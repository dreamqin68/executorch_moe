import importlib
import torch
import torch.nn.functional as F
from copy import deepcopy
from typing import Any
import types
import numpy as np

importlib.import_module("deepseek_moe_split")


# -----------------------------
# Only replace "per-expert for-loop" with C++ (expert_glu_packed)
# -----------------------------
@torch.no_grad()
def _hybrid_moe_infer(
    self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
):
    """
    Hybrid implementation:
      - Python: statistics/sorting/(optional) distributed all_to_all / unsorting / weighted sum
      - C++: Replace the original for i,num_tokens in enumerate(tokens_per_expert) loop
             with a single deepseek_moe_split.expert_glu_packed call
    Convention: x [N,H], topk_ids/topk_weight [N,K]
    """
    N, H = x.shape
    K = topk_ids.size(1)
    E_total = len(self.experts)

    # --- Count & grouping in C++ (no sort op) ---
    idxs, inv, tpe = torch.ops.deepseek_moe_split.group_by_expert(
        topk_ids.to(torch.long), E_total
    )

    if not torch.compiler.is_compiling():
        N, H = x.shape
        M = N * K
        s = int(tpe.sum().to(torch.int64).cpu().item())
        assert s == M, f"Token count mismatch: {s} vs {M}"

        n_idx = idxs // K
        assert n_idx.dtype == torch.long and inv.dtype == torch.long
        mi, ma = int(n_idx.min().cpu()), int(n_idx.max().cpu())
        assert 0 <= mi and ma < N, f"n_idx range [{mi}, {ma}] vs N={N}"
        mi, ma = int(inv.min().cpu()), int(inv.max().cpu())
        assert 0 <= mi and ma < M, f"inv range [{mi}, {ma}] vs M={M}"

    # --- Pack tokens by expert (stable) ---
    n_idx = (idxs // K).to(torch.long)  # [M]
    sorted_tokens = x.index_select(0, n_idx)  # [M, H]
    sorted_tokens_shape = sorted_tokens.shape

    # === Optional: distributed EP (if ep_size>1, keep original logic) ===
    ep_size = int(getattr(self, "ep_size", 1))
    ep_rank = int(getattr(self, "ep_rank", 0))
    experts_per_rank = int(getattr(self, "experts_per_rank", E_total))

    if ep_size > 1:
        import torch.distributed as dist

        # 这里之前误用了未定义的 tokens_per_expert，应使用 tpe
        tokens_per_ep_rank = tpe.view(ep_size, -1).sum(dim=1)
        tokens_per_expert_group = tpe.new_empty(tpe.shape[0])
        dist.all_to_all_single(tokens_per_expert_group, tpe)

        output_splits = (
            tokens_per_expert_group.view(ep_size, -1).sum(1).cpu().numpy().tolist()
        )
        gathered_tokens = sorted_tokens.new_empty(
            tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1]
        )
        input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
        dist.all_to_all(
            list(gathered_tokens.split(output_splits)),
            list(sorted_tokens.split(input_split_sizes)),
        )
        tokens_per_expert_post_gather = tokens_per_expert_group.view(
            ep_size, experts_per_rank
        ).sum(
            dim=0
        )  # [experts_per_rank]

        gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int64)
        s = 0
        for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
            gatherd_idxs[s : s + k] = i % experts_per_rank
            s += k
        gatherd_idxs = gatherd_idxs.argsort()
        sorted_tokens = gathered_tokens[gatherd_idxs]
        # 若后面需要 rank 内本地专家计数，用 tokens_per_expert_post_gather
        tpe = tokens_per_expert_post_gather  # 长度 == experts_per_rank

    # --- Construct weight lists needed by C++ side (only cover experts for this rank) ---
    # base offset (in distributed case, whether experts in module are full or split by rank? keep consistent with your original model)
    base = ep_rank * experts_per_rank if ep_size > 1 else 0
    tokens_per_expert_t = tpe.to(device=sorted_tokens.device, dtype=torch.long)
    E_local = int(tokens_per_expert_t.numel())

    gate_wT = [
        self.experts[base + i].gate_proj.weight.t().contiguous() for i in range(E_local)
    ]
    up_wT = [
        self.experts[base + i].up_proj.weight.t().contiguous() for i in range(E_local)
    ]
    down_wT = [
        self.experts[base + i].down_proj.weight.t().contiguous() for i in range(E_local)
    ]

    # --- Use C++ expert_glu_packed to complete the original per-expert computation and concatenate segments back in order ---
    outs = torch.ops.deepseek_moe_split.expert_glu_packed(
        sorted_tokens.to(torch.float32),
        tpe,  # tokens_per_expert (int64)
        gate_wT,
        up_wT,
        down_wT,
    )
    # outs: [sum(tokens_per_expert), H], order consistent with segment concatenation

    # === Distributed back-transmission to original rank (if ep_size>1) ===
    if ep_size > 1:
        import torch.distributed as dist

        # Use index_put instead of index_put_ for out-of-place operation
        new_x = torch.index_put(torch.empty_like(outs), [gatherd_idxs], outs)
        gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
        dist.all_to_all(
            list(gathered_tokens.split(input_split_sizes)),
            list(new_x.split(output_splits)),
        )
        outs = gathered_tokens

    # --- Unpack back to original (n,k) order WITHOUT argsort / index_put_ ---
    reordered = outs.index_select(0, inv)
    final_out = (
        reordered.view(-1, K, H)
        .to(topk_weight.dtype)
        .mul(topk_weight.unsqueeze(-1).expand(-1, -1, H))
        .sum(dim=1)
        .to(outs.dtype)
    )
    return final_out


def make_moe_with_cpp_expert_loop(moe_py: Any) -> Any:
    """Copy module and replace moe_infer with hybrid implementation: for-loop → expert_glu_packed (C++), rest remains Python"""
    moe = deepcopy(moe_py)
    moe.moe_infer = types.MethodType(_hybrid_moe_infer, moe)
    return moe.eval()


# -----------------------------
# 3) Gate's top-k: call C++ topk_select (optional)
# -----------------------------
def _cpp_topk_values_indices(scores_2d: torch.Tensor, k: int):
    scores_2d = scores_2d.to(torch.float32)
    # establish data edges on the graph
    idx_out, w_out = torch.ops.deepseek_moe_split.topk_select(
        scores_2d,
        k,
        False,
        1.0,
    )
    # Maintain the original "torch.topk-like" return order: (values, indices)
    return w_out, idx_out


def gate_forward_topk_cpp(self, hidden_states: torch.Tensor):
    """
    Use C++ topk_select replace all top-k routing logic in Gate
    """
    bsz, seq_len, h = hidden_states.shape
    x = hidden_states.view(-1, h)

    logits = F.linear(x.float(), self.weight.float(), None)
    if self.scoring_func == "softmax":
        scores = torch.softmax(logits, dim=1)
    elif self.scoring_func == "sigmoid":
        scores = logits.sigmoid()
    else:
        raise NotImplementedError(f"Unsupported scoring_func: {self.scoring_func}")

    # 2) replace top-k selection part
    if self.topk_method == "greedy":
        topk_val, topk_idx = _cpp_topk_values_indices(scores, self.top_k)
        topk_weight = topk_val  # use values as weights

    elif self.topk_method == "group_limited_greedy":
        G = self.n_group
        per = self.n_routed_experts // G
        group_scores = scores.view(-1, G, per).max(dim=-1).values  # [N, G]
        _, group_idx = _cpp_topk_values_indices(group_scores, self.topk_group)

        group_mask = torch.zeros_like(group_scores).scatter(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(-1, G, per).reshape(-1, G * per)
        masked_scores = scores.masked_fill(~score_mask.bool(), 0.0)

        topk_val, topk_idx = _cpp_topk_values_indices(masked_scores, self.top_k)
        topk_weight = scores.gather(1, topk_idx)

    elif self.topk_method == "noaux_tc":
        assert not self.training
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)

        G = self.n_group
        per = self.n_routed_experts // G
        vals2, _ = _cpp_topk_values_indices(
            scores_for_choice.view(-1, G, per).reshape(-1, per), 2
        )  # [(N*G), 2]
        group_scores = vals2.sum(dim=-1).view(-1, G)  # [N, G]
        _, group_idx = _cpp_topk_values_indices(group_scores, self.topk_group)

        group_mask = torch.zeros_like(group_scores).scatter(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(-1, G, per).reshape(-1, G * per)
        masked_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)

        _, topk_idx = _cpp_topk_values_indices(masked_scores, self.top_k)

        topk_weight = scores.gather(1, topk_idx)

    else:
        raise NotImplementedError(f"Unsupported topk_method: {self.topk_method}")

    if self.top_k > 1 and self.norm_topk_prob:
        denom = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denom * self.routed_scaling_factor
    else:
        topk_weight = topk_weight * self.routed_scaling_factor

    if self.training and self.alpha > 0.0:
        aux_topk = self.top_k
        topk_idx_for_aux = topk_idx.view(bsz, -1)
        if self.seq_aux:
            scores_seq = scores.view(bsz, seq_len, -1)
            ce = torch.zeros(bsz, self.n_routed_experts, device=x.device)
            ce = ce.scatter_add(
                1,
                topk_idx_for_aux,
                torch.ones(bsz, seq_len * aux_topk, device=x.device),
            ).div(seq_len * aux_topk / self.n_routed_experts)
            aux_loss = (ce * scores_seq.mean(dim=1)).sum(dim=1).mean() * self.alpha
        else:
            mask_ce = torch.nn.functional.one_hot(
                topk_idx_for_aux.view(-1), num_classes=self.n_routed_experts
            )
            ce = mask_ce.float().mean(0)
            Pi = scores.mean(0)
            fi = ce * self.n_routed_experts
            aux_loss = (Pi * fi).sum() * self.alpha
    else:
        aux_loss = None

    return topk_idx, topk_weight, aux_loss


def patch_gate_topk_inplace(moe):
    """moe.gate.forward replace C++ topk_select version; other unchanged."""
    moe.gate.forward = types.MethodType(gate_forward_topk_cpp, moe.gate)
    return moe
