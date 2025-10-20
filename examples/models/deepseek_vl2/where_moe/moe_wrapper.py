import torch
import torch.nn.functional as F
from copy import deepcopy
import types
from typing import Tuple

@torch.no_grad()
def _moe_infer_where(
    self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor
):
    """
    Python /MoE infer(static graph), **no Python for-loop**:
      Only compute for the experts selected in (n,k), then aggregate by token dimension.
    Shape:
      x            : [N, H]
      topk_idx     : [N, K]   (long)
      topk_weight  : [N, K]   (float)
      self.experts : list, each expert contains gate/up/down Linear
    Return:
      y : [N, H]
    """
    device, dtype = x.device, x.dtype
    N, H = x.shape
    K = topk_idx.size(1)
    E = len(self.experts)

    # 1) Stack all experts weights (still tensor-level operation; no explicit for-loop computation)
    WgT = torch.stack(
        [e.gate_proj.weight.t().contiguous() for e in self.experts], 0
    ).to(
        device=device, dtype=dtype
    )  # [E,H,I]
    WuT = torch.stack([e.up_proj.weight.t().contiguous() for e in self.experts], 0).to(
        device=device, dtype=dtype
    )  # [E,H,I]
    WdT = torch.stack(
        [e.down_proj.weight.t().contiguous() for e in self.experts], 0
    ).to(
        device=device, dtype=dtype
    )  # [E,I,H]

    # 2) Select corresponding expert weights (sparse gather) by (n,k)
    idx_flat = topk_idx.reshape(-1).to(torch.long)  # [N*K]
    NK = idx_flat.numel()
    Wg_sel = WgT.index_select(0, idx_flat)  # [NK,H,I]
    Wu_sel = WuT.index_select(0, idx_flat)  # [NK,H,I]
    Wd_sel = WdT.index_select(0, idx_flat)  # [NK,I,H]

    # 3) Copy token vector for each (n,k), and do batch matmul
    x_sel = x.unsqueeze(1).expand(N, K, H).reshape(NK, 1, H)  # [NK,1,H]
    g = torch.bmm(x_sel, Wg_sel).squeeze(1)  # [NK,I]
    u = torch.bmm(x_sel, Wu_sel).squeeze(1)  # [NK,I]
    h = F.silu(g) * u  # [NK,I]
    y_pair = torch.bmm(h.unsqueeze(1), Wd_sel).squeeze(1)  # [NK,H] = f_{e_{n,k}}(x_n)

    # 4) Multiply by routing weights (use mask+where; weights of 0 are directly cleared)
    alpha = topk_weight.reshape(-1).to(dtype=dtype)  # [NK]
    mask = alpha != 0
    y_pair = torch.where(
        mask.unsqueeze(-1), y_pair * alpha.unsqueeze(-1), torch.zeros_like(y_pair)
    )  # [NK,H]

    # 5) Write back to token dimension and aggregate by K
    y = torch.zeros((N, H), device=device, dtype=dtype)
    n_index = torch.arange(N, device=device).repeat_interleave(
        K
    )  # [NK]：each (n,k) belongs to which n
    y.index_add_(0, n_index, y_pair)  # Sum over (n,k) for the same n
    return y


def make_moe_with_where_infer(moe_py):
    """
    Return a new Module, replace .moe_infer with the static graph version above.
    Other (like gate.forward / topk selection) keep the original pure PyTorch implementation (same ATen primitives).
    """
    moe = deepcopy(moe_py)
    moe.moe_infer = types.MethodType(_moe_infer_where, moe)
    return moe.eval()


def where_topk_values_indices(
    scores_2d: torch.Tensor, k: int, *, largest: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    General Top-K (by dim=1), without using torch.topk/sort/kthvalue.
    Return (values[N,k], indices[N,k]), sorted by descending (largest=True) or ascending (largest=False).
    - The stability of torch.topk on duplicate values may be slightly different; this implementation uses the order of "first appearing index priority" for equal elements.
    """
    if scores_2d.dim() != 2:
        raise ValueError(f"expect 2D tensor [N,E], got {scores_2d.shape}")
    N, E = scores_2d.shape
    if not (1 <= k <= E):
        raise ValueError(f"k must be in [1, {E}], got {k}")
    if not scores_2d.is_floating_point():
        raise TypeError("scores_2d must be floating dtype")

    device = scores_2d.device
    dtype = scores_2d.dtype

    # Copy a working tensor; negate the work tensor to convert "min" to "max"
    work = scores_2d.clone()
    if not largest:
        work = -work

    # Fill the minimum value for the "selected position" (avoid -inf compatibility problem; adapt to dtype)
    very_neg = -torch.finfo(dtype).max

    vals = []
    idxs = []
    # Iterate k times: each time take the global row argmax, then mask the position to very_neg
    for _ in range(k):
        v1, i1 = work.max(dim=1)  # [N], [N]
        vals.append(v1)
        idxs.append(i1)

        # Construct a boolean mask to mask the selected position
        mask = i1.unsqueeze(-1) == torch.arange(E, device=device)  # [N,E] bool
        work = work.masked_fill(mask, very_neg)

    # Stack back [N,k], and negate the values if needed (corresponding to largest=False)
    vals = torch.stack(vals, dim=1)  # [N,k]
    idxs = torch.stack(idxs, dim=1).to(torch.long)  # [N,k]
    if not largest:
        vals = -vals
    return vals, idxs


def gate_forward_topk_where(self, hidden_states: torch.Tensor):
    """Gate.forward: replace all top-k logic with where top-k."""
    bsz, seq_len, h = hidden_states.shape
    x = hidden_states.view(-1, h)

    logits = F.linear(x.float(), self.weight.float(), None)
    if self.scoring_func == "softmax":
        scores = torch.softmax(logits, dim=1)
    elif self.scoring_func == "sigmoid":
        scores = logits.sigmoid()
    else:
        raise NotImplementedError(f"Unsupported scoring_func: {self.scoring_func}")

    if self.topk_method == "greedy":
        topk_weight, topk_idx = where_topk_values_indices(scores, self.top_k)

    elif self.topk_method == "group_limited_greedy":
        group_scores = (
            scores.view(bsz * seq_len, self.n_group, -1).max(dim=-1).values
        )  # [n, n_group]
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[
            1
        ]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(bsz * seq_len, -1)
        )  # [n, e]
        tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
        topk_weight, topk_idx = torch.topk(
            tmp_scores, k=self.top_k, dim=-1, sorted=False
        )
    elif self.topk_method == "noaux_tc":
        assert not self.training
        scores_for_choice = scores.view(
            bsz * seq_len, -1
        ) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(bsz * seq_len, self.n_group, -1)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )  # [n, n_group]
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[
            1
        ]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(bsz * seq_len, -1)
        )  # [n, e]
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)

    ### norm gate to sum 1
    if self.top_k > 1 and self.norm_topk_prob:
        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator * self.routed_scaling_factor
    else:
        topk_weight = topk_weight * self.routed_scaling_factor
    ### expert-level computation auxiliary loss
    if self.training and self.alpha > 0.0:
        scores_for_aux = scores
        aux_topk = self.top_k
        # always compute aux loss based on the naive greedy topk method
        topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
        if self.seq_aux:
            scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
            ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
            ce.scatter_add_(
                1,
                topk_idx_for_aux_loss,
                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
            ).div_(seq_len * aux_topk / self.n_routed_experts)
            aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                dim=1
            ).mean() * self.alpha
        else:
            mask_ce = F.one_hot(
                topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
            )
            ce = mask_ce.float().mean(0)
            Pi = scores_for_aux.mean(0)
            fi = ce * self.n_routed_experts
            aux_loss = (Pi * fi).sum() * self.alpha
    else:
        aux_loss = None
    return topk_idx, topk_weight, aux_loss


def patch_gate_topk_where(moe):
    """Replace moe.gate.forward with where top-k implementation."""
    moe.gate.forward = types.MethodType(gate_forward_topk_where, moe.gate)
    return moe
