import importlib
import torch
from copy import deepcopy
from typing import Any
import types
import torch.nn.functional as F

# ensure C++ extension is loaded
importlib.import_module("deepseek_moe_split")


def _cpp_moe_infer(self, x, topk_idx, topk_w):
    E = len(self.experts)
    # stack List to 3D Tensor, avoid ExecuTorch's TensorList matching problem
    gate_wT_3d = torch.stack(
        [self.experts[i].gate_proj.weight.t().contiguous() for i in range(E)], dim=0
    )  # [E,H,I]
    up_wT_3d = torch.stack(
        [self.experts[i].up_proj.weight.t().contiguous() for i in range(E)], dim=0
    )  # [E,H,I]
    down_wT_3d = torch.stack(
        [self.experts[i].down_proj.weight.t().contiguous() for i in range(E)], dim=0
    )  # [E,I,H]

    # use functional version, let EXIR automatically convert to .out
    return torch.ops.deepseek_moe_split.moe_infer_glu(
        x, topk_idx, topk_w, gate_wT_3d, up_wT_3d, down_wT_3d
    )


def make_moe_with_cpp_infer(moe_py: Any) -> Any:
    """Copy module, and replace moe_infer with C++ version."""
    moe = deepcopy(moe_py)
    moe.moe_infer = types.MethodType(_cpp_moe_infer, moe)
    return moe.eval()


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
