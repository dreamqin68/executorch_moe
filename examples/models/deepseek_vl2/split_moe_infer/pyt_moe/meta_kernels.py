import torch
from torch.library import register_fake


# ---- topk_select ----
@register_fake("deepseek_moe_split::topk_select")
def topk_select_meta(
    scores, top_k: int, norm_topk_prob: bool, routed_scaling_factor: float
):
    assert scores.dim() == 2
    N, E = scores.shape
    K = min(int(top_k), int(E))
    idx = scores.new_empty((N, K), dtype=torch.long)
    w = scores.new_empty((N, K), dtype=scores.dtype)
    return idx, w


@register_fake("deepseek_moe_split::topk_select.out")
def topk_select_out_meta(
    scores,
    top_k: int,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
    *,
    topk_idx_out,
    topk_w_out,
):
    assert scores.dim() == 2
    N, E = scores.shape
    K = min(int(top_k), int(E))
    topk_idx_out.resize_(N, K)
    topk_w_out.resize_(N, K)
    return topk_idx_out, topk_w_out


@register_fake("deepseek_moe_split::expert_glu_packed")
def expert_glu_packed_meta(sorted_tokens, tokens_per_expert, gate_wT, up_wT, down_wT):
    assert sorted_tokens.dim() == 2
    return sorted_tokens.new_empty(sorted_tokens.shape)


@register_fake("deepseek_moe_split::expert_glu_packed.out")
def expert_glu_packed_out_meta(
    sorted_tokens, tokens_per_expert, gate_wT, up_wT, down_wT, *, out
):
    assert sorted_tokens.dim() == 2
    out.resize_(sorted_tokens.shape[0], sorted_tokens.shape[1])
    return out


@register_fake("deepseek_moe_split::group_by_expert")
def group_by_expert_meta(topk_idx, num_experts: int):
    N, K = topk_idx.shape
    idxs = topk_idx.new_empty((N * K,))
    inv = topk_idx.new_empty((N * K,))
    cnt = topk_idx.new_empty((num_experts,))
    return (idxs, inv, cnt)


@register_fake("deepseek_moe_split::group_by_expert.out")
def group_by_expert_out_meta(
    topk_idx, num_experts: int, *, indices_out, inverse_out, tokens_per_expert_out
):
    N, K = topk_idx.shape
    indices_out.resize_(N * K)
    inverse_out.resize_(N * K)
    tokens_per_expert_out.resize_(num_experts)
    return indices_out, inverse_out, tokens_per_expert_out
