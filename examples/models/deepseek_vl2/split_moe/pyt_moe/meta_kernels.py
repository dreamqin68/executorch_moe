import torch
from torch.library import register_fake


# ---- moe_infer_glu ----
@register_fake("deepseek_moe_split::moe_infer_glu")
def moe_infer_glu_meta(x, topk_idx, topk_w, gate_wT_3d, up_wT_3d, down_wT_3d):
    assert x.dim() == 2 and topk_idx.dim() == 2 and topk_w.dim() == 2
    assert gate_wT_3d.dim() == 3 and up_wT_3d.dim() == 3 and down_wT_3d.dim() == 3
    N, H = x.shape
    return x.new_empty((N, H))


@register_fake("deepseek_moe_split::moe_infer_glu.out")
def moe_infer_glu_out_meta(
    x, topk_idx, topk_w, gate_wT_3d, up_wT_3d, down_wT_3d, *, out
):
    assert x.dim() == 2 and topk_idx.dim() == 2 and topk_w.dim() == 2
    assert gate_wT_3d.dim() == 3 and up_wT_3d.dim() == 3 and down_wT_3d.dim() == 3
    N, H = x.shape
    out.resize_(N, H)
    return out


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
