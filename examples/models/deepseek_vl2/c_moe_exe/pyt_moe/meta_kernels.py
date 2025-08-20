import torch
from torch.library import register_fake


# ---------- mlp_glu.out ----------
@register_fake("deepseek_moe::mlp_glu.out")
def mlp_glu_out_meta(x, Wg, Bg, Wu, Bu, Wd, Bd, act_id, act_params, *, out):
    # out is given by caller; here only resize
    out.resize_(x.shape)
    return out


@register_fake("deepseek_moe::mlp_glu")
def mlp_glu_meta(x, Wg, Bg, Wu, Bu, Wd, Bd, act_id, act_params):
    return x.new_empty(x.shape)


# ---------- moe_gate_forward.out ----------
@register_fake("deepseek_moe::moe_gate_forward.out")
def moe_gate_forward_out_meta(
    hidden,
    weight,
    top_k,
    routed_scaling_factor,
    norm_topk_prob,
    scoring_id,
    method_id,
    n_group,
    topk_group,
    training,
    alpha,
    seq_aux,
    e_score_correction_bias,
    *,
    topk_idx_out,
    topk_w_out,
):
    assert hidden.dim() == 3
    B, T, H = hidden.shape
    K = int(top_k)
    BT = B * T
    topk_idx_out.resize_(BT, K)
    topk_w_out.resize_(BT, K)
    return topk_idx_out, topk_w_out


@register_fake("deepseek_moe::moe_gate_forward")
def moe_gate_forward_meta(
    hidden,
    weight,
    top_k,
    routed_scaling_factor,
    norm_topk_prob,
    scoring_id,
    method_id,
    n_group,
    topk_group,
    training,
    alpha,
    seq_aux,
    e_score_correction_bias,
):
    B, T, H = hidden.shape
    K = int(top_k)
    BT = B * T
    idx = hidden.new_empty((BT, K), dtype=torch.long)
    w = hidden.new_empty((BT, K), dtype=hidden.dtype)
    return idx, w


# ---------- moe_forward_glu.out ----------
@register_fake("deepseek_moe::moe_forward_glu.out")
def moe_forward_glu_out_meta(
    hidden, topk_idx, topk_weight, Wg, Bg, Wu, Bu, Wd, Bd, act_id, act_params, *, out
):
    out.resize_(hidden.shape)
    return out


@register_fake("deepseek_moe::moe_forward_glu")
def moe_forward_glu_meta(
    hidden, topk_idx, topk_weight, Wg, Bg, Wu, Bu, Wd, Bd, act_id, act_params
):
    return hidden.new_empty(hidden.shape)
