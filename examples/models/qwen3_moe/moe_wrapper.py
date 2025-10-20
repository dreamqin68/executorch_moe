import torch
import torch.nn.functional as F
from copy import deepcopy
import types


# General topk (by dim=1), without depending on torch.topk/sort/kthvalue
def topk_values_indices(scores_2d: torch.Tensor, k: int, *, largest: bool = True):
    if scores_2d.dim() != 2:
        raise ValueError(f"expect 2D tensor [N,E], got {scores_2d.shape}")
    N, E = scores_2d.shape
    if not (1 <= k <= E):
        raise ValueError(f"k must be in [1, {E}], got {k}")
    if not scores_2d.is_floating_point():
        raise TypeError("scores_2d must be floating dtype")

    device = scores_2d.device
    dtype = scores_2d.dtype

    work = scores_2d.clone()
    if not largest:
        work = -work
    very_neg = -torch.finfo(dtype).max

    vals = []
    idxs = []
    for _ in range(k):
        v, i = work.max(dim=1)  # [N],[N]
        vals.append(v)
        idxs.append(i)
        mask = i.unsqueeze(-1) == torch.arange(E, device=device)  # [N,E] bool
        work = work.masked_fill(mask, very_neg)

    vals = torch.stack(vals, dim=1)  # [N,k]
    idxs = torch.stack(idxs, dim=1).to(torch.long)  # [N,k]
    if not largest:
        vals = -vals
    return vals, idxs


@torch.no_grad()
def qwen3moe_forward_sparse_wrapper(moe_block, hidden_states):
    """
    Sparse Top-K (where_topk) forward, no change original class:
      Input: hidden_states [B,S,H]
      Output: (final_hidden_states [B,S,H], router_logits [B*S,E])
    """
    B, S, H = hidden_states.shape
    x = hidden_states.reshape(-1, H)  # [N,H]
    N = x.size(0)
    device, dtype = x.device, x.dtype
    E = moe_block.num_experts
    K = moe_block.top_k

    # 1) gating + softmax(float32 for stability)
    router_logits = moe_block.gate(x)  # [N,E]
    probs = F.softmax(router_logits, dim=-1, dtype=torch.float)

    # 2) where-topk get Top-K probability and index (without using torch.topk)
    topk_vals, topk_idx = topk_values_indices(probs, K, largest=True)  # [N,K],[N,K]
    if getattr(moe_block, "norm_topk_prob", False):
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
    alpha = topk_vals.to(dtype)  # [N,K]
    idx_flat = topk_idx.reshape(-1).to(torch.long)  # [N*K]
    NK = idx_flat.numel()

    # 3) prepare the weights of the selected experts (sparse gather; no expert loop)
    WgT = torch.stack(
        [e.gate_proj.weight.t().contiguous() for e in moe_block.experts], 0
    ).to(
        device=device, dtype=dtype
    )  # [E,H,I]
    WuT = torch.stack(
        [e.up_proj.weight.t().contiguous() for e in moe_block.experts], 0
    ).to(
        device=device, dtype=dtype
    )  # [E,H,I]
    WdT = torch.stack(
        [e.down_proj.weight.t().contiguous() for e in moe_block.experts], 0
    ).to(
        device=device, dtype=dtype
    )  # [E,I,H]

    Wg_sel = WgT.index_select(0, idx_flat)  # [NK,H,I]
    Wu_sel = WuT.index_select(0, idx_flat)  # [NK,H,I]
    Wd_sel = WdT.index_select(0, idx_flat)  # [NK,I,H]

    # 4) forward the selected (n,k) batch
    x_sel = x.unsqueeze(1).expand(N, K, H).reshape(NK, 1, H)  # [NK,1,H]
    g = torch.bmm(x_sel, Wg_sel).squeeze(1)  # [NK,I]
    u = torch.bmm(x_sel, Wu_sel).squeeze(1)  # [NK,I]
    h = F.silu(g) * u  # [NK,I]
    y_pair = torch.bmm(h.unsqueeze(1), Wd_sel).squeeze(1)  # [NK,H]

    # 5) α weighted + mask (where avoid branch)
    alpha_flat = alpha.reshape(-1)  # [NK]
    y_pair = torch.where(
        (alpha_flat != 0).unsqueeze(-1),
        y_pair * alpha_flat.unsqueeze(-1),
        torch.zeros_like(y_pair),
    )  # [NK,H]

    # 6) aggregate back to token
    y = torch.zeros((N, H), device=device, dtype=dtype)
    n_index = torch.arange(N, device=device).repeat_interleave(K)  # [NK]
    y.index_add_(0, n_index, y_pair)  # sum over K

    final_hidden_states = y.reshape(B, S, H)
    return final_hidden_states, router_logits


def make_moe_with_where_infer(moe_py):
    moe = deepcopy(moe_py)
    moe.forward = types.MethodType(qwen3moe_forward_sparse_wrapper, moe)
    return moe.eval()
