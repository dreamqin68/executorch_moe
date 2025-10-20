from pathlib import Path
from copy import deepcopy
import torch
import torch.nn as nn

from executorch.examples.models.qwen3_moe.print import (
    diff_stats,
    dump_ops,
    dump_edge_ops,
    show_lowered_modules,
)

from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
    VulkanPartitioner,
    VulkanSupportedOperators,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime

from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeSparseMoeBlock,
)
from executorch.examples.models.qwen3_moe.moe_wrapper import make_moe_with_where_infer


def build_rope_cos_sin(
    B: int, T: int, head_dim: int, rope_theta: float = 10000.0, device=None, dtype=None
):
    """
    Return (cos, sin) with shape (B, T, head_dim), using standard RoPE angles:

        theta_i = rope_theta ** (-2i / head_dim),  i in [0, head_dim/2)
        angles  = pos * theta (interleaved even/odd handled by downstream)
    """
    # positions: (T, 1)
    pos = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)  # (T, 1)

    # frequencies for half-dim, then tile to full head_dim by [cos(x0), cos(x0), cos(x1), cos(x1), ...]
    half = head_dim // 2
    if head_dim % 2 != 0:
        half = head_dim // 2
        pad = 1
    else:
        pad = 0

    inv_freq = torch.pow(
        torch.tensor(rope_theta, device=device, dtype=dtype),
        -torch.arange(0, half, device=device, dtype=dtype) * (2.0 / head_dim),
    )  # (half,)
    angles = pos * inv_freq.unsqueeze(0)  # (T, half)

    # interleave to full dim: [a0,a0,a1,a1,...]
    angles_full = torch.stack([angles, angles], dim=-1).reshape(T, -1)  # (T, 2*half)
    if pad == 1:
        # if odd dim, pad one column 0, cos=1,sin=0,don't affect numerical stability
        angles_full = torch.nn.functional.pad(
            angles_full, (0, 1), value=0.0
        )  # (T, 2*half+1) == head_dim

    # (1, T, head_dim) → (B, T, head_dim)
    cos = torch.cos(angles_full).unsqueeze(0).expand(B, -1, -1).contiguous()
    sin = torch.sin(angles_full).unsqueeze(0).expand(B, -1, -1).contiguous()
    return cos, sin


class DecoderRunner(nn.Module):
    """Wrapper to handle RoPE embeddings as buffers and simplify export/runtime inputs."""

    def __init__(self, decoder, cos, sin):
        super().__init__()
        self.decoder = decoder.eval()
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        """Forward pass with RoPE embeddings from buffers and None for optional parameters."""
        y = self.decoder(
            hidden_states=x,
            position_embeddings=(self.cos, self.sin),
            attention_mask=None,  # Use causal attention built-in masking
            position_ids=None,
            past_key_values=None,
            cache_position=None,
        )
        return y


# ===== 2) module-based Decoder with scope-based partitioning =====
class AttnBlock(nn.Module):
    """attention module: explicit scope boundary"""

    def __init__(self, decoder, cos, sin):
        super().__init__()
        self.input_ln = decoder.input_layernorm
        self.self_attn = decoder.self_attn
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        r = x
        h = self.input_ln(x)
        h, _ = self.self_attn(
            hidden_states=h,
            position_embeddings=(self.cos, self.sin),
            attention_mask=None,
            past_key_values=None,
            cache_position=None,
        )
        return r + h


class MoeBlock(nn.Module):
    """MoE module: explicit scope boundary"""

    def __init__(self, decoder):
        super().__init__()
        self.post_ln = decoder.post_attention_layernorm
        self.mlp = decoder.mlp  #  where version

    def forward(self, h):
        r = h
        h = self.post_ln(h)
        h = self.mlp(h)
        if isinstance(h, tuple):
            h, _ = h
        return r + h


class DecoderRunnerScoped(nn.Module):
    """module-based Decoder with scope-based partitioning: manually assign operators to different backends"""

    def __init__(self, decoder, cos, sin):
        super().__init__()
        self.attn = AttnBlock(decoder, cos, sin)  # scope name fixed: attn
        self.moe = MoeBlock(decoder)  # scope name fixed: moe

    def forward(self, x):
        h = self.attn(x)
        h = self.moe(h)
        return h


# ========== only filter "scope==attn" for Vulkan ==========
_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)


def _infer_scope_fx(node: torch.fx.Node) -> str | None:
    sc = node.meta.get("scope", None)
    if sc:
        return sc
    stack = node.meta.get("nn_module_stack", {}) or {}
    if stack:
        key = list(stack.keys())[-1]
        if "attn" in key:
            return "attn"
        if "moe" in key:
            return "moe"
    # conservative backtrace (avoid "contaminating" int path from MoE to attn)
    boring = (
        "view",
        "reshape",
        "contiguous",
        "expand",
        "slice",
        "select",
        "permute",
        "transpose",
        "to",
        "_to_copy",
        "detach",
        "clone",
    )
    seen = {node}
    st = list(getattr(node, "all_input_nodes", []))
    while st:
        cur = st.pop()
        if cur in seen:
            continue
        seen.add(cur)
        try:
            v = cur.meta.get("val", None)
            if v is not None and v.dtype in _INT_DTYPES:
                return None
        except Exception:
            pass
        sc = cur.meta.get("scope", None)
        if sc:
            return sc
        tgt = str(getattr(cur, "target", "")).lower()
        if any(b in tgt for b in boring):
            st.extend(getattr(cur, "all_input_nodes", []))
    return None


_orig_vk_node_is_compatible = VulkanSupportedOperators.node_is_compatible


def _node_is_compatible_attn_only(self, node, features=None):
    ok, reason = _orig_vk_node_is_compatible(self, node, features=features)
    if not ok:
        return ok, reason
    sc = _infer_scope_fx(node)
    if sc != "attn":
        # only accept attn scope, other (including MoE) are rejected, let XNNPACK handle the rest
        return (False, "filtered: not in attn scope")
    return (True, "filtered: attn scope accepted")


VulkanSupportedOperators.node_is_compatible = _node_is_compatible_attn_only
print("[PATCH] VulkanSupportedOperators.node_is_compatible -> attn-only")


def main():
    # ---- Shapes & Hyperparams ----
    B, T = 2, 16  # batch, seq_len
    H = 256  # hidden_size
    N_HEAD = 8  # num_attention_heads
    HEAD_DIM = H // N_HEAD
    I = 1024  # MLP intermediate_size (dense layer use)
    I_MOE = 512  # MoE intermediate_size per expert
    NEXP, K = 4, 2  # num_experts, num_experts_per_tok

    device = torch.device("cpu")
    dtype = torch.float32
    # dtype = torch.float16
    torch.manual_seed(0)

    # ---- Config: force this layer to be a sparse MoE layer ((layer_idx+1)%decoder_sparse_step==0 and num_experts>0)----
    cfg = Qwen3MoeConfig(
        hidden_size=H,
        intermediate_size=I,
        num_attention_heads=N_HEAD,
        moe_intermediate_size=I_MOE,
        num_experts=NEXP,
        num_experts_per_tok=K,
        decoder_sparse_step=1,  # every layer is a MoE layer
        mlp_only_layers=[],  # don't mask any layer
    )
    cfg._attn_implementation = "eager"

    layer_idx = 0
    # 1) Original decoder (for Eager comparison)
    decoder_orig = Qwen3MoeDecoderLayer(cfg, layer_idx=layer_idx).eval()
    # 2) where-based decoder (for export/lowering/runtime)
    decoder_where = deepcopy(decoder_orig).eval()
    if isinstance(decoder_where.mlp, Qwen3MoeSparseMoeBlock):
        decoder_where.mlp = make_moe_with_where_infer(decoder_where.mlp).eval()

    # ---- Inputs ----
    x = torch.randn(B, T, H, device=device, dtype=dtype)

    # RoPE (cos, sin): (B, T, HEAD_DIM)
    cos, sin = build_rope_cos_sin(
        B, T, HEAD_DIM, rope_theta=10000.0, device=device, dtype=dtype
    )

    # Create runners
    runner_orig = DecoderRunner(decoder_orig, cos, sin).eval()
    runner_scoped_orig = DecoderRunnerScoped(decoder_orig, cos, sin).eval()
    runner_where = DecoderRunnerScoped(decoder_where, cos, sin).eval()

    # ---- Eager ----
    with torch.no_grad():
        y_orig = runner_orig(x)
        y_scoped_orig = runner_scoped_orig(x)
        y_where = runner_where(x)

    # ---- Export (using where-based version) ----
    exported = torch.export.export(runner_where, args=(x,))
    gm = exported.module()
    with torch.no_grad():
        y_export = gm(x)

    dump_ops(exported, "exported_program_decoder")

    # ===== 3) module-based Decoder with scope-based partitioning =====
    print(
        "[DEBUG] Using scope-based partitioning via monkey patch (VulkanPartitioner + XnnpackPartitioner)"
    )

    # check before export: confirm nn_module_stack is correctly captured
    gm_fx = exported.module()
    print("\n[Sanity] nn_module_stack on FX:")
    for n in gm_fx.graph.nodes:
        if n.op in ("call_function", "call_module"):
            stack = n.meta.get("nn_module_stack", {})
            print(
                f"{n.op:>12} {str(n.target):40s}  scopes={list(stack.keys())[-1] if stack else None}"
            )

    # lowering + partitioning (using scope-based partitioners)
    edge_cfg = EdgeCompileConfig(_skip_dim_order=True, _check_ir_validity=False)
    edge = to_edge_transform_and_lower(
        exported,
        compile_config=edge_cfg,
        # use standard partitioners with scope filtering via monkey patch
        partitioner=[
            VulkanPartitioner(),  # now filtered to only accept scope=attn via monkey patch
            XnnpackPartitioner(),  # will handle remaining nodes (including moe scope)
        ],
    )

    print("[STEP] after to_edge_transform_and_lower", flush=True)
    show_lowered_modules(edge)

    print("[STEP] after show_lowered_modules", flush=True)

    # dump_edge_ops(edge, "edge_after_lowering_decoder")
    print("[STEP] (skipped dump_edge_ops)", flush=True)

    exec_prog = edge.to_executorch()
    print("[STEP] after to_executorch", flush=True)

    THIS = Path(__file__).resolve()
    PTE = (THIS.parent / "qwen3_decoder_barrier.pte").resolve()
    with open(PTE, "wb") as f:
        exec_prog.write_to_file(f)
    print(f"[save] wrote {PTE}", flush=True)

    rt = Runtime.get()
    print("[STEP] after Runtime.get", flush=True)

    prog = rt.load_program(str(PTE))
    print("[STEP] after load_program", flush=True)
    method = prog.load_method("forward")
    print("[STEP] after load_method", flush=True)

    # With DecoderRunner (where), ExecuTorch runtime only needs x as input
    (y_exec,) = method.execute([x])

    print(f"[exec] y: {tuple(y_exec.shape)}")

    print("\n=== SUMMARY · Whole Decoder Layer (FP32) ===")
    diff_stats("Eager: Original vs Where", y_orig, y_where)

    diff_stats("Where: Eager vs Export", y_where, y_export)
    diff_stats("Where: Eager vs Exec  ", y_where, y_exec)
    diff_stats("Where: Export vs Exec ", y_export, y_exec)

    diff_stats("Original-eager vs where-Exec", y_orig, y_exec)


if __name__ == "__main__":
    main()
