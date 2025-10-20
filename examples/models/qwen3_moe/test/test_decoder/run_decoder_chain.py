import torch
import torch.nn as nn
from copy import deepcopy
from pathlib import Path
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime
from executorch.examples.models.qwen3_moe.moe_wrapper import make_moe_with_where_infer
from executorch.examples.models.qwen3_moe.print import (
    diff_stats,
    dump_ops,
    dump_edge_ops,
    show_lowered_modules,
)
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeSparseMoeBlock,
)


# ---- minimal, deterministic RoPE (cos, sin) builder ----
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


# ---------- Stage 1: Attention ----------
class DecoderStage1_Attn(nn.Module):
    def __init__(self, dec: Qwen3MoeDecoderLayer, cos: torch.Tensor, sin: torch.Tensor):
        super().__init__()

        self.input_layernorm = deepcopy(dec.input_layernorm).eval()
        self.self_attn = deepcopy(dec.self_attn).eval()
        self.self_attn.config._attn_implementation = "eager"
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor):
        residual = x
        x0 = self.input_layernorm(x)
        attn_out, _ = self.self_attn(
            hidden_states=x0,
            position_embeddings=(self.cos, self.sin),
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            cache_position=None,
        )
        h1 = residual + attn_out
        return h1


# ---------- Stage 2: MoE ----------
class DecoderStage2_MoE(nn.Module):
    def __init__(self, dec: Qwen3MoeDecoderLayer):
        super().__init__()
        self.post_attention_layernorm = deepcopy(dec.post_attention_layernorm).eval()

        if isinstance(dec.mlp, Qwen3MoeSparseMoeBlock):
            self.mlp = make_moe_with_where_infer(deepcopy(dec.mlp)).eval()
        else:
            self.mlp = deepcopy(dec.mlp).eval()

    def forward(self, h1: torch.Tensor):
        residual = h1
        x1 = self.post_attention_layernorm(h1)
        mlp_out = self.mlp(x1)
        if isinstance(mlp_out, tuple):  # MoE return (hidden, logits)
            mlp_out, _ = mlp_out
        y = residual + mlp_out
        return y


def lower_to_pte(exported, partitioner, out_pte_path: str):
    edge = to_edge_transform_and_lower(
        exported,
        compile_config=EdgeCompileConfig(
            _skip_dim_order=False, _check_ir_validity=False
        ),
        partitioner=partitioner,
    )

    dump_edge_ops(edge, "edge_after_lowering_decoder")
    show_lowered_modules(edge)

    exec_prog = edge.to_executorch()
    with open(out_pte_path, "wb") as f:
        exec_prog.write_to_file(f)
    return out_pte_path


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
    torch.manual_seed(0)

    # ---- Config: force this layer to be a sparse MoE layer----
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

    decoder_orig = Qwen3MoeDecoderLayer(cfg, layer_idx=0).eval()
    decoder_where = deepcopy(decoder_orig).eval()
    if isinstance(decoder_where.mlp, Qwen3MoeSparseMoeBlock):
        decoder_where.mlp = make_moe_with_where_infer(decoder_where.mlp).eval()

    # ---- Inputs ----
    x = torch.randn(B, T, H, device=device, dtype=dtype)

    # RoPE (cos, sin): (B, T, HEAD_DIM)
    cos, sin = build_rope_cos_sin(B, T, HEAD_DIM, device=device, dtype=dtype)

    # ---- Eager (WHERE variant) ----
    with torch.no_grad():
        _x0 = decoder_where.input_layernorm(x)
        _attn, _ = decoder_where.self_attn(
            _x0,
            (cos, sin),
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            cache_position=None,
        )
        hidden_after_attn_where = x + _attn
        _x1 = decoder_where.post_attention_layernorm(hidden_after_attn_where)
        _moe = decoder_where.mlp(_x1)  # MoE block (Qwen3MoeSparseMoeBlock)
        if isinstance(_moe, tuple):
            _moe, _ = _moe
        hidden_after_moe_where = hidden_after_attn_where + _moe

    # ---- Eager (ORIGINAL MoE baseline) ----
    with torch.no_grad():
        _x0o = decoder_orig.input_layernorm(x)
        _attn_o, _ = decoder_orig.self_attn(
            _x0o,
            (cos, sin),
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            cache_position=None,
        )
        hidden_after_attn_orig = x + _attn_o
        _x1o = decoder_orig.post_attention_layernorm(hidden_after_attn_orig)
        _moe_o = decoder_orig.mlp(_x1o)  # MoE block (Qwen3MoeSparseMoeBlock)
        if isinstance(_moe_o, tuple):
            _moe_o, _ = _moe_o
        hidden_after_moe_orig = hidden_after_attn_orig + _moe_o

    # --- build two-phase runners ---
    stage1 = DecoderStage1_Attn(decoder_where, cos, sin).eval()
    stage2 = DecoderStage2_MoE(decoder_where).eval()

    # --- export ---
    ep1 = torch.export.export(stage1, args=(x,))
    ep2 = torch.export.export(stage2, args=(hidden_after_attn_where,))

    # --- lower to different backends ---
    THIS = Path(__file__).resolve()
    PTE1 = str((THIS.parent / "decoder_stage1_attn_vulkan.pte").resolve())
    PTE2 = str((THIS.parent / "decoder_stage2_moe_xnnpack.pte").resolve())
    lower_to_pte(
        ep1, partitioner=[VulkanPartitioner()], out_pte_path=PTE1
    )  # Attention → Vulkan
    lower_to_pte(
        ep2, partitioner=[XnnpackPartitioner()], out_pte_path=PTE2
    )  # MoE → XNNPACK
    print(f"[save] wrote {PTE1}\n[save] wrote {PTE2}")

    # --- Runtime chain execution ---
    rt = Runtime.get()
    prog1 = rt.load_program(PTE1)
    m1 = prog1.load_method("forward")
    (hidden_after_attn_exec,) = m1.execute([x])  # ATTN on Vulkan

    prog2 = rt.load_program(PTE2)
    m2 = prog2.load_method("forward")
    (hidden_after_moe_exec,) = m2.execute([hidden_after_attn_exec])  # MoE on XNNPACK

    print(
        "[exec] hidden_after_attn:",
        tuple(hidden_after_attn_exec.shape),
        " hidden_after_moe:",
        tuple(hidden_after_moe_exec.shape),
    )

    # --- consistency check ---
    print("\n=== Consistency: WHERE eager vs EXEC (per stage) ===")
    diff_stats(
        "ATTN: where-eager vs exec", hidden_after_attn_where, hidden_after_attn_exec
    )
    diff_stats(
        "MOE : where-eager vs exec", hidden_after_moe_where, hidden_after_moe_exec
    )

    print("\n=== Model delta: ORIGINAL vs WHERE (both eager, isolates 'where') ===")
    diff_stats(
        "ATTN: original-eager vs where-eager",
        hidden_after_attn_orig,
        hidden_after_attn_where,
    )
    diff_stats(
        "MOE : original-eager vs where-eager",
        hidden_after_moe_orig,
        hidden_after_moe_where,
    )

    print("\n=== End-to-end: ORIGINAL eager vs chained EXEC on WHERE ===")
    diff_stats(
        "ATTN: original-eager vs exec", hidden_after_attn_orig, hidden_after_attn_exec
    )
    diff_stats(
        "MOE : original-eager vs exec", hidden_after_moe_orig, hidden_after_moe_exec
    )


if __name__ == "__main__":
    main()
