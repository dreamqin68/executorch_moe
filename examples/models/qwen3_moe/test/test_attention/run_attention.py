import os
import torch
import torch.nn as nn

# ExecuTorch
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime

from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention

from executorch.examples.models.qwen3_moe.print import (
    diff_stats,
    dump_ops,
    dump_edge_ops,
    show_lowered_modules,
)


def build_rope_cos_sin(
    B: int,
    T: int,
    head_dim: int,
    base: float = 10000.0,
    device="cpu",
    dtype=torch.float32,
):
    # basic RoPE frequency: first get [T, head_dim]
    pos = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)  # [T,1]
    idx = torch.arange(0, head_dim, 2, device=device, dtype=dtype)  # [head_dim/2]
    inv_freq = 1.0 / (base ** (idx / head_dim))  # [head_dim/2]
    freqs = pos * inv_freq.unsqueeze(0)  # [T, head_dim/2]
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # [T, head_dim]
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)  # [T, head_dim]

    # key: expand to [B, T, head_dim] to match q/k [B, H, T, head_dim] and unsqueeze_dim=1
    cos = cos.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, T, D]
    sin = sin.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, T, D]
    return cos, sin


# ---------- wrapper so export only needs (x) ----------
class AttentionRunner(nn.Module):
    def __init__(self, attn: Qwen3MoeAttention, cos: torch.Tensor, sin: torch.Tensor):
        super().__init__()
        self.attn = attn.eval()
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor):
        y, w = self.attn(
            x,
            (self.cos, self.sin),
            attention_mask=None,
            past_key_values=None,
            cache_position=None,
        )
        return y, w


def main():
    torch.manual_seed(0)

    # ---- shapes ----
    B, T = 2, 16
    H = 256
    NUM_HEAD = 8
    NUM_KV_HEAD = 4
    HEAD_DIM = H // NUM_HEAD  # 32

    # ---- config ----
    cfg = Qwen3MoeConfig(
        hidden_size=H,
        num_attention_heads=NUM_HEAD,
        num_key_value_heads=NUM_KV_HEAD,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
    )
    cfg._attn_implementation = "eager"

    # ---- module ----
    attn = Qwen3MoeAttention(cfg, layer_idx=0).eval()

    # ---- inputs ----
    x = torch.randn(B, T, H)
    cos, sin = build_rope_cos_sin(B, T, HEAD_DIM, device=x.device, dtype=torch.float32)

    # ---- eager run ----
    runner = AttentionRunner(attn, cos, sin).eval()
    with torch.no_grad():
        y_eager, w_eager = runner(x)

    print("=== EAGER shapes ===")
    print("y:", tuple(y_eager.shape), "  w:", tuple(w_eager.shape))

    # ---- export (only needs x; cos/sin is buffer) ----
    exported = torch.export.export(runner, args=(x,))
    gm = exported.module()
    with torch.no_grad():
        y_exp, w_exp = gm(x)

    print("\n=== In-Python export (eager vs export) ===")
    diff_stats("Hidden: eager vs export", y_eager, y_exp)
    diff_stats("Weights: eager vs export", w_eager, w_exp)

    dump_ops(exported, "exported_program")

    # ---- lower to edge with mixed partitioners ----
    edge = to_edge_transform_and_lower(
        exported,
        # partitioner=[XnnpackPartitioner(), VulkanPartitioner()],
        # partitioner=[XnnpackPartitioner()],
        partitioner=[VulkanPartitioner()],
        # partitioner=[VulkanPartitioner(), XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    dump_edge_ops(edge, "edge_after_lowering")

    # ---- debug: dump delegate information ----
    show_lowered_modules(edge)

    # ---- save .pte ----
    out_pte = os.path.join(
        os.path.dirname(__file__) or ".", "qwen3_moe_attention_vulkan.pte"
    )
    exec_prog = edge.to_executorch()
    with open(out_pte, "wb") as f:
        exec_prog.write_to_file(f)
    print(f"\n[save] wrote {out_pte}")

    # ---- ExecuTorch runtime run ----
    rt = Runtime.get()
    prog = rt.load_program(out_pte)
    method = prog.load_method("forward")

    y_exec, w_exec = method.execute([x])
    print("[exec] y:", tuple(y_exec.shape), "  w:", tuple(w_exec.shape))

    print("\n=== Consistency vs ExecuTorch ===")
    diff_stats("Hidden: export vs exec", y_exp, y_exec)
    diff_stats("Hidden: eager  vs exec", y_eager, y_exec)
    diff_stats("Weights: export vs exec", w_exp, w_exec)
    diff_stats("Weights: eager  vs exec", w_eager, w_exec)


if __name__ == "__main__":
    main()
