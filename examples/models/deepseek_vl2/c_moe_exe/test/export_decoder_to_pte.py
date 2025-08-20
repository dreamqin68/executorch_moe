import os, torch
from copy import deepcopy

import sys
from pathlib import Path

THIS = Path(__file__).resolve()
PYT_MOE = THIS.parents[1] / "pyt_moe"
if str(PYT_MOE) not in sys.path:
    sys.path.insert(0, str(PYT_MOE))

import moe_wrapper

from executorch.examples.models.deepseek_vl2.models.configuration_deepseek import (
    DeepseekV2Config,
)
from executorch.examples.models.deepseek_vl2.models.modeling_deepseek import (
    DeepseekV2DecoderLayer,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig


def build_layer(use_moe: bool = True):
    cfg = DeepseekV2Config(
        hidden_size=256,
        intermediate_size=1024,
        moe_intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        n_routed_experts=4 if use_moe else None,
        num_experts_per_tok=2 if use_moe else None,
        n_group=4 if use_moe else None,
        topk_group=2 if use_moe else None,
        moe_layer_freq=1,
        first_k_dense_replace=0,
        hidden_act="silu",
        use_mla=False,
        attention_dropout=0.0,
        topk_method="greedy",
        norm_topk_prob=False,
        scoring_func="softmax",
    )
    cfg._attn_implementation = "eager"

    layer_py = DeepseekV2DecoderLayer(cfg, 0).eval()
    layer_cpp = deepcopy(layer_py)

    from moe_wrapper import replace_layer_mlp_with_cpp

    replace_layer_mlp_with_cpp(layer_cpp).eval()
    return layer_py, layer_cpp, cfg


class DecoderWrapper(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, cos, sin):
        (y,) = self.layer(x, position_embeddings=(cos, sin))
        return y


def get_rope(layer, cfg, x, pos):
    rot = getattr(layer.self_attn, "rotary_emb", None)
    if rot is None and hasattr(layer.self_attn, "self_attn"):
        rot = getattr(layer.self_attn.self_attn, "rotary_emb", None)
    assert rot is not None
    if hasattr(rot, "dim"):
        qk_dim = int(rot.dim)
    else:
        qk_dim = getattr(
            cfg, "qk_rope_head_dim", cfg.hidden_size // cfg.num_attention_heads
        )
    dummy = x.new_empty(x.size(0), x.size(1), qk_dim)
    return rot(dummy, pos)  # (cos, sin)


def main():
    layer_py, layer_cpp, cfg = build_layer(use_moe=True)

    B, T, H = 2, 16, cfg.hidden_size
    x = torch.randn(B, T, H)
    pos = torch.arange(T, dtype=torch.long).unsqueeze(0).expand(B, -1)
    cos, sin = get_rope(layer_py, cfg, x, pos)

    print("=== Kernel Availability Check ===")
    print(
        f"CPU mlp_glu: {torch._C._dispatch_has_kernel_for_dispatch_key('deepseek_moe::mlp_glu', 'CPU')}"
    )
    print(
        f"CPU moe_gate_forward: {torch._C._dispatch_has_kernel_for_dispatch_key('deepseek_moe::moe_gate_forward', 'CPU')}"
    )
    print(
        f"CPU moe_forward_glu: {torch._C._dispatch_has_kernel_for_dispatch_key('deepseek_moe::moe_forward_glu', 'CPU')}"
    )
    print("=================================")

    print("=== Out Kernel Availability ===")
    for name in ["mlp_glu.out", "moe_gate_forward.out", "moe_forward_glu.out"]:
        ok = torch._C._dispatch_has_kernel_for_dispatch_key(
            f"deepseek_moe::{name}", "CPU"
        )
        print(f"CPU {name}: {ok}")
    print("================================")

    print("=== Schema Validation ===")
    op = torch.ops.deepseek_moe.moe_gate_forward
    print("overloads:", op.overloads())

    def get_schema(o):
        return getattr(o, "schema", getattr(o, "_schema", str(o)))

    print("functional:", get_schema(op.default))
    print("out       :", get_schema(op.out))
    print("================================")

    with torch.no_grad():
        (y_py,) = layer_py(x, position_embeddings=(cos, sin))
        (y_cpp,) = layer_cpp(x, position_embeddings=(cos, sin))
    print(f"[eager vs cpp] max|Δ|={(y_py - y_cpp).abs().max().item():.3e}")

    import meta_kernels

    # export → edge → XNNPACK → .pte
    wrapped = DecoderWrapper(layer_cpp).eval()
    exported_program = torch.export.export(wrapped, args=(x, cos, sin))
    print("[export] OK")

    with torch.no_grad():
        y_export = exported_program.module()(x, cos, sin)
    print(f"[cpp vs export] max|Δ|={(y_cpp - y_export).abs().max().item():.3e}")

    # 1) edge transform & backend
    edge = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
        # if there are custom operators/shapes that are not static, turn off strict validation for a smoother experience
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    # 2) write .pte
    THIS = Path(__file__).resolve()
    out_path = (THIS.parent / "decoder_xnn.pte").resolve()
    exec_prog = edge.to_executorch()
    with open(out_path, "wb") as f:
        exec_prog.write_to_file(f)
    print(f"[save] wrote {out_path}")


if __name__ == "__main__":
    main()
