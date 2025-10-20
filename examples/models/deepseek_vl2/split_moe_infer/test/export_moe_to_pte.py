import sys, importlib, torch
from pathlib import Path

THIS = Path(__file__).resolve()
PYT_MOE = THIS.parents[1] / "pyt_moe"
if str(PYT_MOE) not in sys.path:
    sys.path.insert(0, str(PYT_MOE))


# --- load C++ custom ops (deepseek_moe_split*.so) ---
def _load_split_op():
    try:
        importlib.import_module("deepseek_moe_split")
        return
    except Exception:
        pass
    candidates = list(PYT_MOE.glob("deepseek_moe_split*.so"))
    candidates += list((PYT_MOE / "build").glob("**/deepseek_moe_split*.so"))
    if not candidates:
        raise RuntimeError(
            f"deepseek_moe_split*.so not found.\n  cd {PYT_MOE} && python setup.py build_ext --inplace"
        )
    so_path = str(min(candidates, key=lambda p: len(str(p))))
    torch.ops.load_library(so_path)

    # Check if all required kernels are available
    ok_exp_func = torch._C._dispatch_has_kernel_for_dispatch_key(
        "deepseek_moe_split::expert_glu_packed", "CPU"
    )
    ok_exp_out = torch._C._dispatch_has_kernel_for_dispatch_key(
        "deepseek_moe_split::expert_glu_packed.out", "CPU"
    )
    ok_topk_func = torch._C._dispatch_has_kernel_for_dispatch_key(
        "deepseek_moe_split::topk_select", "CPU"
    )
    ok_topk_out = torch._C._dispatch_has_kernel_for_dispatch_key(
        "deepseek_moe_split::topk_select.out", "CPU"
    )

    if not (ok_exp_func and ok_exp_out and ok_topk_func and ok_topk_out):
        raise RuntimeError(
            f"CPU kernels not found: expert_glu_packed={ok_exp_func}/{ok_exp_out}, "
            f"topk_select={ok_topk_func}/{ok_topk_out}"
        )


_load_split_op()

# --- fake/meta registration
import meta_kernels

from moe_wrapper import (
    patch_gate_topk_inplace,
    make_moe_with_cpp_expert_loop,
)

from executorch.examples.models.deepseek_vl2.models.configuration_deepseek import (
    DeepseekV2Config,
)
from executorch.examples.models.deepseek_vl2.models.modeling_deepseek import (
    DeepseekV2MoE,
)
from executorch.examples.models.deepseek_vl2.utils.print import (
    diff_stats,
    dump_ops,
    dump_edge_ops,
)
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig


def main():
    torch.manual_seed(0)

    cfg = DeepseekV2Config(
        hidden_size=256,
        intermediate_size=1024,
        moe_intermediate_size=512,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=4,
        topk_group=2,
        hidden_act="silu",
        attention_dropout=0.0,
        topk_method="greedy",
        norm_topk_prob=False,
        scoring_func="softmax",
    )

    moe_py = DeepseekV2MoE(cfg).eval()
    # hybrid: gate uses C++ topk_select; expert loop uses expert_glu_packed (C++), rest remains Python
    moe_hybrid = patch_gate_topk_inplace(make_moe_with_cpp_expert_loop(moe_py)).eval()

    B, T, H = 2, 16, cfg.hidden_size
    x = torch.randn(B, T, H)

    with torch.no_grad():
        y_py = moe_py(x)
        y_hb = moe_hybrid(x)

    print("=== EAGER (python vs cpp_infer) ===")
    diff_stats("python vs cpp_infer", y_py, y_hb)

    # Export
    exported = torch.export.export(moe_hybrid, args=(x,))
    gm = exported.module()

    dump_ops(exported, "exported_program")

    with torch.no_grad():
        y_export = gm(x)
    print("=== In-Python export (hybrid vs export) ===")
    diff_stats("hybrid vs export", y_hb, y_export)

    # Self-check: confirm .out variants are visible
    print(
        "\n[check] deepseek_moe_split ops:",
        [n for n in dir(torch.ops.deepseek_moe_split) if not n.startswith("_")],
    )
    print(
        "[check] has expert_glu_packed.out :",
        hasattr(torch.ops.deepseek_moe_split.expert_glu_packed, "out"),
    )
    print(
        "[check] has topk_select.out       :",
        hasattr(torch.ops.deepseek_moe_split.topk_select, "out"),
    )
    print(
        "[check] schema expert_glu_packed.out:",
        torch.ops.deepseek_moe_split.expert_glu_packed.out._schema,
    )
    print(
        "[check] schema topk_select.out      :",
        torch.ops.deepseek_moe_split.topk_select.out._schema,
    )

    # Lower → ExecuTorch
    edge = to_edge_transform_and_lower(
        exported,
        partitioner=[],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    dump_edge_ops(edge, "edge_after_lowering")

    exec_prog = edge.to_executorch()
    out_path = (THIS.parent / "moe_split_portable.pte").resolve()
    with open(out_path, "wb") as f:
        exec_prog.write_to_file(f)
    print(f"[save] wrote {out_path}")


if __name__ == "__main__":
    main()
