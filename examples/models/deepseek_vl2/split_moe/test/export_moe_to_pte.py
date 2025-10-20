import sys, importlib, torch
from pathlib import Path

THIS = Path(__file__).resolve()
PYT_MOE = THIS.parents[1] / "pyt_moe"
if str(PYT_MOE) not in sys.path:
    sys.path.insert(0, str(PYT_MOE))


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
    ok_moe_func = torch._C._dispatch_has_kernel_for_dispatch_key(
        "deepseek_moe_split::moe_infer_glu", "CPU"
    )
    ok_moe_out = torch._C._dispatch_has_kernel_for_dispatch_key(
        "deepseek_moe_split::moe_infer_glu.out", "CPU"
    )
    ok_topk_func = torch._C._dispatch_has_kernel_for_dispatch_key(
        "deepseek_moe_split::topk_select", "CPU"
    )
    ok_topk_out = torch._C._dispatch_has_kernel_for_dispatch_key(
        "deepseek_moe_split::topk_select.out", "CPU"
    )

    if not (ok_moe_func and ok_moe_out and ok_topk_func and ok_topk_out):
        raise RuntimeError(
            f"CPU kernels not found: moe_infer_glu={ok_moe_func}/{ok_moe_out}, "
            f"topk_select={ok_topk_func}/{ok_topk_out}"
        )


_load_split_op()

import meta_kernels
from moe_wrapper import make_moe_with_cpp_infer, patch_gate_topk_inplace

from executorch.examples.models.deepseek_vl2.models.configuration_deepseek import (
    DeepseekV2Config,
)
from executorch.examples.models.deepseek_vl2.models.modeling_deepseek import (
    DeepseekV2MoE,
)
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig


def _dump_ops(tag, gm_or_edge):
    print(f"\n==== OPS @ {tag} ====")
    try:
        # 1) torch.export ExportedProgram
        gm = gm_or_edge.graph_module
        for n in gm.graph.nodes:
            print(" ", n.op, n.target)
    except Exception:
        # 2) EdgeProgramManager（subgraphs）
        for name, gm in getattr(gm_or_edge, "graph_modules", {}).items():
            print(" [subgraph]", name)
            for n in gm.graph.nodes:
                print("  ", n.op, n.target)


def dump_edge_ops(edge_mgr, tag="edge_after_lowering"):
    print(f"\n==== OPS @ {tag} ====")
    # single subgraph (common)
    if hasattr(edge_mgr, "exported_program"):
        ep = edge_mgr.exported_program()
        gm = ep.module()
        for n in gm.graph.nodes:
            print(" ", n.op, n.target)
        return
    # multiple subgraphs (some partitioners will produce multiple)
    if hasattr(edge_mgr, "exported_programs"):
        for i, ep in enumerate(edge_mgr.exported_programs()):
            gm = ep.module()
            print(f" [subgraph {i}]")
            for n in gm.graph.nodes:
                print("  ", n.op, n.target)
        return
    print("  (EdgeProgramManager has no exported_program{,s} interface)")


def _diff(name, a, b):
    d = (a - b).abs()
    print(
        f"[{name}] max|Δ|={d.max().item():.3e}  mean|Δ|={d.mean().item():.3e}  rms={torch.sqrt((d*d).mean()).item():.3e}"
    )


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
    moe_cpp = patch_gate_topk_inplace(make_moe_with_cpp_infer(moe_py)).eval()
    # moe_cpp = make_moe_with_cpp_infer(moe_py).eval()

    B, T, H = 2, 16, cfg.hidden_size
    x = torch.randn(B, T, H)

    with torch.no_grad():
        y_py = moe_py(x)
        y_cpp = moe_cpp(x)

    print("=== EAGER (python vs cpp_infer) ===")
    _diff("python vs cpp_infer", y_py, y_cpp)

    # Verify that the patched model actually uses custom ops
    print("\n[verify] Running dummy forward to confirm custom ops are used...")
    with torch.no_grad():
        dummy_out = moe_cpp(x)
    print("[verify] Dummy forward completed successfully")

    ep = torch.export.export(moe_cpp, args=(x,))
    gm = ep.module()

    # Dump ops at exported_program stage
    _dump_ops("exported_program", ep)

    # B. Confirm that FX graph only shows .out forms
    # print("\n[graph]")
    # print(gm.graph)

    with torch.no_grad():
        y_export = gm(x)
    print("=== In-Python export (cpp vs export) ===")
    _diff("cpp vs export", y_cpp, y_export)

    # A. Self-check prints before export
    print(
        "\n[check] deepseek_moe_split ops:",
        [n for n in dir(torch.ops.deepseek_moe_split) if not n.startswith("_")],
    )

    print(
        "[check] has moe_infer_glu.out   :",
        hasattr(torch.ops.deepseek_moe_split.moe_infer_glu, "out"),
    )
    print(
        "[check] has topk_select.out      :",
        hasattr(torch.ops.deepseek_moe_split.topk_select, "out"),
    )

    print(
        "[check] schema moe_infer_glu.out :",
        torch.ops.deepseek_moe_split.moe_infer_glu.out._schema,
    )
    print(
        "[check] schema topk_select.out   :",
        torch.ops.deepseek_moe_split.topk_select.out._schema,
    )

    edge = to_edge_transform_and_lower(
        ep, partitioner=[], compile_config=EdgeCompileConfig(_check_ir_validity=False)
    )

    # Dump ops at edge_after_lowering stage
    dump_edge_ops(edge)
    exec_prog = edge.to_executorch()
    out_path = (THIS.parent / "moe_split_portable.pte").resolve()
    with open(out_path, "wb") as f:
        exec_prog.write_to_file(f)
    print(f"[save] wrote {out_path}")


if __name__ == "__main__":
    main()
