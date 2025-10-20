import os, sys, ctypes, torch, importlib
from pathlib import Path

os.environ["ET_PREFERRED_BACKENDS"] = "portable"

# ---------- Paths ----------
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]

env_libdir = os.environ.get("ET_LIBDIR", "")

candidates = [
    Path(env_libdir) if env_libdir else None,
    Path.home() / "et_xnn" / "lib",
]

LIBDIR = next((p for p in candidates if p and p.is_dir()), None)

PYT_MOE = ROOT / "pyt_moe"
if str(PYT_MOE) not in sys.path:
    sys.path.insert(0, str(PYT_MOE))

PTE = str(THIS.parent / "moe_split_portable.pte")
PLUGIN = (ROOT / "exe_moe" / "build" / "libdeepseek_moe_execu.so").resolve()
CORE = (LIBDIR / "libexecutorch_core.so") if LIBDIR else None

if LIBDIR:
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if str(LIBDIR) not in ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{LIBDIR}:{ld}" if ld else str(LIBDIR)
        os.execv(sys.executable, [sys.executable] + sys.argv)

CORE_HANDLE = None
if CORE and CORE.exists():
    CORE_HANDLE = ctypes.CDLL(str(CORE), mode=ctypes.RTLD_GLOBAL)
    print(f"[dlopen] keep core resident: {CORE.name} -> {CORE_HANDLE}")

PLUGIN_HANDLE = ctypes.CDLL(str(PLUGIN), mode=ctypes.RTLD_GLOBAL)
print(f"[dlopen] keep plugin resident: {PLUGIN.name} -> {PLUGIN_HANDLE}")

# ---------- Runtime ----------
from executorch.runtime import Runtime

rt = Runtime.get()

# ---------- import meta_kernels for export ----------
try:
    importlib.import_module("deepseek_moe_split")
    import meta_kernels  # fake kernels for export
except Exception as e:
    print(f"[warn] meta_kernels not found: {e}")


def diff_stats(a: torch.Tensor, b: torch.Tensor, name: str):
    d = (a - b).abs()
    print(
        f"[{name}] max|Δ|={d.max().item():.3e}  mean|Δ|={d.mean().item():.3e}  "
        f"rms={torch.sqrt((d*d).mean()).item():.3e}"
    )


def main():
    print("=== ExecuTorch MoE Portable Runner ===")
    print(f"[config] plugin={PLUGIN}")
    print(f"[config] pte={PTE}")

    # Create test input
    B, T, H = 2, 16, 256
    x = torch.randn(B, T, H, dtype=torch.float32)
    print(f"[input] created tensor: {x.shape}, {x.dtype}")

    from executorch.examples.models.deepseek_vl2.models.configuration_deepseek import (
        DeepseekV2Config,
    )
    from executorch.examples.models.deepseek_vl2.models.modeling_deepseek import (
        DeepseekV2MoE,
    )

    from moe_wrapper import make_moe_with_cpp_infer, patch_gate_topk_inplace

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

    # eager pure Python
    moe_py = DeepseekV2MoE(cfg).eval()
    out_py = moe_py(x)  # (B,T,H)

    # cpp version (C++ moe_infer + C++ topk_select)
    moe_cpp = patch_gate_topk_inplace(make_moe_with_cpp_infer(moe_py)).eval()
    out_cpp = moe_cpp(x)

    # export version (run exported GraphModule in Python)
    with torch.no_grad():
        exported = torch.export.export(moe_cpp, args=(x,))
        out_export = exported.module()(x)

    # === 2) Optional: Re-export .pte to ensure weight consistency ===
    if os.environ.get("REBUILD_PTE", "0") == "1":
        from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        exec_prog = edge.to_executorch()
        with open(PTE, "wb") as f:
            exec_prog.write_to_file(f)
        print(f"[save] wrote {PTE}")

    # === 3) ExecuTorch portable execution ===
    prog = rt.load_program(str(PTE))
    method = prog.load_method("forward")
    (y_exec,) = method.execute([x])
    print(
        f"[exec] ExecuTorch output: {y_exec.shape}, {y_exec.dtype}, mean={float(y_exec.mean()):.6f}"
    )

    # === 4) Print comparison ===
    print("\n=== Consistency vs ExecuTorch (portable) ===")
    diff_stats(out_cpp, y_exec, "cpp        vs exec")
    diff_stats(out_export, y_exec, "export_cpp vs exec")
    diff_stats(out_py, y_exec, "eager      vs exec")


if __name__ == "__main__":
    main()
