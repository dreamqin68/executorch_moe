import os, sys, ctypes, torch
from copy import deepcopy
from pathlib import Path

# ---------- Env & dlopen ----------
THIS = Path(__file__).resolve()  # .../c_moe_exe/test/run_pte.py
ROOT = THIS.parents[1]  # .../c_moe_exe
REPO = ROOT.parents[1]  # executorch repo root (optional)

env_libdir = os.environ.get("ET_LIBDIR", "")

candidates = [
    Path(env_libdir) if env_libdir else None,
    Path.home() / "et_xnn" / "lib",
]

LIBDIR = next((p for p in candidates if p and p.is_dir()), None)

PLUGIN = (ROOT / "exe_moe" / "build" / "libdeepseek_moe_execu.so").resolve()

if LIBDIR:
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if str(LIBDIR) not in ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{LIBDIR}:{ld}" if ld else str(LIBDIR)
        # Re-exec current script to make new LD_LIBRARY_PATH effective
        os.execv(sys.executable, [sys.executable] + sys.argv)

# Explicitly dlopen plugin to trigger EXECUTORCH_LIBRARY(...) static registration
ctypes.CDLL(str(PLUGIN), mode=ctypes.RTLD_GLOBAL)

os.environ.setdefault("ET_PREFERRED_BACKENDS", "xnnpack,portable")

# ---------- Paths ----------
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
PYT_MOE = ROOT / "pyt_moe"
if str(PYT_MOE) not in sys.path:
    sys.path.insert(0, str(PYT_MOE))

PTE = str(THIS.parent / "decoder_portable.pte")

# ---------- Imports ----------
import importlib
import torch.nn as nn
from executorch.runtime import Runtime

# ensure custom ops are loaded (pytorch side)
importlib.import_module("deepseek_moe")
import meta_kernels  # fake kernels for export

from executorch.examples.models.deepseek_vl2.models.configuration_deepseek import (
    DeepseekV2Config,
)
from executorch.examples.models.deepseek_vl2.models.modeling_deepseek import (
    DeepseekV2DecoderLayer,
)
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

from moe_wrapper import replace_layer_mlp_with_cpp  # cpp gate + cpp experts


# ---------- Helpers ----------
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


def build_layer(use_moe=True):
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
    replace_layer_mlp_with_cpp(layer_cpp).eval()
    return layer_py, layer_cpp, cfg


def diff_stats(a: torch.Tensor, b: torch.Tensor, name: str):
    d = (a - b).abs()
    print(
        f"[{name}] max|Δ|={d.max().item():.3e}  mean|Δ|={d.mean().item():.3e}  "
        f"rms={torch.sqrt((d*d).mean()).item():.3e}"
    )


# ---------- Inputs ----------
torch.manual_seed(0)
B, T, H, QK = 2, 16, 256, 32
x = torch.randn(B, T, H, dtype=torch.float32)
pos = torch.arange(T, dtype=torch.long).unsqueeze(0).expand(B, -1)

# ---------- Build eager & cpp ----------
layer_py, layer_cpp, cfg = build_layer(use_moe=True)
cos, sin = get_rope(layer_py, cfg, x, pos)

with torch.no_grad():
    (y_python,) = layer_py(x, position_embeddings=(cos, sin))
    (y_cpp,) = layer_cpp(x, position_embeddings=(cos, sin))

print("=== Python vs CPP ===")
diff_stats(y_python, y_cpp, "python vs cpp")

# ---------- (optional) rebuild .pte with portable backend ----------
REBUILD = os.environ.get("REBUILD_PTE", "0") == "1"
if REBUILD:
    print("\n[rebuild] REBUILD_PTE=1 -> export portable .pte with current weights")

    class DecoderWrapper(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, x, cos, sin):
            (y,) = self.layer(x, position_embeddings=(cos, sin))
            return y

    wrapped = DecoderWrapper(layer_cpp).eval()

    # 1) export
    exported_program = torch.export.export(wrapped, args=(x, cos, sin))

    # 2) execute exported graph in PyTorch, get y_export_cpp
    with torch.no_grad():
        y_export_cpp = exported_program.module()(x, cos, sin)

    # 3) edge transform
    edge = to_edge_transform_and_lower(
        exported_program,
        partitioner=[],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    # 4) write .pte
    exec_prog = edge.to_executorch()
    with open(PTE, "wb") as f:
        exec_prog.write_to_file(f)
    print(f"[save] wrote {PTE}")

    # consistency with cpp / python (exported eager replay)
    print("\n=== Export (portable) consistency ===")
    diff_stats(y_cpp, y_export_cpp, "cpp vs export_cpp")
    diff_stats(y_python, y_export_cpp, "python vs export_cpp")
else:
    y_export_cpp = None
    print(
        "\n[hint] not use REBUILD_PTE=1, so current .pte may not be the export result corresponding to the weights above."
    )

# ---------- Executorch runtime (portable) ----------
rt = Runtime.get()
(y_exec,) = rt.load_program(PTE).load_method("forward").execute([x, cos, sin])

print("\n=== Executorch (portable) output ===")
print(
    "exec shape/dtype:",
    tuple(y_exec.shape),
    y_exec.dtype,
    "mean=",
    float(y_exec.mean()),
)

# ---------- Consistency checks ----------
print("\n=== Consistency checks (portable) ===")
diff_stats(y_cpp, y_exec, "cpp vs exec_cpp")
diff_stats(y_python, y_exec, "python vs exec_cpp")

if y_export_cpp is not None:
    diff_stats(y_export_cpp, y_exec, "export_cpp vs exec_cpp")
else:
    print("[skip] export_cpp vs exec_cpp (not rebuild, skip)")
