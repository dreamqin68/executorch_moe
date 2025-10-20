import torch
from pathlib import Path
from executorch.examples.models.qwen3_moe.moe_wrapper import (
    make_moe_with_where_infer,
)
from executorch.examples.models.qwen3_moe.print import (
    diff_stats,
    dump_ops,
    dump_edge_ops,
    show_lowered_modules,
)

from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
    VulkanPartitioner,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


from executorch.runtime import Runtime

from transformers.models.qwen3_moe.configuration_qwen3_moe import (
    Qwen3MoeConfig,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
)


def main():
    B, T = 2, 16
    H = 256  # hidden_size
    I_MOE = 512  # moe_intermediate_size
    NEXP, K = 4, 2  # num_experts, num_experts_per_tok

    cfg = Qwen3MoeConfig(
        hidden_size=H,
        moe_intermediate_size=I_MOE,
        num_experts=NEXP,
        num_experts_per_tok=K,
    )

    moe_py = Qwen3MoeSparseMoeBlock(cfg).eval()
    moe_where = make_moe_with_where_infer(moe_py).eval()

    x = torch.randn(B, T, H)

    # --- Eager ---
    with torch.no_grad():
        out_py_hidden, out_py_logits = moe_py(x)
        out_where_hidden, out_where_logits = moe_where(x)

    print("=== EAGER (Original vs where) ===")
    diff_stats("Hidden:  Original vs where", out_py_hidden, out_where_hidden)
    diff_stats("Logits:  Original vs where", out_py_logits, out_where_logits)

    class _BothOutputs(torch.nn.Module):
        def __init__(self, block: torch.nn.Module):
            super().__init__()
            self.block = block.eval()

        def forward(self, x):
            y, logits = self.block(x)
            return y, logits

    both = _BothOutputs(moe_where).eval()

    # --- Export ---
    exported = torch.export.export(both, args=(x,))
    gm = exported.module()
    with torch.no_grad():
        y_export, logits_export = gm(x)  # now two outputs

    print("=== In-Python export (eager vs export) ===")
    diff_stats("Hidden:  where vs export", out_where_hidden, y_export)
    diff_stats("Logits:  where vs export", out_where_logits, logits_export)

    dump_ops(exported, "exported_program")
    # --- Edge Lowering ---
    edge_cfg = EdgeCompileConfig(_skip_dim_order=False, _check_ir_validity=False)
    edge = to_edge_transform_and_lower(
        exported,
        compile_config=edge_cfg,
        # partitioner=[XnnpackPartitioner(), VulkanPartitioner()],
        partitioner=[XnnpackPartitioner()],
        # partitioner=[VulkanPartitioner()],
        # partitioner=[VulkanPartitioner(), XnnpackPartitioner()],
    )

    dump_edge_ops(edge, "edge_after_lowering")

    # ---- debug: dump delegate information ----
    show_lowered_modules(edge)

    # --- Save .pte ---
    exec_prog = edge.to_executorch()
    THIS = Path(__file__).resolve()
    PTE = (THIS.parent / "qwen3_moe_sparse_xnnpack.pte").resolve()
    with open(PTE, "wb") as f:
        exec_prog.write_to_file(f)
    print(f"[save] wrote {PTE}")

    # --- ExecuTorch mix run ---
    rt = Runtime.get()
    prog = rt.load_program(str(PTE))
    method = prog.load_method("forward")

    y_exec, logits_exec = method.execute([x])
    print(f"[exec] y: {tuple(y_exec.shape)}, logits: {tuple(logits_exec.shape)}")

    print("\n=== Consistency vs ExecuTorch (Xnnpack) ===")
    diff_stats("Hidden:  where        vs exec", out_where_hidden, y_exec)
    diff_stats("Hidden:  export_where vs exec", y_export, y_exec)
    diff_stats("Hidden:  Original     vs exec", out_py_hidden, y_exec)
    diff_stats("Logits:  where        vs exec", out_where_logits, logits_exec)
    diff_stats("Logits:  export_where vs exec", logits_export, logits_exec)
    diff_stats("Logits:  Original     vs exec", out_py_logits, logits_exec)


if __name__ == "__main__":
    main()
