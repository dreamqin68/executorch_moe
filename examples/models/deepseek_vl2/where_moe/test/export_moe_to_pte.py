import torch
from pathlib import Path
from executorch.examples.models.deepseek_vl2.models.configuration_deepseek import (
    DeepseekV2Config,
)
from executorch.examples.models.deepseek_vl2.models.modeling_deepseek import (
    DeepseekV2MoE,
)
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

from executorch.examples.models.deepseek_vl2.where_moe.moe_wrapper import (
    make_moe_with_where_infer,
    patch_gate_topk_where,
)
from executorch.examples.models.deepseek_vl2.utils.print import (
    diff_stats,
    dump_ops,
    dump_edge_ops,
)


def main():
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

    # moe_where = make_moe_with_where_infer(moe_py).eval()
    moe_where = patch_gate_topk_where(make_moe_with_where_infer(moe_py)).eval()

    x = torch.randn(2, 16, 256, dtype=torch.float32)
    out_py = moe_py(x)
    out_where = moe_where(x)

    print("=== EAGER (python vs where) ===")
    diff_stats("python vs where", out_py, out_where)

    # Export
    exported = torch.export.export(moe_where, args=(x,))
    gm = exported.module()

    dump_ops(exported, "exported_program")

    with torch.no_grad():
        y_export = gm(x)
    print("=== In-Python export (where vs export) ===")
    diff_stats("where vs export", out_where, y_export)

    edge = to_edge_transform_and_lower(
        exported,
        partitioner=[],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    dump_edge_ops(edge, "edge_after_lowering")

    exec_prog = edge.to_executorch()
    THIS = Path(__file__).resolve()
    out_path = (THIS.parent / "moe_split_portable.pte").resolve()
    with open(out_path, "wb") as f:
        exec_prog.write_to_file(f)
    print(f"[save] wrote {out_path}")


if __name__ == "__main__":
    main()
