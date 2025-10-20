import torch
from executorch.exir import ExportedProgram, EdgeProgramManager


def diff_stats(name: str, a: torch.Tensor, b: torch.Tensor):
    d = (a - b).abs()
    print(
        f"[{name}] max|Δ|={d.max().item():.3e}  mean|Δ|={d.mean().item():.3e}  rms={torch.sqrt((d*d).mean()).item():.3e}"
    )


def dump_ops(gm_or_edge: ExportedProgram, tag: str = "exported_program"):
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


def dump_edge_ops(edge_mgr: EdgeProgramManager, tag: str = "edge_after_lowering"):
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
