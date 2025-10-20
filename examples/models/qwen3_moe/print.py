import torch
from collections import Counter
from executorch.exir.lowered_backend_module import LoweredBackendModule
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


def show_lowered_modules(edge_manager: EdgeProgramManager, list_ops_topk=8):
    """
    use official API to show backend info and op distribution of LoweredBackendModule
    """
    gm = next(iter(edge_manager._edge_programs.values())).graph_module
    print("\n=== LoweredBackendModules (backend & a peek at ops) ===")
    backend_counts = Counter()

    for name, mod in gm.named_modules():
        if isinstance(mod, LoweredBackendModule):
            backend = getattr(mod, "backend_id", "UNKNOWN")
            backend_counts[backend] += 1

            # count the ops in this partition (based on original_module)
            try:
                ep = mod.original_module  # ExportedProgram
                op_counter = Counter()
                for n in ep.graph.nodes:
                    if n.op == "call_function":
                        # unify to "aten.add.Tensor" / "aten.view_copy.default"
                        tgt = getattr(n.target, "__name__", str(n.target))
                        op_counter[tgt] += 1
                top_ops = ", ".join(
                    [f"{k}x{v}" for k, v in op_counter.most_common(list_ops_topk)]
                )
            except Exception:
                top_ops = "(ops unavailable)"

            print(f"{name:24s} | backend={backend:14s} | ops: {top_ops}")

    print("=== Module counts by backend ===")
    for b, c in backend_counts.items():
        print(f"{b:14s}: {c}")
