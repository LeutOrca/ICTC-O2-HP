"""Audit all native/base/prefix whole-network receipts on a compute node."""
import argparse
import json
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("case")
    args = parser.parse_args()
    results = args.root / "results"
    modes = {}
    for boundary in ("fixed", "e2e"):
        rows = {}
        signatures = set()
        items = {}
        for stage in ("time", "memory"):
            for role in ("native", "base", "prefix"):
                items[(stage, role)] = []
                for repeat in range(5):
                    path = results / f"mace-{stage}-{args.case}-{boundary}-{role}-{repeat}.json"
                    value = json.loads(path.read_text())
                    signatures.add(tuple(value[key] for key in (
                        "source", "checkpoint_sha256", "input_sha256", "gpu_uuid", "torch",
                        "qualification_sha256", "cuda_graph", "tf32", "amp")))
                    items[(stage, role)].append(value)
        if len(signatures) != 1:
            raise RuntimeError("Paired roles used different hardware, inputs or qualification")
        for role in ("native", "base", "prefix"):
            timing, memory = items[("time", role)], items[("memory", role)]
            rows[role] = {key: median(row[key] for row in timing)
                          for key in ("wall_ms", "atoms_s", "structures_s", "capture_prepare_wall_ms")}
            rows[role].update({key: median(row[key] for row in memory)
                              for key in ("peak_allocated", "peak_reserved")})
        for reference in ("native", "base"):
            speeds = [a["wall_ms"] / b["wall_ms"] for a, b in zip(
                items[("time", reference)], items[("time", "prefix")])]
            allocated = [1 - b["peak_allocated"] / a["peak_allocated"] for a, b in zip(
                items[("memory", reference)], items[("memory", "prefix")])]
            reserved = [1 - b["peak_reserved"] / a["peak_reserved"] for a, b in zip(
                items[("memory", reference)], items[("memory", "prefix")])]
            rows[f"prefix_vs_{reference}"] = dict(speedup=median(speeds), speedup_pairs=speeds,
                                                   allocated_reduction=median(allocated),
                                                   reserved_reduction=median(reserved))
        modes[boundary] = rows
    report = dict(case=args.case, boundaries=modes, status="complete")
    (results / f"MACE_PREFIX_PAIR_AUDIT-{args.case}.json").write_text(json.dumps(report, indent=2))
    lines = [f"# MACE {args.case} 前缀缓存五轮三臂配对", "",
             "三臂同一 GPU、严格 FP32、同 Graph、同冻结 checkpoint/输入，含捕获峰值显存。",
             "base 为冻结的 78231f3，prefix 为 base 加常量前缀缓存。准备成本另列于 JSON 回执。", "",
             "| 边界 | 原生→base→prefix ms | prefix 对原生 / base | prefix allocated / reserved MiB |",
             "|---|---|---|---|"]
    for name, row in modes.items():
        lines.append(f"| {name} | {row['native']['wall_ms']:.4f}→{row['base']['wall_ms']:.4f}→{row['prefix']['wall_ms']:.4f} | "
                     f"{row['prefix_vs_native']['speedup']:.4f}× / {row['prefix_vs_base']['speedup']:.4f}× | "
                     f"{row['prefix']['peak_allocated']/2**20:.2f} / {row['prefix']['peak_reserved']/2**20:.2f} |")
    (results / f"REPORT.mace-prefix-{args.case}.zh-CN.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
