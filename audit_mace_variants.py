"""Audit same-card native/base/local-candidate pairs and their exact boundaries."""
import argparse
import json
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("case")
    parser.add_argument("candidate")
    parser.add_argument("--model", choices=("mace", "v3"), default="mace")
    args = parser.parse_args()
    directory = args.root / "results"
    roles = ("native", "base", args.candidate)
    if args.model == "mace" and args.candidate == "final-input" and any(
            directory.glob(f"mace-time-{args.case}-*-final-*.json")):
        roles = ("native", "base", "final", args.candidate)
    paths = {(stage, boundary, role, repeat): directory / f"{args.model}-{stage}-{args.case}-{boundary}-{role}-{repeat}.json"
             for stage in ("time", "memory") for boundary in ("fixed", "e2e")
             for role in roles for repeat in range(5)}
    missing = [path.name for path in paths.values() if not path.exists()]
    title = "MACE" if args.model == "mace" else "V3_LOCAL"
    output = directory / f"{title}_PAIR_AUDIT-{args.case}-{args.candidate}.json"
    if missing:
        output.write_text(json.dumps(dict(status="incomplete", missing=missing), indent=2))
        print(json.dumps(dict(status="incomplete", missing=len(missing))), flush=True)
        return
    receipts = {key: json.loads(path.read_text()) for key, path in paths.items()}
    input_keys = ("input_sha256",) if args.model == "mace" else (
        "input_positions_sha256", "topology_sha256", "input_manifest_sha256")
    signature_keys = ("source", "checkpoint_sha256", *input_keys, "gpu_uuid", "torch",
                      "qualification_sha256", "cuda_graph", "tf32", "amp", "dtype",
                      "atoms", "edges", "structures")
    signatures = {json.dumps({key: value[key] for key in signature_keys}, sort_keys=True)
                  for value in receipts.values()}
    if len(signatures) != 1:
        raise RuntimeError("The native and candidate cohorts differ in inputs, source or GPU")
    rows = {}
    for boundary in ("fixed", "e2e"):
        row = {}
        for role in roles:
            timing = [receipts[("time", boundary, role, repeat)] for repeat in range(5)]
            memory = [receipts[("memory", boundary, role, repeat)] for repeat in range(5)]
            row[role] = {key: median(value[key] for value in timing)
                         for key in ("wall_ms", "atoms_s", "structures_s")}
            row[role].update({key: median(value[key] for value in memory)
                              for key in ("peak_allocated", "peak_reserved")})
        for reference in roles[:-1]:
            speed, allocated, reserved = [], [], []
            for repeat in range(5):
                speed.append(receipts[("time", boundary, reference, repeat)]["wall_ms"] /
                             receipts[("time", boundary, args.candidate, repeat)]["wall_ms"])
                allocated.append(1 - receipts[("memory", boundary, args.candidate, repeat)]["peak_allocated"] /
                                 receipts[("memory", boundary, reference, repeat)]["peak_allocated"])
                reserved.append(1 - receipts[("memory", boundary, args.candidate, repeat)]["peak_reserved"] /
                                receipts[("memory", boundary, reference, repeat)]["peak_reserved"])
            row[f"vs_{reference}"] = dict(speedup=median(speed), paired_speedups=speed,
                                          allocated_reduction=median(allocated), reserved_reduction=median(reserved),
                                          speed_2x=median(speed) >= 2, allocated_10pct=median(allocated) >= .1)
        rows[boundary] = row
    report = dict(status="complete", model=args.model, candidate=args.candidate, case=args.case, boundaries=rows,
                  source=next(iter(receipts.values()))["source"], gpu_uuid=next(iter(receipts.values()))["gpu_uuid"],
                  atoms=next(iter(receipts.values()))["atoms"], edges=next(iter(receipts.values()))["edges"],
                  structures=next(iter(receipts.values()))["structures"], torch=next(iter(receipts.values()))["torch"],
                  tf32=False, amp=False, cuda_graph=next(iter(receipts.values()))["cuda_graph"])
    output.write_text(json.dumps(report, indent=2))
    base = "78231f3" if args.model == "mace" else "9787a3a"
    graph_boundary = "包括 Graph 捕获与回放" if report["cuda_graph"] else "为预热后单次完整 E/F"
    lines = [f"# {title} {args.case} / {args.candidate} 五轮同卡整网配对", "",
             f"严格 FP32、同 Graph 设置、同冻结 checkpoint 和输入，峰值显存{graph_boundary}。",
             "每个阶段独立进程，每臂重新加载模型并清理前一臂分配，五轮交替。", "",
             f"| 边界 | 原生→{base}→候选 ms | 候选对原生 / 对{base} | 候选 allocated / reserved MiB |",
             "|---|---|---|---|"]
    for boundary, row in rows.items():
        candidate = row[args.candidate]
        lines.append(f"| {boundary} | {row['native']['wall_ms']:.4f}→{row['base']['wall_ms']:.4f}→{candidate['wall_ms']:.4f} | "
                     f"{row['vs_native']['speedup']:.4f}× / {row['vs_base']['speedup']:.4f}× | "
                     f"{candidate['peak_allocated']/2**20:.2f} / {candidate['peak_reserved']/2**20:.2f} |")
    if "final" in roles[:-1]:
        lines += ["", "输入优化也直接与联合版本 final 逐轮配对："]
        for boundary, row in rows.items():
            lines.append(f"- {boundary}: {row['vs_final']['speedup']:.5f}×，"
                         f"allocated 下降 {row['vs_final']['allocated_reduction']*100:.4f}%，"
                         f"reserved 下降 {row['vs_final']['reserved_reduction']*100:.4f}%。")
    (directory / f"REPORT.{args.model}-{args.case}-{args.candidate}.zh-CN.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
