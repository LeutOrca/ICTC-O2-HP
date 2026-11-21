"""Generate complete paired tables from independently audited receipts on CPU."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    summary = json.loads((root / "SUMMARY.json").read_text())
    assert summary["performance_complete"]
    lines = ["# V3 全 14 层修复：完整五轮同卡结果", "",
             "三臂分别为官方原生、c591b16 后四个 SO2 替换、全 14 个 SO2 候选。每轮在同一卡交替，阶段和各臂独立进程。",
             "固定拓扑不含邻居重建；e2e 从驻 GPU 的冻结输入开始，含官方构图和 E/F，不含文件载入/H2D。",
             "CUDA Graph 三臂均关闭，FP32、TF32/AMP 关闭，原数值阈值保持。性能中位数与数值 FP32 门控分开。", ""]
    for implementation in ("native-block-tree", "fused-block", "fused-compact"):
        pairs = [row for row in summary["whole_network_pairs"] if row["implementation"] == implementation]
        assert len(pairs) == 2 and all(row["status"] == "complete" for row in pairs)
        lines += [f"## {implementation}", "",
                  "| 输入/边界 | 原生 / c591b16 / 候选 ms | 对原生 / 对 c591b16 | 候选 atoms/s | 候选 structures/s |",
                  "|---|---:|---:|---:|---:|"]
        for row in pairs:
            for boundary, values in row["boundaries"].items():
                native, base, own = (values[role] for role in ("native", "base", "own"))
                lines.append(f"| {row['case']}/{boundary} | {native['wall_ms']:.4f} / {base['wall_ms']:.4f} / {own['wall_ms']:.4f} | "
                    f"{values['vs_native']['speedup']:.4f}× / {values['vs_base']['speedup']:.4f}× | {own['atoms_s']:.2f} | {own['structures_s']:.3f} |")
        lines += ["", "峰值为 PyTorch allocated/reserved，单位 MiB，包含各臂缓存权重。", "",
                  "| 输入/边界 | 原生 allocated / reserved | c591b16 allocated / reserved | 候选 allocated / reserved | 对原生减少 allocated / reserved | 对 c591b16 减少 allocated / reserved |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in pairs:
            for boundary, values in row["boundaries"].items():
                memories = [f"{values[role]['peak_allocated']/2**20:.2f} / {values[role]['peak_reserved']/2**20:.2f}"
                            for role in ("native", "base", "own")]
                reductions = [f"{values[ref]['allocated_reduction']*100:.2f}% / {values[ref]['reserved_reduction']*100:.2f}%"
                              for ref in ("vs_native", "vs_base")]
                lines.append(f"| {row['case']}/{boundary} | " + " | ".join([*memories, *reductions]) + " |")
        lines += ["", "同卡绑定：", ""]
        for row in pairs:
            lines.append(f"- {row['case']}: source `{row['source']}`，GPU `{row['gpu_uuid']}`，[原始审计](../{row['path']})。")
        lines += [""]
    (root / "results/FINAL-PAIRS.zh-CN.md").write_text("\n".join(lines) + "\n")
    gpu_ids = set((root / "provenance/GPU_JOBS.txt").read_text().split())
    with (root / "provenance/SLURM-RECEIPTS.tsv").open() as source:
        jobs = [row for row in csv.DictReader(source, delimiter="|") if row["JobIDRaw"] in gpu_ids]
    assert {row["JobIDRaw"] for row in jobs} == gpu_ids
    assert all(row["State"] not in ("RUNNING", "PENDING") for row in jobs)
    seconds = sum(int(row["ElapsedRaw"]) for row in jobs)
    resource = dict(gpu_hours=seconds/3600, gpu_jobs=len(jobs), nodes=1,
                    accounting="One requested GPU per recorded job, Slurm elapsed seconds; step rows excluded",
                    jobs=jobs)
    (root / "provenance/GPU_ACCOUNTING.json").write_text(json.dumps(resource, indent=2))
    print("\n".join(lines), flush=True)
    print(json.dumps(dict(gpu_hours=resource["gpu_hours"], gpu_jobs=len(jobs))), flush=True)


if __name__ == "__main__":
    main()
