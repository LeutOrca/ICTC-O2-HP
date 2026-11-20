"""Audit five physical-GPU pairs for frozen whole EquiformerV3 E/F."""
import argparse
import json
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cohort", type=Path)
    parser.add_argument("suite", choices=("mptrj-b1", "mptrj-b8"))
    args = parser.parse_args()
    root = args.cohort / "results"
    rows = {}
    for boundary in ("fixed", "e2e"):
        times, memories, speeds, allocated, reserved = [], [], [], [], []
        signatures = set()
        for repeat in range(5):
            timing = {role: json.loads((root / f"v3-time-{args.suite}-{boundary}-{role}-{repeat}.json").read_text())
                      for role in ("native", "own")}
            memory = {role: json.loads((root / f"v3-memory-{args.suite}-{boundary}-{role}-{repeat}.json").read_text())
                      for role in ("native", "own")}
            for value in (*timing.values(), *memory.values()):
                signatures.add(json.dumps({key: value[key] for key in (
                    "checkpoint_sha256", "source", "topology_sha256", "input_positions_sha256",
                    "input_manifest_sha256", "qualification_sha256", "gpu_uuid", "torch", "seed",
                    "cuda_graph", "tf32", "amp")}, sort_keys=True))
            speeds.append(timing["native"]["wall_ms"] / timing["own"]["wall_ms"])
            allocated.append(1 - memory["own"]["peak_allocated"] / memory["native"]["peak_allocated"])
            reserved.append(1 - memory["own"]["peak_reserved"] / memory["native"]["peak_reserved"])
            times.append(timing)
            memories.append(memory)
        if len(signatures) != 1:
            raise RuntimeError("The five pairs did not share frozen inputs, qualification and GPU")
        traces = {role: json.loads((root / f"v3-trace-{args.suite}-{boundary}-{role}.json").read_text())
                  for role in ("native", "own")}
        if traces["own"]["unexpected_library_matrix"]:
            raise RuntimeError(traces["own"]["unexpected_library_matrix"])
        row = dict(speedup=median(speeds), speedup_pairs=speeds,
                   allocated_reduction=median(allocated), reserved_reduction=median(reserved))
        for role in ("native", "own"):
            row[role] = {key: median(pair[role][key] for pair in times)
                         for key in ("wall_ms", "event_ms", "atoms_s", "structures_s")}
            row[role].update({key: median(pair[role][key] for pair in memories)
                              for key in ("peak_allocated", "peak_reserved")})
        row.update(speed_2x=row["speedup"] >= 2, allocated_10pct=row["allocated_reduction"] >= .1)
        rows[boundary] = row
    report = dict(suite=args.suite, boundaries=rows,
                  source=(args.cohort / "provenance/commit.txt").read_text().strip(),
                  status="complete", candidate_library_matrix_kernels=0)
    (root / f"V3_PAIR_AUDIT-{args.suite}.json").write_text(json.dumps(report, indent=2))
    lines = [f"# EquiformerV3 {args.suite} 整网 E/F 五轮同卡配对", "",
             f"快照 `{report['source']}`。同一 A100，冻结官方 checkpoint、EMA、输入及图规则。",
             "两臂关闭 TF32/AMP/Graph，严格 FP32。同步墙钟包含能量前向和位置伴随得到力。",
             "fixed 使用冻结边列表，e2e 包含官方邻居构建。显存单独在新进程测量，包含模型和打包缓存。", "",
             "| 边界 | 原生→候选 ms | 配对中位数 | 原生→候选 atoms/s | allocated MiB | reserved MiB |", "|---|---|---|---|---|---|"]
    for boundary, row in rows.items():
        a, b = row["native"], row["own"]
        lines.append(f"| {boundary} | {a['wall_ms']:.3f}→{b['wall_ms']:.3f} | {row['speedup']:.3f}× | "
                     f"{a['atoms_s']:.1f}→{b['atoms_s']:.1f} | "
                     f"{a['peak_allocated']/2**20:.1f}→{b['peak_allocated']/2**20:.1f} | "
                     f"{a['peak_reserved']/2**20:.1f}→{b['peak_reserved']/2**20:.1f} |")
    lines.extend(["", "本表仅覆盖本 suite。完整 SO2Linear 的 5–6× 算子门另列，不能替代整网或其他输入的结论。"])
    (root / f"REPORT.v3-{args.suite}.zh-CN.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
