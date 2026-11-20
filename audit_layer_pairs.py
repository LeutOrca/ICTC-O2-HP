"""Audit the frozen five-pair SO2Linear cohort on a Slurm CPU node."""
import argparse
import json
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cohort", type=Path)
    args = parser.parse_args()
    directory = args.cohort / "results"
    expected = [directory / f"{stage}-{mode}-{role}-{repeat}.json"
                for stage in ("time", "memory") for mode in ("forward", "input_vjp")
                for role in ("native", "own") for repeat in range(5)]
    missing = [str(path.relative_to(args.cohort)) for path in expected if not path.exists()]
    if missing:
        report = dict(status="incomplete", missing=missing)
        (directory / "PAIR_AUDIT.json").write_text(json.dumps(report, indent=2))
        (directory / "REPORT.zh-CN.md").write_text(
            "# 自写 SO2Linear 配对进度\n\n正式回执尚未齐全，详见 PAIR_AUDIT.json。\n"
        )
        print(json.dumps(report), flush=True)
        return
    qualifier = json.loads((directory / "so2-layer.json").read_text())
    if not qualifier.get("passed"):
        raise RuntimeError("The frozen complete-layer numerical gate failed")
    modes = {}
    for mode in ("forward", "input_vjp"):
        ratios, memory_ratios, reserved_ratios = [], [], []
        native_ms, own_ms, native_peak, own_peak = [], [], [], []
        uuids = set()
        for repeat in range(5):
            times = {role: json.loads((directory / f"time-{mode}-{role}-{repeat}.json").read_text())
                     for role in ("native", "own")}
            memories = {role: json.loads((directory / f"memory-{mode}-{role}-{repeat}.json").read_text())
                        for role in ("native", "own")}
            signatures = {(item["input_sha256"], item["weights_sha256"], item["gpu_uuid"],
                           item["source"], item["edges"], item["channels"], item["lmax"], item["mmax"])
                          for item in (*times.values(), *memories.values())}
            if len(signatures) != 1:
                raise RuntimeError(dict(mode=mode, repeat=repeat, mismatched_signatures=list(signatures)))
            uuids.add(times["own"]["gpu_uuid"])
            ratios.append(times["native"]["wall_ms"] / times["own"]["wall_ms"])
            memory_ratios.append(memories["own"]["peak_allocated"] / memories["native"]["peak_allocated"])
            reserved_ratios.append(memories["own"]["peak_reserved"] / memories["native"]["peak_reserved"])
            native_ms.append(times["native"]["wall_ms"])
            own_ms.append(times["own"]["wall_ms"])
            native_peak.append(memories["native"]["peak_allocated"])
            own_peak.append(memories["own"]["peak_allocated"])
        if len(uuids) != 1:
            raise RuntimeError("The five repetitions used different physical GPUs")
        speed = median(ratios)
        allocated = 1 - median(memory_ratios)
        reserved = 1 - median(reserved_ratios)
        modes[mode] = dict(wall_speedup=speed, speedup_pairs=ratios,
                           native_wall_ms=median(native_ms), own_wall_ms=median(own_ms),
                           allocated_reduction=allocated, reserved_reduction=reserved,
                           native_peak_allocated=median(native_peak), own_peak_allocated=median(own_peak),
                           speed_5x=speed >= 5, speed_6x=speed >= 6,
                           allocated_10pct=allocated >= .1, allocated_20pct=allocated >= .2,
                           gpu_uuid=next(iter(uuids)))
        trace = json.loads((directory / f"trace-{mode}-own.json").read_text())
        if trace.get("unexpected_library_gemm"):
            raise RuntimeError(trace)
    report = dict(status="complete", source=(args.cohort / "provenance/commit.txt").read_text().strip(),
                  modes=modes, numerical_gate=qualifier, tf32=False, amp=False, cuda_graph=False)
    (directory / "PAIR_AUDIT.json").write_text(json.dumps(report, indent=2))
    lines = ["# 自写 SO2Linear 五轮配对", "",
             f"快照为 `{report['source']}`。同一 A100、同源权重和输入哈希，严格 FP32、关闭 TF32/AMP/Graph。",
             "原生参考为官方 V3 SO2Linear，C128、L4、M2、160000 边。候选包含冻结复数权重打包、",
             "自写 IEEE FP32 矩阵核、跨阶输出组装及输入伴随。各臂采用独立进程，时间与显存分开测量。", "",
             "| 范围 | 原生→候选同步 ms | 五轮配对速度中位数 | allocated 下降 | reserved 下降 | 5× / 6× |",
             "|---|---|---|---|---|---|"]
    for mode, row in modes.items():
        lines.append(f"| {mode} | {row['native_wall_ms']:.3f}→{row['own_wall_ms']:.3f} | "
                     f"{row['wall_speedup']:.3f}× | {100*row['allocated_reduction']:.1f}% | "
                     f"{100*row['reserved_reduction']:.1f}% | {row['speed_5x']} / {row['speed_6x']} |")
    lines.extend(["", "本表覆盖完整 SO2Linear 及一次输入 VJP。整网 E/F 另由各自冻结 checkpoint 验收，"
                  "MACE 与 V3 的整网性能另有独立边界。", "候选 trace 中的矩阵 kernel 名和全部逐轮回执保存在本目录。"])
    (directory / "REPORT.zh-CN.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
