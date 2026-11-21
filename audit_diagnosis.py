"""Summarize the layerwise diagnosis using only scheduler-managed CPU work."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    lines = ["# V3 前向与输入反向诊断", "",
             f"source `{report['source']}`，GPU `{report['gpu_uuid']}`，seed {report['seed']}。", "",
             "## 全 14 个交叉消融", "", "| 模式 | 力 relative L2 | max abs | 通过 |", "|---|---|---|---|"]
    for mode, errors in report["whole"].items():
        force = errors["forces"]
        lines.append(f"| {mode} | {force['relative_l2']:.8g} | {force['max_abs']:.8g} | {force['passed']} |")
    lines += ["", "## 相同原生输入和余切的逐层误差", "",
              "| 层 | 前向 relative L2 | 输入 VJP relative L2 | 前向 m0/m1/m2 | VJP m0/m1/m2 |", "|---|---|---|---|---|"]
    for name, row in report["local"].items():
        forward = max(value["relative_l2"] for value in row["forward"])
        part_f = "/".join(f"{value['relative_l2']:.3g}" for value in row["forward_by_order"].values())
        part_b = "/".join(f"{value['relative_l2']:.3g}" for value in row["input_vjp_by_order"].values())
        lines.append(f"| {name} | {forward:.8g} | {row['input_vjp']['relative_l2']:.8g} | {part_f} | {part_b} |")
    lines += ["", "## 单层交叉消融", "", "| 层 | 原生前向/自写反向 | 自写前向/原生反向 | 自写两向 |", "|---|---|---|---|"]
    for name, row in report["individual"].items():
        values = [row[key]["forces"]["relative_l2"] for key in (
            "primal_native-adjoint_own", "primal_own-adjoint_native", "primal_own-adjoint_own")]
        lines.append(f"| {name} | " + " | ".join(f"{value:.8g}" for value in values) + " |")
    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]), flush=True)


if __name__ == "__main__":
    main()
