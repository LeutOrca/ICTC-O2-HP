"""Write evidence-bounded Chinese tables from the audited CPU summary."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    data = json.loads((ROOT / "SUMMARY.json").read_text())
    rows = data["summary"]
    lookup = {(r["runtime"], r["role"], r["case"], r["boundary"]): r for r in rows}
    cases = ("b1", "b8", "n512", "n4096")
    names = dict(b1="B1 / 6 原子", b8="B8 / 68 原子", n512="600 原子", n4096="4100 原子")
    text = ["# 当前 GitHub O2-HP 与同期库：MACE 整网 E/F 比较", "",
        "本轮为新整网比较测试，未改变本地 HPX 核算法。官方 main `96815838` 与 fork `296e61f1` 仅 README 不同，可执行源码一致。公共库没有现成 MACE/V3 转换器，本轮新接入仅替换 MACE 首层标量 TP 的四个 l 路径；其结果属于这份零填充适配方案，不能概括为库的最佳整网性能。本地 `c591b16` HPX 的替换范围更大，单独列示。", "",
        "## 冻结与数值口径", "",
        "MACE-OMAT-0 medium checkpoint SHA256 `d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a`。固定输入、权重、6 Å 周期邻居规则。验收总能、相互作用能和坐标力；模型、参考与误差统计均 FP32，逐分量 `2e-5 + 2e-5|reference|`，relative L2 ≤ `1e-5`。力 RMS < `1e-3` 时沿用原近零规则。TF32、AMP 关闭，没有 FP64 数值参考或 FP64 误差统计。上游离线 CG 系数构造与原生邻居 API 的原实现保留，均不冒充 FP64 推理。", "",
        "原生 MACE 为整数角阶、检查点自然宇称；公共适配使用 canonical 整数 O(2) 块完成首层标量恒等嵌入，不是 full-parity 覆盖测试，无半整数编译。MACE 所有可接受对照均启用 CUDA Graph。训练继续暂停。", "",
        "## 测量边界", "",
        "GBU2 H100 80GB，同一负载的各臂在同一个 GPU UUID 上顺序配对。Torch 2.6 主线与 Torch 2.11 新库运行时分别建立原生分母，不跨版本相除。每臂 5 个独立计时进程，角色与两种边界顺序交替；每边界预热 5 次、同步墙钟采样 15 次，报告轮次中位数及配对比值中位数。固定图从 GPU 输入开始；含构图从 CPU 构型开始，包含原生邻居构建、输入装配、H2D 和整网 E/F。排除文件 I/O、编译和一次捕获。", "",
        "峰值 allocated / reserved 分别以独立新进程测量，每臂每边界 1 次，包含 Graph 捕获、常驻缓冲和回放，不与资格阶段峰值混算。4100 数值参考采用原验证的 128 通道分块、与冻结原生 FP32 参考逐位核对；计时及显存分母保持未分块原生 Graph，并与分块参考逐位一致。OOM、接口不兼容和数值失败分列，无零吞吐填值。", "",
        "## 本地 HPX 对同卡原生 Graph", ""]
    for boundary, label in (("fixed", "固定拓扑"), ("e2e", "含邻居构建")):
        text.extend([f"### {label}", "", "| 负载 | 原生 → HPX ms | 配对加速（IQR） | HPX atoms/s | HPX 构型/s | allocated MiB 原生 → HPX | reserved MiB 原生 → HPX |", "|---|---:|---:|---:|---:|---:|---:|"])
        for case in cases:
            key = ("2.6.0+cu124", "local-hpx", case, boundary)
            if key not in lookup:
                text.append(f"| {names[case]} | 未完成有效配对 | — | — | — | — | — |")
                continue
            r = lookup[key]
            b = lookup[r["runtime"], "native", case, boundary]
            text.append(f"| {names[case]} | {b['latency_ms']:.3f} → {r['latency_ms']:.3f} | {r['speedup']:.3f}× ({r['speedup_q25']:.3f}–{r['speedup_q75']:.3f}) | {r['atoms_s']:.1f} | {r['structures_s']:.3f} | {b['peak_allocated_mib']:.2f} → {r['peak_allocated_mib']:.2f} | {b['peak_reserved_mib']:.0f} → {r['peak_reserved_mib']:.0f} |")
        text.append("")
    text.extend(["### 端到端速度与显存目标分别验收", "", "| 负载 | 端到端加速 | allocated 降幅 | reserved 降幅 | ≥2× 且两类显存均降 ≥10% |", "|---|---:|---:|---:|---|"])
    for case in cases:
        r = lookup.get(("2.6.0+cu124", "local-hpx", case, "e2e"))
        if r:
            passed = r['speedup'] >= 2 and r['allocated_drop_pct'] >= 10 and r['reserved_drop_pct'] >= 10
            text.append(f"| {names[case]} | {r['speedup']:.3f}× | {r['allocated_drop_pct']:.2f}% | {r['reserved_drop_pct']:.2f}% | {'通过' if passed else '未全部达到'} |")
    text.extend(["", "B8 的 2× 目标绑定实际复测结果；不以历史其他节点的结果替代本轮未达门结论。以下所有绝对值与比值都来自本轮，未混入旧微基准。", "",
                 "## 公共 GitHub 首层 TP 适配", "",
                 "| 来源 | 边界 | B1 延迟 ms | 对原生加速 | atoms/s | 构型/s | allocated / reserved MiB |",
                 "|---|---|---:|---:|---:|---:|---:|"])
    for role in ("official", "fork"):
        for boundary in ("fixed", "e2e"):
            r = lookup.get(("2.6.0+cu124", role, "b1", boundary))
            if r:
                text.append(f"| {role} | {boundary} | {r['latency_ms']:.3f} | {r['speedup']:.3f}× | {r['atoms_s']:.1f} | {r['structures_s']:.2f} | {r['peak_allocated_mib']:.2f} / {r['peak_reserved_mib']:.0f} |")
    text.extend(["", "两个 main 的源码相同，独立进程间的小幅差异属于测量波动，不是算法更新收益。B8 / 600 的力 relative L2 约 `4.15e-5` / `2.59e-5`，均超过原 `1e-5` 门；4100 适配资格阶段 OOM。这些点不报告有效加速或吞吐。该接入需要扩展均匀表示和零权重路径，适配代价已包含在整网时间和显存中。", "",
        "## 同期库", "",
        "EquivariantX/TACE 当前冻结 `c646c7000af7adb367d6bd8b7eb5ea74c08dc37c`，cuEquivariance 当前为 `0.13.0`。旧 Torch 2.6 的接口/版本不兼容作为软件状态记录；另建隔离 Torch 2.11 运行时验证，不替换为旧库而冒称最新。新运行时须建立本版本原生 Graph 分母，并额外按原 FP32 门核对冻结 Torch 2.6 参考。当前库完整原始状态见 `results/status.csv` 与 `measurements/`。", "",
        "| 运行时 | 实现 | 负载 | 边界 | 延迟 ms | 对匹配原生加速 | atoms/s | 构型/s | allocated / reserved MiB |", "|---|---|---|---|---:|---:|---:|---:|---:|"])
    contemporary = [r for r in rows if r['role'] in ("eqx-current", "cue-current")]
    for r in contemporary:
        text.append(f"| {r['runtime']} | {r['role']} | {names[r['case']]} | {r['boundary']} | {r['latency_ms']:.3f} | {r['speedup']:.3f}× | {r['atoms_s']:.1f} | {r['structures_s']:.2f} | {r['peak_allocated_mib']:.2f} / {r['peak_reserved_mib']:.0f} |")
    if not contemporary:
        text.append("| — | 当前库未形成有效同口径配对 | — | — | — | — | — | — | — |")
    text.extend(["", "没有用不同模型或不同方程的 eSCN、其他 MLIP 绝对时间作本检查点整网加速分母。不同架构不能满足相同权重和逐点 E/F 配对约束。", "",
        "## V3 保留范围", "",
        "本轮未将公共 O2-HP 替换到 V3 的 SO2Linear。最新 V3 仍为交付快照 `4b17046`、已测核心 `efb5a27`：全 14 个自写 SO2Linear 已过 FP32 E/F，原生自然宇称、整数角阶、CUDA Graph 关闭；A100 B1/B8 含构图历史配对为 `1.0814× / 1.1270×`。这些历史结果独立标注，不进入本轮 H100 MACE 曲线或分母。", "",
        "## 图、数据与恢复", "",
        "图宽 150 mm，Times New Roman，无框图例。横轴是不同化学组成/批次的离散负载，不宣称受控原子数缩放。失败与 OOM 为缺口，不插值，不填 0。5 轮中位数及 IQR 在 CSV 中完整保留。", "",
        "![整网加速与吞吐](../../figures/o2-hp-github-whole-20261121-r1/Whole_MACE_speed_throughput.png)", "",
        "![峰值显存](../../figures/o2-hp-github-whole-20261121-r1/Whole_MACE_peak_memory.png)", "",
        "[CSV](results/summary.csv) · [逐项状态](results/status.csv) · [机器汇总](SUMMARY.json) · [实时源码身份](provenance/SOURCES.md) · [合同](CONTRACT.md)", "",
        "新驱动与每项修复按 Git commit 保存；远端分别使用带 hash 的独立目录。公开上游源码和本地 c591b16 载荷均独立冻结。实际测量源码、作业、GPU UUID、输入/检查点/资格回执 SHA256 均可追溯。Tavotto 编辑记录及导出证明另存于图项目。", ""])
    (ROOT / "REPORT.zh-CN.md").write_text("\n".join(text))
    print("Report written", len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()
