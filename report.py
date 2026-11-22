"""Generate evidence-bounded Chinese results text on a Slurm CPU node."""
import csv
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent


def main():
    data = list(csv.DictReader((ROOT/"summary.csv").open()))
    ratios = list(csv.DictReader((ROOT/"speedups.csv").open()))
    summary = json.loads((ROOT/"SUMMARY.json").read_text())
    coverage = list(csv.DictReader((ROOT/"coverage.csv").open()))
    role_coverage = list(csv.DictReader((ROOT/"role-coverage.csv").open()))
    role_states = Counter(row["status"] for row in role_coverage)
    budget = json.loads((ROOT/"provenance/BUDGET-AUDIT.json").read_text())
    states = Counter(row["status"] for row in coverage)
    text = ["# 算子吞吐随角向阶数变化：H100 同卡配对", "",
        f"当前已汇总 **{summary['measured_configurations']}/88 个配置**、{summary['summary_points']} 个合格曲线点。"
        f"仍待运行配置 {len(summary['pending_configurations'])} 个；配置状态统计 `{dict(states)}`。"
        "measured 仅表示该配置至少一臂有有效数据，不表示所有角色/边界通过。88 个位置含原生 ICTP 不支持的 4–8 阶。",
        f"逐角色/边界共 {len(role_coverage)} 项，终态为 `{dict(role_states)}`，待测 {summary['pending_role_metrics']} 项。"
        "收集完成不代表所有实现通过数值门或构建成功。有效点是五个独立进程的中位数，CSV 同时保留 IQR。",
        "2026-11-22 07:00 自动收集115550已回收到全部测量、日志与远端账本，随后本地记账库连接失败，未生成汇总。"
        "本轮恢复了本地账本，并由CPU115739重新汇总。GBU2当前SSH重连失败，远端终态依据已回收的07:00账本；所有本轮GPU作业已终止。",
        "", "## 读取图表", "",
        "横轴为角向截断 1–8，固定 C64。节点 TP 固定 N4096，边 TP、UVU/RadialUVU、SO2Linear 固定 E16384。"
        "原子/s 是逐节点特征算子的行吞吐，不含邻居构建或整网 E/F；边/s 是逐边算子的行吞吐。"
        "O2 的角阶是 Mmax，SO2Linear 是 Lmax=Mmax，ICTP 是 Lmax。每个角阶使用可复现的随机特征和固定随机权重，各实现共享同一逻辑输入/权重；不同角阶权重形状不同。"
        "这些是独立算子图，不是官方 checkpoint 整网 E/F 的速度图。不同方程和存储宽度只列绝对值，不计算跨方程加速。",
        "", "![节点 TP 吞吐](../../figures/o2-hp-angular-operators-20261121-r1/O2_TP_atoms.png)",
        "", "![边 TP 吞吐](../../figures/o2-hp-angular-operators-20261121-r1/O2_TP_edges.png)",
        "", "![UVU 与径向融合 canonical](../../figures/o2-hp-angular-operators-20261121-r1/O2_UVU_Radial_canonical.png)",
        "", "![UVU 与径向融合 full](../../figures/o2-hp-angular-operators-20261121-r1/O2_UVU_Radial_full.png)",
        "", "![SO2Linear](../../figures/o2-hp-angular-operators-20261121-r1/SO2Linear_edges.png)",
        "", "![ICTP 原生](../../figures/o2-hp-angular-operators-20261121-r1/ICTP_atoms_edges.png)",
        "", "## 版本、方程与精度", "",
        "- O2-HP official main `96815838f470652dce3d0e9d5cf7fe2978e949ca`、fork `296e61f1c13b8ad5ff525cbe3d0bd63d4d6adaca`。"
        "14 份可执行源码逐字节一致，所以同一曲线明确标注 official (=fork)，不是两个不同算法。",
        "- Eqx-O2/TACE `c646c7000af7adb367d6bd8b7eb5ea74c08dc37c`，CUDA 后端，包名 equivariantx。"
        "UUU 使用相同路径与权重；V=1 UVU 精确映射为 u1u。表示、符号和权重变换在准备阶段，核消费各自原生 API 布局。",
        "- cuEq 0.13.0：exact O2 descriptor，uniform_1d，math_dtype=float32，仅作同方程 UUU 对照。",
        "- 本地 HPX `5f278ef91bedece75b0558700afdb76bcf267a1b`：独立输出 preload，保持原路径和权重。"
        "冻结权重适配器只请求输入伴随 mask=3，以跳过原库未被本任务使用的 FP64 参数梯度。它明确是输入反向适配入口，非原生训练性能。原位覆盖输入不混入这些曲线。",
        "- ICTP `f40592a5687ec1d03219300ee557b2660f7d0369`：原生 O3 even/weighted UVU，自然宇称，Cartesian+path-expanded 输出。"
        "原生只支持 L≤3，4–8 阶实际尝试构造后记 unsupported。灰色曲线是不同方程成本参照，独立图给出其原生吞吐。",
        "- SO2Linear：冻结 V3 原生方程、C64→C64、无附加标量头；自写 CUDA/Triton 使用 c591b16 通用源码并修正 CUDA 工作线程上下文绑定。"
        "不同阶数不是固定检查点的整网替换验收。",
        "", "O2 同时测试 canonical 和 full parity，整数角阶，无半整数编译。"
        "所有推理、输入反向、数值参考和误差统计均 FP32，TF32/AMP/CUDA Graph 关闭。上游离线 CG 系数构造保留在计时外。"
        "逐分量同时满足 `1e-5+1e-4|ref|` 与 `2e-5+2e-5|ref|`，relative L2≤1e-5，门槛未放宽。"
        "全负载、两种子×两尺度验证，独立行参考按 512 行流式计算，补测可用 4096 行减少重复启动，但误差仍按原 512 行窗口逐窗统计；改变统计窗口的5992a12试测已被替代并排除。实际分块和检查窗口写入资格收据。冻结所有共享参数，forward+VJP 包含所有位置相关输入，"
        "UVU 包含外部逐边权重，RadialUVU 包含径向隐藏输入；不包含参数梯度或二阶训练。",
        f"当前有效曲线点在所有种子、尺度及原统计窗口中的最大 relative L2 为 `{summary.get('max_accepted_relative_l2')}`。"
        "逐点绝对误差和 relative L2 来自原始 FP32 回执，见 [误差汇总](numerics.csv)。",
        "", "## 同方程配对加速", "",
        "以下范围仅描述已完成的角阶点，不是对未测阶数的预测，也不是整网加速。加速=同卡基线 wall ms / 候选 wall ms。",
        "", "| 算子 / 行数 / 宇称 / 边界 | 候选 / 基线 | 已测角阶 | 配对加速范围 |",
        "|---|---|---|---:|"]
    groups = {}
    for row in ratios:
        key = tuple(row[k] for k in ("family","rows","parity","metric","candidate","baseline"))
        groups.setdefault(key,[]).append(row)
    for key, values in sorted(groups.items()):
        family, rows, parity, metric, candidate, baseline = key
        orders = ",".join(str(n) for n in sorted(int(r["order"]) for r in values))
        vals = [float(r["speedup"]) for r in values]
        text.append(f"| {family}/{rows}/{parity}/{metric} | {candidate}/{baseline} | {orders} | {min(vals):.3f}–{max(vals):.3f}× |")
    text.extend(["", "## 同卡峰值显存配对", "",
        "下表只使用显存阶段双方 GPU UUID/cohort 相同的记录，数值为候选/基线比，小于1表示降低。缺失或不同卡记录不相除。",
        "", "| 算子 / 行数 / 宇称 / 边界 | 候选 / 基线 | 已配对角阶 | allocated 比范围 | reserved 比范围 |",
        "|---|---|---|---:|---:|"])
    for key, values in sorted(groups.items()):
        paired=[r for r in values if r.get("memory_status")=="paired"]
        if not paired:
            continue
        family,rows,parity,metric,candidate,baseline=key
        orders=",".join(str(n) for n in sorted(int(r["order"]) for r in paired))
        allocated=[float(r["allocated_ratio"]) for r in paired]
        reserved=[float(r["reserved_ratio"]) for r in paired]
        text.append(f"| {family}/{rows}/{parity}/{metric} | {candidate}/{baseline} | {orders} | {min(allocated):.3f}–{max(allocated):.3f} | {min(reserved):.3f}–{max(reserved):.3f} |")
    text.extend(["", "![同方程 TP 对照](../../figures/o2-hp-angular-operators-20261121-r1/O2_TP_same_equation_speedup.png)",
        "", "![同方程 SO2 与 Radial 对照](../../figures/o2-hp-angular-operators-20261121-r1/SO2_Radial_same_equation_speedup.png)",
        "", "## 测量和失败记录", "",
        "GBU2 NV_H100 / NVIDIA H100 80GB，PyTorch2.11.0+cu128、Triton3.6.0。各配置的对照臂按顺序轮换/反转，"
        "每边界预热30次、同步墙钟测30次，CUDA event 作为辅助列。构建、数值资格、计时和峰值显存使用分离进程。"
        "没有把编译、源封印或独立参考验证时间计入稳态吞吐。每个比值核对 GPU UUID、输入/权重/资格 SHA、运行时和边界一致。",
        "后续部分测量通过只预载 CPU 代码的 forkserver 减少重复导入。四阶段短测确认不同 PID、相同 GPU 和四臂 FP32 均通过；"
        "每个计时/显存子进程仍从未初始化 CUDA 的状态开始，CUDA 上下文和分配器独立。CSV 明列 process_mode，不能将准备开销减少冒充算子加速。",
        "", "allocated/reserved 峰值均来自新进程；独立显存批次在自己的同卡 GPU 上重新过 FP32 门，CSV 分别记录计时卡和显存卡 UUID。"
        "早期 TP/UVU/Radial 显存准备误留一份未用的原布局 cotangent，旧显存记录已被修正批次替代；这些计时不受影响。"
        f"当前 {summary['memory_missing_points']} 个合格计时点尚无匹配的修正显存。",
        "", "早期 SO2 载荷漏带 matrix.cu，修复后又检测到 cuLaunchKernel status201。该重复失败不计有效吞吐；"
        "修复为在反向工作线程绑定模块所属 CUDA 上下文，调用后恢复。新批次先强制空上下文回归，再完整重做原数值门和配对。"
        "高阶构建诊断按角色/阶段保存期限，超时不是数值失败，也不是不支持；失败/OOM/unsupported 均不填零吞吐。",
        "", "### 本轮未产生有效吞吐的角色/边界", "",
        "以下是最终逐角色状态，历史中间失败仍完整保留于status.csv。数值失败不降低门槛，也不填入最快失败耗时。", "",
        "| 算子 | 行数 | 角阶 | 宇称 | 实现 | 边界 | 终态 |", "|---|---:|---:|---|---|---|---|"])
    for row in role_coverage:
        if row["status"] not in ("measured", "unsupported"):
            text.append("| " + " | ".join(row[k] for k in ("family","rows","order","parity","role","metric","status")) + " |")
    text.extend([
        "", "## 资源、文件与快照", "",
        f"账本快照：本项目已分配 {budget['project_gpu_h']:.4f} GPUh，本角阶轮 {budget['angular_gpu_h']:.4f} GPUh，"
        "最终值以收尾 Slurm 账本为准。按睡前一次提交要求加入有界、跳过已完成点的夜间补测，本轮包含失败的包络36 GPUh，项目原总上限500 GPUh保持。CPU 汇总不记 GPUh。"
        "GBU2 按 GPU TRES 计数；AIDAM 旧记录缺少 GPU TRES，按原生请求/提交行/SBATCH请求补计并保存脚本指纹，避免漏计历史 GPU 作业。",
        "新增夜间载荷d005f68的压缩包已按睡前SHA256核验，其12份控制/测量源码与Git一致；四个GPU源封印的上游逐文件哈希独立核对。"
        "先前98份实际部署文件已独立归档核对。由于当前远端SSH断开，本次不能追加声称已重新下载夜间远端控制源码字节。",
        "", "[逐点数据](summary.csv) · [同方程配对比](speedups.csv) · [失败与状态](status.csv) · [88位置覆盖](coverage.csv) · [机器审计](SUMMARY.json)",
        "", "图宽150mm，Times New Roman，无框图例。初始绘图与汇总在 CPU Slurm，Tavotto 打开、预检及导出使用用户已授权的本轮绘图例外。"
        "最终9图和预检证明位于[最终导出清单](provenance/TAVOTTO-FINAL-EXPORTS.json)，每图0错误、0警告。"
        "重新加载共享绘图模块，避免旧worker缓存；预检的无拟合置信区间建议不适用于这里的原始测量点连线。"
        "原先整网规模图保留为独立附录，不充当本轮角阶吞吐主图。训练及新增整网测量均未启动。"])
    (ROOT/"REPORT.zh-CN.md").write_text("\n".join(text)+"\n")
    print("Report regenerated from audited CSV receipts",flush=True)


if __name__ == "__main__":
    main()
