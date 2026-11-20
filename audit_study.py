"""Build a receipt-bounded progress report on a Slurm CPU node."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    target = ROOT / path
    return json.loads(target.read_text()) if target.exists() else None


operator = read("cohorts/d8aec37/results/PAIR_AUDIT.json")
large = read("cohorts/d8aec37/results/so2-large.json")
pair = read("cohorts/d8aec37/results/time-forward-own-0.json")
assert large["passed"] and operator["status"] == "complete"
assert all(large[key] == pair[key] for key in ("source", "input_sha256", "weights_sha256", "gpu_uuid"))
prefix = read("cohorts/24c2ddc/results/mace-prefix-b8.json")
v3 = read("cohorts/2089373/results/V3_PAIR_AUDIT-mptrj-b1.json")
scalar = read("cohorts/8c2d4d3/results/mace-scalar-b8.json")
qualification_paths = [
    "cohorts/8c2d4d3/results/mace-scalar-b8.json",
    "cohorts/8c2d4d3/results/mace-scalar-600.json",
    "cohorts/3924ed2/results/mace-receiver-b8.json",
    "cohorts/3924ed2/results/mace-receiver-600.json",
    "cohorts/b0d0c01/results/mace-projection-b8.json",
    "cohorts/b0d0c01/results/mace-projection-600.json",
]
qualification_paths += [str(path.relative_to(ROOT)) for pattern in (
    "cohorts/*/measurements/*/results/qualification.json",
    "h100-results/*/measurements/*/results/qualification.json",
) for path in ROOT.glob(pattern)]
qualifications = {path: read(path) for path in qualification_paths if (ROOT / path).exists()}
for path in ROOT.glob("cohorts/64dbeb1/results/mace-input-600.json"):
    qualifications[str(path.relative_to(ROOT))] = json.loads(path.read_text())
v3_local_audits = {str(path.relative_to(ROOT)): json.loads(path.read_text())
                   for path in ROOT.glob("cohorts/*/measurements/*/results/V3_LOCAL_PAIR_AUDIT-*.json")}
v3_local_gate = read("cohorts/9333e8f/results/v3-local-last2-b1.json")
audit_paths = [
    "cohorts/d3ddb6f/results/MACE_PAIR_AUDIT-b8-scalar.json",
    "cohorts/152c619/measurements/114664-projection-b8/results/MACE_PAIR_AUDIT-b8-projection.json",
    "cohorts/152c619/measurements/114665-receiver-b8/results/MACE_PAIR_AUDIT-b8-receiver.json",
]
audit_paths += [str(path.relative_to(ROOT)) for pattern in (
    "cohorts/c429843/measurements/*/results/MACE_PAIR_AUDIT-*.json",
    "cohorts/fb3dc99/measurements/*/results/MACE_PAIR_AUDIT-*.json",
    "h100-results/*/measurements/*/results/MACE_PAIR_AUDIT-*.json",
) for path in ROOT.glob(pattern)]
audits = {path: read(path) for path in dict.fromkeys(audit_paths) if (ROOT / path).exists()}
best_h100 = {}
for path, audit in audits.items():
    if path.startswith("h100-results/") and audit.get("status") == "complete":
        if audit["candidate"] in ("final", "final-input"):
            current = best_h100.get(audit["case"])
            job = int(Path(path).parts[3].split("-")[0])
            selection = (audit["candidate"] == "final-input", "vs_final" in audit["boundaries"]["e2e"], job)
            if current is None or selection > tuple(current["selection"]):
                best_h100[audit["case"]] = dict(path=path, audit=audit, selection=selection)
mace_target = (set(best_h100) == {"b8", "n512", "n4096"} and all(
    item["audit"]["boundaries"]["e2e"]["vs_native"]["speed_2x"] and
    item["audit"]["boundaries"]["e2e"]["vs_native"]["allocated_10pct"]
    for item in best_h100.values()))
summary = dict(operator=operator, exact_shape_validation=large, mace_prefix_b8=prefix,
               v3_first_all_matrix_pairs=v3, mace_scalar=scalar,
               mace_qualifications=qualifications, mace_pairs=audits,
               v3_local_pairs=v3_local_audits, v3_local_b1_gate=v3_local_gate,
               mace_h100_best_cases=best_h100, mace_h100_three_case_target=mace_target)
(ROOT / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
lines = ["# O3-HPX 路径的自写算子续轮", "",
         ("完整 SO2Linear 已达 5× 下限，MACE H100 三点均通过端到端 2× 和 allocated 下降 10% 门；V3 新局部候选单列。"
          if mace_target else "当前已有一项独立算子达到 5× 下限，整网目标仍在验收。MACE 优先，V3 并行推进。"),
         "用户已明确：本轮替换算子必须全自写，未改原生层可保留 cuBLAS。", "",
         "## 已完成的 SO2Linear 五轮同卡配对", "",
         "快照 `d8aec37`，官方 V3 SO2Linear，160000 边、C128、L4、M2，冻结相同输入和权重。",
         "严格 FP32，TF32/AMP/Graph 均关闭。五轮双向交替，时间、显存使用不同进程。",
         "该快照的完整测量形状已重新核验，输入、权重、源码和 GPU UUID 与计时回执一致。", "",
         "| 范围 | 原生→自写同步 ms | 配对中位数 | 原生→自写百万边/s | allocated MiB | reserved 下降 |",
         "|---|---|---|---|---|---|"]
for name, row in operator["modes"].items():
    lines.append(f"| {name} | {row['native_wall_ms']:.3f}→{row['own_wall_ms']:.3f} | "
                 f"{row['wall_speedup']:.3f}× | {160/row['native_wall_ms']:.3f}→{160/row['own_wall_ms']:.3f} | "
                 f"{row['native_peak_allocated']/2**20:.1f}→{row['own_peak_allocated']/2**20:.1f} "
                 f"(−{row['allocated_reduction']*100:.1f}%) | {row['reserved_reduction']*100:.1f}% |")
lines += ["", "两种范围均超过 5×，尚未达到 6×。allocated 均超过 20% 下降门。",
          "前向/输入 VJP 最大绝对差为 "
          f"`{large['errors']['forward']['max_abs']:.6g}` / `{large['errors']['input_vjp']['max_abs']:.6g}`，"
          "相对 L2 为 "
          f"`{large['errors']['forward']['relative_l2']:.6g}` / `{large['errors']['input_vjp']['relative_l2']:.6g}`。",
          "逐分量 `2e-5 + 2e-5×|reference|` 和相对 L2 ≤ `1e-5` 全部通过。",
          "候选前向与一次输入反向 trace 均只有自写矩阵核，没有 cuBLAS/CUTLASS。",
          "[五轮明细](cohorts/d8aec37/results/PAIR_AUDIT.json) · "
          "[完整形状验收](cohorts/d8aec37/results/so2-large.json)", "",
          "## 更新的执行路径和代码", "",
          "| 项目 | 最终实现 | 证据状态 |", "|---|---|---|",
          "| 碎批量乘法合并 | `matrix_kernels.py` 自写 IEEE FP32 Triton GEMM，`csrc/matrix.cu` 自写共享内存 CUDA FMA | 独立算子通过 |",
          "| 冻结复数权重 | `frozen_so2.py` 构造实块矩阵，缓存正反方向连续布局，按参数版本/存储/设备失效 | 前向、VJP、缓存更新通过 |",
          "| 输出和伴随组装 | `so2_layer.py` 直接写最终输出及输入梯度，各阶不再同时保留完整临时结果 | 五配对速度、显存达 5× 下限 |",
          "| 逐元素融合 | `merge_norm.py` 融合中心化、统计量、仿射及一次输入伴随，保留 pow 伴随次序 | 独立 norm 和 V3 B1 通过，B8 未通过 |",
          "| gather 与径向调制 | 沿用冻结 V3 `9787a3a` 的已验收融合 | 本轮整网组合仍单独验收 |",
          "| 小批量矩阵 | `_small_batched` 合并 Wigner 小矩阵线程；`kernel_registry.py` 直接注册 CUDA 矩阵核 | 核和 B1 通过，整网仅筛选 |",
          "| MACE 常量前缀 | `mace_frozen_prefix.py` 缓存 node embedding 和首层 linear_up，动态 skip TP 保留 | B8/600 E/F 与 Graph 逐位通过 |",
          "| MACE 首层 TP | `mace_scalar_tp.py` 保留 `x*w` 再乘 `h`，径向伴随使用非 FMA 原生小树归约 | `8c2d4d3` B8/600 通过，单独版本性能退化 |",
          "| MACE TP 与接收归约融合 | `mace_receiver_tp.py` 自写 Triton；不物化 E×2048 消息及其伴随 | `3924ed2` B8/600 通过，已做五轮整网配对 |",
          "| MACE 常量物种投影 | `mace_species_projection.py` 只替换 TorchScript 的三个常量 one-hot 投影节点，保留原生通道 GEMM | `b0d0c01` B8/600 E/F 逐位一致，已配对 |",
          "| H100 归约兼容 | `native_reduce.py` 用相邻对归约复现 lane 0 的原生升序树 | `40af2f8` 不再依赖旧 Triton 缺失的 tl.gather |",
          "| 端到端组装 | `mace_static_inputs.py` 复用 node_attrs/batch/ptr/head，逐次重建原生邻居 | `fb3dc99` 独立资格、位置更新后的输入逐位核对后计时 |", "",
          "冻结前缀的原生准备成本与缓存驻留量单独计量，它属于消除重复计算，不能记为自写 GEMM 的速度比。",
          "MACE 半边/共享矩等全部 O3-HPX 项目尚未完成移植或验收，不宣称已经逐项完成 O3-HPX 全流程。", "",
          "## 整网证据", "",
          "MACE 继续以 `78231f3` 为冻结性能基线，新候选每次直接比较原生 eager 和原生 Graph。",
          "新增前缀 `24c2ddc` 的 B8 缓存共 69632 字节，600 原子也通过逐位 E/F。H100 前缀作业 `316402`",
          "25 分钟超时：30 份计时完整、显存 29/30，不能记为完成，不能用另一张卡补一个缺片。",
          "[B8 前缀资格](cohorts/24c2ddc/results/mace-prefix-b8.json)", "",
          "首层 TP 旧版 `ccb09db` 力相对 L2 为 4.04e-5。伴随消融确认角向单独替换通过，径向单独替换失败。",
          "原因是原生 einsum 为先舍入逐元素乘积再做 3/5/7 维归约；FMA 点积改变了顺序。",
          "`8c2d4d3` 复现非 FMA 乘法和原生小树后，前向与径向 VJP 逐位一致，B8 固定图力相对 L2 为 "
          f"`{scalar['numerics']['fixed']['forces']['relative_l2']:.6g}`，600 也通过，原容差未放宽。",
          "单独版本使用的通用 CUDA 小 GEMV 分块浪费较大，整网明显退化，保留回执，不晋级。融合版改用专用 Triton 伴随。", "",
          "### 各快照正式五轮配对", "",
          "各行仅来自完整的同 GPU UUID、同输入/权重/源码/资格哈希配对。速度是五个配对比值的中位数，",
          "延迟及显存列是各臂中位数，因此四舍五入后的延迟之比可能与速度略有不同。全部臂同用 CUDA Graph。", "",
          "| 卡 / 快照 / 点 / 候选 | 边界 | 原生→78231f3→候选 ms | 对原生 / 对78231f3 | allocated / reserved MiB | allocated / reserved 相对原生下降 | 候选 原子/s |",
          "|---|---|---|---|---|---|---|"]
for path, audit in audits.items():
    if audit.get("status") != "complete":
        continue
    hardware = "H100" if path.startswith("h100-results/") else "A100"
    identity = path.split("/")[1]
    for boundary, row in audit["boundaries"].items():
        candidate = row[audit["candidate"]]
        lines.append(f"| {hardware} / {identity} / {audit['case']} / {audit['candidate']} | {boundary} | "
                     f"{row['native']['wall_ms']:.4f}→{row['base']['wall_ms']:.4f}→{candidate['wall_ms']:.4f} | "
                     f"{row['vs_native']['speedup']:.4f}× / {row['vs_base']['speedup']:.4f}× | "
                     f"{candidate['peak_allocated']/2**20:.2f} / {candidate['peak_reserved']/2**20:.2f} | "
                     f"{row['vs_native']['allocated_reduction']*100:.2f}% / {row['vs_native']['reserved_reduction']*100:.2f}% | "
                     f"{candidate['atoms_s']:.1f} |")
lines += ["", "原始路径、逐轮比值、structures/s、原子/s、三臂 allocated/reserved 全值在 [SUMMARY.json](SUMMARY.json) 的 mace_pairs。",
          "A100 投影与融合两作业曾共用原生臂文件名，原目录部分覆盖。已从不可变 Slurm stdout 的原始 JSON 行",
          "分别恢复每项 66 份回执，CPU 114671 核对 UUID/输入/源码/资格哈希后审计通过。表中只使用按作业恢复的路径，",
          "旧平铺目录保留作故障证据。`c429843` 起所有新结果位于 measurements/<job>-<variant>-<case>/。", "",
          "联合候选为前缀、物种投影和 TP/CSR 融合三项；final-input 额外复用常量组分字段。",
          "H100 B8/600 必须先通过原阈值，再计时和测显存。4100 点也重新对原生 Graph 验证，失败/OOM 单列。", "",
          "此前全矩阵替换实验中，A100 最佳 B8 力相对 L2 约 `2.94e-5`，H100 初版为 `4.42e-5`。",
          "原生 FP32 相对独立 FP64 的诊断差约 `6.57e-5`，仅用于解释舍入敏感性，原验收阈值未放宽。", "",
          "V3 初次全矩阵候选 `2089373`，B1 的两种子、固定/端到端 E/F 通过，但五配对整网未达标：", "",
          "| B1 整网 E/F | 原生→候选 ms | 同卡配对 | allocated MiB | reserved MiB |",
          "|---|---|---|---|---|"]
for name, row in v3["boundaries"].items():
    a, b = row["native"], row["own"]
    lines.append(f"| {name} | {a['wall_ms']:.3f}→{b['wall_ms']:.3f} | {row['speedup']:.3f}× | "
                 f"{a['peak_allocated']/2**20:.1f}→{b['peak_allocated']/2**20:.1f} | "
                 f"{a['peak_reserved']/2**20:.1f}→{b['peak_reserved']/2**20:.1f} |")
lines += ["", "allocated 下降约 6.9%，速度与整网 10% 显存门均未达到。后续直接注册核的单对筛选改善了分派成本，",
          "仍需各输入点资格与完整五配对后才能替换正式结果。全 14 个 SO2 或联合 norm 的 B8 仍存在相对 L2 超限。",
          "[全矩阵分支五配对](cohorts/2089373/results/V3_PAIR_AUDIT-mptrj-b1.json)", "",
          "### V3 后两层局部替换", "",
          "保留原生 norm 及前五层 SO2Linear，只替换 blocks.5/6 的四个 SO2Linear，并沿用已验收 gather/radial。",
          "B1 和 B8 均按两种子、两图边界通过原 E/F 门。B8 最大力相对 L2 约 9.36e-6，能量逐位一致。",
          "这四个 SO2Linear 内全部使用自写 Triton，未改层保留原生 cuBLAS。正式比较原生、9787a3a、局部候选三臂。", "",
          "| 点 | 边界 | 原生→9787a3a→局部候选 ms | 对原生 / 对9787a3a | allocated / reserved MiB |",
          "|---|---|---|---|---|"]
for path, audit in v3_local_audits.items():
    if audit.get("status") != "complete":
        continue
    for boundary, row in audit["boundaries"].items():
        candidate = row["own"]
        lines.append(f"| {audit['case']} | {boundary} | {row['native']['wall_ms']:.3f}→"
                     f"{row['base']['wall_ms']:.3f}→{candidate['wall_ms']:.3f} | "
                     f"{row['vs_native']['speedup']:.3f}× / {row['vs_base']['speedup']:.3f}× | "
                     f"{candidate['peak_allocated']/2**20:.1f} / {candidate['peak_reserved']/2**20:.1f} |")
lines += ["", "本表只在三臂五配对收齐并审计通过后列数值。B8 数值过门不代表整个 V3 达到 5×，",
          "SO2Linear 大形状的 5× 收益仍单列在算子表。", "",
          "## 固定的科学范围", "",
          "MACE checkpoint SHA256 `d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a`，",
          "V3 checkpoint SHA256 `429ccded98163122e7ba588d78e2441653f37f3e091e106c432807fe373c8f98`。",
          "两者沿用官方表示和图规则。MACE 为自然宇称，V3 沿用官方 SO(2)/SO(3) 表示，未增加独立 full-parity 通道。",
          "均为整数 l/m，未采用半整数表示或半整数编译。数据与累计为 FP32，非 FP16，关闭 TF32 和 AMP。",
          "MACE 两臂同用 Graph。V3 正式配对关闭 Graph，新增 Graph 资格探针在确定性 scatter_add 处被阻断，未产生有效性能。",
          "当前自写冻结算子仅验证前向和一次输入伴随，训练及力损失二阶梯度未放行。", "",
          "## 快照与恢复", "",
          "各项源文件按 Git commit 导出到只读使用的 `cohorts/<hash>`，提交前逐文件 SHA256 预检。",
          "`d8aec37` 是已达 5× 的完整 SO2Linear 快照，MACE 冻结性能基线绑定 `78231f3`，新联合版本由各行对应回执验收。",
          "最新源码及全部实验提交在本 run 的 Git 仓库，未通过的快照不覆盖已验收基线。",
          "完整提交清单和作业账本位于 `provenance/`。全部 Python、编译、聚合和测量均由 Slurm 计算节点执行。"]
(ROOT / "REPORT.zh-CN.md").write_text("\n".join(lines) + "\n")
print(json.dumps(dict(report=str(ROOT / "REPORT.zh-CN.md"), large_gate=True,
                       operator_5x=True, whole_goal_complete=False)), flush=True)
