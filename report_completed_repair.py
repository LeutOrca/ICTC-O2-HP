"""Current report exclusively from independently audited receipts and CSVs."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GALLERY = "../../figures/o2-hp-angular-operators-20261121-r1/repair-20261122-r1"


def main():
    def read(name):
        return json.loads((ROOT / name).read_text())
    with (ROOT / "summary.csv").open() as stream:
        points = list(csv.DictReader(stream))
    with (ROOT / "speedups.csv").open() as stream:
        pairs = [r for r in csv.DictReader(stream) if r["candidate"].endswith("repair")]
    summary = read("SUMMARY.json")
    gate = read("provenance/REPAIR-GATE-AUDIT.json")
    bounded = read("provenance/BOUNDED-GATE-AUDIT.json")
    budget = read("provenance/BUDGET-AUDIT.json")
    cold_paths = list((ROOT / "measurements").glob("*/cold-build/SUMMARY.json"))
    cold = json.loads(cold_paths[-1].read_text()) if cold_paths else None
    new = [r for r in points if r["role"].endswith("repair")]
    lines = ["# Full-parity UVU/RadialUVU 修复与角阶吞吐复测", "",
        "冻结基线 `6f641cf736a97fa6835678c0eb123f5b2694e80d`。"
        "本轮只处理独立算子，整网和训练保持暂停。", "",
        "## 数值与构建验收", "",
        "原11项数值失败/12项超时的分类有误：3个未执行VJP被前向资格回执误标，正确为8个真实数值失败、15个构建超时。"
        "分类修正不改变原453个有效点；原基线及原始失败回执保留。",
        f"数值核心`73a2e61`、H100317553、CPU115807：8项全部通过，两种子×两尺度、原512行窗口，共{sum(r['checks'] for r in gate['accepted'])}个FP32检查，输入/规范权重哈希不变。",
        "分段隔离证明TP权重伴随及其误差传播通过，而同一权重伴随仅改变长K原生矩阵执行形状即可复现隐藏输入伴随误差。"
        "参考仿射层重放完全一致。形状切换选中了不同原生FP32矩阵核；这不能单凭核名确定原生分段数。",
        "自写Triton输入伴随：K<49152采用16路交错、128元素块内舍入与蝶形合并，K≥49152采用8路连续分段与蝶形合并。"
        "两分支均IEEE FP32，不用TF32、FP64、原子累加或vendor矩阵例程。原生冻结仿射前向保留。",
        f"构建候选`e48a156`取消V=1全路径静态展开，调用冻结公共源码的运行时路径循环Triton核；径向按4096行重算私有权重，并在分配权重伴随前释放生成权重。"
        f"独立CPU核对{bounded['gates']}个受影响候选门、{bounded['checks']}个FP32检查通过。"
        "原官方/fork源码未改，修复候选单列，不冒充未修改的GitHub实现。", ""]
    if cold:
        values = [r["elapsed_s"] for r in cold["cases"]]
        lines += [f"独立冷构建作业{cold['job']}：每个配置使用起初不存在的Triton缓存、独立进程、原900秒期限。"
                  f"9个前向+全输入VJP构建覆盖原15个超时槽，passed={cold['passed']}，进程耗时{min(values):.2f}–{max(values):.2f}秒。"
                  "包含导入/初始化，不是算子延迟或编译核本体耗时。", ""]
    else:
        lines += ["独立冷缓存构建回执尚待回收，不用共享缓存下构建成功冒充冷编译验收。", ""]
    lines += ["## 配对结果与吞吐", "",
        f"当前汇总{summary['summary_points']}个有效曲线点，新增修复角色{len(new)}个点，配对比值{summary['paired_ratios']}项。"
        f"缺显存{summary['memory_missing_points']}，审计错误{len(summary['audit_errors'])}。"
        "原生失败槽继续保留失败，图中由明确标注的修复角色补充，不把失败填成零吞吐。",
        "UVU分母是未修改Eqx-O2 CUDA；Radial高阶VJP的分母是Eqx-O2加同一自写仿射输入伴随，因为原生Eqx隐藏输入伴随未过门。"
        "后者不能写成对未修改Eqx整算子的加速。", "",
        "| 算子 | 角阶 | 边界 | 候选/分母 | 同卡加速 | allocated降低 | reserved降低 |",
        "|---|---:|---|---|---:|---:|---:|"]
    for r in pairs:
        assert r["memory_status"] == "paired"
        lines.append(f"| {r['family']} | {r['order']} | {r['metric']} | {r['candidate']}/{r['baseline']} | "
                     f"{float(r['speedup']):.3f}× | {100*(1-float(r['allocated_ratio'])):.1f}% | {100*(1-float(r['reserved_ratio'])):.1f}% |")
    lines += ["", "小于1的加速表示退化，负显存降幅表示增加。构建修复不自动等于性能优化通过。", "",
        f"![Full parity UVU/Radial 吞吐]({GALLERY}/O2_UVU_Radial_full.png)", "",
        "横轴角阶1–8，C64；节点TP N4096，其余边算子E16384；UVU V1、径向H64。"
        "原子/s是节点特征行吞吐，边/s是逐边算子行吞吐，均不含整网、邻居构建、能量或力。"
        "O2 canonical/full分图，SO2/ICTP是不同方程的绝对成本参照，不作跨方程加速。",
        f"[节点TP]({GALLERY}/O2_TP_atoms.pdf) · [边TP]({GALLERY}/O2_TP_edges.pdf) · "
        f"[canonical UVU/Radial]({GALLERY}/O2_UVU_Radial_canonical.pdf) · [full UVU/Radial]({GALLERY}/O2_UVU_Radial_full.pdf) · "
        f"[SO2Linear]({GALLERY}/SO2Linear_edges.pdf) · [ICTP]({GALLERY}/ICTP_atoms_edges.pdf)", "",
        "## 不变的合同与来源", "",
        "GBU2 H10080GB，torch2.11.0+cu128/Triton3.6.0。FP32输入/权重/累加/参考/误差统计，TF32/AMP/CUDA Graph关闭。"
        "同时满足逐分量1e-5+1e-4|ref|和2e-5+2e-5|ref|，relative L2≤1e-5，不放宽门槛。"
        "两种子20261003/20261004、尺度0.1/1，E16384全负载，参考可分4096行，统计始终每512行。",
        "构建、资格、五轮计时、每角色显存用分离进程。每轮30次预热、30次同步墙钟，CUDA event辅助；"
        "五轮角色顺序轮换/反转，同一job/GPU/输入/权重/资格哈希配对。峰值allocated/reserved来自新CUDA上下文与分配器。"
        "整数角阶，无半整数/半精度编译；冻结参数，只验前向和所有位置相关输入一次VJP，不声明训练、参数梯度或二阶通过。",
        "官方96815838与fork296e61f1的14份可执行源码一致；Eqx c646c700/TACE0.2.5 CUDA；ICTP f40592a5仅支持L≤3，4–8不支持留空；"
        "cuEq0.13 exact O2仅用于UUU。旧TP/HPX/SO2测量沿用原合格回执，新增角色的来源见快照和源封印。", "",
        "## 资源、连接与交付", "",
        f"本续轮{budget['repair_gpu_jobs']}个GPU作业、{budget['repair_gpu_h']:.4f}GPUh，含失败与取消；"
        f"角阶轮含续轮共{budget['inclusive_angular_gpu_h']:.4f}GPUh，项目{budget['project_gpu_h']:.4f}/500GPUh。",
        "早前SSH连接曾被关闭，并有一次认证失败；同一地址、端口与密钥后来恢复。主机/sshd无本次重启证据，账号不能读取服务端系统日志，根因未确定。"
        "不能据此断言主机宕机、密钥失效、限流或共享存储故障。",
        "Tavotto插件及独立引擎安装为0.21.3，当前MCP仍加载旧0.21.0，需重连才加载新版。"
        "本轮使用已授权绘图例外完成Tavotto打开/预检/导出；数值、汇总及初始绘图均经Slurm。"
        "150mm、Times New Roman、无框图例；最终预检与导出证据另存，不将安装版本冒充运行中版本。", "",
        "[数值独立审计](provenance/REPAIR-GATE-AUDIT.json) · [构建候选资格审计](provenance/BOUNDED-GATE-AUDIT.json) · "
        "[逐点延迟/吞吐/显存](summary.csv) · [同卡配对](speedups.csv) · [原始状态](status.csv) · [汇总审计](SUMMARY.json) · "
        "[账本](provenance/BUDGET-AUDIT.json)", ""]
    (ROOT / "REPORT.zh-CN.md").write_text("\n".join(lines))
    print("Current report written from audited repair receipts", flush=True)


if __name__ == "__main__":
    main()
