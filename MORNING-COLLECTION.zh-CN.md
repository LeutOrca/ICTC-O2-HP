# 2026-11-22 早间数据收集

已收集本轮全部夜间测量、日志、四个夜间源封印及07:00远端Slurm账本。
317429–317432均COMPLETED：TP两个补测在11秒内确认已完整并跳过，UVU补测5590秒，Radial补测11604秒。
33个本轮GPU作业均为终态，历史超时、失败和取消一并保留。没有新提交GPU、整网或训练作业。

自动收集115550在成功回收文件和远端账本后，本地sacct记账库连接失败，故未进入汇总。
本次刷新本地账本，CPU115739/115743完成汇总与初始绘图，CPU115742独立审计通过。
GBU2当前SSH三次有界重试均Connection closed，不能宣称实时队列已复核为空。
本轮作业结束依据是07:00账本及完整原始回执。collect_morning.sh已改为查询成功后原子替换账本，
预算代码拒绝空账本，防止暂时故障覆盖有效文件并产生费用低估。

## 科学终态

- 496个角色/计时边界项中：453个有效点、11个数值失败、12个构建超时、20个ICTP高阶不支持，pending=0。
- 全部有效点有五个独立进程计时、新进程allocated/reserved显存记录，429个比值均保留同卡配对证据。
- FP32参考、模型及误差统计；TF32/AMP/Graph关闭，原分量与relative L2门不变。
  有效点最大relative L2为2.2551566871698014e-6，分块仍按原512行误差窗口检查。
- 11个数值失败都在full O2输入VJP：public UVU六阶一项；Radial的public/unfused/Eqx四至八阶共十项。
  具体实现、角阶和边界见role-coverage.csv及报告。它们没有进入加速曲线。
- 12个超时为full O2七/八阶public UVU及public/unfused Radial的前向/输入VJP构建。
- ICTP原生只支持一至三阶。四至八阶的两行数、两计时边界共20项记unsupported，不填零或外推。

## 主要结果与边界

| 同方程配对 | 前向 | 前向＋输入VJP |
|---|---:|---:|
| HPX fresh-output / GitHub O2-HP TP，汇总两行数及两宇称 | 0.988–1.449× | 1.521–2.472× |
| 通用Triton / 原生SO2Linear | 1.880–18.046× | 2.514–18.011× |
| 通用CUDA / 原生SO2Linear | 0.724–0.891× | 0.788–1.200× |

HPX TP的前向allocated/reserved与GitHub版本持平，输入VJP的allocated降低约1.4–7.2%，
所以本次普通独立输出接口没有普遍达到显存降低10%的目标，不能引用旧原位接口结果替代。
SO2替换allocated降低约22.8–25.0%/21.4–35.4%，reserved为0–30.8%/19.6–40.3%；
速度和两种显存指标独立报告。Radial融合并非普遍快于Eqx或unfused，完整退化也保留。

这些是固定C64、N4096/E16384、角阶一至八的独立算子结果，不是整网E/F或含构图加速。
官方main96815838和fork296e61f1的14份执行源码相同，合并曲线明确双来源。
本地TP为5f278ef独立输出/冻结权重输入伴随适配；SO2为c591b16通用核加上下文修复，
不等同于efb5a27整网专用版本。ICTP为不同方程成本参照，不计算跨方程加速。

## 图、费用与恢复

九张图已通过Tavotto预检并导出矢量PDF、600dpi PNG，全部0错误、0警告。
保留150mm、Times New Roman、无框图例；共享helper在main内重新加载，防止旧worker缓存。
导出文件、SHA256、字体与证明路径见provenance/TAVOTTO-FINAL-EXPORTS.json。
固定文件名已同步为同一最终Tavotto导出，报告嵌图可离线阅读。

本轮33个单卡GPU作业实耗25.0411 GPUh、涉及8个H100节点（并非同时8节点）。
项目2026-11-15以来累计111.7231/500 GPUh，含失败与AIDAM缺GPU TRES历史请求的补计，未清零预算。

先前98份实际远端控制源码归档逐字节审计通过。夜间d005f68载荷SHA256与睡前冻结值一致，
12份控制/测量文件与该Git提交一致，四个GPU上游源封印均与本地冻结源码一致。
当前SSH故障下不能追加声称已重新下载夜间远端控制源码字节；此限制已写入机器审计。

快照身份见provenance/FROZEN-GIT.txt，独立恢复QA见provenance/DELIVERY-QA.json。
源码Git bundle、冻结/恢复源码tar、上游源码、原始回执和完整图库分别归档，
校验表见provenance/DELIVERY.sha256。恢复审计不把已有数值失败或性能缺口改判为通过。

[完整报告](REPORT.zh-CN.md) · [逐点数据](summary.csv) · [角色终态](role-coverage.csv) · [机器审计](provenance/MORNING-AUDIT.json)
