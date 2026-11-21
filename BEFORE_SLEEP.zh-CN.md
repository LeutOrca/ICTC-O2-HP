# 2026-11-22 睡前交接

用户要求一次性提交全部剩余作业，早上再收集。提交后仅做一次队列/依赖核对，停止人工轮询。
GPU载荷冻结为 **d005f68**，不新增整网或训练任务。

| 作业 | 内容 | 前置作业结束后启动 | 最大时限 |
|---|---|---|---|
| 317429 | TP，N4096 | 317398 | 3小时 |
| 317430 | TP，E16384 | 317399 | 3小时 |
| 317431 | UVU，E16384 | 317369、317400 | 4小时 |
| 317432 | RadialUVU，E16384 | 317370、317401 | 4小时 |
| 115550 | CPU回收、预算、汇总、初始绘图 | 北京时间2026-11-22 07:00 | 1小时 |

补测在计算节点读取已有回执，仅补没有完整五轮同卡计时或没有完整同卡显存配对的配置。
已完成配置自动跳过，不从这些作业自动递交新GPU作业。依赖使用afterany，不把调度完成当作数值通过。
较长时限用于防止夜间构建再次截断，实际消耗按真实占用时间结算。

范围：整数角阶1–8，C64；节点N4096，边E16384。O2 canonical/full分开，
forward与所有位置相关输入的一次VJP分开。FP32参考与误差统计，原分量/relative L2门保持，
TF32/AMP/CUDA Graph关闭。参考执行可按4096行批量，误差仍按原512行窗口检查。
5992a12使用4096行统计窗口的试测已排除，不能进入正式结论。
ICTP原生只支持1–3阶，4–8阶实际构造报L_MAX=3，不填零、不外推。

SO2三臂1–8阶已完成。原生、通用c591b16 Triton、自写CUDA分别列示；
这不是最新efb5a27整网专用实现的E/F速度图。当前图的输入和权重是同角阶内匹配的可复现随机张量。
本地TP采用5f278ef独立输出/冻结权重输入伴随适配，不把原位覆盖输入的收益混入普通接口。

源码部署已有98文件独立核对通过。新增d005f68源封印随夜间作业保存。
AIDAM旧记录不写GPU TRES，预算已按原生请求/提交行/SBATCH请求补计；
上次预算快照为项目103.8517/500 GPUh、本轮17.1697 GPUh，均非终态结算。
包含失败和全部夜间预留的本轮包络为36 GPUh，项目原500 GPUh硬上限保持。

## 明早入口

先看 `provenance/MORNING-QUEUE.txt`、收集作业115550的日志，以及
`SUMMARY.json` 的 pending_role_metrics/audit_errors 和 `role-coverage.csv`。
07:00仅是收集时间，不保证所有GPU作业已完成；运行中、超时、OOM、数值失败仍单列。

如需手动重收，在登录节点只做文件回收，然后把Python汇总交给CPU Slurm：

```bash
cd /public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-hp-angular-operators-20261121-r1
O2_COLLECT_ONLY=1 bash collect_morning.sh
sbatch aggregate.sbatch
```

原始回执和各cohort保留在 `measurements/`。正式数据位于summary.csv、speedups.csv、numerics.csv，
完整角阶图位于 `figures/o2-hp-angular-operators-20261121-r1/`。
当前Tavotto的SO2/ICTP导出仅记作阶段产物；最终全图应在回收完数据后统一预检/导出，
并刷新绘图入口，核对helper缓存、图内条件标注和最新CSV，避免复用旧worker画面。
归档独立恢复/CSV逐字节复算工具已准备，最终封包待数据终态。

[当前报告](REPORT.zh-CN.md) · [合同](CONTRACT.md) · [提交核对](provenance/OVERNIGHT-STARTUP.txt)
