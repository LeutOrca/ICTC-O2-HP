# Full-parity UVU/RadialUVU 修复与角阶吞吐复测

冻结基线 `6f641cf736a97fa6835678c0eb123f5b2694e80d`。本轮只处理独立算子，整网和训练保持暂停。

## 数值与构建验收

原11项数值失败/12项超时的分类有误：3个未执行VJP被前向资格回执误标，正确为8个真实数值失败、15个构建超时。分类修正不改变原453个有效点；原基线及原始失败回执保留。
数值核心`73a2e61`、H100317553、CPU115807：8项全部通过，两种子×两尺度、原512行窗口，共4096个FP32检查，输入/规范权重哈希不变。
分段隔离证明TP权重伴随及其误差传播通过，而同一权重伴随仅改变长K原生矩阵执行形状即可复现隐藏输入伴随误差。参考仿射层重放完全一致。形状切换选中了不同原生FP32矩阵核；这不能单凭核名确定原生分段数。
自写Triton输入伴随：K<49152采用16路交错、128元素块内舍入与蝶形合并，K≥49152采用8路连续分段与蝶形合并。两分支均IEEE FP32，不用TF32、FP64、原子累加或vendor矩阵例程。原生冻结仿射前向保留。
构建候选`e48a156`取消V=1全路径静态展开，调用冻结公共源码的运行时路径循环Triton核；径向按4096行重算私有权重，并在分配权重伴随前释放生成权重。独立CPU核对13个受影响候选门、6656个FP32检查通过。原官方/fork源码未改，修复候选单列，不冒充未修改的GitHub实现。

独立冷构建作业317569：每个配置使用起初不存在的Triton缓存、独立进程、原900秒期限。9个前向+全输入VJP构建覆盖原15个超时槽，passed=True，进程耗时25.50–116.50秒。包含导入/初始化，不是算子延迟或编译核本体耗时。

## 配对结果与吞吐

当前汇总489个有效曲线点，新增修复角色36个点，配对比值454项。缺显存0，审计错误0。原生失败槽继续保留失败，图中由明确标注的修复角色补充，不把失败填成零吞吐。
UVU分母是未修改Eqx-O2 CUDA；Radial高阶VJP的分母是Eqx-O2加同一自写仿射输入伴随，因为原生Eqx隐藏输入伴随未过门。后者不能写成对未修改Eqx整算子的加速。

UVU L6–8对原生Eqx前向1.562–1.592×，前向+VJP1.021–1.078×，allocated基本持平。径向L4–8对加同一伴随修复的Eqx，前向1.125–1.218×；VJP为0.848–0.912×，allocated降低75.0–80.7%、reserved降低73.6–80.1%。径向流式版的速度门未通过，UVU显存10%门未通过，保留为单列修复候选。整网暂停，整网加速结论不更新。

| 算子 | 角阶 | 边界 | 候选/分母 | 同卡加速 | allocated降低 | reserved降低 |
|---|---:|---|---|---:|---:|---:|
| radial | 4 | forward | public-repair/eqx-o2-repair | 1.125× | 61.0% | 58.0% |
| radial | 4 | forward | public-repair/public-unfused-repair | 0.954× | 60.9% | 58.0% |
| radial | 4 | forward_vjp | public-repair/eqx-o2-repair | 0.848× | 75.0% | 73.6% |
| radial | 4 | forward_vjp | public-repair/public-unfused-repair | 0.829× | 75.0% | 73.6% |
| radial | 5 | forward | public-repair/eqx-o2-repair | 1.151× | 63.1% | 60.6% |
| radial | 5 | forward | public-repair/public-unfused-repair | 0.958× | 63.1% | 60.2% |
| radial | 5 | forward_vjp | public-repair/eqx-o2-repair | 0.899× | 77.1% | 76.3% |
| radial | 5 | forward_vjp | public-repair/public-unfused-repair | 0.889× | 77.1% | 76.3% |
| radial | 6 | forward | public-repair/eqx-o2-repair | 1.191× | 64.6% | 63.3% |
| radial | 6 | forward | public-repair/public-unfused-repair | 0.970× | 64.6% | 63.0% |
| radial | 6 | forward_vjp | public-repair/eqx-o2-repair | 0.912× | 78.7% | 78.4% |
| radial | 6 | forward_vjp | public-repair/public-unfused-repair | 0.902× | 78.6% | 78.3% |
| radial | 7 | forward | public-repair/eqx-o2-repair | 1.206× | 65.8% | 65.3% |
| radial | 7 | forward | public-repair/public-unfused-repair | 0.980× | 65.8% | 65.3% |
| radial | 7 | forward_vjp | public-repair/eqx-o2-repair | 0.912× | 79.8% | 79.7% |
| radial | 7 | forward_vjp | public-repair/public-unfused-repair | 0.906× | 79.8% | 79.7% |
| radial | 8 | forward | public-repair/eqx-o2-repair | 1.218× | 66.8% | 65.6% |
| radial | 8 | forward | public-repair/public-unfused-repair | 0.984× | 66.8% | 65.4% |
| radial | 8 | forward_vjp | public-repair/eqx-o2-repair | 0.872× | 80.7% | 80.1% |
| radial | 8 | forward_vjp | public-repair/public-unfused-repair | 0.860× | 80.7% | 80.1% |
| uvu | 6 | forward | public-repair/eqx-o2 | 1.562× | 0.0% | 0.8% |
| uvu | 6 | forward_vjp | public-repair/eqx-o2 | 1.078× | 0.0% | 0.0% |
| uvu | 7 | forward | public-repair/eqx-o2 | 1.582× | 0.0% | 1.3% |
| uvu | 7 | forward_vjp | public-repair/eqx-o2 | 1.031× | 0.0% | 0.3% |
| uvu | 8 | forward | public-repair/eqx-o2 | 1.592× | 0.1% | 0.1% |
| uvu | 8 | forward_vjp | public-repair/eqx-o2 | 1.021× | 0.0% | 0.3% |

小于1的加速表示退化，负显存降幅表示增加。构建修复不自动等于性能优化通过。

![Full parity UVU/Radial 吞吐](../../figures/o2-hp-angular-operators-20261121-r1/repair-20261122-r1/O2_UVU_Radial_full_repair20261122.png)

横轴角阶1–8，C64；节点TP N4096，其余边算子E16384；UVU V1、径向H64。原子/s是节点特征行吞吐，边/s是逐边算子行吞吐，均不含整网、邻居构建、能量或力。O2 canonical/full分图，SO2/ICTP是不同方程的绝对成本参照，不作跨方程加速。
[节点TP](../../figures/o2-hp-angular-operators-20261121-r1/repair-20261122-r1/O2_TP_atoms_repair20261122.pdf) · [边TP](../../figures/o2-hp-angular-operators-20261121-r1/repair-20261122-r1/O2_TP_edges_repair20261122.pdf) · [canonical UVU/Radial](../../figures/o2-hp-angular-operators-20261121-r1/repair-20261122-r1/O2_UVU_Radial_canonical_repair20261122.pdf) · [full UVU/Radial](../../figures/o2-hp-angular-operators-20261121-r1/repair-20261122-r1/O2_UVU_Radial_full_repair20261122.pdf) · [SO2Linear](../../figures/o2-hp-angular-operators-20261121-r1/repair-20261122-r1/SO2Linear_edges_repair20261122.pdf) · [ICTP](../../figures/o2-hp-angular-operators-20261121-r1/repair-20261122-r1/ICTP_atoms_edges_repair20261122.pdf)

## 不变的合同与来源

GBU2 H10080GB，torch2.11.0+cu128/Triton3.6.0。FP32输入/权重/累加/参考/误差统计，TF32/AMP/CUDA Graph关闭。同时满足逐分量1e-5+1e-4|ref|和2e-5+2e-5|ref|，relative L2≤1e-5，不放宽门槛。两种子20261003/20261004、尺度0.1/1，E16384全负载，参考可分4096行，统计始终每512行。
构建、资格、五轮计时、每角色显存用分离进程。每轮30次预热、30次同步墙钟，CUDA event辅助；五轮角色顺序轮换/反转，同一job/GPU/输入/权重/资格哈希配对。峰值allocated/reserved来自新CUDA上下文与分配器。计时消费各自原生API布局，包含径向仿射前向、TP及所需输入VJP，不含输入生成、逻辑基变换、参考验证及构建。这些是预编译后的Triton JIT稳态测量，不是离线动态AOT验收。整数角阶，无半整数/半精度编译；冻结参数，只验前向和所有位置相关输入一次VJP，不声明训练、参数梯度或二阶通过。
官方96815838与fork296e61f1的14份可执行源码一致，执行包0.1.6.dev0；Eqx c646c700/TACE0.2.5 CUDA；ICTP f40592a5仅支持L≤3，4–8不支持留空；cuEq0.13 exact O2仅用于UUU。旧TP/HPX/SO2测量沿用原合格回执，新增角色的来源见快照和源封印。

## 资源、连接与交付

本续轮14个GPU作业、1.7047GPUh，含失败与取消；角阶轮含续轮共26.7458GPUh，项目113.4278/500GPUh。
早前SSH连接曾被关闭，并有一次认证失败；同一地址、端口与密钥后来恢复。主机/sshd无本次重启证据，账号不能读取服务端系统日志，根因未确定。不能据此断言主机宕机、密钥失效、限流或共享存储故障。
Tavotto插件及独立引擎安装为0.21.3，当前MCP仍加载旧0.21.0，需重连才加载新版。本轮使用已授权绘图例外完成Tavotto打开/预检/导出；数值、汇总及初始绘图均经Slurm。150mm、Times New Roman、无框图例；最终预检与导出证据另存，不将安装版本冒充运行中版本。

九图Tavotto最终导出均0错误、0警告、0不可验证项。无量纲角阶/加速比的单位格式提示保留。新图名及数据比值文件均使用唯一名称，避免父子图库同名解析到旧源码；报告PNG/PDF固定文件按最终导出原字节同步。

[逐项快照](SNAPSHOTS.zh-CN.md) · [最终Tavotto导出与哈希](provenance/TAVOTTO-FINAL-EXPORTS.json) · [同卡配对独立审计](provenance/FINAL-PAIRED-AUDIT.json) · [独立解包复算QA](provenance/DELIVERY-QA.json) · [归档索引](provenance/DELIVERY.json)

[数值独立审计](provenance/REPAIR-GATE-AUDIT.json) · [构建候选资格审计](provenance/BOUNDED-GATE-AUDIT.json) · [逐点延迟/吞吐/显存](summary.csv) · [同卡配对](speedups.csv) · [原始状态](status.csv) · [汇总审计](SUMMARY.json) · [账本](provenance/BUDGET-AUDIT.json)
