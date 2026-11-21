# 算子吞吐随角向阶数变化：H100 同卡配对

当前已汇总 **54/88 个配置**、282 个合格曲线点。未测完配置 22 个；状态统计 `{'measured': 54, 'pending': 22, 'timeout': 2, 'unsupported': 10}`。88 个位置含原生 ICTP 不支持的 4–8 阶，完成不能解释为 88 个位置都有有效吞吐。
尚缺五次完整计时或仍在测量的点留空。每个数据点是五个独立进程的中位数，CSV 同时保留 IQR。

## 读取图表

横轴为角向截断 1–8，固定 C64。节点 TP 固定 N4096，边 TP、UVU/RadialUVU、SO2Linear 固定 E16384。原子/s 是逐节点特征算子的行吞吐，不含邻居构建或整网 E/F；边/s 是逐边算子的行吞吐。O2 的角阶是 Mmax，SO2Linear 是 Lmax=Mmax，ICTP 是 Lmax。每个角阶使用可复现的随机特征和固定随机权重，各实现共享同一逻辑输入/权重；不同角阶权重形状不同。这些是独立算子图，不是官方 checkpoint 整网 E/F 的速度图。不同方程和存储宽度只列绝对值，不计算跨方程加速。

![节点 TP 吞吐](../../figures/o2-hp-angular-operators-20261121-r1/O2_TP_atoms.png)

![边 TP 吞吐](../../figures/o2-hp-angular-operators-20261121-r1/O2_TP_edges.png)

![UVU 与径向融合 canonical](../../figures/o2-hp-angular-operators-20261121-r1/O2_UVU_Radial_canonical.png)

![UVU 与径向融合 full](../../figures/o2-hp-angular-operators-20261121-r1/O2_UVU_Radial_full.png)

![SO2Linear](../../figures/o2-hp-angular-operators-20261121-r1/SO2Linear_edges.png)

![ICTP 原生](../../figures/o2-hp-angular-operators-20261121-r1/ICTP_atoms_edges.png)

## 版本、方程与精度

- O2-HP official main `96815838f470652dce3d0e9d5cf7fe2978e949ca`、fork `296e61f1c13b8ad5ff525cbe3d0bd63d4d6adaca`。14 份可执行源码逐字节一致，所以同一曲线明确标注 official (=fork)，不是两个不同算法。
- Eqx-O2/TACE `c646c7000af7adb367d6bd8b7eb5ea74c08dc37c`，CUDA 后端，包名 equivariantx。UUU 使用相同路径与权重；V=1 UVU 精确映射为 u1u。表示、符号和权重变换在准备阶段，核消费各自原生 API 布局。
- cuEq 0.13.0：exact O2 descriptor，uniform_1d，math_dtype=float32，仅作同方程 UUU 对照。
- 本地 HPX `5f278ef91bedece75b0558700afdb76bcf267a1b`：独立输出 preload，保持原路径和权重。冻结权重适配器只请求输入伴随 mask=3，以跳过原库未被本任务使用的 FP64 参数梯度。它明确是输入反向适配入口，非原生训练性能。原位覆盖输入不混入这些曲线。
- ICTP `f40592a5687ec1d03219300ee557b2660f7d0369`：原生 O3 even/weighted UVU，自然宇称，Cartesian+path-expanded 输出。原生只支持 L≤3，4–8 阶实际尝试构造后记 unsupported。灰色曲线是不同方程成本参照，独立图给出其原生吞吐。
- SO2Linear：冻结 V3 原生方程、C64→C64、无附加标量头；自写 CUDA/Triton 使用 c591b16 通用源码并修正 CUDA 工作线程上下文绑定。不同阶数不是固定检查点的整网替换验收。

O2 同时测试 canonical 和 full parity，整数角阶，无半整数编译。所有推理、输入反向、数值参考和误差统计均 FP32，TF32/AMP/CUDA Graph 关闭。上游离线 CG 系数构造保留在计时外。逐分量同时满足 `1e-5+1e-4|ref|` 与 `2e-5+2e-5|ref|`，relative L2≤1e-5，门槛未放宽。全负载、两种子×两尺度验证，独立行参考按 512 行流式计算，补测可用 4096 行减少重复启动，但误差仍按原 512 行窗口逐窗统计；改变统计窗口的5992a12试测已被替代并排除。实际分块和检查窗口写入资格收据。冻结所有共享参数，forward+VJP 包含所有位置相关输入，UVU 包含外部逐边权重，RadialUVU 包含径向隐藏输入；不包含参数梯度或二阶训练。
当前有效曲线点在所有种子、尺度及原统计窗口中的最大 relative L2 为 `1.7900082411870244e-06`。逐点绝对误差和 relative L2 来自原始 FP32 回执，见 [误差汇总](numerics.csv)。

## 同方程配对加速

以下范围仅描述已完成的角阶点，不是对未测阶数的预测，也不是整网加速。加速=同卡基线 wall ms / 候选 wall ms。

| 算子 / 行数 / 宇称 / 边界 | 候选 / 基线 | 已测角阶 | 配对加速范围 |
|---|---|---|---:|
| radial/16384/canonical/forward | public/eqx-o2 | 1,2,3,4,5,6,8 | 0.821–1.138× |
| radial/16384/canonical/forward | public/public-unfused | 1,2,3,4,5,6,8 | 0.604–0.768× |
| radial/16384/canonical/forward_vjp | public/eqx-o2 | 1,2,3,4,5,6,8 | 0.590–0.988× |
| radial/16384/canonical/forward_vjp | public/public-unfused | 1,2,3,4,5,6,8 | 0.470–0.676× |
| radial/16384/full/forward | public/eqx-o2 | 1,2,3 | 0.853–1.144× |
| radial/16384/full/forward | public/public-unfused | 1,2,3 | 0.622–0.851× |
| radial/16384/full/forward_vjp | public/eqx-o2 | 1,2,3 | 0.595–1.004× |
| radial/16384/full/forward_vjp | public/public-unfused | 1,2,3 | 0.500–0.675× |
| so2/16384/natural/forward | so2-cuda2/so2-native | 1,2,3,4,5,6,7,8 | 0.724–0.891× |
| so2/16384/natural/forward | so2-triton/so2-native | 1,2,3,4,5,6,7,8 | 1.880–18.046× |
| so2/16384/natural/forward_vjp | so2-cuda2/so2-native | 1,2,3,4,5,6,7,8 | 0.788–1.200× |
| so2/16384/natural/forward_vjp | so2-triton/so2-native | 1,2,3,4,5,6,7,8 | 2.514–18.011× |
| tp/16384/canonical/forward | local-hpx/cue-o2 | 1 | 1.540–1.540× |
| tp/16384/canonical/forward | local-hpx/eqx-o2 | 1 | 1.934–1.934× |
| tp/16384/canonical/forward | local-hpx/public | 1 | 1.093–1.093× |
| tp/16384/canonical/forward | public/cue-o2 | 1,2,3,4,5 | 1.391–1.720× |
| tp/16384/canonical/forward | public/eqx-o2 | 1,2,3,4,5 | 1.603–1.823× |
| tp/16384/canonical/forward_vjp | local-hpx/cue-o2 | 1 | 2.889–2.889× |
| tp/16384/canonical/forward_vjp | local-hpx/eqx-o2 | 1 | 2.188–2.188× |
| tp/16384/canonical/forward_vjp | local-hpx/public | 1 | 1.569–1.569× |
| tp/16384/canonical/forward_vjp | public/cue-o2 | 1,2,3,4,5 | 1.833–2.207× |
| tp/16384/canonical/forward_vjp | public/eqx-o2 | 1,2,3,4,5 | 1.315–1.402× |
| tp/16384/full/forward | local-hpx/cue-o2 | 1 | 1.640–1.640× |
| tp/16384/full/forward | local-hpx/eqx-o2 | 1 | 1.863–1.863× |
| tp/16384/full/forward | local-hpx/public | 1 | 1.096–1.096× |
| tp/16384/full/forward | public/cue-o2 | 1,2,3,4,5 | 1.518–1.956× |
| tp/16384/full/forward | public/eqx-o2 | 1,2,3,4,5 | 1.793–2.206× |
| tp/16384/full/forward_vjp | local-hpx/cue-o2 | 1 | 2.991–2.991× |
| tp/16384/full/forward_vjp | local-hpx/eqx-o2 | 1 | 2.135–2.135× |
| tp/16384/full/forward_vjp | local-hpx/public | 1 | 1.600–1.600× |
| tp/16384/full/forward_vjp | public/cue-o2 | 1,2,3,4,5 | 1.860–2.225× |
| tp/16384/full/forward_vjp | public/eqx-o2 | 1,2,3,4,5 | 1.226–1.492× |
| tp/4096/canonical/forward | local-hpx/cue-o2 | 1 | 1.490–1.490× |
| tp/4096/canonical/forward | local-hpx/eqx-o2 | 1 | 1.847–1.847× |
| tp/4096/canonical/forward | local-hpx/public | 1 | 1.073–1.073× |
| tp/4096/canonical/forward | public/cue-o2 | 1,2,3,4 | 1.385–1.715× |
| tp/4096/canonical/forward | public/eqx-o2 | 1,2,3,4 | 1.443–1.934× |
| tp/4096/canonical/forward_vjp | local-hpx/cue-o2 | 1 | 2.931–2.931× |
| tp/4096/canonical/forward_vjp | local-hpx/eqx-o2 | 1 | 2.230–2.230× |
| tp/4096/canonical/forward_vjp | local-hpx/public | 1 | 1.594–1.594× |
| tp/4096/canonical/forward_vjp | public/cue-o2 | 1,2,3,4 | 1.865–2.063× |
| tp/4096/canonical/forward_vjp | public/eqx-o2 | 1,2,3,4 | 1.206–1.419× |
| tp/4096/full/forward | public/cue-o2 | 1,2,3,4 | 1.517–1.893× |
| tp/4096/full/forward | public/eqx-o2 | 1,2,3,4 | 1.296–1.730× |
| tp/4096/full/forward_vjp | public/cue-o2 | 1,2,3,4 | 1.840–2.395× |
| tp/4096/full/forward_vjp | public/eqx-o2 | 1,2,3,4 | 1.107–1.402× |
| uvu/16384/canonical/forward | public/eqx-o2 | 1,2,3,4,5,6,8 | 1.638–2.566× |
| uvu/16384/canonical/forward_vjp | public/eqx-o2 | 1,2,3,4,5,6,8 | 1.286–1.737× |
| uvu/16384/full/forward | public/eqx-o2 | 1,2,3,4,5 | 1.682–2.231× |
| uvu/16384/full/forward_vjp | public/eqx-o2 | 1,2,3,4,5 | 1.316–1.940× |

## 同卡峰值显存配对

下表只使用显存阶段双方 GPU UUID/cohort 相同的记录，数值为候选/基线比，小于1表示降低。缺失或不同卡记录不相除。

| 算子 / 行数 / 宇称 / 边界 | 候选 / 基线 | 已配对角阶 | allocated 比范围 | reserved 比范围 |
|---|---|---|---:|---:|
| radial/16384/canonical/forward | public/eqx-o2 | 1,2,3,4,8 | 0.663–0.858 | 0.707–0.901 |
| radial/16384/canonical/forward | public/public-unfused | 1,2,3,4,8 | 0.663–0.858 | 0.707–0.901 |
| radial/16384/canonical/forward_vjp | public/eqx-o2 | 1,2,3,4 | 1.014–1.052 | 0.836–1.078 |
| radial/16384/canonical/forward_vjp | public/public-unfused | 1,2,3,4 | 1.007–1.047 | 0.989–1.007 |
| radial/16384/full/forward | public/eqx-o2 | 1,2,3 | 0.614–0.704 | 0.627–0.812 |
| radial/16384/full/forward | public/public-unfused | 1,2,3 | 0.614–0.704 | 0.651–0.820 |
| radial/16384/full/forward_vjp | public/eqx-o2 | 1,2,3 | 1.005–1.015 | 0.976–1.000 |
| radial/16384/full/forward_vjp | public/public-unfused | 1,2,3 | 1.000–1.010 | 1.000–1.006 |
| so2/16384/natural/forward | so2-cuda2/so2-native | 1,2,3,4,5,6,7,8 | 0.750–0.772 | 0.692–1.000 |
| so2/16384/natural/forward | so2-triton/so2-native | 1,2,3,4,5,6,7,8 | 0.750–0.772 | 0.692–1.000 |
| so2/16384/natural/forward_vjp | so2-cuda2/so2-native | 1,2,3,4,5,6,7,8 | 0.646–0.786 | 0.597–0.804 |
| so2/16384/natural/forward_vjp | so2-triton/so2-native | 1,2,3,4,5,6,7,8 | 0.646–0.786 | 0.597–0.804 |
| tp/16384/canonical/forward | local-hpx/cue-o2 | 1 | 1.000–1.000 | 0.756–0.756 |
| tp/16384/canonical/forward | local-hpx/eqx-o2 | 1 | 1.000–1.000 | 0.756–0.756 |
| tp/16384/canonical/forward | local-hpx/public | 1 | 1.000–1.000 | 1.000–1.000 |
| tp/16384/canonical/forward | public/cue-o2 | 1,2,3,4,5 | 1.000–1.000 | 0.753–0.910 |
| tp/16384/canonical/forward | public/eqx-o2 | 1,2,3,4,5 | 1.000–1.000 | 0.753–0.910 |
| tp/16384/canonical/forward_vjp | local-hpx/cue-o2 | 1 | 1.000–1.000 | 1.000–1.000 |
| tp/16384/canonical/forward_vjp | local-hpx/eqx-o2 | 1 | 1.000–1.000 | 1.000–1.000 |
| tp/16384/canonical/forward_vjp | local-hpx/public | 1 | 0.986–0.986 | 1.000–1.000 |
| tp/16384/canonical/forward_vjp | public/cue-o2 | 1,2,3,4,5 | 1.014–1.044 | 0.901–1.000 |
| tp/16384/canonical/forward_vjp | public/eqx-o2 | 1,2,3,4,5 | 1.014–1.043 | 0.901–1.000 |
| tp/16384/full/forward | local-hpx/cue-o2 | 1 | 1.000–1.000 | 0.718–0.718 |
| tp/16384/full/forward | local-hpx/eqx-o2 | 1 | 1.000–1.000 | 0.718–0.718 |
| tp/16384/full/forward | local-hpx/public | 1 | 1.000–1.000 | 1.000–1.000 |
| tp/16384/full/forward | public/cue-o2 | 1,2,3,4,5 | 1.000–1.000 | 0.718–0.905 |
| tp/16384/full/forward | public/eqx-o2 | 1,2,3,4,5 | 1.000–1.000 | 0.718–0.905 |
| tp/16384/full/forward_vjp | local-hpx/cue-o2 | 1 | 1.000–1.000 | 0.890–0.890 |
| tp/16384/full/forward_vjp | local-hpx/eqx-o2 | 1 | 1.000–1.000 | 0.890–0.890 |
| tp/16384/full/forward_vjp | local-hpx/public | 1 | 0.976–0.976 | 1.000–1.000 |
| tp/16384/full/forward_vjp | public/cue-o2 | 1,2,3,4,5 | 1.024–1.054 | 0.890–0.966 |
| tp/16384/full/forward_vjp | public/eqx-o2 | 1,2,3,4,5 | 1.024–1.054 | 0.890–0.966 |
| tp/4096/canonical/forward | local-hpx/cue-o2 | 1 | 1.001–1.001 | 0.524–0.524 |
| tp/4096/canonical/forward | local-hpx/eqx-o2 | 1 | 1.001–1.001 | 0.524–0.524 |
| tp/4096/canonical/forward | local-hpx/public | 1 | 1.000–1.000 | 1.000–1.000 |
| tp/4096/canonical/forward | public/cue-o2 | 1,2,3,4 | 1.001–1.001 | 0.524–1.000 |
| tp/4096/canonical/forward | public/eqx-o2 | 1,2,3,4 | 0.999–1.001 | 0.524–1.000 |
| tp/4096/canonical/forward_vjp | local-hpx/cue-o2 | 1 | 1.000–1.000 | 0.524–0.524 |
| tp/4096/canonical/forward_vjp | local-hpx/eqx-o2 | 1 | 1.000–1.000 | 0.524–0.524 |
| tp/4096/canonical/forward_vjp | local-hpx/public | 1 | 0.986–0.986 | 1.000–1.000 |
| tp/4096/canonical/forward_vjp | public/cue-o2 | 1,2,3,4 | 1.014–1.037 | 0.524–1.000 |
| tp/4096/canonical/forward_vjp | public/eqx-o2 | 1,2,3,4 | 1.014–1.036 | 0.524–1.000 |
| tp/4096/full/forward | public/cue-o2 | 1,3,4 | 1.001–1.001 | 0.512–1.000 |
| tp/4096/full/forward | public/eqx-o2 | 1,3,4 | 0.999–1.001 | 0.512–1.000 |
| tp/4096/full/forward_vjp | public/cue-o2 | 1,3,4 | 1.025–1.048 | 0.512–0.859 |
| tp/4096/full/forward_vjp | public/eqx-o2 | 1,3,4 | 1.025–1.046 | 0.512–0.859 |
| uvu/16384/canonical/forward | public/eqx-o2 | 1,2,3,4,5,6,8 | 1.000–1.000 | 0.771–1.028 |
| uvu/16384/canonical/forward_vjp | public/eqx-o2 | 1,2,3,4,5,6,8 | 1.005–1.008 | 0.836–1.000 |
| uvu/16384/full/forward | public/eqx-o2 | 1,2,3,4,5 | 1.000–1.000 | 0.771–1.011 |
| uvu/16384/full/forward_vjp | public/eqx-o2 | 1,2,3,4,5 | 1.005–1.005 | 0.895–1.000 |

![同方程 TP 对照](../../figures/o2-hp-angular-operators-20261121-r1/O2_TP_same_equation_speedup.png)

![同方程 SO2 与 Radial 对照](../../figures/o2-hp-angular-operators-20261121-r1/SO2_Radial_same_equation_speedup.png)

## 测量和失败记录

GBU2 NV_H100 / NVIDIA H100 80GB，PyTorch2.11.0+cu128、Triton3.6.0。各配置的对照臂按顺序轮换/反转，每边界预热30次、同步墙钟测30次，CUDA event 作为辅助列。构建、数值资格、计时和峰值显存使用分离进程。没有把编译、源封印或独立参考验证时间计入稳态吞吐。每个比值核对 GPU UUID、输入/权重/资格 SHA、运行时和边界一致。
后续部分测量通过只预载 CPU 代码的 forkserver 减少重复导入。四阶段短测确认不同 PID、相同 GPU 和四臂 FP32 均通过；每个计时/显存子进程仍从未初始化 CUDA 的状态开始，CUDA 上下文和分配器独立。CSV 明列 process_mode，不能将准备开销减少冒充算子加速。

allocated/reserved 峰值均来自新进程；独立显存批次在自己的同卡 GPU 上重新过 FP32 门，CSV 分别记录计时卡和显存卡 UUID。早期 TP/UVU/Radial 显存准备误留一份未用的原布局 cotangent，旧显存记录已被修正批次替代；这些计时不受影响。当前 17 个合格计时点尚无匹配的修正显存。

早期 SO2 载荷漏带 matrix.cu，修复后又检测到 cuLaunchKernel status201。该重复失败不计有效吞吐；修复为在反向工作线程绑定模块所属 CUDA 上下文，调用后恢复。新批次先强制空上下文回归，再完整重做原数值门和配对。高阶构建诊断按角色/阶段保存期限，超时不是数值失败，也不是不支持；失败/OOM/unsupported 均不填零吞吐。

## 资源、文件与快照

账本快照：本项目已分配 34.3956 GPUh，本角阶轮 10.5061 GPUh，最终值以收尾 Slurm 账本为准。本轮包含失败的上限包络20 GPUh，项目原总上限500 GPUh保持。CPU 汇总不记 GPUh。

[逐点数据](summary.csv) · [同方程配对比](speedups.csv) · [失败与状态](status.csv) · [88位置覆盖](coverage.csv) · [机器审计](SUMMARY.json)

图宽150mm，Times New Roman，无框图例。初始绘图与汇总在 CPU Slurm，Tavotto 打开、预检及导出使用用户已授权的本轮绘图例外。原先整网规模图保留为独立附录，不充当本轮角阶吞吐主图。训练及新增整网测量均未启动。
