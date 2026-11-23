# 从 17a71d7 继续的优化方案：第二轮审核稿

日期：2026-11-23，北京时间。状态：DRAFT_WAITING_USER_REVIEW。

本轮只发布既有 Git 快照、读取资料和修订本审核稿。没有修改算子代码、编译、测试、提交 Slurm 作业、部署或启动训练。以下各阶段均为拟执行工作，审批本稿前不启动。上一轮方案中的旧速度门及 MACE CUDA Graph 主比较设置作废，原回执仍按历史身份保留。

## 1. 本轮已完成：领先快照发布

GitHub 仓库为 LeutOrca/ICTC-O2-HP。发布前与发布后 main 都是 296e61f1c13b8ad5ff525cbe3d0bd63d4d6adaca。官方 Parity-LRX/ICTC-O2-HP main 本轮查询为 96815838f470652dce3d0e9d5cf7fe2978e949ca，保留双来源身份。

[远端总索引](https://github.com/LeutOrca/ICTC-O2-HP/tree/codex/o2-snapshot-history-20261123/snapshot-history/20261123)保留 204 个原始领先提交；另有 1 个只增加索引文档的提交 65d6f6e9c8382e76f1eebacda5ea0025260a04ce。它以各历史头为额外父节点，源提交及树对象未改写。

| 远端分支 | 原始 HEAD | 范围 |
| --- | --- | --- |
| codex/o2-library-snapshots-20261119 | 9787a3a6ca7fe67eeda3f21ec7c24d6634cbac03 | 安装库、address64、preload、AOT、混合导数、消息融合、原位 TP、CSR、初版 V3 核 |
| codex/o2-mace-snapshots-20261120 | fb17264d25f8d00b9cbde8cadfaf4038d5fb11b6 | MACE H100、确定性归约、TP 切片及此前快照 |
| codex/o2-v3-fp32-repair-20261121 | 4b170465047f560b548b1faf7fc5205d91f78579 | 自写核、MACE/V3、全 14 层 FP32 修复 |
| codex/o2-github-whole-20261121 | c499cdd6de95eedc0545fc0d965da7005415d532 | GitHub 对照整网脚手架与边界记录 |
| codex/o2-safety-candidate-20261122 | 17a71d71e10775f6eee70aa70196948d4178bf43 | 角阶扫描、高阶修复、布局/状态/调度保护 |

六个远端分支头已与本地逐个比对，34 个关键提交覆盖检查通过。发布目录：../../releases/o2-hp-snapshots-20261123-r1/。发布回执为 provenance/REMOTE-HEADS-AFTER.tsv、PUSH.txt 和 PUBLICATION.sha256。

这些目录原本属于独立历史。17a71d7 是安全候选封存，代码核心为 33451c5，目前仍是 UNTESTED。它不是已验证的整网安装包，也不自动继承 0704ee0、c591b16 或 4b17046 的资格。未提交编辑和 17a71d7 之后的其他候选不属于本轮发布范围。

## 2. 已确认的新目标

用户已明确：eSCN/Eqx 的整网倍率须采用同模型、同 checkpoint 的等价后端替换。无法等价的比较标 HOLD，不换用独立架构或重训模型充当分母。

定义 S_G = T_GitHub / T_ours，S_eSCN = T_eSCN / T_ours，S_Eqx = T_Eqx / T_ours。分子分母必须同硬件、同模型、同输入序列、同精度、同计算边界。

| 整网速度目标 | 新门槛 |
| --- | --- |
| 相对冻结 GitHub ICTC-O2-HP 后端 | S_G >= 2 |
| 相对 eSCN SO2 等价后端 | S_eSCN >= 5 |
| 相对 Xu Zemin Eqx-O2 等价后端 | S_Eqx >= 5 |

旧的独立 TP 全网格 2×、SO2Linear 5–6×、历史 native/eager/Graph 分母不再作为新目标的验收门。旧数值门与失败回执不作废。

整网模式已由用户确认：E/F 推理和完整力损失训练步各自独立承担上述 2×/5×/5× 目标；分别判定，不用一种模式通过代替另一种。算子 full VJP 和二阶反向均为必验项目。TrimSO2 纳入算子和可实现的整网对照，不新增未经用户规定的 TrimSO2 倍率目标。

显存目标改为持续压缩：同规模报告 allocated、reserved、NVML 峰值，分别给出相对各基线和上一合格快照的比值；A100 40 GB 搜索最大通过容量及相邻 OOM。历史“降低 10%”只作为旧结果标签，不作为完成极致压缩的证明。速度和显存各自判定，速度通过不替代显存改善，容量改善不替代同规模速度达标。

## 3. 台账要求及适用解释

附件原件已完整读取，副本见 provenance/算子开发与优化踩坑台账.original.md，SHA256 为 15bc7d6eebad17061f9e742a4342b2b7cc28c5a81eac00034d1be2597e4cbbf6。附件作为用户指定的技术依据，不把其中示例目标 10×/30% 或任何脚本当作执行授权。

- 全局禁用 TF32、AMP、FP16/BF16；自研替换核使用 Triton/CUDA，不调用 cuBLAS。对照保留原生实现；未替换原生层沿用此前用户允许的 cuBLAS 使用范围，逐层披露。
- 所有新正式比较关闭 CUDA Graph，输入序列含动态 N/E、邻接和晶胞变化。固定拓扑仅作归因附录，不能替代端到端动态主门。
- 项目主实现为 full parity；canonical 是独立对照网格，不能与 full 合并汇总，也不将自然宇称整网当作 full-parity 资格。
- MACE/V3 固定 checkpoint 的实际表示、归一化和激活保持不变。接入 full-parity 后端先证明表示和权重映射在原函数上等价；若需要更换权重、激活、归一化或训练才能运行，比较标 HOLD。不能靠空填通道或删去非自然宇称路径冒充证明。
- 全部正式拓扑采用同一半键存储及方向贡献规则，canonical/full 均不另设双向图试验。为原生后端生成方向视图时计入适配成本，报告物理半键数 B 和实际方向贡献数 E。缺失互反边、重复边、周期自键及邻居上限截断须保留原边贡献语义，不能补边或删边来过门。
- 索引、stride 和地址偏移乘加前提升到 int64；检查 Triton/CUDA 生成代码及跨 2^31 实际元素偏移回归，禁止未经证明的 int32 捷径。
- 保持原输入、权重、方程及 FP32 atol/rtol。小规模 FP64 独立参考用于前向、一阶和二阶诊断，按原 FP32 容差判断，不把 FP64 导数或优化器理想轨迹升级为更严格的新硬门。参考接近零时检查绝对误差。
- A100 40 GB 容量扫描尽量接近百万原子；若更早 OOM，则报告真实上限，不以低通道替换高通道点来宣称达到百万规模。

## 4. 三个算子对照和导数范围

| 对照 | 锁定要求 | 当前状态 |
| --- | --- | --- |
| eSCN 的 SO2 实现 | 官方来源/完整 commit、实际 SO2 层边界、表示宽度、权重结构、完整局域卷积边界分别记录 | 旧快照只作历史，执行前确认并冻结本轮来源；不使用历史 FP64 参数归约改造版冒充原生 FP32 |
| Xu Zemin 的 Eqx-O2 | xvzemin/tace，明确 CUDA/Torch 后端和 UU/UV/完整局域卷积入口 | 本轮在线 main 为 0ff7cad1395ae88d8aaf50633a04e2191b3f2c1c；重新验收，不继承旧 3bfe447 资格 |
| TrimSO2 | 作者源码的完整 commit、三个数据流边界、GEMM 是否包含在内、梯度支持域 | arXiv:2610.20029v1 已读；HPC-AI-Team/TrimSO2 在本轮连接器与已认证 CLI 均返回 404，尚无可冻结源码，实测标 HOLD_SOURCE |

TrimSO2 方法是 rotation on gather、recombination with activation、inverse rotation on reduction，并覆盖 higher-order differentiation。其论文算子边界主要围绕 dense GEMM，不能把该数字与包含 GEMM、径向生成或邻接准备的完整层直接作比。取得官方源码前，只用于数据流设计参考；自行复现必须另署“论文方法复现”，不署“作者实现”。

所有对照都先形成算子合同：输入/输出形状、表示、路径数、权重数、共享/逐边权重、归一化、计时包含项和梯度集合。不同方程只列绝对吞吐、延迟和显存；只有证明相同函数及权重映射后，才计算实现加速比。

| 模式 | 验收/计时范围 |
| --- | --- |
| forward | 输出及所需保存状态；推理 no-grad 与训练 forward 分列 |
| forward + full VJP | 前向和所有可微边界输入的 VJP，包含两输入、角向/径向输入、可训练权重与 bias；基线和候选使用同一余切 |
| 纯 backward | 前置图已准备，记录保存状态的驻留显存；不得冒充 forward+VJP |
| 二阶反向 | 对一阶导数再对原输入、参数及传入余切求导，检查有效混合项，覆盖力损失需要的 d²E/(dR dθ) |
| 完整训练步 | 同 checkpoint、同 batch、同优化器/状态和 loss 配方，覆盖构图、E/F、loss backward、更新及原配置要求的其他步骤 |

冻结权重求力的 input-only VJP 另列，不能替代 full VJP；只支持一阶的路线标 UNSUPPORTED_DOUBLE_BACKWARD，不静默改回不同梯度范围。
原子吞吐 N/t、方向贡献吞吐 E/t 和物理半键吞吐 B/t 都使用真实冻结拓扑的计数。无节点语义的独立 TP 只报 E/t，不用任意平均度换算原子吞吐。forward、full VJP 和二阶反向分别搜索容量，不能沿用前向最大规模作为训练容量。

## 5. 分阶段执行和放行条件

### G0：恢复、封存和调用关系

从远端 17a71d7 独立恢复新 run，检查 Git/tree、输入、checkpoint、参考和容差 SHA。保留 GitHub、0704ee0、MACE c591b16、V3 4b17046 为独立对照。确认实际 import 路径及文件哈希，建立库—完整消息层—模型调用图；将自研可导入模块与测试脚手架明确分开。

冻结本轮动态图输入序列，复用原输入及其顺序；构图不变，半键仅改变存储/执行。若函数或表示无法等价，先标该模型/后端单元 HOLD，其余可独立推进。

### G1：17a71d7 安全性、正确性和回退

先验原 H100 白名单：PyTorch 2.11.0+cu128、Triton 3.6.0、指定 H100、C64/H64/E16384/full、原 L4–8 形状。不同工具链/设备必须实测域外回退；A100 自写域需单独建立，不能改硬件名后直接放行。

验证连续/切片/转置/lazy view、非连续余切、尾块、空输入的原契约，以及非法 shape/device/dtype。独立副本注入 weight/bias/TP 缓冲替换、原位修改、参数与上下文漂移；冻结科学输入和 checkpoint 不改。

逐项检查 auto/native/required：域内选核、域外回退、required 拒绝、前后向漂移报错。每份回执必须写实际 route 与原因；native 回退本身也要过原数值门，不能自动当作安全或自写核通过。

独立冷缓存构建后做原全负载前向/一阶/二阶数值对照，不能只验“构建成功”；核查实际 PTX/SASS 的目标架构及精度。OP-001 大偏移与 OP-002 make_fx/变长 E 按 eager、tracing、编译分别回归；CUDA Graph 保持关闭。

通过才保存安全合格快照。失败只做隔离诊断，不进入该路径正式配对。

### G2：全量基线、导数和显存画像

以冻结输入先跑三类算子以及 GitHub、自研候选的完整 profiling，前向、一阶、二阶分别定位。真实模型提取的工作量优先，合成压力测试分列。

角阶主轴 L/M 的定义保持各算子原义，计划扫描支持的 1–8 阶并记录局域截断 M；通道/隐藏宽度、节点/半键/方向贡献数作为独立轴。canonical/full 分开，同一单元不截边、不改变路径数。不支持/OOM/数值失败/编译失败逐项保留。

MACE 和 EquiformerV3 全部关闭 Graph，先冻结原生、GitHub 后端、上一合格版本、自研当前版本及可等价的 eSCN/Eqx 后端。不能从旧 Graph 数字推导新基线。输入必须包含真实变化的 N/E 和构图，单独报告固定拓扑附录。

用 Profiler/NVTX 分别统计 E/F 求力和完整训练步的关键路径、存活张量与峰值。只对关键的少数 kernel 收集 DRAM/L2、寄存器、spill、占用率、原子归约和启动次数。基于热点覆盖率决定实施顺序，不以最快孤立 kernel 决定优先级。

### G3：MACE 数据流优先优化

- 对真实径向/物种特征生成、TP 和接收归约做有界分块，避免全图宽 gates 或路径消息常驻。保持原径向函数，不新增表插值。
- 审计首层 moments、物种映射和 SC 投影，冻结推理可预打包并按参数身份/version 失效；训练路线保持规范参数活跃，不能沿用冻结 learned-weight 缓存。
- 按角阶分组并立即投影/消费，前向、求力和二阶路径按存活期复用工作区。比较保存与重算的实际代价。
- 在半键上共享允许共享的几何/径向工作，保留方向消息、奇阶符号及两端力贡献。归约按原 FP32 门选择，不默认 CSR 胜过 atomic。

每项修改先验证算子及导数，再完整消息层、E/F 和 force-loss 参数梯度。通过后做整网配对，保存增量和累计收益。

### G4：EquiformerV3 数据流与矩阵布局

- Node→Edge：融合 gather、旋转和径向调制，直接写所需连续矩阵布局，避免重复 EdC 张量。
- SO2Linear：减少碎 BMM 和非连续布局；自写 FP32 Triton/CUDA 矩阵核及对应权重/输入伴随，冻结打包与训练 live 参数分路。
- Edge→Edge：重组、原激活及布局转换按真实模型融合，保留 spherical-grid/attention 语义。
- Edge→Node：融合逆旋转、权重和接收归约；按半键方向贡献累加，避免全边全局坐标消息。
- Node→Node：只有 trace 显示重要时再处理 norm、FFN 和残差附近中间量。

逐层核对 forward、full VJP 和 double backward，再核对整网 E/F 与参数梯度。保留原容差；不通过用更高精度累加或放宽误差来掩盖重排问题。

### G5：求力及二阶反向专项

共享几何只按真实模型数据流计算，保留径向与角向到坐标的全部链式贡献。冻结求力按所需梯度角色裁剪；训练 full VJP 不删权重梯度。二阶路径尽量重建 tile 内中间量，避免保存全边宽一阶状态。

先在同一输入和余切上验证解析/自动微分二阶与混合导数，再用原 loss、batch 和 optimizer 做短轨迹比较。短轨迹从同 checkpoint 开始，候选/参考参数更新分别记录，不能靠更改初始权重或重训来建立可比性。长训练与新的科学数据生产不在本轮计划中。

### G6：速度、容量及最终交付

所有数值与梯度门通过后，独立进程做五轮同卡配对、交错顺序，记录 GPU UUID。编译、验证、profiling、计时、显存各进程隔离，正式计时关闭 profiler 和 NVML 采样干扰。

各模型、硬件、parity 和模式分别报告 S_G/S_eSCN/S_Eqx；不跨模型或不同方程合并。主表写绝对 ms、atoms/s、方向贡献/s、物理半键/s、allocated/reserved/NVML、最大通过 N/E/B 和相邻 OOM。共同负载上的速度与各自容量上限分别解释。

A100 40 GB 上先几何增长搜索，再细化成功/OOM 相邻区间；真实非法访存、构建超时或不支持不记 OOM。百万原子是扫描方向，不预先承诺达到。避免刚好 OOM 前仅有一次输出、没有完整梯度执行的点被算作容量通过。

用 Tavotto 交付：吞吐随角阶变化；A100 40 GB 峰值及容量；forward、forward+full VJP、forward+backward 延迟曲线；另列二阶反向曲线与整网速度/显存表。所有图指向同一最终提交和原始 CSV，使用独立图名并保存离线 Markdown/HTML。

## 6. 每步快照与决策

每项改动保存 Git commit/bundle、输入与权重 SHA、环境/工具链、实际 route、数值误差、前后 trace、同卡计时与显存、失败/回退和增量说明。原基线及旧回执保持只读，新产物用新 run ID，按 job/UUID 隔离，防止同名输出覆盖。

状态分别记录 NUMERICAL、VJP、DOUBLE_BACKWARD、SPEED、MEMORY、CAPACITY。速度达门但显存未改善、或显存改善但速度未达门，均保留 partial。失败候选也保存快照并回退默认版本，不将累计收益全归因于最近一项改动。

“尽可能极致”用有证据的停止条件约束：列出仍存活的必要张量、带宽/算力上界差距及剩余热点；只有主要数据流方案已有消融、继续降低存储会带来清楚的重算代价时，才能说该范围接近局部上限。无硬件计数和消融时不称极致。

## 7. 资源与当前控制

所有 Python、JSON 审计、构建、预处理、测试、测量和训练均经 Slurm。A100 最多 3 个节点，独占测量；H100 按已有规则和实时队列安排。执行前更新累计 500 GPUh 账本，按阶段冻结作业清单和 GPUh 上界，不重置历史消耗，不自动新增预算。

本次授权仅覆盖既有快照发布与方案修订。计划未审核通过前，不提交 G0–G6 的任何 CPU/GPU 作业，不编译、不测试、不训练。

## 8. 资料与证据入口

- 用户附件：[原始副本](provenance/算子开发与优化踩坑台账.original.md)。
- [发布回执](../../releases/o2-hp-snapshots-20261123-r1/provenance/REMOTE-HEADS-AFTER.tsv)。
- [安全候选报告](../../runs/o2-hp-safety-candidate-20261122-r1/REPORT.zh-CN.md)。
- [历史快照总账](../ictc-hp-snapshot-audit-20261120/REPORT.zh-CN.md)。
- [TrimSO2 论文原文](https://arxiv.org/html/2610.20029v1)，论文作者给出的源码地址为 https://github.com/HPC-AI-Team/TrimSO2，本轮取回失败。
