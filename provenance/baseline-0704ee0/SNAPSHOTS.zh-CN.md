# 本续轮逐项快照

起点`6f641cf736a97fa6835678c0eb123f5b2694e80d`，原位保持。全部为独立算子，不含新整网或训练作业。

| 快照 | 更新代码/内容 | 验收边界 |
|---|---|---|
| `57855c4` | 汇总分类修正、Tavotto升级记录 | 原11/12更正为8个真实数值失败/15个构建超时，原453点CSV不变 |
| `756c995` | `diagnose_radial.py`、只读分段诊断入口 | H100317539隔离TP权重伴随与仿射输入伴随，不是性能结果 |
| `df591ce` | `candidate/ordered_matrix.py`及矩阵探针 | 自写IEEE FP32分段/合并，H100317540/317549；不使用FP64或vendor矩阵例程 |
| `a5cbc8c` | `candidate/affine_adjoint.py`、`models.py`新候选角色 | H100317541完整门7/8，L8失败保留，不晋级 |
| `8e23cb5` | 连续分段探针及原生矩阵核名诊断 | H100317550选出L8的8路连续分段；不能仅由核名推断原生分段数 |
| `73a2e61` | 按K形状选取16路交错/128块或8路连续归约 | H100317553、CPU115807，8/8原FP32门与4096检查通过，输入/权重哈希不变 |
| `e48a156` | `candidate/bounded_uvu.py`、`models.py`、`repair_campaign.py` | UVU有界运行时路径循环；径向4096行私有权重重算/释放。H100317559–317561、CPU115809通过13门/6656检查 |
| `f523ee0` | `cold_build.py`、CPU汇总/配对/报告控制脚本 | 冷缓存H100317569、CPU115811，9个完整VJP构建覆盖15个超时槽，最长116.5秒；11份实际部署执行源码匹配 |
| 最终交付 | CSV、角阶图、报告、独立恢复QA和本表 | 最终Git hash见`provenance/FROZEN-GIT.txt`；测量算术核心仍是`e48a156`，不把控制/报告更新当算术提速 |

逐步Git bundle及hash在`provenance/snapshots/`，最终bundle保留所有历史。`a5cbc8c`的失败和已取消的辅助构建作业均保留、记账。

UVU/Radial使用full parity，整数角阶，不是半整数或半精度编译。模型、参考、误差统计均FP32，TF32/AMP/CUDA Graph关闭。前向与一次全部输入VJP分开，不声明参数梯度、二阶或训练通过。

本轮未修改官方/fork上游执行源码。修复角色另列：`public-repair`、`public-unfused-repair`、`eqx-o2-repair`。其中Eqx修复角色的TP仍为原生Eqx CUDA，仅径向仿射输入伴随使用自写核，作为合格对照；自写候选TP使用冻结公共源码的Triton核及有界调度。
径向仿射前向仍为未改的原生层，允许保留cuBLAS；新输入伴随不调用cuBLAS。原ICTC Cartesian/Fischer/UVU方程与输入权重保持一致。

恢复数值核心后仍需满足`runtime.sh`记录的GBU2运行时和工具链。Git bundle不携带编译缓存，公共上游依赖在交付源码归档中；数据/图归档分别保存。新机器测试必须重新验FP32及硬件边界，不能沿用另一GPU的资格SHA冒充同卡配对。
