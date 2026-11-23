# 已获批准的续轮入口

用户2026-11-23已直接批准执行，当前控制见AUTHORIZATION.zh-CN.md。17a71d7原目录继续只读；本目录继承的“保持停测”属于批准前历史。

当前执行G1，详见PROGRESS.zh-CN.md及provenance/approved-execution/。全部相关原FP32正确性/full VJP/二阶/回退门通过后，才进入MACE/EquiformerV3整网性能。原FP32门、输入、权重和方程不变，TF32/AMP/CUDA Graph关闭，保持半键/full parity及同模型/checkpoint等价后端要求。

最近数学修改为85510d1：分页径向隐藏输入的一阶及两路二阶长K矩阵均采用有序FP32归约。fddf444追加FP64诊断的原生FP32对照。失败候选14e7d16/d621827保留，9d605b4为整网入口审计文档。

GBU2不可变目录 `/public_hw/home/phy_bjxiao/o2-benchmark/runs/o2-hp-approved-20261123-r1-fddf444` 已提交318804构建、318805径向32项小例、318806补充FP64诊断、318807在小例通过后构建E16384/L8、318808完整径向资格。依赖失败自动终止后续，submit_once.sh持久化ID。318804最后一次可见为RUNNING，之后连续SSH握手超时，尚无最新终态回执，不能宣称通过或失败。

GBU2 UVU全负载318787/L6、318789/L7、318767/L8已COMPLETED；L8已有四项通过，L6/L7仍待统一回收并CPU审计。AIDAM A100域外回退116726→116727因资源PENDING，不可取消其他用户作业或用H100结果替代。

连接恢复后先只读核对上述作业WorkDir、Slurm状态、结果/日志和submissions/*.job，再回收，不启动同名副本。GBU2为10.10.4.99/login02/NV_H100，GBU10.10.3.99不是该硬件。所有Python及JSON/数值审计仍经Slurm。

G1失败则隔离定位、保存新数学快照，不放宽容差或先开整网配对。G1通过后先按WHOLE-MODEL-ROUTES.zh-CN.md证明full-parity/半键映射，再对原完整模型核验E/F、全部参数梯度及短训练轨迹。不能等价的单元HOLD。E/F推理和完整训练分别2×/5×/5×，速度/显存独立。

本轮采用optimize-for-gpu技能的方法流程，来源K-Dense Scientific Agent Skills（arXiv:2609.00065）。500 GPUh预算沿用，刷新账本后记精确实耗，失败和重复构建均计入。
