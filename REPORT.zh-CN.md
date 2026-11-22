# 0704ee0 安全候选：入口、重算状态与归约调度

分支 `codex/o2-safety-candidate-20261122` 从 `0704ee0d3aa4f8d8635a6626949dde4433dc2530` 独立创建，原目录和交付不改。本轮完成源码修复及逐项快照，**没有执行 Python、导入、语法编译、构建、测试、数值或性能测量，没有提交 Slurm 作业、部署 GBU2、恢复整网或训练**。本候选为 `UNTESTED`，不能继承基线通过数、吞吐或显存结果。

## 1. 流式入口布局合同

`candidate/contracts.py` 检查 dense strided 二维 CUDA FP32、同设备、行数及表示/隐藏/权重宽度、Linear 参数形状与冻结状态，以及四份 TP 缓冲的 shape、dtype、设备、连续性与冻结状态。lazy negative/conjugate view 明确拒绝，避免 `.contiguous()` 仍返回带逻辑标志的 view 而 raw Triton 读取底层值。输出余切同样检查形状、设备和布局。

`bounded`、`streamed`、`unfused` 在 custom Function 外连续化输入，保留复制的 autograd 连接。没有 dtype 转换、权重改写、输入 donation 或新的数值近似。非法输入报错；结构合法但超出调度覆盖域则按下文回退。

## 2. 前向拥有的参数和计划

`candidate/frozen_state.py` 捕获前向实际使用的 weight、bias、四份 TP 缓冲及不可变 TP 元数据。记录身份、数据/storage 指针、offset、shape/stride、dtype/device、requires_grad 和 version。前向结束和反向开始检查普通原位修改、替换、迁移、维度或计划变化。

流式前向和反向重算都对保存的同一 weight/bias 调用原生 `F.linear`，TP 核使用保存的计划和缓冲。反向不读取当前模块参数来计算。仿射输入伴随也使用 saved_tensors 中的前向权重。参数变动时拒绝继续，绝不临时换公式或换路线。TP 缓冲同样接入 autograd 的 saved-tensor 版本检查。无版本计数的 inference tensor 不进入自写保护路径。

不复制/哈希参数值，不保证绕过 version 的原始存储写入、`.data` 写入或并发修改安全。仅支持冻结参数的一次输入 VJP，保留 `once_differentiable`，未新增参数梯度、训练或二阶支持。

## 3. 确切适用域与原生回退

新增 `candidate/reduction_policy.py` 替代 `K >= 49152` 广义阈值。只读取 host 元数据，不在线调参、不读 tensor 数值、不改变全局精度设置、不显式同步 CUDA。decision 记录 route、mode、configuration、execution、矩阵行数、profile、回退原因及 `candidate_validated=False`。`models.py` 记录候选路线，原始输入/权重/参考构造不变。

自写结构范围：NVIDIA H100 80GB HBM3 / sm90，torch 2.11.0+cu128 / CUDA 12.8、Triton 3.6.0，`CUBLAS_WORKSPACE_CONFIG=:16:8`；E16384、C64、V1、external/unshared，两个输入均 full parity、相同整数角阶，component / path-none。径向为 H64、有 bias 的冻结普通 Linear。独立 bounded UVU 限 L6–8，径向限 L4–8。

| L | 确切 K | 原归约配置 | 径向矩阵行数 |
|---|---:|---|---|
| 4 | 15616 | `16x1b@128` | unfused 16384 / streamed 4096 |
| 5 | 23296 | `16x1b@128` | 同上 |
| 6 | 32512 | `16x1b@128` | 同上 |
| 7 | 43264 | `16x1b@128` | 同上 |
| 8 | 55552 | `-8x1b` | 同上 |

K 已从 0704ee0 原 `measurements/paired-repair-e48a156/qualify-bounded-radial/` 的 L4/L5 回执及 `qualify-bounded-uvu/` 的 L6/L7/L8 回执逐项核对。它们仅限定历史结构覆盖，**不保证任意 FP32 数据过门**。完整设备/工具链/输入改变都需重新验收。

- `auto`（旧 `shape-v2` 别名）：域内选原配置，域外原生 TP/Linear 回退并记录原因。流式域外恢复完整原生组合，可能失去显存优势。
- `native`：显式原生，不运行本轮自写替换，不能记作候选提速。
- `required`：域外报错，避免要求自写核却悄悄回退。原始归约字符串不再作为公共调度政策；`ordered_matrix.scheduled` 保留为内部/诊断原语。
- 自定义 forward、module hooks、已编译 module、tensor subclass、inference mode、functorch/ProxyTensor/FakeTensor、编译 tracing 或 CUDA Graph capture 不进入自写路线。交给原生语义处理不代表已验证原生支持这些机制。
- TF32/AMP 或强制 TF32 环境覆盖违反共同 FP32 合同，直接拒绝，不以回退降低精度。反向发现运行时/精度上下文变化同样报错，不在中途换调度。

原生回退只恢复原实现，**不代表它的旧数值失败、超时或 OOM 已修复**。候选路线也没有新的数值或性能资格。

## 4. 对照冻结 O3-HPX 历史

参照只读 `runs/o2-hpx-o3-audit-20261117-r1/attachment/ICTC-O3-HPX`，HEAD `45d36f0`：

- `f6c3670` / `67f4dc5`：设备/形状适用域与原生回退，不泛化某张卡的归约配置。
- `45f1e5f` 的 `streamed_gate_message.py`：参数身份、指针、shape、device/dtype/version 失效保护。本轮进一步绑定前向保存对象并拒绝中途变更；同文件的 functorch/proxy/fake 边界一并参考。
- `d9c42a4` 的 `foundation_linear.py`：保留 canonical 仿射前向，避免重新结合造成力舍入差异。本轮未拆改前向方程。
- `1d2ed12` 的 FP64 训练修复仅作背景，未移植 FP64。本轮模型、参考及门控仍为 FP32。

## 5. 冻结边界与静态核对

`candidate/ordered_matrix.py`、`benchmark.py` 相对 0704ee0 逐字节不变。四个公共 TP 核及上游依赖不改；输入/权重生成、种子和原误差门不变，径向分块仍为 4096。两组逐分量门 `1e-5 + 1e-4*abs(ref)`、`2e-5 + 2e-5*abs(ref)` 及 relative L2 ≤ 1e-5 保留，不用 FP64、不放宽阈值。full parity / 整数角阶，TF32、AMP、CUDA Graph 关闭。

仅做文本阅读、Git diff/空白检查、文件哈希和归档封存，见 `provenance/STATIC-REVIEW.txt`。**未做 AST、Python 语法、导入、运行或科学验收，错误路径和调度选择同样未经执行。**

继承 CSV、SUMMARY、旧报告、恢复说明及快照表在 `provenance/baseline-0704ee0/`；其余继承 provenance 回执也均为原历史，详见 `provenance/README.zh-CN.md`。逐项恢复点见 [SNAPSHOTS.zh-CN.md](SNAPSHOTS.zh-CN.md)，最终 Git hash 见 `provenance/CANDIDATE-FROZEN-GIT.txt`。恢复验证须用户另行授权并另建 run，不自动执行继承的 campaign/sbatch/aggregate/绘图脚本。整网和训练继续暂停。
