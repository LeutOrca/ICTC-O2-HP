# 本轮独立安全候选快照

基线 `0704ee0d3aa4f8d8635a6626949dde4433dc2530`，原目录不改。以下均仅源代码与文档快照，未编译、未执行、未获新数值或性能资格。

| 步骤 | Git commit | 修改 |
|---|---|---|
| 1 布局入口 | `d45cb87` | 新增 `contracts.py`；公共 UVU/streamed 入口检查 shape/device/dtype，在 custom Function 外连续化；旧结果移入基线区 |
| 2 前反向一致性 | `4771d0b` | 新增 `frozen_state.py`；保存前向 weight/bias/TP 缓冲和计划；反向重算绑定保存对象并检查身份/version |
| 3 调度覆盖域 | 见 `provenance/snapshots/03-policy.git` | 新增 `reduction_policy.py`；完整设备/软件/形状白名单，auto/native/required；原生回退与上下文漂移报错，补 lazy view/余切/冻结 buffer 边界 |

bundle/hash 在 `provenance/snapshots/`，最终完整历史在 `provenance/candidate-final.bundle`，最新 hash 在 `provenance/CANDIDATE-FROZEN-GIT.txt`。依赖随独立源码压缩包保存，不在 Git bundle 内。旧 0704ee0 快照表在 `provenance/baseline-0704ee0/`，不混为本轮步骤。

full parity、整数角阶、严格 FP32，无半整数/半精度编译；本轮根本未编译。容差、输入/权重/方程不变。自写伴随内无 vendor 矩阵例程，未改仿射前向保留原生。CUDA Graph、整网及训练未启动。
