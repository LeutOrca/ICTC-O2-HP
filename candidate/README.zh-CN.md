# 安全候选 API，未运行验证

公共入口：`bounded_uvu.bounded`、`bounded_uvu.streamed`、`bounded_uvu.unfused`、`affine_adjoint.affine`。默认 `auto`，可选 `native` 或 `required`，旧 `shape-v2` 为 `auto` 别名。单独 `affine` 无 TP 上下文时 auto 原生、required 报错。

入口检查 CUDA FP32、shape/device/layout 和冻结计划/参数，在 custom Function 外连续化特征；lazy negative/conjugate view 需调用者先物化。参数、buffer 或上下文在前反向间变化会报错。低层 `.apply`、`forward_into`、`inputs_into`、`weights_into`、`ordered_affine`、`ordered_matrix.scheduled` 是内部原语，不是绕过公共合同的受保护接口。

归约表严格限定历史 H100/工具链/E16384/C64/H64/full/V1 的 L4–8，独立 UVU 仅 L6–8。decision 的 route/reasons 解释自写还是原生；域外回退不等于数值/构建资格。没有在线 tuning、FP64、TF32 或新公式近似。未改仿射前向保留原生，自写输入伴随不调用 vendor 矩阵例程。

测试、构建、整网、训练继续暂停，本轮仅静态修复。
