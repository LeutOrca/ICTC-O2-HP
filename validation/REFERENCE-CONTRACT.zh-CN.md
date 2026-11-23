# 新增 full VJP/二阶角色的参考合同

本轮仍逐字节保留 benchmark.py 的原 FP32 error 门：relative L2 <= 1e-5，并同时满足原两个逐分量不等式。输入、权重、seed/scale、原512行窗口不改。

历史 radial 只测冻结参数的 x/y/hidden 输入 VJP。新增 affine weight/bias full VJP 与二阶角色在首次全负载测试前定义参考：每512边用独立 PyTorch ReferenceUvuTensorProduct 和原 F.linear 的自动微分计算；所有边对共享参数梯度的贡献按页顺序用FP32加完，再按原误差门比较完整共享参数梯度。不能以仅检查一页、删权重梯度或更宽误差代替。

这一扩展不追认旧输入-only回执为full VJP，也不改变已有 forward/x/y/hidden 的512行原生参考边界。整网仍须与完整原生模型的原FP32门独立比较，算子分页参考通过不代替整网训练梯度/轨迹通过。
