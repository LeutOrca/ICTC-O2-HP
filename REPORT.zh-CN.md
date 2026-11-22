# 从0704ee0分出的安全候选（未测试）

分支`codex/o2-safety-candidate-20261122`。原0704ee0保持不改。用户本轮只授权代码修复、快照和报告；不恢复编译、测试、整网或训练，不提交作业。

## 第一步：入口布局与形状合同

新增`candidate/contracts.py`，检查二维strided CUDA FP32张量、同设备、batch/表示/隐藏维度、逐边权重形状，以及TP计划缓冲的shape/dtype/布局和冻结参数。公共UVU/流式径向包装在custom Function外对输入调用`.contiguous()`，使复制保持正常autograd连接；连续输入直接复用，无dtype转换或数值处理。

当前代码仅经过源文件阅读和Git差异审阅，**尚未运行、编译或做数值/性能验证**。本步骤没有修改矩阵核、TP方程、输入/权重生成器、FP32参考或容差。

继承的489点数据、旧SUMMARY和原报告已移入`provenance/baseline-0704ee0/`，只表示基线结果，不是本候选的验收。参数一致性保护和调度适用域将在后续独立快照完成。
