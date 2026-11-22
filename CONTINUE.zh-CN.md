# 当前候选入口：保持停测

从0704ee0创建独立分支，只授权修改候选代码、保存快照与更新报告。
不得运行继承的campaign/sbatch、diagnose、benchmark、aggregate或绘图入口。
不得提交CPU/GPU测试或编译作业，不恢复整网和训练。

本候选未通过新数值或性能验收，继承数据仅在`provenance/baseline-0704ee0/`作历史参考。
每个修改步骤保存Git commit/bundle。恢复验证需用户另行明确授权。

已保存布局 `d45cb87`、状态保护 `4771d0b`、调度域与原生回退 `33451c5`。最新封存 hash 见 `provenance/CANDIDATE-FROZEN-GIT.txt`。

后续获准后先在新 run 验入口合法/非法布局、weight/bias/TP 缓冲替换及原位修改、auto/native/required 路线和域外回退；再按原输入/权重/方程及 FP32 门验完整前向/一次输入 VJP。正式配对必须报告实际 route，原生回退不能冒充自写核。这里只列待办，不授权或自动启动任何步骤。完整候选源码包另带依赖，Git bundle 只保存跟踪源码和历史。
