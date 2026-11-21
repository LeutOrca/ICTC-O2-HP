# V3 FP32 修复快照恢复

当前修复位于独立 Git 分支 `codex/v3-so2-fp32-repair`，从 c591b16 开始。`git log --oneline` 查看本轮增量，`cohorts/<commit>/` 为每一步实际执行源码，结果保存在相应 results/，作业日志在 logs/。不要修改已提交的 cohort。

本轮合同为 REPAIR_CONTRACT.md；继承的 CONTRACT.md 仅保存前轮上下文。当前仍在数值修复，Git HEAD 不代表全 14 个替换已通过，未放行正式性能测量。全部新误差统计 FP32，原阈值不变。

官方依赖保持 sibling run：`../o2-hpx-equiformerv3-oam-20261119-r2` 的 whole_network.py、输入和 checkpoint，以及 `../o2-hpx-equiformerv3-oam-20261119-r1/upstream/equiformer_v3`。检查点 SHA256 见本轮合同，不重复打包权重。

作业从 cohort 目录通过 stage_a100.sbatch 提交，HPX_STUDY_ROOT 指向本修复 run。所有 Python、NVRTC/Triton 编译、数值验收和 JSON 汇总均经 Slurm；登录节点只做文件、Git 和队列控制。原 c591b16 run 与全部历史结果保持原位。

最终源码包、Git bundle 及独立解包验证回执将在本轮诊断收尾后记录于 provenance/，旧前轮路径不作为本轮已验证交付。
