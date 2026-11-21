# V3 FP32 修复快照恢复

当前修复位于独立 Git 分支 `codex/v3-so2-fp32-repair`，从 c591b16 开始。`git log --oneline` 查看本轮增量，`cohorts/<commit>/` 为每一步实际执行源码，结果保存在相应 results/，作业日志在 logs/。不要修改已提交的 cohort。

本轮合同为 REPAIR_CONTRACT.md；继承的 CONTRACT.md 仅保存前轮上下文。全部新误差统计 FP32，原阈值不变。分段修复版与融合版的全 14 个替换均已通过冻结 B1/B8 的完整 E/F 门；Git HEAD 还包含诊断/报告工具，具体通过版本及性能须按报告中每个独立快照识别。

官方依赖保持 sibling run：`../o2-hpx-equiformerv3-oam-20261119-r2` 的 whole_network.py、输入和 checkpoint，以及 `../o2-hpx-equiformerv3-oam-20261119-r1/upstream/equiformer_v3`。检查点 SHA256 见本轮合同，不重复打包权重。

作业从 cohort 目录通过 stage_a100.sbatch 提交，HPX_STUDY_ROOT 指向本修复 run。所有 Python、NVRTC/Triton 编译、数值验收和 JSON 汇总均经 Slurm；登录节点只做文件、Git 和队列控制。原 c591b16 run 与全部历史结果保持原位。

完整五配对已测核心为 `efb5a27`，实施名 `fused-compact`。最新报告/审计源码的完整提交保存在 `provenance/FINAL_SOURCE_COMMIT.txt`。`provenance/v3-fp32-repair-final.bundle` 保存全部 Git 历史，`provenance/v3-fp32-repair-final-source.tar.gz` 保存最终已跟踪文件；源码包不重复嵌入 checkpoint 或原始 trace。

`provenance/SNAPSHOT_QA.json` 核验 bundle 独立恢复导出、每个 cohort 的文件哈希及最终源码包独立解包的成员集合和 SHA256；`provenance/DELIVERY_QA.json` 另核对官方 checkpoint、FP32 数值资格、完整三臂五配对/trace、基线核不变及当前核与已测源码一致。快照恢复通过不等于速度目标通过，目标状态见 SUMMARY.json 的 acceptance。

原始数值/性能/trace 证据保留在本 run 的各 `cohorts/<source>/measurements/<job>/results/`。完整可离线阅读的表格为 PAIRED_RESULTS.zh-CN.md。`provenance/git-history.txt` 列出本轮所有增量提交，历史失败和未晋级方案均保留。
