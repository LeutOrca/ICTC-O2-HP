# 自写 kernel 快照恢复

本 run 的 `provenance/hpx-custom-kernels.bundle` 包含全部增量 Git 提交。用 `git clone <bundle> <new-directory>` 独立恢复，再按 `provenance/git-history.txt` 切换相应版本。

- 完整 SO2Linear 五配对通过的源码为 `d8aec37`。对应 `cohorts/d8aec37/results/PAIR_AUDIT.json` 和 `so2-large.json`，C128/L4/M2/160000 边，原生权重与输入哈希一致。
- MACE 冻结性能基线为 `78231f3`。首层径向伴随修复为 `8c2d4d3`，TP/CSR 融合为 `3924ed2`，常量物种投影为 `b0d0c01`，三项组合的 H100 兼容版本为 `40af2f8`。`fb3dc99` 另复用常量组分字段，每次端到端调用仍重建原生邻居。
- `64dbeb1` 修复资格脚本，使位置扰动导致邻居数变化时原生和候选对称重新捕获 Graph。稳态测量仍是已冻结几何上的原生邻居重建，不把重新捕获成本隐藏到形状变化的在线延迟中。当前回放 API 在形状变化时要求显式重新捕获。
- H100 原生/78231f3/候选的完整五配对按 `h100-results/<source>/measurements/<job>-<variant>-<case>/results/` 保存。A100 同样按 `cohorts/<source>/measurements/...` 隔离。完整路径、原始指标和资格证据见 `SUMMARY.json`，不要跨 GPU、软件或作业拼接缺片。
- A100 `152c619` 的两个作业曾覆盖同名原生臂文件，分别从各自 Slurm stdout 的原始 JSON 行恢复。只使用 `measurements/114664-projection-b8/` 与 `measurements/114665-receiver-b8/`，对应 `LOG_RECOVERY.json` 记录日志 SHA 和行号，CPU114671 审计通过。
- V3 `9333e8f` 只替换 blocks.5/6 的四个 SO2Linear，保留原生 norm、前五层 SO2 和其他原生层。B1/B8 两种子、两图边界严格 E/F 通过；原生/9787a3a/局部候选另做五配对。全 14 个 SO2 的新核尚未通过 B8，不能将后两层资格外推。
- 全矩阵替换、单独首层 TP、全 norm、Graph 试验及失败回执均保留。Git HEAD 表示最新源码和审计工具，不自动表示通过所有整网目标。各快照的训练和二阶梯度均未放行。

数据与框架依赖沿用已封印的 MACE/V3 run，checkpoint 不重复打包。环境和路径在 `stage_a100.sbatch`、`stage_h100.sbatch` 中固定。MACE 基准依赖 `mace-ictc-o2-hp-gbu2-20261120-r1` 与上一轮的 `frozen/`，V3 依赖 `o2-hpx-equiformerv3-oam-20261119-r2` 和 r1 官方源码及 checkpoint。其身份见 `CONTRACT.md` 与 `REPORT.zh-CN.md`。

GPU 任务从 `cohorts/<commit>/` 执行，先核对其 `provenance/input.sha256`。新编辑不写回旧 cohort。全部 Python、NVRTC/Triton 编译、验收和聚合均经 Slurm，登录节点只提交和查看文本。

`provenance/SNAPSHOT_QA.json` 记录从 bundle 独立导出各 cohort 后的逐文件哈希检查。GPU 数值和性能证据保存在各自 results/ 中，源码恢复检查不替代 GPU 科学验收。
