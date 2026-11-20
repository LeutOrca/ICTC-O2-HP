# 自写 kernel 快照恢复

本 run 的 `provenance/hpx-custom-kernels.bundle` 包含全部增量 Git 提交。用 `git clone <bundle> <new-directory>` 独立恢复，再按 `provenance/git-history.txt` 切换相应版本。

- 完整 SO2Linear 五配对通过的源码为 `d8aec37`。对应 `cohorts/d8aec37/results/PAIR_AUDIT.json` 和 `so2-large.json`，C128/L4/M2/160000 边，原生权重与输入哈希一致。
- MACE 的通过性能基线仍为 `78231f3`，位于上一轮 run。新增常量前缀 `24c2ddc` 通过 A100 B8 严格 E/F 和更新位置的 Graph 资格，速度与峰值仍以新的同卡配对为准。
- 后续首层 TP、局部 SO2、全矩阵分派和 Graph 试验都独立保存。Git HEAD 表示最新开发源码，不自动表示通过整网全部门槛。

数据与框架依赖沿用已封印的 MACE/V3 run，checkpoint 不重复打包。环境和路径在 `stage_a100.sbatch`、`stage_h100.sbatch` 中固定。MACE 基准依赖 `mace-ictc-o2-hp-gbu2-20261120-r1` 与上一轮的 `frozen/`，V3 依赖 `o2-hpx-equiformerv3-oam-20261119-r2` 和 r1 官方源码及 checkpoint。其身份见 `CONTRACT.md` 与 `REPORT.zh-CN.md`。

GPU 任务从 `cohorts/<commit>/` 执行，先核对其 `provenance/input.sha256`。新编辑不写回旧 cohort。全部 Python、NVRTC/Triton 编译、验收和聚合均经 Slurm，登录节点只提交和查看文本。

`provenance/SNAPSHOT_QA.json` 记录从 bundle 独立导出各 cohort 后的逐文件哈希检查。GPU 数值和性能证据保存在各自 results/ 中，源码恢复检查不替代 GPU 科学验收。
