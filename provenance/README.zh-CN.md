# 本候选 provenance 的身份

基线为 `BASELINE-COMMIT.txt` 中的 0704ee0，最新候选为 `CANDIDATE-FROZEN-GIT.txt`。

`baseline-0704ee0/` 保存继承数据表、SUMMARY、报告、快照表及旧恢复说明。此目录以外继承的 `BOUNDED-*`、`REPAIR-*`、`COLD-*`、`FINAL-*`、`TAVOTTO-*`、`BUDGET-*`、旧 `BASELINE-GIT.txt` 等也全部属于 0704ee0 或更早历史，路径/作业号保持原文，不是本候选验证。

本轮新证据只包括 `BASELINE-COMMIT.txt`、`STATIC-REVIEW.txt`、`UPSTREAM*.sha256`、`snapshots/01-layout.*`、`02-frozen-state.*`、`03-policy.*`、最终候选 bundle/hash、封包说明及 SHA。没有新数值或性能报告。

bundle 不含上游依赖或旧测量载荷，源码压缩包另带 `upstream/`。继承工具脚本是历史材料，不是自动执行授权。
