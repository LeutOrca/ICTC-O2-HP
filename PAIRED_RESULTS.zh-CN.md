# V3 全 14 层修复：完整五轮同卡结果

三臂分别为官方原生、c591b16 后四个 SO2 替换、全 14 个 SO2 候选。每轮在同一卡交替，阶段和各臂独立进程。
固定拓扑不含邻居重建；e2e 从驻 GPU 的冻结输入开始，含官方构图和 E/F，不含文件载入/H2D。
CUDA Graph 三臂均关闭，FP32、TF32/AMP 关闭，原数值阈值保持。性能中位数与数值 FP32 门控分开。

## native-block-tree

| 输入/边界 | 原生 / c591b16 / 候选 ms | 对原生 / 对 c591b16 | 候选 atoms/s | 候选 structures/s |
|---|---:|---:|---:|---:|
| mptrj-b1/fixed | 117.5331 / 112.3330 / 127.6623 | 0.9205× / 0.8798× | 219.33 | 7.833 |
| mptrj-b1/e2e | 119.7761 / 114.6486 / 129.9396 | 0.9219× / 0.8819× | 215.48 | 7.696 |
| mptrj-b8/fixed | 208.6320 / 193.1309 / 235.3036 | 0.8867× / 0.8222× | 288.99 | 33.999 |
| mptrj-b8/e2e | 225.1319 / 210.1753 / 251.9552 | 0.8947× / 0.8348× | 269.89 | 31.752 |

峰值为 PyTorch allocated/reserved，单位 MiB，包含各臂缓存权重。

| 输入/边界 | 原生 allocated / reserved | c591b16 allocated / reserved | 候选 allocated / reserved | 对原生减少 allocated / reserved | 对 c591b16 减少 allocated / reserved |
|---|---:|---:|---:|---:|---:|
| mptrj-b1/fixed | 1917.57 / 2020.00 | 1733.10 / 1830.00 | 1706.25 / 1762.00 | 11.02% / 12.77% | 1.55% / 3.72% |
| mptrj-b1/e2e | 1917.61 / 2020.00 | 1733.14 / 1830.00 | 1706.29 / 1762.00 | 11.02% / 12.77% | 1.55% / 3.72% |
| mptrj-b8/fixed | 4575.43 / 4894.00 | 4020.12 / 4306.00 | 3996.45 / 4110.00 | 12.65% / 16.02% | 0.59% / 4.55% |
| mptrj-b8/e2e | 4575.52 / 4894.00 | 4020.21 / 4306.00 | 3996.55 / 4110.00 | 12.65% / 16.02% | 0.59% / 4.55% |

同卡绑定：

- mptrj-b1: source `22825fcbdfbe9dcda82226f8c1ec90a23c90ab06`，GPU `f31def7c-1b59-13f6-e074-5a31665c9a29`，[原始审计](cohorts/22825fc/measurements/115087-v3-native-block-tree-mptrj-b1/results/V3_LOCAL_PAIR_AUDIT-mptrj-b1-own.json)。
- mptrj-b8: source `22825fcbdfbe9dcda82226f8c1ec90a23c90ab06`，GPU `5dce2724-2ce2-55ed-a1e1-c1783fbee235`，[原始审计](cohorts/22825fc/measurements/115088-v3-native-block-tree-mptrj-b8/results/V3_LOCAL_PAIR_AUDIT-mptrj-b8-own.json)。

## fused-block

| 输入/边界 | 原生 / c591b16 / 候选 ms | 对原生 / 对 c591b16 | 候选 atoms/s | 候选 structures/s |
|---|---:|---:|---:|---:|
| mptrj-b1/fixed | 119.6136 / 114.2118 / 111.1396 | 1.0787× / 1.0276× | 251.94 | 8.998 |
| mptrj-b1/e2e | 121.3778 / 116.2592 / 112.7387 | 1.0802× / 1.0308× | 248.36 | 8.870 |
| mptrj-b8/fixed | 211.0822 / 195.8404 / 186.2869 | 1.1332× / 1.0546× | 365.03 | 42.945 |
| mptrj-b8/e2e | 228.4565 / 213.0545 / 203.2180 | 1.1241× / 1.0471× | 334.62 | 39.367 |

峰值为 PyTorch allocated/reserved，单位 MiB，包含各臂缓存权重。

| 输入/边界 | 原生 allocated / reserved | c591b16 allocated / reserved | 候选 allocated / reserved | 对原生减少 allocated / reserved | 对 c591b16 减少 allocated / reserved |
|---|---:|---:|---:|---:|---:|
| mptrj-b1/fixed | 1917.57 / 2020.00 | 1733.10 / 1830.00 | 1732.88 / 1852.00 | 9.63% / 8.32% | 0.01% / -1.20% |
| mptrj-b1/e2e | 1917.61 / 2020.00 | 1733.14 / 1830.00 | 1732.92 / 1852.00 | 9.63% / 8.32% | 0.01% / -1.20% |
| mptrj-b8/fixed | 4575.43 / 4894.00 | 4020.12 / 4306.00 | 4018.37 / 4306.00 | 12.18% / 12.01% | 0.04% / 0.00% |
| mptrj-b8/e2e | 4575.52 / 4894.00 | 4020.21 / 4306.00 | 4018.46 / 4306.00 | 12.17% / 12.01% | 0.04% / 0.00% |

同卡绑定：

- mptrj-b1: source `a97721560565ff9f3f22bf23f5d8b0340f7bd8a6`，GPU `2111e3ca-aefb-2b81-0b3b-95cc162136d8`，[原始审计](cohorts/a977215/measurements/115094-v3-fused-block-mptrj-b1/results/V3_LOCAL_PAIR_AUDIT-mptrj-b1-own.json)。
- mptrj-b8: source `a97721560565ff9f3f22bf23f5d8b0340f7bd8a6`，GPU `4da5e472-bcc6-4df8-e15f-bcbbed9f94af`，[原始审计](cohorts/a977215/measurements/115095-v3-fused-block-mptrj-b8/results/V3_LOCAL_PAIR_AUDIT-mptrj-b8-own.json)。

## fused-compact

| 输入/边界 | 原生 / c591b16 / 候选 ms | 对原生 / 对 c591b16 | 候选 atoms/s | 候选 structures/s |
|---|---:|---:|---:|---:|
| mptrj-b1/fixed | 117.5819 / 112.1205 / 108.2523 | 1.0862× / 1.0308× | 258.65 | 9.238 |
| mptrj-b1/e2e | 119.5108 / 114.4174 / 110.5157 | 1.0814× / 1.0338× | 253.36 | 9.048 |
| mptrj-b8/fixed | 210.3243 / 194.8181 / 184.8576 | 1.1376× / 1.0540× | 367.85 | 43.277 |
| mptrj-b8/e2e | 228.3384 / 213.1755 / 202.6126 | 1.1270× / 1.0531× | 335.62 | 39.484 |

峰值为 PyTorch allocated/reserved，单位 MiB，包含各臂缓存权重。

| 输入/边界 | 原生 allocated / reserved | c591b16 allocated / reserved | 候选 allocated / reserved | 对原生减少 allocated / reserved | 对 c591b16 减少 allocated / reserved |
|---|---:|---:|---:|---:|---:|
| mptrj-b1/fixed | 1917.57 / 2020.00 | 1733.10 / 1830.00 | 1711.39 / 1826.00 | 10.75% / 9.60% | 1.25% / 0.22% |
| mptrj-b1/e2e | 1917.61 / 2020.00 | 1733.14 / 1830.00 | 1711.43 / 1826.00 | 10.75% / 9.60% | 1.25% / 0.22% |
| mptrj-b8/fixed | 4575.43 / 4894.00 | 4020.12 / 4306.00 | 4018.37 / 4306.00 | 12.18% / 12.01% | 0.04% / 0.00% |
| mptrj-b8/e2e | 4575.52 / 4894.00 | 4020.21 / 4306.00 | 4018.46 / 4306.00 | 12.17% / 12.01% | 0.04% / 0.00% |

同卡绑定：

- mptrj-b1: source `efb5a271c58f76fc2f2eaaf49c78fa73f31f2985`，GPU `5dce2724-2ce2-55ed-a1e1-c1783fbee235`，[原始审计](cohorts/efb5a27/measurements/115102-v3-fused-compact-mptrj-b1/results/V3_LOCAL_PAIR_AUDIT-mptrj-b1-own.json)。
- mptrj-b8: source `efb5a271c58f76fc2f2eaaf49c78fa73f31f2985`，GPU `5ad49e16-2802-236e-933b-d315984dbf38`，[原始审计](cohorts/efb5a27/measurements/115103-v3-fused-compact-mptrj-b8/results/V3_LOCAL_PAIR_AUDIT-mptrj-b8-own.json)。

