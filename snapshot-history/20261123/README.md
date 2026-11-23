# O2 optimization snapshot history through 17a71d7

This branch indexes the original committed optimization histories through the
2026-11-22 safety candidate. The repository's `main` remains at
296e61f1c13b8ad5ff525cbe3d0bd63d4d6adaca.

| Branch | Original head | Scope |
| --- | --- | --- |
| codex/o2-library-snapshots-20261119 | 9787a3a6ca7fe67eeda3f21ec7c24d6634cbac03 | Installable library history: address64, preload, AOT, mixed derivatives, streamed graph, donated input, native launcher, CSR, initial V3 kernels |
| codex/o2-mace-snapshots-20261120 | fb17264d25f8d00b9cbde8cadfaf4038d5fb11b6 | MACE H100 qualification, deterministic reduction and TP slicing histories |
| codex/o2-v3-fp32-repair-20261121 | 4b170465047f560b548b1faf7fc5205d91f78579 | Custom Triton/CUDA, MACE/V3 measured candidates and full14 FP32 repair |
| codex/o2-github-whole-20261121 | c499cdd6de95eedc0545fc0d965da7005415d532 | GitHub-baseline whole-MACE comparison harness and scope records |
| codex/o2-safety-candidate-20261122 | 17a71d71e10775f6eee70aa70196948d4178bf43 | Angular campaign, full-parity repair, layout/state/dispatch guards and final safety candidate |

`COMMITS.tsv` contains the original SHAs, parents and subjects of every imported
commit not already reachable from the frozen GitHub main. `SOURCES.tsv` records
which immutable local source supplied each branch. Source repositories and
existing commit objects were not rewritten. The index commit keeps these heads
as additional parents so the full histories remain reachable from this branch;
its library files retain the frozen GitHub main version.

These are several independent source trees, not successive installable package
releases. Check out the appropriate branch for its source layout. No uncommitted
source changes, model checkpoints, ignored dependency directories, build caches
or new measurement results are added by this publication. Existing tracked
records and their original numerical/performance boundaries remain unchanged.
Use each snapshot's dependency seals and recovery instructions before execution.

The safety candidate is **UNTESTED**. Its code core is 33451c5; 17a71d7 seals the
documentation and recovery index. Publication does not grant numerical,
performance, second-derivative or whole-model qualification. Source updates,
compilation, tests, GPU jobs and deployment are not performed by this index.
