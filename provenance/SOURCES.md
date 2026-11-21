# Live source identities checked 2026-11-21

- Official O2-HP main: https://github.com/Parity-LRX/ICTC-O2-HP/commit/96815838f470652dce3d0e9d5cf7fe2978e949ca
- Fork O2-HP main: https://github.com/LeutOrca/ICTC-O2-HP/commit/296e61f1c13b8ad5ff525cbe3d0bd63d4d6adaca
- The two executable trees are identical. `upstream-diff.tsv` contains README.md only.
- Contemporary EquivariantX/TACE: https://github.com/xvzemin/tace/commit/c646c7000af7adb367d6bd8b7eb5ea74c08dc37c (2026-11-20, TACE 0.2.5).
- Contemporary cuEquivariance: 0.13.0 (2026-10-15), https://github.com/NVIDIA/cuEquivariance/releases and https://nvidia.github.io/cuEquivariance/changelog.html . The release changelog requires Torch >=2.11 for recent compiled ops. A pre-existing older runtime is not labelled 0.13.0.
- Local HPX comparator: unpublished c591b16, `runs/hpx-custom-kernels-20261120-r1/cohorts/c591b16`, source preserved separately. These kernels are not the public O2-HP main.

The public library offers an O(2) tensor product, UVU and RadialUVU, without a
MACE or V3 model converter. This run supplies a new, explicitly limited bridge
for the first MACE scalar TP. It does not claim to replace both MACE convolutions
or the V3 SO2Linear modules. Integer canonical O(2) blocks are used for the scalar
identity embedding. This is not a full-parity coverage test or half-integer
compilation. Native MACE parity and all original model parameters are preserved.

Historical V3 results remain in `runs/v3-so2-fp32-repair-20261121-r1` and use
Graph off on A100. They must not enter an H100 MACE speedup denominator. The
public O2-HP GitHub implementation is not substituted into that V3 snapshot.
