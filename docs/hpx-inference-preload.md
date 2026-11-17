# O2 forward input reuse, 2026-11-17

Set `O2_HPX_FORWARD=preload` before importing the library to enable this
experimental inference path. It preloads each input representation block per
edge/channel, reuses it across the original ordered paths, and stores output
accumulators after all contributions. Coefficients, precision, weights, bases,
checkpoint keys and public shapes are unchanged. Training adjoints are unchanged.
The selected forward implementation is part of the registered operator namespace
to prevent an older Inductor lowering being reused for the new implementation.

Confirmed scope: A100-SXM4-40GB, Torch 2.7.1+cu128, Triton 3.3.1, FP32,
TF32/AMP/CUDA Graphs disabled. Twenty-four screen cases cover C=32/128,
M=1/3/6, canonical/full, E=8192/131072. Five independent processes per
implementation, thirty warmups and thirty synchronized samples per process.
Paired geometric mean speedup versus the frozen address64 baseline is 1.4321
for CUDA-event region time and 1.4118 for synchronized wall time. No point has
more than 3% regression. Full per-point process medians and speedups are in
[the receipt](hpx-preload-screen-20261117.json).

Qualification checks every actual-size compiled output against the independent
frozen eager baseline, and 257-row independent FP64 equations at both seeds and
scales 0.1/1, including rotation/reflection actions. Generated Inductor code is
sealed per case. Earlier separate-process checks cover both input gradients
and all path-weight gradients. Earlier batch timing protocols are excluded
because they allowed compiled implementations to be reused across contenders.

This is a screened inference improvement, not a 240-point or new-SOTA claim.
The user target is >=2x over the strongest qualified same-equation contender
on the full preregistered grid, independently for inference and complete force
training. Full-grid qualification and current cuEq 0.13 comparison remain open.

Study provenance: `runs/o2-hpx-opt-20261117-r1` in the parent benchmark workspace.
Frozen source: `frozen/preload-safe-v1`. Numerical/generated-code job:110488.
Timing array:110481_0..9 (even baseline, odd candidate, five paired processes).
Summary:110495. The original source checkout is preserved independently.
