# Full-edge training numerical boundary

The fixed threshold remains `1e-5 + 1e-4*abs(FP64 reference)`. Expanded tests use
the entire C128/M6/full/E131072 candidate call, then independently stream1024-row
FP64 reference chunks. Weight derivatives are summed over every row in FP64
before comparison. Reference chunking only bounds validation memory.

Both committed-core snapshot `aot-library-v2` (job110934) and selective replay
`hpx-selective-replay-v1` (110932) fail22 of19712 first weight derivatives at
seed20261003/unit scale. Maximum absolute error is0.0003817369561; the maximum
error/threshold ratio is10.15976595. Their grouped weight reduction and final
reduction function ASTs are identical. This identifies a shared pre-existing
reduction boundary, rather than qualifying the new candidate from its timing.

Selective replay completes five same-GPU six-case paired allocations. It gives
1.025679x event/1.025652x wall geometric mean versus the committed core, without
a >3% regression. It remains unpromoted because the expanded numerical gate fails.

Accuracy fix `hpx-accurate-weight-v2`, core SHA256
`ea0e39518ef076a6d967364ab7a392f6c5665c59f37a192eba3d4a829fda3ba7`,
retains FP64 grouped weight products/partials and mixed-weight direction sums,
then returns FP32 gradients. Input/output/parameter storage remains FP32. This
is an explicitly different accumulation-precision experiment, not a strict-FP32
kernel result. Its source is now applied, and its original patch is retained in
`experiments/training-accuracy-v2.patch` as a checkpoint of the correction.

It passes the same full-E first and mixed second derivative checks for both
seeds20261003/20261004 and scales0.1/1 (110942/110943/110946/110947). Four small-E
families also pass all derivative selections and20 AdamW states (110941).
CPU110948 audits these receipts and unchanged tolerances. The prior FP64
candidatev1 fails two mixed-weight entries at the second seed (110938); onlyv2
retains wide values until all directional contributions have been added.

The correction has performance costs in some cases. It is a numerical fix,
not a performance promotion. No source/intermediate precision change is hidden.
Neither the full240 training goal nor a current-SOTA claim is accepted.
The subsequent [confirmed tile increment](hpx-training-accurate-tiles.md) retains
this correction and reports its own paired timings and numerical gates.
The compact receipt is [hpx-training-numerical-boundary-20261117.json](hpx-training-numerical-boundary-20261117.json).
