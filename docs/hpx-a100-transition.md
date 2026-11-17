# A100 transition boundary

Existing A100 results retain their scientific meaning: the source versions,
precision, timing boundaries and five same-GPU paired allocations are preserved.
The confirmed full-grid forward preload result is1.533332x event/1.512255x wall
over entry native. The dynamic AOT runner has a separate24-point paired result.
Accurate weight tiles have a six-point paired result versus their corrected
FP64-accumulation baseline. These increments were measured on A100 already.

The current training correction does not change the11 forward-function ASTs or
the AOT loader relative to the qualified deployment source. CPU110998 audits
this equality, source hashes, paired scope and incomplete gates in the
[transition receipt](hpx-a100-transition-audit-20261117.json).

H100 results are separate. No H100 job is running; pending forward jobs are
reversibly held. The actual6.631389 H100 GPU hours include failed and timed-out
allocations. No newly prepared private-source package was uploaded.

The isolated A100 Torch2.11.0+cu128/Triton3.6/cuEq0.13 cohort must rebuild AOT and
external kernels and requalify numerical equivalence. The original Torch2.7 o2
environment remains unchanged. New and historical results are not pooled.

Complete A100 training acceptance remains HOLD: the40GB largest-point capacity
probe is pending, and expanded60-family qualification exposes one AdamW weight
state failure at C256/M6/full/E37. Neither switching hardware nor preserving the
screened gains closes this failure. No tolerance, equation, derivative term or
complete-step boundary is changed. Full240/current-SOTA2x/no3% remains open.
