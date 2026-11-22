# Full-parity repair continuation

Baseline: 6f641cf736a97fa6835678c0eb123f5b2694e80d. Preserve its inputs, seeds,
weights, equations, full parity, FP32 component and relative-L2 tolerances,
512-row statistics windows, disabled TF32/AMP/CUDA Graph, and all timing and
memory boundaries. The inherited CONTRACT.md remains scientific provenance.

Priority: first diagnose and repair the reported11 qualification failures,
then fix high-order builds and complete five same-GPU repetitions, fresh
allocated/reserved memory, and angular-order throughput figures. No network
or training work. Replacement arithmetic must remain self-written Triton/CUDA.
Do not introduce FP64 model/reference/gating or vendor GEMM replacements.
Native unchanged affine layers and diagnostic replays may use their original
PyTorch implementation; diagnostics are never performance qualifications.

The receipt audit found3 of the reported11 are unexecuted VJPs mislabeled by
forward-only qualification receipts. Their matching build records timed out.
Correct totals are8 actual numerical failures and15 build timeouts, not11/12.
Classification correction changes no accepted measurement CSV or tolerance.

True failures are full-parity RadialUVU input VJPs: public-unfused L4/5/6,
Eqx-O2 L4/5/6/7/8. Only the radial hidden-input adjoint fails component bounds;
all relative-L2 values remain below1e-5. No correction to arithmetic is yet
accepted. Initial isolation replays exactly C64/H64/E16384, unit scale,
seed20261003, order4, full parity, the original offending reference region.

GBU2 H100 with torch2.11.0+cu128 remains the formal target. A100 is occupied
by another user's8-GPU allocation and is not used. Direct GBU2 SSH from both
login and CPU compute nodes, standard port22, and a trusted GBU relay all
failed on2026-11-22. Do not silently substitute RTX/CPU timing or publish
diagnostic output as H100 numerical acceptance. CPU audits remain allowed.

The initial H100 isolation allocation is bounded to one GPU and30 minutes,
at most0.5GPUh. No build-timeout campaign is submitted before the numerical
issue is isolated and repaired. The original project500GPUh cap is preserved;
all new failures/compilations count. No new GPU job has been submitted yet.

Tavotto plugin and standalone engine are upgraded to0.21.3. Current MCP process
requires reconnection to load the new plugin. Keep150mm, Times New Roman,
frameless legends and the existing Tavotto-only login plotting exception.
All Python, source/JSON audits, numerical work and initial plots remain Slurm.
