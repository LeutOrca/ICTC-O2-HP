# Trilinear mixed second derivatives, 2026-11-17

The original committed core fails expanded FP64 validation at
C128/M6/full/E131072 in22 shared-weight gradients. The current numerical fix
uses explicit FP64 weight products/reduction and passes the added stress gate.
The historical timings below remain bound to the earlier FP32 core. Full240
training/SOTA qualification is still HOLD. See [the numerical boundary](hpx-training-numerical-boundary.md).

Enable `O2_HPX_FORWARD=preload O2_HPX_ADJOINT=replay` before import for the
qualified experimental path. This replaces the generic Torch reconstruction
inside double backward with exact forward/first-adjoint identities. Canonical
trainable weights remain live. The first-adjoint equations, coefficient source,
normalization, state keys and public layouts are unchanged. Current grouped
weight accumulation uses the precision correction described above.

For the trilinear product F(x,y,w), output cotangent q, and cotangents A/B/C
of its three first gradients, contract each selected cotangent by replacing
exactly one argument of F. A first adjoint of that replaced product contributes
to the other two arguments. Its forward value contributes to dq. The derivative
in the replaced argument is zero. Omitting an unused cotangent omits exactly its
zero contribution. All input/weight mixed terms and the output-cotangent term
are retained.

Numerical gate: independent FP64 equations, both inputs and every weight, two
seeds, scales 0.1/1, input-only/weight-only/all cotangent selections, and twenty
energy/force-loss + AdamW steps with parameter and optimizer-state comparison.
Four families pass: C32/M1/canonical/E37, C32/M3/full/E257, and
C128/M6/canonical or full/E37. Tolerance is fixed at 1e-5 + 1e-4*abs(reference).

The training measurement includes zeroing gradients, eager public forward,
create_graph=True forces from 0.5*sum(output^2), energy/force loss, complete
backward and AdamW. It measures operator force training, not a complete MLIP.
Thirty warmups and thirty samples run in each of five independent paired
allocations. Within each allocation, build and timing use separate processes
on the same GPU. Both original address64 and prior sched_combined are measured.

| C | M | parity | E | speedup vs fastest measured prior | candidate event ms |
|---|---|---|---|---|---|
| 32 | 1 | canonical | 8192 | 2.22x | 2.280 |
| 32 | 3 | canonical | 8192 | 7.03x | 2.861 |
| 32 | 1 | full | 131072 | 5.87x | 10.790 |
| 128 | 3 | full | 8192 | 9.02x | 8.116 |
| 128 | 6 | canonical | 131072 | both prior variants OOM | 165.092 |
| 128 | 6 | full | 131072 | both prior variants OOM | 305.536 |

The four finite comparisons give geometric means 5.3593x event and 5.3375x
synchronized wall time, with no >3% regression. OOM cases have no speedup ratio.
All six candidate cases finish on A100-SXM4-40GB, Torch2.7.1+cu128/Triton3.3.1.
Raw paired medians and memory are in [the receipt](hpx-training-screen-20261117.json).

This confirms a scoped improvement over the frozen prior implementation. The
240-case full-grid target and current external-SOTA comparison remain open.
cuEq0.13 is being qualified separately on H100/Torch2.11.

Source: frozen/hpx-replay-v1 in the parent o2-hpx-opt-20261117-r1 study.
Numerical job110503; prior sched_combined numerical job110579; paired array110578.
