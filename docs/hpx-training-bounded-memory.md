# Bounded replay memory on40GB A100

Mixed replay now adds independent contributions into its own output buffers
when no higher derivative graph is requested. Higher derivative construction
retains out-of-place additions. A consumed forward temporary is released before
the next direction. CPU111100 normalizes these storage statements and proves
all other kernel/math ASTs identical to the corrected8-row baseline.

The caller protocol is `operator_training_live_v6`: after constructing the same
energy/force loss, release Python references to output, row_energy and force
before backward. Autograd retains every tensor it still needs. This is applied
to every backend. The complete zero_grad/forward/create_graph-force/loss/
backward/AdamW timing boundary and equations are unchanged. Historical h4 and
new v6 measurements are separate cohorts.

Two-seed/five-family20-step checks give1600 bitwise comparisons for the caller
change and1600 for the final replay source. Losses, both input gradients, weight
gradients, parameters and AdamW states agree exactly with their prior source.

At C256/M6/full/E131072 on A100-SXM4-40GB, use the explicit16-row capacity
profile before importing the package:

```bash
export O2_HPX_FORWARD=preload O2_HPX_ADJOINT=replay
export O2_RUNTIME_EDGE_TILE=16 O2_OPT_VARIANT=bounded16_a10040
```

Five independent allocations111062/111066/111067/111068/111069 each complete
30 warmups and30 measured complete steps with finite parameters, gradients and
optimizer states. Every allocation peaks at38388584960 allocated bytes,
about35.75GiB. No batch/edge split or derivative term is removed.

Full-E first and mixed derivatives pass both seeds20261003/20261004 and both
scales0.1/1 against independent FP64 AD (111060/111063/111064/111065). Each has
6146 complete row/weight checks. Only the reference uses128-row chunks;
candidate calls use all131072 rows. First-gradient matrices are staged to CPU
and their GradientEdges seed the same VJP to bound validation memory.

Five same-GPU six-point pairs111074/111076..111079 use the same v6 caller and
8-row configuration for both sources. Event/wall speed ratios are0.995624x/
0.995753x; no point regresses by >3%. This snapshot confirms memory support,
with an approximately0.44% mean latency cost. It is not a latency promotion.
The [independent audit](hpx-training-bounded-audit-20261117.json) and
[paired data](hpx-training-bounded-paired-20261117.json) preserve the scope.

FP32 interfaces and explicit FP64 weight accumulation are unchanged. The
original FP64 AdamW-trajectory failure at C256/M6/full/E37 remains HOLD.
Current Torch2.11 tests reproduce an optimizer-state failure in all four FP32
entrants. Higher forward/input-adjoint accumulation precision reduces the
failure but does not close it (unpromoted experiment111080). No threshold is
relaxed. Full240 training/current-SOTA2x acceptance remains open.
