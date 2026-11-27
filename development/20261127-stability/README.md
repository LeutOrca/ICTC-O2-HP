# L1–L8 stability repair from the personal fork

This series starts from `a1dc6c9`, with operator sources corresponding to `5764f66`. It preserves the original loss, precision, numerical gates, active parameters and continuous optimizer/EMA/scheduler/RNG state. GPU work uses one A100 serially, with independent CPU-AMD audits.

`allocator-v7a` removes the local `coverage.py` import collision from the new control directory and uses `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on both backends. Native gradients that the original backward would clear before use are released before diagnostics. No training state is restored inside a trajectory.

With the old allocator policy, new job 120842 preserved an L8 second-mixed-diagnostic OOM even after old-gradient release. With expandable segments, native, native-repeat and candidate job 120845 captured all three steps. CPU-AMD audit 120846 passed L8 E/F, joint VJP, actual force-loss mixed derivatives, native updates and three-step comparisons under the original thresholds. This is an L8 three-step result, not a 20-step stability claim.

The remaining L7 error was localized to one four-row angular tail layout. The earlier plan search stopped at 32 partitions and used an in-gate approximation. Probe 120853 isolated all observed angular differences to that tail. Search 120857 found a 54-partition single-chain FP32 plan with zero differences for both preserved actual mixed-derivative and training operand sets. Full independent training validation of the revised plan is pending.

The complete L1–L8 matrix will be rechecked after the repair, followed by 20 continuous training steps per order and three fresh-process repeated-inference checks. These are bounded validation windows, not an indefinite trajectory guarantee. Historical failures and all earlier source tags remain unchanged.
