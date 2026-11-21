# Angular-order operator comparison, 2026-11-21

User correction: primary plots are throughput versus angular order 1..8,
not whole-network throughput versus atom count. Include ICTP and Eqx-O2.
Whole-network work is frozen separately as an appendix; no more whole-model
jobs are authorized by this operator campaign.

Families: O2 UUU TensorProduct; O2 V=1 UVU and RadialUVU; SO2Linear; native
ICTP O3 weighted UVU. C=64 throughout. Fixed N=4096 for pointwise node TPs,
fixed E=16384 for edge TPs, UVU/RadialUVU and SO2Linear; radial hidden=64.
Both canonical and full parity for O2. SO2Linear Lmax=Mmax=1..8, C64->C64.
Native ICTP natural parity/even couplings, L<=3; test 4..8 constructor guards
and mark unsupported, without extrapolation or zero throughput.

Official O2-HP 96815838 and fork 296e61f1 have byte-identical executable
sources. Both source identities are listed; one measured implementation curve
is labelled official main (= fork), rather than manufacturing two algorithms.
Eqx-O2 c646c700 uses exactly mapped paths, weights and orthogonal coordinate
layouts. cuEq0.13 exact O2 descriptors are a same-equation contender for UUU.
ICTP and SO2Linear have different equations and storage widths; absolute costs
are separate panels. No cross-equation speedup claims.

Strict FP32 model, reference and error statistics. TF32/AMP/Graph disabled.
Keep the original operator component bound 1e-5+1e-4*abs(reference), also
require 2e-5+2e-5*abs(reference) and relative L2<=1e-5. No FP64 numerical
reference, relaxed gates, half-integer compilation, or precision-cohort pooling.
Upstream offline coefficient construction is preserved outside the timed region.
Freeze weights for inference. Measure forward and forward+one VJP for all
position-dependent inputs, including external radial/UVU weights. No training
or parameter-gradient claim. Two seeds, scaled and unit-scale numerical probes.

Layout/orthogonal-basis and weight conversion are preparation only; each
operator consumes its declared native API layout. Validation decodes outputs
and input adjoints. Full workload validation may stream a row-independent
FP32 reference to bound memory; no shared parameter gradients are omitted
from a claimed training measurement, because training is not measured.

Separate build/qualification/timing/memory processes on Slurm compute nodes.
Five independent paired timing processes per configuration, rotating role
order; 30 warmups/30 synchronized samples for each boundary. Report wall and
CUDA event ms, primary rows/s from wall time, both allocated/reserved peaks.
Memory uses fresh processes per role. Timing and memory may use separate
allocations; all arms within each phase use the same GPU and keep phase UUIDs
explicit. The corrected memory sweep requalifies on its own GPU and removes
an unused native-layout cotangent from preparation before resetting peaks.
Earlier memory records retaining that unused tensor are superseded, while
their unaffected timing records are preserved. Failed/OOM/unsupported points have no
accepted throughput and are gaps in plots. Same GPU for all compared arms
within each configured workload, source/input/weight hashes recorded.

Initial operator envelope: 8 GPUh inclusive of failures/builds. This joins the
existing 500 GPUh project accounting, not a reset. GBU2 H100 is authorized;
A100 remains unavailable until the other user's allocation ends.

Pilot re-estimate: the full 88-configuration grid and separate corrected
memory qualification require an inclusive 16 GPUh envelope. This follows the
authorized PLAN.md pilot re-estimation rule and remains inside the unchanged
500 GPUh project hard cap, verified in provenance/BUDGET-AUDIT.json. Preserve
failed points and baselines rather than dropping them to reduce accounting.
High-order forward and VJP builds may be isolated by role and phase with
bounded timeouts; a backward build timeout does not invalidate a separately
qualified inference-forward point. No numerical tolerance is changed.

The inclusive pilot envelope is revised to 20 GPUh to include a direct local
HPX/public comparison and a complete SO2 rerun after correcting a CUDA worker
context launch error. The unchanged 500 GPUh project cap is audited separately.
Local HPX source is 5f278ef, independent-output preload API; donated-input
inference is excluded from these paired first-input-VJP panels. A frozen-weight
adapter requests its existing input adjoints with output_mask=3, bypassing the
stock API's unused FP64 parameter-gradient computation. Thus this is explicitly
the input-only inference adapter, not a stock HPX training benchmark. All new
arms are requalified in FP32 and paired on their own common GPU allocation.

High-order supplemental gates may stream the same row-independent FP32
reference in4096-row chunks rather than512, to reduce repeated reference
launches. Every row and every input adjoint is still checked at the original
tolerances; no shared parameter gradients are claimed. The actual chunk size
is preserved in each seed/scale qualification receipt.
Error statistics remain in the original512-row windows even if reference
execution batches4096 rows. The earlier5992a12 pilot that grouped statistics
by4096 is superseded and excluded from formal timing/memory aggregation.

CPU-only module preload through a CUDA-clean forkserver is allowed after
the four-process/four-arm probe passes. Every timing and memory child still
starts with CUDA uninitialized and a fresh context/allocator, records its PID
and process_mode, and independently reconstructs the frozen workload. Import
and construction time remains outside steady-state operator timing.

The user requests a single complete submission before sleep, then morning
collection. Four dependent overnight allocations are pre-submitted, scanning
validated receipts and skipping complete paired time/memory cases. TP tails
are bounded to3h each, UVU/Radial tails to4h each; the inclusive envelope is
36 GPUh with all earlier failures charged and the unchanged500 GPUh cap.
There is no automatic GPU resubmission from inside these jobs. Whole-model
and training work remain paused. Stop manual polling after one startup audit.

Plots: 150 mm, Times New Roman, frameless legend, x=angular order1..8, separate
atoms/s and edges/s panels and forward/VJP boundaries. Use Tavotto to open,
edit, preflight, export vector PDF and PNG. The user explicitly authorized a
login-node exception for this round's Tavotto editing only; all scientific
computations, aggregation and initial plotting still run via Slurm.
