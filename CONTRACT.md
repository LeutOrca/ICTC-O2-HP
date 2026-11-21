# Current GitHub O2-HP whole-network comparison — 2026-11-21

User selected whole-network E/F as the primary scope, MACE first with V3 retained,
and both public GitHub main branches. No operator timing is promoted as a whole-model speedup.
Frozen sources: Parity-LRX/ICTC-O2-HP 96815838f470652dce3d0e9d5cf7fe2978e949ca;
LeutOrca/ICTC-O2-HP 296e61f1c13b8ad5ff525cbe3d0bd63d4d6adaca.
The only tree difference is README.md; executable sources are identical. Both
provenances remain explicit. New local integration code is identified separately.

Keep the official MACE/V3 checkpoints, input values, weights and graph rules.
Every backend must actually call its named library and report substituted scope.
Existing local HPX kernels are a separately labelled comparator, never relabelled
as the public GitHub implementation. Unsupported mappings and failed FP32 gates
have no accepted timing/speedup/throughput value, and are never plotted as zero.

All numerical gates use FP32 references and FP32 error arithmetic, original
2e-5+2e-5|reference| component thresholds and relative L2 <=1e-5, including the
pre-existing force-RMS near-zero rule. No FP64 numerical reference, TF32 or AMP.
Upstream offline coefficient construction is source-preserved and outside timing.
Backend libraries may use their native vendor kernels; this is a comparison, not
a new self-written kernel replacement. Training remains paused.

Build, qualification, time, memory, trace and aggregation use Slurm compute
nodes. Start on one dedicated A100 GPU, at most three A100 nodes. Reserve an
initial bounded eight GPU hours for qualification and matched comparison,
inclusive of failures and builds; account against the existing project budget.
Five independent alternating role processes, matched GPU UUID per workload,
software, seed, graph settings, inputs and timing boundary. MACE Graph setting
must match every compared arm; V3 preserves the existing Graph-off boundary.
Report fixed topology and end-to-end separately, plus atoms/s, structures/s,
peak allocated and reserved. State whether input assembly/H2D are included.

Contemporary source checks include current EquivariantX/TACE and current
cuEquivariance. Only exact model conversions can enter the same-checkpoint
speedup denominator. Different architectures/equations have absolute costs only.

Plots: editable Matplotlib line charts passed through the Tavotto plugin,
150 mm, Times New Roman, frameless legend, vector PDF and PNG preview.
No publishing, remote push or messages to other people are authorized.
