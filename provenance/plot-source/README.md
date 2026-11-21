# Angular-order operator throughput

Primary x axis: maximum angular order1..8. Fixed C64, N4096 (node TPs), E16384
(edge operators), radial hidden64. O2 uses Mmax; SO2Linear uses Lmax=Mmax.
Separate canonical/full parity panels, forward / forward+input-VJP rows.
Primary throughput uses synchronized wall time. CUDA-event values are in CSV.
No whole-network E/F or neighbor construction enters these timings.
Inputs are reproducible synthetic features, with matched frozen random weights
within each angular point. These are not official-checkpoint whole-model rates.

Official96815838 and fork296e61f1 executable sources are byte-identical and
share the clearly labelled curve. Eqx-O2 c646c700 and exact cuEq0.13 O2
descriptors preserve paths, weights and orthogonal basis mappings. Kernels,
inputs, references and error statistics are FP32; TF32/AMP/Graph are off.

ICTP f40592a is an O3 even/weighted-UVU operator with Cartesian and path-expanded
storage. Its grey curve is a different-equation cost reference, not a speedup
denominator. The official implementation supports only1..3;4..8 are marked
unsupported in status.csv and are never interpolated or assigned zero rates.
SO2Linear is another separate equation, with native V3 source and self-written
CUDA/Triton replacements measured against its own unchanged weights.
The SO2 replacements are the generic c591b16 operator family plus a driver
context fix, not the later efb5a27 whole-checkpoint numerical specialization.
Local O2 HPX uses5f278ef, fresh output and an FP32 input-only adjoint adapter.

Missing/failed/OOM points are gaps. Five independent paired timing processes
are the statistical units. The editable scripts, input CSVs, vector PDFs,
PNGs and Tavotto override/proof files are kept together. Figure width150mm,
Times New Roman, frameless legends. Memory source/phase UUIDs and corrections
are explicit in the report and CSV and do not alter timing data.
