# Confirmed accurate-weight training increment

The numerical correction retains FP64 weight products, partial sums and
mixed-weight directional sums until the final FP32 gradient. Its initial
16-row32-channel/four-warp kernel loses time to register pressure. An8-row tile
and two warps reduce this pressure while keeping every reduction equation and
accumulation type. The launch strategy is fixed across positive E.

Five same-GPU paired allocations110955..110959 compare corrected baselinev2
against tunedv3 in separate build and timing processes. Each complete training
step includes gradient zeroing, forward, create_graph forces, energy/force loss,
full backward and AdamW. Thirty warmups and thirty measured steps are used per
process, with backend order counterbalanced. All six preregistered screening
cases complete. The pointwise paired-median geometric means are1.108500x event
and1.108144x synchronized wall, with no >3% regression.

Across the five independent paired allocation scores, the event95% paired-t
interval on log scores is[1.104250,1.112583]; wall[1.104018,1.112468]. This is a
stable incremental win over the corrected baseline, not the full240/current-SOTA
score and not an additional10.85% over the older strict-FP32 training result.

Four small-E families pass all first/mixed derivative selections and20 AdamW
states (110950). Full C128/M6/full/E131072 first and mixed derivatives pass both
seeds20261003/20261004 and scales0.1/1 (110951..110954) against independent,
row-streamed FP64 AD. CPU110961 audits source seals, unchanged reduction-function
ASTs, all numerical receipts and all five pairs. No threshold is relaxed.

Source core SHA256:
`f534db9597d5d1899e0ccf573a1822905a41dad2ac014a8a8bfdd47a141919c4`.
[Paired measurements](hpx-training-accurate-tiles-paired-20261117.json) and
[promotion audit](hpx-training-accurate-tiles-audit-20261117.json) retain the data.
Full240 training and current external-contender comparisons remain open.

Expanded representation-wide qualification110963 subsequently passes59 of60
small-E families. At C256/M6/full/E37, one of39424 weights after AdamW step0
exceeds the unchanged FP64-reference threshold (max absolute weight difference
0.0001045438068). First and mixed derivative selections at this point finish
before that optimizer-state failure. The screened timing increment remains
bound to its qualified cases. Full training acceptance is HOLD pending the
diagnosis, as recorded in [the full60 receipt](hpx-training-full60-boundary-20261117.json).
