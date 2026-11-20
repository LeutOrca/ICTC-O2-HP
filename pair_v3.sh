#!/bin/bash
set -euo pipefail
py=/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python
suite=${1:?Specify mptrj-b1 or mptrj-b8}
qualification=${2:?Specify the immutable qualified E/F receipt}
qualified_root=$(dirname "$(dirname "$qualification")")
for file in matrix_kernels.py cuda_matrix.py csrc/matrix.cu frozen_so2.py so2_layer.py merge_norm.py v3_native.py qualify_v3.py; do
    cmp "$file" "$qualified_root/$file"
done
for stage in time memory; do
    for repeat in 0 1 2 3 4; do
        roles=(native own)
        if (( repeat % 2 )); then roles=(own native); fi
        for boundary in fixed e2e; do
            for role in "${roles[@]}"; do
                srun "$py" -u measure_v3.py "$stage" "$role" "$suite" "$boundary" \
                    "results/v3-$stage-$suite-$boundary-$role-$repeat.json" --repeat "$repeat" \
                    --qualification "$qualification"
            done
        done
    done
done
for boundary in fixed e2e; do
    for role in native own; do
        srun "$py" -u measure_v3.py trace "$role" "$suite" "$boundary" \
            "results/v3-trace-$suite-$boundary-$role.json" --qualification "$qualification"
    done
done
