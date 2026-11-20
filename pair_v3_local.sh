#!/bin/bash
set -euo pipefail
py=${HPX_PYTHON:?Specify the pinned compute-node Python}
suite=${1:?Specify mptrj-b1 or mptrj-b8}
start=${2:?Specify the first replaced block}
study="measurements/${SLURM_JOB_ID:?Use Slurm}-v3-block${start}-$suite"
mkdir -p "$study/results"
qualification="$study/results/qualification.json"
"$py" -u qualify_v3.py --suite "$suite" --output "$qualification" --without-norm --local-only --start-block "$start"
for stage in time memory trace; do
    repeats=(0 1 2 3 4)
    if test "$stage" = trace; then repeats=(0); fi
    for repeat in "${repeats[@]}"; do
        roles=(native base own)
        if (( repeat % 2 )); then roles=(own base native); fi
        for boundary in fixed e2e; do
            for role in "${roles[@]}"; do
                "$py" -u measure_v3.py "$stage" "$role" "$suite" "$boundary" \
                    "$study/results/v3-$stage-$suite-$boundary-$role-$repeat.json" \
                    --qualification "$qualification" --repeat "$repeat"
            done
        done
    done
done
"$py" -u audit_mace_variants.py "$PWD/$study" "$suite" own --model v3
