#!/bin/bash
# Run only after all frozen E/F qualification jobs have passed.
set -euo pipefail
py=${HPX_PYTHON:?Specify the pinned compute-node Python}
implementation=${1:?Specify the frozen own-kernel implementation}
study="measurements/${SLURM_JOB_ID:?Use Slurm}-v3-$implementation"
mkdir -p "$study/results"
for suite in mptrj-b1 mptrj-b8; do
    "$py" -u qualify_v3.py --suite "$suite" --without-norm --local-only \
        --implementation "$implementation" --output "$study/results/qualification-$suite.json"
    "$py" -u qualify_v3.py --suite "$suite" --without-norm --local-only --start-block 5 \
        --output "$study/results/qualification-c591b16-$suite.json"
done
for suite in mptrj-b1 mptrj-b8; do
    qualification="$study/results/qualification-$suite.json"
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
done
