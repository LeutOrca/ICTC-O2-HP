#!/bin/bash
# A rejection probe; five-pair conclusions require pair_v3_repair.sh.
set -euo pipefail
py=${HPX_PYTHON:?Specify the pinned compute-node Python}
implementation=${1:?Specify implementation}
suite=${2:-mptrj-b8}
study="measurements/${SLURM_JOB_ID:?Use Slurm}-probe-$implementation-$suite"
mkdir -p "$study/results"
for validation in mptrj-b1 mptrj-b8; do
    "$py" -u qualify_v3.py --suite "$validation" --without-norm --local-only \
        --implementation "$implementation" --output "$study/results/qualification-$validation.json"
done
for stage in time memory trace; do
    for boundary in fixed e2e; do
        for role in native base own; do
            "$py" -u measure_v3.py "$stage" "$role" "$suite" "$boundary" \
                "$study/results/v3-$stage-$suite-$boundary-$role-0.json" \
                --qualification "$study/results/qualification-$suite.json"
        done
    done
done
