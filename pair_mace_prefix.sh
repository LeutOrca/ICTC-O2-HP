#!/bin/bash
set -euo pipefail
py=${HPX_PYTHON:?Specify the pinned compute-node Python}
case=${1:?Specify b8 or n512}
qualification="results/mace-prefix-$case.json"
"$py" -u qualify_mace_prefix.py "$case" "$qualification"
for stage in time memory; do
    for repeat in 0 1 2 3 4; do
        roles=(native base prefix)
        if (( repeat % 2 )); then roles=(prefix base native); fi
        for boundary in fixed e2e; do
            for role in "${roles[@]}"; do
                "$py" -u measure_mace_local.py "$stage" "$role" "$case" "$boundary" \
                    "results/mace-$stage-$case-$boundary-$role-$repeat.json" \
                    --qualification "$qualification" --repeat "$repeat"
            done
        done
    done
done
for boundary in fixed e2e; do
    for role in native base prefix; do
        "$py" -u measure_mace_local.py trace "$role" "$case" "$boundary" \
            "results/mace-trace-$case-$boundary-$role.json" --qualification "$qualification"
    done
done
