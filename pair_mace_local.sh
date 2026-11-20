#!/bin/bash
set -euo pipefail
py=${HPX_PYTHON:?Specify the pinned compute-node Python}
variant=${1:?Specify the isolated candidate}
case=${2:?Specify the frozen case}
study="measurements/${SLURM_JOB_ID:?Use a scheduler-managed compute allocation}-$variant-$case"
mkdir -p "$study/results"
qualification="$study/results/qualification.json"
"$py" -u qualify_mace_variant.py "$variant" "$case" "$qualification"
for stage in time memory trace; do
    "$py" -u pair_mace_stage.py "$stage" "$variant" "$case" "$qualification" --result-root "$study"
done
"$py" -u audit_mace_variants.py "$PWD/$study" "$case" "$variant"
