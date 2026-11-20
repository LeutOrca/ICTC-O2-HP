#!/bin/bash
set -euo pipefail
py=${HPX_PYTHON:?Specify the pinned compute-node Python}
variant=${1:?Specify the isolated candidate}
case=${2:?Specify the frozen case}
qualification="results/mace-$variant-$case-qualification.json"
"$py" -u qualify_mace_variant.py "$variant" "$case" "$qualification"
for stage in time memory trace; do
    "$py" -u pair_mace_stage.py "$stage" "$variant" "$case" "$qualification"
done
