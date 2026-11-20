#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 00:07:00
#SBATCH -J mace-h100-trial-audit
#SBATCH -o logs/%x-%j.log
set -euo pipefail
folder=$1 variant=$2
shift 2
py=/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python
echo "Native CUDA Graph comparison"
"$py" audit_force_pairs.py "$folder" --variant "$variant" "$@"
echo "Incremental comparison against receiver CSR"
"$py" audit_force_pairs.py "$folder" --variant "$variant" --baseline receiver-csr-graph \
    --output PAIR_AUDIT-vs-receiver-csr.json "$@"
"$py" audit_traces.py "$folder"
find "$folder" -type f -print0 | sort -z | xargs -0 sha256sum > "provenance/$folder.sha256"
