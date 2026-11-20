#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 00:07:00
#SBATCH -J mace-h100-pair-audit
#SBATCH -o logs/%x-%j.log
set -euo pipefail
folder=$1
shift
py=/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python
"$py" audit_force_pairs.py "$folder" "$@"
"$py" audit_traces.py "$folder"
find "$folder" -type f -print0 | sort -z | xargs -0 sha256sum > "provenance/$folder.sha256"
