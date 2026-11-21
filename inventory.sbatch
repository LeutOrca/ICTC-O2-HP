#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:04:00
#SBATCH --job-name=o2-whole-inventory
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python -u inventory.py provenance/environment-o2.json
../o2-hpx-opt-20261117-r1/environments/torch211-cu128/bin/python -u inventory.py provenance/environment-torch211.json
/public/home/sps-xia/bjxiao/.local/share/uv/tools/tavotto/bin/python3 -u inventory.py provenance/environment-tavotto.json
