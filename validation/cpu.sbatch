#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:10:00
#SBATCH --job-name=o2-approved-static
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python -u validation/static_preflight.py
