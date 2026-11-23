#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --job-name=o2-recover-receipts
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python -u validation/recover_receipts.py
