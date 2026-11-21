#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:05:00
#SBATCH --job-name=o2-angular-night-preflight
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
py=/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python
"$py" -m py_compile night_fill.py phase_campaign.py
bash -n collect_morning.sh night.sbatch collect_morning.sbatch
"$py" night_fill.py tp 4096 --audit-only --receipt-root "$PWD/measurements"
"$py" night_fill.py radial 16384 --audit-only --receipt-root "$PWD/measurements"
