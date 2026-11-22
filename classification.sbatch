#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:04:00
#SBATCH --job-name=o2-full-classification
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
py=/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python
"$py" aggregate.py
"$py" audit_classification.py
