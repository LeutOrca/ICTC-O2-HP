#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:12:00
#SBATCH --job-name=o2-angular-restore-qa
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python finalize.py
