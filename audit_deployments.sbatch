#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --job-name=o2-angular-deployed-source
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python audit_deployments.py
