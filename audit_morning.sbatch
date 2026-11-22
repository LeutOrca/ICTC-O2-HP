#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:06:00
#SBATCH --job-name=o2-angular-morning-audit
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
py=/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python
"$py" audit_deployments.py
"$py" audit_morning.py
