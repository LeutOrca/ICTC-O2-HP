#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --job-name=o2-gh-whole-paired
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
source h100_env.sh
cases=(b1 b8 n512 n4096)
"$py" -u campaign.py --case "${cases[$SLURM_ARRAY_TASK_ID]}"
