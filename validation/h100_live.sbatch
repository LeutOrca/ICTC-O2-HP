#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=o2-approved-live
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
source runtime.sh
sha256sum --check provenance/approved-execution/BASELINE-CODE.sha256
"$py" -S -u validation/g1_live.py "$@"
