#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=o2-angular-fixed
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
source runtime.sh
"$py" -S -u verify_sources.py
if [[ "$1" == so2 ]]; then
    "$py" -S -u cuda_context_probe.py
    "$py" -S -u campaign.py so2 16384
else
    export O2_INCLUDE_LOCAL_HPX=1
    "$py" -S -u continuation.py tp "$2" 1 --memory
fi
