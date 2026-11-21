#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=o2-angular-night-fill
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
source runtime.sh
export O2_PROCESS_MODE=forkserver
if [[ "$1" == tp ]]; then
    export O2_INCLUDE_LOCAL_HPX=1
else
    unset O2_INCLUDE_LOCAL_HPX
fi
"$py" -S -u verify_sources.py
"$py" -S -u night_fill.py "$@"
