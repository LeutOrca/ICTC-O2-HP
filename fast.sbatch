#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=o2-angular-fast-process
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
source runtime.sh
export O2_PROCESS_MODE=forkserver
"$py" -S -u verify_sources.py
if [[ "$1" == probe ]]; then
    export O2_INCLUDE_LOCAL_HPX=1
    "$py" -S -u fast_probe.py
elif [[ "$1" == tp ]]; then
    export O2_INCLUDE_LOCAL_HPX=1
    export O2_REFERENCE_CHUNK=4096
    "$py" -S -u continuation.py tp "$2" 1 --memory
else
    "$py" -S -u phase_campaign.py "$@"
fi
