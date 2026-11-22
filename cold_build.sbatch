#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --job-name=o2-bounded-cold-build
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
source runtime.sh
export O2_PROCESS_MODE=exec O2_RADIAL_ADJOINT=shape-v2 O2_BOUNDED_UVU=1
"$py" -S -u verify_sources.py
"$py" -S -u cold_build.py
