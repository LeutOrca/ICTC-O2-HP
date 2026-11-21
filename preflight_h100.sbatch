#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --job-name=o2-gh-whole-preflight
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -uo pipefail
source h100_env.sh
sha256sum --quiet -c provenance/deployment.sha256 || exit 3
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "provenance/gpu-${SLURM_JOB_ID}.csv"
"$py" -u inventory.py "provenance/environment-h100-${SLURM_JOB_ID}.json"
"$py" -u whole.py ref native b1 fixed results/preflight/26-native-b1.json
for role in official fork local-hpx eqx-current cue-current; do
    "$py" -u whole.py build "$role" b1 fixed "results/preflight/26-${role}-build.json"
    if [ "$?" = 0 ]; then
        "$py" -u whole.py qualify "$role" b1 fixed "results/preflight/26-${role}-b1.json" --reference results/preflight/26-native-b1.pt
    fi
done
