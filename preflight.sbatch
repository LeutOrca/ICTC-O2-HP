#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --job-name=o2-angular-preflight
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -uo pipefail
source runtime.sh
sha256sum --quiet -c provenance/deployment.sha256 || exit 3
for family in tp uvu radial so2 ictp; do
    for order in 1 8; do
        parity=full
        if [[ "$family" = so2 || "$family" = ictp ]]; then parity=natural; fi
        "$py" -S -u benchmark.py build "$family" "$order" "$parity" 37 "results/preflight/build-$family-$order.json"
        if [[ "$?" = 0 ]]; then
            "$py" -S -u benchmark.py qualify "$family" "$order" "$parity" 37 "results/preflight/qualify-$family-$order.json"
        fi
    done
done
