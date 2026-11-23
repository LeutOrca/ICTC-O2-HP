#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --job-name=o2-approved-safety
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
source runtime.sh
sha256sum --check provenance/approved-execution/BASELINE-CODE.sha256
case "$1" in
    guards) "$py" -S -u validation/g1_guards.py ;;
    largeoffset) "$py" -S -u validation/g1_largeoffset.py ;;
    fp64) "$py" -S -u validation/g1_fp64.py ;;
    shortprofile) "$py" -S -u validation/diagnose_short_hidden.py ;;
    *) exit 64 ;;
esac
