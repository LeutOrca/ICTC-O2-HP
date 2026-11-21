#!/bin/bash
#SBATCH -p GPU-A100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=o2-whole-preflight
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -uo pipefail
source env.sh
native_python=/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python
new_python="$PWD/../o2-hpx-opt-20261117-r1/environments/torch211-cu128/bin/python"
mkdir -p results/preflight provenance
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "provenance/gpu-${SLURM_JOB_ID}.csv"
for runtime in 27 211; do
    export O2_RUNTIME="$runtime"
    source env.sh
    if [ "$runtime" = 27 ]; then
        selected_python="$native_python"
        roles='official fork local-hpx eqx-current'
    else
        selected_python="$new_python"
        roles='cue-current'
    fi
    "$selected_python" -u whole.py ref native b1 fixed "results/preflight/${runtime}-native-b1.json"
    for role in $roles; do
        "$selected_python" -u whole.py build "$role" b1 fixed "results/preflight/${runtime}-${role}-build.json"
        if [ "$?" = 0 ]; then
            "$selected_python" -u whole.py qualify "$role" b1 fixed "results/preflight/${runtime}-${role}-b1.json" --reference "results/preflight/${runtime}-native-b1.pt"
        fi
    done
done
