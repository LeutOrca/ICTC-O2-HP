#!/bin/bash
#SBATCH -p NV_H100
#SBATCH -A phy_gjxia
#SBATCH --qos=user_phy_bjxiao
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --job-name=o2-full-exec-probe
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
source runtime.sh
export O2_PROCESS_MODE=exec O2_RADIAL_ADJOINT=16x1b@128
export TORCH_EXTENSIONS_DIR="$PWD/../o2-hp-full-parity-repair-20261122-r1-756c995/cache/extensions"
export TRITON_CACHE_DIR="$PWD/../o2-hp-full-parity-repair-20261122-r1-756c995/cache/triton"
export XDG_CACHE_HOME="$PWD/../o2-hp-full-parity-repair-20261122-r1-756c995/cache/xdg"
mkdir -p results/exec-probe
"$py" -S -u benchmark.py build radial 4 full 16384 results/exec-probe/build-L4-full-eqx-o2-repair.json --role eqx-o2-repair --metric forward_vjp
"$py" -S -u benchmark.py qualify radial 4 full 16384 results/exec-probe/qualify-L4-full-eqx-o2-repair.json --role eqx-o2-repair --metric forward_vjp --reference-chunk 4096
