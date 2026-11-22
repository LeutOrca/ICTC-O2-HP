#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=o2-tavotto-engine-update
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
uv tool upgrade tavotto --index-url https://pypi.org/simple
/public/home/sps-xia/bjxiao/.local/share/uv/tools/tavotto/bin/python3 -c 'from importlib.metadata import version; print("Tavotto engine", version("tavotto"))'
