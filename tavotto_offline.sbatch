#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:04:00
#SBATCH --job-name=o2-tavotto-offline-update
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
sha256sum -c downloads/TAVOTTO-WHEEL.sha256
uv tool upgrade tavotto --offline --find-links "$PWD/downloads"
/public/home/sps-xia/bjxiao/.local/share/uv/tools/tavotto/bin/python3 -c 'from importlib.metadata import version; v=version("tavotto"); print("Tavotto engine",v); assert v=="0.21.3"'
