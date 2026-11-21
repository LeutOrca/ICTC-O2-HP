#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --job-name=o2-angular-morning-collect
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
bash collect_morning.sh
