#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:08:00
#SBATCH --job-name=o2-angular-summary
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
py=/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python
"$py" aggregate.py
"$py" report.py
cd ../../figures/o2-hp-angular-operators-20261121-r1
export MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.mpl-cache"
"$py" tp_throughput.py
"$py" message_throughput.py
"$py" so2_throughput.py
"$py" ictp_throughput.py
"$py" speedups.py
"$py" hpx_speedup.py
