#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:10:00
#SBATCH --job-name=o2-repair-final-summary
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
py=/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python
"$py" budget.py
"$py" aggregate.py
"$py" report_completed_repair.py
cd ../../figures/o2-hp-angular-operators-20261121-r1/repair-20261122-r1
export MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.mpl-cache"
"$py" repair_tp_throughput.py
"$py" repair_message_throughput.py
"$py" repair_so2_throughput.py
"$py" repair_ictp_throughput.py
"$py" repair_speedups.py
"$py" repair_hpx_speedup.py
