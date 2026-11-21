#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --job-name=o2-angular-collector-probe
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i /public/home/sps-xia/bjxiao/.ssh/id_rsa_GBU -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o ConnectTimeout=8 -p 18022 phy_bjxiao@10.10.4.99 hostname
