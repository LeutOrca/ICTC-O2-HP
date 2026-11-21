#!/bin/bash
set -euo pipefail
cd /public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-hp-angular-operators-20261121-r1
remote_root=/public_hw/home/phy_bjxiao/o2-benchmark/runs
ssh_args=(-o BatchMode=yes -o IdentitiesOnly=yes -i /public/home/sps-xia/bjxiao/.ssh/id_rsa_GBU -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o ConnectTimeout=10 -p 18022)
rsync_ssh='ssh -o BatchMode=yes -o IdentitiesOnly=yes -i /public/home/sps-xia/bjxiao/.ssh/id_rsa_GBU -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o ConnectTimeout=10 -p 18022'
while read -r sha label; do
    [[ -z "$sha" ]] && continue
    mkdir -p "measurements/$label/source-seals" "measurements/$label/logs"
    rsync -a -e "$rsync_ssh" "phy_bjxiao@10.10.4.99:$remote_root/o2-hp-angular-operators-20261121-r1-$sha/results/" "measurements/$label/"
    rsync -a --include='SOURCE-SEAL-*.json' --exclude='*' -e "$rsync_ssh" "phy_bjxiao@10.10.4.99:$remote_root/o2-hp-angular-operators-20261121-r1-$sha/provenance/" "measurements/$label/source-seals/"
    rsync -a -e "$rsync_ssh" "phy_bjxiao@10.10.4.99:$remote_root/o2-hp-angular-operators-20261121-r1-$sha/logs/" "measurements/$label/logs/"
done < provenance/REMOTE-COHORTS.tsv
ssh "${ssh_args[@]}" phy_bjxiao@10.10.4.99 'source /etc/profile; sacct -n -X -S 2026-11-15 -u phy_bjxiao --format=JobID,State,ElapsedRaw,AllocTRES,WorkDir,NodeList,Partition,ReqTRES,NNodes,SubmitLine -P' > provenance/project-gbu2-accounting-extended.tsv
sacct -n -X -S 2026-11-15 -u bjxiao --format=JobID,State,ElapsedRaw,AllocTRES,WorkDir,NodeList,Partition,ReqTRES,NNodes,SubmitLine -P > provenance/project-aidam-accounting-extended.tsv
ssh "${ssh_args[@]}" phy_bjxiao@10.10.4.99 'source /etc/profile; squeue -u phy_bjxiao -o "%.18i %.30j %.9T %.10M %.20R"' > provenance/MORNING-QUEUE.txt
if [[ "${O2_COLLECT_ONLY:-0}" == 1 ]]; then
    exit 0
fi
py=/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python
"$py" budget.py
"$py" aggregate.py
"$py" report.py
cd ../../figures/o2-hp-angular-operators-20261121-r1
export MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.mpl-cache"
for script in tp_throughput.py message_throughput.py so2_throughput.py ictp_throughput.py speedups.py hpx_speedup.py; do
    "$py" "$script"
done
