#!/bin/bash
set -euo pipefail
cd /public/home/sps-xia/bjxiao/o2-benchmark/runs/o2-hp-angular-operators-20261121-r1
remote_root=/public_hw/home/phy_bjxiao/o2-benchmark/runs
ssh_args=(-o BatchMode=yes -o IdentitiesOnly=yes -i /public/home/sps-xia/bjxiao/.ssh/id_rsa_GBU -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o ConnectTimeout=10 -p 18022)
rsync_ssh='ssh -o BatchMode=yes -o IdentitiesOnly=yes -i /public/home/sps-xia/bjxiao/.ssh/id_rsa_GBU -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o ConnectTimeout=10 -p 18022'
# A failed scheduler query must never truncate the last usable accounting file.
capture() {
    local target=$1 tmp
    shift
    tmp=$(mktemp "${target}.tmp.XXXXXX")
    if "$@" > "$tmp" && [[ -s "$tmp" ]]; then
        mv "$tmp" "$target"
        printf '%s SUCCESS %s\n' "$(date -Is)" "$target" >> provenance/COLLECTION-ATTEMPTS.txt
    else
        rm -f "$tmp"
        printf '%s FAILED-PRESERVED %s\n' "$(date -Is)" "$target" >> provenance/COLLECTION-ATTEMPTS.txt
        return 1
    fi
}
while read -r sha label; do
    [[ -z "$sha" ]] && continue
    mkdir -p "measurements/$label/source-seals" "measurements/$label/logs"
    rsync -a -e "$rsync_ssh" "phy_bjxiao@10.10.4.99:$remote_root/o2-hp-angular-operators-20261121-r1-$sha/results/" "measurements/$label/"
    rsync -a --include='SOURCE-SEAL-*.json' --exclude='*' -e "$rsync_ssh" "phy_bjxiao@10.10.4.99:$remote_root/o2-hp-angular-operators-20261121-r1-$sha/provenance/" "measurements/$label/source-seals/"
    rsync -a -e "$rsync_ssh" "phy_bjxiao@10.10.4.99:$remote_root/o2-hp-angular-operators-20261121-r1-$sha/logs/" "measurements/$label/logs/"
done < provenance/REMOTE-COHORTS.tsv
capture provenance/project-gbu2-accounting-extended.tsv ssh "${ssh_args[@]}" phy_bjxiao@10.10.4.99 'source /etc/profile; sacct -n -X -S 2026-11-15 -u phy_bjxiao --format=JobID,State,ElapsedRaw,AllocTRES,WorkDir,NodeList,Partition,ReqTRES,NNodes,SubmitLine -P' || true
capture provenance/project-aidam-accounting-extended.tsv sacct -n -X -S 2026-11-15 -u bjxiao --format=JobID,State,ElapsedRaw,AllocTRES,WorkDir,NodeList,Partition,ReqTRES,NNodes,SubmitLine -P || true
capture provenance/MORNING-QUEUE.txt ssh "${ssh_args[@]}" phy_bjxiao@10.10.4.99 'source /etc/profile; squeue -u phy_bjxiao -o "%.18i %.30j %.9T %.10M %.20R"' || true
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
