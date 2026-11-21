"""Conservative project-period Slurm accounting; run on a CPU compute node."""
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent


def main():
    rows=[]
    seen=set()
    for cluster in ("aidam","gbu2"):
        for line in (ROOT/f"provenance/project-{cluster}-accounting.tsv").read_text().splitlines():
            fields=line.split("|")
            if len(fields)<6:
                continue
            job,state,seconds,tres,workdir,nodes=fields[:6]
            match=re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)",tres)
            if not match or (cluster,job) in seen:
                continue
            seen.add((cluster,job))
            if "/o2-benchmark" not in workdir:
                continue
            seconds=int(seconds or 0)
            rows.append(dict(cluster=cluster,job=job,state=state,seconds=seconds,
                gpus=int(match.group(1)),gpu_h=seconds*int(match.group(1))/3600,
                workdir=workdir,nodes=nodes))
    project=sum(r["gpu_h"] for r in rows)
    angular=sum(r["gpu_h"] for r in rows if "o2-hp-angular-operators-20261121-r1" in r["workdir"])
    record=dict(project_period_start="2026-11-15",project_gpu_h=project,
        project_hard_cap=500,remaining_project_gpu_h=500-project,
        angular_gpu_h=angular,proposed_inclusive_angular_envelope=16,
        authorized_total_cap_preserved=project+max(0,16-angular)<=500,rows=rows)
    (ROOT/"provenance/BUDGET-AUDIT.json").write_text(json.dumps(record,indent=2))
    print(json.dumps({k:v for k,v in record.items() if k!="rows"}),flush=True)
    if not record["authorized_total_cap_preserved"]:
        raise RuntimeError("Project hard cap would be exceeded")


if __name__=="__main__":
    main()
