"""Conservative project-period Slurm accounting; run on a CPU compute node."""
import json
import hashlib
from pathlib import Path
import re
import shlex
from collections import Counter

ROOT=Path(__file__).resolve().parent


def tres_gpus(value):
    entries=dict(item.split("=",1) for item in value.split(",") if "=" in item)
    if "gres/gpu" in entries:
        return int(entries["gres/gpu"])
    typed=[int(v) for k,v in entries.items() if k.startswith("gres/gpu:")]
    return sum(typed) if typed else None


def request_gpus(text,node_count):
    words=shlex.split(text)
    for index,word in enumerate(words):
        for option in ("--gres","--gpus","--gpus-per-node"):
            value=word.split("=",1)[1] if word.startswith(option+"=") else (
                words[index+1] if word==option and index+1<len(words) else None)
            if value is None:
                continue
            if option=="--gres":
                counts=[int(item.rsplit(":",1)[1]) if ":" in item else 1
                        for item in value.split(",") if item=="gpu" or item.startswith("gpu:")]
                return sum(counts)*node_count
            return int(value.rsplit(":",1)[-1])*(node_count if option.endswith("per-node") else 1)
    return None


def infer_gpu_count(allocated,requested,submission,workdir,partition,node_count):
    for method,value in (("AllocTRES",allocated),("ReqTRES",requested)):
        count=tres_gpus(value)
        if count is not None:
            return count,method,{}
    count=request_gpus(submission,node_count)
    if count is not None:
        return count,"SubmitLine",{}
    try:
        words=shlex.split(submission)
    except ValueError:
        words=[]
    for word in words:
        if word.startswith("-") or not word.endswith((".sbatch",".sh")):
            continue
        path=Path(word) if word.startswith("/") else Path(workdir)/word
        if not path.is_file():
            continue
        content=path.read_text()
        directives=" ".join(line.strip()[7:].strip() for line in content.splitlines() if line.strip().startswith("#SBATCH"))
        count=request_gpus(directives,node_count)
        if count is not None:
            return count,"SBATCH-request",dict(request_script=str(path),request_script_sha256=hashlib.sha256(content.encode()).hexdigest())
    if re.search("gpu|a100|h100|titan|v100",partition,re.I):
        return 8*node_count,"conservative-node-upper-bound",dict(unresolved=True)
    return 0,"no-gpu-request",{}


def main():
    rows=[]
    seen=set()
    for cluster in ("aidam","gbu2"):
        extended=ROOT/f"provenance/project-{cluster}-accounting-extended.tsv"
        if extended.exists() and extended.stat().st_size == 0:
            raise RuntimeError(f"Empty accounting snapshot: {extended}; refresh on login before aggregation")
        source=extended if extended.exists() else ROOT/f"provenance/project-{cluster}-accounting.tsv"
        for line in source.read_text().splitlines():
            fields=line.split("|",9)
            if len(fields)<6:
                continue
            job,state,seconds,tres,workdir,nodes=fields[:6]
            if (cluster,job) in seen:
                continue
            seen.add((cluster,job))
            partition,requested,node_count,submission=(fields[6:10] if len(fields)>=10 else ("", "", "1", ""))
            if "/o2-benchmark" not in workdir and "/o2-benchmark" not in submission:
                continue
            seconds=int(seconds or 0)
            count,method,evidence=infer_gpu_count(tres,requested,submission,workdir,partition,max(1,int(node_count or 1)))
            if not count:
                continue
            rows.append(dict(cluster=cluster,job=job,state=state,seconds=seconds,
                gpus=count,gpu_h=seconds*count/3600,method=method,
                workdir=workdir,nodes=nodes,partition=partition,submission=submission,**evidence))
    project=sum(r["gpu_h"] for r in rows)
    angular=sum(r["gpu_h"] for r in rows if "o2-hp-angular-operators-20261121-r1" in r["workdir"])
    repair_rows=[r for r in rows if "o2-hp-full-parity-repair-20261122-r1" in r["workdir"]]
    repair=sum(r["gpu_h"] for r in repair_rows)
    angular_rows=[r for r in rows if "o2-hp-angular-operators-20261121-r1" in r["workdir"]]
    record=dict(project_period_start="2026-11-15",project_gpu_h=project,
        project_hard_cap=500,remaining_project_gpu_h=500-project,
        angular_gpu_h=angular,repair_gpu_h=repair,inclusive_angular_gpu_h=angular+repair,
        repair_gpu_jobs=len(repair_rows),proposed_inclusive_angular_envelope=36,
        project_gpu_h_by_cluster={c:sum(r["gpu_h"] for r in rows if r["cluster"]==c) for c in ("aidam","gbu2")},
        accounting_methods=dict(Counter(r["method"] for r in rows)),
        unresolved_gpu_jobs=[r for r in rows if r.get("unresolved")],
        angular_gpu_jobs=len(angular_rows),angular_nodes=sorted({r["nodes"] for r in angular_rows if r["seconds"]>0}),
        authorized_total_cap_preserved=project+max(0,36-angular-repair)<=500,rows=rows)
    (ROOT/"provenance/BUDGET-AUDIT.json").write_text(json.dumps(record,indent=2))
    print(json.dumps({k:v for k,v in record.items() if k!="rows"}),flush=True)
    if not record["authorized_total_cap_preserved"]:
        raise RuntimeError("Project hard cap would be exceeded")


if __name__=="__main__":
    main()
