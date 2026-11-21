"""One pre-submitted allocation fills only missing paired time/memory cases."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
IDENTITY=("family","order","parity","rows","torch","gpu_uuid","input_sha256","canonical_weights_sha256")
ROLES={"tp":("public","local-hpx","eqx-o2","cue-o2"),"uvu":("public","eqx-o2"),
       "radial":("public","public-unfused","eqx-o2")}


def scan(family,rows,receipt_root=None):
    roots=[receipt_root] if receipt_root else [p/"results" for p in ROOT.parent.glob("o2-hp-angular-operators-20261121-r1-*")]
    data=[]
    gates={}
    for root in roots:
        for path in root.rglob("*.json"):
            if f"{family}-{rows}" not in path.parent.name or not path.name.startswith(("qualify","time","memory")):
                continue
            try:
                raw=path.read_bytes()
                r=json.loads(raw)
            except (json.JSONDecodeError,OSError):
                continue
            if r.get("family")!=family or r.get("rows")!=rows or not r.get("passed"):
                continue
            data.append(r)
            if r.get("stage")=="qualify":
                gates[hashlib.sha256(raw).hexdigest()]=r
    times={}
    memory={}
    for r in data:
        if r.get("stage") not in ("time","memory"):
            continue
        q=gates.get(r.get("qualification_sha256"))
        if q is None or any(r.get(k)!=q.get(k) for k in IDENTITY):
            continue
        if q.get("dtype")!="float32" or any(q.get(k,True) for k in ("tf32","amp","cuda_graph")):
            continue
        if any(v.get("reference_chunk_size",512)>512 and v.get("check_window_size")!=512 for v in q.get("numerics",{}).values()):
            continue
        key=(r["order"],r["parity"],r["gpu_uuid"],r["job"],r["input_sha256"],r.get("canonical_weights_sha256"))
        if r["stage"]=="time":
            for m in r.get("measurements",[]):
                pair=(m["role"],m["metric"])
                if m["role"] in q.get("accepted_by_metric",{}).get(m["metric"],[]) and len(m["wall_samples_ms"])==30:
                    times.setdefault(key,{}).setdefault(pair,set()).add(r["repeat"])
        elif "discarded_preparation_cotangent_bytes" in r:
            if r["role"] in q.get("accepted_by_metric",{}).get(r["metric"],[]):
                memory.setdefault(key,set()).add((r["role"],r["metric"]))
    expected={(role,metric) for role in ROLES[family] for metric in ("forward","forward_vjp")}
    complete_times={(k[0],k[1]) for k,groups in times.items() if all(groups.get(pair)==set(range(5)) for pair in expected)}
    complete_memory={(k[0],k[1]) for k,groups in memory.items() if expected<=groups}
    return complete_times,complete_memory


def main():
    p=argparse.ArgumentParser()
    p.add_argument("family",choices=tuple(ROLES))
    p.add_argument("rows",type=int)
    p.add_argument("--audit-only",action="store_true")
    p.add_argument("--receipt-root",type=Path)
    a=p.parse_args()
    timed,measured_memory=scan(a.family,a.rows,a.receipt_root)
    actions=[]
    for order in range(8,0,-1):
        for parity in ("full","canonical"):
            case=(order,parity)
            mode=("skip" if case in timed and case in measured_memory else
                  "memory" if case in timed else "curves" if case in measured_memory else "both")
            actions.append(dict(order=order,parity=parity,mode=mode))
    plan=dict(family=a.family,rows=a.rows,actions=actions,audit_only=a.audit_only)
    if a.audit_only:
        print(json.dumps(plan),flush=True)
        return
    result=ROOT/"results"/f"NIGHT-{a.family}-{a.rows}.json"
    result.parent.mkdir(parents=True,exist_ok=True)
    result.write_text(json.dumps(plan,indent=2))
    import phase_campaign
    original=sys.argv[:]
    for item in actions:
        if item["mode"]=="skip":
            print("skip complete",a.family,a.rows,item["order"],item["parity"],flush=True)
            continue
        sys.argv=[str(ROOT/"phase_campaign.py"),item["mode"],a.family,str(item["order"]),
            "--end",str(item["order"]),"--parity",item["parity"],"--rows",str(a.rows),
            "--build-timeout","900","--qualify-timeout","1800"]
        phase_campaign.main()
        item["attempted"]=True
        result.write_text(json.dumps(plan,indent=2))
    sys.argv=original
    plan["finished"]=True
    result.write_text(json.dumps(plan,indent=2))


if __name__=="__main__":
    main()
