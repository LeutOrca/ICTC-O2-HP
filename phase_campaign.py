"""Isolate remaining roles for high-order memory or missing absolute curves."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("memory","curves","both"))
    p.add_argument("family",choices=("tp","uvu","radial"))
    p.add_argument("start",type=int)
    p.add_argument("--role")
    p.add_argument("--parity",choices=("canonical","full"))
    p.add_argument("--descending",action="store_true")
    p.add_argument("--end",type=int,default=8)
    p.add_argument("--rows",type=int,default=16384)
    p.add_argument("--build-timeout",type=int,default=360)
    p.add_argument("--qualify-timeout",type=int,default=900)
    a=p.parse_args()
    result=ROOT/"results"/f"{a.mode}-phase-{a.family}-{a.rows}"
    result.mkdir(parents=True,exist_ok=True)
    def run(stage,order,parity,role,metric="forward_vjp",timeout=600,**options):
        path=result/f"{stage}-L{order}-{parity}-{role}-{metric}-r{options.get('repeat',0)}.json"
        command=[sys.executable,"-S","-u",str(ROOT/"benchmark.py"),stage,a.family,str(order),parity,str(a.rows),str(path),
                 "--role",role,"--metric",metric]
        if stage=="qualify":
            command.extend(("--reference-chunk","4096"))
        for key,value in options.items():
            command.extend(("--"+key,str(value)))
        if os.environ.get("O2_PROCESS_MODE")=="forkserver":
            from fork_runner import run as fork_run
            code=fork_run(command,path.with_suffix(".log"),timeout)
        else:
            with path.with_suffix(".log").open("w") as log:
                proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    code=proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait()
                    code=124
        if code==124 or not path.exists():
            data=json.loads(path.read_text()) if path.exists() else {}
            data.update(stage=stage,family=a.family,order=order,parity=parity,rows=a.rows,
                        role=role,metric=metric,passed=False,status="timeout" if code==124 else "process-failure",timeout_s=timeout)
            path.write_text(json.dumps(data,indent=2))
        data=json.loads(path.read_text())
        print(a.mode,a.family,order,parity,role,metric,stage,data["status"],flush=True)
        return path,data
    defaults={"tp":("public","local-hpx","eqx-o2","cue-o2"),"uvu":("public","eqx-o2"),
              "radial":("public","public-unfused","eqx-o2")}
    roles=(a.role,) if a.role else defaults[a.family]
    orders=range(a.end,a.start-1,-1) if a.descending else range(a.start,a.end+1)
    for order in orders:
        for parity in ((a.parity,) if a.parity else ("canonical","full")):
            gates=[]
            for role in roles:
                accepted=None
                for metric in ("forward_vjp","forward"):
                    _,built=run("build",order,parity,role,metric,timeout=a.build_timeout)
                    if not built.get("passed"):
                        continue
                    q,gate=run("qualify",order,parity,role,metric,timeout=a.qualify_timeout)
                    if gate.get("passed"):
                        accepted=(q,gate)
                        break
                if accepted is None:
                    continue
                q,gate=accepted
                gates.append((q,gate))
            if not gates:
                continue
            gate=dict(gates[0][1])
            gate.update(role=None,stage="qualify",status="qualified",passed=True,
                accepted=sorted({role for _,g in gates for role in g["accepted"]}),
                accepted_by_metric={metric:sorted({role for _,g in gates for role in g["accepted_by_metric"][metric]})
                    for metric in ("forward","forward_vjp")},
                numerics={key:value for _,g in gates for key,value in g["numerics"].items()},
                component_qualifications=[hashlib.sha256(path.read_bytes()).hexdigest() for path,_ in gates])
            q=result/f"qualify-L{order}-{parity}-merged.json"
            q.write_text(json.dumps(gate,indent=2))
            if a.mode in ("curves","both"):
                for repeat in range(5):
                    run("time",order,parity,"all",qualification=q,repeat=repeat)
            if a.mode in ("memory","both"):
                for metric,good in gate["accepted_by_metric"].items():
                    for role in good:
                        run("memory",order,parity,role,metric,qualification=q)


if __name__=="__main__":
    main()
