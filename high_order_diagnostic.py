"""Bounded per-role/per-phase high-order builds, preserving timeout receipts."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    target=ROOT/"results/high-order-diagnostic"
    target.mkdir(parents=True,exist_ok=True)
    for family,roles in (("uvu",("public","eqx-o2")),("radial",("public","public-unfused","eqx-o2"))):
        for metric in ("forward","forward_vjp"):
            for role in roles:
                output=target/f"{family}-L8-{role}-{metric}.json"
                command=[sys.executable,"-S","-u",str(ROOT/"benchmark.py"),"build",family,"8","full","37",str(output),"--role",role,"--metric",metric]
                with output.with_suffix(".log").open("w") as log:
                    try:
                        done=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=180,check=False)
                        code=done.returncode
                    except subprocess.TimeoutExpired:
                        code=124
                if code==124:
                    previous=json.loads(output.read_text()) if output.exists() else {}
                    previous.update(stage="build",family=family,order=8,parity="full",rows=37,
                        role=role,metric=metric,passed=False,status="build_timeout",timeout_s=180)
                    output.write_text(json.dumps(previous,indent=2))
                print(family,role,metric,code,flush=True)


if __name__=="__main__":
    main()
