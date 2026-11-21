"""Check separate CUDA-clean forkserver qualification/timing/memory processes."""
import json
from pathlib import Path
import sys
from fork_runner import run

ROOT=Path(__file__).resolve().parent


def main():
    out=ROOT/"results/forkserver-probe"
    out.mkdir(parents=True,exist_ok=True)
    q=out/"qualify.json"
    receipts=[]
    for stage,extra in (("build",[]),("qualify",[]),("time",["--qualification",str(q)]),
        ("memory",["--qualification",str(q),"--role","local-hpx","--metric","forward_vjp"])):
        path=q if stage=="qualify" else out/f"{stage}.json"
        command=[sys.executable,"-S","-u",str(ROOT/"benchmark.py"),stage,"tp","1","full","37",str(path),*extra]
        if run(command,path.with_suffix(".log"),300):
            raise RuntimeError(f"Forkserver {stage} failed")
        data=json.loads(path.read_text())
        assert data["passed"] and data["process_mode"]=="forkserver"
        receipts.append(data)
    assert len({r["pid"] for r in receipts})==4
    assert len({r["gpu_uuid"] for r in receipts})==1
    (out/"PROBE.json").write_text(json.dumps(dict(passed=True,pids=[r["pid"] for r in receipts],
        accepted=receipts[1]["accepted_by_metric"],distinct_cuda_processes=True),indent=2))
    print("CUDA-clean forkserver: four independent phases passed",flush=True)


if __name__=="__main__":
    main()
