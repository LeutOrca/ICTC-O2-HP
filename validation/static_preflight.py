"""CPU-node-only source and project-budget audit for the approved continuation."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
assert os.environ.get("SLURM_JOB_ID"), "All Python checks require Slurm"
subprocess.run(["sha256sum", "--check", "provenance/approved-execution/BASELINE-CODE.sha256"], cwd=ROOT, check=True)
subprocess.run(["sha256sum", "--check", "provenance/UPSTREAM.sha256"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
files = [*ROOT.joinpath("candidate").glob("*.py"), *ROOT.joinpath("validation").glob("*.py")]
for path in files:
    ast.parse(path.read_text(), filename=str(path))
from budget import ledger_records,request_gpus
assert request_gpus('sbatch --gres=gpu:1 broken="',1)==1
example='1|COMPLETED|60|gres/gpu=1|/o2-benchmark|node|GPU|gres/gpu=1|1|srun bash -c "a\nb"\n2|FAILED|1||/o2-benchmark|node|CPU||1|true\n'
records=list(ledger_records(example))
assert len(records)==2 and 'a\nb' in records[0]
subprocess.run([os.sys.executable, str(ROOT / "budget.py")], cwd=ROOT, check=True)
budget = json.loads((ROOT / "provenance/BUDGET-AUDIT.json").read_text())
if budget["remaining_project_gpu_h"] < 2:
    raise RuntimeError("Insufficient authorized project budget for the two-GPU-hour G1 envelope")
record = dict(passed=True, stage="static-and-budget-only", numerical_pass=False,
              job=os.environ["SLURM_JOB_ID"], candidate="17a71d71e10775f6eee70aa70196948d4178bf43",
              syntax_files=len(files), project_gpu_h=budget["project_gpu_h"],
              remaining_project_gpu_h=budget["remaining_project_gpu_h"],
              original_error_sha256=hashlib.sha256((ROOT / "benchmark.py").read_bytes()).hexdigest())
(ROOT / "provenance/approved-execution/STATIC-PREFLIGHT.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
