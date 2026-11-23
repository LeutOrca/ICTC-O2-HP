"""CPU-node-only source and project-budget audit for the approved continuation."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
assert os.environ.get("SLURM_JOB_ID"), "All Python checks require Slurm"
subprocess.run(["sha256sum", "--check", "provenance/approved-execution/BASELINE-CODE.sha256"], cwd=ROOT, check=True)
subprocess.run(["sha256sum", "--check", "provenance/UPSTREAM.sha256"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
files = [*ROOT.joinpath("candidate").glob("*.py"), *ROOT.joinpath("validation").glob("*.py")]
for path in files:
    ast.parse(path.read_text(), filename=str(path))
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
