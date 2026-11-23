"""Supplemental real >2**31 element-offset fixture, never a throughput point."""
import json
import os
from pathlib import Path
import sys

assert os.environ.get("SLURM_JOB_ID")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from benchmark import configure, error
from models import Spec
from candidate.live_uvu import live_uvu

configure()
rows = 40000
spec = Spec("uvu", 8, "full", rows)
tp = spec.calls["public"]
x, y, w = spec.logical
boundary = (2**31 + tp.weight_numel - 1) // tp.weight_numel
assert boundary < rows and w.numel() > 2**31
out = live_uvu(tp, x, y, w)
g = torch.full_like(out, .1)
first = torch.autograd.grad(out, (x, y, w), g)
torch.cuda.synchronize()
checks = []
for start, end in ((0, 3), (boundary - 2, boundary + 3), (rows - 3, rows)):
    ref_values = [v[start:end].detach().requires_grad_() for v in (x, y, w)]
    expected = spec.reference(*ref_values)
    df = torch.autograd.grad(expected, ref_values, g[start:end])
    for index, (a, b) in enumerate(zip((out[start:end], *(v[start:end] for v in first)), (expected, *df))):
        checks.append(dict(rows=[start, end], role=index, **error(a, b)))
result = dict(job=os.environ["SLURM_JOB_ID"], passed=all(c["passed"] for c in checks),
              fixture="supplemental true allocation, not frozen benchmark input replacement",
              weight_shape=list(w.shape), max_element_offset=w.numel() - 1,
              crossing_row=boundary, checks=checks,
              input_sha256=spec.metadata["input_sha256"], tf32=False, cuda_graph=False,
              peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
destination = ROOT / "results" / os.environ["SLURM_JOB_ID"]
destination.mkdir(parents=True, exist_ok=True)
(destination / "LARGE-OFFSET.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
if not result["passed"]:
    raise SystemExit(2)
