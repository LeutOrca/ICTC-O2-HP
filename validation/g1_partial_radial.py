"""Validate every nonempty radial gradient-role mask, without inactive parameters."""
import json
import os
from pathlib import Path
import sys

assert os.environ.get("SLURM_JOB_ID")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["O2_DIAGNOSTIC"] = "1"
import torch
import torch.nn.functional as F
from benchmark import configure, error
from models import Spec
from validation.g1_live import derived
from candidate.streamed_live_radial import streamed_live_radial
from ictc_o2_hp import ReferenceUvuTensorProduct

configure()
records = []
for mask in range(1, 32):
    spec = Spec("radial", 1, "full", 37)
    tp, linear = spec.calls["public"].tp, spec.diagnostics["linear"]
    values = [*spec.logical, linear.weight, linear.bias]
    for index, value in enumerate(values):
        value.requires_grad_(bool(mask & (1 << index)))
    active = [i for i, v in enumerate(values) if v.requires_grad]
    targets = [values[i] for i in active]
    # Reinsert only active target leaves; inactive canonical values remain.
    def actual(*args):
        return streamed_live_radial(tp, linear, *values[:3])
    oracle = ReferenceUvuTensorProduct(64, 1, 1, parity="full", internal_weights=False, shared_weights=False).cuda().float()
    refs = [v.detach().requires_grad_(v.requires_grad) for v in values]
    def reference(*args):
        return oracle(refs[0], refs[1], F.linear(refs[2], refs[3], refs[4]))
    g = torch.full((37, tp.representation_dimension), .1, device="cuda", requires_grad=True)
    directions = tuple(torch.full_like(v, .2) for v in targets)
    out, df, ddf = derived(actual, targets, g, directions)
    expected, rf, rrf = derived(reference, [refs[i] for i in active], g.detach().requires_grad_(), directions)
    checks = []
    for a, b in zip((out, *df, *ddf), (expected, *rf, *rrf)):
        if a is None and b is None:
            check = dict(passed=True, zero_unused=True)
        else:
            check = error(torch.zeros_like(b) if a is None else a,
                          torch.zeros_like(a) if b is None else b)
        checks.append(check)
    records.append(dict(mask=mask, roles=active, passed=all(c["passed"] for c in checks), checks=checks))
result = dict(job=os.environ["SLURM_JOB_ID"], passed=all(c["passed"] for c in records), cases=records,
              tf32=False, cuda_graph=False, scope="partial-role radial first and second only")
destination = ROOT / "results" / os.environ["SLURM_JOB_ID"]
destination.mkdir(parents=True, exist_ok=True)
(destination / "PARTIAL-RADIAL.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
if not result["passed"]:
    raise SystemExit(2)
