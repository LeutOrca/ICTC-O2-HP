"""Native fallback on the actual A100/o2 runtime; no H100 kernel promotion."""
import json
import os
from pathlib import Path
import sys

assert os.environ.get("SLURM_JOB_ID")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["O2_DIAGNOSTIC"] = "1"
import torch
from benchmark import configure, error
from models import Spec
from candidate.bounded_uvu import bounded, streamed
from candidate.reduction_policy import choose_uvu, choose_radial

configure()
build = sys.argv[1] == "build"
checks = []
for family in ("uvu", "radial"):
    spec = Spec(family, 1, "full", 37)
    tp = spec.calls["public"] if family == "uvu" else spec.calls["public"].tp
    values = list(spec.logical)
    linear = None if family == "uvu" else spec.diagnostics["linear"]
    for mode in ("auto", "native"):
        decision = choose_uvu(tp, *values, mode) if family == "uvu" else choose_radial(tp, linear, values[2], mode, execution="streamed", features=values[:2])
        assert decision.route == "native"
        out = bounded(tp, *values, policy=mode) if family == "uvu" else streamed(tp, linear, mode, *values)
        g = torch.full_like(out, .1)
        first = torch.autograd.grad(out, values, g)
        if not build:
            rv = [v.detach().requires_grad_() for v in values]
            reference = spec.reference(*rv)
            df = torch.autograd.grad(reference, rv, g)
            comparisons = [error(out, reference), *(error(a, b) for a, b in zip(first, df))]
            checks.append(dict(family=family, mode=mode, decision=decision.metadata(), passed=all(c["passed"] for c in comparisons), checks=comparisons))
        torch.cuda.synchronize()
# This separate structural fixture reaches runtime dispatch with exact covered
# shapes. Its uninitialized tensors are never read by a kernel.
spec = Spec("uvu", 6, "full", 1)
tp = spec.calls["public"]
values = [torch.empty(16384, width, device="cuda") for width in (tp.input1_dimension, tp.input2_dimension, tp.weight_numel)]
decision = choose_uvu(tp, *values, "auto")
checks.append(dict(name="exact-H100-shape-on-A100-runtime", decision=decision.metadata(), passed=decision.route == "native"))
try:
    choose_uvu(tp, *values, "required")
except RuntimeError as exc:
    checks.append(dict(name="required-rejects-A100-runtime", passed=True, message=str(exc)))
else:
    checks.append(dict(name="required-rejects-A100-runtime", passed=False))
record = dict(job=os.environ["SLURM_JOB_ID"], passed=all(c["passed"] for c in checks), built_only=build,
              device=torch.cuda.get_device_name(), torch=str(torch.__version__), checks=checks,
              full_parameter_radial=False, double_backward_qualified=False, h100_custom_on_a100=False,
              tf32=False, cuda_graph=False)
destination = ROOT / "results" / os.environ["SLURM_JOB_ID"]
destination.mkdir(parents=True, exist_ok=True)
(destination / "A100-FALLBACK.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2), flush=True)
if not record["passed"]:
    raise SystemExit(2)
