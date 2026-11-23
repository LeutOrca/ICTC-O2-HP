"""Regression of eager-only rejection and native tracing fallback; not AOTI."""
import json
import argparse
import os
from pathlib import Path
import sys

assert os.environ.get("SLURM_JOB_ID")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from benchmark import configure, error
from models import Spec
from candidate.bounded_uvu import bounded
from candidate.live_uvu import live_uvu

configure()
parser = argparse.ArgumentParser()
parser.add_argument("--makefx-only", action="store_true")
args = parser.parse_args()
spec = Spec("uvu", 1, "full", 37)
tp = spec.calls["public"]
records = []
with FakeTensorMode(allow_non_fake_inputs=True) as mode:
    fake = [mode.from_tensor(v) for v in spec.logical]
    try:
        live_uvu(tp, *fake)
    except RuntimeError as exc:
        assert "proxy/fake" in str(exc) or "context" in str(exc) or "tensor subclasses" in str(exc)
        records.append(dict(mode="explicit-live-fake", status="REFUSED_AS_CONTRACTED", passed=True, message=str(exc)))
    else:
        raise RuntimeError("Explicit raw-kernel entry accepted fake tensors")

function = lambda x, y, w: bounded(tp, x, y, w, policy="auto")
builders = [("make_fx-symbolic", lambda: make_fx(function, tracing_mode="symbolic", _allow_non_fake_inputs=True)(*spec.logical))]
if not args.makefx_only:
    builders.append(("compile-eager-backend", lambda: torch.compile(function, backend="eager", dynamic=True, fullgraph=False)))
for name, build in builders:
    try:
        call = build()
        checks = []
        for rows in (37, 41):
            case = Spec("uvu", 1, "full", rows)
            values = case.logical
            output = call(*values)
            g = torch.full_like(output, .1)
            first = torch.autograd.grad(output, values, g)
            refs = [v.detach().requires_grad_() for v in values]
            expected = case.reference(*refs)
            rf = torch.autograd.grad(expected, refs, g)
            checks.extend([dict(rows=rows, role=0, **error(output, expected)),
                           *(dict(rows=rows, role=i + 1, **error(a, b)) for i, (a, b) in enumerate(zip(first, rf)))])
        records.append(dict(mode=name, status="PASS" if all(c["passed"] for c in checks) else "NUMERICAL_FAIL",
                            passed=all(c["passed"] for c in checks), checks=checks, route="native-fallback"))
    except Exception as exc:
        # Keep unsupported tracing separate from successful eager qualification.
        records.append(dict(mode=name, status="HOLD_TRACING", passed=False,
                            error_type=type(exc).__name__, message=str(exc)[:2500]))
result = dict(job=os.environ["SLURM_JOB_ID"], cases=records, tf32=False, cuda_graph=False,
              tested_modes=[r["mode"] for r in records],
              eager_only_entry_refusal_passed=records[0]["passed"],
              native_tracing_passed=all(r["passed"] for r in records[1:]),
              offline_dynamic_aoti_passed=False, graph_breaks_allowed_for_compile_eager_backend=True,
              own_kernel_compiled_performance_passed=False)
destination = ROOT / "results" / os.environ["SLURM_JOB_ID"]
destination.mkdir(parents=True, exist_ok=True)
(destination / "TRACING.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
