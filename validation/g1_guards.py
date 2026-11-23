"""Capability-aware mutation/layout/partial-gradient regression; Slurm GPU only."""
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
from candidate.live_uvu import live_uvu
from candidate.streamed_live_radial import streamed_live_radial
from candidate.bounded_uvu import bounded

configure()
checks = []


def reject(name, function):
    try:
        function()
    except (RuntimeError, TypeError, ValueError) as exc:
        checks.append(dict(name=name, passed=True, rejected=type(exc).__name__, message=str(exc)[:500]))
    else:
        checks.append(dict(name=name, passed=False, message="Required rejection absent"))


for mask in range(1, 8):
    spec = Spec("uvu", 1, "full", 37)
    tp = spec.calls["public"]
    values = [v.detach().T.contiguous().T.requires_grad_(bool(mask & (1 << i)))
              for i, v in enumerate(spec.logical)]
    targets = [v for v in values if v.requires_grad]
    output = live_uvu(tp, *values)
    g = torch.full_like(output.T, .1).contiguous().T.requires_grad_()
    got = torch.autograd.grad(output, targets, g, create_graph=True)
    directions = [torch.full_like(v.T, .2).contiguous().T for v in got]
    second = torch.autograd.grad(got, (*targets, g), directions, allow_unused=True)
    refs = [v.detach().requires_grad_(v.requires_grad) for v in values]
    rt = [v for v in refs if v.requires_grad]
    rg = g.detach().requires_grad_()
    want_out = spec.reference(*refs)
    want = torch.autograd.grad(want_out, rt, rg, create_graph=True)
    want_second = torch.autograd.grad(want, (*rt, rg), directions, allow_unused=True)
    result = [error(output, want_out), *(error(a, b) for a, b in zip(got, want))]
    for a, b in zip(second, want_second):
        if a is None and b is None:
            result.append(dict(passed=True, zero_unused=True))
        else:
            result.append(error(torch.zeros_like(b) if a is None else a,
                                torch.zeros_like(a) if b is None else b))
    checks.append(dict(name=f"uvu-partial-{mask}-noncontiguous-input-g-dd", passed=all(r["passed"] for r in result), checks=result))

for family in ("uvu", "radial"):
    for mutation in ("buffer-replace", "buffer-inplace", "metadata", "input-inplace", "tf32") + (
            ("weight-replace", "weight-inplace", "bias-replace", "bias-inplace") if family == "radial" else ()):
        for stage in ("between-forward-and-first", "between-first-and-second"):
            spec = Spec(family, 1, "full", 37)
            tp = spec.calls["public"] if family == "uvu" else spec.calls["public"].tp
            values = list(spec.logical)
            linear = None if family == "uvu" else spec.diagnostics["linear"]
            if linear is not None:
                linear.requires_grad_(True)
                values += [linear.weight, linear.bias]
            out = live_uvu(tp, *values) if family == "uvu" else streamed_live_radial(tp, linear, *values[:3])
            g = torch.ones_like(out).requires_grad_()
            first = torch.autograd.grad(out, values, g, create_graph=True) if stage.endswith("second") else None
            with torch.no_grad():
                if mutation == "buffer-replace":
                    tp._coefficients = tp._coefficients.clone()
                elif mutation == "buffer-inplace":
                    tp._coefficients.add_(0)
                elif mutation == "metadata":
                    tp.normalization = "norm"
                elif mutation == "input-inplace":
                    values[0].add_(0)
                elif mutation == "tf32":
                    torch.backends.cuda.matmul.allow_tf32 = True
                elif mutation == "weight-replace":
                    linear.weight = torch.nn.Parameter(linear.weight.detach().clone())
                elif mutation == "weight-inplace":
                    linear.weight.add_(0)
                elif mutation == "bias-replace":
                    linear.bias = torch.nn.Parameter(linear.bias.detach().clone())
                elif mutation == "bias-inplace":
                    linear.bias.add_(0)
            function = (lambda: torch.autograd.grad(out, values, g)) if first is None else (
                lambda: torch.autograd.grad(first, (*values, g), [torch.ones_like(v) for v in first], allow_unused=True))
            reject(f"{family}-{mutation}-{stage}", function)
            torch.backends.cuda.matmul.allow_tf32 = False

for rows in (0, 1, 127, 128, 129, 511, 512, 513):
    spec = Spec("uvu", 1, "full", rows)
    out = live_uvu(spec.calls["public"], *spec.logical)
    checked = error(out, spec.reference(*spec.logical)) if rows else dict(passed=out.shape == (0, spec.calls["public"].representation_dimension))
    checks.append(dict(name=f"dynamic-e-{rows}", **checked))

spec = Spec("uvu", 1, "full", 37)
tp = spec.calls["public"]
x, y, w = spec.logical
for policy in ("auto", "native"):
    checks.append(dict(name=f"inherited-fallback-{policy}", **error(bounded(tp, x, y, w, policy=policy), tp(x, y, w))))
reject("inherited-required-outside-domain", lambda: bounded(tp, x, y, w, policy="required"))
reject("live-lazy-negative", lambda: live_uvu(tp, torch._neg_view(x), y, w))
reject("live-wrong-width", lambda: live_uvu(tp, x[:, :-1], y, w))
reject("live-double-dtype", lambda: live_uvu(tp, x.double(), y, w))
reject("live-cpu", lambda: live_uvu(tp, x.cpu(), y, w))
result = dict(job=os.environ["SLURM_JOB_ID"], passed=all(c["passed"] for c in checks), checks=checks,
              tf32=False, cuda_graph=False, route="explicit-live-kernels-and-inherited-fallback", whole_model_pass=False)
destination = ROOT / "results" / os.environ["SLURM_JOB_ID"]
destination.mkdir(parents=True, exist_ok=True)
(destination / "GUARDS.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
if not result["passed"]:
    raise SystemExit(2)
