"""Supplemental independent small FP64 oracle, judged with original FP32 gates."""
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
from candidate.live_uvu import live_uvu
from candidate.streamed_live_radial import streamed_live_radial
from ictc_o2_hp import ReferenceUvuTensorProduct

configure()
records = []
for family in ("uvu", "radial"):
    for order in (1, 4, 8):
        for seed in (20261003, 20261004):
            for scale in (.1, 1.):
                spec = Spec(family, order, "full", 37, seed, scale)
                tp = spec.calls["public"] if family == "uvu" else spec.calls["public"].tp
                values = list(spec.logical)
                if family == "radial":
                    linear = spec.diagnostics["linear"]
                    linear.requires_grad_(True)
                    values += [linear.weight, linear.bias]
                    actual = lambda x, y, h, w, b: streamed_live_radial(tp, linear, x, y, h)
                else:
                    actual = lambda x, y, w: live_uvu(tp, x, y, w)
                oracle = ReferenceUvuTensorProduct(64, 1, order, parity="full",
                            internal_weights=False, shared_weights=False).cuda().double()
                reference = oracle if family == "uvu" else lambda x, y, h, w, b: oracle(x, y, F.linear(h, w, b))
                generator = torch.Generator(device="cuda").manual_seed(20261121)
                g = (torch.randn(37, tp.representation_dimension, device="cuda", generator=generator) * .1).requires_grad_()
                generator.manual_seed(20261123)
                directions = tuple(torch.randn(v.shape, device="cuda", generator=generator) * .1 for v in values)
                out, first, second = derived(actual, values, g, directions)
                refs = [v.detach().double().requires_grad_() for v in values]
                expected, df, ddf = derived(reference, refs, g.detach().double().requires_grad_(), tuple(d.double() for d in directions))
                checks = []
                for index, (a, b) in enumerate(zip((out, *first, *second), (expected, *df, *ddf))):
                    if a is None and b is None:
                        checked = dict(passed=True, zero_unused=True)
                    else:
                        a = torch.zeros_like(b, dtype=torch.float32) if a is None else a
                        b = torch.zeros_like(a) if b is None else b.float()
                        checked = error(a, b)
                    checks.append(dict(role=index, **checked))
                record = dict(family=family, order=order, seed=seed, scale=scale,
                              passed=all(c["passed"] for c in checks), checks=checks)
                if not record["passed"]:
                    oracle32 = ReferenceUvuTensorProduct(64, 1, order, parity="full",
                                internal_weights=False, shared_weights=False).cuda().float()
                    reference32 = oracle32 if family == "uvu" else lambda x, y, h, w, b: oracle32(x, y, F.linear(h, w, b))
                    native_values = [v.detach().requires_grad_() for v in values]
                    native_out, native_first, native_second = derived(reference32, native_values,
                                            g.detach().requires_grad_(), directions)
                    control = []
                    for index, (a, b) in enumerate(zip((native_out, *native_first, *native_second), (expected, *df, *ddf))):
                        if a is None and b is None:
                            checked = dict(passed=True, zero_unused=True)
                        else:
                            a = torch.zeros_like(b, dtype=torch.float32) if a is None else a
                            b = torch.zeros_like(a) if b is None else b.float()
                            checked = error(a, b)
                        control.append(dict(role=index, **checked))
                    record["native_fp32_control_against_same_fp64"] = control
                    record["native_fp32_control_passed"] = all(c["passed"] for c in control)
                records.append(record)
                print(family, order, seed, scale, record["passed"], flush=True)
result = dict(job=os.environ["SLURM_JOB_ID"], passed=all(c["passed"] for c in records),
              oracle="independent FP64 ReferenceUvu + F.linear; output/derivatives rounded once to FP32 for inherited error()",
              candidate_arithmetic="FP32", original_gate_unchanged=True, cases=records,
              replaces_inherited_fp32_reference=False)
destination = ROOT / "results" / os.environ["SLURM_JOB_ID"]
destination.mkdir(parents=True, exist_ok=True)
(destination / "FP64-ORACLE.json").write_text(json.dumps(result, indent=2))
print(json.dumps(dict(passed=result["passed"], cases=len(records))), flush=True)
if not result["passed"]:
    raise SystemExit(2)
