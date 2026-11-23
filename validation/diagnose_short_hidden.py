"""Isolate FP32 long-K association for the two remaining L8 short-page cases."""
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
from candidate.frozen_state import TPPlan
from candidate.live_uvu import weight_adjoint
from candidate.ordered64 import scheduled
from ictc_o2_hp import ReferenceUvuTensorProduct

configure()
configurations = ("-8x1b", "-16x1b", "16x1b@128", "16x1b@256", "8x1b@128", "8x1b@256",
                  "8x1b", "16x1b", "4x1b@128", "32x1b@128", "32x1b@256", "16x1a@128")
records = []
for seed in (20261003, 20261004):
    spec = Spec("radial", 8, "full", 37, seed, 1.)
    tp, linear = spec.calls["public"].tp, spec.diagnostics["linear"]
    linear.requires_grad_(True)
    x, y, h = spec.logical
    values = [x, y, h, linear.weight, linear.bias]
    generator = torch.Generator(device="cuda").manual_seed(20261121)
    g = (torch.randn(37, tp.representation_dimension, device="cuda", generator=generator) * .1).requires_grad_()
    generator.manual_seed(20261123)
    directions = tuple(torch.randn(v.shape, device="cuda", generator=generator) * .1 for v in values)
    a, b, _, d_weight, _ = directions
    oracle = ReferenceUvuTensorProduct(64, 1, 8, parity="full", internal_weights=False, shared_weights=False).cuda().float()
    reference = lambda x, y, h, w, bias: oracle(x, y, F.linear(h, w, bias))
    _, first, second = derived(reference, values, g, directions)
    with torch.no_grad():
        plan = TPPlan.capture(tp)
        gw = weight_adjoint(plan, x, y, g)
        p = weight_adjoint(plan, a, y, g) + weight_adjoint(plan, x, b, g)
        native_first = gw @ linear.weight
        native_second = p @ linear.weight + gw @ d_weight
        record = dict(seed=seed, input_sha256=spec.metadata["input_sha256"],
                      canonical_weights_sha256=spec.metadata["canonical_weights_sha256"],
                      native_matrix_controls=dict(first=error(native_first, first[2]), second=error(native_second, second[2])),
                      profiles=[])
        for configuration in configurations:
            got_first = scheduled(gw, linear.weight, configuration)
            got_second = scheduled(p, linear.weight, configuration) + scheduled(gw, d_weight, configuration)
            checks = dict(first=error(got_first, first[2]), second=error(got_second, second[2]),
                          matrix_first=error(got_first, native_first), matrix_second=error(got_second, native_second))
            record["profiles"].append(dict(configuration=configuration, checks=checks,
                                            passed=checks["first"]["passed"] and checks["second"]["passed"]))
        records.append(record)
        print(seed, [(r["configuration"], r["passed"]) for r in record["profiles"]], flush=True)
passing = [c for c in configurations if all(next(r for r in case["profiles"] if r["configuration"] == c)["passed"] for case in records)]
result = dict(job=os.environ["SLURM_JOB_ID"], scope="diagnostic only; not an automatic profile promotion",
              candidate_dtype="float32", original_gate_unchanged=True, tf32=False, cuda_graph=False,
              passing_both_seeds=passing, cases=records)
destination = ROOT / "results" / os.environ["SLURM_JOB_ID"]
destination.mkdir(parents=True, exist_ok=True)
(destination / "SHORT-HIDDEN-PROFILES.json").write_text(json.dumps(result, indent=2))
print(json.dumps(dict(passing_both_seeds=passing)), flush=True)
