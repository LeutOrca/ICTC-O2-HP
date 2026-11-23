"""Domain refusal must precede a raw launch or bypassed module semantics."""
import json
import os
from pathlib import Path
import sys

assert os.environ.get("SLURM_JOB_ID")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["O2_DIAGNOSTIC"] = "1"
import torch
from benchmark import configure
from models import Spec
from validation.g1_live import one_case
from candidate.live_uvu import live_uvu
from candidate.streamed_live_radial import streamed_live_radial

configure()
records = [one_case("uvu", 1, 37, 20261003, .1),
           one_case("radial", 1, 37, 20261003, .1, streamed=True)]


class TaggedTensor(torch.Tensor):
    pass


def fresh():
    spec = Spec("radial", 1, "full", 37)
    return spec, spec.calls["public"].tp, spec.diagnostics["linear"]


for name in ("tp-forward", "tp-forward-hook", "tp-backward-hook", "tp-compiled-call",
             "normalization", "angular-cutoff", "linear-forward", "linear-forward-hook",
             "linear-backward-hook", "linear-compiled-call", "weight-layout", "bias-layout",
             "input-subclass", "buffer-subclass", "global-forward-hook"):
    spec, tp, linear = fresh()
    values = list(spec.logical)
    handle = None
    if name == "tp-forward":
        tp.forward = lambda *args: args[0]
    elif name == "tp-forward-hook":
        handle = tp.register_forward_hook(lambda *args: None)
    elif name == "tp-backward-hook":
        handle = tp.register_full_backward_hook(lambda *args: None)
    elif name == "tp-compiled-call":
        tp._compiled_call_impl = lambda *args: args[0]
    elif name == "normalization":
        tp.normalization = "norm"
    elif name == "angular-cutoff":
        tp.m_max_in2 = 0
    elif name == "linear-forward":
        linear.forward = lambda v: v
    elif name == "linear-forward-hook":
        handle = linear.register_forward_hook(lambda *args: None)
    elif name == "linear-backward-hook":
        handle = linear.register_full_backward_hook(lambda *args: None)
    elif name == "linear-compiled-call":
        linear._compiled_call_impl = lambda v: v
    elif name == "weight-layout":
        linear.weight = torch.nn.Parameter(linear.weight.detach().T.contiguous().T)
    elif name == "bias-layout":
        pair = linear.bias.detach()[:, None].expand(-1, 2).contiguous()
        linear.bias = torch.nn.Parameter(pair[:, 0])
    elif name == "input-subclass":
        values[0] = torch.Tensor._make_subclass(TaggedTensor, values[0].detach(), False)
    elif name == "buffer-subclass":
        tp._coefficients = torch.Tensor._make_subclass(TaggedTensor, tp._coefficients, False)
    elif name == "global-forward-hook":
        handle = torch.nn.modules.module.register_module_forward_hook(lambda *args: None)
    try:
        streamed_live_radial(tp, linear, *values)
    except (RuntimeError, ValueError) as exc:
        message = str(exc)
        records.append(dict(name=name, passed="Raw live" in message, message=message))
    else:
        records.append(dict(name=name, passed=False, message="Raw entry bypassed unsupported semantics"))
    finally:
        if handle is not None:
            handle.remove()
result = dict(job=os.environ["SLURM_JOB_ID"], passed=all(r["passed"] for r in records),
              cases=records, tf32=False, cuda_graph=False,
              scope="implementation-domain safety plus small full-VJP/second integration; no full-load promotion")
destination = ROOT / "results" / os.environ["SLURM_JOB_ID"]
destination.mkdir(parents=True, exist_ok=True)
(destination / "LIVE-POLICY.json").write_text(json.dumps(result, indent=2))
print(json.dumps(dict(passed=result["passed"], cases=len(records))), flush=True)
if not result["passed"]:
    raise SystemExit(2)
