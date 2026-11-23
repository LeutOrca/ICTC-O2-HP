"""Measure the unchanged 17a71d7 capability boundary; no performance claims."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
assert os.environ.get("SLURM_JOB_ID"), "GPU validation must run in Slurm"
os.environ.update(O2_BOUNDED_UVU="1", O2_RADIAL_ADJOINT="auto", O2_DIAGNOSTIC="1")

import torch
from benchmark import configure, error, qualify
from models import Spec
from candidate.bounded_uvu import bounded, streamed, BoundedUVU
from candidate.reduction_policy import choose_uvu, choose_radial


def expect_error(label, fn, rows):
    try:
        fn()
    except (RuntimeError, ValueError, TypeError) as exc:
        rows.append(dict(name=label, guard_passed=True, error=type(exc).__name__, message=str(exc)[:800]))
    else:
        rows.append(dict(name=label, guard_passed=False, message="Expected refusal was absent"))


def guards():
    from ictc_o2_hp import UvuTensorProduct
    tp = UvuTensorProduct(64, 1, 1, parity="full", internal_weights=False, shared_weights=False).cuda()
    generator = torch.Generator(device="cuda").manual_seed(20261003)
    x, y, w = [(torch.randn(37, width, generator=generator, device="cuda") * .1).requires_grad_()
               for width in (tp.input1_dimension, tp.input2_dimension, tp.weight_numel)]
    results = []
    for mode in ("auto", "native"):
        decision = choose_uvu(tp, x, y, w, mode)
        if decision.route != "native":
            raise RuntimeError("Out-of-domain input selected a custom route")
        value = bounded(tp, x, y, w, policy=mode)
        expected = tp(x, y, w)
        checked = error(value, expected)
        if not checked["passed"]:
            raise RuntimeError("Native fallback changed the frozen equation")
        results.append(dict(name=mode, route=decision.metadata(), checks=checked))
    expect_error("required-outside-domain", lambda: bounded(tp, x, y, w, policy="required"), results)
    for name, operands in (
        ("wrong-width", (x[:, :-1], y, w)),
        ("wrong-batch", (x, y[:-1], w)),
        ("wrong-dtype", (x.double(), y, w)),
        ("cpu-input", (x.cpu(), y, w)),
        ("lazy-negative", (torch._neg_view(x), y, w)),
    ):
        expect_error(name, lambda values=operands: bounded(tp, *values), results)
    noncontiguous = [value.detach().T.contiguous().T.requires_grad_() for value in (x, y, w)]
    checked = error(bounded(tp, *noncontiguous), tp(x, y, w))
    if not checked["passed"]:
        raise RuntimeError("Layout canonicalization changed values")
    results.append(dict(name="noncontiguous", checks=checked))
    # Isolate the state guard with a small raw kernel call. This is not an
    # extension of the public dispatch's qualified performance domain.
    output = BoundedUVU.apply(x, y, w, tp)
    tp._coefficients = tp._coefficients.clone()
    expect_error("tp-buffer-replacement", lambda: torch.autograd.grad(output.sum(), (x, y, w)), results)
    if not all(row.get("guard_passed", row.get("checks", {}).get("passed", True)) for row in results):
        raise RuntimeError("At least one safety guard failed")
    return results


def inspect_case(family, order, stage):
    spec = Spec(family, order, "full", 16384)
    role = "public-repair"
    route = spec.metadata["uvu_decision" if family == "uvu" else "radial_streamed_decision"]
    if route["route"] != "ordered":
        raise RuntimeError("The original H100 white-list route did not activate: " + str(route))
    inputs = spec.encoded(role)
    output = spec.calls[role](*inputs)
    generator = torch.Generator(device="cuda").manual_seed(20261121)
    cotangent = (torch.randn(output.shape, generator=generator, device="cuda") * .1).requires_grad_()
    gradients = torch.autograd.grad(output, inputs, cotangent, create_graph=True)
    torch.cuda.synchronize()
    result = dict(family=family, order=order, route=route, input_sha256=spec.metadata["input_sha256"],
                  canonical_weights_sha256=spec.metadata.get("canonical_weights_sha256"),
                  input_vjp_constructed=True, full_parameter_vjp=family == "uvu")
    try:
        scalar = sum(g.square().sum() for g in gradients)
        torch.autograd.grad(scalar, (*inputs, cotangent))
        result["double_backward"] = "unexpected-supported"
    except RuntimeError as exc:
        result.update(double_backward="unsupported", double_backward_error=str(exc)[:1200])
    if family == "radial":
        linear = spec.diagnostics["linear"]
        linear.requires_grad_(True)
        try:
            spec.calls[role](*inputs)
        except (ValueError, RuntimeError) as exc:
            result.update(full_parameter_vjp=False, trainable_affine_error=str(exc)[:1200])
        else:
            result["trainable_affine_error"] = "Expected frozen-parameter refusal was absent"
        linear.requires_grad_(False)
    if stage == "qualify":
        # Reuse the original gate and its original 512-row check windows.
        result["original_fp32_input_gate"] = qualify(spec, role, "forward_vjp", 4096)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("build", "qualify"))
    args = parser.parse_args()
    configure()
    if torch.is_autocast_enabled() or torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("Forbidden precision policy")
    result = dict(stage=args.stage, job=os.environ["SLURM_JOB_ID"], base="17a71d7",
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                  tf32=False, cuda_graph=False, amp=False, cases=[])
    try:
        result["guards"] = guards()
        for family, order in (("uvu", 6), ("radial", 4)):
            result["cases"].append(inspect_case(family, order, args.stage))
        result["execution_completed"] = True
        result["g1_all_passed"] = False  # expected missing second derivatives and live radial VJP
        result["capability_gap_confirmed"] = all(row["double_backward"] == "unsupported" for row in result["cases"])
    except Exception as exc:
        result.update(execution_completed=False, g1_all_passed=False, error=str(exc), traceback=traceback.format_exc())
    path = ROOT / "results" / os.environ["SLURM_JOB_ID"] / (args.stage + "-original.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key not in ("guards", "cases")}), flush=True)
    print("receipt", path, flush=True)
    if not result["execution_completed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
