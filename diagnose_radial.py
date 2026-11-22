"""Frozen-input FP32 stage isolation. Diagnostics are never timing qualifications."""
import argparse
import json
import os
from pathlib import Path

import torch
from benchmark import configure, error, prepare
from models import Spec

ROOT = Path(__file__).resolve().parent


def windows(actual, expected, first):
    return [dict(start=first+i, **error(actual[i:i+512], expected[i:i+512]))
            for i in range(0, len(actual), 512)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, choices=range(4, 9), default=4)
    parser.add_argument("--role", choices=("eqx-o2", "public-unfused"), default="eqx-o2")
    parser.add_argument("--seed", type=int, choices=(20261003, 20261004), default=20261003)
    parser.add_argument("--reference-start", type=int, default=8192)
    parser.add_argument("--reference-rows", type=int, choices=(512, 4096), default=4096)
    parser.add_argument("--matrix-probe", action="store_true")
    args = parser.parse_args()
    configure()
    if "H100" not in torch.cuda.get_device_name() or torch.__version__ != "2.11.0+cu128":
        raise RuntimeError("This diagnosis requires the frozen H100 / torch2.11.0+cu128 runtime")
    os.environ["O2_DIAGNOSTIC"] = "1"
    assert args.reference_start % 4096 == 0
    assert 0 <= args.reference_start < args.reference_start + args.reference_rows <= 16384
    record = dict(purpose="stage-diagnosis-only; not a numerical or performance acceptance",
                  dtype="float32", tf32=False, amp=False, cuda_graph=False,
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  gpu_uuid=str(torch.cuda.get_device_properties(0).uuid), job=os.getenv("SLURM_JOB_ID"),
                  seed=args.seed, scale=1., role=args.role, order=args.order, rows=16384,
                  reference_start=args.reference_start, reference_rows=args.reference_rows,
                  parity="full", channels=64, hidden=64)
    spec = Spec("radial", args.order, "full", 16384, args.seed, 1.)
    record.update(spec.metadata)
    linear, mapped, signs = (spec.diagnostics[k] for k in ("linear", "mapped", "signs"))
    capture = []
    def retain(module, inputs, output):
        if output.requires_grad:
            output.retain_grad()
            capture.append(output)
    handle = (mapped if args.role == "eqx-o2" else linear).register_forward_hook(retain)
    try:
        inputs, cotangent, logical_cotangent = prepare(spec, args.role)
        # prepare runs a no-grad forward; capture only the actual adjoint graph.
        capture.clear()
        output = spec.calls[args.role](*inputs)
        native_grads = torch.autograd.grad(output, inputs, cotangent)
        actual_grads = [value.detach() for value in spec.decode_grad(args.role, native_grads)]
        weight_grad = capture[-1].grad.detach()
        if args.role == "eqx-o2":
            weight_grad = weight_grad * signs
        actual = spec.decode[args.role](output.detach())
    finally:
        handle.remove()
    first, last = args.reference_start, args.reference_start + args.reference_rows
    logical = [value[first:last].detach().requires_grad_() for value in spec.logical]
    capture.clear()
    handle = linear.register_forward_hook(retain)
    try:
        expected = spec.reference(*logical)
        expected_grads = torch.autograd.grad(expected, logical, logical_cotangent[first:last])
        reference_weight_grad = capture[-1].grad.detach()
    finally:
        handle.remove()
    record["end_to_end_checks"] = [windows(actual[first:last], expected, first)] + [
        windows(got[first:last], ref, first) for got, ref in zip(actual_grads, expected_grads)]
    local_weight_grad = weight_grad[first:last].contiguous()
    # Same cotangent and frozen affine weight; change only matrix execution shape
    # to distinguish TP weight-adjoint error from long-K affine accumulation.
    local_hidden_grad = local_weight_grad @ linear.weight
    reference_replay = reference_weight_grad @ linear.weight
    record["isolation"] = dict(
        weight_adjoint=windows(local_weight_grad, reference_weight_grad, first),
        affine_shape_effect=windows(actual_grads[2][first:last], local_hidden_grad, first),
        tp_adjoint_effect=windows(local_hidden_grad, expected_grads[2], first),
        reference_linear_replay=windows(reference_replay, expected_grads[2], first),
        strides=dict(weight=list(linear.weight.stride()), actual_gw=list(local_weight_grad.stride()),
                     reference_gw=list(reference_weight_grad.stride())))
    if args.matrix_probe:
        from candidate.ordered_matrix import scheduled
        from matrix_kernels import matrix
        configurations = ["2x1b", "4x1b", "8x1b", "16x1b", "32x1b",
            "4x8b", "8x8b", "16x8b", "16x1a", "16x1l",
            "8x1b@128", "8x1b@256", "16x1b@128", "16x1b@256", "32x1b@256"]
        probes = {}
        for name in ["generic-ieee"] + configurations:
            result = (matrix(local_weight_grad, linear.weight) if name == "generic-ieee" else
                      scheduled(local_weight_grad, linear.weight, name))
            checks = windows(result, expected_grads[2], first)
            probes[name] = dict(passed=all(r["passed"] for r in checks), checks=checks)
            print("matrix-probe", name, probes[name]["passed"],
                  max(r["max_abs"] for r in checks), flush=True)
            del result
        record["own_matrix_probes"] = probes
    torch.cuda.synchronize()
    suffix = "-matrix-probe" if args.matrix_probe else ""
    output_path = ROOT / "results" / f"radial-isolation-L{args.order}-{args.role}-{args.seed}-{first}-{args.reference_rows}{suffix}.json"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_text(json.dumps(record, indent=2))
    print(json.dumps(dict(receipt=str(output_path), isolated_components=list(record["isolation"]),
                         input_sha256=record["input_sha256"], weights_sha256=record["canonical_weights_sha256"])), flush=True)


if __name__ == "__main__":
    main()
