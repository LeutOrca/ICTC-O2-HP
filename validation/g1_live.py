"""Original FP32 gates for full VJP and explicit own-kernel double backward."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
assert os.environ.get("SLURM_JOB_ID")
os.environ["O2_DIAGNOSTIC"] = "1"
os.environ.pop("O2_BOUNDED_UVU", None)
os.environ.pop("O2_RADIAL_ADJOINT", None)
import torch
import torch.nn.functional as F
from benchmark import configure, error
from models import Spec, digest
from candidate.live_uvu import live_uvu, live_radial
from candidate.streamed_live_radial import streamed_live_radial, hidden_configuration, PAGE
from candidate.reduction_policy import COVERAGE


def derived(function, values, g, directions):
    output = function(*values)
    first = torch.autograd.grad(output, values, g, create_graph=True)
    second = torch.autograd.grad(first, (*values, g), directions, allow_unused=True)
    return output, first, second


def one_case(family, order, rows, seed, scale, build=False, streamed=False):
    spec = Spec(family, order, "full", rows, seed, scale)
    tp = spec.calls["public"] if family == "uvu" else spec.calls["public"].tp
    if family == "uvu":
        values = list(spec.logical)
        actual = lambda x, y, w: live_uvu(tp, x, y, w)
        reference = spec.reference
    else:
        linear = spec.diagnostics["linear"]
        linear.requires_grad_(True)
        values = [*spec.logical, linear.weight, linear.bias]
        operation = streamed_live_radial if streamed else live_radial
        actual = lambda x, y, h, weight, bias: operation(tp, linear, x, y, h)
        from ictc_o2_hp import ReferenceUvuTensorProduct
        oracle = ReferenceUvuTensorProduct(64, 1, order, parity="full", internal_weights=False, shared_weights=False).cuda().float()
        reference = lambda x, y, h, weight, bias: oracle(x, y, F.linear(h, weight, bias))
    generator = torch.Generator(device="cuda").manual_seed(20261121)
    g = (torch.randn(rows, tp.representation_dimension, device="cuda", generator=generator) * .1).requires_grad_()
    generator.manual_seed(20261123)
    directions = tuple(torch.randn(value.shape, device="cuda", generator=generator) * .1 for value in values)
    output, first, second = derived(actual, values, g, directions)
    torch.cuda.synchronize()
    record = dict(family=family, order=order, rows=rows, seed=seed, scale=scale,
                  input_sha256=spec.metadata["input_sha256"], canonical_weights_sha256=spec.metadata.get("canonical_weights_sha256"),
                  full_vjp_roles=["left", "angular", "external_weights"] if family == "uvu" else
                      ["left", "angular", "radial_hidden", "affine_weight", "affine_bias"],
                  second_roles="all first-input roles plus incoming output cotangent",
                  route="own64-uvu-first-and-second" if family == "uvu" else
                      ("own64-streamed-live-radial-first-and-second" if streamed else "native-unchanged-affine+own64-uvu-first-and-second"),
                  reference_replay_in_candidate=False, vendor_tp_in_candidate=False,
                  dtype="float32", tf32=False, amp=False, cuda_graph=False)
    if family == "radial" and streamed:
        sizes = sorted({min(PAGE, rows - start) for start in range(0, rows, PAGE)})
        base = COVERAGE.get(order, (None, None))[1]
        record["hidden_adjoint_profiles"] = [dict(rows=size, K=tp.weight_numel,
            configuration=hidden_configuration(size, tp.weight_numel, base)) for size in sizes]
    if build:
        record.update(passed=True, built_only=True, numerical_pass=False)
        return record
    checks = []
    if family == "uvu":
        # Every edge is independent, so preserve original 512-row windows.
        for start in range(0, rows, 512):
            end = min(start + 512, rows)
            reference_values = [value[start:end].detach().requires_grad_() for value in values]
            reference_g = g[start:end].detach().requires_grad_()
            expected, df, ddf = derived(reference, reference_values, reference_g,
                                       tuple(value[start:end] for value in directions))
            got = [output[start:end], *(value[start:end] for value in first),
                   *(None if value is None else value[start:end] for value in second)]
            want = [expected, *df, *ddf]
            for index, (a, b) in enumerate(zip(got, want)):
                if a is None and b is None:
                    result = dict(passed=True, zero_unused=True)
                else:
                    if a is None:
                        a = torch.zeros_like(b)
                    if b is None:
                        b = torch.zeros_like(a)
                    result = error(a, b)
                checks.append(dict(window=[start, end], output_index=index, **result))
            del reference_values, reference_g, expected, df, ddf, got, want
    else:
        # Preserve the inherited 512-row forward/input-reference boundary.
        # The newly introduced shared-parameter roles have no inherited
        # qualification: define and seal their ascending FP32 page sum before
        # the first full-load test, and compare only after all pages contribute.
        shared = {index: torch.zeros_like(value) for index, value in
                  ((4, first[3]), (5, first[4]), (9, second[3]), (10, second[4])) if value is not None}
        for start in range(0, rows, 512):
            end = min(start + 512, rows)
            reference_values = [value[start:end].detach().requires_grad_() for value in values[:3]] + [
                value.detach().requires_grad_() for value in values[3:]]
            reference_g = g[start:end].detach().requires_grad_()
            ref_directions = tuple(value[start:end] for value in directions[:3]) + directions[3:]
            expected, df, ddf = derived(reference, reference_values, reference_g, ref_directions)
            for index, a, b in ((0, output[start:end], expected),
                                (1, first[0][start:end], df[0]), (2, first[1][start:end], df[1]),
                                (3, first[2][start:end], df[2]),
                                (6, None if second[0] is None else second[0][start:end], ddf[0]),
                                (7, None if second[1] is None else second[1][start:end], ddf[1]),
                                (8, None if second[2] is None else second[2][start:end], ddf[2]),
                                (11, None if second[5] is None else second[5][start:end], ddf[5])):
                if a is None and b is None:
                    checked = dict(passed=True, zero_unused=True)
                else:
                    if a is None:
                        a = torch.zeros_like(b)
                    if b is None:
                        b = torch.zeros_like(a)
                    checked = error(a, b)
                checks.append(dict(window=[start, end], output_index=index, **checked))
            for index, value in ((4, df[3]), (5, df[4]), (9, ddf[3]), (10, ddf[4])):
                if value is not None:
                    if index not in shared:
                        shared[index] = torch.zeros_like(value)
                    shared[index].add_(value.detach())
            del reference_values, reference_g, expected, df, ddf
        for index, actual_shared in ((4, first[3]), (5, first[4]), (9, second[3]), (10, second[4])):
            if actual_shared is None and index not in shared:
                continue
            a = actual_shared if actual_shared is not None else torch.zeros_like(shared[index])
            b = shared.get(index, torch.zeros_like(a))
            for start in range(0, len(a), 512):
                end = min(start + 512, len(a))
                checks.append(dict(window=[start, end], output_index=index,
                                   shared_all_edges=True, **error(a[start:end], b[start:end])))
        record["shared_parameter_reference"] = "complete ascending FP32 sum of independent 512-edge reference contributions"
    record.update(passed=all(check["passed"] for check in checks), built_only=False, checks=checks,
                  original_gate="benchmark.error byte-identical to 17a71d7",
                  gradient_directions_sha256=digest(directions))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("build", "small", "full"))
    parser.add_argument("--family", choices=("uvu", "radial"))
    parser.add_argument("--order", type=int)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--streamed", action="store_true")
    args = parser.parse_args()
    configure()
    destination = ROOT / "results" / os.environ["SLURM_JOB_ID"]
    destination.mkdir(parents=True, exist_ok=True)
    cases = []
    families = [args.family] if args.family else ["uvu", "radial"]
    orders = [args.order] if args.order else ([1, 6] if args.stage == "build" else list(range(1, 9)))
    seeds = [20261003] if args.stage == "build" else [20261003, 20261004]
    scales = [.1] if args.stage == "build" else [.1, 1.]
    for family in families:
        for order in orders:
            for seed in seeds:
                for scale in scales:
                    try:
                        rows = args.rows or (16384 if args.stage == "full" else 37)
                        result = one_case(family, order, rows, seed, scale, args.stage == "build", args.streamed)
                    except Exception as exc:
                        result = dict(family=family, order=order, seed=seed, scale=scale, passed=False,
                                      error=str(exc), traceback=traceback.format_exc())
                    result.update(job=os.environ["SLURM_JOB_ID"], gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
                    cases.append(result)
                    path = destination / f"{args.stage}-{family}-L{order}-{seed}-{scale}.json"
                    path.write_text(json.dumps(result, indent=2))
                    print(family, order, seed, scale, result["passed"], result.get("error", ""), flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
    summary = dict(passed=all(result["passed"] for result in cases), stage=args.stage,
                   job=os.environ["SLURM_JOB_ID"], cases=len(cases),
                   source_scope="experimental live_uvu and live_radial only; not whole-model qualification")
    (destination / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
