"""Strict original-TP and whole-network gates for the own first scalar TP."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, os.environ["HPX_MACE_ENTRY"])
import qualify_force as prior
from mace_scalar_tp import ScalarTP, install
from qualify_large_layer import strict_compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=tuple(prior.bench.CASES))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prior.bench.configure()
    torch.use_deterministic_algorithms(True)
    model = prior.setup("receiver-csr-partition-graph")
    geometry = prior.bench.load_geometry(prior.bench.CASES[args.case])
    fixed = prior.inputs_for(geometry, model)
    positions = fixed["positions"].clone()
    positions[0, 0] += .005
    frames = {"fixed": fixed, "e2e": prior.inputs_for(geometry, model),
              "perturbed": dict(fixed, positions=positions)}
    operands = []
    handle = model.interactions[0].conv_tp.register_forward_pre_hook(
        lambda module, values: operands.append(tuple(value.detach() for value in values)))
    reference = {label: prior.graph_evaluate(model, data) for label, data in frames.items()}
    handle.remove()
    product = model.interactions[0].conv_tp
    x, h, weights = operands[0]
    h, weights = h.requires_grad_(), weights.requires_grad_()
    expected = product(x, h, weights)
    torch.manual_seed(20261120)
    incoming = torch.randn_like(expected)
    ref_grad = torch.autograd.grad(expected, (h, weights), incoming)
    own = ScalarTP(product)
    actual = own(x, h, weights)
    own_grad = torch.autograd.grad(actual, (h, weights), incoming)
    numeric = dict(forward=strict_compare(actual, expected),
                   harmonics_vjp=strict_compare(own_grad[0], ref_grad[0]),
                   radial_vjp=strict_compare(own_grad[1], ref_grad[1]))
    report = dict(case=args.case, primitive=numeric,
                  source=Path("provenance/commit.txt").read_text().strip(),
                  base="78231f3482fc9999857cc29e1cf5c220e73aa435",
                  passed=all(row["passed"] for row in numeric.values()))
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Own scalar TP failed the strict native forward/input-VJP gate")
    install(model)
    errors = {label: prior.bench.errors(prior.graph_evaluate(model, data), reference[label])
              for label, data in frames.items()}
    report.update(numerics=errors, passed=all(value["passed"] for row in errors.values()
                                             for value in row.values()))
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Own scalar TP failed the unchanged whole-network FP32 E/F gate")
    replay = prior.replay_for(model, fixed)
    capture = {label: prior.bench.errors(replay.call(data), reference[label])
               for label, data in frames.items()}
    report.update(capture=capture, passed=all(value["passed"] for row in capture.values()
                                             for value in row.values()), cuda_graph=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Own scalar TP Graph failed FP32")


if __name__ == "__main__":
    main()
