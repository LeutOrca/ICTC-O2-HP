"""Strict fused TP/CSR VJP and whole-network E/F/Graph qualification."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, os.environ["HPX_MACE_ENTRY"])
import qualify_force as prior
from mace_receiver_tp import ScalarReceiver, evaluate, install
from qualify_large_layer import strict_compare
from receiver_csr import reduce, schedule


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
    operands, senders = [], []
    block = model.interactions[0]
    handles = [block.conv_tp.register_forward_pre_hook(lambda module, values: operands.append(values)),
               block.linear_up.register_forward_hook(lambda module, values, output: senders.append(output))]
    reference = {label: prior.graph_evaluate(model, data) for label, data in frames.items()}
    for handle in handles:
        handle.remove()
    x = senders[0].detach()
    h, weights = (value.detach().requires_grad_() for value in operands[0][1:])
    receiver = fixed["edge_index"][1]
    rows = schedule(receiver, x.shape[0])
    expected = reduce(block.conv_tp(x[fixed["edge_index"][0]], h, weights), receiver, *rows)
    torch.manual_seed(20261120)
    derivative = torch.randn_like(expected)
    ref_grad = torch.autograd.grad(expected, (h, weights), derivative)
    product = ScalarReceiver(block.conv_tp)
    product.row_schedule = rows
    actual = product(x, h, weights, fixed["edge_index"])
    own_grad = torch.autograd.grad(actual, (h, weights), derivative)
    primitive = dict(forward=strict_compare(actual, expected),
                     harmonics_vjp=strict_compare(own_grad[0], ref_grad[0]),
                     radial_vjp=strict_compare(own_grad[1], ref_grad[1]))
    report = dict(case=args.case, source=Path("provenance/commit.txt").read_text().strip(),
                  primitive=primitive, passed=all(value["passed"] for value in primitive.values()))
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Fused receiver TP failed native FP32 output or VJP")
    install(model)
    errors = {label: prior.bench.errors(evaluate(model, data), reference[label])
              for label, data in frames.items()}
    report.update(numerics=errors, passed=all(value["passed"] for row in errors.values()
                                             for value in row.values()))
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Fused receiver TP failed native whole E/F")
    replay = prior.Replay(evaluate, model, fixed)
    capture = {label: prior.bench.errors(replay.call(data), reference[label])
               for label, data in frames.items()}
    report.update(capture=capture, passed=all(value["passed"] for row in capture.values()
                                             for value in row.values()), cuda_graph=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Fused receiver TP Graph failed native whole E/F")


if __name__ == "__main__":
    main()
