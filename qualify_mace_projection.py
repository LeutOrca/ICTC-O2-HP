"""Qualify one-hot projection replacement while preserving native channel GEMMs."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, os.environ["HPX_MACE_ENTRY"])
import qualify_force as prior
from mace_species_projection import install, project
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
    reference = {label: prior.graph_evaluate(model, data) for label, data in frames.items()}
    attrs = fixed["node_attrs"] * .0093691461124482678
    weight = model.interactions[0].skip_tp.weight[:1458176].view(128, 89, 128)
    expected = torch.tensordot(attrs, weight, dims=([1], [1]))
    actual = project(attrs, weight)
    primitive = strict_compare(actual, expected)
    changed = install(model)
    report = dict(case=args.case, primitive=primitive, changed=changed,
                  source=Path("provenance/commit.txt").read_text().strip(),
                  base="78231f3482fc9999857cc29e1cf5c220e73aa435",
                  passed=primitive["passed"])
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("One-hot projection failed the native FP32 gate")
    errors = {label: prior.bench.errors(prior.graph_evaluate(model, data), reference[label])
              for label, data in frames.items()}
    report.update(numerics=errors, passed=all(value["passed"] for row in errors.values()
                                             for value in row.values()))
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Native TP projection replacement failed whole E/F")
    replay = prior.replay_for(model, fixed)
    capture = {label: prior.bench.errors(replay.call(data), reference[label])
               for label, data in frames.items()}
    report.update(capture=capture, passed=all(value["passed"] for row in capture.values()
                                             for value in row.values()), cuda_graph=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("One-hot projection Graph failed whole E/F")


if __name__ == "__main__":
    main()
