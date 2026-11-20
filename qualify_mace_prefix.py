"""Strict native MACE E/F gate for frozen prefix elimination and CUDA Graph."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, os.environ["HPX_MACE_ENTRY"])
import qualify_force as prior
from mace_frozen_prefix import install, receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=tuple(prior.bench.CASES))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prior.bench.configure()
    torch.use_deterministic_algorithms(True)
    native = prior.setup("native-graph")
    geometry = prior.bench.load_geometry(prior.bench.CASES[args.case])
    fixed = prior.inputs_for(geometry, native)
    positions = fixed["positions"].clone()
    positions[0, 0] += .005
    frames = {"fixed": fixed, "e2e": prior.inputs_for(geometry, native),
              "perturbed": dict(fixed, positions=positions)}
    reference = {label: prior.graph_evaluate(native, data) for label, data in frames.items()}
    model = prior.setup("receiver-csr-partition-graph")
    installed = install(model)
    errors = {}
    for label, data in frames.items():
        actual = prior.graph_evaluate(model, data)
        errors[label] = prior.bench.errors(actual, reference[label])
    report = dict(case=args.case, source=Path("provenance/commit.txt").read_text().strip(),
                  base="78231f3482fc9999857cc29e1cf5c220e73aa435", installed=installed,
                  checkpoint_sha256="d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a",
                  numerics=errors, prefix=receipt(model), torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(), tf32=False, amp=False,
                  passed=all(value["passed"] for row in errors.values() for value in row.values()))
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Frozen prefix failed the unchanged native MACE FP32 gate")
    replay = prior.replay_for(model, fixed)
    captures = {label: prior.bench.errors(replay.call(data), reference[label])
                for label, data in frames.items()}
    report.update(capture=captures, prefix=receipt(model), cuda_graph=True,
                  passed=all(value["passed"] for row in captures.values() for value in row.values()))
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Frozen prefix CUDA Graph failed the native MACE FP32 gate")


if __name__ == "__main__":
    main()
