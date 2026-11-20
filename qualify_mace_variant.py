"""Compare each local candidate directly with native eager and native Graph E/F."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, os.environ["HPX_MACE_ENTRY"])
import qualify_force as prior
from mace_variants import VARIANTS, apply, evaluator


def cpu(values):
    return {name: value.detach().cpu() for name, value in values.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=VARIANTS[2:])
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
    input_bitwise = None
    if args.variant == "final-input":
        from mace_static_inputs import CompositionInputs
        input_builder = CompositionInputs(geometry, native, fixed, prior.bench.get_neighborhood)
        changed = [(position.copy(), atoms.copy(), cell.copy()) for position, atoms, cell in geometry]
        changed[0][0][0, 0] += .005
        native_rebuilt = prior.inputs_for(changed, native)
        own_rebuilt = input_builder(changed)
        own_original = input_builder(geometry)
        input_bitwise = {label: {name: bool(torch.equal(actual[name], expected[name]))
                                 for name in expected}
                         for label, actual, expected in (("original", own_original, fixed),
                                                        ("moved_rebuilt", own_rebuilt, native_rebuilt))}
        if not all(value for row in input_bitwise.values() for value in row.values()):
            raise RuntimeError(dict(input_bitwise=input_bitwise))
        frames["rebuilt_perturbed"] = native_rebuilt
        frames["cached_original"] = own_original
        frames["cached_perturbed"] = own_rebuilt
    for _ in range(3):
        prior.graph_evaluate(native, fixed)
    reference = {name: cpu(prior.graph_evaluate(native, data)) for name, data in frames.items()}
    native_replay = prior.replay_for(native, fixed)
    native_capture = {name: prior.bench.errors(cpu(native_replay.call(data)), reference[name])
                      for name, data in frames.items()}
    if not all(value["passed"] for row in native_capture.values() for value in row.values()):
        raise RuntimeError(dict(native_capture=native_capture))
    del native_replay, native
    gc.collect()
    torch.cuda.empty_cache()
    model = prior.setup("receiver-csr-partition-graph")
    apply(model, args.variant)
    evaluate = evaluator(args.variant, prior)
    errors = {name: prior.bench.errors(cpu(evaluate(model, data)), reference[name])
              for name, data in frames.items()}
    report = dict(case=args.case, variant=args.variant, numerics=errors,
                  native_capture=native_capture,
                  source=Path("provenance/commit.txt").read_text().strip(),
                  base="78231f3482fc9999857cc29e1cf5c220e73aa435",
                  checkpoint_sha256="d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a",
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  input_bitwise=input_bitwise,
                  passed=all(value["passed"] for row in errors.values() for value in row.values()))
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Local candidate failed the unchanged native E/F gate")
    replay = prior.Replay(evaluate, model, fixed)
    capture = {name: prior.bench.errors(cpu(replay.call(data)), reference[name])
               for name, data in frames.items()}
    report.update(capture=capture, cuda_graph=True,
                  passed=all(value["passed"] for row in capture.values() for value in row.values()))
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Local candidate Graph failed the unchanged native E/F gate")


if __name__ == "__main__":
    main()
