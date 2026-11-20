"""Measure the unchanged FP32 force gate against independent double evaluation."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, os.environ["HPX_MACE_ENTRY"])
import qualify_force as prior
from matrix_kernels import OwnMatrixKernels


def errors(actual, reference):
    output = {}
    for name in prior.METRICS:
        value = actual[name].double()
        target = reference[name].double()
        difference = value - target
        output[name] = dict(max_abs=float(difference.abs().max()),
                            relative_l2=float(difference.norm() / target.norm().clamp_min(1e-30)))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prior.bench.configure()
    torch.use_deterministic_algorithms(True)
    model = prior.setup("receiver-csr-partition-graph")
    geometry = prior.bench.load_geometry(prior.bench.CASES["b8"])
    data = prior.inputs_for(geometry, model)
    for _ in range(3):
        prior.graph_evaluate(model, data)
    native = prior.graph_evaluate(model, data)
    variants = {}
    for backend in ("triton", "cuda2", "cuda4", "cuda8"):
        with OwnMatrixKernels(backend):
            variants[backend] = prior.graph_evaluate(model, data)
    double_model = prior.setup("native-graph").double()
    double_data = {name: value.double() if value.is_floating_point() else value
                   for name, value in data.items()}
    double = prior.bench.evaluate(double_model, double_data)
    report = dict(native_fp32_vs_double=errors(native, double),
                  candidates_vs_double={name: errors(value, double) for name, value in variants.items()},
                  candidates_vs_native={name: errors(value, native) for name, value in variants.items()},
                  checkpoint_unchanged=True, diagnostic_only=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
