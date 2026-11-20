"""Ablate native matrix families to attribute strict FP32 MACE force drift."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, os.environ["HPX_MACE_ENTRY"])
import qualify_force as prior
from matrix_kernels import OwnMatrixKernels


class FamilyMode(OwnMatrixKernels):
    def __init__(self, family):
        super().__init__()
        self.family = family
        self.native = Counter()
        self.shapes = Counter()

    def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
        name = operation._schema.name
        if name in ("aten::mm", "aten::bmm", "aten::addmm", "aten::baddbmm"):
            self.shapes[str((name, tuple(tuple(value.shape) for value in args if isinstance(value, torch.Tensor))))] += 1
            batched = name in ("aten::bmm", "aten::baddbmm")
            selected = self.family == "all" or self.family == ("bmm" if batched else "mm")
            if not selected:
                self.native[name] += 1
                return operation(*args, **(kwargs or {}))
        return super().__torch_dispatch__(operation, types, args, kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prior.bench.configure()
    torch.use_deterministic_algorithms(True)
    model = prior.setup("receiver-csr-partition-graph")
    geometry = prior.bench.load_geometry(prior.bench.CASES["b8"])
    data = prior.inputs_for(geometry, model)
    cold = prior.graph_evaluate(model, data)
    for _ in range(3):
        prior.graph_evaluate(model, data)
    expected = prior.graph_evaluate(model, data)
    records = []
    cold_difference = prior.bench.errors(cold, expected)
    print(json.dumps(dict(cold_to_warm=cold_difference)), flush=True)
    for family in ("none", "mm", "bmm", "all"):
        mode = FamilyMode(family)
        with mode:
            actual = prior.graph_evaluate(model, data)
        record = dict(family=family, diagnostic_only=True,
                      numerics=prior.bench.errors(actual, expected),
                      own_calls=dict(mode.calls), native_calls=dict(mode.native),
                      matrix_shapes=dict(mode.shapes))
        records.append(record)
        args.output.write_text(json.dumps(records, indent=2))
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
