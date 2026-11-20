"""Compare blocked warp-K accumulation with native MACE matrix operands.

This diagnostic returns the native matrix output to the network throughout.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, os.environ["HPX_MACE_ENTRY"])
import qualify_force as prior
from cuda_matrix import compiled_matrix


class Inspect(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.records = []

    def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if operation._schema.name not in ("aten::mm", "aten::bmm"):
            return operation(*args, **kwargs)
        native = operation(*args, **kwargs)
        left, right = args
        norm = native.double().norm().clamp_min(1e-30)
        variants = {}
        for groups in (2, 4):
            for stripe in (1, 2, 4, 8, 16, 32):
                own = compiled_matrix(left, right, None, 1., 1., groups=groups, stripe=stripe)
                difference = own.double() - native.double()
                variants[f"g{groups}-s{stripe}"] = dict(
                    relative_l2=float(difference.norm() / norm),
                    exact=bool(torch.equal(own, native)))
        self.records.append(dict(index=len(self.records), operation=operation._schema.name,
                                 left_shape=list(left.shape), right_shape=list(right.shape),
                                 left_stride=list(left.stride()), right_stride=list(right.stride()),
                                 variants=variants))
        return native


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prior.bench.configure()
    torch.use_deterministic_algorithms(True)
    model = prior.setup("receiver-csr-partition-graph")
    data = prior.inputs_for(prior.bench.load_geometry(prior.bench.CASES["b8"]), model)
    for _ in range(3):
        prior.graph_evaluate(model, data)
    inspect = Inspect()
    with inspect:
        prior.graph_evaluate(model, data)
    args.output.write_text(json.dumps(dict(diagnostic_only=True, records=inspect.records), indent=2))
    for row in inspect.records:
        best = min(row["variants"], key=lambda name: row["variants"][name]["relative_l2"])
        print(json.dumps(dict(index=row["index"], left=row["left_shape"], right=row["right_shape"],
                              best=best, error=row["variants"][best]["relative_l2"],
                              exact=[name for name, value in row["variants"].items() if value["exact"]])),
              flush=True)


if __name__ == "__main__":
    main()
