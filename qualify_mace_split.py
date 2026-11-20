"""Check contiguous split-K FP32 accumulation through MACE E/F, unchanged gates."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, os.environ["HPX_MACE_ENTRY"])
import qualify_force as prior
from matrix_kernels import OwnMatrixKernels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prior.bench.configure()
    torch.use_deterministic_algorithms(True)
    model = prior.setup("receiver-csr-partition-graph")
    fixed = prior.inputs_for(prior.bench.load_geometry(prior.bench.CASES["b8"]), model)
    for _ in range(3):
        prior.graph_evaluate(model, fixed)
    reference = prior.graph_evaluate(model, fixed)
    records = []
    for backend in ("split2", "split4", "split8"):
        mode = OwnMatrixKernels(backend)
        with mode:
            result = prior.graph_evaluate(model, fixed)
        numeric = prior.bench.errors(result, reference)
        records.append(dict(backend=backend, numerics=numeric,
                            passed=all(row["passed"] for row in numeric.values()),
                            matrix_calls=dict(mode.calls)))
        args.output.write_text(json.dumps(records, indent=2))
        print(json.dumps(records[-1]), flush=True)


if __name__ == "__main__":
    main()
