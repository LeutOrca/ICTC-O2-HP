"""Qualify the complete original V3 SO2Linear against own frozen kernels."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import torch

from frozen_so2 import FrozenWeightCache
from matrix_kernels import OwnMatrixKernels
from qualify_matrix import compare
from so2_layer import frozen_layer


def official():
    path = Path(os.environ["HPX_V3_UPSTREAM"]) / "experimental/models/equiformer_v3/so2_ops.py"
    spec = importlib.util.spec_from_file_location("official_v3_so2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SO2Linear


def tuple_output(value):
    return value if isinstance(value, tuple) else (value,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(20261120)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = []
    constructor = official()
    for edges in (37, 4096):
        for head in (None, 31):
            module = constructor(128, 128, 4, 2, extra_m0_out_channels=head).cuda().requires_grad_(False)
            inputs = (torch.randn(edges, 19, 128, device="cuda") * .1).requires_grad_()
            caches = tuple(FrozenWeightCache() for _ in range(2))
            reference = tuple_output(module(inputs))
            cotangents = tuple(torch.randn_like(value) for value in reference)
            expected, = torch.autograd.grad(reference, inputs, cotangents)
            with OwnMatrixKernels():
                actual = tuple_output(frozen_layer(inputs, module, caches, "triton"))
                gradient, = torch.autograd.grad(actual, inputs, cotangents)
            results = [compare(got, ref) for got, ref in zip(actual, reference)]
            results.append(compare(gradient, expected))
            record = dict(edges=edges, extra_scalar_channels=head, results=results)
            records.append(record)
            args.output.write_text(json.dumps(dict(records=records), indent=2))
            print(json.dumps(record), flush=True)
            if not all(result["passed"] for result in results):
                raise RuntimeError(record)
    args.output.write_text(json.dumps(dict(records=records, passed=True), indent=2))


if __name__ == "__main__":
    main()
