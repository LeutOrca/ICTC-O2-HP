"""Screen own IEEE FP32 tile schedules on the frozen SO(2) workload."""
import argparse
import gc
import json
from pathlib import Path

import torch
import triton

from frozen_so2 import FrozenWeightCache, _pack_input
from matrix_kernels import _matrix
from qualify_matrix import compare, time_ms
from qualify_so2 import reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(20261120)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    edges, width, outputs = 160000, 128, 128
    # Retain the original sliced, non-contiguous row stride.
    backing = torch.randn(edges, 4, width, device="cuda") * .1
    source = backing[:, :2]
    weights = torch.randn(2 * outputs, width, device="cuda") * .1
    weight = FrozenWeightCache().prepare(weights).T
    expected = reference(source, weights)
    baseline = time_ms(lambda: reference(source, weights))
    records = []
    for bm, bn, bk, warps in ((32, 32, 32, 4), (32, 64, 32, 4),
                              (64, 64, 32, 4), (64, 128, 32, 4),
                              (32, 128, 32, 4), (64, 32, 32, 4),
                              (32, 64, 16, 4), (64, 64, 16, 4),
                              (64, 64, 32, 8), (64, 128, 32, 8)):
        def call():
            packed = source.new_empty((edges, 2 * width))
            output = source.new_empty((edges, 2 * outputs))
            _pack_input[(triton.cdiv(packed.numel(), 256),)](
                source, packed, edges, width, *source.stride(), 256,
            )
            compiled = _matrix[(triton.cdiv(edges, bm), triton.cdiv(2 * outputs, bn))](
                packed, weight, output, output, edges, 2 * outputs, 2 * width,
                0, *packed.stride(), 0, *weight.stride(), 0, 0, 0, 0, *output.stride(),
                False, 1., 1., bm, bn, bk, num_warps=warps,
            )
            if ".tf32" in compiled.asm["ptx"]:
                raise RuntimeError("Unexpected TF32")
            return output.view(edges, 2, outputs)
        actual = call()
        numerical = compare(actual, expected)
        if not numerical["passed"]:
            raise RuntimeError(numerical)
        latency = time_ms(call)
        record = dict(tile=(bm, bn, bk, warps), event_ms=latency,
                      baseline_ms=baseline, speedup=baseline / latency, numerics=numerical)
        records.append(record)
        args.output.write_text(json.dumps(dict(records=records), indent=2))
        print(json.dumps(record), flush=True)
        del actual
        gc.collect()
    args.output.write_text(json.dumps(dict(records=records, passed=True), indent=2))


if __name__ == "__main__":
    main()
