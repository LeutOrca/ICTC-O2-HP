"""Frozen SO(2) FP32 qualification and paired operator measurements."""
import argparse
import gc
import json
from pathlib import Path

import torch

from frozen_so2 import FrozenWeightCache, frozen_so2
from qualify_matrix import compare, time_ms


def reference(inputs, weight):
    values = torch.nn.functional.linear(inputs, weight)
    width = weight.shape[0] // 2
    real, imaginary = values.split(width, dim=-1)
    return torch.stack((real[:, 0] - imaginary[:, 1],
                        real[:, 1] + imaginary[:, 0]), dim=1)


def run(backend, target):
    torch.manual_seed(20261120)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = []
    for edges, width, outputs in ((0, 17, 13), (37, 65, 93), (257, 128, 64),
                                   (16384, 128, 128), (160000, 128, 128)):
        backing = torch.randn(edges, 4, width, device="cuda") * .1
        inputs = backing[:, :2, :].requires_grad_()
        weight = torch.randn(2 * outputs, width, device="cuda") * .1
        cache = FrozenWeightCache()
        expected = reference(inputs, weight)
        actual = frozen_so2(inputs, weight, cache, backend)
        cotangent = torch.randn_like(expected)
        ref_gradient, = torch.autograd.grad(expected, inputs, cotangent)
        gradient, = torch.autograd.grad(actual, inputs, cotangent)
        forward = compare(actual, expected)
        backward = compare(gradient, ref_gradient)
        row = dict(shape=(edges, width, outputs), forward=forward, input_adjoint=backward,
                   input_stride=inputs.stride())
        if not forward["passed"] or not backward["passed"]:
            target.write_text(json.dumps(dict(backend=backend, records=records + [row]), indent=2))
            raise RuntimeError(row)
        if edges:
            source = inputs.detach()
            row["native_event_ms"] = time_ms(lambda: reference(source, weight))
            row["own_event_ms"] = time_ms(lambda: frozen_so2(source, weight, cache, backend))
            row["forward_speedup"] = row["native_event_ms"] / row["own_event_ms"]
            baseline_ptr = cache.packed.data_ptr()
            frozen_so2(source, weight, cache, backend)
            row["cache_reused"] = cache.packed.data_ptr() == baseline_ptr
            weight.add_(.01)
            updated = frozen_so2(source, weight, cache, backend)
            row["weight_update"] = compare(updated, reference(source, weight))
            if not row["cache_reused"] or not row["weight_update"]["passed"]:
                raise RuntimeError(row)
        records.append(row)
        target.write_text(json.dumps(dict(backend=backend, records=records), indent=2))
        del backing, inputs, weight, cache, expected, actual, cotangent, ref_gradient, gradient
        gc.collect()
        torch.cuda.empty_cache()
    target.write_text(json.dumps(dict(backend=backend, gpu=torch.cuda.get_device_name(),
                                     records=records, passed=True), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("triton", "cuda"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.backend, args.output)
