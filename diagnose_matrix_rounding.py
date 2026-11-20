"""Native MACE matrix operands, own FP32 errors and native CUDA kernel trace.

Native matrix outputs feed this diagnostic forward. It is not a candidate run.
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
from matrix_kernels import matrix


class Inspect(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.records = []

    def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = operation._schema.name
        if name not in ("aten::mm", "aten::bmm"):
            return operation(*args, **kwargs)
        native = operation(*args, **kwargs)
        left, right = args
        norm = native.double().norm().clamp_min(1e-30)
        variants = {}
        for backend in ("triton", "cuda2", "cuda4", "cuda8"):
            own = matrix(left, right, backend=backend)
            diff = own.double() - native.double()
            variants[backend] = dict(relative_l2=float(diff.norm() / norm),
                                     max_abs=float(diff.abs().max()),
                                     equal_fraction=float((own == native).float().mean()))
        self.records.append(dict(index=len(self.records), operation=name,
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
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as trace:
        prior.graph_evaluate(model, data)
        torch.cuda.synchronize()
    kernels = sorted({event.name for event in trace.events()
                      if event.device_type == torch.autograd.DeviceType.CUDA})
    report = dict(diagnostic_only=True, records=inspect.records, native_cuda_kernels=kernels,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    args.output.write_text(json.dumps(report, indent=2))
    trace.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
