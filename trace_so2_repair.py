"""Trace only the replaced SO2 primals and input adjoints on real operands."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, os.environ["HPX_V3_ENTRY"])
import whole_network as native
from diagnose_v3_so2 import tensors
from so2_ordered import ordered_layer, PROFILES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--implementation", choices=tuple(PROFILES), required=True)
    args = parser.parse_args()
    native.strict(torch)
    native.checked_checkpoint()
    torch.set_num_threads(4)
    net = native.Network("native", "cuda")
    net.freeze_ema()
    net.model.requires_grad_(False)
    raw = net.batch(net.manifest["suites"]["mptrj-b8"])
    fixed, topology = net.topology(raw)
    layers = {name: module for name, module in net.model.named_modules()
              if module.__class__.__name__ == "SO2Linear"}
    captures, handles = {}, []
    def hook(name):
        def forward(module, inputs, outputs):
            row = dict(input=inputs[0].detach().clone(), cotangents=[None] * len(tensors(outputs)))
            captures[name] = row
            for index, value in enumerate(tensors(outputs)):
                value.register_hook(lambda gradient, index=index: row["cotangents"].__setitem__(
                    index, gradient.detach().clone()))
        return forward
    for name, module in layers.items():
        handles.append(module.register_forward_hook(hook(name)))
    torch.manual_seed(native.SEEDS[0])
    net.forward(fixed, True)
    for handle in handles:
        handle.remove()
    def call():
        for name, module in layers.items():
            row = captures[name]
            inputs = row["input"].detach().requires_grad_()
            with torch.profiler.record_function("own-SO2-forward/" + name):
                outputs = tensors(ordered_layer(inputs, module, *([args.implementation] * 3)))
            assert all(output.dtype == torch.float32 for output in outputs)
            with torch.profiler.record_function("own-SO2-input-vjp/" + name):
                gradients = tuple(torch.zeros_like(output) if value is None else value
                                  for output, value in zip(outputs, row["cotangents"]))
                torch.autograd.grad(outputs, inputs, gradients)
    call()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as trace:
        call()
        torch.cuda.synchronize()
    kernels = sorted({event.name for event in trace.events()
                      if event.device_type == torch.autograd.DeviceType.CUDA})
    forbidden = [name for name in kernels if any(token in name.lower()
                 for token in ("cublas", "gemm", "gemv", "cutlass"))]
    report = dict(source=Path("provenance/commit.txt").read_text().strip(),
                  implementation=args.implementation, checkpoint_sha256=native.CHECKPOINT_SHA,
                  topology=topology, layers=list(layers), kernels=kernels,
                  forbidden_library_matrix=forbidden, metrics_dtype="torch.float32",
                  scope="all 14 own SO2 forward and input VJP, real native operands",
                  passed=not forbidden)
    args.output.write_text(json.dumps(report, indent=2))
    trace.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
    print(json.dumps(report), flush=True)
    assert not forbidden


if __name__ == "__main__":
    main()
