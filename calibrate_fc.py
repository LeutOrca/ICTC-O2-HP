"""Identify native FP32 FMA accumulation schedules from real SO2 FC operands."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, os.environ["HPX_V3_ENTRY"])
import whole_network as native
from frozen_so2 import _pack_input
from so2_ordered import product
import triton


def error(actual, expected):
    actual, expected = actual.detach().double(), expected.detach().double()
    delta = actual - expected
    return dict(max_abs=float(delta.abs().max()),
                relative_l2=float(delta.norm() / expected.norm().clamp_min(1e-30)))


def flatten(value):
    if value.ndim == 2:
        return value
    assert value.shape[1] == 2
    if value.is_contiguous():
        return value.reshape(-1, value.shape[-1])
    packed = value.new_empty((value.shape[0] * 2, value.shape[-1]))
    _pack_input[(triton.cdiv(packed.numel(), 256),)](
        value, packed, value.shape[0], value.shape[-1], *value.stride(), 256)
    return packed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed-index", type=int, default=0)
    args = parser.parse_args()
    native.strict(torch)
    native.checked_checkpoint()
    torch.set_num_threads(4)
    net = native.Network("native", "cuda")
    net.freeze_ema()
    net.model.requires_grad_(False)
    raw = net.batch(net.manifest["suites"]["mptrj-b8"])
    fixed, topology = net.topology(raw)
    selected = {name: module for name, module in net.model.named_modules()
                if isinstance(module, torch.nn.Linear) and ".so2_linear_" in name
                and name.split(".")[1] in ("0", "3", "6")}
    assert len(selected) == 18
    captures, handles = {}, []
    def hook(name):
        def forward(module, inputs, output):
            row = dict(input=inputs[0].detach().clone(), output=output.detach().clone())
            captures[name] = row
            inputs[0].register_hook(lambda value: row.update(gradient=value.detach().clone()))
            output.register_hook(lambda value: row.update(cotangent=value.detach().clone()))
        return forward
    def run():
        torch.manual_seed(native.SEEDS[args.seed_index])
        return net.forward(fixed, True)
    for _ in range(3):
        run()
    for name, module in selected.items():
        handles.append(module.register_forward_hook(hook(name)))
    reference = run()
    for handle in handles:
        handle.remove()
    repeat = native.gate(torch, run(), reference, raw.natoms)
    assert all(value["max_abs"] == 0 for value in repeat.values())
    configs = ["triton", "1x1l"]
    configs += [f"2x{stripe}l" for stripe in (1, 2, 4, 8, 16)]
    configs += [f"4x{stripe}{join}" for stripe in (1, 2, 4, 8, 16) for join in "lab"]
    configs += [f"8x{stripe}{join}" for stripe in (1, 4, 8) for join in "lab"]
    configs += [f"-{groups}x1{join}" for groups in (2, 4, 8) for join in "lab"]
    report = dict(source=Path("provenance/commit.txt").read_text().strip(),
                  checkpoint_sha256=native.CHECKPOINT_SHA, topology=topology,
                  seed=native.SEEDS[args.seed_index], gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                  tf32=False, amp=False, records={})
    for name, module in selected.items():
        row = captures[name]
        left, derivative = flatten(row["input"]), flatten(row["cotangent"])
        results = {}
        for config in configs:
            actual = product(left, module.weight.t(), config, bias=module.bias)
            gradient = product(derivative, module.weight, config)
            results[config] = dict(forward=error(actual, row["output"].reshape_as(actual)),
                                   backward=error(gradient, row["gradient"].reshape_as(gradient)))
        report["records"][name] = dict(input_shape=list(row["input"].shape),
                                      weight_shape=list(module.weight.shape), results=results)
        args.output.write_text(json.dumps(report, indent=2))
        for direction in ("forward", "backward"):
            best = min(results, key=lambda config: results[config][direction]["relative_l2"])
            print(json.dumps(dict(layer=name, direction=direction, best=best,
                                  **results[best][direction])), flush=True)


if __name__ == "__main__":
    main()
