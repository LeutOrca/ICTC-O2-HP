"""Isolate native/own primal and input-adjoint drift at each official SO2 layer."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from types import MethodType

import torch

sys.path.insert(0, os.environ["HPX_V3_ENTRY"])
import whole_network as native
from optimizations import install as install_previous
from frozen_so2 import FrozenWeightCache
from so2_layer import frozen_layer
from fp32_gate import gate
from so2_ordered import ordered_layer, PROFILES


def tensors(value):
    return tuple(value) if isinstance(value, tuple) else (value,)


def error(actual, expected):
    value, reference = actual.detach().float(), expected.detach().float()
    delta = value - reference
    relative = float(delta.norm() / reference.norm().clamp_min(1e-30))
    violations = int((delta.abs() > native.ATOL + native.RTOL * reference.abs()).sum())
    return dict(max_abs=float(delta.abs().max()), relative_l2=relative,
                violations=violations, passed=violations == 0 and relative <= native.REL_L2)


def fingerprint(value):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


class Hybrid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, native_forward, own_forward, own_primal, own_adjoint):
        ctx.native_forward, ctx.own_forward = native_forward, own_forward
        ctx.own_adjoint = own_adjoint
        ctx.save_for_backward(inputs)
        return (own_forward if own_primal else native_forward)(inputs)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *derivatives):
        inputs, = ctx.saved_tensors
        with torch.enable_grad():
            value = inputs.detach().requires_grad_()
            outputs = tensors((ctx.own_forward if ctx.own_adjoint else ctx.native_forward)(value))
            derivative, = torch.autograd.grad(outputs, value, derivatives)
        return derivative, None, None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--suite", default="mptrj-b8")
    parser.add_argument("--individual", action="store_true")
    parser.add_argument("--implementation", choices=("packed", "ordered", "calibrated", *PROFILES), default="packed")
    parser.add_argument("--seed-index", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    native.strict(torch)
    native.checked_checkpoint()
    torch.set_num_threads(4)
    network = native.Network("native", "cuda")
    network.freeze_ema()
    network.model.requires_grad_(False)
    raw = network.batch(network.manifest["suites"][args.suite])
    fixed, topology = network.topology(raw)
    layers = {name: module for name, module in network.model.named_modules()
              if module.__class__.__name__ == "SO2Linear"}
    assert len(layers) == 14
    original = {name: module.forward for name, module in layers.items()}
    caches = {name: tuple(FrozenWeightCache() for _ in module.so2_m_linear)
              for name, module in layers.items()}
    captures, handles = {}, []

    def own_layer(inputs, module, cache):
        if args.implementation == "packed":
            return frozen_layer(inputs, module, cache, "triton")
        configuration = "triton" if args.implementation == "ordered" else args.implementation
        return ordered_layer(inputs, module, configuration, configuration, configuration)

    def hook(name):
        def forward(module, inputs, outputs):
            row = dict(input=inputs[0].detach().clone(),
                       outputs=tuple(value.detach().clone() for value in tensors(outputs)),
                       cotangents=[None] * len(tensors(outputs)))
            captures[name] = row
            inputs[0].register_hook(lambda derivative: row.update(input_gradient=derivative.detach().clone()))
            for index, value in enumerate(tensors(outputs)):
                value.register_hook(lambda derivative, index=index: row["cotangents"].__setitem__(
                    index, derivative.detach().clone()))
        return forward

    def run():
        torch.manual_seed(native.SEEDS[args.seed_index])
        return network.forward(fixed, True)

    for _ in range(3):
        run()
    for name, module in layers.items():
        handles.append(module.register_forward_hook(hook(name)))
    reference = run()
    for handle in handles:
        handle.remove()
    repeat = gate(torch, run(), reference, raw.natoms)
    assert all(row["passed"] and row["max_abs"] == 0 for row in repeat.values())
    report = dict(source=Path("provenance/commit.txt").read_text().strip(),
                  baseline="c591b16", checkpoint_sha256=native.CHECKPOINT_SHA,
                  suite=args.suite, seed=native.SEEDS[args.seed_index], topology=topology,
                  implementation=args.implementation,
                  input_positions_sha256=fingerprint(raw.pos), gpu=torch.cuda.get_device_name(),
                  gpu_uuid=str(torch.cuda.get_device_properties(0).uuid), torch=torch.__version__,
                  tf32=False, amp=False, cuda_graph=False, native_repeat=repeat, metrics_dtype="torch.float32",
                  local={}, whole={}, individual={})

    def save():
        args.output.write_text(json.dumps(report, indent=2))

    for name, module in layers.items():
        row = captures[name]
        inputs = row["input"].requires_grad_()
        cotangents = tuple(value if value is not None else torch.zeros_like(output)
                           for value, output in zip(row["cotangents"], row["outputs"]))
        actual = tensors(own_layer(inputs, module, caches[name]))
        gradient, = torch.autograd.grad(actual, inputs, cotangents)
        native_outputs = tensors(original[name](inputs))
        native_gradient, = torch.autograd.grad(native_outputs, inputs, cotangents)
        assert error(native_gradient, row["input_gradient"])["max_abs"] == 0
        forward_parts, backward_parts = {}, {}
        start = 0
        for order in range(module.mmax + 1):
            count = module.lmax + 1 if order == 0 else 2 * (module.lmax + 1 - order)
            forward_parts[str(order)] = error(actual[0][:, start:start+count], row["outputs"][0][:, start:start+count])
            backward_parts[str(order)] = error(gradient[:, start:start+count], native_gradient[:, start:start+count])
            start += count
        report["local"][name] = dict(input_shape=list(inputs.shape), input_strides=list(row["input"].stride()),
                                     input_sha256=fingerprint(inputs),
                                     forward=[error(value, expected) for value, expected in zip(actual, row["outputs"])],
                                     input_vjp=error(gradient, native_gradient),
                                     forward_by_order=forward_parts, input_vjp_by_order=backward_parts)
        save()
        print(json.dumps(dict(layer=name, **report["local"][name])), flush=True)
        del actual, gradient, native_outputs, native_gradient
    del captures
    gc.collect()

    install_previous("gather-radial", network.model)
    control = gate(torch, run(), reference, raw.natoms)
    assert all(row["passed"] for row in control.values())
    report["gather_control"] = control

    def select(names, own_primal, own_adjoint):
        for name, module in layers.items():
            module.forward = original[name]
            if name not in names:
                continue
            def own(value, module=module, cache=caches[name]):
                return own_layer(value, module, cache)
            def forward(layer, value, native_forward=original[name], own_forward=own):
                return Hybrid.apply(value, native_forward, own_forward, own_primal, own_adjoint)
            module.forward = MethodType(forward, module)

    for primal, adjoint in ((False, False), (False, True), (True, False), (True, True)):
        select(layers, primal, adjoint)
        errors = gate(torch, run(), reference, raw.natoms)
        key = f"primal_{'own' if primal else 'native'}-adjoint_{'own' if adjoint else 'native'}"
        report["whole"][key] = errors
        save()
        print(json.dumps(dict(mode=key, numerics=errors)), flush=True)
    if args.individual:
        for name in layers:
            report["individual"][name] = {}
            for primal, adjoint in ((False, True), (True, False), (True, True)):
                select((name,), primal, adjoint)
                errors = gate(torch, run(), reference, raw.natoms)
                key = f"primal_{'own' if primal else 'native'}-adjoint_{'own' if adjoint else 'native'}"
                report["individual"][name][key] = errors
                save()
                print(json.dumps(dict(layer=name, mode=key, numerics=errors)), flush=True)


if __name__ == "__main__":
    main()
