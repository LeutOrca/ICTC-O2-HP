"""Isolate scalar-TP angular versus radial adjoints, using native diagnostics."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, os.environ["HPX_MACE_ENTRY"])
import qualify_force as prior
from mace_scalar_tp import ScalarTP
from matrix_kernels import matrix


class Inspect(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.rows = []

    def __torch_dispatch__(self, op, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        value = op(*args, **kwargs)
        if op._schema.name != "aten::bmm":
            return value
        variants = {}
        for backend in ("triton", "cuda", "cuda2", "cuda4", "cuda8"):
            own = matrix(*args, backend=backend)
            variants[backend] = dict(exact=torch.equal(own, value),
                                     max_abs=float((own - value).abs().max()))
        self.rows.append(dict(left=list(args[0].shape), right=list(args[1].shape),
                              strides=[list(arg.stride()) for arg in args],
                              output_stride=list(value.stride()), variants=variants))
        return value


class _Family(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h, weights, reference, angular, radial):
        ctx.save_for_backward(x, h, weights)
        ctx.reference, ctx.angular, ctx.radial = reference, angular, radial
        return reference(x, h, weights)

    @staticmethod
    def backward(ctx, derivative):
        x, h, weights = ctx.saved_tensors
        with torch.enable_grad():
            h = h.detach().requires_grad_()
            weights = weights.detach().requires_grad_()
            expected = ctx.reference(x.detach(), h, weights)
            native = torch.autograd.grad(expected, (h, weights), derivative)
            actual = ScalarTP(ctx.reference)(x.detach(), h, weights)
            own = torch.autograd.grad(actual, (h, weights), derivative)
        return None, own[0] if ctx.angular else native[0], own[1] if ctx.radial else native[1], None, None, None


class Family(torch.nn.Module):
    def __init__(self, reference, angular, radial):
        super().__init__()
        self.reference, self.angular, self.radial = reference, angular, radial

    def forward(self, x, h, weights):
        return _Family.apply(x, h, weights, self.reference, self.angular, self.radial)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prior.bench.configure()
    torch.use_deterministic_algorithms(True)
    model = prior.setup("receiver-csr-partition-graph")
    fixed = prior.inputs_for(prior.bench.load_geometry(prior.bench.CASES["b8"]), model)
    operands = []
    tp = model.interactions[0].conv_tp
    handle = tp.register_forward_pre_hook(lambda module, values: operands.append(values))
    reference = prior.graph_evaluate(model, fixed)
    handle.remove()
    x, h, weights = (value.detach() for value in operands[0])
    h, weights = h.requires_grad_(), weights.requires_grad_()
    trace = Inspect()
    torch.manual_seed(20261120)
    with trace:
        value = tp(x, h, weights)
        torch.autograd.grad(value, (h, weights), torch.randn_like(value))
    rows = []
    for angular, radial in ((False, False), (True, False), (False, True), (True, True)):
        model.interactions[0].conv_tp = Family(tp, angular, radial)
        actual = prior.graph_evaluate(model, fixed)
        rows.append(dict(own_angular=angular, own_radial=radial,
                         numerics=prior.bench.errors(actual, reference)))
        print(json.dumps(rows[-1]), flush=True)
    args.output.write_text(json.dumps(dict(diagnostic_only=True, families=rows,
                                           native_batched_matmuls=trace.rows), indent=2))


if __name__ == "__main__":
    main()
