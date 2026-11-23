"""Bounded live-parameter radial UVU with own first and mixed second adjoints.

Only x/y/hidden and the original affine parameters are saved. Generated path
weights are private to a 512-edge page and recomputed for differentiation.
Every matrix operation inside this candidate uses IEEE FP32 Triton kernels.
This remains experimental until all original FP32 gates have passed.
"""
import torch
from torch.autograd.function import once_differentiable

from .contracts import check_features, prepare_cotangent, tensor_contract
from .dense64 import bias_rows, mm, sum_rows
from .frozen_state import ForwardState
from .live_uvu import add_private, forward, input_adjoint, weight_adjoint
from .live_contracts import original_plan, original_linear, tensor_types
from .ordered64 import scheduled
from .reduction_policy import COVERAGE, assert_reduction_context, require_fp32_policy

PAGE = 512


def hidden_configuration(rows, columns, configuration):
    # Frozen shape-only profiles measured independently of input values.
    if (rows, columns) == (37, 55552):
        return "16x1b@128"
    if (rows, columns) == (512, 55552):
        return "-16x1b"
    return configuration


def hidden_adjoint(values, weight, configuration):
    # The reduction length is the TP path-weight width, even for a short
    # edge page. A small row count does not make that FP32 reduction short.
    # This profile extension is experimental until both row domains pass.
    # Select only frozen measured shapes, independently of seed or values.
    configuration = hidden_configuration(*values.shape, configuration)
    return scheduled(values, weight, configuration) if configuration else mm(values, weight)


def add_shared(destination, value):
    destination.add_(value)


class _RadialAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, h, weight, bias, g, state, needs):
        state.assert_current()
        plan = state.plan
        configuration = COVERAGE.get(state.tp.m_max, (None, None))[1]
        outputs = [torch.empty_like(value) if active else None
                   for value, active in zip((x, y, h, weight, bias), needs)]
        for index in (3, 4):
            if outputs[index] is not None:
                outputs[index].zero_()
        for start in range(0, len(x), PAGE):
            end = min(start + PAGE, len(x))
            xs, ys, hs, gs = x[start:end], y[start:end], h[start:end], g[start:end]
            if needs[0] or needs[1]:
                w = mm(hs, weight.T, bias)
                if needs[0]:
                    outputs[0][start:end] = input_adjoint(plan, xs, ys, w, gs, True)
                if needs[1]:
                    outputs[1][start:end] = input_adjoint(plan, xs, ys, w, gs, False)
                del w
            if needs[2] or needs[3] or needs[4]:
                gw = weight_adjoint(plan, xs, ys, gs)
                if needs[2]:
                    outputs[2][start:end] = hidden_adjoint(gw, weight, configuration)
                if needs[3]:
                    add_shared(outputs[3], mm(gw.T, hs))
                if needs[4]:
                    add_shared(outputs[4], sum_rows(gw))
                del gw
        ctx.save_for_backward(x, y, h, weight, bias, g, *plan.buffers)
        ctx.state, ctx.configuration = state, configuration
        ctx.set_materialize_grads(False)
        return tuple(outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, a, b, c, d_weight, d_bias):
        ctx.state.assert_current()
        x, y, h, weight, bias, g, *buffers = ctx.saved_tensors
        assert_reduction_context(x)
        plan = ctx.state.plan.with_buffers(buffers)
        directions = []
        for name, value, original in (("dd_left", a, x), ("dd_angular", b, y),
                                      ("dd_hidden", c, h), ("dd_affine_weight", d_weight, weight),
                                      ("dd_affine_bias", d_bias, bias)):
            if value is not None:
                tensor_contract(name, value, device=x.device, ndim=original.ndim)
                if value.shape != original.shape:
                    raise ValueError(name + " has the wrong shape")
                value = value.contiguous()
            directions.append(value)
        a, b, c, d_weight, d_bias = directions
        needs = ctx.needs_input_grad[:6]
        output = [torch.zeros_like(value) if need else None
                  for value, need in zip((x, y, h, weight, bias, g), needs)]
        for start in range(0, len(x), PAGE):
            end = min(start + PAGE, len(x))
            xs, ys, hs, gs = x[start:end], y[start:end], h[start:end], g[start:end]
            aa = None if a is None else a[start:end]
            bb = None if b is None else b[start:end]
            cc = None if c is None else c[start:end]
            # q = c A^T + h D^T + e is the cotangent of the TP weight VJP.
            q = None
            if cc is not None:
                q = mm(cc, weight.T)
            if d_weight is not None:
                q = add_private(q, mm(hs, d_weight.T))
            if d_bias is not None:
                q = add_private(q, bias_rows(d_bias, end - start))
            dx = dy = dg = p = None
            if aa is not None or bb is not None:
                w = mm(hs, weight.T, bias)
                if aa is not None:
                    if needs[1]:
                        dy = add_private(dy, input_adjoint(plan, aa, ys, w, gs, False))
                    p = add_private(p, weight_adjoint(plan, aa, ys, gs))
                    if needs[5]:
                        dg = add_private(dg, forward(plan, aa, ys, w))
                if bb is not None:
                    if needs[0]:
                        dx = add_private(dx, input_adjoint(plan, xs, bb, w, gs, True))
                    p = add_private(p, weight_adjoint(plan, xs, bb, gs))
                    if needs[5]:
                        dg = add_private(dg, forward(plan, xs, bb, w))
                del w
            if q is not None:
                if needs[0]:
                    dx = add_private(dx, input_adjoint(plan, xs, ys, q, gs, True))
                if needs[1]:
                    dy = add_private(dy, input_adjoint(plan, xs, ys, q, gs, False))
                if needs[5]:
                    dg = add_private(dg, forward(plan, xs, ys, q))
                del q
            for index, value in ((0, dx), (1, dy), (5, dg)):
                if output[index] is not None and value is not None:
                    output[index][start:end] = value
            if p is not None:
                if needs[2]:
                    output[2][start:end] = hidden_adjoint(p, weight, ctx.configuration)
                if needs[3]:
                    add_shared(output[3], mm(p.T, hs))
                if needs[4]:
                    add_shared(output[4], sum_rows(p))
                del p
            if (d_weight is not None and needs[2]) or (cc is not None and needs[3]):
                gw = weight_adjoint(plan, xs, ys, gs)
                if d_weight is not None and needs[2]:
                    output[2][start:end].add_(hidden_adjoint(gw, d_weight, ctx.configuration))
                if cc is not None and needs[3]:
                    add_shared(output[3], mm(gw.T, cc))
                del gw
        return *output, None, None


class _Radial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, h, weight, bias, tp, linear):
        state = ForwardState(tp, linear)
        out = x.new_empty((len(x), tp.representation_dimension))
        for start in range(0, len(x), PAGE):
            end = min(start + PAGE, len(x))
            generated = mm(h[start:end], weight.T, bias)
            out[start:end] = forward(state.plan, x[start:end], y[start:end], generated)
            del generated
        state.assert_current()
        ctx.state = state
        ctx.save_for_backward(x, y, h, weight, bias, *state.plan.buffers)
        return out

    @staticmethod
    def backward(ctx, grad):
        ctx.state.assert_current()
        x, y, h, weight, bias, *buffers = ctx.saved_tensors
        assert_reduction_context(x)
        del buffers
        g = prepare_cotangent(grad, rows=len(x), width=ctx.state.plan.representation_dimension, device=x.device)
        values = _RadialAdjoint.apply(x, y, h, weight, bias, g, ctx.state, ctx.needs_input_grad[:5])
        return *values, None, None


def streamed_live_radial(tp, linear, x, y, h):
    tensor_types(x, y, h)
    original_plan(tp)
    original_linear(linear)
    check_features(tp, x, y)
    tensor_contract("radial_hidden", h, device=x.device)
    require_fp32_policy()
    assert_reduction_context(x)
    if (type(linear) is not torch.nn.Linear or linear.in_features != 64 or
            linear.out_features != tp.weight_numel or linear.bias is None or
            h.shape != (len(x), 64) or tp.channels != 64 or tp.parity != "full" or
            tp.parity_in2 != "full" or not 1 <= tp.m_max <= 8):
        raise ValueError("Experimental streaming scope is C64/H64/V1/full, biased affine, L1..8")
    tensor_contract("weight", linear.weight, device=x.device)
    tensor_contract("bias", linear.bias, device=x.device, ndim=1)
    return _Radial.apply(x.contiguous(), y.contiguous(), h.contiguous(),
                         linear.weight, linear.bias, tp, linear)
