"""Experimental own-kernel full VJP and double backward for the sealed V=1 UVU.

No reference replay or vendor TP is used for either derivative order. The
second adjoint follows trilinearity and reuses the same four bounded kernels.
The original 17a71d7 dispatch remains untouched. This entry is explicit and
restricted to the runtime under validation, not a promoted default route.
"""
import torch
import triton
from torch.autograd.function import once_differentiable

from .contracts import check_features, prepare_cotangent, prepare_uvu, tensor_contract
from .frozen_state import ForwardState
from .live_contracts import original_plan, tensor_types
from .reduction_policy import assert_reduction_context, require_fp32_policy
from .uvu64_kernels import _forward, _input_grad, _weight_unshared


def forward(plan, x, y, w):
    output = x.new_empty((len(x), plan.representation_dimension))
    if len(x):
        _forward[(triton.cdiv(len(x) * plan.channels, 128), len(plan.entries_out))](
            x, y, w, plan._path_info, plan._coefficients, output,
            len(x), x.shape[1], y.shape[1], plan.channels, 1, len(plan.paths),
            plan.weight_numel, False, 128, 1, num_warps=4, enable_fp_fusion=False)
    return output


def input_adjoint(plan, x, y, w, g, first):
    output = torch.empty_like(x if first else y)
    if len(x):
        block = 128 if first else 16
        order = plan._first_order if first else plan._second_order
        count = len(plan.entries_in1 if first else plan.entries_in2)
        width = plan.channels if first else 1
        _input_grad[(triton.cdiv(len(x) * width, block), count)](
            x, y, w, g, plan._path_info, plan._coefficients, order, output,
            len(x), x.shape[1], y.shape[1], plan.channels, 1, len(plan.paths),
            len(plan.entries_out), len(plan.entries_in1), plan.weight_numel,
            False, first, block, 1 if first else triton.next_power_of_2(plan.channels),
            num_warps=4, enable_fp_fusion=False)
    return output


def weight_adjoint(plan, x, y, g):
    output = x.new_empty((len(x), plan.weight_numel))
    if len(x):
        _weight_unshared[(triton.cdiv(len(x) * plan.channels, 128), len(plan.paths))](
            x, y, g, plan._path_info, plan._coefficients, output,
            len(x), x.shape[1], y.shape[1], plan.channels, 1, plan.weight_numel,
            128, num_warps=4, enable_fp_fusion=False)
    return output


def add_private(accumulator, term):
    if accumulator is None:
        return term
    # These are freshly allocated second-adjoint outputs, never input aliases.
    accumulator.add_(term)
    return accumulator


class _Adjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, w, g, state, needs):
        state.assert_current()
        plan = state.plan
        outputs = (input_adjoint(plan, x, y, w, g, True) if needs[0] else None,
                   input_adjoint(plan, x, y, w, g, False) if needs[1] else None,
                   weight_adjoint(plan, x, y, g) if needs[2] else None)
        ctx.save_for_backward(x, y, w, g, *plan.buffers)
        ctx.state = state
        ctx.set_materialize_grads(False)
        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, a, b, c):
        ctx.state.assert_current()
        x, y, w, g, *buffers = ctx.saved_tensors
        assert_reduction_context(x)
        plan = ctx.state.plan.with_buffers(buffers)
        directions = []
        for name, value, original in (("dd_left", a, x), ("dd_right", b, y), ("dd_weight", c, w)):
            if value is not None:
                tensor_contract(name, value, device=x.device)
                if value.shape != original.shape:
                    raise ValueError(name + " has the wrong shape")
                value = value.contiguous()
            directions.append(value)
        a, b, c = directions
        dx = dy = dw = dg = None
        # If A_x,A_y,A_w denote the three VJPs of trilinear T, then
        # dx=A_x(b,w,g)+A_x(y,c,g), dy=A_y(a,w,g)+A_y(x,c,g),
        # dw=A_w(a,y,g)+A_w(x,b,g), dg=T(a,y,w)+T(x,b,w)+T(x,y,c).
        if a is not None:
            if ctx.needs_input_grad[1]:
                dy = add_private(dy, input_adjoint(plan, a, y, w, g, False))
            if ctx.needs_input_grad[2]:
                dw = add_private(dw, weight_adjoint(plan, a, y, g))
            if ctx.needs_input_grad[3]:
                dg = add_private(dg, forward(plan, a, y, w))
        if b is not None:
            if ctx.needs_input_grad[0]:
                dx = add_private(dx, input_adjoint(plan, x, b, w, g, True))
            if ctx.needs_input_grad[2]:
                dw = add_private(dw, weight_adjoint(plan, x, b, g))
            if ctx.needs_input_grad[3]:
                dg = add_private(dg, forward(plan, x, b, w))
        if c is not None:
            if ctx.needs_input_grad[0]:
                dx = add_private(dx, input_adjoint(plan, x, y, c, g, True))
            if ctx.needs_input_grad[1]:
                dy = add_private(dy, input_adjoint(plan, x, y, c, g, False))
            if ctx.needs_input_grad[3]:
                dg = add_private(dg, forward(plan, x, y, c))
        return dx, dy, dw, dg, None, None


class _UVU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, w, tp):
        state = ForwardState(tp)
        result = forward(state.plan, x, y, w)
        state.assert_current()
        ctx.state = state
        ctx.save_for_backward(x, y, w, *state.plan.buffers)
        return result

    @staticmethod
    def backward(ctx, grad):
        ctx.state.assert_current()
        x, y, w, *buffers = ctx.saved_tensors
        assert_reduction_context(x)
        g = prepare_cotangent(grad, rows=len(x), width=ctx.state.plan.representation_dimension, device=x.device)
        # save_for_backward has already checked the original buffers' versions;
        # state checks additionally reject replacement or metadata mutation.
        del buffers
        gx, gy, gw = _Adjoint.apply(x, y, w, g, ctx.state, ctx.needs_input_grad[:3])
        return gx, gy, gw, None


def live_uvu(tp, x, y, w):
    tensor_types(x, y, w)
    original_plan(tp)
    x, y, w = prepare_uvu(tp, x, y, w)
    require_fp32_policy()
    assert_reduction_context(x)
    if tp.parity != "full" or tp.parity_in2 != "full" or tp.channels != 64 or not 1 <= tp.m_max <= 8:
        raise ValueError("Experimental qualification entry requires C64/V1/full, L1..8")
    return _UVU.apply(x, y, w, tp)


def live_radial(tp, linear, x, y, h):
    """Keep the original affine as-is, with own UVU first/second adjoints.

    Affine forward/derivatives remain the unchanged PyTorch layer in this
    initial capability candidate. This is not an own-GEMM performance claim.
    """
    check_features(tp, x, y)
    tensor_contract("radial_hidden", h, device=x.device)
    if type(linear) is not torch.nn.Linear or linear.out_features != tp.weight_numel:
        raise ValueError("Requires the original nn.Linear with TP weight width")
    if h.shape != (len(x), linear.in_features):
        raise ValueError("Incorrect radial hidden shape")
    for name, value in (("weight", linear.weight), ("bias", linear.bias)):
        if value is not None:
            tensor_contract(name, value, device=x.device, ndim=2 if name == "weight" else 1)
    return live_uvu(tp, x, y, linear(h.contiguous()))
