"""Replace only the frozen affine input adjoint with own IEEE FP32 sums.

The existing native affine forward runs unchanged. The custom operation is
an identity on those generated weights; its sole arithmetic is the input VJP.
No vendor GEMM is called by this replacement and no parameter VJP is claimed.
"""
import torch
from torch.autograd.function import once_differentiable

from .ordered_matrix import scheduled
from .contracts import check_linear, check_tp, prepare_cotangent
from .frozen_state import ForwardState
from .reduction_policy import assert_reduction_context, choose_radial


class FrozenAffineInputAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, generated, state, configuration):
        weight = state.weight
        if weight.requires_grad or weight.dtype != torch.float32:
            raise ValueError("Only frozen FP32 affine input adjoints are supported")
        state.assert_current()
        ctx.save_for_backward(weight, state.bias, *state.plan.buffers)
        ctx.state = state
        ctx.configuration = configuration
        ctx.rows = len(hidden)
        return generated

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_generated):
        ctx.state.assert_current()
        weight, _, *_ = ctx.saved_tensors
        assert_reduction_context(weight)
        grad = prepare_cotangent(grad_generated, rows=ctx.rows, width=weight.shape[0], device=weight.device)
        hidden_grad = scheduled(grad, weight, ctx.configuration)
        return hidden_grad, None, None, None


def ordered_affine(linear, hidden, tp, decision):
    """Internal entry after a radial coverage decision; forward stays native."""
    state = ForwardState(tp, linear)
    generated = linear(hidden)
    state.assert_current()
    return FrozenAffineInputAdjoint.apply(hidden, generated, state, decision.configuration)


def affine(linear, hidden, configuration="auto", *, tp=None):
    check_linear(linear, hidden, outputs=None if tp is None else tp.weight_numel)
    if tp is not None:
        check_tp(tp, hidden.device)
    hidden = hidden.contiguous()
    decision = choose_radial(tp, linear, hidden, configuration, execution="unfused")
    if decision.route == "native":
        return linear(hidden)
    return ordered_affine(linear, hidden, tp, decision)
