"""Replace only the frozen affine input adjoint with own IEEE FP32 sums.

The existing native affine forward runs unchanged. The custom operation is
an identity on those generated weights; its sole arithmetic is the input VJP.
No vendor GEMM is called by this replacement and no parameter VJP is claimed.
"""
import torch
from torch.autograd.function import once_differentiable

from .ordered_matrix import scheduled


class FrozenAffineInputAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, generated, weight, configuration):
        if weight.requires_grad or weight.dtype != torch.float32:
            raise ValueError("Only frozen FP32 affine input adjoints are supported")
        ctx.save_for_backward(weight)
        ctx.configuration = configuration
        return generated

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_generated):
        weight, = ctx.saved_tensors
        hidden_grad = scheduled(grad_generated.contiguous(), weight, ctx.configuration)
        return hidden_grad, None, None, None


def affine(linear, hidden, configuration):
    if linear.weight.requires_grad or (linear.bias is not None and linear.bias.requires_grad):
        raise ValueError("Affine parameters must be frozen")
    generated = linear(hidden)
    return FrozenAffineInputAdjoint.apply(hidden, generated, linear.weight, configuration)
