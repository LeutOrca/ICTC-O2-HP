"""FP32 SO(2) linear output reconstruction and first-order input VJP."""

import torch
import triton
import triton.language as tl


@triton.jit
def _reconstruct(values, real, imag, rows, width: tl.constexpr, tile: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * tile + tl.arange(0, tile)
    col = i % width
    base = (i // width) * (4 * width) + col
    valid = i < rows * width
    ar = tl.load(values + base, valid, 0.)
    br = tl.load(values + base + width, valid, 0.)
    ai = tl.load(values + base + 2 * width, valid, 0.)
    bi = tl.load(values + base + 3 * width, valid, 0.)
    tl.store(real + i, ar - bi, valid)
    tl.store(imag + i, ai + br, valid)


@triton.jit
def _reconstruct_vjp(real, imag, grad, rows, width: tl.constexpr,
                     rs0, rs2, is0, is2, tile: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * tile + tl.arange(0, tile)
    row, col = i // width, i % width
    base = row * (4 * width) + col
    valid = i < rows * width
    gr = tl.load(real + row * rs0 + col * rs2, valid, 0.)
    gi = tl.load(imag + row * is0 + col * is2, valid, 0.)
    tl.store(grad + base, gr, valid)
    tl.store(grad + base + width, gi, valid)
    tl.store(grad + base + 2 * width, gi, valid)
    tl.store(grad + base + 3 * width, -gr, valid)


class _Reconstruction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values):
        rows, _, twice_width = values.shape
        width = twice_width // 2
        ctx.rows, ctx.width = rows, width
        real = values.new_empty((rows, 1, width))
        imag = values.new_empty((rows, 1, width))
        tile = 256
        with torch.cuda.device(values.device):
            _reconstruct[(triton.cdiv(rows * width, tile),)](values, real, imag, rows, width, tile)
        return real, imag

    @staticmethod
    def backward(ctx, real_grad, imag_grad):
        rows, width = ctx.rows, ctx.width
        if torch.is_grad_enabled():
            # Keep higher-order differentiation on ordinary PyTorch operations.
            return torch.cat((torch.cat((real_grad, imag_grad), dim=-1),
                              torch.cat((imag_grad, -real_grad), dim=-1)), dim=1)
        grad = real_grad.new_empty((rows, 2, 2 * width))
        tile = 256
        with torch.cuda.device(real_grad.device):
            _reconstruct_vjp[(triton.cdiv(rows * width, tile),)](
                real_grad, imag_grad, grad, rows, width, real_grad.stride(0),
                real_grad.stride(2), imag_grad.stride(0), imag_grad.stride(2), tile,
            )
        return grad


def reconstruct_so2(values):
    """Return the real and imaginary components of a nonzero-order SO(2) linear product."""
    if values.ndim != 3 or values.size(1) != 2 or values.size(2) % 2:
        raise ValueError("Expected an (edges, 2, 2*out_features) linear output.")
    if values.is_cuda and values.dtype == torch.float32 and values.is_contiguous() and values.numel():
        return _Reconstruction.apply(values)
    real_weight, imag_weight = values.chunk(2, dim=-1)
    return (real_weight[:, :1] - imag_weight[:, 1:],
            real_weight[:, 1:] + imag_weight[:, :1])
