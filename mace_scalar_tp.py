"""Frozen first MACE TP preserving the generated native x*w, then h order."""
import torch
import triton
import triton.language as tl

from matrix_kernels import matrix
from native_reduce import sum32_ascending as _warp_sum_ascending


@triton.jit
def _forward(X, H, W, Y, Product, E: tl.constexpr, C: tl.constexpr,
             XS: tl.constexpr, HS: tl.constexpr, WS: tl.constexpr, TILE: tl.constexpr):
    item = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    edge = item // (C * 16)
    output = item % (C * 16)
    order = tl.where(output < C, 0, tl.where(output < C * 4, 1, tl.where(output < C * 9, 2, 3)))
    harmonic_start = order * order
    width = order * 2 + 1
    position = output - C * harmonic_start
    channel, component = position // width, position % width
    mask = edge < E
    value = tl.load(X + edge * XS + channel, mask, other=0.)
    weight = tl.load(W + edge * WS + order * C + channel, mask, other=0.)
    product = value * weight
    harmonic = tl.load(H + edge * HS + harmonic_start + component, mask, other=0.)
    tl.store(Y + item, product * harmonic, mask)
    tl.store(Product + edge * C * 4 + order * C + channel, product, mask & (component == 0))


@triton.jit
def _h0_gradient(DY, Product, DH, E: tl.constexpr, C: tl.constexpr, DS: tl.constexpr):
    edge = tl.program_id(0)
    lane = tl.arange(0, 32)
    total = tl.full((32,), 0., tl.float32)
    for part in tl.static_range(4):
        channel = lane + part * 32
        incoming = tl.load(DY + edge * DS + channel)
        product = tl.load(Product + edge * C * 4 + channel)
        total = total + incoming * product
    value = tl.sum(_warp_sum_ascending(total[None, :]), 0)
    tl.store(DH + edge * 16, value)


@triton.jit
def _finish_weights(X, H, DY, DW, E: tl.constexpr, C: tl.constexpr,
                    XS: tl.constexpr, HS: tl.constexpr, DS: tl.constexpr, TILE: tl.constexpr):
    item = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    edge, column = item // (4 * C), item % (4 * C)
    valid = edge < E
    source = tl.load(X + edge * XS + column % C, valid, other=0.)
    degree = column // C
    width = degree * 2 + 1
    first = degree * degree
    channel = column % C
    p0 = tl.load(DY + edge * DS + first * C + channel * width, valid, other=0.) * tl.load(H + edge * HS + first, valid, other=0.)
    p1 = tl.load(DY + edge * DS + first * C + channel * width + 1, valid & (width > 1), other=0.) * tl.load(H + edge * HS + first + 1, valid & (width > 1), other=0.)
    p2 = tl.load(DY + edge * DS + first * C + channel * width + 2, valid & (width > 2), other=0.) * tl.load(H + edge * HS + first + 2, valid & (width > 2), other=0.)
    p3 = tl.load(DY + edge * DS + first * C + channel * width + 3, valid & (width > 3), other=0.) * tl.load(H + edge * HS + first + 3, valid & (width > 3), other=0.)
    p4 = tl.load(DY + edge * DS + first * C + channel * width + 4, valid & (width > 4), other=0.) * tl.load(H + edge * HS + first + 4, valid & (width > 4), other=0.)
    p5 = tl.load(DY + edge * DS + first * C + channel * width + 5, valid & (width > 5), other=0.) * tl.load(H + edge * HS + first + 5, valid & (width > 5), other=0.)
    p6 = tl.load(DY + edge * DS + first * C + channel * width + 6, valid & (width > 6), other=0.) * tl.load(H + edge * HS + first + 6, valid & (width > 6), other=0.)
    # The native einsum is an outer product. Its radial adjoint is a rounded
    # elementwise multiply followed by Reduce.cuh, rather than an FMA dot.
    # Last dimensions 3/5/7 use 2/4/4 lanes and ascending shuffle offsets.
    sum3 = (p0 + p2) + p1
    sum5 = ((p0 + p4) + p1) + (p2 + p3)
    sum7 = ((p0 + p4) + (p1 + p5)) + ((p2 + p6) + p3)
    value = tl.where(degree == 0, p0, tl.where(degree == 1, sum3, tl.where(degree == 2, sum5, sum7)))
    tl.store(DW + item, value * source, valid)


class _ScalarTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h, weights):
        if x.requires_grad:
            raise RuntimeError("Frozen first-layer scalar senders must be position-independent")
        edges, channels = x.shape
        if channels != 128 or h.shape != (edges, 16) or weights.shape != (edges, channels * 4):
            raise ValueError("Expected the frozen C128, l<=3 canonical scalar TP")
        if any(value.dtype != torch.float32 or not value.is_cuda or value.stride(-1) != 1
               for value in (x, h, weights)):
            raise ValueError("Scalar TP requires CUDA FP32 tensors with unit inner strides")
        output = x.new_empty((edges, channels * 16))
        product = x.new_empty((edges, channels * 4))
        if edges:
            _forward[(triton.cdiv(output.numel(), 256),)](
                x, h, weights, output, product, edges, channels,
                x.stride(0), h.stride(0), weights.stride(0), 256,
                num_warps=4, enable_fp_fusion=False,
            )
        ctx.save_for_backward(x, h, product)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, incoming):
        x, h, product = ctx.saved_tensors
        edges, channels = x.shape
        dh = h.new_empty(h.shape)
        dw = x.new_empty((edges, channels * 4))
        if edges:
            _h0_gradient[(edges,)](incoming, product, dh, edges, channels, incoming.stride(0),
                                   num_warps=1, enable_fp_fusion=False)
            for degree in (1, 2, 3):
                width = degree * 2 + 1
                first = degree * degree
                derivative = incoming[:, first * channels:(first + width) * channels].reshape(
                    edges, channels, width)
                weighted = product[:, degree * channels:(degree + 1) * channels].unsqueeze(1)
                matrix(weighted, derivative, backend="cuda2",
                       output=dh[:, first:first + width].unsqueeze(1))
            _finish_weights[(triton.cdiv(dw.numel(), 256),)](
                x, h, incoming, dw, edges, channels, x.stride(0), h.stride(0), incoming.stride(0),
                256, num_warps=4, enable_fp_fusion=False,
            )
        return None, dh, dw


class ScalarTP(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        if (str(reference.irreps_in1) != "128x0e" or reference.weight_numel != 512
                or reference.irreps_in2.dim != 16 or reference.irreps_out.dim != 2048
                or len(reference.instructions) != 4):
            raise ValueError("Unsupported first scalar TP identity")
        for degree, instruction in enumerate(reference.instructions):
            if (instruction.i_in1 != 0 or instruction.i_in2 != degree
                    or instruction.i_out != degree or instruction.connection_mode != "uvu"
                    or instruction.path_shape != (128, 1)):
                raise ValueError("Scalar TP path order differs from the frozen generated code")
        self.reference = reference

    def forward(self, x, h, weights):
        return _ScalarTP.apply(x, h, weights)


def install(model):
    block = model.interactions[0]
    block.conv_tp = ScalarTP(block.conv_tp)
