"""Fuse the qualified scalar TP with the existing native-order receiver CSR."""
from contextlib import contextmanager

from mace.modules import blocks
import torch
import triton
import triton.language as tl

from mace_scalar_tp import ScalarTP
from native_reduce import sum32_ascending as _warp_sum_ascending
from native_graph_scatter import native_capture_compat
from receiver_csr import _fold, _ReceiverSum, schedule


@triton.jit
def _message(X, H, W, Edges, Product, edge, column, E: tl.constexpr, C: tl.constexpr,
             XS: tl.constexpr, HS: tl.constexpr, WS: tl.constexpr):
    degree = tl.where(column < C, 0, tl.where(column < 4 * C, 1, tl.where(column < 9 * C, 2, 3)))
    first, width = degree * degree, degree * 2 + 1
    channel = (column - first * C) // width
    component = (column - first * C) % width
    source = tl.load(Edges + edge).to(tl.int32)
    scalar = tl.load(X + source * XS + channel, column < C * 16, other=0.)
    weight = tl.load(W + edge * WS + degree * C + channel, column < C * 16, other=0.)
    product = scalar * weight
    harmonic = tl.load(H + edge * HS + first + component, column < C * 16, other=0.)
    tl.store(Product + edge * C * 4 + degree * C + channel, product,
             (column < C * 16) & (component == 0))
    return product * harmonic


@triton.jit
def _receiver(X, H, W, Edges, Permutation, Offsets, Output, Product,
              E: tl.constexpr, C: tl.constexpr, XS: tl.constexpr, HS: tl.constexpr,
              WS: tl.constexpr, CHANNELS: tl.constexpr):
    node = tl.program_id(0)
    column = tl.program_id(1) * CHANNELS + tl.arange(0, CHANNELS)
    lane = tl.arange(0, 32)
    start = tl.load(Offsets + node)
    length = tl.load(Offsets + node + 1) - start
    groups = length // 32
    partial = tl.full((CHANNELS, 32), 0., tl.float32)
    for group in range(groups):
        edge = tl.load(Permutation + start + group * 32 + lane).to(tl.int32)
        partial = partial + _message(X, H, W, Edges, Product, edge[None, :], column[:, None],
                                      E, C, XS, HS, WS)
    partial = _fold(partial, CHANNELS, 16)
    partial = _fold(partial, CHANNELS, 8)
    partial = _fold(partial, CHANNELS, 4)
    partial = _fold(partial, CHANNELS, 2)
    partial = _fold(partial, CHANNELS, 1)
    total = tl.reshape(partial, (CHANNELS,))
    for item in range(groups * 32, length):
        edge = tl.load(Permutation + start + item).to(tl.int32)
        total = total + _message(X, H, W, Edges, Product, edge, column, E, C, XS, HS, WS)
    tl.store(Output + node * C * 16 + column, total, column < C * 16)


@triton.jit
def _angular(DY, Product, Edges, DH, E: tl.constexpr, C: tl.constexpr,
              DS: tl.constexpr, TILE: tl.constexpr):
    item = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    edge, harmonic = item // 16, item % 16
    valid = (edge < E) & (harmonic != 0)
    degree = tl.where(harmonic < 4, 1, tl.where(harmonic < 9, 2, 3))
    first, width = degree * degree, degree * 2 + 1
    target = tl.load(Edges + E + edge, valid, other=0).to(tl.int32)
    base = target * DS + first * C + harmonic - first
    even = tl.full((TILE,), 0., tl.float32)
    odd = tl.full((TILE,), 0., tl.float32)
    for channel in range(0, C, 2):
        p0 = tl.load(Product + edge * C * 4 + degree * C + channel, valid, other=0.)
        p1 = tl.load(Product + edge * C * 4 + degree * C + channel + 1, valid, other=0.)
        dy0 = tl.load(DY + base + channel * width, valid, other=0.)
        dy1 = tl.load(DY + base + (channel + 1) * width, valid, other=0.)
        even = tl.fma(p0, dy0, even)
        odd = tl.fma(p1, dy1, odd)
    tl.store(DH + item, even + odd, valid)


@triton.jit
def _angular_zero(DY, Product, Edges, DH, E: tl.constexpr, C: tl.constexpr, DS: tl.constexpr):
    edge = tl.program_id(0)
    target = tl.load(Edges + E + edge).to(tl.int32)
    lane = tl.arange(0, 32)
    total = tl.full((32,), 0., tl.float32)
    for part in tl.static_range(4):
        channel = lane + part * 32
        total += tl.load(DY + target * DS + channel) * tl.load(Product + edge * C * 4 + channel)
    tl.store(DH + edge * 16, tl.sum(_warp_sum_ascending(total[None, :]), 0))


@triton.jit
def _radial(X, H, DY, Edges, DW, E: tl.constexpr, C: tl.constexpr,
             XS: tl.constexpr, HS: tl.constexpr, DS: tl.constexpr, TILE: tl.constexpr):
    item = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    edge, column = item // (4 * C), item % (4 * C)
    valid = edge < E
    source = tl.load(Edges + edge, valid, other=0).to(tl.int32)
    target = tl.load(Edges + E + edge, valid, other=0).to(tl.int32)
    x = tl.load(X + source * XS + column % C, valid, other=0.)
    degree, channel = column // C, column % C
    first, width = degree * degree, degree * 2 + 1
    base = target * DS + first * C + channel * width
    p0 = tl.load(DY + base, valid, other=0.) * tl.load(H + edge * HS + first, valid, other=0.)
    p1 = tl.load(DY + base + 1, valid & (width > 1), other=0.) * tl.load(H + edge * HS + first + 1, valid & (width > 1), other=0.)
    p2 = tl.load(DY + base + 2, valid & (width > 2), other=0.) * tl.load(H + edge * HS + first + 2, valid & (width > 2), other=0.)
    p3 = tl.load(DY + base + 3, valid & (width > 3), other=0.) * tl.load(H + edge * HS + first + 3, valid & (width > 3), other=0.)
    p4 = tl.load(DY + base + 4, valid & (width > 4), other=0.) * tl.load(H + edge * HS + first + 4, valid & (width > 4), other=0.)
    p5 = tl.load(DY + base + 5, valid & (width > 5), other=0.) * tl.load(H + edge * HS + first + 5, valid & (width > 5), other=0.)
    p6 = tl.load(DY + base + 6, valid & (width > 6), other=0.) * tl.load(H + edge * HS + first + 6, valid & (width > 6), other=0.)
    sum3 = (p0 + p2) + p1
    sum5 = ((p0 + p4) + p1) + (p2 + p3)
    sum7 = ((p0 + p4) + (p1 + p5)) + ((p2 + p6) + p3)
    value = tl.where(degree == 0, p0, tl.where(degree == 1, sum3, tl.where(degree == 2, sum5, sum7)))
    tl.store(DW + item, value * x, valid)


class _Fused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h, weights, edges, permutation, offsets):
        if x.requires_grad or x.shape[1] != 128:
            raise ValueError("Fused scalar receiver requires frozen C128 senders")
        count, channels = h.shape[0], x.shape[1]
        output = x.new_empty((x.shape[0], channels * 16))
        product = x.new_empty((count, channels * 4))
        if x.shape[0]:
            _receiver[(x.shape[0], triton.cdiv(channels * 16, 16))](
                x, h, weights, edges, permutation, offsets, output, product,
                count, channels, x.stride(0), h.stride(0), weights.stride(0), 16,
                num_warps=4, enable_fp_fusion=False,
            )
        ctx.save_for_backward(x, h, product, edges)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, incoming):
        x, h, product, edges = ctx.saved_tensors
        count, channels = h.shape[0], x.shape[1]
        dh = torch.empty_like(h)
        dw = x.new_empty((count, channels * 4))
        if count:
            _angular_zero[(count,)](incoming, product, edges, dh, count, channels,
                                    incoming.stride(0), num_warps=1, enable_fp_fusion=False)
            _angular[(triton.cdiv(count * 16, 128),)](
                incoming, product, edges, dh, count, channels, incoming.stride(0), 128,
                num_warps=4, enable_fp_fusion=False)
            _radial[(triton.cdiv(dw.numel(), 256),)](
                x, h, incoming, edges, dw, count, channels, x.stride(0), h.stride(0), incoming.stride(0),
                256, num_warps=4, enable_fp_fusion=False)
        return None, dh, dw, None, None, None


class ScalarReceiver(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        ScalarTP(reference)
        self.reference = reference
        self.row_schedule = None

    def forward(self, x, h, weights, edges):
        if self.row_schedule is None:
            raise RuntimeError("The fused scalar receiver requires its shared CSR schedule")
        return _Fused.apply(x, h, weights, edges, *self.row_schedule)


def install(model):
    block = model.interactions[0]
    block.conv_tp = ScalarReceiver(block.conv_tp)
    block.conv_fusion = True


def evaluate(model, data):
    import qualify_force as prior
    with native_capture_compat(model._scatter_tile_width):
        receiver, nodes = data["edge_index"][1], data["positions"].shape[0]
        permutation, offsets = schedule(receiver, nodes)
        original = blocks.scatter_sum
        product = model.interactions[0].conv_tp
        product.row_schedule = (permutation, offsets)

        def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
            if (src.is_cuda and src.dtype == torch.float32 and src.ndim == 2 and dim == 0
                    and out is None and dim_size == nodes and reduce == "sum"
                    and index.shape == receiver.shape and index.data_ptr() == receiver.data_ptr()):
                return _ReceiverSum.apply(src, index, permutation, offsets)
            return original(src, index, dim, out, dim_size, reduce)

        blocks.scatter_sum = scatter
        try:
            return prior.evaluate(model, data)
        finally:
            blocks.scatter_sum = original
            product.row_schedule = None
