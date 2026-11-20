"""Stable row-wise receiver sums with the native Torch 2.6 FP32 addition order."""
from contextlib import contextmanager

from mace.modules import blocks
import torch
import triton
import triton.language as tl


@triton.jit
def _fold(partial, CHANNELS: tl.constexpr, HALF: tl.constexpr):
    pairs = tl.permute(tl.reshape(partial, (CHANNELS, 2, HALF)), (0, 2, 1))
    lower, upper = tl.split(pairs)
    return lower + upper


@triton.jit
def _receiver_sum(Messages, Permutation, Offsets, Output,
                  WIDTH: tl.constexpr, ROW_STRIDE: tl.constexpr,
                  COL_STRIDE: tl.constexpr, CHANNELS: tl.constexpr):
    row = tl.program_id(0)
    column = tl.program_id(1) * CHANNELS + tl.arange(0, CHANNELS)
    lane = tl.arange(0, 32)
    lo = tl.load(Offsets + row)
    length = tl.load(Offsets + row + 1) - lo

    # Native flattened scatter uses one warp per output element. Complete
    # groups of 32 accumulate by lane, then fold at offsets 16, 8, 4, 2, 1.
    group_count = length // 32
    partial = tl.full((CHANNELS, 32), 0, tl.float32)
    for group in range(group_count):
        edge = tl.load(Permutation + lo + 32 * group + lane).to(tl.int32)
        value = tl.load(Messages + column[:, None] * COL_STRIDE + edge[None, :] * ROW_STRIDE,
                        column[:, None] < WIDTH, other=0)
        partial = partial + value
    partial = _fold(partial, CHANNELS, 16)
    partial = _fold(partial, CHANNELS, 8)
    partial = _fold(partial, CHANNELS, 4)
    partial = _fold(partial, CHANNELS, 2)
    partial = _fold(partial, CHANNELS, 1)
    total = tl.reshape(partial, (CHANNELS,))

    # The native warp sums remaining duplicates sequentially in lane zero.
    for item in range(group_count * 32, length):
        edge = tl.load(Permutation + lo + item).to(tl.int32)
        value = tl.load(Messages + column * COL_STRIDE + edge * ROW_STRIDE, column < WIDTH, other=0)
        total = total + value
    tl.store(Output + row * WIDTH + column, total, column < WIDTH)


def schedule(index, size):
    ordered, permutation = torch.sort(index, stable=True)
    offsets = torch.searchsorted(ordered, torch.arange(size + 1, device=index.device), out_int32=True)
    return permutation.to(torch.int32), offsets


class _ReceiverSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, messages, receiver, permutation, offsets):
        nodes = offsets.numel() - 1
        width = messages.shape[1]
        result = messages.new_empty((nodes, width))
        if nodes and width:
            _receiver_sum[(nodes, triton.cdiv(width, 16))](
                messages, permutation, offsets, result, width,
                messages.stride(0), messages.stride(1), 16,
                num_warps=4, enable_fp_fusion=False,
            )
        ctx.save_for_backward(receiver)
        return result

    @staticmethod
    def backward(ctx, derivative):
        receiver, = ctx.saved_tensors
        return derivative.index_select(0, receiver), None, None, None


def reduce(messages, receiver, permutation, offsets):
    return _ReceiverSum.apply(messages, receiver, permutation, offsets)


@contextmanager
def use_receiver_csr(data):
    receiver = data["edge_index"][1]
    nodes = data["positions"].shape[0]
    permutation, offsets = schedule(receiver, nodes)
    original = blocks.scatter_sum

    def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
        if (src.is_cuda and src.dtype == torch.float32 and src.ndim == 2
                and dim == 0 and out is None and dim_size == nodes and reduce == "sum"
                and index.shape == receiver.shape and index.data_ptr() == receiver.data_ptr()):
            return _ReceiverSum.apply(src, index, permutation, offsets)
        return original(src, index, dim, out, dim_size, reduce)

    blocks.scatter_sum = scatter
    try:
        yield
    finally:
        blocks.scatter_sum = original
