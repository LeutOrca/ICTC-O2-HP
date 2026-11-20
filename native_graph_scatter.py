"""Capture-compatible adapter for MACE's deterministic CUDA scatter on Torch 2.6."""
from contextlib import contextmanager

from mace.modules import blocks, models, radial, utils
from mace.tools.scatter import scatter_sum as native_sum
import torch


class _FlatScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, index, size, tile_width):
        width = src.shape[1]
        if tile_width is None or width <= tile_width:
            out = flat_tile(src, index, size)
        else:
            out = torch.cat([flat_tile(tile, index, size) for tile in src.split(tile_width, dim=1)], dim=1)
        ctx.save_for_backward(index)
        return out

    @staticmethod
    def backward(ctx, derivative):
        index, = ctx.saved_tensors
        return derivative.index_select(0, index), None, None, None


def flat_tile(src, index, size):
    width = src.shape[1]
    columns = torch.arange(width, device=index.device)
    # ATen's deterministic scatter flattens these same integer coordinates
    # before using the sorted index_put kernel. Construct them on the GPU.
    coords = index[:, None] * width + columns
    out = src.new_zeros(size * width)
    out.index_put_((coords.flatten(),), src.flatten(), accumulate=True)
    return out.view(size, width)


def graph_scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum", *, tile_width=None):
    if (src.is_cuda and src.ndim == 2 and dim in (0, -2) and out is None
            and dim_size is not None and reduce == "sum"
            and index.ndim == 1 and index.numel() == src.shape[0]):
        return _FlatScatter.apply(src, index, dim_size, tile_width)
    return native_sum(src, index, dim, out, dim_size, reduce)


@contextmanager
def native_capture_compat(tile_width=None):
    modules = (blocks, models, radial, utils)
    original = [module.scatter_sum for module in modules]
    def scatter(*args, **kwargs):
        return graph_scatter(*args, **kwargs, tile_width=tile_width)

    try:
        for module in modules:
            module.scatter_sum = scatter
        yield
    finally:
        for module, scatter in zip(modules, original):
            module.scatter_sum = scatter
