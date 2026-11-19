"""Fused gather and radial modulation for a source/receiver message pair."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_radial(x, edge_index, weights, output, edges, width: tl.constexpr,
                   components: tl.constexpr, tile: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * tile + tl.arange(0, tile)
    slot = i % (2 * width)
    component = (i // (2 * width)) % components
    edge = i // (2 * width * components)
    mask = edge < edges
    atom = tl.load(edge_index + (slot // width) * edges + edge, mask, 0)
    xv = tl.load(x + (atom * components + component) * width + slot % width, mask, 0.)
    w = tl.load(weights + i, mask, 0.)
    tl.store(output + i, xv * w, mask)


@triton.jit
def _gather_radial_vjp(x, edge_index, weights, grad_out, grad_w, grad_src, grad_dst, edges,
                       width: tl.constexpr, components: tl.constexpr, tile: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * tile + tl.arange(0, tile)
    slot = i % (2 * width)
    component = (i // (2 * width)) % components
    edge = i // (2 * width * components)
    mask = edge < edges
    is_dst = slot >= width
    atom = tl.load(edge_index + is_dst.to(tl.int64) * edges + edge, mask, 0)
    xv = tl.load(x + (atom * components + component) * width + slot % width, mask, 0.)
    w = tl.load(weights + i, mask, 0.)
    out = tl.load(grad_out + i, mask, 0.)
    target = (edge * components + component) * width + slot % width
    tl.store(grad_w + i, xv * out, mask)
    tl.store(grad_src + target, w * out, mask & ~is_dst)
    tl.store(grad_dst + target, w * out, mask & is_dst)


def _reference(x, edge_index, weights):
    return torch.cat((x.index_select(0, edge_index[0]),
                      x.index_select(0, edge_index[1])), dim=-1) * weights


class _GatherRadial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, edge_index, weights):
        _, components, width = x.shape
        edges = edge_index.size(1)
        result = weights.new_empty(weights.shape)
        tile = 256
        with torch.cuda.device(x.device):
            _gather_radial[(triton.cdiv(result.numel(), tile),)](
                x, edge_index, weights, result, edges, width, components, tile,
            )
        ctx.save_for_backward(x, edge_index, weights)
        return result

    @staticmethod
    def backward(ctx, incoming):
        x, edge_index, weights = ctx.saved_tensors
        need_x, _, need_weight = ctx.needs_input_grad
        if torch.is_grad_enabled():
            wx = incoming * weights
            src, dst = wx.chunk(2, dim=-1)
            grad_x = _reduce_node_gradients(x, edge_index, src, dst) if need_x else None
            features = torch.cat((x.index_select(0, edge_index[0]), x.index_select(0, edge_index[1])), dim=-1)
            grad_w = incoming * features if need_weight else None
            return grad_x, None, grad_w
        grad_out = incoming.contiguous()
        _, components, width = x.shape
        edges = edge_index.size(1)
        grad_w = weights.new_empty(weights.shape)
        src = x.new_empty((edges, components, width))
        dst = x.new_empty((edges, components, width))
        tile = 256
        with torch.cuda.device(x.device):
            _gather_radial_vjp[(triton.cdiv(grad_out.numel(), tile),)](
                x, edge_index, weights, grad_out, grad_w, src, dst, edges, width, components, tile,
            )
        grad_x = _reduce_node_gradients(x, edge_index, src, dst) if need_x else None
        return grad_x, None, grad_w if need_weight else None


def _reduce_node_gradients(x, edge_index, src, dst):
    index_select_backward = torch.ops.aten.index_select_backward.default
    source_grad = index_select_backward(src, x.shape, 0, edge_index[0])
    target_grad = index_select_backward(dst, x.shape, 0, edge_index[1])
    return source_grad + target_grad


def gather_radial(x, edge_index, weights):
    """Compute `cat(x[src], x[dst]) * weights` while retaining node-sized inputs for VJP."""
    eligible = (x.is_cuda and x.dtype == weights.dtype == torch.float32 and x.ndim == 3
                and x.is_contiguous() and edge_index.is_contiguous() and weights.is_contiguous()
                and edge_index.dtype == torch.int64 and edge_index.ndim == 2 and edge_index.size(0) == 2
                and x.device == edge_index.device == weights.device
                and weights.shape == (edge_index.size(1), x.size(1), 2 * x.size(2)) and weights.numel())
    if eligible:
        return _GatherRadial.apply(x, edge_index, weights)
    return _reference(x, edge_index, weights)
