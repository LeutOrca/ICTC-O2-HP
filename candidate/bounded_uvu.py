"""Bounded-IR V=1 UVU dispatch and recomputing frozen radial composition.

The path loop stays in the GPU program instead of statically unrolling every
path over every representation entry. Kernels and coefficient/order tables
come from the frozen public Triton implementation. No vendor TP is used.
The unchanged radial affine forward is native; its input adjoint is our
ordered IEEE FP32 implementation. Only first input derivatives are supported.
"""
import torch
import triton
from torch.autograd.function import once_differentiable

from ictc_o2_hp.uvu_triton import _forward, _input_grad, _weight_unshared
from .ordered_matrix import scheduled
from .contracts import prepare_radial, prepare_uvu


def check(tp):
    if tp.channels_in2 != 1 or tp.shared_weights or tp.internal_weights:
        raise ValueError("Bounded candidate requires V=1 external unshared weights")


def forward_into(tp, x, y, w, output):
    n, u = len(x), tp.channels
    _forward[(triton.cdiv(n * u, 128), len(tp.entries_out))](
        x, y, w, tp._path_info, tp._coefficients, output,
        n, x.shape[1], y.shape[1], u, 1, len(tp.paths), tp.weight_numel,
        False, 128, 1, num_warps=4, enable_fp_fusion=False)


def inputs_into(tp, x, y, w, g, gx, gy):
    n, u = len(x), tp.channels
    for first, dest, order, block, reduce in (
        (True, gx, tp._first_order, 128, 1),
        (False, gy, tp._second_order, 16, triton.next_power_of_2(u)),
    ):
        if dest is None:
            continue
        entries = tp.entries_in1 if first else tp.entries_in2
        width = u if first else 1
        _input_grad[(triton.cdiv(n * width, block), len(entries))](
            x, y, w, g, tp._path_info, tp._coefficients, order, dest,
            n, x.shape[1], y.shape[1], u, 1, len(tp.paths),
            len(tp.entries_out), len(tp.entries_in1), tp.weight_numel,
            False, first, block, reduce, num_warps=4, enable_fp_fusion=False)


def weights_into(tp, x, y, g, gw):
    _weight_unshared[(triton.cdiv(len(x) * tp.channels, 128), len(tp.paths))](
        x, y, g, tp._path_info, tp._coefficients, gw,
        len(x), x.shape[1], y.shape[1], tp.channels, 1, tp.weight_numel,
        128, num_warps=4, enable_fp_fusion=False)


class BoundedUVU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, w, tp):
        check(tp)
        if not all(v.is_contiguous() and v.dtype == torch.float32 and v.is_cuda for v in (x, y, w)):
            raise ValueError("Candidate requires contiguous CUDA FP32 operands")
        out = x.new_empty((len(x), tp.representation_dimension))
        forward_into(tp, x, y, w, out)
        ctx.save_for_backward(x, y, w)
        ctx.tp = tp
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, y, w = ctx.saved_tensors
        g = grad.contiguous()
        gx = torch.empty_like(x) if ctx.needs_input_grad[0] else None
        gy = torch.empty_like(y) if ctx.needs_input_grad[1] else None
        inputs_into(ctx.tp, x, y, w, g, gx, gy)
        gw = torch.empty_like(w) if ctx.needs_input_grad[2] else None
        if gw is not None:
            weights_into(ctx.tp, x, y, g, gw)
        return gx, gy, gw, None


def bounded(tp, x, y, w):
    x, y, w = prepare_uvu(tp, x, y, w)
    return BoundedUVU.apply(x, y, w, tp)


class StreamedRadial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, h, tp, linear, configuration):
        check(tp)
        if any(p.requires_grad for p in linear.parameters()):
            raise ValueError("Only frozen affine parameters are supported")
        out = x.new_empty((len(x), tp.representation_dimension))
        # Same affine equation, bounded private weights; no input donation.
        for start in range(0, len(x), 4096):
            stop = min(start + 4096, len(x))
            w = linear(h[start:stop])
            forward_into(tp, x[start:stop], y[start:stop], w, out[start:stop])
            del w
        ctx.save_for_backward(x, y, h)
        ctx.tp, ctx.linear, ctx.configuration = tp, linear, configuration
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, y, h = ctx.saved_tensors
        g = grad.contiguous()
        gx = torch.empty_like(x) if ctx.needs_input_grad[0] else None
        gy = torch.empty_like(y) if ctx.needs_input_grad[1] else None
        gh = torch.empty_like(h) if ctx.needs_input_grad[2] else None
        for start in range(0, len(x), 4096):
            stop = min(start + 4096, len(x))
            xs, ys, gs = x[start:stop], y[start:stop], g[start:stop]
            if gx is not None or gy is not None:
                w = ctx.linear(h[start:stop])
                inputs_into(ctx.tp, xs, ys, w, gs,
                            None if gx is None else gx[start:stop],
                            None if gy is None else gy[start:stop])
                # gW does not depend on generated weights. Release before
                # allocating gW, allowing safe same-stream allocator reuse.
                del w
            if gh is not None:
                gw = x.new_empty((stop - start, ctx.tp.weight_numel))
                weights_into(ctx.tp, xs, ys, gs, gw)
                scheduled(gw, ctx.linear.weight, ctx.configuration, output=gh[start:stop])
                del gw
        return gx, gy, gh, None, None, None


def streamed(tp, linear, configuration, x, y, h):
    x, y, h = prepare_radial(tp, linear, x, y, h)
    return StreamedRadial.apply(x, y, h, tp, linear, configuration)
