"""Own frozen FP32 SO2 preserving native real products before complex assembly."""
from types import MethodType

import torch
import triton
import triton.language as tl

from cuda_matrix import compiled_matrix
from frozen_so2 import _pack_input
from matrix_kernels import matrix
from so2_layer import assemble, zero_order_weight


@triton.jit
def _complex_output(Parts, Result, E: tl.constexpr, O: tl.constexpr,
                    RS: tl.constexpr, OFFSET: tl.constexpr, TILE: tl.constexpr):
    item = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    edge, column = item // (2 * O), item % (2 * O)
    channel = column % O
    valid = edge < E
    real_real = tl.load(Parts + edge * 4 * O + channel, valid, other=0.)
    imag_imag = tl.load(Parts + edge * 4 * O + 3 * O + channel, valid, other=0.)
    imag_real = tl.load(Parts + edge * 4 * O + 2 * O + channel, valid, other=0.)
    real_imag = tl.load(Parts + edge * 4 * O + O + channel, valid, other=0.)
    result = tl.where(column < O, real_real - imag_imag, imag_real + real_imag)
    tl.store(Result + edge * RS + OFFSET + column, result, valid)


@triton.jit
def _complex_adjoint(DY, Packed, E: tl.constexpr, O: tl.constexpr,
                      DS: tl.constexpr, OFFSET: tl.constexpr, TILE: tl.constexpr):
    item = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    edge, column = item // (4 * O), item % (4 * O)
    part, channel = column // O, column % O
    real = tl.load(DY + edge * DS + OFFSET + channel, edge < E, other=0.)
    imag = tl.load(DY + edge * DS + OFFSET + O + channel, edge < E, other=0.)
    value = tl.where(part == 0, real, tl.where(part == 3, -real, imag))
    tl.store(Packed + item, value, edge < E)


@triton.jit
def _copy_matrix(Source, Target, ROWS: tl.constexpr, COLS: tl.constexpr,
                  S0: tl.constexpr, S1: tl.constexpr, D0: tl.constexpr, D1: tl.constexpr,
                  TILE: tl.constexpr):
    item = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    row, column = item // COLS, item % COLS
    value = tl.load(Source + row * S0 + column * S1, row < ROWS, other=0.)
    tl.store(Target + row * D0 + column * D1, value, row < ROWS)


def product(left, right, configuration, output=None, bias=None):
    if configuration == "triton":
        return matrix(left, right, bias=bias, backend="triton", output=output)
    join = configuration[-1] if configuration[-1] in "lab" else "l"
    bare = configuration[:-1] if configuration[-1] in "lab" else configuration
    groups, stripe = map(int, bare.split("x"))
    if bias is not None:
        bias = bias.expand((*left.shape[:-1], right.shape[-1]))
    return compiled_matrix(left, right, bias, 1., 1., groups=groups, stripe=stripe,
                           linear_join=join == "l", adjacent_join=join == "a", output=output)


class _Ordered(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, module, forward_config, backward_config, zero_backward_config):
        edges, components, channels = inputs.shape
        out_channels, head = module.num_out_channels, module.extra_m0_out_channels or 0
        result = inputs.new_empty((edges, components * out_channels))
        extra = inputs.new_empty((edges, head))
        zero_input = inputs[:, :module.lmax + 1].reshape(edges, -1)
        zero = matrix(zero_input, zero_order_weight(module), bias=module.fc_m0.bias)
        if head:
            _copy_matrix[(triton.cdiv(extra.numel(), 256),)](
                zero, extra, edges, head, *zero.stride(), *extra.stride(), 256)
        zero_width = (module.lmax + 1) * out_channels
        _copy_matrix[(triton.cdiv(edges * zero_width, 256),)](
            zero[:, head:], result[:, :zero_width], edges, zero_width,
            *zero[:, head:].stride(), *result[:, :zero_width].stride(), 256)
        weights = []
        backward_configs = []
        first_mixer = module.extra_m0_out_channels is not None
        if "calibrated" in (forward_config, backward_config, zero_backward_config):
            if (channels, out_channels) not in ((256, 64), (32, 128)):
                raise ValueError("Calibrated schedule supports only the frozen official FC shapes")
        if zero_backward_config == "calibrated":
            zero_backward_config = "-2x1l" if first_mixer else "-4x1a"
        offset = module.lmax + 1
        for layer in module.so2_m_linear:
            width, outputs = layer.in_features, layer.out_features
            value = inputs[:, offset:offset + 2 * (module.lmax + 1 - layer.m)].reshape(edges, 2, width)
            packed = inputs.new_empty((edges * 2, width))
            if edges:
                _pack_input[(triton.cdiv(packed.numel(), 256),)](
                    value, packed, edges, width, *value.stride(), 256)
            current_forward = ("8x1b" if first_mixer else "triton") if forward_config == "calibrated" else forward_config
            products = product(packed, layer.fc.weight.t(), current_forward)
            if edges:
                _complex_output[(triton.cdiv(edges * 2 * outputs, 256),)](
                    products, result, edges, outputs, result.stride(0), offset * out_channels,
                    256, enable_fp_fusion=False)
            offset += 2 * (module.lmax + 1 - layer.m)
            weights.append(layer.fc.weight)
            backward_configs.append(("-2x1l" if first_mixer else "-8x1a" if layer.m == 1 else "-4x1a")
                                    if backward_config == "calibrated" else backward_config)
        ctx.save_for_backward(module.fc_m0.weight, *weights)
        ctx.shape, ctx.lmax, ctx.head = inputs.shape, module.lmax, head
        ctx.backward_configs, ctx.zero_backward_config = backward_configs, zero_backward_config
        return result.reshape(edges, components, out_channels), extra

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, incoming, extra_gradient):
        zero_weight, *weights = ctx.saved_tensors
        edges, components, channels = ctx.shape
        out_channels = incoming.shape[-1]
        derivative = incoming.reshape(edges, -1)
        result = incoming.new_empty((edges, components * channels))
        zero_width = (ctx.lmax + 1) * out_channels
        zero_derivative = derivative[:, :zero_width]
        if ctx.head:
            zero_derivative, _ = assemble((extra_gradient, zero_derivative), 0)
        product(zero_derivative, zero_weight, ctx.zero_backward_config,
                output=result[:, :(ctx.lmax + 1) * channels])
        offset = ctx.lmax + 1
        for weight, backward_config in zip(weights, ctx.backward_configs):
            width, outputs = weight.shape[1], weight.shape[0] // 2
            packed = incoming.new_empty((edges * 2, outputs * 2))
            if edges:
                _complex_adjoint[(triton.cdiv(packed.numel(), 256),)](
                    derivative, packed, edges, outputs, derivative.stride(0), offset * out_channels,
                    256, enable_fp_fusion=False)
            gradient = product(packed, weight, backward_config).reshape(edges, width * 2)
            if edges:
                _copy_matrix[(triton.cdiv(gradient.numel(), 256),)](
                    gradient, result[:, offset * channels:offset * channels + width * 2], edges, width * 2,
                    *gradient.stride(), *result[:, offset * channels:offset * channels + width * 2].stride(), 256)
            offset += width * 2 // channels
        return result.reshape(ctx.shape), None, None, None, None


def ordered_layer(inputs, module, forward_config="triton", backward_config="triton", zero_backward_config="triton"):
    outputs, extra = _Ordered.apply(inputs, module, forward_config, backward_config, zero_backward_config)
    return (outputs, extra) if module.extra_m0_out_channels is not None else outputs


def install(model, forward_config="triton", backward_config="triton", zero_backward_config="triton"):
    model.requires_grad_(False)
    names = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != "SO2Linear":
            continue
        def forward(layer, inputs):
            return ordered_layer(inputs, layer, forward_config, backward_config, zero_backward_config)
        module.forward = MethodType(forward, module)
        names.append(name)
    assert len(names) == 14
    return names
