"""Fused multi-order SO(2) assembly and input adjoints with frozen channel weights."""
import torch
import triton
import triton.language as tl

from matrix_kernels import matrix


@triton.jit
def _assemble(Parts, Merged, Extra, E, WIDTHS, TOTAL: tl.constexpr, HEAD: tl.constexpr,
              PS0, PS1, TILE: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    total: tl.constexpr = TOTAL - HEAD
    edge = index // (total + HEAD)
    col = index % (total + HEAD)
    valid = edge < E
    offset = 0
    for part in tl.static_range(len(WIDTHS)):
        skip = HEAD if part == 0 else 0
        width = WIDTHS[part] - skip
        selected = valid & (col >= offset) & (col < offset + width)
        value = tl.load(Parts[part] + edge * PS0[part] + (col - offset + skip) * PS1[part],
                        selected, other=0.)
        tl.store(Merged + edge * total + col, value, selected)
        offset += width
    if HEAD:
        selected = valid & (col >= total)
        value = tl.load(Parts[0] + edge * PS0[0] + (col - total) * PS1[0], selected, other=0.)
        tl.store(Extra + edge * HEAD + col - total, value, selected)


def assemble(parts, head):
    widths = tuple(part.shape[1] for part in parts)
    edges = parts[0].shape[0]
    output = parts[0].new_empty((edges, sum(widths) - head))
    extra = parts[0].new_empty((edges, head))
    if edges:
        _assemble[(triton.cdiv(edges * sum(widths), 256),)](
            tuple(parts), output, extra, edges, widths, sum(widths), head,
            tuple(part.stride(0) for part in parts), tuple(part.stride(1) for part in parts), 256,
        )
    return output, extra


@triton.jit
def _transpose_weight(Weight, Packed, ROWS: tl.constexpr, COLS: tl.constexpr,
                      S0: tl.constexpr, S1: tl.constexpr, TILE: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    row, col = index % ROWS, index // ROWS
    value = tl.load(Weight + row * S0 + col * S1, col < COLS, other=0.)
    tl.store(Packed + index, value, col < COLS)


def zero_order_weight(module):
    weight = module.fc_m0.weight
    key = (weight._version, weight.data_ptr(), weight.shape, weight.stride(), weight.device, weight.dtype)
    cached = getattr(module, "_native_zero_cache", None)
    if cached is None or cached[0] != key or cached[2] is not weight:
        packed = weight.new_empty((weight.shape[1], weight.shape[0]))
        _transpose_weight[(triton.cdiv(weight.numel(), 256),)](
            weight, packed, *weight.shape, *weight.stride(), 256,
        )
        module._native_zero_cache = (key, packed, weight)
    return module._native_zero_cache[1]


class _DirectLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, module, caches, backend):
        edges, components, channels = inputs.shape
        head = module.extra_m0_out_channels or 0
        zero_weight = module.fc_m0.weight
        zero_forward = zero_order_weight(module)
        backward_weights = tuple(cache.prepare(layer.fc.weight)
                                 for cache, layer in zip(caches, module.so2_m_linear))
        output_widths = (zero_weight.shape[0] - head,
                         *(weight.shape[0] for weight in backward_weights))
        output_width = sum(output_widths)
        result = inputs.new_empty((edges, output_width))
        extra = inputs.new_empty((edges, head))
        input_start = (module.lmax + 1) * channels
        scalar = inputs[:, :module.lmax + 1].reshape(edges, input_start)
        bias = module.fc_m0.bias
        if head:
            matrix(scalar, zero_forward[:, :head], bias=None if bias is None else bias[:head],
                   backend=backend, output=extra)
        matrix(scalar, zero_forward[:, head:], bias=None if bias is None else bias[head:],
               backend=backend, output=result[:, :output_widths[0]])
        output_start = output_widths[0]
        inner_parts = 2 if getattr(module, "_own_split_complex", False) else 1
        input_widths = [input_start]
        for order, (cache, width) in enumerate(zip(caches, output_widths[1:]), 1):
            component_count = 2 * (module.lmax + 1 - order)
            input_width = component_count * channels
            source = inputs.narrow(1, input_start // channels, component_count)
            matrix(source.reshape(edges, input_width), cache.forward_weight,
                   backend=backend, output=result[:, output_start:output_start + width],
                   inner_parts=inner_parts)
            input_widths.append(input_width)
            input_start += input_width
            output_start += width
        if input_start != components * channels:
            raise ValueError("Frozen SO2 input coverage mismatch")
        ctx.save_for_backward(zero_weight, *backward_weights)
        ctx.input_shape = inputs.shape
        ctx.input_widths, ctx.output_widths = tuple(input_widths), output_widths
        ctx.head, ctx.backend = head, backend
        ctx.inner_parts = inner_parts
        return result.reshape(edges, output_width // module.num_out_channels,
                              module.num_out_channels), extra

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, incoming, extra_gradient):
        zero_weight, *backward_weights = ctx.saved_tensors
        edges = incoming.shape[0]
        derivative = incoming.reshape(edges, sum(ctx.output_widths))
        result = incoming.new_empty((edges, sum(ctx.input_widths)))
        zero_derivative = derivative[:, :ctx.output_widths[0]]
        if ctx.head:
            zero_derivative, _ = assemble((extra_gradient, zero_derivative), 0)
        matrix(zero_derivative, zero_weight, backend=ctx.backend,
               output=result[:, :ctx.input_widths[0]])
        input_start, output_start = ctx.input_widths[0], ctx.output_widths[0]
        for input_width, output_width, weight in zip(
                ctx.input_widths[1:], ctx.output_widths[1:], backward_weights):
            matrix(derivative[:, output_start:output_start + output_width], weight,
                   backend=ctx.backend, output=result[:, input_start:input_start + input_width],
                   inner_parts=ctx.inner_parts)
            input_start += input_width
            output_start += output_width
        return result.reshape(ctx.input_shape), None, None, None


def frozen_layer(inputs, module, caches, backend):
    if inputs.dtype != torch.float32 or not inputs.is_cuda:
        raise ValueError("Frozen SO2 requires CUDA FP32 inputs")
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise ValueError("Frozen SO2 requires immutable channel weights")
    result, extra = _DirectLayer.apply(inputs, module, caches, backend)
    return (result, extra) if module.extra_m0_out_channels is not None else result
