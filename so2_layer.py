"""Fused multi-order SO(2) assembly and input adjoints with frozen channel weights."""
import torch
import triton
import triton.language as tl

from frozen_so2 import FrozenWeightCache, frozen_so2


@triton.jit
def _assemble(Parts, Merged, Extra, E, WIDTHS: tl.constexpr, TOTAL: tl.constexpr, HEAD: tl.constexpr,
              PS0: tl.constexpr, PS1: tl.constexpr, TILE: tl.constexpr):
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


@triton.jit
def _disassemble(Merged, Extra, Parts, E, WIDTHS: tl.constexpr, TOTAL: tl.constexpr, HEAD: tl.constexpr,
                 GS0: tl.constexpr, GS1: tl.constexpr, ES0: tl.constexpr, ES1: tl.constexpr,
                 TILE: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    total: tl.constexpr = TOTAL
    edge, col = index // total, index % total
    offset = 0
    merged_offset = 0
    for part in tl.static_range(len(WIDTHS)):
        width = WIDTHS[part]
        skip = HEAD if part == 0 else 0
        selected = (edge < E) & (col >= offset) & (col < offset + width)
        local = col - offset
        value = tl.load(Merged + edge * GS0 + (merged_offset + local - skip) * GS1,
                        selected & (local >= skip), other=0.)
        if part == 0 and HEAD:
            extra = tl.load(Extra + edge * ES0 + local * ES1,
                            selected & (local < HEAD), other=0.)
            value += extra
        tl.store(Parts[part] + edge * width + local, value, selected)
        offset += width
        merged_offset += width - skip


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


class _Assemble(torch.autograd.Function):
    @staticmethod
    def forward(ctx, head, *parts):
        ctx.widths = tuple(part.shape[1] for part in parts)
        ctx.head = head
        return assemble(parts, head)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, incoming, extra):
        edges = incoming.shape[0]
        gradients = tuple(incoming.new_empty((edges, width)) for width in ctx.widths)
        if edges:
            _disassemble[(triton.cdiv(edges * sum(ctx.widths), 256),)](
                incoming, extra, gradients, edges, ctx.widths, sum(ctx.widths), ctx.head,
                *incoming.stride(), *extra.stride(), 256,
            )
        return (None, *gradients)


class _SplitInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, widths):
        ctx.widths = widths
        ctx.shape = inputs.shape
        edges, _, channels = inputs.shape
        pieces, start = [], 0
        for width in widths:
            piece = inputs.narrow(1, start // channels, width // channels)
            pieces.append(piece.reshape(edges, width))
            start += width
        return tuple(pieces)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *gradients):
        merged, _ = assemble(gradients, 0)
        return merged.reshape(ctx.shape), None


def frozen_layer(inputs, module, caches, backend):
    edges, _, channels = inputs.shape
    widths = ((module.lmax + 1) * channels, *(
        2 * (module.lmax + 1 - order) * channels for order in range(1, module.mmax + 1)
    ))
    pieces = _SplitInput.apply(inputs, widths)
    outputs = [module.fc_m0(pieces[0])]
    for order, (piece, cache) in enumerate(zip(pieces[1:], caches), 1):
        input_width = (module.lmax + 1 - order) * channels
        value = frozen_so2(piece.reshape(edges, 2, input_width),
                           module.so2_m_linear[order - 1].fc.weight, cache, backend)
        outputs.append(value.reshape(edges, 2 * value.shape[2]))
    head = module.extra_m0_out_channels or 0
    result, extra = _Assemble.apply(head, *outputs)
    result = result.reshape(edges, result.shape[1] // module.num_out_channels, module.num_out_channels)
    return (result, extra) if module.extra_m0_out_channels is not None else result
