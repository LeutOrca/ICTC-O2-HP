"""Bounded UVU kernels copied from the sealed upstream with pre-multiply int64 addressing."""
import triton
import triton.language as tl

@triton.jit
def _path(info, coefficients, p):
    p = p.to(tl.int64)
    xoff = tl.load(info + p * 7)
    yoff = tl.load(info + p * 7 + 1)
    outoff = tl.load(info + p * 7 + 2)
    dx = tl.load(info + p * 7 + 3)
    dy = tl.load(info + p * 7 + 4)
    dout = tl.load(info + p * 7 + 5)
    woff = tl.load(info + p * 7 + 6)
    c0 = tl.load(coefficients + p * 8)
    c1 = tl.load(coefficients + p * 8 + 1)
    c2 = tl.load(coefficients + p * 8 + 2)
    c3 = tl.load(coefficients + p * 8 + 3)
    c4 = tl.load(coefficients + p * 8 + 4)
    c5 = tl.load(coefficients + p * 8 + 5)
    c6 = tl.load(coefficients + p * 8 + 6)
    c7 = tl.load(coefficients + p * 8 + 7)
    return xoff, yoff, outoff, dx, dy, dout, woff, c0, c1, c2, c3, c4, c5, c6, c7


@triton.jit
def _forward(x, y, w, info, coefficients, out, N: tl.constexpr,
             WX: tl.constexpr, WY: tl.constexpr, U: tl.constexpr, V: tl.constexpr,
             P: tl.constexpr, WNUM: tl.constexpr, SHARED: tl.constexpr,
             B: tl.constexpr, BV: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    row, u = offsets // U, offsets % U
    mask = row < N
    entry = P + tl.program_id(1).to(tl.int64)
    outoff = tl.load(info + entry * 7)
    dout = tl.load(info + entry * 7 + 1)
    begin = tl.load(info + entry * 7 + 2)
    end = tl.load(info + entry * 7 + 3)
    v = tl.arange(0, BV)
    acc0, acc1 = tl.full((B,), 0, tl.float32), tl.full((B,), 0, tl.float32)
    for p in range(begin, end):
        xo, yo, oo, dx, dy, do, wo, c0, c1, c2, c3, c4, c5, c6, c7 = _path(info, coefficients, p)
        a0 = tl.load(x + row * WX + xo + u * dx, mask, 0)
        a1 = tl.load(x + row * WX + xo + u * dx + 1, mask & (dx == 2), 0)
        b0 = tl.load(y + row[:, None] * WY + yo + v[None, :] * dy,
                     mask[:, None] & (v[None, :] < V), 0)
        b1 = tl.load(y + row[:, None] * WY + yo + v[None, :] * dy + 1,
                     mask[:, None] & (v[None, :] < V) & (dy == 2), 0)
        weight_offset = wo + u[:, None] * V + v[None, :]
        if not SHARED:
            weight_offset += row[:, None] * WNUM
        weights = tl.load(w + weight_offset, mask[:, None] & (v[None, :] < V), 0)
        r0, r1 = tl.sum(weights * b0, 1), tl.sum(weights * b1, 1)
        acc0 += a0 * (c0 * r0 + c1 * r1) + a1 * (c2 * r0 + c3 * r1)
        acc1 += a0 * (c4 * r0 + c5 * r1) + a1 * (c6 * r0 + c7 * r1)
    tl.store(out + row * WX + outoff + u * dout, acc0, mask)
    tl.store(out + row * WX + outoff + u * dout + 1, acc1, mask & (dout == 2))


@triton.jit
def _input_grad(x, y, w, g, info, coefficients, order, dest, N: tl.constexpr,
                WX: tl.constexpr, WY: tl.constexpr, U: tl.constexpr, V: tl.constexpr,
                P: tl.constexpr, NO: tl.constexpr, NX: tl.constexpr,
                WNUM: tl.constexpr, SHARED: tl.constexpr, FIRST: tl.constexpr,
                B: tl.constexpr, BR: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    if FIRST:
        row, channel = offsets // U, offsets % U
        entry = P + NO + tl.program_id(1).to(tl.int64)
        width = WX
        reduction = V
    else:
        row, channel = offsets // V, offsets % V
        entry = P + NO + NX + tl.program_id(1).to(tl.int64)
        width = WY
        reduction = U
    mask = row < N
    offset = tl.load(info + entry * 7)
    dim = tl.load(info + entry * 7 + 1)
    begin = tl.load(info + entry * 7 + 2)
    end = tl.load(info + entry * 7 + 3)
    r = tl.arange(0, BR)
    active = mask[:, None] & (r[None, :] < reduction)
    acc0, acc1 = tl.full((B,), 0, tl.float32), tl.full((B,), 0, tl.float32)
    for q in range(begin, end):
        p = tl.load(order + q)
        xo, yo, oo, dx, dy, do, wo, c0, c1, c2, c3, c4, c5, c6, c7 = _path(info, coefficients, p)
        if FIRST:
            b0 = tl.load(y + row[:, None] * WY + yo + r[None, :] * dy, active, 0)
            b1 = tl.load(y + row[:, None] * WY + yo + r[None, :] * dy + 1, active & (dy == 2), 0)
            wi = wo + channel[:, None] * V + r[None, :]
            if not SHARED:
                wi += row[:, None] * WNUM
            weights = tl.load(w + wi, active, 0)
            b0, b1 = tl.sum(weights * b0, 1), tl.sum(weights * b1, 1)
            g0 = tl.load(g + row * WX + oo + channel * do, mask, 0)
            g1 = tl.load(g + row * WX + oo + channel * do + 1, mask & (do == 2), 0)
            acc0 += g0 * (c0 * b0 + c1 * b1) + g1 * (c4 * b0 + c5 * b1)
            acc1 += g0 * (c2 * b0 + c3 * b1) + g1 * (c6 * b0 + c7 * b1)
        else:
            a0 = tl.load(x + row[:, None] * WX + xo + r[None, :] * dx, active, 0)
            a1 = tl.load(x + row[:, None] * WX + xo + r[None, :] * dx + 1, active & (dx == 2), 0)
            g0 = tl.load(g + row[:, None] * WX + oo + r[None, :] * do, active, 0)
            g1 = tl.load(g + row[:, None] * WX + oo + r[None, :] * do + 1, active & (do == 2), 0)
            wi = wo + r[None, :] * V + channel[:, None]
            if not SHARED:
                wi += row[:, None] * WNUM
            weights = tl.load(w + wi, active, 0)
            b0 = a0 * (c0 * g0 + c4 * g1) + a1 * (c2 * g0 + c6 * g1)
            b1 = a0 * (c1 * g0 + c5 * g1) + a1 * (c3 * g0 + c7 * g1)
            acc0 += tl.sum(weights * b0, 1)
            acc1 += tl.sum(weights * b1, 1)
    tl.store(dest + row * width + offset + channel * dim, acc0, mask)
    tl.store(dest + row * width + offset + channel * dim + 1, acc1, mask & (dim == 2))


@triton.jit
def _weight_unshared(x, y, g, info, coefficients, dest, N: tl.constexpr,
                     WX: tl.constexpr, WY: tl.constexpr, U: tl.constexpr, V: tl.constexpr,
                     WNUM: tl.constexpr, B: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    row, uv = offsets // (U * V), offsets % (U * V)
    u, v = uv // V, uv % V
    active = row < N
    p = tl.program_id(1).to(tl.int64)
    xo, yo, oo, dx, dy, do, wo, c0, c1, c2, c3, c4, c5, c6, c7 = _path(info, coefficients, p)
    a0 = tl.load(x + row * WX + xo + u * dx, active, 0)
    a1 = tl.load(x + row * WX + xo + u * dx + 1, active & (dx == 2), 0)
    b0 = tl.load(y + row * WY + yo + v * dy, active, 0)
    b1 = tl.load(y + row * WY + yo + v * dy + 1, active & (dy == 2), 0)
    g0 = tl.load(g + row * WX + oo + u * do, active, 0)
    g1 = tl.load(g + row * WX + oo + u * do + 1, active & (do == 2), 0)
    r0 = a0 * (c0 * g0 + c4 * g1) + a1 * (c2 * g0 + c6 * g1)
    r1 = a0 * (c1 * g0 + c5 * g1) + a1 * (c3 * g0 + c7 * g1)
    tl.store(dest + row * WNUM + wo + uv, r0 * b0 + r1 * b1, active)
