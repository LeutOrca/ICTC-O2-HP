"""Fused frozen-weight EquiformerV3 merge normalization and input adjoint."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _warp_sum_ascending(values):
    rows: tl.constexpr = values.shape[0]
    lane = tl.arange(0, 32)
    for offset in tl.static_range(5):
        other = tl.broadcast_to(((lane + (1 << offset)) % 32)[None, :], (rows, 32))
        values = values + tl.gather(values, other, axis=1)
    return tl.sum(tl.where(lane[None, :] == 0, values, 0.), 1)


@triton.jit
def _mean128(values):
    # ATen Reduce.cuh: four 32-strided accumulators, then ascending warp shuffles.
    rows: tl.constexpr = values.shape[0]
    lane = tl.arange(0, 32)
    total = tl.gather(values, tl.broadcast_to(lane[None, :], (rows, 32)), axis=1)
    for part in tl.static_range(1, 4):
        indices = tl.broadcast_to((lane + part * 32)[None, :], (rows, 32))
        total = total + tl.gather(values, indices, axis=1)
    return _warp_sum_ascending(total) / 128.


@triton.jit
def _merge_forward(X, Weight, Bias, Degree, Balance, Y, Stats,
                   R: tl.constexpr, C: tl.constexpr,
                   S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                   EPS: tl.constexpr, CENTER: tl.constexpr, AFFINE: tl.constexpr,
                   BR: tl.constexpr, BC: tl.constexpr):
    node = tl.program_id(0)
    row = tl.arange(0, BR)
    col = tl.arange(0, BC)
    x = tl.load(X + node * S0 + row[:, None] * S1 + col[None, :] * S2,
                (row[:, None] < R) & (col[None, :] < C), other=0.)
    if C == 128:
        means = _mean128(x)
        mean = tl.sum(tl.where(row == 0, means, 0.), 0)
    else:
        mean = tl.sum(tl.where(row[:, None] == 0, x, 0.), 0)
        mean = tl.sum(mean, 0) / C
    if CENTER:
        z = x - tl.where((row[:, None] == 0) & (col[None, :] < C), mean, 0.)
    else:
        z = x
    balance = tl.load(Balance + row, row < R, other=0.)
    if C == 128:
        variance = tl.sum(_mean128(z * z) * balance, 0)
    else:
        variance = tl.sum(tl.sum(z * z, 1) / C * balance, 0)
    inverse = tl.rsqrt(variance + EPS)
    coefficient = tl.full((BR, BC), 1., tl.float32)
    if AFFINE:
        degree = tl.load(Degree + row, row < R, other=0)
        coefficient = tl.load(Weight + degree[:, None] * C + col[None, :],
                              (row[:, None] < R) & (col[None, :] < C), other=0.)
    result = z * (inverse * coefficient)
    if CENTER and AFFINE:
        bias = tl.load(Bias + col, col < C, other=0.)
        result += tl.where(row[:, None] == 0, bias[None, :], 0.)
    tl.store(Y + (node * R + row[:, None]) * C + col[None, :], result,
             (row[:, None] < R) & (col[None, :] < C))
    tl.store(Stats + node * 3, mean)
    tl.store(Stats + node * 3 + 1, inverse)
    tl.store(Stats + node * 3 + 2, variance + EPS)


@triton.jit
def _merge_backward(X, DY, Weight, Degree, Balance, Stats, DX,
                    R: tl.constexpr, C: tl.constexpr,
                    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                    D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr,
                    CENTER: tl.constexpr, AFFINE: tl.constexpr,
                    BR: tl.constexpr, BC: tl.constexpr):
    node = tl.program_id(0)
    row = tl.arange(0, BR)
    col = tl.arange(0, BC)
    mask = (row[:, None] < R) & (col[None, :] < C)
    x = tl.load(X + node * S0 + row[:, None] * S1 + col[None, :] * S2, mask, other=0.)
    mean = tl.load(Stats + node * 3)
    inverse = tl.load(Stats + node * 3 + 1)
    variance_with_epsilon = tl.load(Stats + node * 3 + 2)
    if CENTER:
        z = x - tl.where((row[:, None] == 0) & (col[None, :] < C), mean, 0.)
    else:
        z = x
    incoming = tl.load(DY + node * D0 + row[:, None] * D1 + col[None, :] * D2, mask, other=0.)
    coefficient = tl.full((BR, BC), 1., tl.float32)
    if AFFINE:
        degree = tl.load(Degree + row, row < R, other=0)
        coefficient = tl.load(Weight + degree[:, None] * C + col[None, :], mask, other=0.)
    if C == 128:
        # Native sum_to_size reduces flattened [R,128] with vector loads.
        columns = tl.arange(0, 128)
        partial = tl.full((128,), 0., tl.float32)
        for component in range(R):
            raw = tl.load(X + node * S0 + component * S1 + columns * S2)
            if CENTER:
                raw = raw - tl.where(component == 0, mean, 0.)
            grad = tl.load(DY + node * D0 + component * D1 + columns * D2)
            affine = tl.full((128,), 1., tl.float32)
            if AFFINE:
                level = tl.load(Degree + component)
                affine = tl.load(Weight + level * C + columns)
            partial = partial + (grad * raw) * affine
        lane = tl.arange(0, 32)
        packed = tl.gather(partial, lane * 4, axis=0)
        for part in tl.static_range(1, 4):
            packed = packed + tl.gather(partial, lane * 4 + part, axis=0)
        projection = tl.sum(_warp_sum_ascending(packed[None, :]), 0)
    else:
        projection = tl.sum(tl.sum((incoming * z) * coefficient, 1), 0)
    balance = tl.load(Balance + row, row < R, other=0.)
    variance_gradient = projection * (-0.5 * libdevice.pow(variance_with_epsilon, -1.5))
    square_gradient = (variance_gradient * balance[:, None]) / C
    gradient = incoming * (inverse * coefficient) + square_gradient * (2. * z)
    if CENTER:
        if C == 128:
            scalar_sum = tl.sum(tl.where(row == 0, _mean128(gradient), 0.), 0)
        else:
            scalar_sum = tl.sum(tl.sum(tl.where(row[:, None] == 0, gradient, 0.), 0), 0) / C
        gradient -= tl.where((row[:, None] == 0) & (col[None, :] < C), scalar_sum, 0.)
    tl.store(DX + (node * R + row[:, None]) * C + col[None, :], gradient, mask)


class _MergeNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias, degree, balance, epsilon, centering, affine):
        nodes, rows, channels = inputs.shape
        result = torch.empty_like(inputs, memory_format=torch.contiguous_format)
        stats = inputs.new_empty((nodes, 3))
        br, bc = triton.next_power_of_2(rows), triton.next_power_of_2(channels)
        if nodes:
            _merge_forward[(nodes,)](
                inputs, weight, bias, degree, balance, result, stats,
                rows, channels, *inputs.stride(), epsilon, centering, affine, br, bc,
                num_warps=4, enable_fp_fusion=False,
            )
        ctx.save_for_backward(inputs, weight, degree, balance, stats)
        ctx.centering, ctx.affine = centering, affine
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, derivative):
        inputs, weight, degree, balance, stats = ctx.saved_tensors
        nodes, rows, channels = inputs.shape
        result = torch.empty_like(inputs, memory_format=torch.contiguous_format)
        if nodes:
            _merge_backward[(nodes,)](
                inputs, derivative, weight, degree, balance, stats, result,
                rows, channels, *inputs.stride(), *derivative.stride(),
                ctx.centering, ctx.affine, triton.next_power_of_2(rows),
                triton.next_power_of_2(channels), num_warps=4, enable_fp_fusion=False,
            )
        return result, None, None, None, None, None, None, None


def merge_norm(inputs, module):
    if inputs.dtype != torch.float32 or not inputs.is_cuda or inputs.ndim != 3:
        raise ValueError("Expected CUDA FP32 [nodes, components, channels]")
    rows, channels = inputs.shape[1:]
    if rows != (module.lmax + 1) ** 2 or channels != module.num_channels:
        raise ValueError("Input shape does not match the frozen norm")
    if any(value.requires_grad for value in module.parameters()):
        raise ValueError("The fused input-adjoint norm requires frozen affine parameters")
    if not hasattr(module, "_native_norm_schedule"):
        degrees = [level for level in range(module.lmax + 1) for _ in range(2 * level + 1)]
        degree = torch.tensor(degrees, device=inputs.device, dtype=torch.int32)
        if module.normalization == "norm":
            balance = inputs.new_ones(rows)
        elif module.std_balance_degrees:
            balance = module.balance_degree_weight.reshape(-1)
        else:
            balance = inputs.new_full((rows,), 1. / rows)
        module._native_norm_schedule = (degree, balance)
    degree, balance = module._native_norm_schedule
    if degree.device != inputs.device:
        raise ValueError("Rebuild the norm schedule after moving the model")
    weight = module.affine_weight if module.affine else inputs
    bias = module.affine_bias if module.affine and module.centering else inputs
    return _MergeNorm.apply(inputs, weight, bias, degree, balance,
                           module.eps, module.centering, module.affine)
