"""Triton local O(2) UVU convolution for MACE frozen E/F inference.

Apply the existing e3nn path weights and the order-zero CG slice. Keep
edge-local intermediates inside Triton and differentiate both rotations.
"""
import math

from e3nn import o3
from eqx_geometry.wigner import WignerD
import torch
import triton
import triton.language as tl


@triton.jit
def _rotate(M, V, dim: tl.constexpr, row: tl.constexpr, valid):
    value = tl.full(valid.shape, 0, tl.float32)
    for column in tl.static_range(dim):
        matrix = tl.load(M + row * dim + column)
        vector = tl.load(V + column, valid, other=0)
        value += matrix * vector
    return value


@triton.jit
def _message_path(x, din_ptr, dout_ptr, w, y, valid,
                  dim_in: tl.constexpr, dim_out: tl.constexpr,
                  terms: tl.constexpr, tile: tl.constexpr):
    for m in tl.static_range(dim_out):
        value = tl.full((tile,), 0, tl.float32)
        for k in tl.static_range(len(terms.value)):
            x_local = _rotate(din_ptr, x, dim_in, terms.value[k][0], valid)
            rotation = tl.load(dout_ptr + tl.constexpr(terms.value[k][1]) * dim_out + m)
            value += ((x_local * terms.value[k][2]) * w) * rotation
        tl.atomic_add(y + m, value, valid, sem="relaxed")


@triton.jit
def _local_o2_forward(X, D, W, Edge, Y, E: tl.constexpr, XW: tl.constexpr,
                      DW: tl.constexpr, WW: tl.constexpr, YW: tl.constexpr,
                      PATHS: tl.constexpr, C: tl.constexpr, TILE: tl.constexpr):
    edge = tl.program_id(0)
    path_id = tl.program_id(1)
    channel = tl.program_id(2) * TILE + tl.arange(0, TILE)
    src = tl.load(Edge + edge).to(tl.int32)
    dst = tl.load(Edge + E + edge).to(tl.int32)
    valid = channel < C
    for p in tl.static_range(len(PATHS.value)):
        if path_id == p:
            x = X + src * XW + PATHS.value[p][0] + channel * PATHS.value[p][2]
            y = Y + dst * YW + PATHS.value[p][3] + channel * PATHS.value[p][5]
            din_ptr = D + edge * DW + PATHS.value[p][1]
            dout_ptr = D + edge * DW + PATHS.value[p][4]
            w = tl.load(W + edge * WW + PATHS.value[p][6] + channel, valid, other=0)
            _message_path(x, din_ptr, dout_ptr, w, y, valid,
                          PATHS.value[p][2], PATHS.value[p][5], tl.constexpr(PATHS.value[p][7]), TILE)


@triton.jit
def _message_path_vjp(x, din_ptr, dout_ptr, w, dy, dx, ddin, ddout, dw,
                      valid, dim_in: tl.constexpr, dim_out: tl.constexpr,
                      terms: tl.constexpr, tile: tl.constexpr):
    weight_grad = tl.full((tile,), 0, tl.float32)
    for k in tl.static_range(len(terms.value)):
        x_local = _rotate(din_ptr, x, dim_in, terms.value[k][0], valid)
        dy_local = _rotate(dout_ptr, dy, dim_out, terms.value[k][1], valid)
        weight_grad += (dy_local * terms.value[k][2]) * x_local
        gx = (dy_local * terms.value[k][2]) * w
        gy = (x_local * terms.value[k][2]) * w
        for i in tl.static_range(dim_in):
            xv = tl.load(x + i, valid, other=0)
            di = tl.sum(gx * xv, 0)
            tl.atomic_add(ddin + tl.constexpr(terms.value[k][0]) * dim_in + i, di, sem="relaxed")
        for j in tl.static_range(dim_out):
            dyv = tl.load(dy + j, valid, other=0)
            do = tl.sum(gy * dyv, 0)
            tl.atomic_add(ddout + tl.constexpr(terms.value[k][1]) * dim_out + j, do, sem="relaxed")
    for i in tl.static_range(dim_in):
        node_grad = tl.full((tile,), 0, tl.float32)
        for k in tl.static_range(len(terms.value)):
            dy_local = _rotate(dout_ptr, dy, dim_out, terms.value[k][1], valid)
            gx = (dy_local * terms.value[k][2]) * w
            node_grad += gx * tl.load(din_ptr + tl.constexpr(terms.value[k][0]) * dim_in + i)
        tl.atomic_add(dx + i, node_grad, valid, sem="relaxed")
    tl.store(dw, weight_grad, valid)


@triton.jit
def _local_o2_backward(X, D, W, Edge, DY, DX, DD, DWeight,
                       E: tl.constexpr, XW: tl.constexpr, DW: tl.constexpr,
                       WW: tl.constexpr, YW: tl.constexpr, PATHS: tl.constexpr,
                       C: tl.constexpr, TILE: tl.constexpr):
    edge = tl.program_id(0)
    path_id = tl.program_id(1)
    channel = tl.arange(0, TILE)
    src = tl.load(Edge + edge).to(tl.int32)
    dst = tl.load(Edge + E + edge).to(tl.int32)
    valid = channel < C
    for p in tl.static_range(len(PATHS.value)):
        if path_id == p:
            xi = src * XW + PATHS.value[p][0] + channel * PATHS.value[p][2]
            yi = dst * YW + PATHS.value[p][3] + channel * PATHS.value[p][5]
            di = edge * DW + PATHS.value[p][1]
            do = edge * DW + PATHS.value[p][4]
            wi = edge * WW + PATHS.value[p][6] + channel
            w = tl.load(W + wi, valid, other=0)
            _message_path_vjp(X + xi, D + di, D + do, w, DY + yi, DX + xi,
                              DD + di, DD + do, DWeight + wi, valid,
                              PATHS.value[p][2], PATHS.value[p][5], tl.constexpr(PATHS.value[p][7]), TILE)


class _LocalO2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, frame, weights, edge_index, paths, channels, output_width):
        x, frame, weights, edge_index = (v.contiguous() for v in (x, frame, weights, edge_index))
        output = x.new_zeros((x.shape[0], output_width))
        edges = frame.shape[0]
        if edges:
            _local_o2_forward[(edges, len(paths), triton.cdiv(channels, 32))](
                x, frame, weights, edge_index, output, edges, x.shape[1],
                frame.shape[1], weights.shape[1], output_width, paths, channels, 32,
                num_warps=1, enable_fp_fusion=False,
            )
        ctx.save_for_backward(x, frame, weights, edge_index)
        ctx.path_metadata = (paths, channels)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        x, frame, weights, edge_index = ctx.saved_tensors
        paths, channels = ctx.path_metadata
        dx, df, dw = torch.zeros_like(x), torch.zeros_like(frame), torch.empty_like(weights)
        edges = frame.shape[0]
        if edges:
            tile = max(32, triton.next_power_of_2(channels))
            _local_o2_backward[(edges, len(paths))](
                x, frame, weights, edge_index, grad_output.contiguous(), dx, df, dw,
                edges, x.shape[1], frame.shape[1], weights.shape[1],
                grad_output.shape[1], paths, channels, tile, num_warps=min(8, tile // 32),
                enable_fp_fusion=False,
            )
        return dx, df, dw, None, None, None, None


class SharedGeometry(torch.nn.Module):
    def __init__(self, harmonics, degree):
        super().__init__()
        if not harmonics.normalize or harmonics.normalization != "component":
            raise ValueError("Expected normalized, component-scaled harmonics")
        self.harmonics = harmonics
        self.wigner = WignerD(degree, degree)

    def forward(self, edge_vector):
        harmonics = self.harmonics(edge_vector)
        frame = self.wigner.forward_packed(edge_vector, method="recursive")
        return harmonics, frame


class LocalO2Message(torch.nn.Module):
    """Preserve the canonical order, parity and scaling of all original paths."""
    def __init__(self, product):
        super().__init__()
        inputs = product.irreps_in1.slices()
        outputs = product.irreps_out.slices()
        channels = product.irreps_in1[0].mul
        degree = max(product.irreps_in1.lmax, product.irreps_out.lmax)
        offsets = [sum((2 * j + 1) ** 2 for j in range(l)) for l in range(degree + 1)]
        paths, weight = [], 0
        for ins in product.instructions:
            mi, ir_in = product.irreps_in1[ins.i_in1]
            mh, ir_h = product.irreps_in2[ins.i_in2]
            mo, ir_out = product.irreps_out[ins.i_out]
            if (mi != channels or mo != channels or mh != 1 or not ins.has_weight
                    or ins.connection_mode != "uvu" or ins.path_shape != (channels, 1)
                    or ir_h.p != (-1) ** ir_h.l):
                raise ValueError("Unsupported local UVU checkpoint path")
            cg = o3.wigner_3j(ir_in.l, ir_h.l, ir_out.l, dtype=torch.float64)
            cg = cg[:, ir_h.l, :] * (math.sqrt(ir_h.dim) * ins.path_weight)
            terms = tuple((m, n, float(cg[m, n])) for m, n in cg.nonzero().tolist())
            if not terms:
                raise ValueError("Empty order-zero slice")
            paths.append((inputs[ins.i_in1].start, offsets[ir_in.l], ir_in.dim,
                          outputs[ins.i_out].start, offsets[ir_out.l], ir_out.dim,
                          weight, terms))
            weight += channels
        if weight != product.weight_numel:
            raise ValueError("Canonical path weight coverage mismatch")
        self.paths = tuple(paths)
        self.channels = channels
        self.output_width = product.irreps_out.dim
        self.degree = degree

    def forward(self, x, edge_attrs, weights, edge_index):
        _, frame = edge_attrs
        if x.is_cuda and x.dtype == torch.float32:
            return _LocalO2.apply(x, frame, weights, edge_index, self.paths,
                                  self.channels, self.output_width)
        source, target = edge_index
        result = x.new_zeros((x.size(0), self.output_width))
        for xi, di, ni, yi, do, no, wi, terms in self.paths:
            x_lab = x[source, xi:xi + self.channels * ni].reshape(-1, self.channels, ni)
            din = frame[:, di:di + ni * ni].reshape(-1, ni, ni)
            dout = frame[:, do:do + no * no].reshape(-1, no, no)
            x_local = x_lab @ din.transpose(1, 2)
            cg = x.new_zeros((ni, no))
            for a, b, coefficient in terms:
                cg[a, b] = coefficient
            w = weights[:, wi:wi + self.channels, None]
            message = ((x_local @ cg) * w) @ dout
            width = self.channels * no
            reduced = message.new_zeros((x.size(0), width)).index_add(0, target, message.flatten(1))
            result[:, yi:yi + width] = result[:, yi:yi + width] + reduced
        return result


def configure_local_o2(model):
    from candidate import configure_candidate

    model = configure_candidate(model, "hp-scalar")
    second = model.interactions[1]
    second.conv_tp = LocalO2Message(second.conv_tp)
    second.conv_fusion = True
    harmonics = model.spherical_harmonics
    model.spherical_harmonics = SharedGeometry(harmonics, second.conv_tp.degree).to(
        next(model.parameters())
    )

    def take_harmonics(module, args):
        node_feats, edge_attrs, weights, edge_index = args
        return node_feats, edge_attrs[0], weights, edge_index

    model.interactions[0].conv_tp.register_forward_pre_hook(take_harmonics)
    return model
