"""Low-order UVU message and reduction using the original lab-frame harmonics."""
from e3nn import o3
import torch
import triton
import triton.language as tl


@triton.jit
def _direct_forward(X, H, W, Edge, Y, E: tl.constexpr, XW: tl.constexpr,
                    HW: tl.constexpr, WW: tl.constexpr, YW: tl.constexpr,
                    PATHS: tl.constexpr, C: tl.constexpr, TILE: tl.constexpr):
    edge = tl.program_id(0)
    path_id = tl.program_id(1)
    channel = tl.program_id(2) * TILE + tl.arange(0, TILE)
    src = tl.load(Edge + edge).to(tl.int32)
    dst = tl.load(Edge + E + edge).to(tl.int32)
    valid = channel < C
    for p in tl.static_range(len(PATHS.value)):
        if path_id == p:
            x = X + src * XW + PATHS.value[p][0] + channel * PATHS.value[p][1]
            h = H + edge * HW + PATHS.value[p][2]
            y = Y + dst * YW + PATHS.value[p][4] + channel * PATHS.value[p][5]
            w = tl.load(W + edge * WW + PATHS.value[p][6] + channel, valid, other=0)
            for j in tl.static_range(len(PATHS.value[p][7])):
                value = tl.full((TILE,), 0, tl.float32)
                for k in tl.static_range(len(PATHS.value[p][7][j])):
                    xv = tl.load(x + PATHS.value[p][7][j][k][0], valid, other=0)
                    hv = tl.load(h + PATHS.value[p][7][j][k][1])
                    value += (xv * hv) * PATHS.value[p][7][j][k][2]
                tl.atomic_add(y + j, value * w, valid, sem="relaxed")


@triton.jit
def _direct_backward(X, H, W, Edge, DY, DX, DH, DW, E: tl.constexpr, XW: tl.constexpr,
                     HW: tl.constexpr, WW: tl.constexpr, YW: tl.constexpr,
                     PATHS: tl.constexpr, C: tl.constexpr, TILE: tl.constexpr):
    edge = tl.program_id(0)
    path_id = tl.program_id(1)
    channel = tl.arange(0, TILE)
    src = tl.load(Edge + edge).to(tl.int32)
    dst = tl.load(Edge + E + edge).to(tl.int32)
    valid = channel < C
    for p in tl.static_range(len(PATHS.value)):
        if path_id == p:
            xi = src * XW + PATHS.value[p][0] + channel * PATHS.value[p][1]
            hi = edge * HW + PATHS.value[p][2]
            yi = dst * YW + PATHS.value[p][4] + channel * PATHS.value[p][5]
            wi = edge * WW + PATHS.value[p][6] + channel
            w = tl.load(W + wi, valid, other=0)
            dw = tl.full((TILE,), 0, tl.float32)
            for j in tl.static_range(len(PATHS.value[p][7])):
                dy = tl.load(DY + yi + j, valid, other=0)
                dyw = dy * w
                value = tl.full((TILE,), 0, tl.float32)
                for k in tl.static_range(len(PATHS.value[p][7][j])):
                    xv = tl.load(X + xi + PATHS.value[p][7][j][k][0], valid, other=0)
                    hv = tl.load(H + hi + PATHS.value[p][7][j][k][1])
                    coefficient = PATHS.value[p][7][j][k][2]
                    value += (xv * hv) * coefficient
                    d_term = dyw * coefficient
                    tl.atomic_add(DX + xi + PATHS.value[p][7][j][k][0],
                                  d_term * hv, valid, sem="relaxed")
                    tl.atomic_add(DH + hi + PATHS.value[p][7][j][k][1],
                                  tl.sum(d_term * xv, 0), sem="relaxed")
                dw += dy * value
            tl.store(DW + wi, dw, valid)


class _DirectMessage(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h, weights, edge, paths, channels, output_width):
        x, h, weights, edge = (value.contiguous() for value in (x, h, weights, edge))
        output = x.new_zeros((x.shape[0], output_width))
        edges = h.shape[0]
        if edges:
            _direct_forward[(edges, len(paths), triton.cdiv(channels, 32))](
                x, h, weights, edge, output, edges, x.shape[1], h.shape[1],
                weights.shape[1], output_width, paths, channels, 32,
                num_warps=1, enable_fp_fusion=False,
            )
        ctx.save_for_backward(x, h, weights, edge)
        ctx.paths = paths
        ctx.channels = channels
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        x, h, weights, edge = ctx.saved_tensors
        dx, dh, dw = torch.zeros_like(x), torch.zeros_like(h), torch.empty_like(weights)
        if h.shape[0]:
            tile = max(32, triton.next_power_of_2(ctx.channels))
            _direct_backward[(h.shape[0], len(ctx.paths))](
                x, h, weights, edge, dy.contiguous(), dx, dh, dw, h.shape[0],
                x.shape[1], h.shape[1], weights.shape[1], dy.shape[1],
                ctx.paths, ctx.channels, tile, num_warps=min(8, tile // 32),
                enable_fp_fusion=False,
            )
        return dx, dh, dw, None, None, None, None


class DirectMessage(torch.nn.Module):
    """Keep the original low-order path and avoid the full per-edge messages."""
    def __init__(self, product):
        super().__init__()
        source = product.irreps_in1.slices()
        harmonic = product.irreps_in2.slices()
        destination = product.irreps_out.slices()
        self.channels = product.irreps_in1[0].mul
        self.output_width = product.irreps_out.dim
        self.reference = product
        paths, weight = [], 0
        for inst in product.instructions:
            mi, ri = product.irreps_in1[inst.i_in1]
            mh, rh = product.irreps_in2[inst.i_in2]
            mo, ro = product.irreps_out[inst.i_out]
            if (mi != self.channels or mo != self.channels or mh != 1 or not inst.has_weight
                    or inst.connection_mode != "uvu" or inst.path_shape != (self.channels, 1)
                    or ri.l > 1):
                raise ValueError("Expected the frozen checkpoint's low-order UVU interaction")
            cg = o3.wigner_3j(ri.l, rh.l, ro.l, dtype=torch.float64) * inst.path_weight
            terms = tuple(
                tuple((a, b, float(cg[a, b, j])) for a, b in cg[:, :, j].nonzero().tolist())
                for j in range(ro.dim)
            )
            paths.append((source[inst.i_in1].start, ri.dim, harmonic[inst.i_in2].start,
                          rh.dim, destination[inst.i_out].start, ro.dim, weight, terms))
            weight += self.channels
        if weight != product.weight_numel:
            raise ValueError("Original radial weight coverage mismatch")
        self.paths = tuple(paths)

    def forward(self, x, harmonics, weights, edge_index):
        if x.is_cuda and x.dtype == torch.float32:
            return _DirectMessage.apply(x, harmonics, weights, edge_index,
                                        self.paths, self.channels, self.output_width)
        source, target = edge_index
        message = self.reference(x[source], harmonics, weights)
        return x.new_zeros((x.shape[0], self.output_width)).index_add(0, target, message)


def configure_direct(model):
    from candidate import configure_candidate

    model = configure_candidate(model, "hp-scalar")
    layer = model.interactions[1]
    layer.conv_tp = DirectMessage(layer.conv_tp)
    layer.conv_fusion = True
    return model
