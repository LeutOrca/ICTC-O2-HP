"""Strict-FP32 Triton spatial message fusion for frozen MACE inference."""
import torch
from e3nn import o3
import triton
import triton.language as tl


@triton.jit
def _scalar_message_forward(X, H, W, Edges, Y, E: tl.constexpr,
                            C: tl.constexpr, H_WIDTH: tl.constexpr,
                            W_WIDTH: tl.constexpr, Y_WIDTH: tl.constexpr,
                            PATHS: tl.constexpr, TILE: tl.constexpr):
    edge = tl.program_id(0)
    channel = tl.program_id(1) * TILE + tl.arange(0, TILE)
    src = tl.load(Edges + edge).to(tl.int32)
    dst = tl.load(Edges + E + edge).to(tl.int32)
    x = tl.load(X + src * C + channel, channel < C, other=0)
    for path in tl.static_range(len(PATHS.value)):
        w = tl.load(W + edge * W_WIDTH + PATHS.value[path][1] + channel, channel < C, other=0)
        for m in tl.static_range(tl.constexpr(PATHS.value[path][3])):
            h = tl.load(H + edge * H_WIDTH + PATHS.value[path][0] + m)
            value = ((PATHS.value[path][4] * h) * x) * w
            tl.atomic_add(Y + dst * Y_WIDTH + PATHS.value[path][2] + channel * PATHS.value[path][3] + m,
                          value, channel < C, sem="relaxed")


@triton.jit
def _scalar_message_backward(X, H, W, Edges, DY, DX, DH, DW, E: tl.constexpr,
                             C: tl.constexpr, H_WIDTH: tl.constexpr,
                             W_WIDTH: tl.constexpr, Y_WIDTH: tl.constexpr,
                             PATHS: tl.constexpr, TILE: tl.constexpr):
    edge = tl.program_id(0)
    channel = tl.arange(0, TILE)
    src = tl.load(Edges + edge).to(tl.int32)
    dst = tl.load(Edges + E + edge).to(tl.int32)
    x = tl.load(X + src * C + channel, channel < C, other=0)
    dx = tl.full((TILE,), 0, tl.float32)
    for path in tl.static_range(len(PATHS.value)):
        w = tl.load(W + edge * W_WIDTH + PATHS.value[path][1] + channel, channel < C, other=0)
        dw = tl.full((TILE,), 0, tl.float32)
        for m in tl.static_range(tl.constexpr(PATHS.value[path][3])):
            h = tl.load(H + edge * H_WIDTH + PATHS.value[path][0] + m)
            dy = tl.load(DY + dst * Y_WIDTH + PATHS.value[path][2] + channel * PATHS.value[path][3] + m,
                         channel < C, other=0)
            angular_weight = dy * (PATHS.value[path][4] * h)
            dx += angular_weight * w
            dw += angular_weight * x
            dh = tl.sum((dy * w) * x, 0) * PATHS.value[path][4]
            tl.store(DH + edge * H_WIDTH + PATHS.value[path][0] + m, dh)
        tl.store(DW + edge * W_WIDTH + PATHS.value[path][1] + channel, dw, channel < C)
    tl.atomic_add(DX + src * C + channel, dx, channel < C, sem="relaxed")


class _ScalarMessage(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, harmonics, weights, edge_index, paths, output_width):
        x, harmonics, weights, edge_index = (
            tensor.contiguous() for tensor in (x, harmonics, weights, edge_index)
        )
        output = x.new_zeros((x.shape[0], output_width))
        edges, channels = harmonics.shape[0], x.shape[1]
        if edges:
            _scalar_message_forward[(edges, triton.cdiv(channels, 32))](
                x, harmonics, weights, edge_index, output, edges, channels,
                harmonics.shape[1], weights.shape[1], output_width, paths, 32,
                num_warps=1, enable_fp_fusion=False,
            )
        ctx.save_for_backward(x, harmonics, weights, edge_index)
        ctx.paths = paths
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        x, harmonics, weights, edge_index = ctx.saved_tensors
        dx = torch.zeros_like(x)
        dh, dw = torch.empty_like(harmonics), torch.empty_like(weights)
        edges, channels = harmonics.shape[0], x.shape[1]
        if edges:
            tile = max(32, triton.next_power_of_2(channels))
            _scalar_message_backward[(edges,)](
                x, harmonics, weights, edge_index, grad_output.contiguous(),
                dx, dh, dw, edges, channels, harmonics.shape[1], weights.shape[1],
                grad_output.shape[1], ctx.paths, tile,
                num_warps=min(8, tile // 32), enable_fp_fusion=False,
            )
        return dx, dh, dw, None, None, None


class ScalarMessage(torch.nn.Module):
    """Scalar sender times a single copy of each angular order, then scatter.

    The local O(2) sender has only order zero. The output frame rotation is
    equivalent to multiplying by the original lab-frame harmonic, so neither
    frame materialization nor per-edge output storage is needed in this layer.
    """

    def __init__(self, product):
        super().__init__()
        source_mul, source_ir = product.irreps_in1[0]
        if len(product.irreps_in1) != 1 or source_ir != o3.Irrep("0e"):
            raise ValueError("The first MACE message must have scalar senders")
        harmonics = product.irreps_in2.slices()
        outputs = product.irreps_out.slices()
        paths = []
        weight_start = 0
        for ins in product.instructions:
            angular_mul, angular_ir = product.irreps_in2[ins.i_in2]
            output_mul, output_ir = product.irreps_out[ins.i_out]
            if (ins.i_in1 != 0 or ins.connection_mode != "uvu" or not ins.has_weight
                    or angular_mul != 1 or output_mul != source_mul or angular_ir != output_ir
                    or ins.path_shape != (source_mul, 1)):
                raise ValueError("Expected checkpoint canonical scalar × harmonic UVU paths")
            scale = ins.path_weight * float(
                o3.wigner_3j(0, angular_ir.l, output_ir.l, dtype=torch.float64)[0, 0, 0]
            )
            paths.append((harmonics[ins.i_in2].start, weight_start, outputs[ins.i_out].start,
                          angular_ir.dim, scale))
            weight_start += source_mul
        harmonic_coverage = sorted(h + index for h, _, _, dim, _ in paths for index in range(dim))
        if harmonic_coverage != list(range(product.irreps_in2.dim)) or weight_start != product.weight_numel:
            raise ValueError("Triton scalar message requires exact harmonic and weight coverage")
        self.paths = tuple(paths)
        self.output_width = product.irreps_out.dim
        self.reference = product

    def forward(self, node_features, harmonics, weights, edge_index):
        if node_features.is_cuda and node_features.dtype == torch.float32:
            return _ScalarMessage.apply(
                node_features, harmonics, weights, edge_index, self.paths, self.output_width,
            )
        source, target = edge_index
        edges = self.reference(node_features[source], harmonics, weights)
        return edges.new_zeros((node_features.shape[0], edges.shape[-1])).index_add(0, target, edges)


def configure_candidate(model, backend):
    if backend != "hp-scalar":
        raise ValueError(f"Unknown candidate {backend}")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("This MACE path supports frozen E/F inference")
    first = model.interactions[0]
    if hasattr(first, "conv_fusion"):
        raise ValueError("The first MACE convolution must use the native checkpoint weights")
    first.conv_tp = ScalarMessage(first.conv_tp)
    first.conv_fusion = True
    return model
