"""Frozen native-layout operators and independent FP32 references."""
from contextlib import nullcontext
import hashlib
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "upstream/official/src"), str(ROOT / "upstream/eqx"),
                str(ROOT / "upstream/ictp"), str(ROOT / "upstream/local-so2"),
                str(ROOT / "upstream")]
from ictc_o2_hp import (TensorProduct, ReferenceTensorProduct, UvuTensorProduct,
                        ReferenceUvuTensorProduct, RadialUvuTensorProduct)
from ictc_o2_hp import o2_math

C = 64
H = 64


def digest(values):
    result = hashlib.sha256()
    for item in values:
        result.update(str(tuple(item.shape)).encode())
        for chunk in item.detach().split(512, 0):
            result.update(chunk.contiguous().cpu().numpy().tobytes())
    return result.hexdigest()


def entries(blocks, channels):
    result, offset = [], 0
    for block in blocks:
        for copy in range(block.parity_copies):
            result.append((block.key, copy, offset, block.irrep_dimension))
            offset += channels * block.irrep_dimension
    return result


def transform(value, table, channels, inverse=False):
    pieces = []
    for key, copy, offset, dim in table:
        part = value[:, offset:offset + channels * dim]
        if inverse:
            part = part.reshape(len(value), dim, channels).transpose(-1, -2)
            if key[0] == "freq" and copy:
                part = torch.stack((part[..., 1], -part[..., 0]), -1)
        else:
            part = part.reshape(len(value), channels, dim)
            if key[0] == "freq" and copy:
                part = torch.stack((-part[..., 1], part[..., 0]), -1)
            part = part.transpose(-1, -2)
        pieces.append(part.flatten(1))
    return torch.cat(pieces, -1).contiguous()


def label(key):
    return ("0e" if key[1] == 1 else "0o") if key[0] == "scalar" else f"{key[1]}m"


def eqx_module(tables, channels, paths, keys, mode, external=False):
    from equivariantx.o2 import TensorProduct as EqxTP
    from equivariantx.o2.clebsch_gordan import coupling_entries
    irreps = [[(label(entry[0]), channel) for entry in table]
              for table, channel in zip(tables, channels)]
    instructions = [(i, j, k, mode, True) for i, j, k in paths]
    model = EqxTP(*irreps, instructions, irrep_normalization="component",
                  path_normalization="none", internal_weights=not external,
                  shared_weights=not external, backend="cuda")
    signs = []
    for instruction, (key1, key2, key3) in zip(model.instructions, keys):
        a = model.irreps_in1[instruction.i_in1].ir
        b = model.irreps_in2[instruction.i_in2].ir
        out = model.irreps_out[instruction.i_out].ir
        native = o2_math.o2_cg_tensor(key1, key2, key3, dtype=torch.float32)
        native = native * (out.dim ** .5 / native.norm())
        other = torch.zeros_like(native)
        for i, j, k, value in coupling_entries(a, b, out):
            other[i, j, k] = value * instruction.path_weight
        sign = 1. if float((native * other).sum()) > 0 else -1.
        if not torch.allclose(native, other * sign, atol=2e-6, rtol=2e-6):
            raise ValueError((key1, key2, key3, "analytic FP32 basis mapping failed"))
        signs.extend([sign] * channels[2])
    return model.cuda().float().requires_grad_(False), torch.tensor(signs, device="cuda")


def tp_reference(native):
    reference = ReferenceTensorProduct(C, native.max_frequency, parity=native.parity).cuda().float()
    offset = 0
    with torch.no_grad():
        for weight in reference.weights:
            weight.copy_(native.weight[offset:offset + weight.numel()])
            offset += weight.numel()
    reference.requires_grad_(False)
    sizes = [b.multiplicity * b.irrep_dimension for b in native.blocks]
    def call(left, right):
        def unpack(value):
            return tuple(part.reshape(len(value), b.multiplicity, b.irrep_dimension)
                         for part, b in zip(value.split(sizes, -1), native.blocks))
        return torch.cat([part.flatten(1) for part in reference(unpack(left), unpack(right))], -1)
    return call


class Spec:
    def __init__(self, family, order, parity, rows, seed=20261003, scale=.1):
        self.family, self.order, self.parity, self.rows = family, order, parity, rows
        self.calls, self.encode, self.decode = {}, {}, {}
        self.parameters = {}
        self.metadata = dict(family=family, order=order, parity=parity, rows=rows,
                             channels=C, hidden=H, scale=scale, seed=seed)
        torch.manual_seed(17000 + 100 * order + C)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        def random(shape):
            return (torch.randn(shape, generator=generator, device="cuda") * scale).requires_grad_()
        identity = lambda x: x
        if family == "tp":
            native = TensorProduct(C, order, parity=parity).cuda().float().requires_grad_(False)
            self.metadata["canonical_weights_sha256"] = digest([native.weight])
            self.logical = [random((rows, native.representation_dimension)) for _ in range(2)]
            self.reference = tp_reference(native)
            table = entries(native.blocks, C)
            indices = {(key, copy): i for i, (key, copy, _, _) in enumerate(table)}
            paths, keys = [], []
            for k1, k2, k3 in native.paths:
                for copy in range(native.block_by_key[k3].parity_copies):
                    paths.append((indices[k1, copy if k1[0] == "freq" else 0],
                                  indices[k2, copy if k2[0] == "freq" else 0], indices[k3, copy]))
                    keys.append((k1, k2, k3))
            self.calls["public"] = native
            self.encode["public"] = [identity, identity]
            self.decode["public"] = identity
            eqx, signs = eqx_module([table] * 3, [C] * 3, paths, keys, "uuu")
            with torch.no_grad():
                eqx.weight.copy_(native.weight * signs)
            self.calls["eqx-o2"] = eqx
            conv = lambda x: transform(x, table, C)
            self.encode["eqx-o2"] = [conv, conv]
            self.decode["eqx-o2"] = lambda x: transform(x, table, C, True)
            try:
                from cue_adapter import CueO2
                cue = CueO2(native).cuda().float().requires_grad_(False)
                self.calls["cue-o2"] = cue
                self.encode["cue-o2"] = [conv, conv]
                self.decode["cue-o2"] = self.decode["eqx-o2"]
            except Exception as exc:
                self.metadata["cue_build_error"] = repr(exc)
            self.parameters = {"public": list(native.parameters()), "eqx-o2": list(eqx.parameters())}
            self.metadata.update(input_dimension=native.representation_dimension,
                                 output_dimension=native.representation_dimension,
                                 paths=len(native.paths), weights=native.weight.numel(), equation="O2-uuu")
        elif family in ("uvu", "radial"):
            native = UvuTensorProduct(C, 1, order, parity=parity,
                                     internal_weights=False, shared_weights=False).cuda().float()
            ref = ReferenceUvuTensorProduct(C, 1, order, parity=parity,
                                           internal_weights=False, shared_weights=False).cuda().float()
            left, right = random((rows, native.input1_dimension)), random((rows, native.input2_dimension))
            first = [(e.key, e.copy, e.offset, e.dim) for e in native.entries_in1]
            second = [(e.key, e.copy, e.offset, e.dim) for e in native.entries_in2]
            keys = [(first[i][0], second[j][0], first[k][0]) for i, j, k in native.paths]
            eqx, signs = eqx_module([first, second, first], [C, 1, C], native.paths, keys,
                                     "u1u", external=True)
            enc_left = lambda x: transform(x, first, C)
            enc_right = lambda x: transform(x, second, 1)
            self.decode["public"] = identity
            self.decode["eqx-o2"] = lambda x: transform(x, first, C, True)
            if family == "uvu":
                weights = random((rows, native.weight_numel))
                self.logical = [left, right, weights]
                self.reference = ref
                self.calls["public"] = native
                self.calls["eqx-o2"] = eqx
                self.encode["public"] = [identity] * 3
                self.encode["eqx-o2"] = [enc_left, enc_right, lambda x: x * signs]
            else:
                linear = torch.nn.Linear(H, native.weight_numel).cuda().float().requires_grad_(False)
                self.metadata["canonical_weights_sha256"] = digest(list(linear.parameters()))
                hidden = random((rows, H))
                self.logical = [left, right, hidden]
                self.reference = lambda x, y, h: ref(x, y, linear(h))
                fused = RadialUvuTensorProduct(native, linear).requires_grad_(False)
                self.calls["public"] = fused
                self.calls["public-unfused"] = lambda x, y, h: native(x, y, linear(h))
                self.encode["public-unfused"] = [identity] * 3
                self.decode["public-unfused"] = identity
                mapped = torch.nn.Linear(H, native.weight_numel).cuda().float().requires_grad_(False)
                with torch.no_grad():
                    mapped.weight.copy_(linear.weight * signs[:, None])
                    mapped.bias.copy_(linear.bias * signs)
                self.calls["eqx-o2"] = lambda x, y, h: eqx(x, y, mapped(h))
                self.encode["public"] = [identity] * 3
                self.encode["eqx-o2"] = [enc_left, enc_right, identity]
            self.metadata.update(input_dimension=native.input1_dimension,
                angular_dimension=native.input2_dimension, output_dimension=native.representation_dimension,
                paths=len(native.paths), external_weights=native.weight_numel, equation="O2-uvu-V1")
        elif family == "so2":
            from so2_ops import SO2Linear
            from frozen_so2 import FrozenWeightCache
            from so2_layer import frozen_layer
            from matrix_kernels import OwnMatrixKernels
            native = SO2Linear(C, C, order, order).cuda().float().requires_grad_(False)
            self.metadata["canonical_weights_sha256"] = digest(list(native.parameters()))
            self.logical = [random((rows, (order + 1) ** 2, C))]
            self.calls["so2-native"] = native
            for backend in ("cuda2", "triton"):
                caches = tuple(FrozenWeightCache() for _ in range(order))
                def own(x, backend=backend, caches=caches):
                    with OwnMatrixKernels(backend):
                        return frozen_layer(x, native, caches, backend)
                self.calls["so2-" + backend] = own
            for role in self.calls:
                self.encode[role], self.decode[role] = [identity], identity
            def reference(x):
                out = [torch.nn.functional.linear(x[:, :order + 1].flatten(1),
                       native.fc_m0.weight, native.fc_m0.bias).reshape(rows if len(x)==rows else len(x), order+1, C)]
                offset = order + 1
                for m, layer in enumerate(native.so2_m_linear, 1):
                    count = order + 1 - m
                    value = x[:, offset:offset + 2 * count].reshape(len(x), 2, -1)
                    a, b = layer.fc.weight.chunk(2, 0)
                    real = torch.nn.functional.linear(value[:, 0], a) - torch.nn.functional.linear(value[:, 1], b)
                    imag = torch.nn.functional.linear(value[:, 0], b) + torch.nn.functional.linear(value[:, 1], a)
                    out.extend((real.reshape(len(x), count, C), imag.reshape(len(x), count, C)))
                    offset += 2 * count
                return torch.cat(out, 1)
            self.reference = reference
            self.metadata.update(input_dimension=(order+1)**2*C, output_dimension=(order+1)**2*C,
                                 mmax=order, equation="SO2Linear")
        elif family == "ictp":
            from ictp.o3.tensor_product import WeightedTensorProduct, PlainTensorProduct
            from ictp.o3.cartesian_harmonics import CartesianHarmonics
            native = WeightedTensorProduct(order, order, order, C, 1, C,
                                           connection_mode="uvu").cuda().float().requires_grad_(False)
            self.metadata["canonical_weights_sha256"] = digest(list(native.parameters()))
            plain = PlainTensorProduct(order, order, order, C, C, C).cuda().float()
            harmonic = CartesianHarmonics(order).cuda().float()
            with torch.no_grad():
                hx, hy = harmonic(random((rows, 3))), harmonic(random((rows, 3)))
                features = random((rows, C))
                left = (hx[:, :, None] * features[:, None]).flatten(1)
            self.logical = [left.requires_grad_(), hy.detach().requires_grad_()]
            def reference(x, y):
                expanded = torch.cat([part.reshape(len(y), 3**l, 1).expand(-1, -1, C).flatten(1)
                    for l, part in enumerate(y.split([3**l for l in range(order+1)], -1))], -1)
                value = plain(x, expanded)
                offset, weight_start, out = 0, 0, []
                for l, count in enumerate(native.n_paths):
                    width = 3**l * C * count
                    block = value[:, offset:offset+width].reshape(len(x), 3**l, C, count)
                    factors = torch.stack([native.weight[p][:,0] * native.alpha[p]
                                           for p in range(weight_start, weight_start+count)], -1)
                    out.append((block * factors).flatten(1))
                    offset += width
                    weight_start += count
                return torch.cat(out, -1)
            self.calls["ictp-native"] = native
            self.encode["ictp-native"], self.decode["ictp-native"] = [identity]*2, identity
            self.reference = reference
            self.metadata.update(input_dimension=native.in1_dim, angular_dimension=native.in2_dim,
                output_dimension=sum(3**l*C*n for l,n in enumerate(native.n_paths)),
                paths=native.n_total_paths, equation="O3-ICTP-even-uvu-path-expanded")
        else:
            raise ValueError(family)
        self.metadata["input_sha256"] = digest(self.logical)

    def encoded(self, role, logical=None):
        return [fn(value.detach()).detach().requires_grad_(True)
                for fn, value in zip(self.encode[role], logical or self.logical)]

    def decode_grad(self, role, gradients):
        if role not in ("eqx-o2", "cue-o2"):
            return gradients
        # Transforms are orthogonal; their transposes map cotangents back.
        decoded = []
        for value, original, encode in zip(gradients, self.logical, self.encode[role]):
            source = torch.zeros_like(original, requires_grad=True)
            mapped = encode(source)
            decoded.append(torch.autograd.grad(mapped, source, value)[0])
        return decoded
