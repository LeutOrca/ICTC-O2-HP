"""FP32-only diagnostic of FMA chains and bit-ordered reduction trees.

Uses real native FC operands and output/input-VJP samples, never force values
to choose a schedule. Native pointwise joins here are diagnostic only.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.environ["HPX_V3_ENTRY"])
import whole_network as native
from calibrate_fc import flatten


@triton.jit
def chains(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
           A0: tl.constexpr, A1: tl.constexpr, B0: tl.constexpr, B1: tl.constexpr,
           GROUPS: tl.constexpr, STRIPE: tl.constexpr, CONTIGUOUS: tl.constexpr,
           TILE: tl.constexpr):
    item = tl.program_id(0) * TILE + tl.arange(0, TILE)
    group = tl.program_id(1)
    row, col = item // N, item % N
    total = tl.full((TILE,), 0., tl.float32)
    for inner in range(K // GROUPS):
        if CONTIGUOUS:
            k = group * (K // GROUPS) + inner
        else:
            k = (inner // STRIPE) * GROUPS * STRIPE + group * STRIPE + inner % STRIPE
        a = tl.load(A + row * A0 + k * A1, row < M, other=0.)
        b = tl.load(B + k * B0 + col * B1, row < M, other=0.)
        total = tl.fma(a, b, total)
    tl.store(C + group * M * N + item, total, row < M)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    native.strict(torch)
    native.checked_checkpoint()
    torch.set_num_threads(4)
    net = native.Network("native", "cuda")
    net.freeze_ema()
    net.model.requires_grad_(False)
    raw = net.batch(net.manifest["suites"]["mptrj-b8"])
    fixed, topology = net.topology(raw)
    selected = {name: module for name, module in net.model.named_modules()
                if isinstance(module, torch.nn.Linear) and name.startswith("blocks.0.ga.so2_linear_")}
    captures, handles = {}, []
    def hook(name):
        def forward(module, inputs, output):
            row = dict(input=inputs[0].detach().clone(), output=output.detach().clone())
            captures[name] = row
            inputs[0].register_hook(lambda value: row.update(gradient=value.detach().clone()))
            output.register_hook(lambda value: row.update(cotangent=value.detach().clone()))
        return forward
    for name, module in selected.items():
        handles.append(module.register_forward_hook(hook(name)))
    torch.manual_seed(native.SEEDS[0])
    net.forward(fixed, True)
    for handle in handles:
        handle.remove()
    report = dict(source=Path("provenance/commit.txt").read_text().strip(),
                  checkpoint_sha256=native.CHECKPOINT_SHA, topology=topology,
                  metrics_dtype="torch.float32", records={})
    for name, module in selected.items():
        first = "so2_linear_1" in name
        direction = "forward" if first else "backward"
        if first and name.endswith("fc_m0"):
            continue
        row = captures[name]
        left = flatten(row["input"] if first else row["cotangent"])[:32]
        right = (module.weight.t() if first else module.weight)[:, :32]
        reference_width = module.out_features if first else module.in_features
        expected = (row["output"] if first else row["gradient"]).reshape(-1, reference_width)[:32, :32]
        m, k, n = left.shape[0], left.shape[1], right.shape[1]
        results = []
        for count in (2, 4, 8, 16, 32, 64, 128):
            bits = count.bit_length() - 1
            orders = list(itertools.permutations(range(bits))) if bits <= 5 else [
                tuple(range(bits)), tuple(reversed(range(bits))),
                *[tuple(range(start, bits)) + tuple(range(start)) for start in range(1, bits)]]
            for sign, stripe in [(1, s) for s in (1, 2, 4, 8, 16)] + [(-1, 1)]:
                if k % (count * stripe):
                    continue
                partial = left.new_empty((count, m, n))
                chains[(triton.cdiv(m * n, 128), count)](
                    left, right, partial, m, n, k, *left.stride(), *right.stride(),
                    count, stripe, sign < 0, 128, enable_fp_fusion=False)
                for order in orders:
                    values = partial
                    remaining = list(range(bits))
                    for bit in order:
                        axis = remaining.index(bit)
                        size = 1 << axis
                        view = values.reshape(-1, 2, size, m, n)
                        values = (view[:, 0] + view[:, 1]).reshape(-1, m, n)
                        remaining.remove(bit)
                    actual = values[0]
                    delta = actual - expected
                    results.append(dict(groups=sign * count, stripe=stripe, order=order,
                        relative_l2=float(delta.norm() / expected.norm()),
                        max_abs=float(delta.abs().max()), exact=int((delta == 0).sum()), elements=m * n))
        results.sort(key=lambda value: value["relative_l2"])
        report["records"][name] = dict(direction=direction, input_shape=list(left.shape),
                                       best=results[:12], tested=len(results))
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(layer=name, **report["records"][name])), flush=True)


if __name__ == "__main__":
    main()
