"""Compare the receiver kernel and its adjoint with flattened scatter on H100."""
import argparse

import torch

import qualify_force as qualification
from receiver_csr import reduce, schedule

bench = qualification.bench


def check(index, nodes, widths, label):
    permutation, offsets = schedule(index, nodes)
    stats = []
    for width in widths:
        src = torch.randn((width, len(index)), device="cuda").t().requires_grad_(True)
        signs = torch.where(torch.arange(len(index), device="cuda") % 3 == 0, 1e3, 1e-3)
        src = src * signs[:, None]
        expected = src.new_zeros((nodes, width)).scatter_add_(
            0, index[:, None].expand_as(src), src)
        actual = reduce(src, index, permutation, offsets)
        adjoint = torch.randn_like(expected)
        backward_reference, = torch.autograd.grad(expected, src, adjoint)
        backward_actual, = torch.autograd.grad(actual, src, adjoint)
        equal = torch.equal(actual, expected)
        adjoint_equal = torch.equal(backward_actual, backward_reference)
        stats.append(dict(width=width, equal=equal, adjoint_equal=adjoint_equal,
                          max_abs=float((actual - expected).abs().max())))
        if not (equal and adjoint_equal):
            raise RuntimeError(f"Receiver order differs at {label}, width {width}: {stats[-1]}")
    print(label, stats, flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    bench.configure()
    torch.use_deterministic_algorithms(True)
    stats = {}
    degrees = torch.tensor([0, 1, 7, 31, 32, 33, 63, 64, 65, 97, 128, 129], device="cuda")
    receivers = torch.repeat_interleave(torch.arange(len(degrees), device="cuda"), degrees)
    receivers = receivers[torch.randperm(len(receivers), device="cuda")]
    stats["synthetic"] = check(receivers, len(degrees), (1, 3, 127, 128, 2048), "synthetic")
    for case in qualification.CASES:
        model = qualification.setup("native-graph")
        data = qualification.inputs_for(bench.load_geometry(bench.CASES[case]), model)
        widths = (1,) + tuple(layer.conv_tp.irreps_out.dim for layer in model.interactions)
        stats[case] = check(data["edge_index"][1], len(data["positions"]), widths, case)
    bench.write(bench.ROOT / "RECEIVER_CHECK.json", stats)


if __name__ == "__main__":
    main()
