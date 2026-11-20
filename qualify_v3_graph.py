"""Bounded qualification of whole frozen-topology V3 E/F CUDA Graph capture."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, os.environ["HPX_V3_ENTRY"])
import whole_network as native
from graph_rotation import init_edge_rot_mat
from graph_topology import install as install_topology
from matrix_kernels import OwnMatrixKernels
from v3_native import install


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("native", "own"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    native.strict(torch)
    torch.set_num_threads(4)
    net = native.Network("native", "cuda")
    net.freeze_ema()
    net.model.requires_grad_(False)
    raw = net.batch(net.manifest["suites"]["mptrj-b1"])
    fixed, topology = net.topology(raw)
    reference = {}
    for seed in native.SEEDS:
        torch.manual_seed(seed)
        reference[seed] = net.forward(fixed, True)
    net.model._init_edge_rot_mat = lambda edge: init_edge_rot_mat(edge, use_rotation_mask=True)
    install_topology(net, fixed)
    if args.role == "own":
        from optimizations import install as install_previous
        install_previous("gather-radial", net.model)
        install(net.model)
    context = OwnMatrixKernels() if args.role == "own" else nullcontext()
    with context:
        for _ in range(3):
            warm = net.forward(fixed, True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                warm = net.forward(fixed, True)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = net.forward(fixed, True)
    errors = []
    for seed in native.SEEDS:
        torch.manual_seed(seed)
        graph.replay()
        torch.cuda.synchronize()
        numeric = native.gate(torch, output, reference[seed], raw.natoms)
        errors.append(dict(seed=seed, numerics=numeric))
    report = dict(role=args.role, topology_sha256=topology, errors=errors,
                  passed=all(value["passed"] for row in errors for value in row["numerics"].values()),
                  cuda_graph=True, tf32=False, amp=False,
                  checkpoint_sha256=native.CHECKPOINT_SHA,
                  source=Path("provenance/commit.txt").read_text().strip())
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("The captured whole-network E/F failed the unchanged gate")


if __name__ == "__main__":
    main()
