"""Screen only whole-network E/F, without any performance measurements."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, os.environ["HPX_V3_ENTRY"])
import whole_network as native
from optimizations import install as install_previous
from so2_ordered import install, PROFILES
from fp32_gate import gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--profiles", action="store_true")
    parser.add_argument("--profile-prefix", default="")
    args = parser.parse_args()
    native.strict(torch)
    torch.set_num_threads(4)
    network = native.Network("native", "cuda")
    network.freeze_ema()
    network.model.requires_grad_(False)
    raw = network.batch(network.manifest["suites"]["mptrj-b8"])
    fixed, topology = network.topology(raw)
    def run(seed):
        torch.manual_seed(seed)
        return network.forward(fixed, True)
    for _ in range(3):
        run(native.SEEDS[0])
    references = {seed: run(seed) for seed in native.SEEDS}
    install_previous("gather-radial", network.model)
    records = []
    configurations = (("triton", "triton", "triton"),
                      ("2x1", "2x1", "triton"),
                      ("2x1", "2x1", "4x8"),
                      ("triton", "triton", "4x8"),
                      ("4x1", "4x1", "4x8"),
                      ("4x8", "4x8", "4x8"),
                      ("2x1", "4x8", "4x8"),
                      ("4x8", "2x1", "4x8"))
    if args.profiles:
        configurations = [(profile,) * 3 for profile in PROFILES if profile.startswith(args.profile_prefix)]
    for forward, backward, zero in configurations:
        installed = install(network.model, forward, backward, zero)
        for seed in native.SEEDS:
            errors = gate(torch, run(seed), references[seed], raw.natoms)
            record = dict(forward=forward, backward=backward, zero_backward=zero,
                          seed=seed, numerics=errors, installed=installed,
                          passed=all(value["passed"] for value in errors.values()))
            records.append(record)
            args.output.write_text(json.dumps(dict(source=Path("provenance/commit.txt").read_text().strip(),
                                                   checkpoint_sha256=native.CHECKPOINT_SHA, topology=topology,
                                                   records=records, metrics_dtype="torch.float32"), indent=2))
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
