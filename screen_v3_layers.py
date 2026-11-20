"""Find the largest contiguous SO2 suffix that satisfies unchanged V3 E/F."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, os.environ["HPX_V3_ENTRY"])
import whole_network as native
from optimizations import install as install_previous
from v3_native import install


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    native.strict(torch)
    torch.set_num_threads(4)
    network = native.Network("native", "cuda")
    network.freeze_ema()
    network.model.requires_grad_(False)
    suite = "mptrj-b8"
    raw = network.batch(network.manifest["suites"][suite])
    fixed, topology = network.topology(raw)
    references = {}
    for seed in native.SEEDS:
        for boundary, data in (("fixed", fixed), ("e2e", raw)):
            torch.manual_seed(seed)
            references[(seed, boundary)] = network.forward(data, boundary == "fixed")
    install_previous("gather-radial", network.model)
    original = {name: module.forward for name, module in network.model.named_modules()
                if module.__class__.__name__ == "SO2Linear"}
    modules = dict(network.model.named_modules())
    trials = []
    for start in range(1, 7):
        for name, forward in original.items():
            modules[name].forward = forward
        installed = install(network.model, use_norm=False, start_block=start)
        records = []
        passed = True
        for seed in native.SEEDS:
            for boundary, data in (("fixed", fixed), ("e2e", raw)):
                torch.manual_seed(seed)
                errors = native.gate(torch, network.forward(data, boundary == "fixed"),
                                     references[(seed, boundary)], raw.natoms)
                records.append(dict(seed=seed, boundary=boundary, numerics=errors, matrix_calls={}))
                passed = all(row["passed"] for row in errors.values())
                if not passed:
                    break
            if not passed:
                break
        report = dict(suite=suite, base="9787a3a6ca7fe67eeda3f21ec7c24d6634cbac03",
                      source=Path("provenance/commit.txt").read_text().strip(),
                      checkpoint_sha256=native.CHECKPOINT_SHA, installed=installed,
                      records=records, passed=passed, start_block=start, kernel="triton",
                      local_only=True, registry=False, split_complex=False,
                      tf32=False, amp=False, cuda_graph=False)
        trials.append(report)
        args.output.write_text(json.dumps(trials, indent=2))
        print(json.dumps(report), flush=True)
        if passed:
            args.output.with_name(f"qualification-v3-block{start}.json").write_text(json.dumps(report, indent=2))
            break


if __name__ == "__main__":
    main()
