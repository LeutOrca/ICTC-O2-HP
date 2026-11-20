"""Strict E/F qualification of own kernels inside frozen official EquiformerV3."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

ENTRY = Path(os.environ["HPX_V3_ENTRY"])
sys.path.insert(0, str(ENTRY))
import whole_network as native
from matrix_kernels import OwnMatrixKernels
from v3_native import install


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="mptrj-b1")
    parser.add_argument("--kernel", choices=("triton", "cuda"), default="triton")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native.strict(torch)
    torch.set_num_threads(4)
    model = native.Network("native", "cuda")
    model.freeze_ema()
    model.model.requires_grad_(False)
    raw = model.batch(args.suite)
    fixed, topology = model.topology(raw)
    references = {}
    for seed in native.SEEDS:
        for boundary, data in (("fixed", fixed), ("e2e", raw)):
            torch.manual_seed(seed)
            references[(seed, boundary)] = model.forward(data, boundary == "fixed")
    from optimizations import install as install_previous
    install_previous("gather-radial", model.model)
    installed = install(model.model, args.kernel)
    records = []
    for seed in native.SEEDS:
        for boundary, data in (("fixed", fixed), ("e2e", raw)):
            torch.manual_seed(seed)
            mode = OwnMatrixKernels(args.kernel)
            with mode:
                actual = model.forward(data, boundary == "fixed")
            errors = native.gate(torch, actual, references[(seed, boundary)], raw.natoms)
            record = dict(seed=seed, boundary=boundary, numerics=errors, matrix_calls=dict(mode.calls))
            records.append(record)
            passed = all(item["passed"] for item in errors.values())
            report = dict(suite=args.suite, base="9787a3a6ca7fe67eeda3f21ec7c24d6634cbac03",
                          checkpoint_sha256=native.CHECKPOINT_SHA, installed=installed,
                          records=records, passed=passed, tf32=False, amp=False, cuda_graph=False)
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(record), flush=True)
            if not passed:
                raise RuntimeError("Own V3 kernels failed the unchanged whole-network FP32 gate")
    print("V3_NATIVE_KERNEL_EF_PASS", flush=True)


if __name__ == "__main__":
    main()
