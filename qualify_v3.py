"""Strict E/F qualification of own kernels inside frozen official EquiformerV3."""
import argparse
from contextlib import nullcontext
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
    parser.add_argument("--kernel", choices=("triton", "cuda", "warp4"), default="triton")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--without-norm", action="store_true")
    parser.add_argument("--split-complex", action="store_true")
    parser.add_argument("--registry", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--start-block", type=int, choices=range(7), default=0)
    parser.add_argument("--implementation", choices=("packed", "ordered", "calibrated"), default="packed")
    args = parser.parse_args()
    native.strict(torch)
    native.checked_checkpoint()
    torch.set_num_threads(4)
    model = native.Network("native", "cuda")
    model.freeze_ema()
    model.model.requires_grad_(False)
    raw = model.batch(model.manifest["suites"][args.suite])
    fixed, topology = model.topology(raw)
    for _ in range(3):
        torch.manual_seed(native.SEEDS[0])
        model.forward(fixed, True)
    references = {}
    for seed in native.SEEDS:
        for boundary, data in (("fixed", fixed), ("e2e", raw)):
            torch.manual_seed(seed)
            references[(seed, boundary)] = model.forward(data, boundary == "fixed")
    from optimizations import install as install_previous
    install_previous("gather-radial", model.model)
    if args.implementation in ("ordered", "calibrated"):
        if not args.without_norm or not args.local_only or args.start_block:
            raise ValueError("Ordered qualification requires all 14 SO2 and unchanged native norm")
        from so2_ordered import install as install_ordered
        configuration = "calibrated" if args.implementation == "calibrated" else "triton"
        installed = dict(so2=install_ordered(model.model, configuration, configuration, configuration), norm=[])
    else:
        installed = install(model.model, args.kernel, use_norm=not args.without_norm,
                            split_complex=args.split_complex, start_block=args.start_block)
    if args.registry:
        from kernel_registry import install as register
        registered_calls = register(args.kernel)
    records = []
    for seed in native.SEEDS:
        for boundary, data in (("fixed", fixed), ("e2e", raw)):
            torch.manual_seed(seed)
            mode = nullcontext() if args.registry or args.local_only else OwnMatrixKernels(args.kernel)
            with mode:
                actual = model.forward(data, boundary == "fixed")
            errors = native.gate(torch, actual, references[(seed, boundary)], raw.natoms)
            record = dict(seed=seed, boundary=boundary, numerics=errors,
                          matrix_calls=dict(registered_calls if args.registry else
                                            {} if args.local_only else mode.calls))
            records.append(record)
            passed = all(item["passed"] for item in errors.values())
            report = dict(suite=args.suite, base="9787a3a6ca7fe67eeda3f21ec7c24d6634cbac03",
                          checkpoint_sha256=native.CHECKPOINT_SHA, installed=installed,
                          records=records, passed=passed, split_complex=args.split_complex,
                          registry=args.registry,
                          local_only=args.local_only,
                          start_block=args.start_block, kernel=args.kernel,
                          implementation=args.implementation,
                          source=Path("provenance/commit.txt").read_text().strip(),
                          input_positions_sha256=native.tensor_sha(raw.pos), topology_sha256=topology,
                          input_manifest_sha256=native.sha(native.PROV / "geometry.json"),
                          gpu_uuid=str(torch.cuda.get_device_properties(0).uuid), torch=torch.__version__,
                          tf32=False, amp=False, cuda_graph=False)
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(record), flush=True)
            if not passed:
                raise RuntimeError("Own V3 kernels failed the unchanged whole-network FP32 gate")
    print("V3_NATIVE_KERNEL_EF_PASS", flush=True)


if __name__ == "__main__":
    main()
