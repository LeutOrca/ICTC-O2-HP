"""Counterbalance fresh model arms inside one process per measurement stage."""
import argparse
import gc
from pathlib import Path

import torch

from mace_variants import VARIANTS
import measure_mace_local


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("time", "memory", "trace"))
    parser.add_argument("variant", choices=VARIANTS[2:])
    parser.add_argument("case")
    parser.add_argument("qualification", type=Path)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    for repeat in range(5 if args.stage != "trace" else 1):
        roles = ("native", "base", args.variant)
        if repeat % 2:
            roles = roles[::-1]
        for boundary in ("fixed", "e2e"):
            for role in roles:
                if args.stage == "memory" and torch.cuda.memory_allocated() > 1024 ** 2:
                    raise RuntimeError("GPU tensors carried over from the preceding memory arm")
                measure_mace_local.main([
                    args.stage, role, args.case, boundary,
                    str(args.result_root / "results" / f"mace-{args.stage}-{args.case}-{boundary}-{role}-{repeat}.json"),
                    "--qualification", str(args.qualification), "--repeat", str(repeat),
                ])
                gc.collect()
                torch.cuda.synchronize()
                torch._C._cuda_clearCublasWorkspaces()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
