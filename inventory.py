"""Inspect package versions and frozen input metadata on a CPU compute node."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys


def main():
    rows = {}
    for package in ("torch", "triton", "e3nn", "mace-torch", "cuequivariance",
                    "cuequivariance-torch", "cuequivariance-ops-torch-cu12",
                    "openequivariance", "matplotlib", "numpy", "matscipy"):
        try:
            distribution = metadata.distribution(package)
            rows[package] = dict(version=distribution.version, path=str(distribution.locate_file("")))
        except metadata.PackageNotFoundError:
            rows[package] = None
    output = Path(sys.argv[1])
    output.write_text(json.dumps(dict(python=sys.executable, job=os.environ.get("SLURM_JOB_ID"),
                                      packages=rows), indent=2))
    print(output.read_text(), flush=True)


if __name__ == "__main__":
    main()
