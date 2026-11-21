"""Build a read-only dependency view for the isolated Torch 2.11 runtime.

No base environment is modified. Incompatible optional vision/audio/PyG binary
extensions are absent from this inference environment, rather than imported
from the Torch 2.6 environment by accident.
"""
import json
from pathlib import Path
import sys


def main():
    source, target, manifest = map(Path, sys.argv[1:])
    target.mkdir(exist_ok=False)
    excluded, included = [], []
    prefixes = ("torch-", "torchvision", "torchaudio", "torchtext", "torchdata",
                "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv",
                "pyg_lib", "triton", "nvidia", "cuequivariance")
    for item in sorted(source.iterdir()):
        if (item.name in ("torch", "functorch", "torchgen", "__pycache__")
                or item.name.startswith(prefixes)):
            excluded.append(item.name)
            continue
        (target / item.name).symlink_to(item, target_is_directory=item.is_dir())
        included.append(dict(name=item.name, source=str(item)))
    manifest.write_text(json.dumps(dict(source=str(source), target=str(target),
                                       excluded=excluded, included=included), indent=2))


if __name__ == "__main__":
    main()
