"""Prepare and load architecture-specific FP32 donated-left inference artifacts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil

import torch

from .aoti import _equation
from .donated import DonatedLeftInference


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def build_donated_launcher(build_directory):
    """Build the optional host extension with the target PyTorch/CUDA toolchain."""
    from torch.utils.cpp_extension import load
    src = Path(__file__).with_name("donated_launch.cpp")
    key = _sha(src.read_bytes() + torch.__version__.encode())[:16]
    path = Path(build_directory) / key
    path.mkdir(parents=True, exist_ok=True)
    ext = load(name="o2_native_" + key, sources=[str(src)], build_directory=str(path),
               extra_cflags=["-O3", "-std=c++17"], extra_ldflags=["-ldl"], with_cuda=True)
    return Path(ext.__file__)


def prepare_donated_inference(model, directory, launcher):
    """Create a new family bundle on the destination GPU, reading live weights."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with torch.cuda.device(model.weight.device), torch.no_grad():
        fn = DonatedLeftInference(model)
        example = torch.zeros((37, model.representation_dimension), device=model.weight.device)
        fn(example, torch.zeros_like(example))
        kernel = fn.last_kernel
        ptx = kernel.asm["ptx"]
        signature = re.search(r"\.entry\s+(\w+)\s*\((.*?)\)", ptx, re.S)
        if (not signature or len(re.findall(r"\.param\s", signature[2])) != 5
                or getattr(kernel.metadata, "global_scratch_size", 0)
                or ptx.count(signature[1] + "_param_4") != 1):
            raise ValueError("Unsupported CUDA launch ABI or scratch use")
        if re.search(r"(?:mma|wgmma|tcgen|cvt)[^;\n]*tf32", ptx, re.I):
            raise ValueError("TF32 is disabled for this kernel")
        cubin = kernel.asm["cubin"]
        host = Path(launcher)
        shutil.copyfile(host, directory / host.name)
        (directory / "launch.cubin").write_bytes(cubin)
        metadata = dict(name=signature[1], shared=kernel.metadata.shared, warps=kernel.metadata.num_warps,
                        block=128, device=list(torch.cuda.get_device_capability()), equation=_equation(model),
                        torch=torch.__version__, cubin_sha=_sha(cubin), ptx_sha=_sha(ptx.encode()),
                        host_name=host.name, host_sha=_sha(host.read_bytes()))
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


class PreparedDonatedLeftInference:
    """Call a prepared FP32 TP and return the overwritten, exclusively owned left input.

    The model's weights remain live. The prepared kernel requires 16-byte aligned
    contiguous tensors. The caller synchronizes producer streams before donation.
    """
    def __init__(self, model, directory):
        directory = Path(directory)
        meta = json.loads((directory / "metadata.json").read_text())
        device = model.weight.device
        if (device.type != "cuda" or meta["torch"] != torch.__version__
                or meta["device"] != list(torch.cuda.get_device_capability(device))
                or meta["equation"] != _equation(model)):
            raise ValueError("Artifact GPU, Torch version or O(2) equation differs")
        cubin = (directory / "launch.cubin").read_bytes()
        host = directory / meta["host_name"]
        if _sha(cubin) != meta["cubin_sha"] or _sha(host.read_bytes()) != meta["host_sha"]:
            raise ValueError("CUDA or host binary hash differs")
        spec = importlib.util.spec_from_file_location(host.stem, host)
        ext = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ext)
        self.model = model
        self.ptx_sha = meta["ptx_sha"]
        self.impl = ext.DonatedKernel(cubin, meta["name"], model.representation_dimension, model.channels,
                                      model.weight.numel(), meta["block"], meta["warps"], meta["shared"], device.index)

    def __call__(self, left, right):
        return self.impl(left, right, self.model.weight)
