"""Build and call the shared-memory IEEE FP32 CUDA kernel."""
from pathlib import Path

from torch.utils.cpp_extension import load

_EXTENSION = None


def prepare():
    global _EXTENSION
    if _EXTENSION is None:
        _EXTENSION = load(
            name="hpx_ieee_matrix_r1",
            sources=[str(Path(__file__).parent / "csrc/matrix.cu")],
            extra_cuda_cflags=["-O3", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"],
            extra_cflags=["-O3"],
            verbose=True,
        )
    return _EXTENSION


def compiled_matrix(left, right, bias, alpha, beta):
    return prepare().matrix(left, right, bias, alpha, beta)
