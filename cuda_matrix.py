"""Compile own CUDA source with NVRTC and launch through the CUDA driver."""
import ctypes
from pathlib import Path

import torch

_MODULES = {}


def _checked(status, action):
    if status:
        raise RuntimeError(f"{action} returned CUDA status {status}")


def prepare(groups=1):
    device = torch.cuda.current_device()
    key = (device, groups)
    if key in _MODULES:
        return _MODULES[key]
    torch.cuda.init()
    torch.cuda.set_device(device)
    context_anchor = torch.empty(1, device=torch.device("cuda", device))
    torch.cuda.synchronize()
    library_root = Path(torch.__file__).resolve().parent.parent / "nvidia/cuda_nvrtc/lib"
    libraries = sorted(library_root.glob("libnvrtc.so*"))
    if not libraries:
        raise RuntimeError(f"NVRTC is missing from {library_root}")
    compiler = ctypes.CDLL(str(libraries[0]))
    source = (Path(__file__).parent / "csrc/matrix.cu").read_bytes()
    program = ctypes.c_void_p()
    compiler.nvrtcCreateProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
    ]
    _checked(compiler.nvrtcCreateProgram(ctypes.byref(program), source, b"matrix.cu", 0, None, None),
             "nvrtcCreateProgram")
    major, minor = torch.cuda.get_device_capability()
    options = [f"--gpu-architecture=sm_{major}{minor}".encode(),
               b"--std=c++17", b"--ftz=false", b"--prec-div=true", b"--prec-sqrt=true",
               f"-DK_GROUPS={abs(groups)}".encode(), f"-DK_CONTIGUOUS={int(groups < 0)}".encode()]
    option_array = (ctypes.c_char_p * len(options))(*options)
    compiler.nvrtcCompileProgram.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p),
    ]
    status = compiler.nvrtcCompileProgram(program, len(options), option_array)
    size = ctypes.c_size_t()
    compiler.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    compiler.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    compiler.nvrtcGetProgramLogSize(program, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    compiler.nvrtcGetProgramLog(program, log)
    if status:
        raise RuntimeError(log.value.decode())
    compiler.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    compiler.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _checked(compiler.nvrtcGetPTXSize(program, ctypes.byref(size)), "nvrtcGetPTXSize")
    ptx = ctypes.create_string_buffer(size.value)
    _checked(compiler.nvrtcGetPTX(program, ptx), "nvrtcGetPTX")
    compiler.nvrtcGetCUBINSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    compiler.nvrtcGetCUBIN.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _checked(compiler.nvrtcGetCUBINSize(program, ctypes.byref(size)), "nvrtcGetCUBINSize")
    cubin = ctypes.create_string_buffer(size.value)
    _checked(compiler.nvrtcGetCUBIN(program, cubin), "nvrtcGetCUBIN")
    compiler.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _checked(compiler.nvrtcDestroyProgram(ctypes.byref(program)), "nvrtcDestroyProgram")
    if b".tf32" in ptx.value:
        raise RuntimeError("Unexpected TF32 in own CUDA kernel")
    driver = ctypes.CDLL("libcuda.so.1")
    _checked(driver.cuInit(0), "cuInit")
    module, kernel = ctypes.c_void_p(), ctypes.c_void_p()
    driver.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    driver.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p,
    ]
    _checked(driver.cuModuleLoadData(ctypes.byref(module), cubin), "cuModuleLoadData")
    _checked(driver.cuModuleGetFunction(ctypes.byref(kernel), module, b"ieee_matrix"),
             "cuModuleGetFunction")
    driver.cuLaunchKernel.argtypes = [
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    _MODULES[key] = (driver, module, kernel, cubin)
    return _MODULES[key]


def compiled_matrix(left, right, bias, alpha, beta, groups=1, output=None):
    driver, _, kernel, _ = prepare(groups)
    rows, inner, columns = left.shape[-2], left.shape[-1], right.shape[-1]
    batches = left.shape[0] if left.ndim == 3 else 1
    if output is None:
        output = left.new_empty((*left.shape[:-1], columns))
    if not output.numel():
        return output
    pointers = [ctypes.c_void_p(value.data_ptr()) if value is not None else ctypes.c_void_p()
                for value in (left, right, bias, output)]
    strides = [left.stride(0) if left.ndim == 3 else 0, left.stride(-2), left.stride(-1),
               right.stride(0) if right.ndim == 3 else 0, right.stride(-2), right.stride(-1),
               bias.stride(0) if bias is not None and bias.ndim == 3 else 0,
               bias.stride(-2) if bias is not None else 0,
               bias.stride(-1) if bias is not None else 0]
    arguments = pointers + [ctypes.c_int64(value) for value in (rows, columns, inner, *strides)]
    arguments += [ctypes.c_int64(value) for value in (
        output.stride(0) if output.ndim == 3 else 0, output.stride(-2), output.stride(-1)
    )]
    arguments += [ctypes.c_float(alpha), ctypes.c_float(beta)]
    addresses = (ctypes.c_void_p * len(arguments))(
        *[ctypes.cast(ctypes.byref(value), ctypes.c_void_p).value for value in arguments]
    )
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    _checked(driver.cuLaunchKernel(kernel, ((rows + 63) // 64) * batches,
                                  (columns + 63) // 64, 1, 16, 16, 1, 0,
                                  stream, addresses, None), "cuLaunchKernel")
    return output
