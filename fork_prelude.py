"""Preload CPU code only; each measurement child owns a new CUDA context."""
import torch
import models
import benchmark

if torch.cuda.is_initialized():
    raise RuntimeError("Forkserver preload unexpectedly initialized CUDA")
