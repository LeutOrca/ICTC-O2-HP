"""Check whether ordered PyTorch reductions resolve the native FP32 repeat gate."""
import argparse
import gc
from pathlib import Path
import traceback

import torch

import paired_runner as bench
from local_o2_compat import configure_local_o2

BASE_MODEL = bench.model_for
BASE_CONFIGURE = bench.configure


def configure():
    BASE_CONFIGURE()
    torch.use_deterministic_algorithms(True)


def model_for(backend, device):
    if backend == "hp-local-o2":
        return configure_local_o2(BASE_MODEL("native", device))
    return BASE_MODEL(backend, device)


bench.configure = configure
bench.model_for = model_for
for case in ("b8", "n512", "n4096"):
    for backend in ("native", "hp-scalar", "hp-local-o2", "hp-uvu"):
        args = argparse.Namespace(device="cuda", case=case, backend=backend,
                                  stage="qualify", boundary="fixed", round=0)
        print(f"deterministic {backend} {case}", flush=True)
        try:
            bench.run(args)
        except Exception as exc:
            traceback.print_exc()
            with (bench.ROOT / "errors.tsv").open("a") as log:
                log.write(f"{backend}\t{case}\t{type(exc).__name__}\n")
        finally:
            gc.collect()
            torch.cuda.empty_cache()
