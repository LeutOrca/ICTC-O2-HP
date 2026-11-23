"""Capture the exact native FP32 hidden-adjoint kernel and its loaded library."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

assert os.environ.get("SLURM_JOB_ID")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["O2_DIAGNOSTIC"] = "1"
import torch
import triton
from benchmark import configure
from models import Spec
from candidate.frozen_state import TPPlan
from candidate.live_uvu import weight_adjoint

configure()
spec = Spec("radial", 8, "full", 16384, 20261003, 1.)
tp, linear = spec.calls["public"].tp, spec.diagnostics["linear"]
linear.requires_grad_(True)
values = [*spec.logical, linear.weight, linear.bias]
generator = torch.Generator(device="cuda").manual_seed(20261121)
g = torch.randn(16384, tp.representation_dimension, device="cuda", generator=generator)[:512] * .1
generator.manual_seed(20261123)
directions = tuple(torch.randn(v.shape, device="cuda", generator=generator) * .1 for v in values)
x, y = (v[:512] for v in values[:2])
a, b, _, d_weight, _ = directions
plan = TPPlan.capture(tp)
destination = ROOT / "results" / os.environ["SLURM_JOB_ID"]
destination.mkdir(parents=True, exist_ok=True)
with torch.no_grad():
    gw = weight_adjoint(plan, x, y, g)
    p = weight_adjoint(plan, a[:512], y, g) + weight_adjoint(plan, x, b[:512], g)
    for _ in range(3):
        result = gw @ linear.weight
        result2 = p @ linear.weight + gw @ d_weight
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.profiler.record_function("native_hidden_first"):
            result = gw @ linear.weight
        with torch.profiler.record_function("native_hidden_second"):
            result2 = p @ linear.weight + gw @ d_weight
        torch.cuda.synchronize()
    trace = destination / "NATIVE-HIDDEN-TRACE.json"
    prof.export_chrome_trace(str(trace))
events = json.loads(trace.read_text())["traceEvents"]
names = sorted({v["name"] for v in events if v.get("cat") == "kernel" and v.get("ph") == "X"})
libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                    if "libcublas" in line and line.split()[-1].startswith("/")})
metadata = []
tool = Path(triton.__file__).parent / "backends/nvidia/bin/cuobjdump"
for index, path in enumerate(libraries):
    library = Path(path)
    with library.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    entry = dict(path=path, sha256=checksum)
    if tool.is_file():
        symbols = destination / f"NATIVE-SYMBOLS-{index}.txt"
        with symbols.open("w") as output:
            run = subprocess.run([str(tool), "--gpu-architecture", "sm_90", "--dump-elf-symbols", path], stdout=output, stderr=subprocess.PIPE, text=True, timeout=180)
        entry.update(symbol_file=str(symbols), symbol_returncode=run.returncode, symbol_stderr=run.stderr[-1500:])
    metadata.append(entry)
record = dict(job=os.environ["SLURM_JOB_ID"], kernel_names=names, libraries=metadata,
              input_sha256=spec.metadata["input_sha256"], weight_sha256=spec.metadata["canonical_weights_sha256"],
              matrix_shape=[512, 55552, 64], dtype="float32", tf32=False, cuda_graph=False,
              scope="native numerical-association diagnostic; no performance acceptance")
(destination / "NATIVE-HIDDEN.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2), flush=True)
