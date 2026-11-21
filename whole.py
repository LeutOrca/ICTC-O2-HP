"""Isolated-process whole-MACE qualification and measurement, FP32 throughout."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
SOURCE = Path(os.getenv("O2_MACE_SOURCE", str(RUNS / "mace-ictc-o2-hp-gbu2-20261119-r1/frozen")))
os.environ["MACE_SOURCE"] = str(SOURCE)
os.environ["MACE_RESULT_ROOT"] = str(HERE)
# Append, never prepend the old environment. Torch/Triton remain those of the
# selected interpreter. This supplies missing frozen MACE dependencies on 2.11.
sys.path.append("/public/home/sps-xia/bjxiao/miniconda3/envs/o2/lib/python3.11/site-packages")
sys.path.insert(0, os.getenv("HPX_MACE_ENTRY", str(RUNS / "mace-ictc-o2-hp-gbu2-20261120-r1")))
import torch
import qualify_force as prior


def errors(actual, expected):
    result = {}
    for key, value in actual.items():
        reference = expected[key]
        dtype_ok = value.dtype == reference.dtype == torch.float32
        value, reference = value.detach().cpu(), reference.detach().cpu()
        delta = (value - reference).abs()
        relative = float(delta.norm() / reference.norm().clamp_min(1e-30))
        rms = float(reference.square().mean().sqrt())
        finite = bool(torch.isfinite(value).all() and torch.isfinite(reference).all())
        need_relative = key != "forces" or rms >= 1e-3
        result[key] = dict(passed=dtype_ok and finite and
                          (not need_relative or relative <= 1e-5) and
                          bool((delta <= 2e-5 + 2e-5 * reference.abs()).all()),
                          max_abs=float(delta.max()), relative_l2=relative,
                          rms_reference=rms, arithmetic="float32")
    return result


def setup(role):
    model = prior.setup("receiver-csr-partition-graph" if role == "local-hpx"
                        else "native-graph")
    evaluator = prior.graph_evaluate
    coverage = "native MACE"
    if role in ("official", "fork"):
        sys.path.insert(0, str(HERE / "upstream" / role / "src"))
        from public_bridge import install
        install(model)
        coverage = "first scalar conv_tp, 4 l paths; other layers native"
    elif role == "local-hpx":
        sys.path.insert(0, os.getenv("O2_HPX_SOURCE", str(RUNS / "hpx-custom-kernels-20261120-r1")))
        from mace_variants import apply, evaluator as select
        apply(model, "final-input")
        evaluator = select("final-input", prior)
        coverage = "unpublished c591b16 final-input HPX cumulative MACE"
    elif role == "eqx-current":
        sys.path.insert(0, str(HERE / "upstream/eqx"))
        from equivariantx.models.mace import convert_mace_to_eqx
        model = convert_mace_to_eqx(model, implementation="o2", inplace=True,
                                    backend="cuda", stream_radial=True)
        coverage = "current EquivariantX O2 MACE converter, streamed radial"
    elif role == "cue-current":
        from mace.cli.convert_e3nn_cueq import run
        elements, cutoff = model._benchmark_elements, model.r_max_cpu
        model = run(model, device="cuda", return_model=True)
        model = model.float().requires_grad_(False).eval()
        model._benchmark_elements, model.r_max_cpu = elements, cutoff
        model._scatter_tile_width, model._receiver_csr = None, False
        coverage = "official MACE cuEquivariance converter optimize_all"
    return model, evaluator, coverage


def cpu(value):
    return {key: item.detach().cpu().clone() for key, item in value.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("build", "ref", "qualify", "time", "memory", "trace"))
    p.add_argument("role")
    p.add_argument("case", choices=tuple(prior.bench.CASES))
    p.add_argument("boundary", choices=("fixed", "e2e"))
    p.add_argument("output", type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--qualification", type=Path)
    p.add_argument("--repeat", type=int, default=0)
    a = p.parse_args()
    record = dict(stage=a.stage, role=a.role, case=a.case, boundary=a.boundary,
                  repeat=a.repeat, status="starting", cuda_graph=True, dtype="float32",
                  tf32=False, amp=False, error_arithmetic="float32", job=os.getenv("SLURM_JOB_ID"))
    try:
        prior.bench.configure()
        torch.use_deterministic_algorithms(True)
        record.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                      gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
        if a.stage in ("time", "memory", "trace"):
            qualification = json.loads(a.qualification.read_text())
            if not (qualification["passed"] and qualification["case"] == a.case
                    and qualification["role"] == a.role and qualification["torch"] == torch.__version__):
                raise RuntimeError("Matching whole-network FP32/Graph qualification missing")
            record["qualification_sha256"] = hashlib.sha256(a.qualification.read_bytes()).hexdigest()
        model, evaluate, coverage = setup(a.role)
        record["coverage"] = coverage
        geometry = prior.bench.load_geometry(prior.bench.CASES[a.case])
        fixed = prior.inputs_for(geometry, model)
        record.update(atoms=len(fixed["positions"]), edges=fixed["edge_index"].shape[1],
                      structures=len(geometry), checkpoint_sha256=hashlib.sha256(
                          prior.bench.CHECKPOINT.read_bytes()).hexdigest())
        digest = hashlib.sha256()
        for name, value in sorted(fixed.items()):
            digest.update(name.encode())
            digest.update(value.detach().contiguous().cpu().numpy().tobytes())
        record["input_sha256"] = digest.hexdigest()
        changed = dict(fixed, positions=fixed["positions"].detach().clone())
        changed["positions"][0, 0] += 0.005
        if a.stage == "build":
            evaluate(model, fixed)
            torch.cuda.synchronize()
            record.update(status="built", passed=True)
        elif a.stage == "ref":
            expected = {label: cpu(prior.evaluate(model, data))
                        for label, data in (("fixed", fixed), ("perturbed", changed))}
            runner = prior.Replay(evaluate, model, fixed)
            comparisons = {label: errors(cpu(runner.call(data)), expected[label])
                           for label, data in (("fixed", fixed), ("perturbed", changed))}
            record["numerics"] = comparisons
            record["passed"] = all(m["passed"] for group in comparisons.values() for m in group.values())
            if record["passed"]:
                torch.save(dict(values=expected, input_sha256=record["input_sha256"],
                                torch=torch.__version__), a.output.with_suffix(".pt"))
            record["status"] = "passed" if record["passed"] else "numerical-failure"
        elif a.stage == "qualify":
            reference = torch.load(a.reference, map_location="cpu", weights_only=False)
            if reference["input_sha256"] != record["input_sha256"] or reference["torch"] != torch.__version__:
                raise ValueError("Frozen inputs or runtime differ from reference")
            eager = {label: cpu(evaluate(model, data))
                     for label, data in (("fixed", fixed), ("perturbed", changed))}
            runner = prior.Replay(evaluate, model, fixed)
            builder = None
            if a.role == "local-hpx":
                from mace_static_inputs import CompositionInputs
                builder = CompositionInputs(geometry, model, fixed, prior.bench.get_neighborhood)
            rebuilt = builder(geometry) if builder else prior.inputs_for(geometry, model)
            compare = {}
            for label, data in (("fixed", fixed), ("e2e", rebuilt), ("perturbed", changed)):
                actual, repeated = cpu(runner.call(data)), cpu(runner.call(data))
                ref_key = "fixed" if label == "e2e" else label
                compare[label] = dict(reference=errors(actual, reference["values"][ref_key]),
                                      repeat=errors(repeated, actual),
                                      graph_vs_eager=errors(actual, eager[ref_key]))
            record["numerics"] = compare
            record["passed"] = all(m["passed"] for group in compare.values()
                                   for kind in group.values() for m in kind.values())
            record["status"] = "passed" if record["passed"] else "numerical-failure"
        else:
            builder = None
            if a.role == "local-hpx":
                from mace_static_inputs import CompositionInputs
                builder = CompositionInputs(geometry, model, fixed, prior.bench.get_neighborhood)
            if a.stage == "memory":
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            tick = time.perf_counter()
            runner = prior.Replay(evaluate, model, fixed)
            torch.cuda.synchronize()
            record["capture_prepare_wall_ms"] = (time.perf_counter() - tick) * 1000
            def call():
                data = fixed if a.boundary == "fixed" else (
                    builder(geometry) if builder else prior.inputs_for(geometry, model))
                return runner.call(data)
            if a.stage == "time":
                for _ in range(5):
                    out = call()
                    del out
                walls = []
                for _ in range(15):
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    out = call()
                    torch.cuda.synchronize()
                    walls.append((time.perf_counter() - tick) * 1000)
                    del out
                record.update(wall_ms=median(walls), samples_ms=walls,
                              atoms_s=record["atoms"] * 1000 / median(walls),
                              structures_s=record["structures"] * 1000 / median(walls))
            elif a.stage == "memory":
                out = call()
                torch.cuda.synchronize()
                record.update(peak_allocated=torch.cuda.max_memory_allocated(),
                              peak_reserved=torch.cuda.max_memory_reserved(), includes_capture=True)
                del out
            else:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    out = call()
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(a.output.with_suffix(".trace.json")))
                record["cuda_kernels"] = sorted({e.name for e in prof.events()
                    if e.device_type == torch.autograd.DeviceType.CUDA})
            record.update(status="measured", passed=True)
    except Exception as exc:
        record.update(status="oom" if isinstance(exc, torch.OutOfMemoryError) else "error",
                      passed=False, exception_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("traceback", "numerics", "samples_ms")}), flush=True)
    if not record.get("passed"):
        sys.exit(2)


if __name__ == "__main__":
    main()
