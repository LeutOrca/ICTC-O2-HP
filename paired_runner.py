"""MACE whole-network E/F qualification and measurement on frozen periodic inputs."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(os.environ["MACE_RESULT_ROOT"])
SOURCE = Path(os.environ["MACE_SOURCE"])
sys.path[:0] = [str(SOURCE / item) for item in ("source/python", "source/hp3", "source/tace", "code")]

import numpy as np
import torch
from mace.data.neighborhood import get_neighborhood
from omat_adapter import load_native

CHECKPOINT = SOURCE / "models/pretrained/native.model"
INPUTS = SOURCE / "inputs/pretrained"
CASES = {
    "b1": ("validation-013",),
    "b8": tuple(f"validation-{index:03d}" for index in range(8)),
    "n512": ("scan-N512",),
    "n4096": ("scan-N4096",),
}
FP32_ATOL = 2e-5
FP32_RTOL = 2e-5
RELATIVE_L2 = 1e-5


def configure():
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261119)


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def model_for(backend, device):
    model = load_native(CHECKPOINT).float().to(device).requires_grad_(False).eval()
    if backend.startswith("eqx"):
        from eqx.models.mace import convert_mace_to_eqx

        model = convert_mace_to_eqx(
            model, implementation="o2", inplace=True,
            backend="torch" if backend == "eqx-torch" else "cuda",
            stream_radial=backend != "eqx-eager-radial",
        )
    if backend == "hp-uvu":
        from direct_uvu import configure_direct

        return configure_direct(model)
    if backend == "hp-local-o2":
        from local_o2 import configure_local_o2

        return configure_local_o2(model)
    if backend.startswith("hp"):
        from candidate import configure_candidate

        model = configure_candidate(model, backend)
    return model


def load_geometry(names):
    geometry = []
    for name in names:
        source = torch.load(INPUTS / f"{name}.pt", map_location="cpu", weights_only=False)
        pointers = source["ptr"].tolist()
        for index, (lo, hi) in enumerate(zip(pointers, pointers[1:])):
            geometry.append((
                source["positions"][lo:hi].numpy().astype(np.float64),
                source["atomic_numbers"][lo:hi].numpy(),
                source["cell"].reshape(-1, 3, 3)[index].numpy().astype(np.float64),
            ))
    return geometry


def native_graph(geometry, model, device):
    edge_indexes, unit_shifts, shifts, cells, positions, species, batches = (
        [] for _ in range(7)
    )
    pointers = [0]
    for batch, (position, atoms, cell) in enumerate(geometry):
        edge, shift, unit, cell = get_neighborhood(
            position, cutoff=model.r_max_cpu, pbc=(True, True, True), cell=cell.copy(),
        )
        edge_indexes.append(edge + pointers[-1])
        unit_shifts.append(unit)
        shifts.append(shift)
        cells.append(cell)
        positions.append(position)
        species.append(atoms)
        batches.append(np.full(len(atoms), batch, dtype=np.int64))
        pointers.append(pointers[-1] + len(atoms))

    # The model's element table is constant and is supplied from the frozen checkpoint.
    elements = model._benchmark_elements
    indices = np.searchsorted(elements, np.concatenate(species))
    attrs = np.eye(len(elements), dtype=np.float32)[indices]
    return {
        "edge_index": torch.from_numpy(np.concatenate(edge_indexes, axis=1).astype(np.int64, copy=False)).to(device),
        "shifts": torch.from_numpy(np.concatenate(shifts).astype(np.float32)).to(device),
        "unit_shifts": torch.from_numpy(np.concatenate(unit_shifts).astype(np.float32)).to(device),
        "positions": torch.from_numpy(np.concatenate(positions).astype(np.float32)).to(device),
        "cell": torch.from_numpy(np.concatenate(cells, axis=0).astype(np.float32)).to(device),
        "node_attrs": torch.from_numpy(attrs).to(device),
        "batch": torch.from_numpy(np.concatenate(batches)).to(device),
        "ptr": torch.tensor(pointers, device=device),
        "head": torch.zeros(len(geometry), dtype=torch.long, device=device),
    }


def evaluate(model, network_input):
    with torch.enable_grad(), torch.autocast(network_input["positions"].device.type, enabled=False):
        out = model(dict(network_input), training=False, compute_force=True)
    return {key: out[key].detach() for key in ("energy", "interaction_energy", "forces")}


def errors(actual, expected):
    output = {}
    for name, value in actual.items():
        reference = expected[name]
        ok_dtype = value.dtype == reference.dtype == torch.float32
        value, reference = value.detach().double().cpu(), reference.detach().double().cpu()
        delta = (value - reference).abs()
        finite = bool(torch.isfinite(value).all() and torch.isfinite(reference).all())
        rms = float(reference.square().mean().sqrt())
        relative = float(delta.norm() / reference.norm().clamp_min(1e-30))
        require_relative = name != "forces" or rms >= 1e-3
        passed = ok_dtype and finite and (
            not require_relative or relative <= RELATIVE_L2
        ) and bool((delta <= FP32_ATOL + FP32_RTOL * reference.abs()).all())
        output[name] = dict(passed=passed, max_abs=float(delta.max()),
                            relative_l2=relative, rms_reference=rms)
    return output


def run(args):
    configure()
    device = torch.device(args.device)
    model = model_for(args.backend, device)
    model._benchmark_elements = model.atomic_numbers.cpu().numpy()
    if not bool(np.all(model._benchmark_elements[:-1] < model._benchmark_elements[1:])):
        raise ValueError("Expected the frozen checkpoint's sorted element table")
    model.r_max_cpu = float(model.r_max)
    geometry = load_geometry(CASES[args.case])
    fixed = native_graph(geometry, model, device)
    receipt = dict(job=os.environ["SLURM_JOB_ID"], backend=args.backend, case=args.case,
                   boundary=args.boundary, stage=args.stage, torch=torch.__version__,
                   atoms=len(fixed["positions"]), edges=fixed["edge_index"].size(1),
                   structures=len(geometry), cutoff=model.r_max_cpu, tf32=False,
                   device=str(device), amp=False, parameters=sum(p.numel() for p in model.parameters()))
    if device.type == "cuda":
        receipt["gpu"] = torch.cuda.get_device_name(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def fixed_call():
        return evaluate(model, fixed)

    def e2e_call():
        with torch.profiler.record_function("mace_native_neighbors_and_input_transfer"):
            data = native_graph(geometry, model, device)
        return evaluate(model, data)

    call = fixed_call if args.boundary == "fixed" else e2e_call
    reference_file = ROOT / "receipts" / f"{args.device}-{args.case}-reference.pt"
    output_file = ROOT / "receipts" / (
        f"{args.device}-{args.backend}-{args.case}-{args.boundary}-{args.stage}-{args.round}.json"
    )
    if args.stage == "qualify":
        result = {label: fn() for label, fn in (("fixed", fixed_call), ("e2e", e2e_call))}
        repeat = {label: fn() for label, fn in (("fixed", fixed_call), ("e2e", e2e_call))}
        receipt["self_repeat"] = {label: errors(repeat[label], value) for label, value in result.items()}
        if args.backend == "native":
            torch.save({label: {key: v.cpu() for key, v in value.items()}
                        for label, value in result.items()}, reference_file)
        reference = torch.load(reference_file, weights_only=True)
        receipt["numerics"] = {label: errors(value, reference[label]) for label, value in result.items()}
        receipt["passed"] = all(row["passed"] for group in ("self_repeat", "numerics")
                                for boundary in receipt[group].values() for row in boundary.values())
        write(output_file, receipt)
        print(json.dumps(receipt), flush=True)
        return

    for _ in range(args.warmup):
        call()
    sync()
    if args.stage == "trace":
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        # The trace is intentionally separate from the steady measurements.
        with torch.profiler.profile(activities=activities, profile_memory=True, record_shapes=True) as profile:
            with torch.profiler.record_function("mace_whole_network_ef"):
                call()
                sync()
        trace = ROOT / "traces" / f"{args.backend}-{args.case}-{args.boundary}.json"
        profile.export_chrome_trace(str(trace))
        receipt.update(trace=str(trace.relative_to(ROOT)), summary=[
            dict(name=row.key, calls=row.count, self_cuda_us=row.self_device_time_total,
                 self_cpu_us=row.self_cpu_time_total, self_cuda_bytes=row.self_device_memory_usage)
            for row in sorted(profile.key_averages(), key=lambda row: row.self_device_time_total,
                              reverse=True)[:40]
        ])
    elif args.stage == "memory":
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.repeats):
            call()
        sync()
        receipt.update(peak_allocated=torch.cuda.max_memory_allocated(device),
                       peak_reserved=torch.cuda.max_memory_reserved(device))
    else:
        samples = []
        for _ in range(args.repeats):
            sync()
            start = time.perf_counter_ns()
            call()
            sync()
            samples.append((time.perf_counter_ns() - start) / 1e6)
        median = statistics.median(samples)
        receipt.update(wall_ms=samples, median_ms=median,
                       atoms_s=1000 * receipt["atoms"] / median,
                       structures_s=1000 * receipt["structures"] / median)
    write(output_file, receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--stage", choices=("qualify", "trace", "time", "memory"), required=True)
    parser.add_argument("--boundary", choices=("fixed", "e2e"), default="fixed")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--round", type=int, default=0)
    run(parser.parse_args())
