"""Locate strict FP32 force drift in the frozen small and medium MACE networks."""
import argparse
import gc

import numpy as np
import torch

import paired_runner as bench
from candidate import configure_candidate
from capture_both_variants import Replay
from compact_species import PackedSpeciesProjection
from direct_uvu import DirectMessage
from native_graph_scatter import native_capture_compat


CASES = ("b8", "n512")
METRICS = ("energy", "interaction_energy", "forces")
MODES = {
    "native-graph": (False, False, ()),
    "scalar-graph": (True, False, ()),
    "uvu-only-graph": (False, True, ()),
    "uvu-graph": (True, True, ()),
    "packed-first-graph": (False, False, (0,)),
    "packed-second-graph": (False, False, (1,)),
    "packed-both-graph": (False, False, (0, 1)),
    "frozen-1b65a8c-graph": (True, True, (0, 1)),
    "receiver-csr-graph": (False, False, ()),
    "native-graph-tiled": (False, False, ()),
    "receiver-csr-parallel-graph": (False, False, ()),
}


def setup(mode):
    scalar, uvu, pack = MODES[mode]
    model = bench.model_for("native", "cuda")
    model._receiver_csr = mode in ("receiver-csr-graph", "receiver-csr-parallel-graph")
    model._parallel_neighbors = mode == "receiver-csr-parallel-graph"
    model._scatter_tile_width = 128 if mode == "native-graph-tiled" else None
    model._benchmark_elements = model.atomic_numbers.cpu().numpy()
    model.r_max_cpu = float(model.r_max)
    if scalar:
        configure_candidate(model, "hp-scalar")
    if uvu:
        block = model.interactions[1]
        block.conv_tp = DirectMessage(block.conv_tp)
        block.conv_fusion = True
    for layer in pack:
        block = model.interactions[layer]
        block.skip_tp = PackedSpeciesProjection(block.skip_tp)
    return model


def inputs_for(geometry, model, device="cuda"):
    if model._parallel_neighbors and len(geometry) > 1:
        from parallel_neighbors import neighborhoods
        inputs = bench.native_graph(geometry, model, device,
                                    neighbor_batches=neighborhoods(geometry, model.r_max_cpu))
    else:
        inputs = bench.native_graph(geometry, model, device)
    if any(isinstance(layer.skip_tp, PackedSpeciesProjection) for layer in model.interactions):
        species = np.searchsorted(model._benchmark_elements, np.concatenate([row[1] for row in geometry]))
        elements = np.unique(species)
        inputs["_selector_groups"] = tuple(
            (int(z), None if len(elements) == 1 else
             torch.from_numpy(np.flatnonzero(species == z)).to(device))
            for z in elements
        )
    return inputs


def evaluate(model, inputs):
    selectors = [layer.skip_tp for layer in model.interactions
                 if isinstance(layer.skip_tp, PackedSpeciesProjection)]
    data = dict(inputs)
    groups = data.pop("_selector_groups", None)
    for selector in selectors:
        selector.groups = groups
    try:
        return bench.evaluate(model, data)
    finally:
        for selector in selectors:
            selector.groups = None


def graph_evaluate(model, inputs):
    with native_capture_compat(model._scatter_tile_width):
        if model._receiver_csr:
            from receiver_csr import use_receiver_csr
            with use_receiver_csr(inputs):
                return evaluate(model, inputs)
        return evaluate(model, inputs)


def exact(actual, reference):
    return {key: torch.equal(actual[key], reference[key]) for key in METRICS}


def cpu(result):
    return {key: item.cpu() for key, item in result.items()}


def collect(point, mode, reference):
    print(f"start {point} {mode}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    model = setup(mode)
    geometry = bench.load_geometry(bench.CASES[point])
    fixed = inputs_for(geometry, model)
    if model._parallel_neighbors:
        changed = list(geometry)
        position, atoms, cell = changed[0]
        position = position.copy()
        position[0, 0] += 0.005
        changed[0] = (position, atoms, cell)
        for frame in (geometry, changed):
            expected = bench.native_graph(frame, model, "cpu")
            actual = inputs_for(frame, model, "cpu")
            if actual.keys() != expected.keys() or not all(
                    torch.equal(tensor, expected[key]) for key, tensor in actual.items()):
                raise RuntimeError("Parallel native neighbor assembly changed an input tensor")
    perturbed = fixed["positions"].clone()
    perturbed[0, 0] += 0.005
    changed = dict(fixed, positions=perturbed)
    if reference is None:
        reference = {label: cpu(evaluate(model, data)) for label, data in
                     (("fixed", fixed), ("e2e", inputs_for(geometry, model)), ("perturbed", changed))}
    runner = Replay(graph_evaluate, model, fixed)
    calls = {
        "fixed": lambda: runner.call(fixed),
        "e2e": lambda: runner.call(inputs_for(geometry, model)),
        "perturbed": lambda: runner.call(changed),
    }
    numerics = {}
    case_errors = {}
    for label, call in calls.items():
        value, repeat = cpu(call()), cpu(call())
        numerics[label] = {
            "reference": bench.errors(value, reference[label]),
            "reference_exact": exact(value, reference[label]),
            "repeat": bench.errors(repeat, value),
            "repeat_exact": exact(repeat, value),
        }
        starts = fixed["ptr"].tolist()
        case_errors[label] = [
            bench.errors({"forces": value["forces"][lo:hi]}, {"forces": reference[label]["forces"][lo:hi]})["forces"]
            for lo, hi in zip(starts[:-1], starts[1:])
        ]
    passed = all(item["passed"] for group in numerics.values()
                 for key in ("reference", "repeat") for item in group[key].values())
    if mode in ("native-graph", "native-graph-tiled"):
        passed &= all(all(group["reference_exact"].values()) for group in numerics.values())
    receipt = dict(case=point, backend=mode, atoms=fixed["positions"].shape[0],
                   edges=fixed["edge_index"].shape[1], numerics=numerics,
                   per_structure_force=case_errors, passed=passed,
                   qualification_peak_allocated=torch.cuda.max_memory_allocated(),
                   qualification_peak_reserved=torch.cuda.max_memory_reserved(),
                   deterministic=torch.are_deterministic_algorithms_enabled())
    bench.write(bench.ROOT / "receipts" / f"{point}-{mode}.json", receipt)
    print(point, mode, passed, {name: (data["reference"]["forces"]["relative_l2"],
                                     data["repeat"]["forces"]["relative_l2"])
                               for name, data in numerics.items()}, flush=True)
    if mode in ("native-graph", "native-graph-tiled") and not passed:
        raise RuntimeError("Capture-compatible native graph differs from original native eager")
    return reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument("--mode", choices=tuple(MODES)[1:], action="append")
    args = parser.parse_args()
    bench.configure()
    torch.use_deterministic_algorithms(True)
    for case in args.case or CASES:
        reference = None
        for mode in ("native-graph", *(args.mode or tuple(MODES)[1:])):
            reference = collect(case, mode, reference)
            gc.collect()
            torch.cuda.synchronize()
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
