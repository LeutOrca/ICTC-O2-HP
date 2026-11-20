"""Qualify 4100 atoms against the native flattened deterministic scatter reduction."""
import argparse
import traceback

import torch

from qualify_force import bench, collect, cpu, evaluate, exact, inputs_for, setup
from native_graph_scatter import native_capture_compat

CASE = "n4096"
REFERENCE = bench.ROOT / "receipts" / "n4096-tiled-eager.pt"


def reference():
    model = setup("native-graph-tiled")
    geometry = bench.load_geometry(bench.CASES[CASE])
    fixed = inputs_for(geometry, model)
    positions = fixed["positions"].clone()
    positions[0, 0] += 0.005
    changed = dict(fixed, positions=positions)
    with native_capture_compat(model._scatter_tile_width):
        outputs = {label: cpu(evaluate(model, data)) for label, data in
                   (("fixed", fixed), ("e2e", inputs_for(geometry, model)), ("perturbed", changed))}
        repeat = cpu(evaluate(model, fixed))
    bitwise = exact(repeat, outputs["fixed"])
    if not all(bitwise.values()) or not all(exact(outputs["fixed"], outputs["e2e"]).values()):
        raise RuntimeError("Tiled original eager reference is not repeatable")
    torch.save(outputs, REFERENCE)
    bench.write(bench.ROOT / "TILED_REFERENCE.json",
                dict(tile_width=model._scatter_tile_width, atoms=len(fixed["positions"]),
                     edges=fixed["edge_index"].size(1), bitwise_repeat=bitwise,
                     peak_allocated=torch.cuda.max_memory_allocated(),
                     peak_reserved=torch.cuda.max_memory_reserved()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "native-graph", "native-graph-tiled", "receiver-csr-graph"))
    args = parser.parse_args()
    bench.configure()
    torch.use_deterministic_algorithms(True)
    try:
        if args.mode == "reference":
            reference()
        else:
            collect(CASE, args.mode, torch.load(REFERENCE, weights_only=True))
    except torch.cuda.OutOfMemoryError as exc:
        bench.write(bench.ROOT / f"OOM-{args.mode}.json", {
            "mode": args.mode, "case": CASE, "oom": str(exc),
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            "traceback": traceback.format_exc(),
        })
        return 42
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
