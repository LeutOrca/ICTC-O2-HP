"""Recompute full-network same-card medians and decision gates from frozen receipts."""
import argparse
import json
from pathlib import Path
from statistics import median


def read(path):
    return json.loads(path.read_text())


def compare(folder, case, boundary, base, variant):
    matched = []
    job = folder.name.removeprefix("paired-")
    for repeat in range(1, 6):
        def arm(mode, stage):
            path = folder / "receipts" / f"cuda-{mode}-{case}-{boundary}-{stage}-{repeat}.json"
            row = read(path)
            if row["job"] != job or row["tf32"] or row["amp"]:
                raise ValueError(f"Invalid measurement provenance in {path}")
            if stage == "time" and (len(row["wall_ms"]) != 15
                                    or median(row["wall_ms"]) != row["median_ms"]):
                raise ValueError(f"Invalid timing series in {path}")
            return row

        time_base, memory_base = arm(base, "time"), arm(base, "memory")
        time_variant, memory_variant = arm(variant, "time"), arm(variant, "memory")
        shapes = [(row["atoms"], row["edges"], row["structures"], row["gpu"], row["torch"])
                  for row in (time_base, memory_base, time_variant, memory_variant)]
        if len(set(shapes)) != 1:
            raise ValueError(f"Unmatched geometry or GPU in {folder}, {case}, {boundary}, {repeat}")
        matched.append({
            "round": repeat,
            "baseline_ms": time_base["median_ms"], "candidate_ms": time_variant["median_ms"],
            "speedup": time_base["median_ms"] / time_variant["median_ms"],
            "baseline_atoms_s": time_base["atoms_s"], "candidate_atoms_s": time_variant["atoms_s"],
            "baseline_structures_s": time_base["structures_s"],
            "candidate_structures_s": time_variant["structures_s"],
            "baseline_allocated": memory_base["peak_allocated"],
            "candidate_allocated": memory_variant["peak_allocated"],
            "allocated_savings": 1 - memory_variant["peak_allocated"] / memory_base["peak_allocated"],
            "baseline_reserved": memory_base["peak_reserved"],
            "candidate_reserved": memory_variant["peak_reserved"],
            "reserved_savings": 1 - memory_variant["peak_reserved"] / memory_base["peak_reserved"],
        })
    stats = {name: median(row[name] for row in matched) for name in matched[0] if name != "round"}
    return dict(case=case, boundary=boundary, medians=stats, pairs=matched,
                speed_pass=stats["speedup"] >= 2,
                allocated_pass=stats["allocated_savings"] >= 0.10,
                reserved_pass=stats["reserved_savings"] >= 0.10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--baseline", default="native-graph")
    parser.add_argument("--variant", default="receiver-csr-graph")
    parser.add_argument("--case", action="append")
    parser.add_argument("--output", default="PAIR_AUDIT.json")
    args = parser.parse_args()
    folder = args.folder
    cases = args.case or ("b8", "n512")
    for case in cases:
        for backend in (args.baseline, args.variant):
            if not read(folder / "receipts" / f"{case}-{backend}.json")["passed"]:
                raise ValueError(f"Missing E/F qualification for {case} {backend}")
    rows = [compare(folder, case, boundary, args.baseline, args.variant)
            for case in cases for boundary in ("fixed", "e2e")]
    receipt = {"gpu": (folder / "gpu.csv").read_text().strip(), "baseline": args.baseline,
               "candidate": args.variant, "comparisons": rows}
    (folder / args.output).write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    for row in rows:
        data = row["medians"]
        values = [row["case"], row["boundary"],
                  f"{data['baseline_ms']:.3f} → {data['candidate_ms']:.3f}",
                  f"{data['speedup']:.3f}×",
                  f"{data['baseline_atoms_s']:.1f} → {data['candidate_atoms_s']:.1f}",
                  f"{data['baseline_structures_s']:.1f} → {data['candidate_structures_s']:.1f}",
                  f"{data['baseline_allocated'] / 2 ** 20:.2f} → {data['candidate_allocated'] / 2 ** 20:.2f}",
                  f"{data['baseline_reserved'] / 2 ** 20:.2f} → {data['candidate_reserved'] / 2 ** 20:.2f}",
                  f"{data['allocated_savings']:.1%} / {data['reserved_savings']:.1%}"]
        print("| " + " | ".join(values) + " |", flush=True)


if __name__ == "__main__":
    main()
