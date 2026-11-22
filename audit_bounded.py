"""Independent CPU receipt audit for all affected bounded-build gates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "o2-hp-angular-operators-20261121-r1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cohort")
    parser.add_argument("--measurement", action="store_true")
    args = parser.parse_args()
    folder = ROOT / "measurements" / args.cohort
    baseline = list(csv.DictReader((BASE / "summary.csv").open()))
    targets = [("uvu", n, "public-repair") for n in range(6, 9)]
    targets += [("radial", n, role) for n in range(4, 9)
                for role in ("public-repair", "public-unfused-repair")]
    if args.measurement:
        targets += [("uvu", n, "eqx-o2") for n in range(6, 9)]
        targets += [("radial", n, "eqx-o2-repair") for n in range(4, 9)]
    accepted = []
    for family, order, role in targets:
        stage = "measure" if args.measurement else "qualify"
        matches = list((folder / f"{stage}-bounded-{family}").glob(f"qualify-L{order}-full-{role}-forward_vjp-r0.json"))
        assert len(matches) == 1, (family, order, role, matches)
        path = matches[0]
        receipt = json.loads(path.read_text())
        assert receipt["family"] == family
        assert role in receipt["accepted_by_metric"]["forward_vjp"], str(path)
        ref = next(r for r in baseline if r["family"] == family and int(r["order"]) == order and r["parity"] == "full")
        for key in ("input_sha256", "canonical_weights_sha256"):
            assert (receipt.get(key) or "") == ref[key], (path, key)
        assert receipt["dtype"] == "float32" and "H100" in receipt["gpu"]
        assert receipt["torch"] == "2.11.0+cu128"
        assert not any(receipt[k] for k in ("tf32", "amp", "cuda_graph"))
        expected = {f"{role}/{s}/{v}" for s in (20261003, 20261004) for v in (.1, 1.)}
        assert set(receipt["numerics"]) == expected
        checks = []
        for record in receipt["numerics"].values():
            assert record["evaluated_metric"] == "forward_vjp"
            assert record["check_window_size"] == 512 and record["full_workload"]
            assert [(r["start"], r["end"]) for r in record["chunks"]] == [(n, n+512) for n in range(0, 16384, 512)]
            for region in record["chunks"]:
                assert len(region["checks"]) == 4
                checks.extend(region["checks"])
        assert len(checks) == 512
        assert all(c["passed"] and c["arithmetic"] == "float32" for c in checks)
        accepted.append(dict(family=family, order=order, role=role,
            checks=len(checks), max_abs=max(c["max_abs"] for c in checks),
            max_relative_l2=max(c["relative_l2"] for c in checks),
            receipt=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            job=receipt["job"], gpu_uuid=receipt["gpu_uuid"]))
    data = dict(passed=True, cohort=args.cohort, gates=len(accepted),
                checks=sum(r["checks"] for r in accepted), accepted=accepted,
                original_input_weight_hashes=True, original_fp32_windows=True)
    output = ROOT / "provenance" / ("BOUNDED-PAIRED-GATE-AUDIT.json" if args.measurement else "BOUNDED-GATE-AUDIT.json")
    output.write_text(json.dumps(data, indent=2))
    print(json.dumps({k: v for k, v in data.items() if k != "accepted"}), flush=True)


if __name__ == "__main__":
    main()
