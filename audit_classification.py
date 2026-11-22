"""Reaggregate frozen receipts and prove that corrected labels change no measurements."""
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "o2-hp-angular-operators-20261121-r1"


def main():
    checks = {}
    for name in ("summary.csv", "speedups.csv", "numerics.csv"):
        current, old = (ROOT / name).read_bytes(), (BASE / name).read_bytes()
        assert current == old, name
        checks[name] = hashlib.sha256(current).hexdigest()
    with (ROOT / "role-coverage.csv").open() as stream:
        roles = list(csv.DictReader(stream))
    counts = dict(Counter(r["status"] for r in roles))
    assert counts == {"measured": 453, "timeout": 15, "numerical-failure": 8, "unsupported": 20}, counts
    result = dict(passed=True, baseline_commit="6f641cf736a97fa6835678c0eb123f5b2694e80d",
                  role_status_counts=counts, measurement_csv_byte_identical=checks,
                  corrected_cases=[r for r in roles if (r["family"], r["order"], r["role"], r["parity"], r["metric"]) in (
                      ("uvu", "6", "public", "full", "forward_vjp"),
                      ("radial", "5", "public", "full", "forward_vjp"),
                      ("radial", "6", "public", "full", "forward_vjp"))])
    (ROOT / "provenance/CLASSIFICATION-AUDIT.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
