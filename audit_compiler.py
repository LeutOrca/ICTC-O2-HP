"""Audit FP32 compiler boundaries and 14-layer numerical receipts on CPU."""
import argparse
import hashlib
import json
from pathlib import Path


def scan(cohort):
    kernels, forbidden = [], []
    for path in sorted((cohort / "cache/triton").glob("*/*.ptx")):
        source = path.read_text()
        if ".f64" in source or ".tf32" in source:
            forbidden.append(str(path))
    for path in sorted((cohort / "cache/triton").glob("*/_partials.ttgir")):
        source = path.read_text()
        lines = [line.strip() for line in source.splitlines() if any(
            token in line for token in ("tt.dot", "inline_asm", "arith.addf", "scf.yield"))]
        kernels.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            preserves_boundary="elementwise_inline_asm" in source and "arith.addf" in source,
            lines=lines))
    return dict(forbidden=forbidden, kernels=kernels,
                preserved=sum(row["preserves_boundary"] for row in kernels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    before = scan(args.root / "cohorts/73372ca")
    after = scan(args.root / "cohorts/8cf3a8f")
    diagnosis = json.loads((args.root / "cohorts/7f2eec4/results/DIAGNOSIS-NATIVE-BLOCK.json").read_text())
    local = diagnosis["local"]
    forward_exact = sum(all(value["max_abs"] == 0 for value in row["forward"]) for row in local.values())
    first_reverse_exact = sum(row["input_vjp"]["max_abs"] == 0 for name, row in local.items()
                              if name.endswith("so2_linear_1"))
    passed = (not before["forbidden"] and not after["forbidden"] and after["preserved"] > 0
              and forward_exact == 14 and first_reverse_exact == 7
              and diagnosis["metrics_dtype"] == "torch.float32")
    report = dict(passed=passed, before=before, after=after, forward_exact_layers=forward_exact,
                  first_reverse_exact_layers=first_reverse_exact,
                  source=diagnosis["source"], metrics_dtype="torch.float32")
    (args.root / "provenance/COMPILER_QA.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(passed=passed, before_preserved=before["preserved"],
                         after_preserved=after["preserved"], forward_exact_layers=forward_exact,
                         first_reverse_exact_layers=first_reverse_exact)), flush=True)
    assert passed


if __name__ == "__main__":
    main()
