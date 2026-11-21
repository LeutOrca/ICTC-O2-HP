"""Read and summarize numerical receipts only on a scheduler CPU node."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    qualifications, screens = [], []
    paths = [*args.root.glob("cohorts/*/results/QUALIFY-*.json"),
             *args.root.glob("cohorts/*/measurements/*/results/qualification*.json")]
    for path in sorted(paths):
        report = json.loads(path.read_text())
        records = report["records"]
        qualifications.append(dict(path=str(path.relative_to(args.root)),
            source=report["source"], suite=report["suite"],
            implementation=report.get("implementation", "packed"),
            start_block=report.get("start_block", 0),
            metrics_dtype=report.get("metrics_dtype", "historical FP64 statistics"),
            records=len(records), complete=len(records) == 4,
            passed=len(records) == 4 and report["passed"] and all(
                value["passed"] for row in records for value in row["numerics"].values()),
            force_relative_l2=[row["numerics"]["forces"]["relative_l2"] for row in records],
            seeds=[row["seed"] for row in records],
            boundaries=[row["boundary"] for row in records]))
    for path in sorted(args.root.glob("cohorts/*/results/*SCREEN.json")):
        report = json.loads(path.read_text())
        screens.append(dict(path=str(path.relative_to(args.root)), source=report["source"],
                            metrics_dtype=report.get("metrics_dtype", "historical FP64 statistics"),
                            records=report["records"]))
    pairs = []
    for path in sorted(args.root.glob("cohorts/*/measurements/*/results/V3_LOCAL_PAIR_AUDIT-*.json")):
        row = json.loads(path.read_text())
        qualification = path.parent / f"qualification-{row['case']}.json"
        row["implementation"] = (json.loads(qualification.read_text())["implementation"]
                                 if row.get("status") == "complete" else "unresolved")
        pairs.append(dict(path=str(path.relative_to(args.root)), **row))
    pair_coverage = {}
    for row in pairs:
        if row.get("status") == "complete":
            pair_coverage.setdefault(row["implementation"], []).append(row["case"])
    timing_started = any(args.root.glob("cohorts/*/measurements/*/results/v3-time-*.json"))
    latest = [row for row in pairs if row["implementation"] == "fused-compact" and row.get("status") == "complete"]
    complete = {row["case"] for row in latest} == {"mptrj-b1", "mptrj-b8"}
    acceptance = dict(numerics_fp32=None, whole_network_speed_2x=None,
                      allocated_10pct=None, reserved_10pct=None,
                      actual_checkpoint_so2_speed_5x="not established by these whole-network measurements")
    if complete:
        sources = {row["source"] for row in latest}
        current_quals = [row for row in qualifications if row["source"] in sources
                         and row["implementation"] == "fused-compact"]
        acceptance["numerics_fp32"] = (bool(current_quals) and all(
            row["complete"] and row["passed"] and row["metrics_dtype"] == "torch.float32" for row in current_quals))
        acceptance["whole_network_speed_2x"] = all(
            values["vs_native"]["speedup"] >= 2 for row in latest for values in row["boundaries"].values())
        for target, field in (("allocated_10pct", "allocated_reduction"), ("reserved_10pct", "reserved_reduction")):
            acceptance[target] = all(values["vs_native"][field] >= .1
                                     for row in latest for values in row["boundaries"].values())
    summary = dict(baseline="c591b16", model="EquiformerV3 + DeNS-OAM",
        metrics_dtype="torch.float32", thresholds=dict(atol=2e-5, rtol=2e-5, relative_l2=1e-5),
        checkpoint_sha256="429ccded98163122e7ba588d78e2441653f37f3e091e106c432807fe373c8f98",
        scope="14 SO2Linear forward and input VJP; native norm unchanged",
        parity="official natural parity", angular_order="integer", tf32=False, amp=False,
        cuda_graph=False, timing_started=timing_started, whole_network_pairs=pairs,
        performance_coverage=pair_coverage,
        latest_implementation="fused-compact", acceptance=acceptance,
        performance_complete=set(pair_coverage.get("fused-compact", [])) == {"mptrj-b1", "mptrj-b8"},
        qualifications=qualifications, screens=screens)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    for row in qualifications:
        print(json.dumps(row), flush=True)
    for screen in screens:
        for row in screen["records"]:
            print(json.dumps(dict(screen=screen["path"], implementation=row["forward"],
                                  seed=row.get("seed"), passed=row["passed"],
                                  force_relative_l2=row["numerics"]["forces"]["relative_l2"])), flush=True)


if __name__ == "__main__":
    main()
