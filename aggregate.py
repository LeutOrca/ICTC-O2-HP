"""Audit immutable receipts and aggregate paired operator curves on a CPU node."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import median, quantiles

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT.parents[1] / "figures/o2-hp-angular-operators-20261121-r1"
IDENTITY = ("family", "order", "parity", "rows", "torch", "gpu_uuid",
            "input_sha256", "canonical_weights_sha256")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def spread(values):
    q = quantiles(values, n=4, method="inclusive")
    return median(values), q[0], q[2]


def main():
    records, errors = [], []
    for path in sorted((ROOT / "measurements").rglob("*.json")):
        if "preflight" in path.parts or path.name == "attempts.json":
            continue
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            errors.append(dict(path=str(path.relative_to(ROOT)), error="incomplete JSON during live sync"))
            continue
        if record.get("stage") not in ("build", "qualify", "time", "memory"):
            continue
        record["receipt"] = str(path.relative_to(ROOT))
        record["cohort"] = path.relative_to(ROOT).parts[1]
        records.append(record)
    qualified = {digest(ROOT / r["receipt"]): r for r in records if r["stage"] == "qualify"}
    statuses = [{k: r.get(k, "") for k in ("cohort", "family", "rows", "order", "parity",
                "stage", "role", "metric", "status", "job", "receipt", "error")}
                for r in records if r["stage"] in ("build", "qualify") or not r.get("passed")]
    for record in records:
        if record["stage"] == "build":
            for role, built in record.get("builds", {}).items():
                if not built.get("passed"):
                    statuses.append(dict({k:record.get(k,"") for k in ("cohort","family","rows","order","parity","job","receipt")},
                        stage="build",role=role,metric=record.get("metric",""),status="build-error",error=built.get("error","")))
        if record["stage"] == "qualify":
            roles = sorted({key.split("/")[0] for key in record.get("numerics",{})})
            for metric in ("forward","forward_vjp"):
                for role in roles:
                    if role in record.get("accepted_by_metric",{}).get(metric,[]):
                        continue
                    items = [v for k,v in record["numerics"].items() if k.startswith(role+"/")]
                    exception = next((v["error"] for v in items if "error" in v), "")
                    statuses.append(dict({k:record.get(k,"") for k in ("cohort","family","rows","order","parity","job","receipt")},
                        stage="qualify",role=role,metric=metric,
                        status="qualification-error" if exception else "numerical-failure",error=exception))
    times, memory = {}, {}
    audit = []
    for record in records:
        if record["stage"] not in ("time", "memory") or not record.get("passed"):
            continue
        try:
            q = qualified[record["qualification_sha256"]]
            assert q.get("passed"), "qualification failed"
            if q["family"] in ("tp","uvu","radial") and any(
                v.get("reference_chunk_size",512)>512 and v.get("check_window_size")!=512
                for v in q.get("numerics",{}).values()):
                statuses.append(dict({k:record.get(k,"") for k in ("cohort","family","rows","order","parity","job","receipt","role","metric")},
                    stage=record["stage"],status="superseded-reference-window",error="4096-row reference statistics must be rechecked in original 512-row windows"))
                continue
            if "component_qualifications" in q:
                components = [qualified[sha] for sha in q["component_qualifications"]]
                assert all(all(c.get(k)==q.get(k) for k in IDENTITY) for c in components), "merged gate identities differ"
                for metric,roles in q["accepted_by_metric"].items():
                    assert set(roles)=={role for c in components for role in c["accepted_by_metric"][metric]}, "merged acceptance changed"
            assert all(q.get(k) == record.get(k) for k in IDENTITY), "qualification identity mismatch"
            assert record["dtype"] == "float32" and not any(record[k] for k in ("cuda_graph", "tf32", "amp"))
            assert "H100" in record["gpu"]
            if record["stage"] == "memory":
                family = record["family"]
                if family in ("tp", "uvu", "radial") and "memory-692de50" != record["cohort"] and "discarded_preparation_cotangent_bytes" not in record:
                    continue
                assert record["role"] in q["accepted_by_metric"][record["metric"]]
                key = tuple(record[k] for k in ("family", "rows", "order", "parity", "role", "metric"))
                memory.setdefault(key, []).append(record)
            else:
                for item in record["measurements"]:
                    role, metric = item["role"], item["metric"]
                    assert role in q["accepted_by_metric"][metric]
                    assert len(item["wall_samples_ms"]) == len(item["event_samples_ms"]) == 30
                    assert item["wall_ms"] > 0 and item["event_ms"] > 0
                    assert abs(item["throughput_rows_s"] * item["wall_ms"] / 1000 - record["rows"]) < 1e-5
                    key = tuple(record[k] for k in ("family", "rows", "order", "parity")) + (role, metric, record["cohort"], record["gpu_uuid"])
                    times.setdefault(key, []).append(dict({k: record[k] for k in record if k != "measurements"}, **item))
            audit.append(dict(receipt=record["receipt"], sha256=digest(ROOT / record["receipt"]),
                              qualification=q["receipt"], qualification_sha256=record["qualification_sha256"]))
        except (KeyError, AssertionError) as exc:
            errors.append(dict(path=record["receipt"], error=str(exc)))
    candidates = {}
    incomplete = []
    for key, values in times.items():
        repeats = {r["repeat"]: r for r in values}
        if set(repeats) != set(range(5)):
            incomplete.append(dict(case=list(key), repeats=sorted(repeats)))
            continue
        assert len({tuple(v.get(k) for k in IDENTITY) for v in values}) == 1
        candidates.setdefault(key[:6], []).append((key, [repeats[n] for n in range(5)]))
    summary, picked = [], {}
    for case, options in sorted(candidates.items()):
        # Complete CUDA payload supersedes the earlier incomplete SO2 package.
        key, values = max(options, key=lambda option: (
            "local_hpx_source" in option[1][0], option[1][0].get("process_mode")=="forkserver", "fixed" in option[0][6],
            "2f1e68c" in option[0][6] or "continuation" in option[0][6], option[0][6]))
        family, rows, order, parity, role, metric = case
        wall, wall_lo, wall_hi = spread([r["wall_ms"] for r in values])
        event, _, _ = spread([r["event_ms"] for r in values])
        throughput, low, high = spread([r["throughput_rows_s"] for r in values])
        row = dict(zip(("family", "rows", "order", "parity", "role", "metric"), case))
        row.update(wall_ms=wall, wall_q25_ms=wall_lo, wall_q75_ms=wall_hi,
            event_ms=event, throughput_rows_s=throughput, throughput_q25=low, throughput_q75=high,
            time_gpu_uuid=key[7], time_cohort=key[6], repeats=5,
            input_sha256=values[0]["input_sha256"], canonical_weights_sha256=values[0].get("canonical_weights_sha256", ""),
            equation=values[0].get("equation", ""), input_dimension=values[0].get("input_dimension", ""),
            output_dimension=values[0].get("output_dimension", ""), paths=values[0].get("paths", ""))
        row.update({field:values[0].get(field,"") for field in (
            "torch","gpu","dtype","tf32","amp","cuda_graph","channels","hidden","seed","scale")})
        row.update(process_mode=values[0].get("process_mode","exec"),time_job=values[0]["job"],memory_job="")
        memories = [m for m in memory.get(case, []) if all(m.get(k) == values[0].get(k)
                    for k in ("input_sha256", "canonical_weights_sha256", "torch"))]
        if memories:
            m = max(memories, key=lambda v: (v["cohort"] == key[6], "memory-692de50" == v["cohort"],
                "fixed" in v["cohort"], "2f1e68c" in v["cohort"], v["cohort"]))
            row.update(peak_allocated=m["peak_allocated"], peak_reserved=m["peak_reserved"],
                       memory_gpu_uuid=m["gpu_uuid"], memory_cohort=m["cohort"],memory_job=m["job"])
        else:
            row.update(peak_allocated="", peak_reserved="", memory_gpu_uuid="", memory_cohort="")
        summary.append(row)
        picked[case] = values
    ratios = []
    summary_by_case={tuple(r[k] for k in ("family","rows","order","parity","role","metric")):r for r in summary}
    pairs = {"tp": (("public", "eqx-o2"), ("public", "cue-o2"), ("local-hpx","public"),
                    ("local-hpx","eqx-o2"), ("local-hpx","cue-o2")),
             "uvu": (("public", "eqx-o2"),),
             "radial": (("public", "eqx-o2"), ("public", "public-unfused")),
             "so2": (("so2-cuda2", "so2-native"), ("so2-triton", "so2-native"))}
    for case, candidate in picked.items():
        family, rows, order, parity, role, metric = case
        for candidate_role, baseline_role in pairs.get(family, ()):
            if role != candidate_role:
                continue
            baseline = picked.get(case[:4] + (baseline_role, metric))
            if baseline is None:
                continue
            if any(c["cohort"] != b["cohort"] or any(c.get(k) != b.get(k) for k in IDENTITY)
                   for c, b in zip(candidate, baseline)):
                continue
            speedup, low, high = spread([b["wall_ms"] / c["wall_ms"] for c, b in zip(candidate, baseline)])
            paired_memory=dict(allocated_ratio="",reserved_ratio="",memory_gpu_uuid="",memory_cohort="",memory_status="missing-or-unpaired")
            c_mem=summary_by_case[case]
            b_mem=summary_by_case[case[:4]+(baseline_role,metric)]
            if c_mem["peak_allocated"]!="" and b_mem["peak_allocated"]!="" and (
                c_mem["memory_gpu_uuid"],c_mem["memory_cohort"]) == (b_mem["memory_gpu_uuid"],b_mem["memory_cohort"]):
                paired_memory.update(allocated_ratio=c_mem["peak_allocated"]/b_mem["peak_allocated"],
                    reserved_ratio=c_mem["peak_reserved"]/b_mem["peak_reserved"],memory_gpu_uuid=c_mem["memory_gpu_uuid"],
                    memory_cohort=c_mem["memory_cohort"],memory_status="paired")
            ratios.append(dict(family=family, rows=rows, order=order, parity=parity, metric=metric,
                candidate=role, baseline=baseline_role, speedup=speedup, q25=low, q75=high,
                gpu_uuid=candidate[0]["gpu_uuid"], cohort=candidate[0]["cohort"],**paired_memory))
    summary_fields = ["family", "rows", "order", "parity", "role", "metric", "wall_ms", "wall_q25_ms", "wall_q75_ms",
        "event_ms", "throughput_rows_s", "throughput_q25", "throughput_q75", "peak_allocated", "peak_reserved",
        "time_gpu_uuid", "memory_gpu_uuid", "time_cohort", "memory_cohort", "repeats", "input_sha256", "canonical_weights_sha256",
        "equation", "input_dimension", "output_dimension", "paths","process_mode","time_job","memory_job",
        "torch","gpu","dtype","tf32","amp","cuda_graph","channels","hidden","seed","scale"]
    ratio_fields = ["family", "rows", "order", "parity", "metric", "candidate", "baseline", "speedup", "q25", "q75", "gpu_uuid", "cohort",
        "allocated_ratio","reserved_ratio","memory_gpu_uuid","memory_cohort","memory_status"]
    for target in (ROOT, FIGURES / "data"):
        write_csv(target / "summary.csv", summary, summary_fields)
        write_csv(target / "speedups.csv", ratios, ratio_fields)
        write_csv(target / "status.csv", statuses, ["cohort", "family", "rows", "order", "parity", "stage", "role", "metric", "status", "job", "receipt", "error"])
    coverage = []
    for family, rows in (("tp",4096),("tp",16384),("uvu",16384),("radial",16384),("so2",16384),("ictp",4096),("ictp",16384)):
        for order in range(1,9):
            for parity in (("canonical","full") if family in ("tp","uvu","radial") else ("natural",)):
                points = [r for r in summary if (r["family"],r["rows"],r["order"],r["parity"]) == (family,rows,order,parity)]
                terminal = [r for r in statuses if (r["family"],r["rows"],r["order"],r["parity"]) == (family,rows,order,parity)
                            and r["status"] in ("unsupported","error","oom","numerical-failure","timeout","build_timeout")]
                coverage.append(dict(family=family,rows=rows,order=order,parity=parity,
                    accepted_curves=len(points), status="measured" if points else (terminal[-1]["status"] if terminal else "pending")))
    write_csv(ROOT / "coverage.csv", coverage, ["family","rows","order","parity","accepted_curves","status"])
    role_coverage = []
    expected_roles = {"tp":("public","local-hpx","eqx-o2","cue-o2"),"uvu":("public","eqx-o2"),
        "radial":("public","public-unfused","eqx-o2"),"so2":("so2-native","so2-cuda2","so2-triton"),"ictp":("ictp-native",)}
    for config in coverage:
        for role in expected_roles[config["family"]]:
            for metric in ("forward","forward_vjp"):
                case = tuple(config[k] for k in ("family","rows","order","parity")) + (role,metric)
                if case in picked:
                    status = "measured"
                elif config["family"] == "ictp" and config["order"] > 3:
                    status = "unsupported"
                else:
                    failed = [r for r in statuses if tuple(r.get(k) for k in ("family","rows","order","parity","role","metric")) == case
                              and r["status"] in ("build-error","qualification-error","numerical-failure","oom","error","timeout","build_timeout")]
                    status = failed[-1]["status"] if failed else "pending"
                role_coverage.append(dict(zip(("family","rows","order","parity","role","metric"),case),status=status))
    write_csv(ROOT/"role-coverage.csv",role_coverage,["family","rows","order","parity","role","metric","status"])
    result = dict(summary_points=len(summary), paired_ratios=len(ratios), measured_configurations=sum(r["status"]=="measured" for r in coverage),
        grid_configurations=len(coverage), pending_configurations=[r for r in coverage if r["status"]=="pending"],
        memory_missing_points=sum(r["peak_allocated"]=="" for r in summary), incomplete_repeat_groups=incomplete,
        audit_errors=errors, receipts=audit, strict_fp32=True, cuda_graph=False,
        official_and_fork_same_executables=True, no_cross_equation_speedups=True,
        expected_role_metrics=len(role_coverage),pending_role_metrics=sum(r["status"]=="pending" for r in role_coverage))
    (ROOT / "SUMMARY.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ("receipts","incomplete_repeat_groups","pending_configurations")}), flush=True)
    if errors:
        raise RuntimeError("Receipt audit errors must be resolved before final release")


if __name__ == "__main__":
    main()
