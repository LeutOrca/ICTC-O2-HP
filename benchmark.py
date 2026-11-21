"""Separate-process FP32 build, qualification, paired timing and memory."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import time
import traceback

import torch
from models import Spec, ROOT


def error(actual, reference):
    if actual.dtype != torch.float32 or reference.dtype != torch.float32:
        raise TypeError("Numerical references and statistics must be FP32")
    delta = (actual.detach() - reference.detach()).abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    if not finite:
        return dict(passed=False, max_abs=None, relative_l2=None, arithmetic="float32", finite=False)
    reference = reference.detach()
    relative = float(delta.norm() / reference.norm().clamp_min(1e-30))
    passed = finite and relative <= 1e-5 and bool((
        (delta <= 1e-5 + 1e-4 * reference.abs()) &
        (delta <= 2e-5 + 2e-5 * reference.abs())).all())
    return dict(passed=passed, max_abs=float(delta.max()), relative_l2=relative,
                arithmetic="float32")


def configure():
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import torch._inductor.config as cfg
    cfg.triton.cudagraphs = False


def prepare(spec, role, cotangent_seed=20261121):
    inputs = spec.encoded(role)
    with torch.no_grad():
        output = spec.calls[role](*inputs)
        decoded = spec.decode[role](output)
    generator = torch.Generator(device="cuda").manual_seed(cotangent_seed)
    cotangent = torch.randn(decoded.shape, device="cuda", generator=generator) * .1
    dummy = torch.zeros_like(output, requires_grad=True)
    native_cotangent = torch.autograd.grad(spec.decode[role](dummy), dummy, cotangent)[0].detach()
    return inputs, native_cotangent, cotangent


def invoke(function, inputs, cotangent, metric):
    if metric == "forward":
        with torch.no_grad():
            return function(*inputs)
    output = function(*inputs)
    gradients = torch.autograd.grad(output, inputs, cotangent)
    if len(gradients) != len(inputs) or any(value is None for value in gradients):
        raise RuntimeError("An input VJP was omitted")
    return output, gradients


def qualify(spec, role, metric="forward_vjp", reference_chunk=512):
    inputs, cotangent, logical_cotangent = prepare(spec, role)
    if metric == "forward":
        with torch.no_grad():
            actual = spec.calls[role](*inputs)
            value = spec.decode[role](actual)
        gradients = []
    else:
        actual = spec.calls[role](*inputs)
        gradients = torch.autograd.grad(actual, inputs, cotangent)
        value = spec.decode[role](actual.detach())
        gradients = [g.detach() for g in spec.decode_grad(role, gradients)]
    chunks = []
    chunk_size = spec.rows if spec.family == "so2" else reference_chunk
    for start in range(0, spec.rows, chunk_size):
        end = min(spec.rows, start + chunk_size)
        logical = [v[start:end].detach().requires_grad_() for v in spec.logical]
        if metric == "forward":
            with torch.no_grad():
                expected = spec.reference(*logical)
            expected_gradients = []
        else:
            expected = spec.reference(*logical)
            expected_gradients = torch.autograd.grad(expected, logical, logical_cotangent[start:end])
        check_window = spec.rows if spec.family == "so2" else 512
        # Reference batching must not enlarge the original error-statistics
        # window: check every original 512-row region independently.
        for window in range(start,end,check_window):
            stop=min(end,window+check_window)
            local=slice(window-start,stop-start)
            checks = [error(value[window:stop], expected[local])] + [
                error(got[window:stop], ref[local]) for got, ref in zip(gradients, expected_gradients)]
            chunks.append(dict(start=window, end=stop, checks=checks))
        del logical, expected, expected_gradients
    return dict(passed=all(c["passed"] for row in chunks for c in row["checks"]),
                forward_passed=all(row["checks"][0]["passed"] for row in chunks),
                vjp_passed=metric=="forward_vjp" and all(c["passed"] for row in chunks for c in row["checks"]),
                chunks=chunks, full_workload=True, reference_chunk_size=chunk_size,
                check_window_size=spec.rows if spec.family == "so2" else 512)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("build", "qualify", "time", "memory"))
    p.add_argument("family", choices=("tp", "uvu", "radial", "so2", "ictp"))
    p.add_argument("order", type=int, choices=range(1, 9))
    p.add_argument("parity", choices=("canonical", "full", "natural"))
    p.add_argument("rows", type=int)
    p.add_argument("output", type=Path)
    p.add_argument("--qualification", type=Path)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--role")
    p.add_argument("--metric", choices=("forward", "forward_vjp"), default="forward_vjp")
    p.add_argument("--reference-chunk",type=int,choices=(512,4096),default=int(os.getenv("O2_REFERENCE_CHUNK","512")))
    a = p.parse_args()
    record = dict(stage=a.stage, family=a.family, order=a.order, parity=a.parity,
                  rows=a.rows, repeat=a.repeat, role=a.role, metric=a.metric,
                  job=os.getenv("SLURM_JOB_ID"), status="starting", passed=False,
                  cuda_graph=False, tf32=False, amp=False, dtype="float32")
    record.update(pid=os.getpid(),process_mode=os.getenv("O2_PROCESS_MODE","exec"))
    try:
        configure()
        record.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                      gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
        spec = Spec(a.family, a.order, a.parity, a.rows)
        record.update(spec.metadata)
        selected_roles = [a.role] if a.role else list(spec.calls)
        if a.stage == "build":
            results = {}
            for role in selected_roles:
                function = spec.calls[role]
                record.update(status="building", active_role=role)
                a.output.parent.mkdir(parents=True, exist_ok=True)
                a.output.write_text(json.dumps(record, indent=2))
                print("building", role, a.metric, flush=True)
                try:
                    inputs, cotangent, _ = prepare(spec, role)
                    invoke(function, inputs, cotangent, a.metric)
                    torch.cuda.synchronize()
                    results[role] = dict(passed=True)
                except Exception as exc:
                    results[role] = dict(passed=False, error=str(exc), traceback=traceback.format_exc())
            record.update(status="built", passed=any(r["passed"] for r in results.values()), builds=results)
        elif a.stage == "qualify":
            results = {}
            for seed in (20261003, 20261004):
                for scale in (.1, 1.):
                    if seed != 20261003 or scale != .1:
                        del spec
                        gc.collect()
                        torch.cuda.empty_cache()
                        spec = Spec(a.family, a.order, a.parity, a.rows, seed, scale)
                    for role in selected_roles:
                        key = f"{role}/{seed}/{scale}"
                        try:
                            results[key] = qualify(spec, role, a.metric,a.reference_chunk)
                        except Exception as exc:
                            results[key] = dict(passed=False, error=str(exc), traceback=traceback.format_exc())
            roles = selected_roles
            by_metric = {}
            for metric, field in (("forward", "forward_passed"), ("forward_vjp", "vjp_passed")):
                by_metric[metric] = [role for role in roles if all(
                    results[f"{role}/{seed}/{scale}"].get(field, False)
                    for seed in (20261003, 20261004) for scale in (.1, 1.))]
            accepted = [role for role in roles if any(role in values for values in by_metric.values())]
            record.update(status="qualified" if accepted else "numerical-failure",
                          passed=bool(accepted), accepted=accepted, accepted_by_metric=by_metric, numerics=results)
        else:
            q = json.loads(a.qualification.read_text())
            if not q["passed"] or (q["family"], q["order"], q["parity"], q["rows"], q["torch"]) != (
                    a.family, a.order, a.parity, a.rows, torch.__version__):
                raise RuntimeError("Matching accepted qualification is required")
            record["qualification_sha256"] = hashlib.sha256(a.qualification.read_bytes()).hexdigest()
            if q["input_sha256"] != spec.metadata["input_sha256"] or q["gpu_uuid"] != record["gpu_uuid"]:
                raise RuntimeError("Inputs or GPU differ from the accepted paired qualification")
            if q.get("canonical_weights_sha256") != spec.metadata.get("canonical_weights_sha256"):
                raise RuntimeError("Frozen weights differ from qualification")
            record["accepted"] = q["accepted"]
            if a.stage == "memory":
                if a.role not in q.get("accepted_by_metric", {}).get(a.metric, q["accepted"]):
                    raise RuntimeError("Role failed qualification")
                inputs, cotangent, unused_native_cotangent = prepare(spec, a.role)
                record["discarded_preparation_cotangent_bytes"] = (
                    0 if cotangent.data_ptr() == unused_native_cotangent.data_ptr()
                    else unused_native_cotangent.numel() * unused_native_cotangent.element_size())
                del unused_native_cotangent
                function = spec.calls[a.role]
                del spec
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                output = invoke(function, inputs, cotangent, a.metric)
                torch.cuda.synchronize()
                record.update(peak_allocated=torch.cuda.max_memory_allocated(),
                              peak_reserved=torch.cuda.max_memory_reserved())
                del output
            else:
                roles = [role for role in q["accepted"] if role in spec.calls]
                roles = roles[a.repeat % len(roles):] + roles[:a.repeat % len(roles)]
                if a.repeat % 2:
                    roles.reverse()
                prepared = {role: prepare(spec, role)[:2] for role in roles}
                measurements = []
                for role in roles:
                    for metric in (("forward", "forward_vjp") if a.repeat % 2 == 0 else ("forward_vjp", "forward")):
                        if role not in q.get("accepted_by_metric", {}).get(metric, q["accepted"]):
                            continue
                        inputs, cotangent = prepared[role]
                        for _ in range(30):
                            output = invoke(spec.calls[role], inputs, cotangent, metric)
                            del output
                        samples, events = [], []
                        for _ in range(30):
                            torch.cuda.synchronize()
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            tick = time.perf_counter()
                            start.record()
                            output = invoke(spec.calls[role], inputs, cotangent, metric)
                            end.record()
                            torch.cuda.synchronize()
                            samples.append((time.perf_counter() - tick) * 1000)
                            events.append(start.elapsed_time(end))
                            del output
                        measurements.append(dict(role=role, metric=metric,
                            wall_ms=median(samples), event_ms=median(events), wall_samples_ms=samples,
                            event_samples_ms=events, throughput_rows_s=a.rows*1000/median(samples)))
                record["measurements"] = measurements
            record.update(status="measured", passed=True)
    except Exception as exc:
        status = "unsupported" if a.family == "ictp" and a.order > 3 else (
            "oom" if isinstance(exc, torch.OutOfMemoryError) else "error")
        record.update(status=status, error=str(exc), traceback=traceback.format_exc(), passed=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(record, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in record.items() if k not in
          ("traceback", "numerics", "measurements", "builds")}), flush=True)
    if not record["passed"] and record["status"] != "unsupported":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
