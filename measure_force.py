"""Crossed same-GPU order for qualified MACE CUDA Graph variants."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

import conservative_runner as bench
from qualify_force import CASES, MODES, inputs_for, replay_for, setup

BASE = "native-graph"
configure_fp32 = bench.configure


def configure():
    configure_fp32()
    torch.use_deterministic_algorithms(True)


def model_for(backend, device):
    model = setup(backend)
    model._experimental_cuda_graph = True
    return model


def evaluate(model, data):
    if not hasattr(model, "_graph_execution"):
        model._graph_execution = replay_for(model, data)
    return model._graph_execution.call(data)


def build_graph(geometry, model, device):
    return inputs_for(geometry, model, device)


def passed(path, case, backend):
    return json.loads((path / "receipts" / f"{case}-{backend}.json").read_text())["passed"]


def cleanup():
    gc.collect()
    torch.cuda.synchronize()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("trace", "time", "memory"))
    parser.add_argument("--variant", choices=tuple(MODES)[1:], default="receiver-csr-graph")
    parser.add_argument("--case", choices=tuple(bench.CASES), action="append")
    parser.add_argument("--baseline", choices=(BASE, "native-graph-tiled"), default=BASE)
    parser.add_argument("--compare-prior", choices=("receiver-csr-graph", "receiver-csr-partition-graph"))
    parser.add_argument("--qualification", type=Path, default=bench.ROOT)
    parser.add_argument("--wait-for", type=Path)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.wait_for:
        while not args.wait_for.exists():
            if (bench.ROOT / "failed").exists():
                raise SystemExit("A preceding CUDA stage failed")
            time.sleep(0.5)
    cases = args.case or CASES
    backends = (args.baseline, args.variant) if args.compare_prior is None else (
        args.baseline, args.compare_prior, args.variant)
    if not all(passed(args.qualification, case, backend) for case in cases for backend in backends):
        raise SystemExit("The requested paired cases did not pass strict FP32")
    bench.configure = configure
    bench.model_for = model_for
    bench.native_graph = build_graph
    bench.evaluate = evaluate
    cleanup()
    for repeat in range(1, (1 if args.stage == "trace" else args.rounds) + 1):
        order = backends if repeat % 2 else backends[::-1]
        for case in cases:
            for boundary in ("fixed", "e2e"):
                for backend in order:
                    print(args.stage, case, boundary, backend, repeat, flush=True)
                    if torch.cuda.memory_allocated() > 1024 ** 2:
                        raise RuntimeError("A preceding arm retained more than 1 MiB")
                    bench.run(argparse.Namespace(device="cuda", case=case, backend=backend,
                              stage=args.stage, boundary=boundary, warmup=6, repeats=15, round=repeat))
                    cleanup()
    (bench.ROOT / f"{args.stage}.ok").touch()


if __name__ == "__main__":
    main()
