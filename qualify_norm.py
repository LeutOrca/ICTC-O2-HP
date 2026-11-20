"""Compare fused V3 merge norm and its input adjoint with official equations."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import torch

from merge_norm import merge_norm
from qualify_matrix import compare, time_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    path = Path(os.environ["HPX_V3_UPSTREAM"]) / "experimental/models/equiformer_v3/layer_norm.py"
    spec = importlib.util.spec_from_file_location("official_v3_norm", path)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    torch.manual_seed(20261120)
    records = []
    for nodes, lmax, channels in ((0, 4, 128), (7, 0, 65), (68, 4, 128), (600, 4, 128)):
        for centering in (False, True):
            norm = source.EquivariantMergeLayerNorm(lmax, channels, centering=centering)
            norm.cuda().requires_grad_(False)
            inputs = torch.randn(nodes, (lmax + 1) ** 2, channels, device="cuda", requires_grad=True)
            reference = norm(inputs)
            result = merge_norm(inputs, norm)
            incoming = torch.randn_like(result)
            ref_gradient, = torch.autograd.grad(reference, inputs, incoming)
            gradient, = torch.autograd.grad(result, inputs, incoming)
            row = dict(shape=inputs.shape, centering=centering, forward=compare(result, reference),
                       input_adjoint=compare(gradient, ref_gradient))
            if not row["forward"]["passed"] or not row["input_adjoint"]["passed"]:
                args.output.write_text(json.dumps(dict(records=records + [row], passed=False), indent=2))
                raise RuntimeError(row)
            if nodes:
                row["native_event_ms"] = time_ms(lambda: norm(inputs))
                row["own_event_ms"] = time_ms(lambda: merge_norm(inputs, norm))
            records.append(row)
    args.output.write_text(json.dumps(dict(records=records, passed=True), indent=2))
    print(json.dumps(dict(records=records, passed=True)), flush=True)


if __name__ == "__main__":
    main()
