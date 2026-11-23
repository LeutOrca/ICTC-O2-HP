"""CPU topology audit: preserve every original directed contribution via pairs."""
import hashlib
import json
import os
from pathlib import Path
import sys

assert os.environ.get("SLURM_JOB_ID")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "comparisons/eqx-0ff7cad"))
import torch
from equivariantx.ops.common.edge_pairs import undirected_edge_pairs

source = ROOT.parent / "mace-ictc-o2-hp-gbu2-20261119-r1/frozen/inputs/pretrained"
records = []
for index in (*range(8), 13):
    path = source / f"validation-{index:03d}.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    edges = data["edge_index"]
    shifts = data["unit_shifts"].to(torch.int64)
    nodes = len(data["positions"])
    pairs = undirected_edge_pairs(edges, shifts, num_nodes=nodes)
    base = edges[:, pairs.representatives]
    base_shifts = shifts[pairs.representatives]
    expanded, expanded_shifts = base[:, pairs.inverse], base_shifts[pairs.inverse]
    same = (edges == expanded).all(0) & (shifts == expanded_shifts).all(1)
    reverse = (edges == expanded.flip(0)).all(0) & (shifts == -expanded_shifts).all(1)
    assert bool((same | reverse).all())
    sign = torch.where(same, 1, -1)
    reconstructed_edges = torch.where(same[None], expanded, expanded.flip(0))
    reconstructed_shifts = expanded_shifts * sign[:, None]
    assert torch.equal(reconstructed_edges, edges) and torch.equal(reconstructed_shifts, shifts)
    counts = torch.bincount(pairs.inverse, minlength=len(base_shifts))
    positive = torch.bincount(pairs.inverse, weights=same.float(), minlength=len(base_shifts)).long()
    negative = counts - positive
    # More than two contributions may belong to one invariant pair; their
    # original multiplicity and direction order remain in inverse/sign.
    records.append(dict(input=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        N=nodes, E=edges.shape[1], B=len(base_shifts),
                        singleton_pairs=int((counts == 1).sum()),
                        pairs_with_both_directions=int(((positive > 0) & (negative > 0)).sum()),
                        extra_repeated_contributions=int((counts - 2).clamp_min(0).sum()),
                        periodic_self_contributions=int((edges[0] == edges[1]).sum()),
                        exact_direction_sequence_and_integer_lattice_shifts=True,
                        directed_contributions_added_or_removed=0))
result = dict(job=os.environ["SLURM_JOB_ID"], passed=True,
              eqx_source="0ff7cad1395ae88d8aaf50633a04e2191b3f2c1c", cases=records,
              scope="topology indexing only; no model/geometry/kernel numerical qualification or performance",
              whole_model_pass=False)
(ROOT / "provenance/approved-execution/HALF-GRAPH-AUDIT.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
