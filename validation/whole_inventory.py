"""Read checkpoint/config/input provenance on a CPU compute node, without runs."""
import hashlib
import json
import os
from pathlib import Path

assert os.environ.get("SLURM_JOB_ID")
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT.parent
import torch


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def simple(value):
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), dtype=str(value.dtype))
    if isinstance(value, dict):
        return {str(k): simple(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [simple(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


v3path = RUNS / "o2-hpx-equiformerv3-oam-20261119-r1/checkpoint/omat24-mptrj-salex_gradient.pt"
assert sha(v3path) == "429ccded98163122e7ba588d78e2441653f37f3e091e106c432807fe373c8f98"
checkpoint = torch.load(v3path, map_location="cpu", weights_only=False)
config = checkpoint["config"]
optimizer = checkpoint.get("optimizer")
v3 = dict(path=str(v3path), sha256=sha(v3path), keys=sorted(checkpoint),
          model=simple(config.get("model_attributes", config.get("model"))),
          optimizer_config=simple(config.get("optim", config.get("optimizer"))),
          loss=simple({k: v for k, v in config.items() if any(t in k.lower() for t in ("loss", "dens", "noise", "output", "normal"))}),
          trainer=simple(config.get("trainer")),
          checkpoint_optimizer_present=optimizer is not None,
          optimizer_groups=[] if optimizer is None else [simple({k: v for k, v in group.items() if k != "params"}) | dict(parameter_count=len(group.get("params", []))) for group in optimizer.get("param_groups", [])],
          optimizer_state_entries=0 if optimizer is None else len(optimizer.get("state", {})),
          normalizers=simple(checkpoint.get("normalizers")), elementrefs=simple(checkpoint.get("elementrefs")),
          state_keys=len(checkpoint["state_dict"]))
del checkpoint, optimizer
mace = RUNS / "mace-ictc-o2-hp-gbu2-20261119-r1/frozen"
macepath = mace / "models/pretrained/native.model"
assert sha(macepath) == "d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a"
inputs = []
for index in (*range(8), 13):
    path = mace / "inputs/pretrained" / f"validation-{index:03d}.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    inputs.append(dict(path=str(path), sha256=sha(path), keys=sorted(data), schema=simple(data)))
    del data
record = dict(job=os.environ["SLURM_JOB_ID"], stage="whole-model-source-inventory-only", gpu_jobs_started=False,
              v3=v3, mace=dict(checkpoint=str(macepath), sha256=sha(macepath), inputs=inputs),
              whole_model_numerical_pass=False, speed_pass=False, memory_pass=False)
destination = ROOT / "provenance/approved-execution/WHOLE-INVENTORY.json"
destination.write_text(json.dumps(record, indent=2))
print(json.dumps(dict(job=record["job"], v3_optimizer_present=v3["checkpoint_optimizer_present"], mace_inputs=len(inputs))), flush=True)
