"""CPU-node inspection of saved compiled PTX; numerical receipts remain separate."""
import hashlib
import json
import os
from pathlib import Path
import re

assert os.environ.get("SLURM_JOB_ID")
ROOT = Path(__file__).resolve().parents[1]
rows = []
for path in sorted((ROOT / "receipts/compiled").rglob("*.ptx")):
    content = path.read_text()
    instructions = "\n".join(line for line in content.splitlines()
                             if not line.lstrip().startswith(("//", ".file", ".loc")))
    forbidden = re.findall(r"\b(?:mma|wgmma)[^;\n]*\.tf32\b|\b(?:fma|add|mul|div|cvt)[^;\n]*\.f64\b|\b(?:mma|wgmma|fma)[^;\n]*\.(?:f16|bf16)\b", instructions)
    rows.append(dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     entries=re.findall(r"\.entry\s+([\w$]+)", content),
                     target=re.findall(r"\.target\s+([^\n]+)", content),
                     fma_fp32=len(re.findall(r"fma\.[^\s;]*f32", instructions)),
                     wide_address_multiply=len(re.findall(r"mul\.wide\.[us]32", instructions)),
                     mul64=len(re.findall(r"(?:mul|mad)\.[^\s;]*[us]64", instructions)), forbidden=forbidden))
record = dict(job=os.environ["SLURM_JOB_ID"], passed=bool(rows) and not any(row["forbidden"] for row in rows),
              ptx_files=len(rows), rows=rows, numerical_pass=False,
              scope="saved explicit-candidate Triton compilation artifacts, not whole-network vendor trace")
(ROOT / "provenance/approved-execution/PTX-AUDIT.json").write_text(json.dumps(record, indent=2))
print(json.dumps(dict(passed=record["passed"], ptx_files=len(rows))), flush=True)
if not record["passed"]:
    raise SystemExit(2)
