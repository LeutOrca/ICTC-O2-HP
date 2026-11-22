"""Read official wheel metadata on a CPU Slurm node."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = json.loads((root / "downloads/tavotto-0.21.3-pypi.json").read_text())
assert data["info"]["version"] == "0.21.3"
wheel = next(item for item in data["urls"] if item["filename"].endswith("py3-none-any.whl"))
(root / "downloads/TAVOTTO-WHEEL.tsv").write_text(wheel["url"] + "\t" + wheel["filename"] + "\t" + wheel["digests"]["sha256"] + "\n")
print(wheel["filename"], wheel["digests"]["sha256"], flush=True)
print("requires", data["info"]["requires_dist"], flush=True)
