"""Freeze explicit source/receipt/figure member sets on a Slurm CPU node."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT.parents[1] / "figures/o2-hp-angular-operators-20261121-r1/repair-20261122-r1"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    tracked = (ROOT / "provenance/DELIVERY-TRACKED.nul").read_bytes().split(b"\0")
    source = {s.decode() for s in tracked if s}
    source.update(str(p.relative_to(ROOT)) for p in (ROOT / "upstream").rglob("*")
                  if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts)
    receipt = set()
    for directory in ("measurements", "logs", "provenance"):
        for path in (ROOT / directory).rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            if path.name.startswith("DELIVERY") or path.name.endswith((".tgz", ".tar.gz", ".whl")):
                continue
            if path.name.startswith("o2-repair-delivery-manifest-"):
                continue  # This job's stdout is still being written.
            if "__pycache__" in path.parts or path.name == "SSH-CONTROL-PATH.txt":
                continue
            receipt.add(str(path.relative_to(ROOT)))
    figures = {str(p.relative_to(FIGURES)) for p in FIGURES.rglob("*") if p.is_file()
               and not any(x in p.parts for x in (".mpl-cache", "__pycache__"))}
    groups = {"sources": (ROOT, source), "receipts": (ROOT, receipt), "figures": (FIGURES, figures)}
    manifest = {}
    for name, (base, paths) in groups.items():
        manifest[name] = {str(p): digest(base / p) for p in sorted(paths)}
        (ROOT / f"provenance/DELIVERY-{name}.nul").write_bytes(b"".join(p.encode()+b"\0" for p in sorted(paths)))
    (ROOT / "provenance/DELIVERY-MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: len(v) for k, v in manifest.items()}), flush=True)


if __name__ == "__main__":
    main()
