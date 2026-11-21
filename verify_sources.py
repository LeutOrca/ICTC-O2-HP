"""Compute-node seal of byte-identical official/fork executables and sources."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    def hashes(folder):
        return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(folder.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    official = hashes(root / "upstream/official/src")
    fork = hashes(root / "upstream/fork/src")
    if official != fork:
        raise RuntimeError("Official and fork executables diverged; they cannot share a curve")
    result = dict(public_executables_identical=True, official="96815838f470652dce3d0e9d5cf7fe2978e949ca",
        fork="296e61f1c13b8ad5ff525cbe3d0bd63d4d6adaca", public=official,
        eqx="c646c7000af7adb367d6bd8b7eb5ea74c08dc37c", ictp="f40592a5687ec1d03219300ee557b2660f7d0369",
        all_source_hashes=hashes(root / "upstream"))
    (root / "provenance/SOURCE-SEAL.json").write_text(json.dumps(result, indent=2))
    print("Official and fork executables identical:", len(official), "files", flush=True)


if __name__ == "__main__":
    main()
