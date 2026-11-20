"""Independently restore the Git bundle and every exported cohort on CPU Slurm."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--exports", type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--current-cohort", type=Path)
    args = parser.parse_args()
    rows = []
    with tempfile.TemporaryDirectory(prefix="hpx-independent-restore-") as temporary:
        root = Path(temporary)
        repository = root / "repository"
        if args.exports is None:
            subprocess.run(["git", "clone", "-q", str(args.bundle.resolve()), str(repository)], check=True)
        for directory in sorted((args.root / "cohorts").iterdir()):
            identity = directory / "provenance/commit.txt"
            manifest = directory / "provenance/input.sha256"
            if not identity.exists() or not manifest.exists():
                continue
            commit = identity.read_text().strip()
            if args.exports is None:
                archive = root / "source.tar"
                with archive.open("wb") as output:
                    subprocess.run(["git", "archive", commit], cwd=repository, stdout=output, check=True)
            else:
                archive = args.exports / f"{directory.name}.tar"
            export = root / directory.name
            export.mkdir()
            with tarfile.open(archive) as contents:
                contents.extractall(export, filter="data")
            files = 0
            for line in manifest.read_text().splitlines():
                digest, name = line.split("  ", 1)
                path = export / name
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise RuntimeError(dict(cohort=directory.name, file=name, mismatch=True))
                files += 1
            rows.append(dict(cohort=directory.name, commit=commit, files=files, passed=True))
        source_archive = None
        if args.source_archive is not None:
            if args.current_cohort is None:
                raise ValueError("Specify the current source cohort for archive validation")
            unpacked = root / "source-archive"
            unpacked.mkdir()
            with tarfile.open(args.source_archive) as contents:
                contents.extractall(unpacked, filter="data")
            export = unpacked / "hpx-custom-kernels"
            expected = set()
            for line in (args.current_cohort / "provenance/input.sha256").read_text().splitlines():
                expected_hash, name = line.split("  ", 1)
                expected.add(name)
                if hashlib.sha256((export / name).read_bytes()).hexdigest() != expected_hash:
                    raise RuntimeError(dict(archive_source_mismatch=name))
            actual = {str(path.relative_to(export)) for path in export.rglob("*") if path.is_file()}
            if actual != expected:
                raise RuntimeError("Source archive member set differs from the frozen source")
            source_archive = dict(passed=True, files=len(expected),
                                  sha256=hashlib.sha256(args.source_archive.read_bytes()).hexdigest())
    result = dict(passed=True, independent_restores=rows,
                  exports_from_bundle_on_control_host=args.exports is not None,
                  source_archive=source_archive,
                  bundle_sha256=hashlib.sha256(args.bundle.read_bytes()).hexdigest())
    (args.root / "provenance/SNAPSHOT_QA.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(passed=True, cohorts=len(rows), bundle_sha256=result["bundle_sha256"])), flush=True)


if __name__ == "__main__":
    main()
