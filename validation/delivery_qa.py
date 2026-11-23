"""Freeze source/history and independently unpack/hash them on a CPU node."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

assert os.environ.get("SLURM_JOB_ID")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "provenance/approved-execution"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


head = (OUT / "SEAL-HEAD.txt").read_text().strip()
tracked = (OUT / "SEAL-TRACKED.nul").read_bytes().decode().rstrip("\0").split("\0")
dependencies = [line.split("  ", 1)[1] for line in (ROOT / "provenance/UPSTREAM.sha256").read_text().splitlines() if line.strip()]
names = sorted(set(tracked + dependencies))
assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in names)
manifest = {name: sha(ROOT / name) for name in names}
archive = OUT / f"source-{head[:7]}-sealed.tar.gz"
with tarfile.open(archive, "w:gz") as tar:
    for name in names:
        tar.add(ROOT / name, arcname=name, recursive=False)
bundle = OUT / f"source-{head[:7]}-history.bundle"
assert bundle.is_file()
scratch = Path(tempfile.mkdtemp(prefix="o2-approved-delivery-"))
try:
    extracted = scratch / "source"
    extracted.mkdir()
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert all(member.isfile() for member in members)
        assert sorted(member.name for member in members) == names
        for member in members:
            target = extracted / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    assert all(sha(extracted / name) == value for name, value in manifest.items())
    for table in ("provenance/UPSTREAM.sha256", "provenance/approved-execution/BASELINE-CODE.sha256"):
        subprocess.run(["sha256sum", "--check", table], cwd=extracted, check=True, stdout=subprocess.DEVNULL)
    # Git object restoration is file organization on the login host; this
    # compute image has no git executable. Independently verify its complete
    # restored-file hash table here instead of installing a new toolchain.
    assert (OUT / "SEAL-RESTORED-HEAD.txt").read_text().strip() == head
    restored_hashes = dict((line.split("  ", 1)[1], line.split("  ", 1)[0])
                          for line in (OUT / "SEAL-RESTORED.sha256").read_text().splitlines())
    assert set(restored_hashes) == set(tracked)
    assert all(restored_hashes[name] == manifest[name] for name in tracked)
    record = dict(job=os.environ["SLURM_JOB_ID"], passed=True, head=head,
                  tracked_files=len(tracked), dependency_entries=len(dependencies), archived_files=len(names),
                  archive=str(archive.relative_to(ROOT)), archive_sha256=sha(archive),
                  bundle=str(bundle.relative_to(ROOT)), bundle_sha256=sha(bundle),
                  member_set_and_hashes_passed=True, independent_git_restore_passed=True,
                  original_baseline_and_dependency_sha_passed=True,
                  numerical_pass=False, g1_all_passed=False,
                  scope="source and recoverability only; GPU receipt recovery remains independent")
    (OUT / "DELIVERY-QA.json").write_text(json.dumps(record, indent=2))
    (OUT / f"source-{head[:7]}-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(record, indent=2), flush=True)
finally:
    shutil.rmtree(scratch)
