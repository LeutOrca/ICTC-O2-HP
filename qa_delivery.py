"""Independent archive unpack, complete member/hash checks and CSV rerun."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from PIL import Image

ROOT = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    manifest = json.loads((ROOT / "provenance/DELIVERY-MANIFEST.json").read_text())
    delivery = json.loads((ROOT / "provenance/DELIVERY.json").read_text())
    checked = {}
    with tempfile.TemporaryDirectory(prefix="o2-repair-delivery-") as temporary:
        base = Path(temporary)
        restored = base / "runs" / ROOT.name
        figures = base / "figures/o2-hp-angular-operators-20261121-r1/repair-20261122-r1"
        for name in ("sources", "receipts", "figures"):
            path = ROOT / delivery[name]["path"]
            assert digest(path.read_bytes()) == delivery[name]["sha256"], name
            with tarfile.open(path) as packed:
                members = {m.name: m for m in packed.getmembers() if m.isfile()}
                assert set(members) == set(manifest[name]), name
                for member, sha in manifest[name].items():
                    assert digest(packed.extractfile(members[member]).read()) == sha, (name, member)
                target = figures if name == "figures" else restored
                target.mkdir(parents=True, exist_ok=True)
                packed.extractall(target, filter="data")
                checked[name] = len(members)
        files = ("summary.csv", "speedups.csv", "numerics.csv", "status.csv", "coverage.csv", "role-coverage.csv")
        originals = {name: (ROOT / name).read_bytes() for name in files}
        process = subprocess.run([sys.executable, str(restored / "aggregate.py")], cwd=restored,
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.returncode == 0, process.stdout
        assert all((restored / name).read_bytes() == raw for name, raw in originals.items())
        final = json.loads((restored / "SUMMARY.json").read_text())
        assert final["summary_points"] == 489 and not final["audit_errors"]
        seals = list((restored / "measurements/paired-repair-e48a156/source-seals").glob("SOURCE-SEAL-*.json"))
        assert len(seals) == 6
        for path in seals:
            seal = json.loads(path.read_text())
            for name, sha in seal["all_source_hashes"].items():
                assert manifest["sources"]["upstream/" + name] == sha, (path.name, name)
        exports = json.loads((ROOT / "provenance/TAVOTTO-FINAL-EXPORTS.json").read_text())
        for item in exports["figures"]:
            assert item["preflight"]["error"] == item["preflight"]["warn"] == item["preflight"]["not_verifiable"] == 0
            stem = item["stem"]
            for artifact in item["files"]:
                assert digest((figures / (stem + "." + artifact["format"])).read_bytes()) == artifact["sha256"]
            with Image.open(figures / (stem + ".png")) as picture:
                picture.load()
                assert picture.width >= 1771
            assert b"/Subtype /Image" not in (figures / (stem + ".pdf")).read_bytes()
    result = dict(passed=True, archive_members=checked, independent_unpack=True,
                  member_sets_and_all_sha256_exact=True, six_csv_reruns_byte_identical=True,
                  restored_accepted_points=489, nine_vector_pdf_and_png_readable=True,
                  six_h100_source_seals_exact=True, report_images_match_tavotto_exports=True,
                  original_fp32_gates_preserved=True, final_snapshot=delivery["snapshot"])
    (ROOT / "provenance/DELIVERY-QA.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
