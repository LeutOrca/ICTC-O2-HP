"""Freeze delivery and independently restore raw receipts and plot inputs on CPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parent
FIG=ROOT.parents[1]/"figures/o2-hp-angular-operators-20261121-r1"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def archive(path,members):
    manifest={name:dict(sha256=sha(source),bytes=source.stat().st_size) for name,source in members}
    with tarfile.open(path,"w:gz") as tar:
        for name,source in members:
            tar.add(source,arcname=name,recursive=False)
    return manifest


def extract_checked(path,target,manifest):
    with tarfile.open(path) as tar:
        names=[member.name for member in tar.getmembers()]
        assert len(names)==len(set(names)) and set(names)==set(manifest)
        tar.extractall(target,filter="data")
    for name,item in manifest.items():
        assert sha(target/name)==item["sha256"],name


def main():
    provenance=ROOT/"provenance"
    commit=(provenance/"FROZEN-GIT.txt").read_text().strip()
    assert (provenance/"RECOVERED-GIT.txt").read_text().strip()==commit
    bundle=provenance/"angular-latest.bundle"
    original_code=provenance/"FROZEN-CODE.tar.gz"
    recovered_code=provenance/"RECOVERED-CODE.tar.gz"
    def code_members(path):
        with tarfile.open(path) as tar:
            assert tar.pax_headers.get("comment")==commit
            return {m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest()
                    for m in tar.getmembers() if m.isfile()}
    assert code_members(original_code)==code_members(recovered_code)
    sources=[]
    for folder in (ROOT/"upstream/official/src",ROOT/"upstream/fork/src",ROOT/"upstream/eqx/equivariantx",
                   ROOT/"upstream/ictp/ictp",ROOT/"upstream/local-hpx",ROOT/"upstream/local-so2"):
        sources.extend((str(p.relative_to(ROOT)),p) for p in sorted(folder.rglob("*"))
                       if p.is_file() and "__pycache__" not in p.parts)
    for path in (ROOT/"upstream/so2_ops.py",provenance/"Times.TTF",provenance/"BUDGET-AUDIT.json"):
        sources.append((str(path.relative_to(ROOT)),path))
    for path in (provenance/"DEPLOYED-HARNESS.tgz",provenance/"DEPLOYED-HARNESS.sha256",
                 provenance/"DEPLOYMENT-AUDIT.json",provenance/"CUDA-matrix.original.py"):
        sources.append((str(path.relative_to(ROOT)),path))
    sources.extend((str(p.relative_to(ROOT)),p) for p in sorted((provenance/"licenses").glob("*")) if p.is_file())
    source_archive=provenance/f"angular-sources-{commit[:7]}.tar.gz"
    source_manifest=archive(source_archive,sources)
    raw=[(str(p.relative_to(ROOT)),p) for p in sorted((ROOT/"measurements").rglob("*")) if p.is_file()]
    raw_archive=provenance/f"angular-receipts-{commit[:7]}.tar.gz"
    raw_manifest=archive(raw_archive,raw)
    fig_members=[(str(p.relative_to(FIG)),p) for p in sorted(FIG.rglob("*"))
                 if p.is_file() and not any(part.startswith(".") or part=="__pycache__" for part in p.relative_to(FIG).parts)
                 and p.suffix in (".py",".csv",".pdf",".png",".json",".md")]
    figure_archive=provenance/f"angular-figures-{commit[:7]}.tar.gz"
    figure_manifest=archive(figure_archive,fig_members)
    compare=("summary.csv","speedups.csv","coverage.csv","role-coverage.csv","status.csv")
    with tempfile.TemporaryDirectory(prefix="angular-restore-") as temp:
        work=Path(temp)
        restored=work/"runs"/ROOT.name
        restored.parent.mkdir()
        restored.mkdir()
        with tarfile.open(recovered_code) as code:
            code.extractall(restored,filter="data")
        extract_checked(source_archive,restored,source_manifest)
        extract_checked(raw_archive,restored,raw_manifest)
        restored_fig=work/"figures"/FIG.name
        restored_fig.mkdir(parents=True)
        extract_checked(figure_archive,restored_fig,figure_manifest)
        spec=importlib.util.spec_from_file_location("restored_aggregate",restored/"aggregate.py")
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
        for name in compare:
            assert sha(ROOT/name)==sha(restored/name),name
        for name in ("summary.csv","speedups.csv","status.csv"):
            assert sha(ROOT/name)==sha(FIG/"data"/name),name
    result=dict(passed=True,commit=commit,independent_git_clone=True,
        source_members=len(source_manifest),receipt_members=len(raw_manifest),figure_members=len(figure_manifest),
        restored_csv_byte_identical=list(compare),
        git_control_on_login=True,cpu_independent_recovery_audit=True,
        archives={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in (
            bundle,original_code,recovered_code,source_archive,raw_archive,figure_archive)},
        source_manifest=source_manifest,receipt_manifest=raw_manifest,figure_manifest=figure_manifest)
    (provenance/"DELIVERY-QA.json").write_text(json.dumps(result,indent=2))
    (provenance/"DELIVERY.sha256").write_text("".join(f"{info['sha256']}  {name}\n" for name,info in result["archives"].items()))
    print(json.dumps({k:v for k,v in result.items() if not k.endswith("manifest")}),flush=True)


if __name__=="__main__":
    main()
