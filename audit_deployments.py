"""Check actual remote harness bytes against independent hashes and Git receipts."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parent
PREFIX="o2-hp-angular-operators-20261121-r1-"


def main():
    expected={}
    for line in (ROOT/"provenance/DEPLOYED-HARNESS.sha256").read_text().splitlines():
        value,name=line.split(maxsplit=1)
        expected[name.strip()]=value
    checked=[]
    mismatches=[]
    with tarfile.open(ROOT/"provenance/DEPLOYED-HARNESS.tgz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            folder,name=member.name.split("/",1)
            commit=folder.removeprefix(PREFIX)
            key=commit+"/"+name
            actual=tar.extractfile(member).read()
            actual_sha=hashlib.sha256(actual).hexdigest()
            assert expected[key]==actual_sha,key
            result=subprocess.run(["git","-C",str(ROOT),"show",f"{commit}:{name}"],capture_output=True,check=False)
            same=result.returncode==0 and result.stdout==actual
            checked.append(dict(cohort=commit,file=name,sha256=actual_sha,git_exact=same))
            if not same:
                mismatches.append(dict(cohort=commit,file=name,sha256=actual_sha,
                    reason="not-in-git-tree" if result.returncode else "different-from-git-tree"))
    assert {r["cohort"]+"/"+r["file"] for r in checked}==set(expected)
    formal=[r for r in mismatches if r["cohort"]!="a674d1c"]
    output=dict(passed=not formal,archive_and_independent_hashes_match=True,checked_files=len(checked),
        git_mismatches=mismatches,formal_mismatches=formal,files=checked)
    (ROOT/"provenance/DEPLOYMENT-AUDIT.json").write_text(json.dumps(output,indent=2))
    print(json.dumps({k:v for k,v in output.items() if k!="files"}),flush=True)
    if formal:
        raise RuntimeError("Formal deployed-source differences require review")


if __name__=="__main__":
    main()
