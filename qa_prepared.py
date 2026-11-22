"""CPU syntax/source freeze audit; explicitly not an H100 numerical acceptance."""
import ast
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "o2-hp-angular-operators-20261121-r1"


def main():
    for name in ("diagnose_radial.py", "models.py", "benchmark.py", "aggregate.py"):
        ast.parse((ROOT / name).read_text(), filename=name)
    checked = []
    for folder in ("official/src", "fork/src", "eqx/equivariantx"):
        for path in sorted((BASE / "upstream" / folder).rglob("*.py")):
            other = ROOT / path.relative_to(BASE)
            assert path.read_bytes() == other.read_bytes(), str(other)
            checked.append(str(path.relative_to(BASE)))
    before = ast.parse((BASE / "models.py").read_text())
    after = ast.parse((ROOT / "models.py").read_text())
    # Remove only the diagnostic object references, then prove the constructor
    # and all RNG/input/weight/reference expressions are unchanged.
    class RemoveObserver(ast.NodeTransformer):
        def visit_If(self, node):
            if 'O2_DIAGNOSTIC' in ast.unparse(node.test):
                return None
            return self.generic_visit(node)
    assert ast.dump(before, include_attributes=False) == ast.dump(RemoveObserver().visit(after), include_attributes=False)
    data = dict(passed=True, baseline_commit="6f641cf736a97fa6835678c0eb123f5b2694e80d",
                scope="CPU source preparation only; H100 numerical/paired gates are not run",
                syntax_files=4, operator_source_files_unchanged=len(checked),
                inputs_weights_rng_equation_reference_ast_unchanged=True,
                sources=checked)
    (ROOT / "provenance/PREPARED-QA.json").write_text(json.dumps(data, indent=2))
    print(json.dumps({k:v for k,v in data.items() if k != "sources"}), flush=True)


if __name__ == "__main__":
    main()
