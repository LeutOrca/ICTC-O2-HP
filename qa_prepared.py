"""CPU syntax/source freeze audit; explicitly not an H100 numerical acceptance."""
import ast
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "o2-hp-angular-operators-20261121-r1"


def main():
    names = ["diagnose_radial.py", "models.py", "benchmark.py", "aggregate.py", "repair_campaign.py"]
    names += [str(p.relative_to(ROOT)) for p in (ROOT / "candidate").glob("*.py")]
    for name in names:
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
            if any(key in ast.unparse(node.test) for key in ('O2_DIAGNOSTIC', 'O2_BOUNDED_UVU')) or ast.unparse(node.test) == "repair":
                return None
            return self.generic_visit(node)
        def visit_Assign(self, node):
            if ast.unparse(node.targets[0]) == "repair":
                return None
            return self.generic_visit(node)
        def visit_Tuple(self, node):
            node.elts = [v for v in node.elts if not (isinstance(v, ast.Constant) and v.value == "eqx-o2-repair")]
            return self.generic_visit(node)
    assert ast.dump(before, include_attributes=False) == ast.dump(RemoveObserver().visit(after), include_attributes=False)
    data = dict(passed=True, baseline_commit="6f641cf736a97fa6835678c0eb123f5b2694e80d",
                scope="CPU source preparation only; H100 numerical/paired gates are not run",
                syntax_files=len(names), operator_source_files_unchanged=len(checked),
                inputs_weights_rng_equation_reference_ast_unchanged=True,
                sources=checked)
    before_gate = ast.parse((BASE / "benchmark.py").read_text()).body
    after_gate = ast.parse((ROOT / "benchmark.py").read_text()).body
    before_error = next(n for n in before_gate if isinstance(n, ast.FunctionDef) and n.name == "error")
    after_error = next(n for n in after_gate if isinstance(n, ast.FunctionDef) and n.name == "error")
    assert ast.dump(before_error) == ast.dump(after_error)
    data["original_fp32_error_ast_unchanged"] = True
    (ROOT / "provenance/BOUNDED-SOURCE-QA.json").write_text(json.dumps(data, indent=2))
    print(json.dumps({k:v for k,v in data.items() if k != "sources"}), flush=True)


if __name__ == "__main__":
    main()
