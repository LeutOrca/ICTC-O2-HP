"""Inspect frozen native tensor-product expressions on a Slurm CPU node."""
import os
from pathlib import Path
import sys

sys.path.insert(0, os.environ["HPX_MACE_ENTRY"])
import paired_runner as bench

bench.configure()
model = bench.model_for("native", "cpu")
lines = []
for index, block in enumerate(model.interactions):
    lines.append(f"INTERACTION {index} {type(block).__name__}")
    lines.append(str(block.conv_tp))
    lines.append(block.conv_tp._compiled_main_left_right.code)
    lines.append(str(block.skip_tp))
    lines.append(block.skip_tp._compiled_main_left_right.code)
Path(sys.argv[1]).write_text("\n".join(lines))
print("\n".join(lines[:3]), flush=True)
