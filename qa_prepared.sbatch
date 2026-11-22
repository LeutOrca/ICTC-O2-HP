#!/bin/bash
#SBATCH -p CPU-AMD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:04:00
#SBATCH --job-name=o2-full-prepared-qa
#SBATCH --output=logs/%x-%j.log
source /etc/profile
set -euo pipefail
py=/public/home/sps-xia/bjxiao/miniconda3/envs/o2/bin/python
"$py" qa_prepared.py
"$py" -c 'import json; from pathlib import Path; from importlib.metadata import distributions; root=Path.cwd(); pyenv=Path("/public/home/sps-xia/bjxiao/.local/share/uv/tools/tavotto/lib/python3.11/site-packages"); v=next(x.version for x in distributions(path=[str(pyenv)]) if x.metadata["Name"]=="tavotto"); assert v=="0.21.3"; p=Path("/public/home/sps-xia/bjxiao/.codex/plugins/cache/tavotto/tavotto/0.21.3/.codex-plugin/plugin.json"); d=json.loads(p.read_text()); assert d["version"]==v; r=dict(plugin_version=d["version"],installed_engine_version=v,requires_mcp_reconnect=True); (root/"provenance/TAVOTTO-UPDATE.json").write_text(json.dumps(r,indent=2)); print(json.dumps(r))'
