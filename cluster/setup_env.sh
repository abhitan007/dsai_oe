#!/bin/bash
# One-time setup of the project Python environment ON THE CLUSTER (login node or JupyterHub CPU terminal;
# needs access to PyPI).  Run from the project root:   bash cluster/setup_env.sh
#
# Creates  ./.venv  inside the approved container with --system-site-packages, so the container's
# P100-compatible torch (and numpy) are reused.  A constraints file pins torch/numpy to the container's
# versions: pip will fail loudly rather than ever replacing them.
set -euo pipefail

CONTAINER="${CONTAINER:-/userhome/containers/jupyter.sif}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

[[ -r "$CONTAINER" ]] || { echo "Container not readable: $CONTAINER" >&2; exit 2; }
run() { apptainer exec --bind "$PROJECT_DIR:$PROJECT_DIR" --pwd "$PROJECT_DIR" "$CONTAINER" "$@"; }

echo "== container python / torch"
run python -c 'import sys, torch, numpy; print(sys.version); print("torch", torch.__version__, "cuda build", torch.version.cuda, "| numpy", numpy.__version__)'

if [[ ! -f .venv/pyvenv.cfg ]]; then     # bin/python is a symlink into the container: test the marker file
  echo "== creating .venv (with system site packages)"
  run python -m venv --system-site-packages .venv
fi

run .venv/bin/python -c 'import torch, numpy; print(f"torch=={torch.__version__}"); print(f"numpy=={numpy.__version__}")' > cluster/constraints-container.txt
echo "== constraints (kept from the container):"; cat cluster/constraints-container.txt

run .venv/bin/python -m pip install --upgrade pip
run .venv/bin/python -m pip install -r cluster/requirements-cluster.txt -c cluster/constraints-container.txt

echo "== verifying"
run .venv/bin/python -m pip check || echo "(pip check reported issues above; review them)"
run .venv/bin/python -c '
import torch, transformers, datasets, lm_eval, accelerate, scipy, yaml
print("torch", torch.__version__, "| transformers", transformers.__version__, "| datasets", datasets.__version__)
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface
print("imports OK")'
run .venv/bin/python -m pytest -q tests/ -p no:cacheprovider
echo "== setup finished: use $PROJECT_DIR/.venv/bin/python inside $CONTAINER"
