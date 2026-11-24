#!/bin/bash
#SBATCH --job-name=qft-check
#SBATCH --partition=gpu-P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=qft_check_%j.out
#SBATCH --error=qft_check_%j.err
#
# One-off verification on a P100 before launching any suite.  Submit from the project root:
#   sbatch cluster/gpu_check_p100.slurm
# Checks: GPU visible in the container, unit tests, custom attention == SDPA on the real model,
# transforms preserve perplexity, and a 3-point PTQ ladder (fp16 / w4a8 / qoq_intattn).

set -euo pipefail
# Required by the CSE cluster guide. Do not remove this line.
cd "$TMPDIR" || exit 1

PROJECT_DIR="${SLURM_SUBMIT_DIR:?Submit this script from the project root}"
CONTAINER="${CONTAINER:-/userhome/containers/jupyter.sif}"
PYTHON="$PROJECT_DIR/.venv/bin/python"
HF_CACHE_SRC="${HF_CACHE_SRC:-$PROJECT_DIR/hf_cache}"

[[ -r "$CONTAINER" ]] || { echo "Container not readable: $CONTAINER" >&2; exit 2; }
# .venv/bin/python is a symlink into the container, so test the venv marker file on the host, not the binary
[[ -f "$PROJECT_DIR/.venv/pyvenv.cfg" ]] || { echo "Project venv missing: $PROJECT_DIR/.venv (run: bash cluster/setup_env.sh)" >&2; exit 2; }
[[ -d "$HF_CACHE_SRC/hub" ]] || { echo "HF cache missing: $HF_CACHE_SRC/hub" >&2; exit 2; }

mkdir -p "$TMPDIR/code"
cp -r "$PROJECT_DIR/qft" "$PROJECT_DIR/scripts" "$PROJECT_DIR/configs" "$PROJECT_DIR/tests" "$TMPDIR/code/"
cp -r "$HF_CACHE_SRC" "$TMPDIR/hf_cache"

export TOKENIZERS_PARALLELISM=false
export HF_HOME="$TMPDIR/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export APPTAINERENV_HF_HOME="$HF_HOME" APPTAINERENV_HF_HUB_OFFLINE=1 \
       APPTAINERENV_TRANSFORMERS_OFFLINE=1 APPTAINERENV_HF_DATASETS_OFFLINE=1
APPT=(apptainer exec --nv --bind "$TMPDIR:$TMPDIR" --bind "$PROJECT_DIR:$PROJECT_DIR" --pwd "$TMPDIR/code" "$CONTAINER")

echo "== GPU / packages ($(hostname))"
"${APPT[@]}" "$PYTHON" -c 'import torch, transformers, datasets, lm_eval; assert torch.cuda.is_available(), "no GPU visible"; p = torch.cuda.get_device_properties(0); print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| transformers", transformers.__version__, "| datasets", datasets.__version__); print("GPU", p.name, "| capability", torch.cuda.get_device_capability(0), "| VRAM GiB", round(p.total_memory / 2**30, 1))'

echo "== unit tests"
"${APPT[@]}" "$PYTHON" -m pytest -q tests/ -p no:cacheprovider

echo "== sanity check on Qwen2.5-0.5B (GPU)"
"${APPT[@]}" "$PYTHON" scripts/sanity_check.py --device cuda --dtype auto --chunks 8 \
  --presets fp16 w4a8 qoq_intattn

echo "== all checks finished at $(date)"
