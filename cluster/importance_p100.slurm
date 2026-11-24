#!/bin/bash
#SBATCH --job-name=qft-importance
#SBATCH --partition=gpu-P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=qft_importance_%j.out
#SBATCH --error=qft_importance_%j.err
#
# Training-free importance proxies for RQ2 (scripts/compute_importance.py). Submit from the project root:
#   sbatch cluster/importance_p100.slurm                    # quantized base (qoq_intattn)
#   QUANT=fp16 sbatch cluster/importance_p100.slurm         # full-precision base (only grad_norm meaningful)
# Output: <project>/results/importance/gsm8k_<QUANT>_seed0.json  (read automatically by scripts/analyze.py)

set -euo pipefail
# Required by the CSE cluster guide. Do not remove this line.
cd "$TMPDIR" || exit 1

PROJECT_DIR="${SLURM_SUBMIT_DIR:?Submit this script from the project root}"
CONTAINER="${CONTAINER:-/userhome/containers/jupyter.sif}"
PYTHON="$PROJECT_DIR/.venv/bin/python"
HF_CACHE_SRC="${HF_CACHE_SRC:-$PROJECT_DIR/hf_cache}"
QUANT="${QUANT:-qoq_intattn}"

[[ -r "$CONTAINER" ]] || { echo "Container not readable: $CONTAINER" >&2; exit 2; }
# .venv/bin/python is a symlink into the container, so test the venv marker file on the host, not the binary
[[ -f "$PROJECT_DIR/.venv/pyvenv.cfg" ]] || { echo "Project venv missing: $PROJECT_DIR/.venv (run: bash cluster/setup_env.sh)" >&2; exit 2; }
[[ -d "$HF_CACHE_SRC/hub" ]] || { echo "HF cache missing: $HF_CACHE_SRC/hub" >&2; exit 2; }

mkdir -p "$TMPDIR/code" "$TMPDIR/results/importance" "$PROJECT_DIR/results/importance"
cp -r "$PROJECT_DIR/qft" "$PROJECT_DIR/scripts" "$PROJECT_DIR/configs" "$TMPDIR/code/"
cp -r "$HF_CACHE_SRC" "$TMPDIR/hf_cache"
OUT="$TMPDIR/results/importance/gsm8k_${QUANT}_seed0.json"
trap 'cp -f "$OUT" "$PROJECT_DIR/results/importance/" 2>/dev/null || true' EXIT TERM

export TOKENIZERS_PARALLELISM=false
export HF_HOME="$TMPDIR/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export APPTAINERENV_HF_HOME="$HF_HOME" APPTAINERENV_HF_HUB_OFFLINE=1 \
       APPTAINERENV_TRANSFORMERS_OFFLINE=1 APPTAINERENV_HF_DATASETS_OFFLINE=1

apptainer exec --nv --bind "$TMPDIR:$TMPDIR" --bind "$PROJECT_DIR:$PROJECT_DIR" --pwd "$TMPDIR/code" "$CONTAINER" \
  "$PYTHON" scripts/compute_importance.py --config configs/base.yaml \
    --set "quant=$QUANT" device=cuda --tag "$QUANT" --batch_size 8 --n_batches 32 --out "$OUT"

echo "Wrote $PROJECT_DIR/results/importance/$(basename "$OUT")"
