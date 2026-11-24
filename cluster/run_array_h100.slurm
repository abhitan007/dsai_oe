#!/bin/bash
#SBATCH --job-name=qft-h100
#SBATCH --partition=gpu-H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=qft_%A_%a.out
#SBATCH --error=qft_%A_%a.err
#
# One array task = one (run, seed) of an experiment suite.  Submit from the project root, either directly
#   python cluster/make_job_list.py --suite configs/suites/rq1_pipelines.yaml --profile h100
#   sbatch --array=1-<N>%4 cluster/run_array_h100.slurm cluster/jobs/rq1_pipelines
# or, when the account may only have a few jobs at once, through cluster/submit_queue.sh.
#
# Results go to <project>/results/<suite>/<run>/seed<k>/ (result.json, log.txt), or to results_c4/ for
# suites built with --base configs/base_c4.yaml. They are copied back even if the run fails or hits the
# time limit. Tasks whose result.json already exists are skipped, so resubmitting is always safe.

set -euo pipefail
# Required by the CSE cluster guide. Do not remove this line.
cd "$TMPDIR" || exit 1

PROJECT_DIR="${SLURM_SUBMIT_DIR:?Submit this script from the project root}"
JOBDIR="${1:?Pass the job directory created by cluster/make_job_list.py, e.g. cluster/jobs/rq1_pipelines}"
[[ "$JOBDIR" = /* ]] || JOBDIR="$PROJECT_DIR/$JOBDIR"
CONTAINER="${CONTAINER:-/userhome/containers/jupyter.sif}"
PYTHON="$PROJECT_DIR/.venv/bin/python"
HF_CACHE_SRC="${HF_CACHE_SRC:-$PROJECT_DIR/hf_cache}"

LINE="$(awk -F'\t' -v i="${SLURM_ARRAY_TASK_ID:?Submit with --array=1-N}" '$1 == i' "$JOBDIR/index.tsv")"
[[ -n "$LINE" ]] || { echo "No task $SLURM_ARRAY_TASK_ID in $JOBDIR/index.tsv" >&2; exit 2; }
IFS=$'\t' read -r IDX RUN_NAME SEED CFG_FILE <<< "$LINE"
RESULTS_ROOT="results"                       # make_job_list.py writes e.g. results_c4 for configs/base_c4.yaml
[[ -f "$JOBDIR/results_root" ]] && RESULTS_ROOT="$(tr -d '[:space:]' < "$JOBDIR/results_root")"
PERSIST="$PROJECT_DIR/$RESULTS_ROOT/$RUN_NAME/seed$SEED"
OUT="$TMPDIR/results/$RUN_NAME/seed$SEED"

echo "Task $SLURM_ARRAY_TASK_ID of array $SLURM_ARRAY_JOB_ID (job $SLURM_JOB_ID) on $(hostname): $RUN_NAME seed=$SEED"
echo "Config: $JOBDIR/$CFG_FILE | Persistent results: $PERSIST"

if [[ -f "$PERSIST/result.json" ]]; then
  echo "Already finished ($PERSIST/result.json) -- skipping"; exit 0
fi
[[ -r "$CONTAINER" ]] || { echo "Container not readable: $CONTAINER" >&2; exit 2; }
# .venv/bin/python is a symlink into the container, so test the venv marker file on the host, not the binary
[[ -f "$PROJECT_DIR/.venv/pyvenv.cfg" ]] || { echo "Project venv missing: $PROJECT_DIR/.venv (run: bash cluster/setup_env.sh)" >&2; exit 2; }
[[ -d "$HF_CACHE_SRC/hub" ]] || { echo "HF cache missing: $HF_CACHE_SRC/hub (see INSTRUCTIONS.md, section 2.1)" >&2; exit 2; }
[[ -f "$JOBDIR/$CFG_FILE" ]] || { echo "Missing config $JOBDIR/$CFG_FILE" >&2; exit 2; }

# Code, config and HF cache go to node-local scratch (fast reads, no write contention between tasks).
mkdir -p "$TMPDIR/code" "$TMPDIR/results"
cp -r "$PROJECT_DIR/qft" "$PROJECT_DIR/scripts" "$PROJECT_DIR/configs" "$TMPDIR/code/"
cp "$JOBDIR/$CFG_FILE" "$TMPDIR/code/run_config.yaml"
cp -r "$HF_CACHE_SRC" "$TMPDIR/hf_cache"

copy_back() {
  if [[ -d "$OUT" ]]; then mkdir -p "$PERSIST" && cp -r "$OUT/." "$PERSIST/" || true; fi
}
trap copy_back EXIT TERM

export TOKENIZERS_PARALLELISM=false
export HF_HOME="$TMPDIR/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
# Make sure the container cannot override these with its own defaults.
export APPTAINERENV_HF_HOME="$HF_HOME" APPTAINERENV_HF_HUB_OFFLINE=1 \
       APPTAINERENV_TRANSFORMERS_OFFLINE=1 APPTAINERENV_HF_DATASETS_OFFLINE=1

APPT=(apptainer exec --nv --bind "$TMPDIR:$TMPDIR" --bind "$PROJECT_DIR:$PROJECT_DIR" --pwd "$TMPDIR" "$CONTAINER")

"${APPT[@]}" "$PYTHON" -c 'import torch, transformers; assert torch.cuda.is_available(), "no GPU visible"; print("torch", torch.__version__, "| transformers", transformers.__version__, "| GPU", torch.cuda.get_device_name(0))'

"${APPT[@]}" "$PYTHON" "$TMPDIR/code/scripts/run_experiment.py" \
  --config "$TMPDIR/code/run_config.yaml" \
  --set "output_dir=$TMPDIR/results"

echo "Finished $RUN_NAME seed=$SEED at $(date)"
