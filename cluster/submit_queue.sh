#!/bin/bash
# Submit a job list one run at a time while staying under the cluster's per-user job limit
# (QOSMaxSubmitJobPerUserLimit: every array task counts as one submitted job, and so does a JupyterHub session).
#
# Run on the LOGIN node from the project root, in the background so it survives logout:
#   nohup bash cluster/submit_queue.sh cluster/jobs/smoke 2 > submit_smoke.log 2>&1 &
#   tail -f submit_smoke.log
# Arguments:  JOBDIR  [MAXJOBS=2]  [SLURM_SCRIPT=cluster/run_array_p100.slurm]  [extra sbatch args...]
#   MAXJOBS = how many of YOUR jobs (all kinds, pending + running) may exist at once = your QOS limit.
# Example with a longer time limit:
#   nohup bash cluster/submit_queue.sh cluster/jobs/rq1_pipelines 2 cluster/run_array_p100.slurm --time=24:00:00 > submit_rq1.log 2>&1 &
# Stop it with:  kill <PID printed at start>   (already submitted jobs keep running; scancel those separately)
# Safe to restart at any time: runs with a result.json, and tasks already queued/running, are not resubmitted.
set -uo pipefail

JOBDIR="${1:?usage: bash cluster/submit_queue.sh JOBDIR [MAXJOBS] [SLURM_SCRIPT] [sbatch args...]}"
MAXJOBS="${2:-2}"
SCRIPT="${3:-cluster/run_array_p100.slurm}"
shift $(( $# < 3 ? $# : 3 ))
EXTRA=("$@")
POLL="${POLL:-60}"

[[ -f "$JOBDIR/index.tsv" ]] || { echo "No $JOBDIR/index.tsv (run cluster/make_job_list.py first)"; exit 2; }
[[ -f "$SCRIPT" ]] || { echo "No $SCRIPT (run from the project root)"; exit 2; }
RESULTS_ROOT="results"
[[ -f "$JOBDIR/results_root" ]] && RESULTS_ROOT="$(tr -d '[:space:]' < "$JOBDIR/results_root")"
TAG="qq-$(basename "$JOBDIR")"          # job names are $TAG-<task>, to recognise this list's tasks in squeue
echo "PID $$ | $JOBDIR | max $MAXJOBS jobs | $SCRIPT ${EXTRA[*]} | job name $TAG | $(date)"

my_jobs()  { squeue -h -u "$USER" -o "%i" | wc -l; }
list_jobs() { squeue -h -u "$USER" -o "%j" | grep "^$TAG-[0-9]*$"; }
in_queue() { squeue -h -u "$USER" -o "%j" | grep -qx "$TAG-$1"; }

while IFS=$'\t' read -r IDX NAME SEED CFG <&3; do
  [[ -n "$IDX" ]] || continue
  if [[ -f "$RESULTS_ROOT/$NAME/seed$SEED/result.json" ]]; then echo "[$IDX] $NAME seed$SEED already done"; continue; fi
  if in_queue "$IDX"; then echo "[$IDX] $NAME seed$SEED already queued"; continue; fi
  while (( $(my_jobs) >= MAXJOBS )); do sleep "$POLL"; done
  until OUT=$(sbatch --array="$IDX" --job-name="$TAG-$IDX" "${EXTRA[@]}" "$SCRIPT" "$JOBDIR" 2>&1); do
    echo "[$IDX] sbatch refused: $OUT -- retrying in ${POLL}s"; sleep "$POLL"
  done
  echo "[$IDX] $NAME seed$SEED -> $OUT ($(date +%H:%M))"
  sleep 2
done 3< "$JOBDIR/index.tsv"

echo "All tasks of $JOBDIR submitted at $(date). Waiting for them to finish ..."
while list_jobs >/dev/null; do sleep "$POLL"; done
done_n=0; total=0
while IFS=$'\t' read -r IDX NAME SEED CFG; do
  [[ -n "$IDX" ]] || continue; total=$((total + 1))
  [[ -f "$RESULTS_ROOT/$NAME/seed$SEED/result.json" ]] && done_n=$((done_n + 1))
done < "$JOBDIR/index.tsv"
echo "Finished at $(date): $done_n of $total runs have result.json"
(( done_n == total )) || echo "Some runs failed: check sacct and the qft_*.err files, then rerun this command (finished runs are skipped)."
