# How to run the experiments

This guide covers three things:

1. **Setup** and a quick check that everything works.
2. **Running on the GPU cluster.** This is how all results in `Report.md` were produced, and it is the
   section to follow if you use the same cluster.
3. **Running on any GPU machine without SLURM**, and analysing the results.

Run all commands from the repository root unless a step says otherwise.

---

## 1. Setup and verification

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126     # the wheel for your CUDA version
pip install -r requirements.txt
python -m pytest -q tests/                                               # 21 passed, CPU, under a minute
```

The first run downloads Qwen2.5-0.5B (about 1 GB), GSM8K, WikiText-2 and one C4 shard. Qwen2.5 is
public, so no Hugging Face login is needed. To keep the cache on a large disk, set `HF_HOME`.

On a GPU, check the model end to end before starting long runs:

```bash
python scripts/sanity_check.py --chunks 8 --presets fp16 w4a8 qoq_intattn
```

You should see:

* `[check 1] ... OK`: our attention matches PyTorch's SDPA attention.
* `[check 2] ... delta` close to 0: the rotation and smoothing transforms do not change the model.
* A short ladder in which `w4a8` is clearly worse than `fp16` and `qoq_intattn` recovers much of the gap.

---

> ## 2. Running on the GPU cluster (SLURM + Apptainer, NVIDIA Tesla P100)
>
> **This is the setup that produced every number in the report:**
> * **Cluster:** the CSE SLURM cluster, login node `172.16.112.202`, partition `gpu-P100`.
> * **GPU:** one NVIDIA Tesla P100-PCIE-16GB per job.
> * **Per job:** 8 CPU cores, 32 GB RAM, a 12-hour limit.
> * **Container:** the approved Apptainer image `/userhome/containers/jupyter.sif`.
> * **No internet on compute nodes:** models and datasets are copied to the cluster in advance.
> * **One job at a time:** our account was limited to this, so runs are submitted one after another
>   by `cluster/submit_queue.sh`.
>
> The P100 has no native bfloat16, so the `p100` profile runs in float32 with batch 8 × 2
> gradient-accumulation steps (the same effective batch of 16) and gradient checkpointing.

Commands are labelled by where they run: **[PC]** is your own computer (Linux or WSL), and
**[LOGIN]** is the cluster login node, reached with `ssh`. Training never runs on the login node;
SLURM starts it on a GPU node. First set your username once in each terminal:

```bash
export CLUSTER=your_username@172.16.112.202
```

### 2.1 [PC] Build the offline cache and pack the code

Compute nodes cannot download anything. So download the model, the datasets and the evaluation tasks
once into one folder, and verify that they load with the network switched off:

```bash
python3 -m venv ~/qft_prefetch_venv
~/qft_prefetch_venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
~/qft_prefetch_venv/bin/pip install -r cluster/requirements-cluster.txt
HF_HOME=$HOME/dsai_hf_cache ~/qft_prefetch_venv/bin/python cluster/prefetch_assets.py
# must end with: [assets] OFFLINE CHECK PASSED

tar --exclude='./.git' --exclude='./results*' --exclude='./cluster/jobs' --exclude='*.pdf' \
    --exclude='__pycache__' --exclude='./.venv' --exclude='./hf_cache' -czf ~/dsai_project_code.tar.gz .
tar -C "$HOME/dsai_hf_cache" -czf ~/dsai_hf_cache.tar.gz .      # about 1.2 GB
```

### 2.2 [PC] Copy to the cluster

```bash
scp ~/dsai_project_code.tar.gz ~/dsai_hf_cache.tar.gz $CLUSTER:~/
```

If the large copy is interrupted, `rsync -avP ~/dsai_hf_cache.tar.gz $CLUSTER:~/` resumes it. To
update one file later, copy just that file to the same place, for example
`scp qft/quant/attention.py $CLUSTER:~/dsai_project/qft/quant/`.

### 2.3 [LOGIN] Unpack and create the environment (once)

```bash
ssh $CLUSTER
mkdir -p ~/dsai_project/hf_cache
tar -xzf ~/dsai_project_code.tar.gz -C ~/dsai_project
tar -xzf ~/dsai_hf_cache.tar.gz     -C ~/dsai_project/hf_cache
cd ~/dsai_project
bash cluster/setup_env.sh          # shows "21 passed" and ends with "== setup finished"
PY="apptainer exec /userhome/containers/jupyter.sif $HOME/dsai_project/.venv/bin/python"
```

`setup_env.sh` creates `.venv` inside the container on top of the container's PyTorch. It pins
PyTorch and NumPy, so the GPU build is never replaced. It needs internet access on the login node. The
`PY=...` shortcut must be set again in every new SSH session.

### 2.4 [LOGIN] Check the GPU and run the smoke test

```bash
sbatch cluster/gpu_check_p100.slurm                 # GPU visible, unit tests, sanity check (~5 min)
sacct -u $USER -S today --format=JobID,JobName%20,State,ExitCode,Elapsed   # wait for COMPLETED 0:0
cat qft_check_<JOBID>.out

$PY cluster/make_job_list.py --suite configs/suites/smoke.yaml --profile p100
nohup bash cluster/submit_queue.sh cluster/jobs/smoke 1 > submit_smoke.log 2>&1 &
tail -f submit_smoke.log      # ends with "6 of 6 runs have result.json" (Ctrl+C stops watching only)
grep -h "s/it\|done in" results/smoke/*/seed0/log.txt               # speed per training step
rm -rf results/smoke cluster/jobs/smoke
```

On the P100, one training step took about 3 s for the FP model and 4 s for the quantized one. A full
run of 1,000 steps plus a full evaluation therefore takes 1–2 hours, well within the 12-hour limit.

### 2.5 [LOGIN] Run the experiments

`make_job_list.py` turns a suite into one config per run, and `submit_queue.sh` submits them one at a
time. The second argument of `submit_queue.sh` is the number of jobs your account may have at once;
for us that was 1. The submitter keeps running after you log out.

**The core study used in the report** (62 runs, seed 0, no zero-shot benchmarks, about 72 GPU-hours):

```bash
cd ~/dsai_project
for s in rq0_ptq_ladder rq1_pipelines rq3_components; do
  $PY cluster/make_job_list.py --suite configs/suites/$s.yaml --profile p100 --seeds 0 --set "eval.lm_eval_tasks=[]"
done
nohup bash -c '
for s in rq0_ptq_ladder rq1_pipelines rq3_components; do bash cluster/submit_queue.sh cluster/jobs/$s 1; done
' > submit_core.log 2>&1 &
```

**The remaining suites** (not run for the report) work the same way:

| suite | runs (seed 0) | studies |
|---|---|---|
| `rq2_modules` | 40 | which modules and layers to finetune |
| `rq3_tricks` | 16 | whether each QoQ trick still matters after finetuning |
| `rq4_intattn` | 29 | IndexSoftmax (b, c) sensitivity |
| `rq5_rank` | 12 | LoRA rank |

For RQ2, first run the importance proxies:

* `sbatch cluster/importance_p100.slurm`
* `QUANT=fp16 sbatch cluster/importance_p100.slurm`

Submit these only when no other job of yours is queued, because each counts towards your job limit.
Drop `--seeds 0` to get the three seeds the suites define. Drop `--set "eval.lm_eval_tasks=[]"` to add
the zero-shot benchmarks.

**Rules:**

* Submit from `~/dsai_project`.
* Run one submitter at a time.
* Never regenerate a job list while its runs are queued.
* Rerunning a submitter is always safe: finished runs are skipped, and queued ones are not submitted twice.

### 2.6 [LOGIN] Monitor and fix

```bash
tail -5 submit_core.log; pgrep -af submit_queue        # nothing printed = submitter finished or stopped
squeue -u $USER
find results -name result.json | wc -l                 # finished runs
sacct -u $USER -S now-1days --format=JobID,JobName%22,State,ExitCode,Elapsed | grep -v "batch\|extern"
cat qft_<JOBID>_<TASK>.err                             # console of one run; also results/<suite>/<run>/seed0/log.txt
```

| problem | what to do |
|---|---|
| `QOSMaxSubmitJobPerUserLimit` from `sbatch` | You are at your job limit: use `submit_queue.sh` with the right limit, and close JupyterHub sessions (they count). |
| `TIMEOUT` in `sacct` | Rerun the submitter with a longer limit: `... submit_queue.sh cluster/jobs/<suite> 1 cluster/run_array_p100.slurm --time=24:00:00` |
| `CUDA out of memory` | Regenerate the job list with `--set train.batch_size=4 train.grad_accum=4` and rerun the submitter. |
| `Project venv missing` | `bash cluster/setup_env.sh` |
| `HF cache missing` / `LocalEntryNotFoundError` | The cache is not in `~/dsai_project/hf_cache`: redo the `tar -xzf ~/dsai_hf_cache.tar.gz` line. |
| submitter stopped before all runs finished | Start it again; finished runs are skipped. |

To debug interactively on a GPU node:
`srun --partition=gpu-P100 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash`.

### 2.7 Analyse and bring the results home

```bash
# [LOGIN]
$PY scripts/analyze.py --results results
# [PC]
rsync -avP $CLUSTER:~/dsai_project/results/ ./results_cluster/     # or: scp -r $CLUSTER:~/dsai_project/results ./results_cluster
```

`rsync` copies only new files, so you can repeat it while runs are still going.

The `cluster/` folder contains everything used here:

| file | what it does |
|---|---|
| `run_array_p100.slurm` | one experiment run per job; skips finished runs and copies results back even after a failure or timeout |
| `submit_queue.sh` | feeds a job list to SLURM while staying under the per-user job limit |
| `make_job_list.py` | expands a suite into per-run configs, with the `p100` or `h100` profile |
| `gpu_check_p100.slurm` | one-off GPU and environment check |
| `importance_p100.slurm` | importance proxies for RQ2 |
| `setup_env.sh` | creates the environment on the cluster |
| `prefetch_assets.py` | builds and verifies the offline cache |
| `requirements-cluster.txt` | pinned package versions for the cluster |
| `run_array_h100.slurm` | the same job for an H100 partition |

---

## 3. Running on a GPU machine without SLURM

`scripts/launch.py` runs a whole suite over the GPUs of one machine, one run per GPU at a time. It can
be resumed after an interruption.

```bash
python scripts/launch.py --suite configs/suites/rq0_ptq_ladder.yaml --gpus 0,1 --seeds 0 --set "eval.lm_eval_tasks=[]"
python scripts/launch.py --suite configs/suites/rq1_pipelines.yaml  --gpus 0,1 --seeds 0 --set "eval.lm_eval_tasks=[]"
python scripts/launch.py --suite configs/suites/rq3_components.yaml --gpus 0,1 --seeds 0 --set "eval.lm_eval_tasks=[]"
```

* `--dry-run` lists the runs without starting them.
* `--only <text>` keeps only runs whose name contains that text.
* On GPUs with bfloat16 support (A100, H100, RTX 30xx and newer) the default `dtype: bfloat16` is used.
  On older GPUs add `--set dtype=float32`.

A single experiment can be run directly:

```bash
python scripts/run_experiment.py --config configs/base.yaml \
    --set name=my_test pipeline=ptq_then_ft quant=qoq_intattn seed=0 eval.lm_eval_tasks=[]
```

`--set` takes dotted keys with YAML values, for example `lora.r=8`, `"lora.targets=[q_proj,v_proj]"` or
`"train_runtime={attn_mode: fp}"`. The quantization presets are listed in `qft/config.py`
(`QUANT_PRESETS`), and any preset field can be overridden: `"quant={preset: qoq_intattn, kv_bits: 8}"`.

A language-modelling variant (finetuning on C4 instead of GSM8K) is available with
`--base configs/base_c4.yaml`. Its results go to `results_c4/`.

## 4. Analysing results

```bash
python scripts/analyze.py --results results_cluster
```

This writes three things into the results folder:

* `REPORT.md`: every table, with paired McNemar significance tests, the train/deploy mismatch table and
  the Shapley decomposition.
* `summary.csv`: one row per run.
* `figures/`: the plots.

Each run folder contains `result.json` (configuration, metrics, per-question GSM8K correctness, the
quantization details and training curves) and `log.txt`.

## 5. Practical notes

* **Evaluation is the slow part.** With simulated IntAttention, a full GSM8K evaluation takes up to
  about 1.5 hours on a P100, because the integer attention is emulated densely in float32. For quick
  experiments use `eval.gsm8k_limit=300 eval.ppl_max_chunks=40`, and never mix limited and full
  evaluations in one table.
* **Rerunning.** Delete the run's `seed<k>` folder, or pass `--force` to `run_experiment.py`.
* **`KeyError: 'qft'` from transformers** means the wrong version is installed; install the pinned
  `transformers==4.56.2`.
