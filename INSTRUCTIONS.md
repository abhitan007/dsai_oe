# How to run the experiments

All commands are run from the repository root. Results go to `results/<suite>/<run>/seed<k>/`
(`config.yaml`, `log.txt`, `stdout.txt`, `result.json`). **Every launch is resumable**: finished runs
(those with a `result.json`) are skipped, so you can re-run the same command after a crash or pre-emption.

---

## 0. Setup (≈10 min)

```bash
python -m venv .venv && source .venv/bin/activate        # or conda create -n qft python=3.11
pip install torch --index-url https://download.pytorch.org/whl/cu121   # pick the wheel for your CUDA
pip install -r requirements.txt
huggingface-cli login            # not needed for Qwen2.5 (public), needed if you switch to Llama
```

Optional: put the HF cache on a big disk: `export HF_HOME=/path/to/big/disk/hf`.
The first run downloads Qwen2.5-0.5B (~1 GB), GSM8K, WikiText-2 and one C4 shard (~320 MB train
+ ~40 MB validation). To pre-download everything on a login node:

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B')
from qft.data import _c4, _wikitext, _gsm8k
_c4('train'); _c4('validation'); _wikitext('train'); _wikitext('test'); _gsm8k('train'); _gsm8k('test')"
python -c "import lm_eval; from lm_eval.tasks import TaskManager; print('lm_eval ok')"
```

## 1. Verify the installation (≈5-15 min) — do not skip

```bash
python -m pytest -q tests/                      # 20 unit tests, CPU, < 1 min
python scripts/sanity_check.py                  # real model on GPU: equivalence checks + PTQ ladder
```

What to look for in `sanity_check.py` output:
* `[check 1] max |logit diff| … OK` — our attention equals HF's SDPA attention (< 1e-3).
* `[check 2] … delta` ≈ 0 (|Δ| < 0.05 PPL in bf16) — rotation/smoothing/SmoothAttention are exact.
* `[check 3]` ladder: `w8a8` ≈ `fp16`; plain `w4a8` clearly worse; `w4a8_rot*`/`qoq` recover much of
  the gap; `qoq_intattn` close to `qoq`. If `w4a8_rot` is *worse* than `w4a8`, something is broken —
  stop and investigate before launching sweeps.

Then an end-to-end smoke test of every pipeline (~15-25 min on one GPU):

```bash
python scripts/launch.py --suite configs/suites/smoke.yaml --gpus 0
cat results/smoke/*/seed0/log.txt | grep -E "metrics|Error"
rm -rf results/smoke                            # smoke numbers are meaningless (20 steps)
```

## 2. Running a single experiment

```bash
python scripts/run_experiment.py --config configs/base.yaml \
    --set name=my_test pipeline=ptq_then_ft quant=qoq_intattn seed=0
```

`--set` accepts dotted keys with YAML values, e.g.
`"lora.targets=[q_proj,v_proj]" lora.r=8 quant.a_bits=16 "train_runtime={attn_mode: fp}" eval.lm_eval_tasks=[]`.
Quantization presets are listed in `qft/config.py` (`QUANT_PRESETS`); any field can be overridden:
`"quant={preset: qoq_intattn, int_attn_c: 8.0}"`.

## 3. Running the study

`--gpus` takes a comma-separated list; one run per GPU at a time (`--jobs_per_gpu 2` packs two
0.5B runs on an 80 GB GPU). Use `--dry-run` to list the runs, `--seeds 0` to do a first pass with a
single seed, `--only <substring>` to filter runs.

Recommended order (each tier is useful on its own):

### Tier 1 — core results (≈35-50 GPU-h)
```bash
python scripts/launch.py --suite configs/suites/rq0_ptq_ladder.yaml --gpus 0,1,2,3   # 17 eval-only runs
python scripts/launch.py --suite configs/suites/rq1_pipelines.yaml  --gpus 0,1,2,3   # 35 runs (3 seeds)
python scripts/launch.py --suite configs/suites/rq3_components.yaml --gpus 0,1,2,3   # 16 PTQ + 48 FT
python scripts/analyze.py                                                             # -> results/REPORT.md
```

### Tier 2 — module importance and PTQ tricks (≈40-60 GPU-h)
```bash
# importance proxies (one GPU, a few minutes each)
python scripts/compute_importance.py --config configs/base.yaml --set quant=qoq_intattn --tag qoq_intattn
python scripts/compute_importance.py --config configs/base.yaml --set quant=fp16 --tag fp16   # gradient-only proxy for FP

python scripts/launch.py --suite configs/suites/rq2_modules.yaml --gpus 0,1,2,3 --seeds 0   # first pass: 40 runs
python scripts/launch.py --suite configs/suites/rq2_modules.yaml --gpus 0,1,2,3             # add seed 1
python scripts/launch.py --suite configs/suites/rq3_tricks.yaml  --gpus 0,1,2,3
```
(For the `fp16` proxy file the quantization error ΔW is zero, so only `grad_norm` is meaningful; the
analysis correlates every proxy file it finds in `results/importance/`.)

### Tier 3 — IntAttention and capacity (≈25-35 GPU-h)
```bash
python scripts/launch.py --suite configs/suites/rq4_intattn.yaml --gpus 0,1,2,3   # 24 cheap PTQ + 15 FT
python scripts/launch.py --suite configs/suites/rq5_rank.yaml    --gpus 0,1,2,3
python scripts/analyze.py
```

### Optional — second task (language-modelling recovery on C4)
Re-runs any suite with C4 finetuning and C4/WikiText perplexity as the main metrics, into a separate
results folder:
```bash
python scripts/launch.py --suite configs/suites/rq1_pipelines.yaml --base configs/base_c4.yaml --gpus 0,1,2,3
python scripts/analyze.py --results results_c4
```
(`analyze.py` ranks by GSM8K where available; for the C4 variant read the PPL columns.)

### Optional — larger model
`--set model=Qwen/Qwen2.5-1.5B` (hidden 1536 = 12·128 → Paley Hadamard is available). Use a different
`output_dir` (`--set output_dir=results_1p5b`) and consider `train.grad_checkpointing=true`.

## 4. Reading the results

`python scripts/analyze.py` writes
* `results/summary.csv` — one row per (run, seed), for your own plots/statistics;
* `results/REPORT.md` — all tables, importance orders, Shapley decomposition, Kendall/Spearman tests;
* `results/figures/*.png` — pipeline bar chart, Shapley bars, IntAttention (b, c) heat-map, rank curves.

Suggested structure of the final course report (map each section to a suite):
1. Introduction & the two papers (README §1-2).
2. Setup (README §2.5) + validation (unit tests, sanity check numbers).
3. RQ0: where accuracy is lost (ladder table).
4. RQ1: the order of operations → *importance order of pipelines*.
5. RQ2: *importance order of modules/layers*, quantized vs FP (Kendall τ), proxies.
6. RQ3: *importance order of quantization components* (Shapley before vs after FT), PTQ tricks after FT.
7. RQ4/RQ5: integer attention under finetuning, rank.
8. Discussion: confirm/refute H1-H4 (README §1), limitations.

## 5. Time & memory notes / troubleshooting

* Per finetuning run: ~1,000 optimizer steps of batch 16 (≤512 tokens) + evaluation. With bf16 on an
  A100 expect roughly 20-40 min; GPU memory ~10-20 GB. If you run out of memory: `train.grad_checkpointing=true`,
  or `train.batch_size=8 train.grad_accum=2` (same effective batch).
* Evaluation is the slow part of short runs. For quick iterations: `eval.lm_eval_tasks=[]`,
  `eval.gsm8k_limit=300`, `eval.ppl_max_chunks=40`. Don't mix limited and full evaluations in one table.
* IntAttention runs a dense fp32 simulation of the integer attention (plus an SDPA reference for the STE
  gradient during training) — expect it to be slower than `attn_mode: fp`.
* `KeyError: 'qft'` from transformers ⇒ wrong transformers version; install the pinned `4.56.2`.
* `lm_eval` failing on a task download: pre-download (see §0) or drop that task from `eval.lm_eval_tasks`.
* Non-finite loss ⇒ the run aborts with `FloatingPointError`; lower `train.lr` (e.g. 1e-4) for that config.
* To rerun something, delete its `seed<k>` folder or pass `--force`.
* The launcher writes each run's exact resolved config to `config.yaml` in the run folder — use it to
  reproduce a single run with `python scripts/run_experiment.py --config <that file>`.
