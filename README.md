# What is the importance order of finetuning such quantized models?

**Course project: finetuning W4A8KV4 + integer-attention quantized LLMs (Qwen2.5-0.5B)**

This repository is a complete, self-contained experimental framework for studying **parameter-efficient
finetuning (LoRA / QLoRA-style) of aggressively quantized language models**. It combines the
two papers in this folder:

| Paper | What we take from it |
|---|---|
| **QServe / QoQ** — *W4A8KV4 Quantization and System Co-design for Efficient LLM Serving* (Lin et al., MLSys 2025) | The full **QoQ** post-training quantization algorithm: 4-bit weights with **progressive group quantization** (INT8 protective range [-119,119] → UINT4 groups of 128), 8-bit per-token activations, 4-bit per-head KV cache, **SmoothAttention**, **block-input Hadamard rotation**, **block-output smoothing**, **output-MSE weight clipping**, **activation-aware channel reordering**. |
| **IntAttention** — *A Fully Integer Attention Pipeline for Efficient Edge Inference* (Zhong et al., MLSys 2026) | The **IndexSoftmax** integer attention: INT8 Q/K/V, INT32 logits, integer max-subtraction, **sparsity-aware clipping** at `c_int = round(c·√d / (s_Q s_K))`, a **32-entry UINT8 LUT exponential** (b = 5, c = 6.6), **integer row normalisation** to UINT8 probabilities (×255), integer P·V. Also its **"Quant-Only"** baseline (dequantize → FP softmax → requantize). |
| **QLoRA / LoRA / LoftQ** (background) | Low-rank adapters trained on top of a *frozen quantized* base; LoftQ-style SVD initialisation of the adapters from the quantization residual. |

Stacking both papers gives an (as far as we know) unexplored deployment target: **every GEMM and the
whole attention path in integer arithmetic** (W4A8 linear layers, KV4 cache, integer softmax). The
project asks how one should *finetune* such a model, and what matters most when doing so.

---

## 1. Research questions — what "importance order" means here

The title is deliberately broad, so we split it into five concrete, falsifiable questions. Each one
produces an explicit *ranking* (an importance order) in the auto-generated report.

| | Question | Ranking produced | Suite |
|---|---|---|---|
| **RQ0** | Where does accuracy go when we quantize? (QServe Fig. 12 ladder + IntAttention) | Damage per quantization step | `rq0_ptq_ladder` |
| **RQ1** | **Order of operations.** Quantize-then-finetune or finetune-then-quantize? Which quantizers must be *in the loop* during finetuning? How should the adapter be deployed (separate branch vs. merged + re-quantized)? | Ranking of 11 pipelines by accuracy of the **deployed** integer model | `rq1_pipelines` |
| **RQ2** | **Which modules / layers** (q, k, v, o, gate, up, down; early / middle / late layers) are most important to adapt, and does quantization *change* this order compared with full-precision LoRA? Can a training-free proxy predict it? | Module and layer-band importance orders, Kendall τ (quantized vs FP), Spearman ρ (proxy vs measured) | `rq2_modules` + `compute_importance.py` |
| **RQ3** | **Which quantization component** (W4, A8, KV4, IntAttention) causes most of the damage, and which one does finetuning *repair*? Do the QoQ PTQ tricks still matter once we finetune? | **Shapley values** of the accuracy drop per component before vs after finetuning; leave-one-trick-out deltas | `rq3_components`, `rq3_tricks` |
| **RQ4** | Can finetuning adapt *through* the non-differentiable IndexSoftmax (STE), and does it shift IntAttention's optimal (b, c)? | Train-with-IndexSoftmax vs train-with-exact-softmax; (b, c) heat-map | `rq4_intattn` |
| **RQ5** | Does recovering from quantization need more adapter *capacity* (rank) than ordinary finetuning? | Accuracy-vs-rank curves (FP vs quantized base) | `rq5_rank` |

### Hypotheses (to be confirmed or refuted by your runs)

* **H1 (order).** Quantization-aware LoRA (`ptq_then_ft` with *all* deployment quantizers simulated in the
  forward pass) beats `ft_then_ptq`, because the adapter can compensate the exact errors that will be present
  at inference; training with fewer quantizers in the loop (QLoRA-style weight-only) degrades as the
  train/deploy mismatch grows. Merging the adapter and re-quantizing (`merge_deq`, `merge_fp`) loses part
  of the gain, since the low-rank update is mostly below the 4-bit step size.
* **H2 (modules).** Under quantization the importance order shifts towards the modules whose *inputs* are
  hardest to quantize (`down_proj`, whose input is the post-SwiGLU activation with huge outliers, and
  `o_proj`) and towards `k`/`v` (they compensate KV4 + IntAttention error), whereas FP LoRA on GSM8K
  favours MLP modules.
* **H3 (components).** Before finetuning, KV4 and A8 dominate the damage on a 0.5B model (its small head
  dim and narrow hidden size make activations relatively spiky); W4 damage is the most *repairable* by
  LoRA (a weight-space error, exactly what LoRA parameterises), activation/KV damage the least. IntAttention
  adds little on top (the paper reports near-lossless behaviour), but Qwen models were reported to be
  the most sensitive in the IntAttention paper.
* **H4.** PTQ tricks that are *function-preserving* (rotation, smoothing) remain important after
  finetuning, while *error-tuning* tricks (clipping, reordering) become largely redundant.

---

## 2. Method

### 2.1 Simulated (fake) quantization that is bit-faithful
Everything runs in PyTorch on top of HuggingFace `transformers`, so it trains on any GPU. Each integer
operation is *simulated*: tensors are mapped to integer codes, rounded and clamped exactly as the paper
specifies, and mapped back. The integer GEMMs of IntAttention (`Q̂K̂ᵀ` with INT8 inputs and head dim 64,
`P̂V̂` with UINT8 P) are computed in fp32 where all intermediate integers are below 2²⁴, so they are
**exact**, not approximations. Speed-ups are *not* measured (the papers already do that with custom
kernels) — this project is about accuracy and finetuning dynamics.

### 2.2 The quantized model (deployment target `qoq_intattn`)

```
                      ┌──────────── offline (once) ─────────────┐
 FP model ─► fold RMSNorm scales ─► rotate residual stream by H (Hadamard 896 = Paley(28) ⊗ Sylvester(32))
          ─► calibration pass (128 × 512 C4 tokens)
          ─► SmoothAttention  (W_q·λ, W_k/λ, λ_i = λ_{i+D/2} = max|K|^0.5, RoPE-safe, GQA-aware)
          ─► block-output smoothing v→o, up→down (α grid-searched per layer on output MSE)
          ─► per linear: reorder input channels by max|X| ─► clip-ratio search per (row, group)
          ─► progressive group quant: INT8 per-channel (±119) → UINT4 per-group(128), UINT8 int scales
                      └──────────────────────────────────────────┘
 runtime:  x ─► INT8 per-token ─► W4 GEMM ─► …     K,V ─► UINT4 per-(token,head) asym  (KV4)
           attention: INT8 Q,K,V ─► INT32 logits ─► IndexSoftmax (clip c_int, 32-entry LUT, ∑, ×255) ─► UINT8 P ─► P̂V̂
```

Implementation map: `qft/quant/transforms.py` (rotation, smoothing, SmoothAttention),
`qft/quant/weight_quant.py` (RTN / progressive / NF4, clip search, reordering),
`qft/quant/attention.py` (KV4, Quant-Only, IndexSoftmax), `qft/quant/apply.py` (pipeline driver).
The custom attention is registered through HuggingFace's `AttentionInterface`, so it sees exactly the
post-RoPE, post-KV-cache Q/K/V a serving kernel would see — including during `generate()`.

### 2.3 Finetuning the quantized model
`QuantLinear` (`qft/quant/quant_linear.py`) holds the frozen quantized weight and an optional LoRA branch:

```
y = Q_A(x) · Q_W(W)ᵀ + b  +  (α/r) · B · A · Q_A(x)
```

* **Quantization-aware LoRA.** All deployment quantizers (A8, KV4, IndexSoftmax) can be active during
  training. Gradients pass through them with the **straight-through estimator** (`x + (Q(x) − x).detach()`).
  For the integer attention the forward value is IndexSoftmax's output and the backward pass uses exact
  softmax attention (`ref + (int − ref).detach()`).
* **Train/deploy decoupling.** `train_runtime` overrides the runtime quantizers *only during training*
  (e.g. QLoRA-style weight-only training, then deployment with A8KV4 + IntAttention) — this is how RQ1 measures
  which quantizers have to be "in the loop".
* **Adapter deployment forms.** `none` (adapter stays a side branch fed with the INT8 activation),
  `requant_deq` (merge into the dequantized W4 weight and re-quantize with the same scales/clip/perm —
  no FP weights needed), `requant_fp` (merge into the original FP weights, then quantize).
* **Initialisation.** standard LoRA (B = 0) or **LoftQ** (SVD of the quantization residual `W − Q(W)`).

### 2.4 Pipelines (RQ1)

| pipeline | what happens | deployed model |
|---|---|---|
| `fp_eval` | nothing | FP16 |
| `fp_ft` | LoRA on the FP model | FP16 + adapter (upper bound) |
| `ptq_only` | QoQ + IntAttention PTQ | integer (lower bound) |
| `ft_then_ptq` | LoRA on FP → merge → PTQ (calibrated on the merged model) | integer |
| `ptq_then_ft` | PTQ → LoRA on the frozen quantized base | integer + adapter (or merged) |

### 2.5 Task, training and evaluation protocol
* **Model:** `Qwen/Qwen2.5-0.5B` (base). 24 layers, hidden 896, 14 query / 2 KV heads (GQA), head dim 64,
  QKV biases, tied embeddings — all handled (biases are scaled by SmoothAttention/smoothing; the LM head is
  untied before rotation; GQA-aware smoothing). Llama/Mistral-style models also work (`model:` field).
  Qwen3 is *not* supported (its q/k-norm prevents folding SmoothAttention into the weights).
* **Finetuning task:** GSM8K supervised finetuning (7,217 train problems; 256 held out for validation loss),
  loss on the answer only, 1,000 steps × batch 16 (≈2.2 epochs), AdamW, lr 2e-4, cosine, warm-up 3%,
  grad-clip 0.3, LoRA r = 16, α = 32, dropout 0.05 on all 7 linear types (QLoRA defaults). An alternative
  *language-modelling recovery* task on C4 is provided (`configs/base_c4.yaml`).
* **Metrics:** GSM8K test exact-match (1,319 problems, greedy, 0-shot after finetuning; 8-shot for models
  that were not finetuned), WikiText-2 perplexity (2048-token windows, GPTQ protocol — measures *general*
  damage / forgetting), zero-shot ARC-e/ARC-c/PIQA/HellaSwag/WinoGrande via `lm-evaluation-harness`,
  held-out task loss. C4 perplexity optional.
* **Calibration:** 128 random 512-token windows of C4 (QServe used the Pile; WikiText would leak into the
  perplexity test set).
* **Statistics:** 3 seeds for the key suites (2 for the larger RQ2/RQ5 sweeps), mean ± sample std.
  GSM8K has a binomial standard error of ≈1.3 pp at 35% accuracy, so differences under ~2 pp should be
  treated as ties unless they are consistent across seeds. Importance orders are reported with rank
  correlations (Kendall τ, Spearman ρ); component importance uses exact **Shapley values** over the 2⁴
  factorial design (fair attribution with interactions).

### 2.6 Training-free importance proxies (RQ2)
`scripts/compute_importance.py` computes, per linear module and in the rotated/smoothed coordinate system:
weight error `‖ΔW‖/‖W‖`, layer-output error, first-order Taylor `|⟨G, ΔW⟩|`, diagonal-Fisher
`½ Σ F ΔW²`, and task-gradient norm `‖G‖` (G = GSM8K loss gradient). `analyze.py` correlates them with the
measured RQ2 importance — a positive result would let practitioners choose LoRA targets without sweeps.

---

## 3. Repository layout

```
qft/                          the library
  config.py                   dataclasses, YAML loading, named quantization presets (QUANT_PRESETS)
  model_utils.py              model loading (registers the "qft" attention), structure helpers
  data.py                     C4 / WikiText / GSM8K: calibration, SFT data, perplexity corpora
  lora.py                     add / merge / save LoRA adapters
  train.py                    minimal transparent training loop (AdamW, cosine, grad-clip, val loss)
  evaluate.py                 perplexity, GSM8K generation + answer extraction, lm-eval wrapper
  runner.py                   one experiment = pipeline × quant × LoRA × task × seed -> result.json
  quant/
    primitives.py             sym/asym fake quant, STE, per-token activation quant
    weight_quant.py           RTN, QoQ progressive group quant, NF4, clip search, reordering
    hadamard.py               Sylvester / Paley-I / Kronecker Hadamard matrices, random orthogonal
    transforms.py             norm folding, rotation, block-output smoothing, SmoothAttention
    calibration.py            activation statistics, token samples, max|K|
    attention.py              KV4, Quant-Only attention, IntAttention / IndexSoftmax (+STE)
    quant_linear.py           QuantLinear (W/A quant + LoRA, LoftQ init, merge/requant)
    apply.py                  PTQ pipeline driver, runtime switches
scripts/
  run_experiment.py           run one config (with --set overrides)
  launch.py                   expand a suite and schedule its runs over several GPUs (resumable)
  sanity_check.py             equivalence checks + quick PTQ ladder on the real model
  compute_importance.py       training-free importance proxies (RQ2)
  analyze.py                  aggregate -> results/REPORT.md, summary.csv, figures/
configs/
  base.yaml, base_c4.yaml     base run configs (GSM8K SFT / C4 recovery)
  suites/*.yaml               rq0 … rq5 + smoke
tests/test_qft.py             20 unit tests (CPU, tiny random Qwen2) — run before anything else
INSTRUCTIONS.md               step-by-step how to run everything
```

---

## 4. Verification that was done while building this repository

The unit tests (`python -m pytest -q tests/`, all 20 pass on CPU) check, among other things:

* Hadamard matrices of order 12, 28, 64, 96, 896, 1536 are exactly orthogonal (896 uses the Paley-I
  construction over GF(27)).
* **Rotation + block-output smoothing + SmoothAttention are function-preserving**: logits of a tiny
  random Qwen2 (with injected key outliers, non-trivial RMSNorm scales, QKV biases, GQA, tied embeddings)
  are unchanged to < 2·10⁻⁴.
* The custom `"qft"` attention reproduces HF eager attention, with and without left padding, and greedy
  `generate()` with a KV cache gives identical tokens.
* Progressive group quantization never overflows the INT8 intermediate range; at most 16 levels per group.
* Clip search + reordering do not increase the layer-output error.
* IndexSoftmax and Quant-Only attention have cosine similarity > 0.99 with exact attention, and larger LUTs
  reduce error.
* Gradients reach every LoRA matrix through A8 + KV4 + IndexSoftmax (STE) and never reach base weights;
  `exact` merge reproduces the unmerged model; `requant_*` merges are finite; LoftQ reduces ‖W − (Q + sBA)‖.
* `ft_then_ptq` unwrap → re-quantize round trip works.
* The memory-efficient training loss (`qft.train.causal_lm_loss`: LM head applied only at labelled positions, in
  checkpointed chunks) equals HF's loss and LoRA gradients to 1e-5/1e-6. With Qwen's 152k vocabulary the naive
  full-vocab fp32 logits cost ≈4 GB per 16×438 batch (plus HF's upcast/log-softmax copies); this was found
  during the CPU smoke test (OOM) and fixed.

`scripts/sanity_check.py` runs the same equivalence checks on the real Qwen2.5-0.5B plus a quick PTQ
ladder; run it first on your GPU (see INSTRUCTIONS.md). See §7 for the numbers we obtained on CPU.

---

## 5. Faithfulness to the papers and deliberate deviations

* **Simulation, not kernels.** QServe's CUDA kernels (compute-aware weight reorder, register-level
  parallelism, subtraction-after-multiplication) and IntAttention's ARM NEON kernels only change *speed*,
  not numerics, so they are not reimplemented. The numerics (rounding, clamping, protective range, integer
  scales, LUT values, ×255 normalisation) follow the papers' equations.
* **Weight clipping objective.** QServe minimises the *block* output MSE for `q_proj`/`k_proj` and the layer
  output MSE elsewhere; we use AWQ-style per-(row, group) layer-output MSE for all layers (cheaper, and it
  decomposes per group).
* **Smoothing α search** measures only the consumer layer's (o_proj / down_proj) output error. This is exact
  with respect to weight quantization because per-row (per-group) quantization of the producer
  (v_proj / up_proj) is invariant to row scaling.
* **IntAttention granularity.** The paper uses per-tensor scales for single-sequence edge inference; with
  batches we use one scale per (sequence, head) by default (`int_attn_granularity: per_head`; `per_tensor`
  and `per_token` are available — Eq. 16-18 of the paper). Padded positions are excluded from the scale
  statistics.
* **KV4** is per-token, per-head dynamic asymmetric UINT4 (group = head dim 64), applied post-RoPE, as in QServe.
* **Activations** are quantized at the input of every linear (per-token symmetric INT8). Embeddings and LM head
  stay in 16 bit (as in QServe/QuaRot).
* **NF4** uses fp32 block scales (QLoRA's double quantization of the scales is omitted — negligible).
* **LoRA branch input** is the INT8-quantized activation by default (`lora.input_quant`), i.e. the adapter is
  deployable on the same INT8 activation path; RQ1 includes the FP-input variant.

## 6. Compute budget (estimates)
On a single A100/H100-class GPU one finetuning run (1,000 steps + full evaluation) is expected to take
roughly 20-40 min; a PTQ-only run 10-20 min (dominated by GSM8K 8-shot generation and lm-eval).
Complete study ≈ 290 runs ≈ 100-150 GPU-hours; the core subset (RQ0, RQ1, RQ3a) ≈ 35-50 GPU-hours.
Priorities and reduced variants are listed in INSTRUCTIONS.md.

## 7. Results
Run `python scripts/analyze.py` — it writes `results/REPORT.md` (tables + importance orders + Shapley
decomposition + correlations) and `results/figures/*.png`. No results are included in this repository:
all numbers in your report must come from your own runs.

**Preliminary sanity numbers (measured while building the repo; CPU, WikiText-2 PPL on only 4×512 tokens,
calibration on 16×256 WikiText-2 tokens — indicative only, rerun `sanity_check.py` on GPU for real numbers):**

| FP | W8A8 | W4A8 RTN | +rotation/smoothing | +clip | +KV4 | +SmoothAttention | QoQ (g128 progressive, reorder) | QoQ + IntAttention | IntAttention only | W4A8KV4-g128 RTN (no tricks) |
|---|---|---|---|---|---|---|---|---|---|---|
| 17.41 | 17.79 | 30.91 | 24.84 | 21.19 | 123.4 | 27.80 | 26.96 | 27.59 | 18.42 | 180.1 |

Custom attention vs. HF SDPA: max |Δlogit| = 0; rotation + smoothing + SmoothAttention: ΔPPL = 0.0000.

## 8. Limitations / threats to validity
* One small model (0.5B) and one main finetuning task; generality to larger models is not established (the
  code supports Qwen2.5-1.5B/3B, Llama-3.2-1B etc. through the `model:` field).
* Simulated quantization: real kernels may differ in accumulation order/rounding of the fp16 epilogue.
* GSM8K accuracy differences of 1-2 pp are within noise; rely on seeds and on the WikiText-2/val-loss metrics,
  which are much less noisy.
* STE through IndexSoftmax is a heuristic; a learned/soft surrogate might do better (future work).

## 9. References
1. Y. Lin, H. Tang, S. Yang, Z. Zhang, G. Xiao, C. Gan, S. Han. *QServe: W4A8KV4 Quantization and System
   Co-design for Efficient LLM Serving.* MLSys 2025.
2. W. Zhong, H. Feng, Z. Zhou, H. Peng, S. Yu. *IntAttention: A Fully Integer Attention Pipeline for Efficient
   Edge Inference.* MLSys 2026.
3. T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.
4. E. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
5. Y. Li et al. *LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models.* ICLR 2024.
6. G. Xiao et al. *SmoothQuant.* ICML 2023. · J. Lin et al. *AWQ.* MLSys 2024. · S. Ashkboos et al. *QuaRot.* 2024.
7. L. Shapley. *A value for n-person games.* 1953 (used for component attribution).
