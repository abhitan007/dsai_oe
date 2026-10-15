# What Is the Importance Order of Finetuning Quantized SLMs?

**Goal.** Measure *what matters, and in which order*, when LoRA-finetuning an SLM deployed in a **fully integer** form, by combining two papers:
**QServe/QoQ** (W4A8KV4: progressive group quantization, SmoothAttention, Hadamard rotation, output smoothing, weight clipping, channel reordering) and
**IntAttention** (IndexSoftmax: INT8 Q/K/V, integer clipping, 32-entry UINT8 LUT exponential, integer normalization to UINT8 P).

**Model: Qwen2.5-0.5B (base).** (i) Small enough for ~290 runs with 2–3 seeds each, needed for statistically meaningful *rankings*.
(ii) A hard case: GQA (14 Q / 2 KV heads), QKV biases, tied embeddings, and a non-power-of-two width (896). IntAttention reported Qwen as its most attention-sensitive family.
(iii) Strong math pretraining gives a measurable GSM8K signal at 0.5B. (iv) Open licence, no gating. Llama-3.2-1B and Qwen2.5-1.5B/3B also run unchanged.
Qwen3 is excluded because its q/k-norm blocks folding SmoothAttention into the weights.

**Novelties**
1. **First W4A8KV4 + integer-softmax stack**: every GEMM *and* the whole attention path in integer arithmetic, simulated bit-faithfully (integer GEMMs exact in fp32).
2. **Quantization-aware LoRA through IndexSoftmax**: STE whose forward pass is the LUT/integer attention and whose backward pass is exact softmax.
   Gradients also flow through A8 and KV4.
3. **Train/deploy decoupling**: tests *which* quantizers must be in the training loop (W-only → +A8 → +KV4 → +IntAttention).
4. **Shapley attribution** over a 2⁴ factorial {W4, A8, KV4, IntAttn}: a fair importance order of the damage *before vs. after* finetuning (what LoRA repairs).
5. **Module/layer importance under quantization vs. full precision** (single, leave-one-out and layer-band LoRA; Kendall τ), checked against training-free proxies (Taylor, Fisher, gradient norm; Spearman ρ).
6. **Do PTQ tricks survive finetuning?** Leave-one-trick-out across all six QoQ techniques, before and after LoRA.
7. **Adapter deployment forms**: side branch on INT8 input vs. merge + re-quantize (dequantized or FP base), plus LoftQ initialisation.
8. **Engineering**: exact Hadamard of order 896 = Paley(28, over GF(27)) ⊗ Sylvester(32); GQA-, bias- and RoPE-aware SmoothAttention and smoothing.
   Padding-aware integer-attention scales. HF `AttentionInterface` hook, so KV4/IntAttention are also active during `generate()`.

**Research questions** RQ0 PTQ ladder · RQ1 order of quantize/finetune (11 pipelines) · RQ2 modules/layers · RQ3 components (Shapley) and PTQ tricks · RQ4 IntAttention (b, c) and STE · RQ5 LoRA rank.
**Protocol:** GSM8K SFT, 1k steps, r = 16. Metrics: GSM8K exact match, WikiText-2 PPL (2048 ctx), 5 zero-shot tasks (lm-eval), held-out loss.
Calibration on C4; mean ± std over seeds.

**Validation so far (measured).** All 20 unit tests pass (incl. memory-efficient loss = HF loss): exact orthogonality; transforms preserve logits to < 2·10⁻⁴; attention matches HF incl. padding and KV-cache generation; STE gradients reach only LoRA.
On the real model, our attention matches SDPA (max |Δlogit| = 0) and rotation + smoothing + SmoothAttention leave PPL unchanged (Δ = 0.0000).
Preliminary PTQ ladder below: CPU run, WikiText-2 PPL on only 4×512 tokens, so indicative only.

| FP | W8A8 | W4A8 RTN | +rotation | +clip | +KV4 | +SmoothAttn | QoQ g128 | QoQ+IntAttn | IntAttn only | W4A8KV4 no tricks |
|---|---|---|---|---|---|---|---|---|---|---|
| 17.41 | 17.79 | 30.91 | 24.84 | 21.19 | 123.4 | 27.80 | 26.96 | 27.59 | 18.42 | 180.1 |

Early reading: KV4 is the dominant PTQ damage on Qwen (key outliers), and SmoothAttention removes most of it. Rotation and clipping each matter. IntAttention alone costs about 1 PPL.
These are the PTQ "before" rankings that finetuning will be tested against.

**Status / next.** The full pipeline (PTQ → LoftQ → QA-LoRA training → `requant_fp` merge → all evaluations) runs end-to-end, verified on a tiny Qwen2 with the real tokenizer. CPU smoke testing exposed and fixed a 152k-vocab logits memory blow-up (~4 GB per batch). The full 0.5B smoke run needs more than this 7.8 GB test VM provides; run it first on the GPU machine.
**Hypotheses to test:** quantization-aware LoRA > finetune-then-quantize; W4 damage is the most LoRA-repairable; function-preserving tricks stay important after finetuning while clipping/reordering become redundant.
**Limitations:** one model and one main task; simulated (not kernel) quantization; GSM8K differences < 2 pp are within noise.
