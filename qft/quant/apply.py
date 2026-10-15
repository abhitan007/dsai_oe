"""End-to-end post-training quantization pipeline (QoQ + IntAttention), simulated in PyTorch.

Order of operations (mirrors QServe / QoQ):
  1. block-input rotation (fold RMSNorm scales, rotate residual stream)        [offline, exact]
  2. calibration pass in full precision (activation stats, token samples, max|K|)
  3. SmoothAttention (fold lambda into W_q / W_k)                              [offline, exact]
  4. block-output smoothing v->o, up->down with per-layer alpha search         [offline, exact]
  5. per linear: activation-aware channel reorder, clip-ratio search, weight quantization
  6. runtime flags: A8 per-token activations, KV4 cache, attention mode (fp / quant_only / int)
"""
from __future__ import annotations

import time

import torch

from ..config import QuantConfig
from ..model_utils import iter_linears, layers, linear_name, set_linear
from .attention import AttnQuantState
from .calibration import CalibStats, collect_calibration
from .quant_linear import QuantLinear
from .transforms import rotate_model, smooth_attention, smooth_output_modules
from .weight_quant import WeightQuantSpec, quantize_weight, search_clip_ratios


def init_attention_states(model):
    for layer in layers(model):
        if getattr(layer.self_attn, "qft_attn", None) is None:
            layer.self_attn.qft_attn = AttnQuantState()


def set_runtime(model, rt: dict):
    """Apply runtime quantization flags (activation bits, KV bits, attention mode)."""
    for _, _, lin in iter_linears(model):
        if isinstance(lin, QuantLinear):
            lin.a_bits = rt.get("a_bits", lin.a_bits)
            lin.a_granularity = rt.get("a_granularity", lin.a_granularity)
    for layer in layers(model):
        st = layer.self_attn.qft_attn
        st.kv_bits = rt.get("kv_bits", st.kv_bits)
        st.mode = rt.get("attn_mode", st.mode)
        st.lut_bits = rt.get("int_attn_b", st.lut_bits)
        st.clip_c = rt.get("int_attn_c", st.clip_c)
        st.granularity = rt.get("int_attn_granularity", st.granularity)


FP_RUNTIME = dict(a_bits=16, kv_bits=16, attn_mode="fp")


def weight_spec(q: QuantConfig) -> WeightQuantSpec:
    return WeightQuantSpec(bits=q.w_bits, method=q.w_method, group_size=q.w_group_size, sym=q.w_sym)


@torch.no_grad()
def quantize_linear_weight(W: torch.Tensor, i: int, short: str, q: QuantConfig, stats: CalibStats | None):
    """Quantize one linear weight with QoQ's per-layer options. Returns (W_q, perm, clip)."""
    spec = weight_spec(q)
    perm = clip = None
    if q.reorder and spec.enabled and spec.method != "nf4" and q.w_group_size > 0:
        _, amax = stats.get(i, short)
        perm = torch.argsort(amax.to(W.device), descending=True)
    if q.weight_clip and spec.enabled and spec.method != "nf4":
        X, _ = stats.get(i, short)
        clip = search_clip_ratios(W, X, spec, perm=perm, a_bits=q.a_bits, ratios=q.clip_ratios)
    return quantize_weight(W, spec, clip=clip, perm=perm), perm, clip


@torch.no_grad()
def quantize_model(model, q: QuantConfig, calib_batches=None, seed: int = 0, keep_fp_weights: bool = False,
                   log=print, stop_after_transforms: bool = False):
    """Quantize ``model`` in place.  Every decoder linear becomes a :class:`QuantLinear`.

    Returns (artifacts, fp_weights, calib_stats_or_None).  ``fp_weights`` maps linear names to the
    transformed full-precision weights (needed for LoftQ init, requant_fp merging and importance
    scores).  With ``stop_after_transforms`` the linears are left as nn.Linear (FP, transformed)
    and the calibration statistics are returned for further analysis.
    """
    t0 = time.time()
    init_attention_states(model)
    set_runtime(model, FP_RUNTIME)
    art: dict = {"quant": q.__dict__.copy()}
    spec = weight_spec(q)

    if q.rotation != "none":
        rotate_model(model, q.rotation, seed)
        art["rotation"] = q.rotation
        log(f"[ptq] rotated residual stream ({q.rotation})")

    use_reorder = q.reorder and spec.enabled and spec.method != "nf4" and q.w_group_size > 0
    needs_calib = q.smooth_output or q.smooth_attention or (q.weight_clip and spec.enabled) or use_reorder \
        or stop_after_transforms
    stats: CalibStats | None = None
    if needs_calib:
        if calib_batches is None:
            raise ValueError("this quantization config needs calibration data")
        stats = collect_calibration(model, calib_batches, q.calib_tokens_per_batch, seed)
        log(f"[ptq] calibration done ({len(calib_batches)} batches)")
        if q.smooth_attention:
            smooth_attention(model, stats.k_amax, q.smooth_attention_alpha)
            log(f"[ptq] SmoothAttention folded (alpha={q.smooth_attention_alpha})")
        if q.smooth_output:
            art["smooth_alphas"] = smooth_output_modules(model, stats, spec, q.a_bits, q.smooth_alphas)
            log("[ptq] block-output smoothing done")

    if stop_after_transforms:
        return art, None, stats

    fp_weights = {}
    errs = {}
    for i, short, lin in iter_linears(model):
        name = linear_name(i, short)
        W = lin.weight.data.float()
        Wq, perm, clip = quantize_linear_weight(W, i, short, q, stats)
        if spec.enabled:
            errs[name] = float((Wq - W).norm() / W.norm().clamp(min=1e-12))
        if keep_fp_weights:
            fp_weights[name] = lin.weight.data.detach().to("cpu", copy=True)
        ql = QuantLinear(Wq.to(lin.weight.dtype), lin.bias, spec, perm, clip, name)
        set_linear(layers(model)[i], short, ql)
    set_runtime(model, q.runtime())
    if errs:
        art["weight_rel_err_mean"] = sum(errs.values()) / len(errs)
        art["weight_rel_err"] = errs
    art["ptq_seconds"] = time.time() - t0
    del stats
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    log(f"[ptq] done in {art['ptq_seconds']:.1f}s"
        + (f", mean rel. weight error {art['weight_rel_err_mean']:.4f}" if errs else ""))
    return art, fp_weights, None


def unwrap_model(model):
    """Replace QuantLinear modules by plain nn.Linear (after merging adapters)."""
    for i, short, lin in iter_linears(model):
        if isinstance(lin, QuantLinear):
            if lin.has_lora():
                raise RuntimeError("merge LoRA adapters before unwrapping")
            set_linear(layers(model)[i], short, lin.to_linear())
