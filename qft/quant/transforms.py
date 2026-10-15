"""Function-preserving weight transformations from QoQ (QServe Sec. 3.2 and App. B.2).

All transforms are mathematically exact in full precision; they only redistribute outliers so
that the subsequent quantization is easier:

* :func:`fold_norms` + :func:`rotate_model` – block-input rotation with an orthogonal (Hadamard)
  matrix Q, absorbed into the embedding, the input modules (q/k/v, gate/up, lm_head: W <- W Q) and
  the output modules (o_proj, down_proj: W <- Q^T W).  Requires RMSNorm scales to be folded first
  (RMSNorm(xQ) = RMSNorm(x) Q when the norm has unit scale).
* :func:`smooth_output_modules` – SmoothQuant-style per-channel scaling of the block intermediate
  activations (v_proj -> o_proj, up_proj -> down_proj), migration strength searched per layer.
* :func:`smooth_attention` – SmoothAttention: K <- K / lambda, Q <- Q * lambda with
  lambda_i = lambda_{i + D/2} = max(max|K_i|, max|K_{i+D/2}|)^alpha (RoPE-compatible).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..model_utils import attn_geometry, layers
from .hadamard import get_rotation
from .primitives import fake_quant_activation
from .weight_quant import WeightQuantSpec, quantize_weight


# --------------------------------------------------------------------------------------
# norm folding + rotation
# --------------------------------------------------------------------------------------
def untie_embeddings(model):
    if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = nn.Parameter(model.lm_head.weight.data.clone())
    model.config.tie_word_embeddings = False


@torch.no_grad()
def _fold_norm_into(norm: nn.Module, linears):
    g = norm.weight.data.double()
    for lin in linears:
        W = lin.weight.data
        lin.weight.data = (W.double() * g[None, :]).to(W.dtype)
    norm.weight.data.fill_(1.0)


@torch.no_grad()
def fold_norms(model):
    untie_embeddings(model)
    for layer in layers(model):
        a, m = layer.self_attn, layer.mlp
        _fold_norm_into(layer.input_layernorm, [a.q_proj, a.k_proj, a.v_proj])
        _fold_norm_into(layer.post_attention_layernorm, [m.gate_proj, m.up_proj])
    _fold_norm_into(model.model.norm, [model.lm_head])


@torch.no_grad()
def rotate_model(model, kind: str = "hadamard", seed: int = 0):
    """Rotate the residual stream by an orthogonal matrix Q (hidden x hidden)."""
    fold_norms(model)
    d = model.config.hidden_size
    dev = model.model.embed_tokens.weight.device
    Q = get_rotation(d, kind, seed).to(dev)

    def right(lin):      # W <- W Q   (module reads the rotated residual stream)
        W = lin.weight.data
        lin.weight.data = (W.double() @ Q).to(W.dtype)

    def left(lin):       # W <- Q^T W (module writes into the residual stream)
        W = lin.weight.data
        lin.weight.data = (Q.T @ W.double()).to(W.dtype)
        if lin.bias is not None:
            lin.bias.data = (lin.bias.data.double() @ Q).to(lin.bias.dtype)

    E = model.model.embed_tokens.weight
    E.data = (E.data.double() @ Q).to(E.dtype)
    for layer in layers(model):
        a, m = layer.self_attn, layer.mlp
        for lin in (a.q_proj, a.k_proj, a.v_proj, m.gate_proj, m.up_proj):
            right(lin)
        left(a.o_proj)
        left(m.down_proj)
    right(model.lm_head)
    return Q


# --------------------------------------------------------------------------------------
# block-output smoothing (v -> o, up -> down)
# --------------------------------------------------------------------------------------
def o_to_v_channel_map(model) -> torch.Tensor:
    """Index map: o_proj input channel -> v_proj output channel (handles GQA)."""
    n_heads, n_kv, D = attn_geometry(model)
    rep = n_heads // n_kv
    c = torch.arange(n_heads * D)
    head, d = c // D, c % D
    return (head // rep) * D + d


@torch.no_grad()
def _search_smooth(W: torch.Tensor, X: torch.Tensor, x_amax: torch.Tensor, w_amax: torch.Tensor,
                   expand, w_spec: WeightQuantSpec, a_bits: int, alphas) -> tuple[torch.Tensor, float]:
    """Return per-channel smoothing factors s (activations divided by s, weight columns multiplied)."""
    W = W.float()
    X = X.float()
    ref = X @ W.T
    best, best_err, best_alpha = None, float("inf"), None
    for alpha in alphas:
        s = (x_amax.clamp(min=1e-5).pow(alpha) / w_amax.clamp(min=1e-5).pow(1 - alpha)).clamp(1e-4, 1e4)
        s_full = expand(s)
        Wq = quantize_weight(W * s_full[None, :], w_spec)
        Xq = fake_quant_activation(X / s_full[None, :], a_bits, use_ste=False)
        err = (Xq @ Wq.T - ref).pow(2).mean().item()
        if err < best_err:
            best, best_err, best_alpha = s, err, alpha
    return best, best_alpha


@torch.no_grad()
def smooth_output_modules(model, calib, w_spec: WeightQuantSpec, a_bits: int, alphas):
    """SmoothQuant-style scaling of block intermediates, with QServe's per-layer alpha search.

    ``calib`` is the :class:`~qft.quant.calibration.CalibStats` object; its samples/amax for the
    smoothed inputs are updated in place.  Returns {layer: {"o": alpha, "down": alpha}}.
    """
    chosen = {}
    vmap = o_to_v_channel_map(model)
    n_v = int(vmap.max()) + 1
    for i, layer in enumerate(layers(model)):
        a, m = layer.self_attn, layer.mlp
        dev = a.o_proj.weight.device
        vm = vmap.to(dev)
        # ---- v_proj -> o_proj (scale shared by all q heads reading the same kv head)
        key = f"layers.{i}.o_proj"
        X, xa = calib.samples[key].to(dev), calib.amax[key].to(dev)
        wa = a.o_proj.weight.abs().amax(0).float()
        xa_v = torch.zeros(n_v, device=dev).scatter_reduce(0, vm, xa.float(), "amax", include_self=False)
        wa_v = torch.zeros(n_v, device=dev).scatter_reduce(0, vm, wa, "amax", include_self=False)
        s_v, alpha_o = _search_smooth(a.o_proj.weight, X, xa_v, wa_v, lambda s: s[vm], w_spec, a_bits, alphas)
        s_o = s_v[vm]
        a.v_proj.weight.data = (a.v_proj.weight.data.float() / s_v[:, None]).to(a.v_proj.weight.dtype)
        if a.v_proj.bias is not None:
            a.v_proj.bias.data = (a.v_proj.bias.data.float() / s_v).to(a.v_proj.bias.dtype)
        a.o_proj.weight.data = (a.o_proj.weight.data.float() * s_o[None, :]).to(a.o_proj.weight.dtype)
        calib.rescale_input(key, s_o)
        # ---- up_proj -> down_proj
        key = f"layers.{i}.down_proj"
        X, xa = calib.samples[key].to(dev), calib.amax[key].to(dev)
        wa = m.down_proj.weight.abs().amax(0).float()
        s, alpha_d = _search_smooth(m.down_proj.weight, X, xa.float(), wa, lambda s: s, w_spec, a_bits, alphas)
        m.up_proj.weight.data = (m.up_proj.weight.data.float() / s[:, None]).to(m.up_proj.weight.dtype)
        if m.up_proj.bias is not None:
            m.up_proj.bias.data = (m.up_proj.bias.data.float() / s).to(m.up_proj.bias.dtype)
        m.down_proj.weight.data = (m.down_proj.weight.data.float() * s[None, :]).to(m.down_proj.weight.dtype)
        calib.rescale_input(key, s)
        chosen[i] = {"o_proj": alpha_o, "down_proj": alpha_d}
    return chosen


# --------------------------------------------------------------------------------------
# SmoothAttention
# --------------------------------------------------------------------------------------
@torch.no_grad()
def smooth_attention(model, k_amax: dict, alpha: float = 0.5):
    """Fold SmoothAttention scales into W_q (x lambda) and W_k (/ lambda), incl. biases."""
    n_heads, n_kv, D = attn_geometry(model)
    rep = n_heads // n_kv
    for i, layer in enumerate(layers(model)):
        a = layer.self_attn
        amax = k_amax[i].float().to(a.k_proj.weight.device)          # [n_kv, D]
        half = D // 2
        lam_h = torch.maximum(amax[:, :half], amax[:, half:]).clamp(min=1e-5).pow(alpha)
        lam = torch.cat([lam_h, lam_h], dim=1)                         # lambda_i = lambda_{i+D/2}
        lam_k = lam.reshape(-1)                                        # [n_kv*D]
        lam_q = lam.repeat_interleave(rep, dim=0).reshape(-1)          # [n_heads*D]
        a.q_proj.weight.data = (a.q_proj.weight.data.float() * lam_q[:, None]).to(a.q_proj.weight.dtype)
        a.k_proj.weight.data = (a.k_proj.weight.data.float() / lam_k[:, None]).to(a.k_proj.weight.dtype)
        if a.q_proj.bias is not None:
            a.q_proj.bias.data = (a.q_proj.bias.data.float() * lam_q).to(a.q_proj.bias.dtype)
        if a.k_proj.bias is not None:
            a.k_proj.bias.data = (a.k_proj.bias.data.float() / lam_k).to(a.k_proj.bias.dtype)
