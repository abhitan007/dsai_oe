"""Quantized attention: KV4 cache quantization (QServe) + IntAttention / IndexSoftmax.

The function :func:`qft_attention_forward` is registered with HuggingFace's ``AttentionInterface``
under the name ``"qft"``.  HF calls it *after* RoPE and the KV-cache update and *before* the KV
heads are repeated for GQA, i.e. exactly where a real serving kernel sees Q, K, V.

Per decoder layer, the behaviour is controlled by ``module.qft_attn`` (an :class:`AttnQuantState`):

* ``kv_bits < 16``  – dynamic asymmetric per-(token, head) quantization of K and V (QServe KV4).
* ``mode="fp"``           – exact softmax attention (SDPA).
* ``mode="quant_only"``   – IntAttention's "Quant-Only" baseline: INT8 Q/K/V, INT32 logits,
                            dequantize -> FP32 softmax -> requantize P to INT8 (x127).
* ``mode="int"``          – IntAttention: INT8 Q/K/V, INT32 logits, IndexSoftmax
                            (integer clipping, 2^b-entry UINT8 LUT exponential, integer row
                            normalization to UINT8 P (x255)), integer P.V.

Training through the integer paths uses a straight-through estimator: the forward value is the
integer pipeline's output, the gradient is that of exact softmax attention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .primitives import fake_quant_asym, quant_sym

ATTN_IMPL_NAME = "qft"


@dataclass
class AttnQuantState:
    kv_bits: int = 16
    mode: str = "fp"                  # fp | quant_only | int
    lut_bits: int = 5                 # IntAttention b
    clip_c: float = 6.6               # IntAttention c
    granularity: str = "per_head"     # per_head | per_tensor | per_token  (scales of Q, K, V)
    p_scale: int = 255                # 255 -> UINT8 P (IntAttention); 127 -> INT8 P
    calibrate: bool = False           # record per-channel max|K| (for SmoothAttention)
    k_amax: torch.Tensor | None = None


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, l, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, l, d).reshape(b, h * n_rep, l, d)


def _valid_mask(attention_mask, q_len: int, k_len: int, device) -> torch.Tensor:
    """Boolean [B|1, 1, q, k] mask, True = query may attend key."""
    if attention_mask is None:
        q_pos = torch.arange(k_len - q_len, k_len, device=device)[:, None]
        k_pos = torch.arange(k_len, device=device)[None, :]
        return (k_pos <= q_pos)[None, None]
    m = attention_mask[:, :, :, :k_len]
    if m.dtype == torch.bool:
        return m
    return m > -1e4   # additive masks use dtype-min for masked entries


def build_lut(lut_bits: int, clip_c: float, device) -> torch.Tensor:
    """UINT8 LUT:  round(255 * exp(-c i / (2^b - 1))) for i < 2^b - 1, and 0 for the last entry."""
    n = 2 ** lut_bits
    i = torch.arange(n, device=device, dtype=torch.float32)
    lut = torch.round(255.0 * torch.exp(-clip_c * i / (n - 1)))
    lut[-1] = 0.0
    return lut


def _scale_dims(granularity: str):
    # tensors are [B, H, L, D]
    if granularity == "per_head":
        return (2, 3)
    if granularity == "per_tensor":
        return (0, 1, 2, 3)
    if granularity == "per_token":
        return (3,)
    raise ValueError(granularity)


@torch.no_grad()
def integer_attention(q, k, v, valid, scaling: float, st: AttnQuantState) -> torch.Tensor:
    """Simulated integer attention.  q:[B,H,Lq,D], k,v:[B,H,Lk,D] (fp32), valid: bool [B|1,1,Lq,Lk]."""
    dims = _scale_dims(st.granularity)
    key_valid = valid.any(dim=-2).unsqueeze(-1)       # [B|1,1,Lk,1]  key is attended by some query
    query_valid = valid.any(dim=-1).unsqueeze(-1)     # [B|1,1,Lq,1]
    q_hat, s_q = quant_sym(q, 8, dims, valid=query_valid)
    k_hat, s_k = quant_sym(k, 8, dims, valid=key_valid)
    v_hat, s_v = quant_sym(v, 8, dims, valid=key_valid)
    a_hat = q_hat @ k_hat.transpose(-1, -2)          # INT32 logits (exact in fp32 for D <= 128)
    alpha = (s_q * s_k.transpose(-1, -2)) * scaling   # broadcastable to [B,H,Lq,Lk]

    if st.mode == "quant_only":
        logits = (alpha * a_hat).masked_fill(~valid, float("-inf"))
        p = torch.softmax(logits, dim=-1).nan_to_num(0.0)              # FP32 softmax
        p_hat = torch.round(p * st.p_scale)                             # requantize P
    elif st.mode == "int":
        n_lut = 2 ** st.lut_bits
        lut = build_lut(st.lut_bits, st.clip_c, q.device)
        c_int = torch.round(st.clip_c / alpha).clamp(min=1.0)           # Eq. 8
        row_max = a_hat.masked_fill(~valid, float("-inf")).amax(-1, keepdim=True)
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        delta = row_max - a_hat                                          # Eq. 7 (non-negative ints)
        delta = torch.where(valid, delta, c_int)                         # masked keys -> LUT zero entry
        delta = torch.minimum(delta, c_int)                              # Eq. 9 clipping
        idx = torch.round(delta / c_int * (n_lut - 1)).long().clamp(0, n_lut - 1)   # Eq. 11
        e_hat = lut[idx]                                                 # Eq. 14 (UINT8 gather)
        row_sum = e_hat.sum(-1, keepdim=True)                            # INT32 accumulation
        p_hat = torch.where(row_sum > 0, torch.round(st.p_scale * e_hat / row_sum.clamp(min=1)),
                            torch.zeros_like(e_hat))                     # Eq. 15 UINT8 P
    else:
        raise ValueError(st.mode)
    out = (p_hat @ v_hat) * (s_v / st.p_scale) if st.granularity != "per_token" else \
        (p_hat @ (v_hat * s_v)) / st.p_scale
    return out


def qft_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout: float = 0.0, **kwargs):
    """HF AttentionInterface-compatible attention with KV and integer-attention simulation."""
    st: AttnQuantState | None = getattr(module, "qft_attn", None)
    if scaling is None:
        scaling = 1.0 / math.sqrt(query.shape[-1])
    if st is not None and st.calibrate:
        amax = key.detach().abs().amax(dim=(0, 2)).float()             # [kv_heads, D]
        st.k_amax = amax if st.k_amax is None else torch.maximum(st.k_amax, amax)

    if st is not None and st.kv_bits < 16:
        # QServe KV4: per-head, per-token dynamic asymmetric quantization (group = head_dim)
        key = fake_quant_asym(key, st.kv_bits, dim=-1)
        value = fake_quant_asym(value, st.kv_bits, dim=-1)

    n_rep = query.shape[1] // key.shape[1]
    key = repeat_kv(key, n_rep)
    value = repeat_kv(value, n_rep)
    q_len, k_len = query.shape[-2], key.shape[-2]
    valid = _valid_mask(attention_mask, q_len, k_len, query.device)

    mode = st.mode if st is not None else "fp"
    need_ref = mode == "fp" or (torch.is_grad_enabled() and (query.requires_grad or key.requires_grad))
    ref = None
    if need_ref:
        add_mask = torch.zeros(valid.shape, dtype=query.dtype, device=query.device)
        add_mask = add_mask.masked_fill(~valid, torch.finfo(query.dtype).min)
        ref = F.scaled_dot_product_attention(query, key, value, attn_mask=add_mask,
                                             dropout_p=dropout if module.training else 0.0, scale=scaling)
    if mode == "fp":
        out = ref
    else:
        with torch.autocast(device_type=query.device.type, enabled=False):
            out_int = integer_attention(query.float(), key.float(), value.float(), valid, scaling, st)
        out_int = out_int.to(query.dtype)
        out = out_int if ref is None else ref + (out_int - ref).detach()   # STE
    return out.transpose(1, 2).contiguous(), None


def register_attention():
    """Register the "qft" attention (and its mask format) with transformers."""
    from transformers import AttentionInterface
    AttentionInterface.register(ATTN_IMPL_NAME, qft_attention_forward)
    try:
        from transformers.masking_utils import AttentionMaskInterface, eager_mask
        AttentionMaskInterface.register(ATTN_IMPL_NAME, eager_mask)
    except ImportError:  # older transformers build masks inside the model
        pass
