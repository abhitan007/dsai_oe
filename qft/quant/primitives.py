"""Low-level (fake-)quantization primitives.

All quantizers here *simulate* integer arithmetic in floating point ("fake quantization"):
a tensor is mapped to its integer grid, rounded/clamped, and mapped back.  Integer values up
to 2^24 are exactly representable in fp32, so every integer GEMM we simulate (INT8 x INT8 with
INT32 accumulation, head_dim <= 128) is *bit exact* as long as it is computed in fp32.

Training through quantizers uses the straight-through estimator (STE):
    y = x + (Q(x) - x).detach()
i.e. forward = quantized value, backward = identity.
"""
from __future__ import annotations

import torch

EPS = 1e-8


def ste(x: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: value of ``xq``, gradient of ``x``."""
    if not x.requires_grad:
        return xq
    return x + (xq - x).detach()


# --------------------------------------------------------------------------------------
# symmetric / asymmetric integer quantizers (return integer codes + scales)
# --------------------------------------------------------------------------------------
def quant_sym(x: torch.Tensor, n_bits: int, dim, qmax: int | None = None, valid: torch.Tensor | None = None):
    """Symmetric quantization with zero point fixed at 0.

    Returns (q, scale) with q integer-valued in [-qmax, qmax].  ``dim`` are the dimensions the
    scale is shared over (reduced with abs-max).  ``valid`` (broadcastable bool) excludes entries
    (e.g. padded tokens) from the abs-max statistic.
    """
    qmax = qmax if qmax is not None else 2 ** (n_bits - 1) - 1
    a = x.abs()
    if valid is not None:
        a = a.masked_fill(~valid, 0.0)
    scale = a.amax(dim=dim, keepdim=True).clamp(min=EPS) / qmax
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return q, scale


def fake_quant_sym(x: torch.Tensor, n_bits: int, dim=-1, use_ste: bool = True) -> torch.Tensor:
    """Dynamic symmetric fake quantization (computed in fp32, returned in x.dtype)."""
    if n_bits >= 16:
        return x
    xf = x.float()
    q, s = quant_sym(xf, n_bits, dim)
    xq = (q * s).to(x.dtype)
    return ste(x, xq) if use_ste else xq


def quant_asym(x: torch.Tensor, n_bits: int, dim, include_zero: bool = True):
    """Asymmetric (min/max) quantization to unsigned codes in [0, 2^b - 1]."""
    qmax = 2 ** n_bits - 1
    xmin = x.amin(dim=dim, keepdim=True)
    xmax = x.amax(dim=dim, keepdim=True)
    if include_zero:
        xmin = xmin.clamp(max=0)
        xmax = xmax.clamp(min=0)
    scale = (xmax - xmin).clamp(min=EPS) / qmax
    zero = torch.round(-xmin / scale)
    q = torch.clamp(torch.round(x / scale) + zero, 0, qmax)
    return q, scale, zero


def fake_quant_asym(x: torch.Tensor, n_bits: int, dim=-1, use_ste: bool = True, include_zero: bool = False) -> torch.Tensor:
    """Dynamic asymmetric fake quantization (used for KV cache: per-token, per-head)."""
    if n_bits >= 16:
        return x
    xf = x.float()
    q, s, z = quant_asym(xf, n_bits, dim, include_zero=include_zero)
    xq = ((q - z) * s).to(x.dtype)
    return ste(x, xq) if use_ste else xq


def fake_quant_activation(x: torch.Tensor, n_bits: int, granularity: str = "per_token", use_ste: bool = True) -> torch.Tensor:
    """Activation quantization used in front of every linear layer (QServe: per-token symmetric INT8)."""
    if n_bits >= 16:
        return x
    if granularity == "per_token":
        return fake_quant_sym(x, n_bits, dim=-1, use_ste=use_ste)
    if granularity == "per_tensor":
        return fake_quant_sym(x, n_bits, dim=tuple(range(x.dim())), use_ste=use_ste)
    raise ValueError(f"unknown activation granularity {granularity}")
