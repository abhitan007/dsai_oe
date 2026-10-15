"""Weight quantizers.

* ``rtn``         – round-to-nearest, per-channel (group_size=-1) or per-group, sym/asym.
* ``progressive`` – QoQ progressive group quantization (QServe, Sec. 3.1):
                    level 1: per-output-channel symmetric INT8 with *protective range* [-119, 119]
                    level 2: per-group asymmetric UINT4 on the INT8 codes with integer (UINT8) scales
                    so that the de-quantized level-1 codes always fit in INT8 and the GEMM can run on
                    INT8 tensor cores.  With group_size=-1 QoQ falls back to per-channel asym INT4.
* ``nf4``         – QLoRA NormalFloat-4 with block-wise absmax scaling (block = 64).  Used as the
                    "standard QLoRA" reference point.

Extras from QoQ (QServe App. B.2): activation-aware channel reordering (``perm``) and
output-MSE based weight clipping (``clip`` ratios, searched by :func:`search_clip_ratios`).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .primitives import EPS, fake_quant_activation

NF4_CODEBOOK = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
])


def _groups(W: torch.Tensor, group_size: int):
    O, I = W.shape
    g = I if group_size is None or group_size <= 0 else group_size
    if I % g != 0:
        raise ValueError(f"in_features={I} not divisible by group_size={g}")
    return W.reshape(O, I // g, g), g


def _clip_tensor(clip, O, G, device):
    if clip is None:
        return None
    if not torch.is_tensor(clip):
        clip = torch.full((O, G), float(clip), device=device)
    return clip.to(device).reshape(O, G, 1)


@torch.no_grad()
def quant_rtn(W: torch.Tensor, bits: int, group_size: int = -1, sym: bool = False, clip=None) -> torch.Tensor:
    O, _ = W.shape
    Wg, _ = _groups(W.float(), group_size)
    c = _clip_tensor(clip, O, Wg.shape[1], W.device)
    if sym:
        qmax = 2 ** (bits - 1) - 1
        amax = Wg.abs().amax(-1, keepdim=True)
        if c is not None:
            amax = amax * c
        s = amax.clamp(min=EPS) / qmax
        q = torch.clamp(torch.round(Wg / s), -qmax - 1, qmax)
        Wq = q * s
    else:
        qmax = 2 ** bits - 1
        wmax = Wg.amax(-1, keepdim=True).clamp(min=0)
        wmin = Wg.amin(-1, keepdim=True).clamp(max=0)
        if c is not None:
            wmax, wmin = wmax * c, wmin * c
        s = (wmax - wmin).clamp(min=EPS) / qmax
        z = torch.round(-wmin / s)
        q = torch.clamp(torch.round(Wg / s) + z, 0, qmax)
        Wq = (q - z) * s
    return Wq.reshape(O, -1)


@torch.no_grad()
def quant_progressive(W: torch.Tensor, group_size: int = 128, clip=None, protect: int = 119,
                      return_stats: bool = False):
    """QoQ two-level progressive group quantization (W4 with INT8 intermediates)."""
    Wf = W.float()
    O, _ = Wf.shape
    # level 1: per-channel symmetric INT8 inside the protective range
    s0 = Wf.abs().amax(1, keepdim=True).clamp(min=EPS) / protect
    q8 = torch.clamp(torch.round(Wf / s0), -protect, protect)
    # level 2: per-group asymmetric UINT4 on integer codes, integer UINT8 scale
    q8g, _ = _groups(q8, group_size)
    c = _clip_tensor(clip, O, q8g.shape[1], W.device)
    qmax = q8g.amax(-1, keepdim=True).clamp(min=0)
    qmin = q8g.amin(-1, keepdim=True).clamp(max=0)
    if c is not None:
        qmax, qmin = torch.round(qmax * c), torch.round(qmin * c)
    s1 = torch.ceil((qmax - qmin) / 15.0).clamp(min=1)          # UINT8 integer scale
    z = torch.clamp(torch.round(-qmin / s1), 0, 15)              # UINT4 zero point
    qu4 = torch.clamp(torch.round(q8g / s1) + z, 0, 15)
    q8_hat = (qu4 - z) * s1                                       # INT8 intermediate (fed to INT8 GEMM)
    overflow = int(((q8_hat > 127) | (q8_hat < -128)).sum())
    q8_hat = q8_hat.clamp(-128, 127)
    Wq = (q8_hat.reshape(O, -1)) * s0
    if return_stats:
        return Wq, {"int8_overflow": overflow}
    return Wq


@torch.no_grad()
def quant_nf4(W: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    O, I = W.shape
    Wb = W.float().reshape(-1, block_size)
    absmax = Wb.abs().amax(-1, keepdim=True).clamp(min=EPS)
    xn = Wb / absmax
    cb = NF4_CODEBOOK.to(W.device)
    idx = (xn.unsqueeze(-1) - cb).abs().argmin(-1)
    return (cb[idx] * absmax).reshape(O, I)


@dataclass
class WeightQuantSpec:
    bits: int = 16
    method: str = "rtn"          # rtn | progressive | nf4
    group_size: int = -1
    sym: bool = False

    @property
    def enabled(self) -> bool:
        return self.bits < 16


@torch.no_grad()
def quantize_weight(W: torch.Tensor, spec: WeightQuantSpec, clip=None, perm=None) -> torch.Tensor:
    """Fake-quantize a weight matrix [out, in].  ``perm`` reorders *input* channels before grouping."""
    if not spec.enabled:
        return W.clone()
    Wp = W[:, perm] if perm is not None else W
    if spec.method == "rtn":
        Wq = quant_rtn(Wp, spec.bits, spec.group_size, spec.sym, clip)
    elif spec.method == "progressive":
        if spec.bits != 4:
            raise ValueError("progressive group quantization is defined for 4-bit weights")
        if spec.group_size is None or spec.group_size <= 0:
            Wq = quant_rtn(Wp, 4, -1, sym=False, clip=clip)   # QoQ per-channel W4A8 variant
        else:
            Wq = quant_progressive(Wp, spec.group_size, clip)
    elif spec.method == "nf4":
        Wq = quant_nf4(Wp)
    else:
        raise ValueError(f"unknown weight quant method {spec.method}")
    if perm is not None:
        inv = torch.argsort(perm)
        Wq = Wq[:, inv]
    return Wq.to(W.dtype)


def n_clip_groups(spec: WeightQuantSpec, in_features: int) -> int:
    if spec.method == "nf4" or spec.group_size is None or spec.group_size <= 0:
        return 1
    return in_features // spec.group_size


@torch.no_grad()
def search_clip_ratios(W: torch.Tensor, X: torch.Tensor, spec: WeightQuantSpec, perm=None, a_bits: int = 16,
                       ratios=None, max_tokens: int = 1024) -> torch.Tensor:
    """Grid-search a clip ratio per (output channel, group) minimizing the layer-output MSE
    ||X W^T - Q_A(X) Q_W(W; alpha)^T||^2 (QServe App. B.2.4 / AWQ auto-clip).

    The error is decomposed over input groups (cross-group terms ignored, as in AWQ), which makes
    the per-group search independent.  Returns a tensor [out, n_groups] of ratios.
    """
    if not spec.enabled or spec.method == "nf4":
        return None
    if ratios is None:
        ratios = [1.0 - 0.05 * i for i in range(10)]   # 1.00 ... 0.55
    W = W.float()
    X = X.float().to(W.device)
    if X.shape[0] > max_tokens:
        X = X[torch.randperm(X.shape[0], device=X.device)[:max_tokens]]
    if perm is not None:
        W, X = W[:, perm], X[:, perm]
    Xq = fake_quant_activation(X, a_bits, use_ste=False) if a_bits < 16 else X
    O, I = W.shape
    G = n_clip_groups(spec, I)
    g = I // G
    Xg = X.reshape(-1, G, g)
    Xqg = Xq.reshape(-1, G, g)
    ref = torch.einsum("ngk,ogk->gno", Xg, W.reshape(O, G, g))       # [G, N, O]
    best_err = torch.full((O, G), float("inf"), device=W.device)
    best = torch.ones((O, G), device=W.device)
    spec_np = WeightQuantSpec(spec.bits, spec.method, spec.group_size, spec.sym)
    for r in ratios:
        Wq = quantize_weight(W, spec_np, clip=torch.full((O, G), r, device=W.device))
        out = torch.einsum("ngk,ogk->gno", Xqg, Wq.reshape(O, G, g))
        err = (out - ref).pow(2).mean(1).T                               # [O, G]
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best = torch.where(better, torch.full_like(best, r), best)
    return best
