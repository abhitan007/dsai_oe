"""Linear layer with simulated W/A quantization and an optional LoRA side branch.

    y = Q_A(x) Q_W(W)^T + b  +  (alpha / r) * B A Q_A(x)          (LoRA input quantized by default)

The quantized base weight is frozen and stored de-quantized (simulation).  The quantizer metadata
(spec, channel permutation, clip ratios) is kept so that merged weights can be re-quantized
exactly like the original ones.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import fake_quant_activation
from .weight_quant import WeightQuantSpec, quantize_weight


class QuantLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, w_spec: WeightQuantSpec | None = None,
                 perm: torch.Tensor | None = None, clip: torch.Tensor | None = None, name: str = ""):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False) if bias is not None else None
        self.w_spec = w_spec or WeightQuantSpec()
        self.register_buffer("perm", perm, persistent=False)
        self.register_buffer("clip", clip, persistent=False)
        self.name = name
        # runtime activation quantization
        self.a_bits = 16
        self.a_granularity = "per_token"
        # LoRA
        self.lora_A: nn.Parameter | None = None
        self.lora_B: nn.Parameter | None = None
        self.lora_scaling = 0.0
        self.lora_dropout: nn.Module = nn.Identity()
        self.lora_input_quant = True

    # ------------------------------------------------------------------ quantization helpers
    @torch.no_grad()
    def requantize(self, W: torch.Tensor) -> torch.Tensor:
        """Quantize an arbitrary weight with this layer's quantizer (same perm / clip ratios)."""
        Wf = W.float()
        return quantize_weight(Wf, self.w_spec, clip=self.clip, perm=self.perm)

    # ------------------------------------------------------------------ LoRA
    def add_lora(self, r: int, alpha: float, dropout: float = 0.0, init: str = "default",
                 w_fp: torch.Tensor | None = None, loftq_iters: int = 1, input_quant: bool = True):
        dev = self.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_input_quant = input_quant
        if init == "loftq":
            if w_fp is None:
                raise ValueError("LoftQ initialisation needs the full-precision weight")
            self._loftq_init(w_fp.to(dev).float(), r, loftq_iters)
        elif init != "default":
            raise ValueError(f"unknown LoRA init {init}")

    @torch.no_grad()
    def _loftq_init(self, W: torch.Tensor, r: int, iters: int):
        """LoftQ (Li et al., 2023): alternate  Q_t = Q(W - s B A),  (B, A) = SVD_r(W - Q_t)."""
        s = self.lora_scaling
        BA = torch.zeros_like(W)
        Wq = self.weight.float()
        for _ in range(max(1, iters)):
            Wq = self.requantize(W - BA) if self.w_spec.enabled else W.clone()
            U, S, Vh = torch.linalg.svd(W - Wq, full_matrices=False)
            sq = S[:r].sqrt()
            B = U[:, :r] * sq / math.sqrt(s)
            A = sq[:, None] * Vh[:r] / math.sqrt(s)
            BA = s * (B @ A)
        self.weight.data.copy_(Wq.to(self.weight.dtype))
        self.lora_A.data.copy_(A)
        self.lora_B.data.copy_(B)

    def has_lora(self) -> bool:
        return self.lora_A is not None

    @torch.no_grad()
    def lora_delta(self) -> torch.Tensor:
        return self.lora_scaling * (self.lora_B.float() @ self.lora_A.float())

    @torch.no_grad()
    def merge_lora(self, mode: str = "exact", w_fp: torch.Tensor | None = None):
        """Fold the adapter into the base weight and drop it.

        mode = "exact"        W <- W + sBA                  (full-precision models)
               "requant_deq"  W <- Q(W_q + sBA)             (no FP weights needed at deploy time)
               "requant_fp"   W <- Q(W_fp + sBA)            (adapter trained on quantized base,
                                                             merged into the original FP weights)
        """
        if not self.has_lora():
            return
        delta = self.lora_delta().to(self.weight.device)
        if mode == "exact":
            new = self.weight.float() + delta
        elif mode == "requant_deq":
            new = self.requantize(self.weight.float() + delta)
        elif mode == "requant_fp":
            if w_fp is None:
                raise ValueError("requant_fp needs the full-precision weight")
            new = self.requantize(w_fp.to(self.weight.device).float() + delta)
        else:
            raise ValueError(mode)
        self.weight.data.copy_(new.to(self.weight.dtype))
        self.lora_A = None
        self.lora_B = None
        self.lora_scaling = 0.0
        self.lora_dropout = nn.Identity()

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xq = fake_quant_activation(x, self.a_bits, self.a_granularity) if self.a_bits < 16 else x
        y = F.linear(xq, self.weight, self.bias)
        if self.lora_A is not None:
            xl = xq if self.lora_input_quant else x
            xl = self.lora_dropout(xl).to(self.lora_A.dtype)
            y = y + (F.linear(F.linear(xl, self.lora_A), self.lora_B) * self.lora_scaling).to(y.dtype)
        return y

    def to_linear(self) -> nn.Linear:
        lin = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None,
                        device=self.weight.device, dtype=self.weight.dtype)
        lin.weight.data.copy_(self.weight.data)
        if self.bias is not None:
            lin.bias.data.copy_(self.bias.data)
        return lin

    def extra_repr(self) -> str:
        w = f"W{self.w_spec.bits}:{self.w_spec.method}:g{self.w_spec.group_size}" if self.w_spec.enabled else "W16"
        lora = f", lora_r={self.lora_A.shape[0]}" if self.lora_A is not None else ""
        return f"{self.in_features}->{self.out_features}, {w}, A{self.a_bits}{lora}"
