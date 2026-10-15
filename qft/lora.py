"""Attach / merge / save LoRA adapters on QuantLinear modules."""
from __future__ import annotations

import torch

from .config import LoraConfig
from .model_utils import iter_linears, layers, linear_name, parse_layer_spec
from .quant.quant_linear import QuantLinear


def add_lora(model, cfg: LoraConfig, fp_weights: dict | None = None) -> dict:
    """Freeze everything, then add trainable LoRA adapters to the selected linears."""
    for p in model.parameters():
        p.requires_grad_(False)
    sel_layers = set(parse_layer_spec(cfg.layers, len(layers(model))))
    targets = set(cfg.targets)
    unknown = targets - {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if unknown:
        raise ValueError(f"unknown LoRA targets {unknown}")
    n_mod, n_par = 0, 0
    for i, short, lin in iter_linears(model):
        if i not in sel_layers or short not in targets:
            continue
        assert isinstance(lin, QuantLinear)
        w_fp = fp_weights.get(linear_name(i, short)) if fp_weights else None
        lin.add_lora(cfg.r, cfg.alpha, cfg.dropout, cfg.init, w_fp=w_fp, loftq_iters=cfg.loftq_iters,
                     input_quant=cfg.input_quant)
        n_mod += 1
        n_par += lin.lora_A.numel() + lin.lora_B.numel()
    return {"lora_modules": n_mod, "lora_params": n_par}


def lora_parameters(model):
    return [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]


@torch.no_grad()
def merge_lora(model, mode: str = "exact", fp_weights: dict | None = None):
    for i, short, lin in iter_linears(model):
        if isinstance(lin, QuantLinear) and lin.has_lora():
            w_fp = fp_weights.get(linear_name(i, short)) if fp_weights else None
            lin.merge_lora(mode, w_fp)


def lora_state_dict(model) -> dict:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n}


@torch.no_grad()
def lora_stats(model) -> dict:
    """Per-module norm of the learned update ||s B A||_F (relative to ||W||_F)."""
    out = {}
    for i, short, lin in iter_linears(model):
        if isinstance(lin, QuantLinear) and lin.has_lora():
            d = lin.lora_delta()
            out[linear_name(i, short)] = float(d.norm() / lin.weight.float().norm().clamp(min=1e-12))
    return out
