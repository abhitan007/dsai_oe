"""Model loading and structural helpers (Qwen2 / Llama / Mistral style decoder-only models)."""
from __future__ import annotations

import re

import torch
import torch.nn as nn

ATTN_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_LINEARS = ("gate_proj", "up_proj", "down_proj")
ALL_LINEARS = ATTN_LINEARS + MLP_LINEARS
SUPPORTED_ARCHS = ("qwen2", "llama", "mistral")


def load_model(name: str, device: str = "cuda", dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .quant.attention import ATTN_IMPL_NAME, register_attention

    register_attention()
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation=ATTN_IMPL_NAME)
    if model.config.model_type not in SUPPORTED_ARCHS:
        raise ValueError(f"model_type={model.config.model_type} not supported (need one of {SUPPORTED_ARCHS}); "
                         "models with q/k-norm (e.g. Qwen3) break the SmoothAttention weight folding")
    model.to(device)
    model.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return model, tok


def layers(model):
    return model.model.layers


def get_submodule_parent(layer: nn.Module, short: str):
    parent = layer.self_attn if short in ATTN_LINEARS else layer.mlp
    return parent, short


def get_linear(layer: nn.Module, short: str) -> nn.Module:
    parent, attr = get_submodule_parent(layer, short)
    return getattr(parent, attr)


def set_linear(layer: nn.Module, short: str, module: nn.Module):
    parent, attr = get_submodule_parent(layer, short)
    setattr(parent, attr, module)


def iter_linears(model):
    for i, layer in enumerate(layers(model)):
        for short in ALL_LINEARS:
            yield i, short, get_linear(layer, short)


def linear_name(i: int, short: str) -> str:
    return f"layers.{i}.{short}"


def parse_layer_spec(spec, n_layers: int):
    """'all' | 'first_third' | 'middle_third' | 'last_third' | '0-7' | '0,3,5' | [0, 1, 2]"""
    if spec is None or spec == "all":
        return list(range(n_layers))
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec]
    third = n_layers // 3
    bands = {
        "first_third": list(range(0, third)),
        "middle_third": list(range(third, n_layers - third)),
        "last_third": list(range(n_layers - third, n_layers)),
        "first_half": list(range(0, n_layers // 2)),
        "last_half": list(range(n_layers // 2, n_layers)),
    }
    if spec in bands:
        return bands[spec]
    out = []
    for part in str(spec).split(","):
        if re.fullmatch(r"\d+-\d+", part.strip()):
            a, b = map(int, part.split("-"))
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return out


def attn_geometry(model):
    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    return n_heads, n_kv, head_dim


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))
