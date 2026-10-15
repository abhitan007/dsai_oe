"""Calibration: per-channel activation statistics and token samples for every linear input,
plus per-channel max|K| (post-RoPE) for SmoothAttention."""
from __future__ import annotations

import torch

from ..model_utils import layers

# linears that share the same input tensor share one statistics entry
INPUT_GROUP = {"q_proj": "attn_in", "k_proj": "attn_in", "v_proj": "attn_in", "o_proj": "o_proj",
               "gate_proj": "mlp_in", "up_proj": "mlp_in", "down_proj": "down_proj"}
_HOOKED = {"q_proj": "attn_in", "o_proj": "o_proj", "gate_proj": "mlp_in", "down_proj": "down_proj"}


class CalibStats:
    def __init__(self):
        self.amax: dict[str, torch.Tensor] = {}
        self.samples: dict[str, torch.Tensor] = {}
        self.k_amax: dict[int, torch.Tensor] = {}

    @staticmethod
    def key(i: int, short: str) -> str:
        return f"layers.{i}.{INPUT_GROUP[short]}"

    def get(self, i: int, short: str):
        k = self.key(i, short)
        return self.samples[k], self.amax[k]

    def rescale_input(self, key: str, s: torch.Tensor):
        """Inputs of a smoothed layer were divided by s (key: 'layers.i.o_proj' / 'layers.i.down_proj')."""
        self.samples[key] = self.samples[key] / s.to(self.samples[key].device)[None, :]
        self.amax[key] = self.amax[key] / s.to(self.amax[key].device)


@torch.no_grad()
def collect_calibration(model, batches, tokens_per_batch: int = 64, seed: int = 0) -> CalibStats:
    """Run the (full-precision) model on calibration batches and record statistics.

    batches: list of LongTensor [B, L] (packed text, no padding).
    tokens_per_batch: number of token rows sampled per batch for MSE-based searches.
    """
    stats = CalibStats()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    state = {"idx": None}
    handles = []

    def make_hook(name):
        def hook(mod, inp, out):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1]).float()
            am = x.abs().amax(0)
            stats.amax[name] = am if name not in stats.amax else torch.maximum(stats.amax[name], am)
            smp = x[state["idx"].to(x.device)]
            stats.samples.setdefault(name, []).append(smp)
        return hook

    for i, layer in enumerate(layers(model)):
        for short, grp in _HOOKED.items():
            mod = layer.self_attn if short in ("q_proj", "o_proj") else layer.mlp
            handles.append(getattr(mod, short).register_forward_hook(make_hook(f"layers.{i}.{grp}")))
        layer.self_attn.qft_attn.calibrate = True
        layer.self_attn.qft_attn.k_amax = None

    dev = next(model.parameters()).device
    try:
        for ids in batches:
            n_tok = ids.numel()
            state["idx"] = torch.randperm(n_tok, generator=gen)[: min(tokens_per_batch, n_tok)]
            model(input_ids=ids.to(dev), use_cache=False)
    finally:
        for h in handles:
            h.remove()
        for i, layer in enumerate(layers(model)):
            st = layer.self_attn.qft_attn
            st.calibrate = False
            stats.k_amax[i] = st.k_amax
            st.k_amax = None
    stats.samples = {k: torch.cat(v, 0) for k, v in stats.samples.items()}
    return stats
