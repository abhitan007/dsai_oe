"""Simulated integer quantization: QoQ (W4A8KV4) transforms and quantizers, and IntAttention."""
from .apply import quantize_model, set_runtime, unwrap_model
from .attention import AttnQuantState, register_attention
from .quant_linear import QuantLinear

__all__ = ["AttnQuantState", "QuantLinear", "quantize_model", "register_attention", "set_runtime", "unwrap_model"]
