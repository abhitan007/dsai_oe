"""Sanity checks on the real model before launching sweeps.

1. the custom "qft" attention reproduces HF's SDPA attention (logits match)
2. rotation + smoothing + SmoothAttention are function preserving (PPL unchanged)
3. a short PTQ ladder on WikiText-2 behaves as expected (W8A8 ~ FP, W4A8 RTN worse, QoQ recovers)

    python scripts/sanity_check.py --model Qwen/Qwen2.5-0.5B --chunks 16
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qft.config import QuantConfig, make_quant_config  # noqa: E402
from qft.data import get_calibration_batches, wikitext2_test_ids  # noqa: E402
from qft.evaluate import eval_ppl  # noqa: E402
from qft.model_utils import load_model  # noqa: E402
from qft.quant.apply import quantize_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunks", type=int, default=16, help="#WikiText-2 windows used for PPL")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--calib", default="c4")
    ap.add_argument("--calib_nsamples", type=int, default=128)
    ap.add_argument("--calib_seqlen", type=int, default=512)
    ap.add_argument("--presets", nargs="+", default=["fp16", "w8a8", "w4a8", "w4a8_rot", "w4a8_rot_clip",
                                                     "w4a8kv4_rot_clip", "w4a8kv4_smoothattn", "w4a8kv4_g128",
                                                     "qoq", "qoq_quantonly_attn", "qoq_intattn",
                                                     "w4a8kv4_g128_rtn", "intattn_only", "nf4"])
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32

    from transformers import AutoModelForCausalLM
    model, tok = load_model(args.model, args.device, torch.float32)
    ids = torch.randint(0, 1000, (2, 128), device=args.device)
    ref = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, attn_implementation="sdpa")
    ref.to(args.device).eval()
    with torch.no_grad():
        d = (model(input_ids=ids).logits - ref(input_ids=ids).logits).abs().max().item()
    print(f"[check 1] max |logit diff| qft-attention vs sdpa: {d:.2e}  {'OK' if d < 1e-3 else 'MISMATCH'}")
    del ref

    test_ids = wikitext2_test_ids(tok)
    calib = get_calibration_batches(tok, args.calib, args.calib_nsamples, args.calib_seqlen, seed=0)

    def ppl(m):
        return eval_ppl(m, test_ids, args.seq_len, batch_size=1, max_chunks=args.chunks)

    base = ppl(model.to(dtype))
    print(f"[check 2] FP wikitext2 ppl ({args.chunks} x {args.seq_len}): {base:.3f}")
    del model
    model, _ = load_model(args.model, args.device, torch.float32)
    quantize_model(model, QuantConfig(rotation="hadamard", smooth_output=True, smooth_attention=True), calib,
                   log=lambda *_: None)
    p = ppl(model.to(dtype))
    print(f"[check 2] transformed (no quant) ppl: {p:.3f}  (delta {p - base:+.4f}, should be ~0)")
    del model

    print("[check 3] PTQ ladder")
    for name in args.presets:
        model, _ = load_model(args.model, args.device, torch.float32)
        t0 = time.time()
        quantize_model(model, make_quant_config(name), calib, log=lambda *_: None)
        p = ppl(model.to(dtype))
        print(f"   {name:28s} ppl {p:9.3f}   ({time.time() - t0:.0f}s)", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
