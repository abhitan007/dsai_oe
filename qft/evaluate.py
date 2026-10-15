"""Evaluation: perplexity (WikiText-2 / C4), GSM8K exact match, lm-evaluation-harness zero-shot."""
from __future__ import annotations

import re
import time

import torch
import torch.nn.functional as F

from .data import _gsm8k, format_gsm8k, gsm8k_prompt


@torch.no_grad()
def eval_ppl(model, ids: torch.Tensor, seq_len: int = 2048, batch_size: int = 4, max_chunks: int | None = None,
             logit_chunk: int = 512) -> float:
    """Non-overlapping-window perplexity (GPTQ protocol).  The LM head is applied in chunks to
    avoid materializing [B, L, vocab] fp32 logits."""
    model.eval()
    dev = next(model.parameters()).device
    n = ids.numel() // seq_len
    if max_chunks is not None:
        n = min(n, max_chunks)
    chunks = ids[: n * seq_len].view(n, seq_len)
    nll, count = 0.0, 0
    for s in range(0, n, batch_size):
        x = chunks[s:s + batch_size].to(dev)
        h = model.model(input_ids=x, use_cache=False).last_hidden_state
        for c in range(0, seq_len - 1, logit_chunk):
            e = min(c + logit_chunk, seq_len - 1)
            logits = model.lm_head(h[:, c:e]).float()
            tgt = x[:, c + 1:e + 1]
            nll += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum").item()
            count += tgt.numel()
    return float(torch.exp(torch.tensor(nll / count)))


# --------------------------------------------------------------------------------------
# GSM8K
# --------------------------------------------------------------------------------------
_NUM = re.compile(r"-?[\d,]*\.?\d+")


def _to_number(s: str):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def extract_answer(text: str):
    text = text.split("Question:")[0]
    m = re.search(r"####\s*(-?[\d,\.\$]+)", text)
    if m:
        return _to_number(m.group(1))
    nums = _NUM.findall(text)
    return _to_number(nums[-1]) if nums else None


def gold_answer(ans: str):
    return _to_number(ans.split("####")[-1])


@torch.no_grad()
def eval_gsm8k(model, tok, fewshot: int = 0, limit: int | None = None, batch_size: int = 64,
               max_new_tokens: int = 256, log=print) -> dict:
    model.eval()
    model.config.use_cache = True
    dev = next(model.parameters()).device
    test = _gsm8k("test")
    n = len(test) if limit is None else min(limit, len(test))
    prefix = ""
    if fewshot > 0:
        train = _gsm8k("train")
        prefix = "".join(format_gsm8k(train[i]["question"], train[i]["answer"]) + "\n\n" for i in range(fewshot))
    prompts = [prefix + gsm8k_prompt(test[i]["question"]) for i in range(n)]
    golds = [gold_answer(test[i]["answer"]) for i in range(n)]
    # sort by length for efficient batching
    order = sorted(range(n), key=lambda i: len(prompts[i]))
    preds = [None] * n
    outs = [None] * n
    t0 = time.time()
    for s in range(0, n, batch_size):
        idx = order[s:s + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(dev)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                             top_k=None, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                             stop_strings=["Question:"], tokenizer=tok)
        texts = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            outs[i] = t
            preds[i] = extract_answer(t)
    model.config.use_cache = False
    correct = [p is not None and g is not None and abs(p - g) < 1e-4 for p, g in zip(preds, golds)]
    acc = sum(correct) / n
    log(f"[eval] gsm8k ({fewshot}-shot, n={n}) acc={acc:.4f} ({time.time() - t0:.0f}s)")
    samples = [{"q": test[i]["question"], "gold": golds[i], "pred": preds[i], "out": outs[i]} for i in range(min(n, 20))]
    return {"gsm8k_acc": acc, "gsm8k_n": n, "gsm8k_fewshot": fewshot, "gsm8k_samples": samples,
            "gsm8k_correct": correct}


# --------------------------------------------------------------------------------------
# lm-evaluation-harness
# --------------------------------------------------------------------------------------
def eval_lm_harness(model, tok, tasks, limit=None, batch_size: int = 32, log=print) -> dict:
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        log("[eval] lm_eval not installed; skipping zero-shot tasks")
        return {}
    model.eval()
    model.config.use_cache = False
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size, max_length=2048)
    res = lm_eval.simple_evaluate(model=lm, tasks=list(tasks), num_fewshot=0, limit=limit, log_samples=False,
                                  bootstrap_iters=0)
    out = {}
    for t in tasks:
        r = res["results"].get(t, {})
        acc = r.get("acc_norm,none", r.get("acc,none"))
        if acc is not None:
            out[t] = float(acc)
    if out:
        out["avg"] = sum(out.values()) / len(out)
    log(f"[eval] lm-eval: {out}")
    return out
