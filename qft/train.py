"""Minimal, transparent LoRA training loop (single GPU)."""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import TrainConfig
from .data import collate_left_pad, infinite_loader
from .lora import lora_parameters


def _lr_lambda(cfg: TrainConfig):
    warm = max(1, int(cfg.warmup_ratio * cfg.max_steps))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, cfg.max_steps - warm)
        if cfg.scheduler == "cosine":
            return 0.5 * (1 + math.cos(math.pi * prog))
        if cfg.scheduler == "linear":
            return 1 - prog
        return 1.0
    return f


def _to(batch, dev):
    return {k: v.to(dev) for k, v in batch.items()}


def causal_lm_loss(model, batch: dict, chunk: int = 1024) -> torch.Tensor:
    """Next-token cross-entropy, identical to HF's loss (shifted labels, -100 ignored, mean over targets),
    but memory-efficient: the LM head is applied only at labelled positions, in chunks, under activation
    checkpointing.  With Qwen's 152k vocabulary the naive [B, L, V] fp32 logits (+ HF's upcast and
    log-softmax copies) cost ~4 GB per 16x438 batch; here the peak is ~chunk x V x 4 B (~0.6 GB)."""
    h = model.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"),
                    position_ids=batch.get("position_ids"), use_cache=False).last_hidden_state
    tgt = batch["labels"][:, 1:]
    sel = tgt != -100
    hs, ts = h[:, :-1][sel], tgt[sel]
    head = model.lm_head

    def ce_sum(hc, tc):
        return F.cross_entropy(head(hc).float(), tc, reduction="sum")

    total = h.new_zeros((), dtype=torch.float32)
    for s in range(0, ts.numel(), chunk):
        hc, tc = hs[s:s + chunk], ts[s:s + chunk]
        total = total + (checkpoint(ce_sum, hc, tc, use_reentrant=False) if torch.is_grad_enabled()
                         else ce_sum(hc, tc))
    return total / max(1, ts.numel())


@torch.no_grad()
def eval_loss(model, dataset, pad_id: int, batch_size: int = 16, max_batches: int | None = None) -> float:
    model.eval()
    dev = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         collate_fn=lambda b: collate_left_pad(b, pad_id))
    tot, n = 0.0, 0
    for j, batch in enumerate(loader):
        if max_batches is not None and j >= max_batches:
            break
        batch = _to(batch, dev)
        loss = causal_lm_loss(model, batch)
        ntok = int((batch["labels"][:, 1:] != -100).sum())
        tot += float(loss) * ntok
        n += ntok
    return tot / max(1, n)


def train_lora(model, train_ds, cfg: TrainConfig, pad_id: int, seed: int, val_ds=None, log=print) -> dict:
    dev = next(model.parameters()).device
    params = lora_parameters(model)
    if not params:
        raise RuntimeError("no trainable LoRA parameters")
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(cfg))
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    it = infinite_loader(train_ds, cfg.batch_size, pad_id, seed)
    history, val_hist = [], []
    t0 = time.time()
    run_loss, run_n = 0.0, 0
    for step in range(cfg.max_steps):
        model.train()
        step_loss = 0.0
        for _ in range(cfg.grad_accum):
            batch = _to(next(it), dev)
            loss = causal_lm_loss(model, batch) / cfg.grad_accum
            loss.backward()
            step_loss += float(loss)
        if not math.isfinite(step_loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        run_loss += step_loss
        run_n += 1
        if (step + 1) % cfg.log_every == 0 or step == cfg.max_steps - 1:
            avg = run_loss / run_n
            history.append({"step": step + 1, "loss": avg, "lr": sched.get_last_lr()[0], "grad_norm": float(gnorm)})
            el = time.time() - t0
            log(f"[train] step {step + 1}/{cfg.max_steps} loss {avg:.4f} gnorm {float(gnorm):.3f} "
                f"lr {sched.get_last_lr()[0]:.2e} ({el / (step + 1):.2f}s/it)")
            run_loss, run_n = 0.0, 0
        if val_ds is not None and cfg.val_every > 0 and (step + 1) % cfg.val_every == 0:
            vl = eval_loss(model, val_ds, pad_id, max_batches=cfg.val_batches)
            val_hist.append({"step": step + 1, "val_loss": vl})
            log(f"[train] step {step + 1} val_loss {vl:.4f}")
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_disable()
    model.eval()
    return {"history": history, "val_history": val_hist, "train_seconds": time.time() - t0,
            "final_train_loss": history[-1]["loss"] if history else None}
