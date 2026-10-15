"""One experiment = (pipeline, quantization config, LoRA config, task, seed) -> result.json.

Pipelines (the *order* in which quantization and finetuning are applied):

  fp_eval       full-precision model, no finetuning                           (reference)
  fp_ft         full-precision LoRA finetuning                                (upper bound)
  ptq_only      PTQ (QoQ / IntAttention ...) without finetuning               (lower bound)
  ptq_then_ft   PTQ first, then LoRA on the frozen quantized model            (QLoRA-style;
                quantizers active in the forward pass -> quantization-aware LoRA)
  ft_then_ptq   LoRA on the FP model, merge, then PTQ of the merged model     (classic deploy)
"""
from __future__ import annotations

import json
import os
import platform
import time

import torch

from .config import RunConfig, to_plain
from .data import build_train_data, c4_val_ids, get_calibration_batches, wikitext2_test_ids
from .evaluate import eval_gsm8k, eval_lm_harness, eval_ppl
from .lora import add_lora, lora_state_dict, lora_stats, merge_lora
from .model_utils import count_params, load_model
from .quant.apply import FP_RUNTIME, quantize_model, set_runtime, unwrap_model
from .train import eval_loss, train_lora
from .utils import Logger, set_seed

PIPELINES = ("fp_eval", "fp_ft", "ptq_only", "ptq_then_ft", "ft_then_ptq")
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def run_dir(cfg: RunConfig) -> str:
    return os.path.join(cfg.output_dir, cfg.name, f"seed{cfg.seed}")


def run_experiment(cfg: RunConfig, force: bool = False) -> dict:
    if cfg.pipeline not in PIPELINES:
        raise ValueError(f"pipeline must be one of {PIPELINES}")
    out_dir = run_dir(cfg)
    res_path = os.path.join(out_dir, "result.json")
    if os.path.exists(res_path) and not force:
        print(f"[run] {res_path} exists, skipping (use --force to rerun)")
        with open(res_path) as f:
            return json.load(f)
    os.makedirs(out_dir, exist_ok=True)
    log = Logger(os.path.join(out_dir, "log.txt"))
    log(f"[run] {cfg.name} seed={cfg.seed} pipeline={cfg.pipeline}")
    log("[run] config: " + json.dumps(to_plain(cfg), default=str))
    set_seed(cfg.seed)
    t_start = time.time()
    dtype = DTYPES[cfg.dtype]
    if cfg.device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model, tok = load_model(cfg.model, cfg.device, dtype=torch.float32)   # transforms in fp32
    result: dict = {"config": to_plain(cfg), "metrics": {}, "env": _env()}
    q = cfg.quant
    needs_ft = cfg.pipeline in ("fp_ft", "ptq_then_ft", "ft_then_ptq") and cfg.lora.enabled

    def calib():
        return get_calibration_batches(tok, q.calib_dataset, q.calib_nsamples, q.calib_seqlen, seed=cfg.seed)

    train_ds = val_ds = None
    if needs_ft:
        train_ds, val_ds = build_train_data(tok, cfg.task, cfg.train, cfg.seed)
        log(f"[run] train examples: {len(train_ds)}, val examples: {len(val_ds)}")

    fp_weights = None
    # ------------------------------------------------------------------ stage 1
    if cfg.pipeline in ("fp_eval", "fp_ft", "ft_then_ptq"):
        from .config import QuantConfig
        quantize_model(model, QuantConfig(), log=log)                     # wrap only (no quantization)
    else:
        keep = cfg.lora.init == "loftq" or cfg.merge_for_eval == "requant_fp"
        art, fp_weights, _ = quantize_model(model, q, calib(), seed=cfg.seed, keep_fp_weights=keep, log=log)
        result["ptq"] = {k: v for k, v in art.items() if k != "weight_rel_err"}
        result["ptq_weight_rel_err"] = art.get("weight_rel_err")
    model.to(dtype)

    # ------------------------------------------------------------------ stage 2: finetuning
    if needs_ft:
        info = add_lora(model, cfg.lora, fp_weights)
        result["lora"] = info
        log(f"[run] LoRA: {info['lora_modules']} modules, {info['lora_params'] / 1e6:.2f}M params "
            f"({100 * info['lora_params'] / count_params(model):.2f}% of total)")
        if cfg.pipeline == "ptq_then_ft":
            set_runtime(model, {**q.runtime(), **cfg.train_runtime})
            log(f"[run] training runtime: {({**q.runtime(), **cfg.train_runtime})}")
        if cfg.eval.task_val_loss:
            result["metrics"]["task_val_loss_before"] = eval_loss(model, val_ds, tok.pad_token_id, max_batches=16)
        result["train"] = train_lora(model, train_ds, cfg.train, tok.pad_token_id, cfg.seed, val_ds, log)
        result["lora_update_norms"] = lora_stats(model)
        if cfg.save_adapter:
            torch.save(lora_state_dict(model), os.path.join(out_dir, "adapter.pt"))

    # ------------------------------------------------------------------ stage 3: deploy form
    if cfg.pipeline == "ft_then_ptq":
        model.float()
        merge_lora(model, "exact")
        unwrap_model(model)
        art, _, _ = quantize_model(model, q, calib(), seed=cfg.seed, log=log)
        result["ptq"] = {k: v for k, v in art.items() if k != "weight_rel_err"}
        model.to(dtype)
    elif cfg.pipeline == "ptq_then_ft":
        set_runtime(model, q.runtime())                                   # deploy-time runtime
        if cfg.merge_for_eval != "none":
            merge_lora(model, cfg.merge_for_eval, fp_weights)
            log(f"[run] merged adapters ({cfg.merge_for_eval})")
    elif cfg.pipeline == "fp_ft":
        set_runtime(model, FP_RUNTIME)

    # ------------------------------------------------------------------ stage 4: evaluation
    result["metrics"].update(evaluate_all(model, tok, cfg, val_ds, finetuned=needs_ft, log=log))
    result["total_seconds"] = time.time() - t_start
    with open(res_path, "w") as f:
        json.dump(result, f, indent=1, default=float)
    log(f"[run] done in {result['total_seconds'] / 60:.1f} min -> {res_path}")
    log(f"[run] metrics: { {k: v for k, v in result['metrics'].items() if not isinstance(v, (list, dict))} }")
    return result


def evaluate_all(model, tok, cfg: RunConfig, val_ds=None, finetuned: bool = False, log=print) -> dict:
    ev = cfg.eval
    m: dict = {}
    if val_ds is not None and ev.task_val_loss:
        m["task_val_loss"] = eval_loss(model, val_ds, tok.pad_token_id, max_batches=16)
        log(f"[eval] task val loss {m['task_val_loss']:.4f}")
    if ev.wikitext2:
        m["wikitext2_ppl"] = eval_ppl(model, wikitext2_test_ids(tok), ev.ppl_seq_len, ev.ppl_batch_size,
                                      ev.ppl_max_chunks)
        log(f"[eval] wikitext2 ppl {m['wikitext2_ppl']:.3f}")
    if ev.c4:
        m["c4_ppl"] = eval_ppl(model, c4_val_ids(tok, ev.c4_nsamples, ev.ppl_seq_len), ev.ppl_seq_len,
                               ev.ppl_batch_size, ev.ppl_max_chunks)
        log(f"[eval] c4 ppl {m['c4_ppl']:.3f}")
    if ev.gsm8k:
        shots = ev.gsm8k_fewshot
        if shots == "auto":
            shots = 0 if (finetuned and cfg.task == "gsm8k") else 8
        g = eval_gsm8k(model, tok, int(shots), ev.gsm8k_limit, ev.gsm8k_batch_size, ev.gsm8k_max_new_tokens, log)
        m.update(g)
    if ev.lm_eval_tasks:
        m["lm_eval"] = eval_lm_harness(model, tok, ev.lm_eval_tasks, ev.lm_eval_limit, ev.lm_eval_batch_size, log)
    if ev.runtime_sweep:
        m["runtime_sweep"] = runtime_sweep(model, tok, cfg, val_ds, log)
    return m


def runtime_sweep(model, tok, cfg: RunConfig, val_ds=None, log=print) -> list:
    """Re-evaluate the *same* weights/adapters under different runtime quantization settings."""
    from .config import RUNTIME_FIELDS
    deploy = cfg.quant.runtime()
    ids = wikitext2_test_ids(tok)
    out = []
    for ov in cfg.eval.runtime_sweep:
        bad = set(ov) - RUNTIME_FIELDS
        if bad:
            raise KeyError(f"runtime_sweep entries may only set runtime fields; got {bad}")
        set_runtime(model, {**deploy, **ov})
        r = {"override": ov,
             "wikitext2_ppl": eval_ppl(model, ids, cfg.eval.ppl_seq_len, cfg.eval.ppl_batch_size,
                                       cfg.eval.ppl_max_chunks)}
        if val_ds is not None:
            r["task_val_loss"] = eval_loss(model, val_ds, tok.pad_token_id, max_batches=16)
        log(f"[eval] runtime sweep {ov}: {r}")
        out.append(r)
    set_runtime(model, deploy)
    return out


def _env():
    import transformers
    e = {"torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version()}
    if torch.cuda.is_available():
        e["gpu"] = torch.cuda.get_device_name(0)
    return e
