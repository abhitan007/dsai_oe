"""Training-free importance proxies for every linear module (RQ2).

For each decoder linear (layer l, type t in {q,k,v,o,gate,up,down}) we compute, in the *same*
transformed coordinate system the quantized model uses:

  w_rel_err    ||Q(W) - W||_F / ||W||_F                              (weight quantization error)
  out_rel_err  ||Q_A(X) Q(W)^T - X W^T|| / ||X W^T||                  (layer output error, calib X)
  taylor       | <G, Q(W) - W> |                                      (1st-order loss change)
  fisher       1/2 * sum F (Q(W) - W)^2,  F = E_batch[G^2]              (2nd-order / OBS-style)
  grad_norm    ||G||_F                                                (how much the task wants W to move)

G is the gradient of the *finetuning-task* loss w.r.t. the full-precision weight.  The analysis
script correlates these proxies with the empirical recovery measured in the rq2 suite to test
whether a cheap proxy can predict which modules are worth adapting.

    python scripts/compute_importance.py --config configs/base.yaml --set quant=qoq_intattn
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qft.config import load_config  # noqa: E402
from qft.data import build_train_data, collate_left_pad, get_calibration_batches  # noqa: E402
from qft.model_utils import iter_linears, linear_name, load_model  # noqa: E402
from qft.quant.apply import quantize_linear_weight, quantize_model  # noqa: E402
from qft.quant.primitives import fake_quant_activation  # noqa: E402
from qft.train import causal_lm_loss  # noqa: E402
from qft.utils import set_seed  # noqa: E402
from run_experiment import parse_sets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--n_batches", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--tag", default="qoq_intattn", help="label used in the output file name")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, parse_sets(args.set))
    set_seed(cfg.seed)
    q = cfg.quant
    out = args.out or os.path.join(cfg.output_dir, "importance", f"{cfg.task}_{args.tag}_seed{cfg.seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    model, tok = load_model(cfg.model, cfg.device, torch.float32)
    calib = get_calibration_batches(tok, q.calib_dataset, q.calib_nsamples, q.calib_seqlen, seed=cfg.seed)
    _, _, stats = quantize_model(model, q, calib, seed=cfg.seed, stop_after_transforms=True)

    # ---------------- task gradients on the transformed full-precision model
    train_ds, _ = build_train_data(tok, cfg.task, cfg.train, cfg.seed)
    for p in model.parameters():
        p.requires_grad_(False)
    lins = {linear_name(i, s): lin for i, s, lin in iter_linears(model)}
    for lin in lins.values():
        lin.weight.requires_grad_(True)
    G = {n: torch.zeros_like(l.weight) for n, l in lins.items()}
    F2 = {n: torch.zeros_like(l.weight) for n, l in lins.items()}
    g = torch.Generator().manual_seed(cfg.seed)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g,
                                         collate_fn=lambda b: collate_left_pad(b, tok.pad_token_id))
    model.train()
    for b, batch in enumerate(loader):
        if b >= args.n_batches:
            break
        batch = {k: v.to(cfg.device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.device == "cuda"):
            loss = causal_lm_loss(model, batch)
        loss.backward()
        for n, l in lins.items():
            G[n] += l.weight.grad
            F2[n] += l.weight.grad.pow(2)
            l.weight.grad = None
    model.eval()
    nb = min(args.n_batches, len(loader))

    # ---------------- proxies
    res = {}
    with torch.no_grad():
        for i, short, lin in iter_linears(model):
            n = linear_name(i, short)
            W = lin.weight.data.float()
            Wq, _, _ = quantize_linear_weight(W, i, short, q, stats)
            dW = Wq - W
            X, _ = stats.get(i, short)
            X = X[:1024].to(W.device).float()
            ref = X @ W.T
            outq = fake_quant_activation(X, q.a_bits, use_ste=False) @ Wq.T
            Gm, Fm = G[n] / nb, F2[n] / nb
            res[n] = {
                "layer": i, "type": short, "numel": W.numel(),
                "w_rel_err": float(dW.norm() / W.norm()),
                "out_rel_err": float((outq - ref).norm() / ref.norm()),
                "taylor": float((Gm * dW).sum().abs()),
                "fisher": float(0.5 * (Fm * dW.pow(2)).sum()),
                "grad_norm": float(Gm.norm()),
            }
    summary = {}
    for key in ("type", "layer"):
        agg = {}
        for r in res.values():
            a = agg.setdefault(str(r[key]), {m: 0.0 for m in ("taylor", "fisher", "grad_norm")} | {"n": 0})
            for m in ("taylor", "fisher", "grad_norm"):
                a[m] += r[m]
            a["n"] += 1
        summary[f"by_{key}"] = agg
    with open(out, "w") as f:
        json.dump({"config": {"tag": args.tag, "quant": q.__dict__, "task": cfg.task, "model": cfg.model, "seed": cfg.seed},
                   "modules": res, "summary": summary}, f, indent=1)
    print(f"[importance] wrote {out}")
    for t, a in summary["by_type"].items():
        print(f"   {t:10s} taylor {a['taylor']:.3e}  fisher {a['fisher']:.3e}  grad_norm {a['grad_norm']:.3e}")


if __name__ == "__main__":
    main()
