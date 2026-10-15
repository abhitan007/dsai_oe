"""Aggregate all result.json files into tables, importance rankings and figures.

    python scripts/analyze.py --results results            # writes results/REPORT.md, summary.csv, figures/

Everything is computed from whatever runs exist; sections with missing data are skipped.
Means are over seeds; +- is the sample standard deviation (n shown).
"""
import argparse
import glob
import itertools
import json
import math
import os
from collections import defaultdict

import numpy as np

# ------------------------------------------------------------------ plotting style (reference palette)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3 (validated all-pairs)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
        "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
    })
    return plt


# ------------------------------------------------------------------ loading
METRICS = ["gsm8k_acc", "wikitext2_ppl", "c4_ppl", "lm_eval_avg", "task_val_loss"]
HIGHER_BETTER = {"gsm8k_acc": True, "wikitext2_ppl": False, "c4_ppl": False, "lm_eval_avg": True,
                 "task_val_loss": False}


def load(results_dir):
    rows = []
    for path in glob.glob(os.path.join(results_dir, "**", "result.json"), recursive=True):
        with open(path) as f:
            r = json.load(f)
        cfg, m = r["config"], r["metrics"]
        tags = cfg.get("tags") or {}
        name = cfg["name"]
        suite = tags.get("suite", name.split("/")[0] if "/" in name else "adhoc")
        run = tags.get("run", name.split("/", 1)[-1])
        row = {"suite": suite, "run": run, "seed": cfg["seed"], "pipeline": cfg["pipeline"], "cfg": cfg,
               "tags": tags, "lora_params": (r.get("lora") or {}).get("lora_params"),
               "train_seconds": (r.get("train") or {}).get("train_seconds"),
               "runtime_sweep": m.get("runtime_sweep"), "lm_eval": m.get("lm_eval") or {}}
        for k in METRICS:
            row[k] = m.get(k)
        if row["lm_eval"]:
            row["lm_eval_avg"] = row["lm_eval"].get("avg")
        rows.append(row)
    return rows


def agg(rows, metric):
    vals = [r[metric] for r in rows if r.get(metric) is not None]
    if not vals:
        return None
    a = np.array(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n": len(a)}


def by_run(rows, suite):
    d = defaultdict(list)
    for r in rows:
        if r["suite"] == suite:
            d[r["run"]].append(r)
    return d


def fmt(a, metric, pct=None):
    if a is None:
        return "-"
    pct = metric in ("gsm8k_acc", "lm_eval_avg") if pct is None else pct
    s = 100 if pct else 1
    dec = 2 if pct or "ppl" in metric else 4
    txt = f"{a['mean'] * s:.{dec}f}"
    if a["n"] > 1:
        txt += f" ± {a['std'] * s:.{dec}f}"
    return txt + (f" (n={a['n']})" if a["n"] > 1 else "")


def table(header, lines):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(x) for x in ln) + " |" for ln in lines]
    return "\n".join(out)


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3:
        return float("nan"), float("nan")
    r = spearmanr(x, y)
    return float(r.statistic if hasattr(r, "statistic") else r.correlation), float(r.pvalue)


def kendall(x, y):
    from scipy.stats import kendalltau
    if len(x) < 3:
        return float("nan"), float("nan")
    r = kendalltau(x, y)
    return float(r.statistic if hasattr(r, "statistic") else r.correlation), float(r.pvalue)


# ------------------------------------------------------------------ sections
def rq0(rows, out, figs):
    runs = by_run(rows, "rq0_ptq_ladder")
    if not runs:
        return
    order = ["fp16", "w8a8", "w4a16_g128", "nf4", "w4a8", "w4a8_rot", "w4a8_rot_clip", "w4a8kv4_rot_clip",
             "w4a8kv4_smoothattn", "w4a8kv4_g128", "qoq", "qoq_quantonly_attn", "qoq_intattn",
             "w4a8kv4_g128_rtn", "w4a8kv4_g128_rtn_intattn", "intattn_only", "quantonly_attn_only"]
    lines = []
    for name in order + sorted(set(runs) - set(order)):
        if name in runs:
            rs = runs[name]
            lines.append([name] + [fmt(agg(rs, m), m) for m in ("wikitext2_ppl", "gsm8k_acc", "lm_eval_avg")])
    out.append("## RQ0 — PTQ ladder (no finetuning)\n")
    out.append("GSM8K is 8-shot here (the base model is not instruction-tuned).\n")
    out.append(table(["config", "WikiText-2 PPL ↓", "GSM8K 8-shot (%) ↑", "zero-shot avg (%) ↑"], lines))
    out.append("")


def rq1(rows, out, figs):
    runs = by_run(rows, "rq1_pipelines")
    if not runs:
        return
    ref = agg(runs.get("fp_ft", []), "gsm8k_acc")
    lines = []
    ranked = []
    for name, rs in runs.items():
        a = agg(rs, "gsm8k_acc")
        gap = f"{(a['mean'] - ref['mean']) * 100:+.2f}" if (a and ref and rs[0]["pipeline"] != "fp_eval"
                                                           and rs[0]["pipeline"] != "ptq_only") else "-"
        lines.append([name, rs[0]["pipeline"], fmt(a, "gsm8k_acc"), gap, fmt(agg(rs, "wikitext2_ppl"), "wikitext2_ppl"),
                      fmt(agg(rs, "lm_eval_avg"), "lm_eval_avg"), fmt(agg(rs, "task_val_loss"), "task_val_loss")])
        if a and rs[0]["pipeline"] not in ("fp_eval", "ptq_only"):
            ranked.append((a["mean"], name))
    lines.sort(key=lambda ln: ln[0])
    out.append("## RQ1 — Order of operations (quantize ↔ finetune)\n")
    out.append("Deployment target for every row: QoQ W4A8KV4-g128 + IntAttention (except `fp_*`, `qlora_nf4`). "
               "`gap` = GSM8K difference to full-precision LoRA (pp).\n")
    out.append(table(["run", "pipeline", "GSM8K (%)", "gap vs fp_ft", "WikiText-2 PPL", "zero-shot avg (%)",
                      "task val loss"], lines))
    ranked.sort(reverse=True)
    out.append("\n**Importance order of pipelines (GSM8K, best first):** " +
               " > ".join(f"`{n}` ({m * 100:.1f})" for m, n in ranked) + "\n")
    _bar_figure(figs, "rq1_pipelines.png", [(n, runs[n]) for _, n in ranked], "gsm8k_acc",
                "GSM8K accuracy by pipeline (mean ± std over seeds)", ref)


def _bar_figure(figs, fname, items, metric, title, ref=None):
    if not items:
        return
    plt = _plt()
    names = [n for n, _ in items]
    a = [agg(rs, metric) for _, rs in items]
    means = [x["mean"] * 100 for x in a]
    errs = [x["std"] * 100 for x in a]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(names) + 1.2))
    y = np.arange(len(names))[::-1]
    ax.barh(y, means, xerr=errs, color=SERIES[0], height=0.6, error_kw={"ecolor": INK2, "elinewidth": 1})
    if ref:
        ax.axvline(ref["mean"] * 100, color=SERIES[1], linewidth=1.5, linestyle="--")
        ax.text(ref["mean"] * 100, y.max() + 0.6, " FP LoRA", color=INK2, fontsize=8, va="bottom")
    for yi, m, e in zip(y, means, errs):
        ax.text(m + e, yi, f"  {m:.1f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, names)
    ax.set_xlabel("GSM8K accuracy (%)" if metric == "gsm8k_acc" else metric)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    lo = min(m - e for m, e in zip(means, errs))
    ax.set_xlim(max(0, lo - 5), max(m + e for m, e in zip(means, errs)) + 6)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, fname), dpi=160)
    plt.close(fig)


def rq2(rows, out, figs, importance_dir):
    runs = by_run(rows, "rq2_modules")
    if not runs:
        return
    out.append("## RQ2 — Which modules / layers to finetune\n")
    types = ["q", "k", "v", "o", "gate", "up", "down"]
    orders = {}
    for base, label in (("q", "quantized base (QoQ+IntAttention, quantization-aware LoRA)"),
                        ("fp", "full-precision base")):
        full = agg(runs.get(f"{base}_all", []), "gsm8k_acc")
        if full is None:
            continue
        lines = []
        single, loo = {}, {}
        for t in types:
            s = agg(runs.get(f"{base}_only_{t}", []), "gsm8k_acc")
            d = agg(runs.get(f"{base}_drop_{t}", []), "gsm8k_acc")
            params = (runs.get(f"{base}_only_{t}") or [{}])[0].get("lora_params")
            if s:
                single[t] = s["mean"]
            if d:
                loo[t] = full["mean"] - d["mean"]
            lines.append([t, fmt(s, "gsm8k_acc"), f"{params / 1e6:.2f}M" if params else "-",
                          f"{s['mean'] * 100 / (params / 1e6):.1f}" if (s and params) else "-",
                          fmt(d, "gsm8k_acc"), f"{loo[t] * 100:+.2f}" if t in loo else "-"])
        out.append(f"### {label}\nAll modules: {fmt(full, 'gsm8k_acc')}\n")
        out.append(table(["module", "only this module (%)", "LoRA params", "acc per M params",
                          "all but this (%)", "drop when removed (pp)"], lines))
        o1 = sorted(single, key=single.get, reverse=True)
        o2 = sorted(loo, key=loo.get, reverse=True)
        orders[base] = (single, loo)
        out.append(f"\n**Order (single-module):** {' > '.join(o1)}  \n**Order (leave-one-out drop):** {' > '.join(o2)}\n")
        extra = []
        for n in ("attn_only", "mlp_only", "layers_first_third", "layers_middle_third", "layers_last_third"):
            a = agg(runs.get(f"{base}_{n}", []), "gsm8k_acc")
            if a:
                extra.append([n, fmt(a, "gsm8k_acc")])
        if extra:
            out.append(table(["subset", "GSM8K (%)"], extra))
            out.append("")
    if "q" in orders and "fp" in orders:
        for k, label in ((0, "single-module"), (1, "leave-one-out")):
            common = [t for t in types if t in orders["q"][k] and t in orders["fp"][k]]
            tau, p = kendall([orders["q"][k][t] for t in common], [orders["fp"][k][t] for t in common])
            out.append(f"- Kendall τ between quantized and FP importance order ({label}): **{tau:.2f}** (p={p:.3f})")
        out.append("")
    # proxies
    files = glob.glob(os.path.join(importance_dir, "*.json"))
    if files and "q" in orders:
        out.append("### Training-free importance proxies vs. measured importance (quantized base)\n")
        lines = []
        for fpath in sorted(files):
            imp = json.load(open(fpath))
            by_type = imp["summary"]["by_type"]
            for metric in ("taylor", "fisher", "grad_norm"):
                for k, label in ((0, "single"), (1, "leave-one-out")):
                    meas = orders["q"][k]
                    common = [t for t in types if t in meas and f"{t}_proj" in by_type]
                    rho, p = spearman([by_type[f"{t}_proj"][metric] for t in common], [meas[t] for t in common])
                    lines.append([os.path.basename(fpath), metric, label, f"{rho:.2f}", f"{p:.3f}"])
        out.append(table(["proxy file", "proxy", "vs. measured", "Spearman ρ", "p"], lines))
        out.append("\n(Only 7 module types → treat p-values with care; ρ ≥ 0.79 is needed for p < 0.05.)\n")


def shapley(values: dict, factors):
    """values: frozenset(active factors) -> metric.  Returns {factor: shapley value of (metric - v(∅))}."""
    n = len(factors)
    phi = {}
    for f in factors:
        tot = 0.0
        others = [g for g in factors if g != f]
        for k in range(n):
            for S in itertools.combinations(others, k):
                S = frozenset(S)
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                tot += w * (values[S | {f}] - values[S])
        phi[f] = tot
    return phi


def rq3(rows, out, figs):
    runs = by_run(rows, "rq3_components")
    if runs:
        out.append("## RQ3a — Importance of each quantization component (Shapley decomposition)\n")
        out.append("Factors: W4 weights, A8 activations, KV4 cache, IntAttention softmax. The Shapley value "
                   "splits the total metric change (all-on vs all-off) fairly among the factors, averaging "
                   "over all orders in which they could be switched on, i.e. it accounts for interactions.\n")
        factors = ["W4", "A8", "KV4", "IntAttn"]
        res = {}
        for prefix in ("ptq", "ft"):
            for metric in ("gsm8k_acc", "wikitext2_ppl", "lm_eval_avg"):
                vals = {}
                for w, a, kv, at in itertools.product([16, 4], [16, 8], [16, 4], ["fp", "int"]):
                    name = f"{prefix}_w{w}_a{a}_kv{kv}_{at}"
                    ag = agg(runs.get(name, []), metric)
                    if ag is None:
                        break
                    S = frozenset(f for f, on in zip(factors, (w == 4, a == 8, kv == 4, at == "int")) if on)
                    vals[S] = ag["mean"] if metric != "wikitext2_ppl" else math.log(ag["mean"])
                else:
                    res[(prefix, metric)] = (shapley(vals, factors), vals[frozenset(factors)] - vals[frozenset()])
        lines = []
        for (prefix, metric), (phi, total) in res.items():
            unit = "Δ log-PPL" if metric == "wikitext2_ppl" else "Δ pp"
            s = 1 if metric == "wikitext2_ppl" else 100
            dmg = (lambda f: phi[f]) if metric == "wikitext2_ppl" else (lambda f: -phi[f])
            order = sorted(factors, key=dmg, reverse=True)          # most damaging first
            lines.append([("PTQ only" if prefix == "ptq" else "after QA-LoRA"), metric, unit,
                          *[f"{phi[f] * s:+.3f}" for f in factors], f"{total * s:+.3f}", " > ".join(order)])
        if lines:
            out.append(table(["stage", "metric", "unit", *factors, "total", "damage order (worst first)"], lines))
            out.append("\nNegative accuracy / positive log-PPL entries are damage attributed to that component. "
                       "Comparing the PTQ and post-finetuning rows shows which damage finetuning repairs.\n")
            _shapley_figure(figs, res, factors)
    runs = by_run(rows, "rq3_tricks")
    if runs:
        out.append("## RQ3b — Do the QoQ PTQ tricks still matter after finetuning?\n")
        lines = []
        for prefix in ("ptq_only", "ptq_then_ft"):
            full = runs.get(prefix)
            if not full:
                continue
            fa, fp_ = agg(full, "gsm8k_acc"), agg(full, "wikitext2_ppl")
            for name, rs in sorted(runs.items()):
                if not name.startswith(prefix + "_"):
                    continue
                a, p = agg(rs, "gsm8k_acc"), agg(rs, "wikitext2_ppl")
                lines.append([prefix, name[len(prefix) + 1:], fmt(a, "gsm8k_acc"),
                              f"{(a['mean'] - fa['mean']) * 100:+.2f}" if a and fa else "-",
                              fmt(p, "wikitext2_ppl"), f"{p['mean'] - fp_['mean']:+.3f}" if p and fp_ else "-"])
        out.append(table(["stage", "removed trick", "GSM8K (%)", "Δ GSM8K (pp)", "WikiText-2 PPL", "Δ PPL"], lines))
        out.append("")


def _shapley_figure(figs, res, factors):
    plt = _plt()
    keys = [k for k in res if k[1] == "wikitext2_ppl"] + [k for k in res if k[1] == "gsm8k_acc"]
    metrics = sorted({k[1] for k in keys}, key=lambda m: m != "wikitext2_ppl")
    if not metrics:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.0))
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        x = np.arange(len(factors))
        for j, prefix in enumerate(("ptq", "ft")):
            if (prefix, metric) not in res:
                continue
            phi = res[(prefix, metric)][0]
            s = 1 if metric == "wikitext2_ppl" else 100
            ax.bar(x + (j - 0.5) * 0.38, [phi[f] * s for f in factors], width=0.36, color=SERIES[j],
                   label="PTQ only" if prefix == "ptq" else "after QA-LoRA")
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xticks(x, factors)
        ax.set_title(("Shapley share of Δ log-PPL" if metric == "wikitext2_ppl" else "Shapley share of Δ GSM8K (pp)"),
                     loc="left")
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "rq3_shapley.png"), dpi=160)
    plt.close(fig)


def rq4(rows, out, figs):
    runs = by_run(rows, "rq4_intattn")
    if not runs:
        return
    out.append("## RQ4 — IntAttention (IndexSoftmax) under finetuning\n")
    grid = {}
    for name, rs in runs.items():
        if name.startswith("ptq_b"):
            b, c = name[len("ptq_b"):].split("_c")
            a = agg(rs, "wikitext2_ppl")
            if a:
                grid[(int(b), float(c))] = a["mean"]
    if grid:
        bs = sorted({k[0] for k in grid})
        cs = sorted({k[1] for k in grid})
        lines = [[f"b={b}"] + [f"{grid.get((b, c), float('nan')):.3f}" for c in cs] for b in bs]
        out.append("PTQ WikiText-2 PPL of QoQ+IntAttention vs LUT bits b and clipping threshold c:\n")
        out.append(table(["", *[f"c={c}" for c in cs]], lines))
        out.append("")
        plt = _plt()
        M = np.array([[grid.get((b, c), np.nan) for c in cs] for b in bs])
        fig, ax = plt.subplots(figsize=(5.5, 2.6))
        im = ax.imshow(M, cmap="Blues_r", aspect="auto")
        ax.set_xticks(range(len(cs)), [str(c) for c in cs])
        ax.set_yticks(range(len(bs)), [str(b) for b in bs])
        ax.set_xlabel("clipping threshold c")
        ax.set_ylabel("LUT bits b")
        ax.grid(False)
        for i in range(len(bs)):
            for j in range(len(cs)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color=SURFACE if M[i, j] < np.nanmedian(M) else INK)
        fig.colorbar(im, ax=ax, label="WikiText-2 PPL (lower = darker)")
        ax.set_title("IndexSoftmax sensitivity after PTQ", loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(figs, "rq4_bc_heatmap.png"), dpi=160)
        plt.close(fig)
    lines = []
    for name, rs in sorted(runs.items()):
        if name.startswith("ft_"):
            lines.append([name, fmt(agg(rs, "gsm8k_acc"), "gsm8k_acc"), fmt(agg(rs, "wikitext2_ppl"), "wikitext2_ppl"),
                          fmt(agg(rs, "task_val_loss"), "task_val_loss")])
    if lines:
        out.append(table(["finetuning variant", "GSM8K (%)", "WikiText-2 PPL", "task val loss"], lines))
        out.append("")
    # runtime sweeps
    sweep_lines = []
    for name in ("ft_intattn_in_loop", "ft_exact_softmax_in_loop"):
        rs = runs.get(name, [])
        acc = defaultdict(list)
        for r in rs:
            for e in r.get("runtime_sweep") or []:
                acc[json.dumps(e["override"], sort_keys=True)].append(e["wikitext2_ppl"])
        for k, v in sorted(acc.items()):
            sweep_lines.append([name, k, f"{np.mean(v):.3f}"])
    if sweep_lines:
        out.append("Same adapters re-evaluated under other attention runtimes (WikiText-2 PPL):\n")
        out.append(table(["trained with", "eval runtime override", "PPL"], sweep_lines))
        out.append("")


def rq5(rows, out, figs):
    runs = by_run(rows, "rq5_rank")
    if not runs:
        return
    out.append("## RQ5 — LoRA rank needed: full-precision vs quantized base\n")
    ranks = [2, 8, 16, 64, 128]
    lines = []
    series = {}
    for base in ("fp", "q"):
        pts = []
        for r in ranks:
            a = agg(runs.get(f"{base}_r{r}", []), "gsm8k_acc")
            if a:
                pts.append((r, a))
        series[base] = pts
        lines.append([base] + [fmt(dict(a), "gsm8k_acc") for _, a in pts])
    out.append(table(["base", *[f"r={r}" for r in ranks]], lines))
    for n in ("q_r16_loftq", "q_r64_loftq"):
        a = agg(runs.get(n, []), "gsm8k_acc")
        if a:
            out.append(f"\n- `{n}`: {fmt(a, 'gsm8k_acc')}")
    out.append("")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 3))
    for j, (base, label) in enumerate((("fp", "FP base"), ("q", "W4A8KV4+IntAttn base"))):
        pts = series.get(base) or []
        if not pts:
            continue
        x = [p[0] for p in pts]
        y = [p[1]["mean"] * 100 for p in pts]
        e = [p[1]["std"] * 100 for p in pts]
        ax.errorbar(x, y, yerr=e, color=SERIES[j], marker="o", capsize=2, label=label)
        ax.text(x[-1], y[-1], f"  {label}", color=INK2, fontsize=8, va="center")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("GSM8K accuracy (%)")
    ax.legend(fontsize=8)
    ax.set_title("Accuracy vs LoRA rank", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "rq5_rank.png"), dpi=160)
    plt.close(fig)


def write_csv(rows, path):
    import csv
    keys = ["suite", "run", "seed", "pipeline", *METRICS, "lora_params", "train_seconds"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in sorted(rows, key=lambda r: (r["suite"], r["run"], r["seed"])):
            w.writerow([r.get(k) for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    rows = load(args.results)
    if not rows:
        print("no results found")
        return
    figs = os.path.join(args.results, "figures")
    os.makedirs(figs, exist_ok=True)
    write_csv(rows, os.path.join(args.results, "summary.csv"))
    out = ["# Results report (auto-generated by scripts/analyze.py)\n",
           f"{len(rows)} runs found. Mean ± sample std over seeds.\n"]
    rq0(rows, out, figs)
    rq1(rows, out, figs)
    rq2(rows, out, figs, os.path.join(args.results, "importance"))
    rq3(rows, out, figs)
    rq4(rows, out, figs)
    rq5(rows, out, figs)
    fig_list = sorted(os.listdir(figs))
    if fig_list:
        out.append("## Figures\n")
        out += [f"![{f}](figures/{f})" for f in fig_list]
    with open(os.path.join(args.results, "REPORT.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {os.path.join(args.results, 'REPORT.md')} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
