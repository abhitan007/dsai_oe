"""Expand an experiment suite (YAML) into runs and execute them on a pool of GPUs.

    python scripts/launch.py --suite configs/suites/rq1_pipelines.yaml --gpus 0,1,2,3
    python scripts/launch.py --suite configs/suites/rq3_components.yaml --gpus 0 --seeds 0 --dry-run

Suite file format:
    base: configs/base.yaml          # base run config
    common: {...}                    # overrides applied to every run (dotted keys allowed)
    seeds: [0, 1, 2]                 # default seeds (CLI --seeds wins)
    runs:                            # explicit runs
      - name: my_run
        set: {pipeline: ptq_then_ft, quant: qoq_intattn}
        seeds: [0]                   # optional per-run seeds (e.g. deterministic PTQ-only runs)
    grid:                            # optional cartesian products -> extra runs
      - name: "fact_w{quant.w_bits}_a{quant.a_bits}"
        axes: {quant.w_bits: [16, 4], quant.a_bits: [16, 8]}
        set: {...}

Each run is executed as a separate process (`run_experiment.py --config <resolved.yaml>`), one
process per GPU at a time.  Finished runs (result.json exists) are skipped, so a suite can be
re-launched after interruption.
"""
import argparse
import copy
import itertools
import os
import subprocess
import sys
import time

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from qft.config import config_from_dict, deep_update, load_yaml  # noqa: E402


def expand_suite(path, seeds_override=None, only=None, base_override=None):
    suite = load_yaml(path)
    suite_name = suite.get("name") or os.path.splitext(os.path.basename(path))[0]
    base = load_yaml(os.path.join(ROOT, base_override or suite["base"]))
    base = deep_update(base, suite.get("common", {}))
    default_seeds = seeds_override or suite.get("seeds", [0])
    runs = []
    for r in suite.get("runs", []):
        runs.append((r["name"], r.get("set", {}), r.get("seeds")))
    for g in suite.get("grid", []):
        keys = list(g["axes"])
        for vals in itertools.product(*[g["axes"][k] for k in keys]):
            assign = dict(zip(keys, vals))
            fmt = {k.replace(".", "_"): v for k, v in assign.items()}
            name = g["name"].format(**fmt)
            st = deep_update(copy.deepcopy(g.get("set", {})), {})
            st.update(assign)
            runs.append((name, st, g.get("seeds")))
    jobs = []
    for name, st, seeds in runs:
        if only and not any(o in name for o in only):
            continue
        for seed in (seeds_override or seeds or default_seeds):
            d = deep_update(base, st)
            d["name"] = f"{suite_name}/{name}"
            d["seed"] = seed
            d.setdefault("tags", {})
            d["tags"] = {**d["tags"], "suite": suite_name, "run": name}
            config_from_dict(d)                    # validate early
            jobs.append(d)
    return suite_name, jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--gpus", default="0", help="comma separated GPU ids; jobs run one per GPU")
    ap.add_argument("--jobs_per_gpu", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--base", default=None, help="replace the suite's base config (e.g. configs/base_c4.yaml)")
    ap.add_argument("--only", nargs="*", default=None, help="substring filter on run names")
    ap.add_argument("--set", nargs="*", default=[], help="extra dotted overrides for every run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    suite_name, jobs = expand_suite(args.suite, args.seeds, args.only, args.base)
    extra = {}
    for it in args.set:
        k, v = it.split("=", 1)
        extra[k] = yaml.safe_load(v)
    if extra:
        jobs = [deep_update(j, extra) for j in jobs]

    todo = []
    for j in jobs:
        rdir = os.path.join(ROOT, j.get("output_dir", "results"), j["name"], f"seed{j['seed']}")
        if os.path.exists(os.path.join(rdir, "result.json")) and not args.force:
            continue
        todo.append((j, rdir))
    print(f"[launch] suite {suite_name}: {len(jobs)} jobs, {len(todo)} to run")
    for j, rdir in todo:
        print(f"   - {j['name']} seed={j['seed']} pipeline={j.get('pipeline')}")
    if args.dry_run or not todo:
        return

    slots = [g for g in args.gpus.split(",") for _ in range(args.jobs_per_gpu)]
    running = {}   # slot index -> (Popen, name)
    queue = list(todo)
    failed = []
    while queue or running:
        for s, gpu in enumerate(slots):
            if s in running:
                p, name = running[s]
                if p.poll() is None:
                    continue
                if p.returncode != 0:
                    failed.append(name)
                    print(f"[launch] FAILED ({p.returncode}): {name}")
                else:
                    print(f"[launch] finished: {name}")
                del running[s]
            if queue and s not in running:
                j, rdir = queue.pop(0)
                os.makedirs(rdir, exist_ok=True)
                cfg_path = os.path.join(rdir, "config.yaml")
                with open(cfg_path, "w") as f:
                    yaml.safe_dump(j, f, sort_keys=False)
                cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_experiment.py"), "--config", cfg_path]
                if args.force:
                    cmd.append("--force")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
                out = open(os.path.join(rdir, "stdout.txt"), "w")
                running[s] = (subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT),
                              f"{j['name']}/seed{j['seed']}")
                print(f"[launch] GPU {gpu}: started {j['name']} seed={j['seed']}")
        time.sleep(5)
    print(f"[launch] all done. failed: {failed if failed else 'none'}")


if __name__ == "__main__":
    main()
