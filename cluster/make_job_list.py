"""Expand an experiment suite into one resolved config per (run, seed) for a SLURM array job.

    python cluster/make_job_list.py --suite configs/suites/rq1_pipelines.yaml --profile p100
    -> cluster/jobs/rq1_pipelines/0001.yaml, 0002.yaml, ...  and  cluster/jobs/rq1_pipelines/index.tsv
       (prints the sbatch command to submit it)

Profiles adapt the runs to the GPU without changing what is measured:
  p100  P100 16 GB, no native bf16 -> float32 compute; batch 8 x grad-accum 2 (same effective batch 16)
        with gradient checkpointing (verified bit-identical); smaller evaluation batches.
  h100  bfloat16 and the default batch sizes from configs/base.yaml.
Use the same profile for every run you compare in one table.
"""
import argparse
import os
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from launch import expand_suite  # noqa: E402
from qft.config import config_from_dict, deep_update  # noqa: E402

PROFILES = {
    "p100": {"dtype": "float32", "train.batch_size": 8, "train.grad_accum": 2, "train.grad_checkpointing": True,
             "eval.ppl_batch_size": 1, "eval.gsm8k_batch_size": 16, "eval.lm_eval_batch_size": 16},
    "h100": {"dtype": "bfloat16"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--profile", choices=sorted(PROFILES), required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="substring filter on run names")
    ap.add_argument("--base", default=None, help="replace the suite's base config (e.g. configs/base_c4.yaml)")
    ap.add_argument("--set", nargs="*", default=[], help="extra dotted overrides, e.g. eval.lm_eval_tasks=[]")
    ap.add_argument("--out", default=None, help="job directory (default cluster/jobs/<suite name>)")
    args = ap.parse_args()

    suite_name, jobs = expand_suite(args.suite, args.seeds, args.only, args.base)
    extra = dict(PROFILES[args.profile])
    for it in args.set:
        k, v = it.split("=", 1)
        extra[k] = yaml.safe_load(v)
    # Where the SLURM job stores results: the (base config's) output_dir, e.g. results_c4 for configs/base_c4.yaml.
    roots = {deep_update(j, extra).get("output_dir", "results") for j in jobs}
    if len(roots) != 1:
        sys.exit(f"runs disagree on output_dir: {sorted(roots)}")
    results_root = os.path.normpath(roots.pop())
    if os.path.isabs(results_root) or results_root.startswith(".."):
        sys.exit(f"output_dir must be a folder inside the project, got {results_root}")
    default_dir = suite_name if results_root == "results" else f"{suite_name}_{results_root.removeprefix('results_')}"
    out = args.out or os.path.join(ROOT, "cluster", "jobs", default_dir)
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):                        # a job list is regenerated from scratch
        if f.endswith(".yaml") or f in ("index.tsv", "results_root"):
            os.remove(os.path.join(out, f))

    lines = []
    for i, j in enumerate(jobs, 1):
        j = deep_update(j, extra)
        j.pop("output_dir", None)                    # set by the SLURM job ($TMPDIR/results)
        j.setdefault("tags", {})["profile"] = args.profile
        config_from_dict(j)                          # validate
        fname = f"{i:04d}.yaml"
        with open(os.path.join(out, fname), "w") as f:
            yaml.safe_dump(j, f, sort_keys=False)
        lines.append(f"{i}\t{j['name']}\t{j['seed']}\t{fname}")
    with open(os.path.join(out, "index.tsv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out, "results_root"), "w") as f:
        f.write(results_root + "\n")

    rel = os.path.relpath(out, ROOT)
    print(f"[jobs] suite {suite_name}: {len(jobs)} runs (profile {args.profile}) -> {rel}/   results -> {results_root}/")
    for ln in lines:
        print("   " + ln.replace("\t", "  "))
    print(f"\nSubmit from the project root with:\n  sbatch --array=1-{len(jobs)}%4 cluster/run_array_{args.profile}.slurm {rel}")


if __name__ == "__main__":
    main()
