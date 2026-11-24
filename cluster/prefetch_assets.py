"""Download everything the experiments need into one Hugging Face cache directory, then verify that it
loads with the network switched off (cluster compute nodes run with HF_*_OFFLINE=1).

Run on a machine WITH internet (e.g. WSL), using the same package versions as the cluster venv
(cluster/requirements-cluster.txt):

    HF_HOME=$HOME/dsai_hf_cache python cluster/prefetch_assets.py

Then tar the cache and copy it to <project>/hf_cache on the cluster (see INSTRUCTIONS.md, section 2.1).
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

LM_EVAL_TASKS = ["arc_easy", "arc_challenge", "piqa", "hellaswag", "winogrande"]


def fetch_and_check(model: str, tasks, wikitext103: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from qft.data import _c4, _gsm8k, _wikitext

    print(f"[assets] model {model}")
    AutoTokenizer.from_pretrained(model)
    AutoModelForCausalLM.from_pretrained(model)
    for split in ("train", "validation"):
        print(f"[assets] c4 {split}: {len(_c4(split))} docs")
    for split in ("train", "validation", "test"):
        print(f"[assets] wikitext-2 {split}: {len(_wikitext(split))} rows")
    if wikitext103:
        for split in ("train", "validation"):
            print(f"[assets] wikitext-103 {split}: {len(_wikitext(split, 'wikitext-103-raw-v1'))} rows")
    for split in ("train", "test"):
        print(f"[assets] gsm8k {split}: {len(_gsm8k(split))} rows")
    if tasks:
        from lm_eval.tasks import TaskManager, get_task_dict
        td = get_task_dict(list(tasks), TaskManager())
        for name, task in td.items():
            docs = task.eval_docs if hasattr(task, "eval_docs") else None
            print(f"[assets] lm-eval {name}: {len(docs) if docs is not None else '?'} eval docs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--tasks", nargs="*", default=LM_EVAL_TASKS)
    ap.add_argument("--wikitext103", action="store_true", help="also fetch WikiText-103 (only for task=wikitext103)")
    ap.add_argument("--_offline_check", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        sys.exit("Set HF_HOME to the cache directory you want to fill, e.g. HF_HOME=$HOME/dsai_hf_cache")

    if args._offline_check:
        fetch_and_check(args.model, args.tasks, args.wikitext103)
        return

    print(f"[assets] downloading into HF_HOME={hf_home}")
    fetch_and_check(args.model, args.tasks, args.wikitext103)

    print("\n[assets] re-loading everything with the network disabled (as on the compute nodes) ...")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    cmd = [sys.executable, os.path.abspath(__file__), "--_offline_check", "--model", args.model, "--tasks", *args.tasks]
    if args.wikitext103:
        cmd.append("--wikitext103")
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        sys.exit("[assets] OFFLINE CHECK FAILED - the cache is incomplete; do not copy it yet.")
    # The C4 arrow cache built by the local json builder is keyed on the absolute cache path, so it is never
    # reused once the cache is moved (the cluster jobs rebuild it offline from the shipped .json.gz in about a
    # minute).  Drop it to keep the tarball small.
    shutil.rmtree(os.path.join(hf_home, "datasets", "json"), ignore_errors=True)
    print(f"\n[assets] OFFLINE CHECK PASSED. Cache is ready: {hf_home}")


if __name__ == "__main__":
    main()
