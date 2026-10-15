"""Run a single experiment.

    python scripts/run_experiment.py --config configs/base.yaml \
        --set name=debug pipeline=ptq_then_ft quant=qoq_intattn train.max_steps=50 seed=0

``--set`` takes dotted keys; values are parsed as YAML (so lists/bools/numbers work):
    --set "lora.targets=[q_proj,v_proj]" quant.a_bits=16 "train_runtime={attn_mode: fp}"
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qft.config import load_config  # noqa: E402
from qft.runner import run_experiment  # noqa: E402


def parse_sets(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--set expects key=value, got {it}")
        k, v = it.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_sets(args.set))
    run_experiment(cfg, force=args.force)


if __name__ == "__main__":
    main()
