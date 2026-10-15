from __future__ import annotations

import os
import random
import sys
import time

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Logger:
    """print + append to a log file."""

    def __init__(self, path: str | None = None):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def __call__(self, msg: str):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, file=sys.stdout, flush=True)
        if self.path:
            with open(self.path, "a") as f:
                f.write(line + "\n")
