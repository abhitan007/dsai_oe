"""Datasets: calibration text, finetuning data (GSM8K SFT / C4 / WikiText LM) and eval corpora."""
from __future__ import annotations

import random

import torch

C4_TRAIN = "en/c4-train.00000-of-01024.json.gz"
C4_VAL = "en/c4-validation.00000-of-00008.json.gz"


# --------------------------------------------------------------------------------------
# raw text sources
# --------------------------------------------------------------------------------------
def _c4(split: str):
    from datasets import load_dataset
    files = {"train": C4_TRAIN} if split == "train" else {"validation": C4_VAL}
    return load_dataset("allenai/c4", data_files=files, split="train" if split == "train" else "validation")


def _wikitext(split: str, name: str = "wikitext-2-raw-v1"):
    from datasets import load_dataset
    return load_dataset("Salesforce/wikitext", name, split=split)


def _gsm8k(split: str):
    from datasets import load_dataset
    return load_dataset("openai/gsm8k", "main", split=split)


# --------------------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------------------
def get_calibration_batches(tok, dataset: str = "c4", nsamples: int = 128, seqlen: int = 512, seed: int = 0,
                            batch_size: int = 8):
    """GPTQ-style random windows of ``seqlen`` tokens; returns list of LongTensor [B, seqlen]."""
    rng = random.Random(seed)
    samples = []
    if dataset == "c4":
        ds = _c4("train")
        while len(samples) < nsamples:
            doc = ds[rng.randint(0, len(ds) - 1)]["text"]
            ids = tok(doc, return_tensors="pt").input_ids[0]
            if ids.numel() <= seqlen:
                continue
            s = rng.randint(0, ids.numel() - seqlen - 1)
            samples.append(ids[s:s + seqlen])
    else:
        if dataset == "wikitext2":
            text = "\n\n".join(_wikitext("train")["text"])
        elif dataset == "gsm8k":
            ds = _gsm8k("train")
            text = "\n\n".join(format_gsm8k(q, a) for q, a in zip(ds["question"], ds["answer"]))
        else:
            raise ValueError(dataset)
        ids = tok(text, return_tensors="pt").input_ids[0]
        for _ in range(nsamples):
            s = rng.randint(0, ids.numel() - seqlen - 1)
            samples.append(ids[s:s + seqlen])
    return [torch.stack(samples[i:i + batch_size]) for i in range(0, len(samples), batch_size)]


# --------------------------------------------------------------------------------------
# GSM8K
# --------------------------------------------------------------------------------------
def gsm8k_prompt(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def format_gsm8k(question: str, answer: str) -> str:
    return f"{gsm8k_prompt(question)} {answer.strip()}"


class GSM8KSFT(torch.utils.data.Dataset):
    """Prompt/answer pairs; loss only on answer tokens (+ EOS)."""

    def __init__(self, tok, split: str = "train", max_len: int = 512, indices=None):
        ds = _gsm8k(split)
        self.items = []
        idx = range(len(ds)) if indices is None else indices
        for i in idx:
            ex = ds[int(i)]
            p = tok(gsm8k_prompt(ex["question"]), add_special_tokens=False).input_ids
            a = tok(" " + ex["answer"].strip(), add_special_tokens=False).input_ids + [tok.eos_token_id]
            ids = (p + a)[:max_len]
            labels = ([-100] * len(p) + a)[:max_len]
            self.items.append((ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate_left_pad(batch, pad_id: int):
    L = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), L), dtype=torch.long)
    for j, (ids, lab) in enumerate(batch):
        n = len(ids)
        input_ids[j, L - n:] = torch.tensor(ids)
        labels[j, L - n:] = torch.tensor(lab)
        attn[j, L - n:] = 1
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn, "position_ids": pos}


# --------------------------------------------------------------------------------------
# language-modelling finetuning data (packed)
# --------------------------------------------------------------------------------------
class PackedLM(torch.utils.data.Dataset):
    def __init__(self, tok, source: str, n_chunks: int, seq_len: int, seed: int = 0, split: str = "train"):
        rng = random.Random(seed)
        buf: list[int] = []
        self.chunks = []
        if source == "c4":
            ds = _c4(split)
            order = list(range(len(ds)))
            rng.shuffle(order)
            it = (ds[i]["text"] for i in order)
        elif source in ("wikitext2", "wikitext103"):
            name = "wikitext-2-raw-v1" if source == "wikitext2" else "wikitext-103-raw-v1"
            ds = _wikitext(split, name)
            it = iter(["\n\n".join(ds["text"])])
        else:
            raise ValueError(source)
        for text in it:
            buf.extend(tok(text, add_special_tokens=False).input_ids + [tok.eos_token_id])
            while len(buf) >= seq_len and len(self.chunks) < n_chunks:
                self.chunks.append(buf[:seq_len])
                buf = buf[seq_len:]
            if len(self.chunks) >= n_chunks:
                break
        if len(self.chunks) < n_chunks:
            print(f"[data] only {len(self.chunks)} chunks available from {source}/{split}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        ids = self.chunks[i]
        return ids, list(ids)


def infinite_loader(dataset, batch_size: int, pad_id: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    while True:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g,
                                             drop_last=True, collate_fn=lambda b: collate_left_pad(b, pad_id))
        yield from loader


def build_train_data(tok, task: str, train_cfg, seed: int):
    """Returns (train_dataset, val_dataset)."""
    total = train_cfg.max_steps * train_cfg.batch_size * train_cfg.grad_accum
    if task == "gsm8k":
        n = len(_gsm8k("train"))
        rng = random.Random(1234)                      # fixed split independent of the run seed
        order = list(range(n))
        rng.shuffle(order)
        val_idx, train_idx = order[:256], order[256:]
        return (GSM8KSFT(tok, "train", train_cfg.seq_len, train_idx),
                GSM8KSFT(tok, "train", train_cfg.seq_len, val_idx))
    if task in ("c4", "wikitext2", "wikitext103"):
        train = PackedLM(tok, task, total, train_cfg.seq_len, seed=seed, split="train")
        val = PackedLM(tok, task, 64, train_cfg.seq_len, seed=0, split="validation")
        return train, val
    raise ValueError(f"unknown task {task}")


# --------------------------------------------------------------------------------------
# perplexity corpora
# --------------------------------------------------------------------------------------
def wikitext2_test_ids(tok) -> torch.Tensor:
    text = "\n\n".join(_wikitext("test")["text"])
    return tok(text, return_tensors="pt").input_ids[0]


def c4_val_ids(tok, nsamples: int = 256, seqlen: int = 2048, seed: int = 0) -> torch.Tensor:
    """GPTQ protocol: random ``seqlen`` windows from the C4 validation shard, concatenated."""
    ds = _c4("validation")
    rng = random.Random(seed)
    out = []
    while len(out) < nsamples:
        ids = tok(ds[rng.randint(0, len(ds) - 1)]["text"], return_tensors="pt").input_ids[0]
        if ids.numel() <= seqlen:
            continue
        s = rng.randint(0, ids.numel() - seqlen - 1)
        out.append(ids[s:s + seqlen])
    return torch.cat(out)
