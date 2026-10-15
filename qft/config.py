"""Experiment configuration (dataclasses + YAML loading + named quantization presets)."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass

import yaml

ALL_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# fields of QuantConfig that only change *runtime* behaviour (can differ between train and eval)
RUNTIME_FIELDS = {"a_bits", "a_granularity", "kv_bits", "attn_mode", "int_attn_b", "int_attn_c",
                  "int_attn_granularity"}


@dataclass
class QuantConfig:
    # ---------------- weights (offline)
    w_bits: int = 16
    w_method: str = "rtn"              # rtn | progressive | nf4
    w_group_size: int = -1             # -1 = per output channel
    w_sym: bool = False
    weight_clip: bool = False          # QoQ: output-MSE clip-ratio search
    clip_ratios: list = field(default_factory=lambda: [1.0 - 0.05 * i for i in range(10)])
    reorder: bool = False              # QoQ: activation-aware channel reordering
    # ---------------- function-preserving transforms (offline)
    rotation: str = "none"             # none | hadamard | random_hadamard | orthogonal
    smooth_output: bool = False        # QoQ block-output smoothing (v->o, up->down)
    smooth_alphas: list = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    smooth_attention: bool = False     # QoQ SmoothAttention
    smooth_attention_alpha: float = 0.5
    # ---------------- runtime
    a_bits: int = 16                   # activation bits (input of every linear)
    a_granularity: str = "per_token"
    kv_bits: int = 16                  # K/V cache bits (per-token, per-head, asymmetric)
    attn_mode: str = "fp"              # fp | quant_only | int   (int = IntAttention / IndexSoftmax)
    int_attn_b: int = 5
    int_attn_c: float = 6.6
    int_attn_granularity: str = "per_head"
    # ---------------- calibration
    calib_dataset: str = "c4"          # c4 | wikitext2 | gsm8k
    calib_nsamples: int = 128
    calib_seqlen: int = 512
    calib_tokens_per_batch: int = 256  # sampled token rows per batch of 8 sequences

    def runtime(self) -> dict:
        return {k: getattr(self, k) for k in RUNTIME_FIELDS}

    @property
    def is_fp(self) -> bool:
        return (self.w_bits >= 16 and self.a_bits >= 16 and self.kv_bits >= 16 and self.attn_mode == "fp"
                and self.rotation == "none" and not self.smooth_output and not self.smooth_attention)


# Named presets.  Each is a dict of overrides on top of QuantConfig() defaults.
_QOQ = dict(w_bits=4, w_method="progressive", w_group_size=128, a_bits=8, kv_bits=4, rotation="hadamard",
            smooth_output=True, smooth_attention=True, weight_clip=True, reorder=True)
QUANT_PRESETS: dict[str, dict] = {
    "fp16": {},
    # ---- weight-only references (QLoRA family)
    "nf4": dict(w_bits=4, w_method="nf4"),
    "w4a16_g128": dict(w_bits=4, w_method="rtn", w_group_size=128),
    # ---- QServe Fig. 12 ladder (+ IntAttention as the final rung)
    "w8a8": dict(w_bits=8, w_sym=True, a_bits=8),
    "w4a8": dict(w_bits=4, a_bits=8),
    "w4a8_rot": dict(w_bits=4, a_bits=8, rotation="hadamard", smooth_output=True),
    "w4a8_rot_clip": dict(w_bits=4, a_bits=8, rotation="hadamard", smooth_output=True, weight_clip=True),
    "w4a8kv4_rot_clip": dict(w_bits=4, a_bits=8, kv_bits=4, rotation="hadamard", smooth_output=True,
                             weight_clip=True),
    "w4a8kv4_smoothattn": dict(w_bits=4, a_bits=8, kv_bits=4, rotation="hadamard", smooth_output=True,
                               weight_clip=True, smooth_attention=True),
    "w4a8kv4_g128": dict(w_bits=4, w_method="progressive", w_group_size=128, a_bits=8, kv_bits=4,
                         rotation="hadamard", smooth_output=True, weight_clip=True, smooth_attention=True),
    "qoq": dict(_QOQ),                                       # full QoQ W4A8KV4 g128
    "qoq_quantonly_attn": dict(_QOQ, attn_mode="quant_only"),
    "qoq_intattn": dict(_QOQ, attn_mode="int"),              # QoQ + IntAttention (full integer stack)
    # ---- same bit widths, no QoQ tricks (plain RTN)
    "w4a8kv4_g128_rtn": dict(w_bits=4, w_group_size=128, a_bits=8, kv_bits=4),
    "w4a8kv4_g128_rtn_intattn": dict(w_bits=4, w_group_size=128, a_bits=8, kv_bits=4, attn_mode="int"),
    # ---- attention-only quantization
    "intattn_only": dict(attn_mode="int"),
    "quantonly_attn_only": dict(attn_mode="quant_only"),
}


def make_quant_config(spec) -> QuantConfig:
    """spec: preset name | dict (optionally with key 'preset') | QuantConfig."""
    if isinstance(spec, QuantConfig):
        return spec
    if isinstance(spec, str):
        spec = {"preset": spec}
    spec = dict(spec or {})
    preset = spec.pop("preset", "fp16")
    if preset not in QUANT_PRESETS:
        raise KeyError(f"unknown quant preset '{preset}'. Available: {sorted(QUANT_PRESETS)}")
    kw = dict(QUANT_PRESETS[preset])
    kw.update(spec)
    _check_keys(QuantConfig, kw)
    return QuantConfig(**kw)


@dataclass
class LoraConfig:
    enabled: bool = True
    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    targets: list = field(default_factory=lambda: list(ALL_TARGETS))
    layers: object = "all"            # all | first_third | middle_third | last_third | "0-7" | [..]
    init: str = "default"             # default | loftq
    loftq_iters: int = 1
    input_quant: bool = True          # LoRA branch consumes the quantized activation


@dataclass
class TrainConfig:
    max_steps: int = 1000
    batch_size: int = 16
    grad_accum: int = 1
    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    scheduler: str = "cosine"         # cosine | linear | constant
    max_grad_norm: float = 0.3
    seq_len: int = 512                # max length (gsm8k) or packed chunk length (c4 / wikitext)
    grad_checkpointing: bool = False
    log_every: int = 10
    val_every: int = 0                # >0: compute held-out loss every N steps
    val_batches: int = 8


@dataclass
class EvalConfig:
    wikitext2: bool = True
    c4: bool = False
    ppl_seq_len: int = 2048
    ppl_batch_size: int = 4
    ppl_max_chunks: int | None = None  # limit #chunks for quick runs
    c4_nsamples: int = 256
    gsm8k: bool = True
    gsm8k_limit: int | None = None
    gsm8k_batch_size: int = 64
    gsm8k_max_new_tokens: int = 256
    gsm8k_fewshot: object = "auto"     # auto: 8-shot without finetuning, 0-shot after finetuning
    lm_eval_tasks: list = field(default_factory=lambda: ["arc_easy", "arc_challenge", "piqa", "hellaswag",
                                                         "winogrande"])
    lm_eval_limit: int | None = None
    lm_eval_batch_size: int = 32
    task_val_loss: bool = True         # held-out loss on the finetuning task
    # extra cheap evaluations (WikiText-2 PPL + task val loss) under modified *runtime* settings,
    # e.g. [{int_attn_c: 4.5}, {int_attn_c: 9.0}]  -> metrics["runtime_sweep"]
    runtime_sweep: list = field(default_factory=list)


@dataclass
class RunConfig:
    name: str = "debug"
    output_dir: str = "results"
    model: str = "Qwen/Qwen2.5-0.5B"
    task: str = "gsm8k"               # gsm8k | c4 | wikitext2   (finetuning data)
    # fp_eval | fp_ft | ptq_only | ptq_then_ft | ft_then_ptq
    pipeline: str = "ptq_then_ft"
    quant: QuantConfig = field(default_factory=QuantConfig)
    train_runtime: dict = field(default_factory=dict)   # runtime-field overrides used *during training*
    merge_for_eval: str = "none"      # none | requant_deq | requant_fp
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 0
    dtype: str = "bfloat16"           # compute dtype for training / evaluation
    device: str = "cuda"
    save_adapter: bool = False
    tags: dict = field(default_factory=dict)            # free-form labels used by the analysis

    def to_dict(self):
        return asdict(self)


def _check_keys(cls, d: dict):
    names = {f.name for f in fields(cls)}
    bad = set(d) - names
    if bad:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(bad)}")


def _build(cls, d):
    d = dict(d or {})
    _check_keys(cls, d)
    return cls(**d)


def config_from_dict(d: dict) -> RunConfig:
    d = copy.deepcopy(d)
    q = make_quant_config(d.pop("quant", "fp16"))
    lora = _build(LoraConfig, d.pop("lora", {}))
    train = _build(TrainConfig, d.pop("train", {}))
    ev = _build(EvalConfig, d.pop("eval", {}))
    tr = d.get("train_runtime") or {}
    bad = set(tr) - RUNTIME_FIELDS
    if bad:
        raise KeyError(f"train_runtime may only override runtime fields {sorted(RUNTIME_FIELDS)}; got {bad}")
    _check_keys(RunConfig, d)
    return RunConfig(quant=q, lora=lora, train=train, eval=ev, **d)


def deep_update(base: dict, upd: dict) -> dict:
    """Recursive dict update; keys may be dotted ('lora.r': 8)."""
    out = copy.deepcopy(base)
    if isinstance(out.get("quant"), str):
        out["quant"] = {"preset": out["quant"]}
    for k, v in (upd or {}).items():
        if k == "quant" and (isinstance(v, str) or (isinstance(v, dict) and "preset" in v)):
            out[k] = {"preset": v} if isinstance(v, str) else copy.deepcopy(v)   # new preset: replace
        elif "." in k:
            head, rest = k.split(".", 1)
            out[head] = deep_update(out.get(head) or {}, {rest: v})
        elif isinstance(v, dict) and isinstance(out.get(k), dict) and k != "train_runtime":
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(path: str, overrides: dict | None = None) -> RunConfig:
    d = load_yaml(path)
    if overrides:
        d = deep_update(d, overrides)
    return config_from_dict(d)


def to_plain(obj):
    if is_dataclass(obj):
        return {k: to_plain(v) for k, v in asdict(obj).items()}
    return obj
