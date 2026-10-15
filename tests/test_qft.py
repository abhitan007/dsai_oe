"""Unit tests (CPU, tiny random Qwen2).  Run:  python -m pytest -q tests/"""
import copy
import math

import pytest
import torch

from qft.config import QuantConfig, config_from_dict, deep_update, make_quant_config
from qft.lora import add_lora, merge_lora
from qft.model_utils import iter_linears
from qft.quant.apply import quantize_model, set_runtime, unwrap_model
from qft.quant.attention import AttnQuantState, integer_attention, register_attention
from qft.quant.hadamard import get_rotation, hadamard_matrix
from qft.quant.weight_quant import WeightQuantSpec, quant_progressive, quantize_weight, search_clip_ratios
from qft.config import LoraConfig

torch.manual_seed(0)


# ----------------------------------------------------------------------------- helpers
def tiny_model(seed=0, hidden=128, tie=True):
    from transformers import AutoModelForCausalLM, Qwen2Config
    register_attention()
    cfg = Qwen2Config(vocab_size=128, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=2,
                      num_attention_heads=hidden // 16, num_key_value_heads=2, max_position_embeddings=256,
                      tie_word_embeddings=tie, rope_theta=10000.0)
    torch.manual_seed(seed)
    ref = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager", torch_dtype=torch.float32)
    # inject outliers so that the transforms have something to do
    with torch.no_grad():
        for layer in ref.model.layers:
            layer.input_layernorm.weight.uniform_(0.5, 2.0)
            layer.post_attention_layernorm.weight.uniform_(0.5, 2.0)
            layer.self_attn.k_proj.weight[3] *= 8
            layer.self_attn.k_proj.bias.normal_()
            layer.self_attn.q_proj.bias.normal_()
    m = AutoModelForCausalLM.from_config(cfg, attn_implementation="qft", torch_dtype=torch.float32)
    m.load_state_dict(ref.state_dict())
    m.eval()
    ref.eval()
    return m, ref


def calib_batches(n=4, L=32):
    g = torch.Generator().manual_seed(1)
    return [torch.randint(0, 128, (2, L), generator=g) for _ in range(n)]


def logits(m, ids, **kw):
    with torch.no_grad():
        return m(input_ids=ids, **kw).logits


# ----------------------------------------------------------------------------- rotations
@pytest.mark.parametrize("n", [12, 28, 64, 96, 896, 1536])
def test_hadamard(n):
    H = hadamard_matrix(n)
    assert H is not None
    assert torch.allclose(H @ H.T, n * torch.eye(n, dtype=H.dtype))
    for kind in ("hadamard", "random_hadamard", "orthogonal"):
        Q = get_rotation(n, kind)
        assert torch.allclose(Q @ Q.T, torch.eye(n, dtype=Q.dtype), atol=1e-10)


# ----------------------------------------------------------------------------- weight quant
def test_rtn_and_progressive():
    W = torch.randn(64, 256)
    for spec in [WeightQuantSpec(8, "rtn", -1, True), WeightQuantSpec(4, "rtn", -1, False),
                 WeightQuantSpec(4, "rtn", 128, False), WeightQuantSpec(4, "progressive", 128),
                 WeightQuantSpec(4, "nf4")]:
        Wq = quantize_weight(W, spec)
        rel = ((Wq - W).norm() / W.norm()).item()
        assert rel < (0.02 if spec.bits == 8 else 0.15), (spec, rel)
    # finer granularity must help
    e_pc = (quantize_weight(W, WeightQuantSpec(4, "rtn", -1)) - W).norm()
    e_g = (quantize_weight(W, WeightQuantSpec(4, "rtn", 128)) - W).norm()
    assert e_g < e_pc
    # progressive: intermediate INT8 codes never overflow thanks to the protective range
    _, st = quant_progressive(torch.randn(256, 512) * torch.rand(256, 1) * 5, 128, return_stats=True)
    assert st["int8_overflow"] == 0
    # number of distinct values per group <= 16
    Wq = quantize_weight(W, WeightQuantSpec(4, "progressive", 128))
    assert all(len(torch.unique(Wq[r, :128])) <= 16 for r in range(8))


def test_perm_and_clip():
    W = torch.randn(32, 256)
    X = torch.randn(512, 256)
    X[:, :4] *= 20
    spec = WeightQuantSpec(4, "progressive", 128)
    perm = torch.argsort(X.abs().amax(0), descending=True)
    clip = search_clip_ratios(W, X, spec, perm=perm, a_bits=8)
    assert clip.shape == (32, 2) and (clip <= 1).all() and (clip >= 0.5).all()
    Wq0 = quantize_weight(W, spec)
    Wq1 = quantize_weight(W, spec, clip=clip, perm=perm)
    err0 = ((X @ Wq0.T - X @ W.T) ** 2).mean()
    err1 = ((X @ Wq1.T - X @ W.T) ** 2).mean()
    assert err1 <= err0 * 1.05


# ----------------------------------------------------------------------------- attention
def test_index_softmax_close_to_softmax():
    torch.manual_seed(0)
    B, H, L, D = 2, 4, 64, 64
    q, k, v = torch.randn(B, H, L, D), torch.randn(B, H, L, D), torch.randn(B, H, L, D)
    valid = torch.ones(L, L, dtype=torch.bool).tril()[None, None]
    scaling = 1 / math.sqrt(D)
    ref = torch.softmax((q @ k.transpose(-1, -2) * scaling).masked_fill(~valid, -1e9), -1) @ v
    for mode in ("int", "quant_only"):
        out = integer_attention(q, k, v, valid, scaling, AttnQuantState(mode=mode))
        cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
        assert cos > 0.99, (mode, cos)
    # more LUT entries -> better approximation
    e = []
    for b in (3, 5, 7):
        out = integer_attention(q, k, v, valid, scaling, AttnQuantState(mode="int", lut_bits=b))
        e.append((out - ref).pow(2).mean().item())
    assert e[2] < e[0]


def test_custom_attention_matches_eager():
    m, ref = tiny_model()
    ids = torch.randint(0, 128, (2, 24))
    assert torch.allclose(logits(m, ids), logits(ref, ids), atol=1e-4)
    # with left padding
    am = torch.ones_like(ids)
    am[0, :5] = 0
    a, b = logits(m, ids, attention_mask=am), logits(ref, ids, attention_mask=am)
    assert torch.allclose(a[0, 5:], b[0, 5:], atol=1e-4) and torch.allclose(a[1], b[1], atol=1e-4)


# ----------------------------------------------------------------------------- transforms
@pytest.mark.parametrize("hidden", [96, 128])   # 96 = 12 * 8 exercises the Paley construction
def test_transforms_are_function_preserving(hidden):
    m, ref = tiny_model(hidden=hidden)
    ids = torch.randint(0, 128, (2, 24))
    base = logits(ref, ids)
    q = QuantConfig(rotation="hadamard", smooth_output=True, smooth_attention=True)
    quantize_model(m, q, calib_batches(), log=lambda *_: None)
    assert m.lm_head.weight.data_ptr() != m.model.embed_tokens.weight.data_ptr()
    out = logits(m, ids)
    assert torch.allclose(out, base, atol=2e-4), (out - base).abs().max()


def test_full_qoq_intattn_pipeline():
    m, ref = tiny_model()
    ids = torch.randint(0, 128, (2, 24))
    base = logits(ref, ids)
    for preset in ["w8a8", "w4a8", "qoq", "qoq_intattn", "nf4", "w4a8kv4_g128_rtn_intattn"]:
        mm, _ = tiny_model()
        art, _, _ = quantize_model(mm, make_quant_config(preset), calib_batches(), log=lambda *_: None)
        out = logits(mm, ids)
        assert torch.isfinite(out).all(), preset
        cos = torch.nn.functional.cosine_similarity(out.flatten(), base.flatten(), dim=0)
        assert cos > 0.9, (preset, cos)


def test_runtime_switch():
    m, _ = tiny_model()
    quantize_model(m, make_quant_config("qoq_intattn"), calib_batches(), log=lambda *_: None)
    ids = torch.randint(0, 128, (2, 16))
    a = logits(m, ids)
    set_runtime(m, dict(a_bits=16, kv_bits=16, attn_mode="fp"))
    b = logits(m, ids)
    assert not torch.allclose(a, b)
    assert m.model.layers[0].self_attn.qft_attn.mode == "fp"


# ----------------------------------------------------------------------------- LoRA
def test_lora_train_merge():
    m, _ = tiny_model()
    _, fpw, _ = quantize_model(m, make_quant_config("qoq_intattn"), calib_batches(), keep_fp_weights=True,
                               log=lambda *_: None)
    ids = torch.randint(0, 128, (2, 16))
    before = logits(m, ids)
    info = add_lora(m, LoraConfig(r=4, alpha=8, dropout=0.0))
    assert info["lora_modules"] == 14
    assert torch.allclose(logits(m, ids), before)               # B = 0
    # gradients flow through all quantizers (STE) into LoRA params
    m.train()
    loss = m(input_ids=ids, labels=ids).loss
    loss.backward()
    grads = [p.grad for n, p in m.named_parameters() if "lora_B" in n]
    assert all(g is not None and g.abs().sum() > 0 for g in grads)
    base_grads = [p.grad for n, p in m.named_parameters() if "lora_" not in n]
    assert all(g is None for g in base_grads)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.01)
    m.eval()
    unmerged = logits(m, ids)
    m2 = copy.deepcopy(m)
    merge_lora(m2, "exact")
    assert torch.allclose(logits(m2, ids), unmerged, atol=1e-3)
    m3 = copy.deepcopy(m)
    merge_lora(m3, "requant_deq")                                  # re-quantized: close but not equal
    out3 = logits(m3, ids)
    assert torch.isfinite(out3).all()
    m4 = copy.deepcopy(m)
    merge_lora(m4, "requant_fp", fpw)
    assert torch.isfinite(logits(m4, ids)).all()


def test_loftq_reduces_error():
    m, _ = tiny_model()
    _, fpw, _ = quantize_model(m, make_quant_config("w4a16_g128"), calib_batches(), keep_fp_weights=True,
                               log=lambda *_: None)
    lin = m.model.layers[0].mlp.down_proj
    W = fpw["layers.0.down_proj"].float()
    e0 = (W - lin.weight.float()).norm()
    add_lora(m, LoraConfig(r=8, alpha=16, init="loftq", loftq_iters=3), fpw)
    e1 = (W - lin.weight.float() - lin.lora_delta()).norm()
    assert e1 < e0


def test_ft_then_ptq_unwrap():
    m, _ = tiny_model()
    quantize_model(m, QuantConfig(), log=lambda *_: None)
    add_lora(m, LoraConfig(r=4))
    merge_lora(m, "exact")
    unwrap_model(m)
    assert all(type(l).__name__ == "Linear" for _, _, l in iter_linears(m))
    quantize_model(m, make_quant_config("qoq_intattn"), calib_batches(), log=lambda *_: None)
    assert torch.isfinite(logits(m, torch.randint(0, 128, (1, 8)))).all()


# ----------------------------------------------------------------------------- generation
def test_generation_with_cache():
    m, ref = tiny_model()
    ids = torch.randint(3, 128, (2, 10))
    am = torch.ones_like(ids)
    am[0, :3] = 0
    kw = dict(max_new_tokens=8, do_sample=False, pad_token_id=0)
    quantize_model(m, QuantConfig(), log=lambda *_: None)
    g1 = m.generate(input_ids=ids, attention_mask=am, **kw)
    g2 = ref.generate(input_ids=ids, attention_mask=am, **kw)
    assert torch.equal(g1, g2)
    m2, _ = tiny_model()
    quantize_model(m2, make_quant_config("qoq_intattn"), calib_batches(), log=lambda *_: None)
    g3 = m2.generate(input_ids=ids, attention_mask=am, **kw)
    assert g3.shape == g1.shape


# ----------------------------------------------------------------------------- config
def test_config_merge():
    base = {"quant": "qoq_intattn", "lora": {"r": 16}}
    d = deep_update(base, {"quant": {"a_bits": 16}, "lora.r": 8})
    cfg = config_from_dict(d)
    assert cfg.quant.a_bits == 16 and cfg.quant.attn_mode == "int" and cfg.lora.r == 8
    d = deep_update(base, {"quant": "fp16"})
    assert config_from_dict(d).quant.is_fp
    with pytest.raises(KeyError):
        config_from_dict({"train_runtime": {"w_bits": 4}})


# ----------------------------------------------------------------------------- loss
def test_chunked_loss_matches_hf():
    from qft.train import causal_lm_loss
    m, _ = tiny_model()
    quantize_model(m, make_quant_config("qoq_intattn"), calib_batches(), log=lambda *_: None)
    add_lora(m, LoraConfig(r=4, alpha=8, dropout=0.0))
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.01)
    ids = torch.randint(0, 128, (3, 20))
    labels = ids.clone()
    labels[:, :7] = -100                       # prompt tokens ignored
    am = torch.ones_like(ids)
    am[0, :4] = 0                              # left padding
    labels[0, :4] = -100
    pos = (am.cumsum(-1) - 1).clamp(min=0)
    batch = dict(input_ids=ids, attention_mask=am, position_ids=pos, labels=labels)
    m.train()
    ref = m(**batch).loss
    g_ref = torch.autograd.grad(ref, [p for n, p in m.named_parameters() if "lora_B" in n])
    ours = causal_lm_loss(m, batch, chunk=5)    # several chunks
    g_ours = torch.autograd.grad(ours, [p for n, p in m.named_parameters() if "lora_B" in n])
    assert torch.allclose(ours, ref, atol=1e-5), (ours, ref)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(g_ours, g_ref))
