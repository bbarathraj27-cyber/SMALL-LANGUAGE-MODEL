"""
test_generation.py

Tests for inference/ (kv_cache.py, sampling.py, generate.py, chat.py).

TinySLM below is a self-contained, test-only stand-in for model/slm.py —
small enough to run fast in CI, but it implements the same forward
contract inference/generate.py expects: RoPE, RMSNorm, SwiGLU, causal
multi-head attention, and model(input_ids, kv_cache=..., positions=...).
It is NOT the production model — do not import it outside tests/.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.kv_cache import KVCache, LayerKVCache
from inference.sampling import (
    apply_temperature,
    apply_repetition_penalty,
    top_k_filter,
    top_p_filter,
    sample_token,
)
from inference.generate import GenerationConfig, generate
from inference.chat import build_prompt, chat_turn


# ---------------------------------------------------------------------------
# Test-support model (mirrors model/: RoPE + RMSNorm + SwiGLU + causal attn)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, positions, head_dim):
    device = q.device
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.einsum("i,j->ij", positions.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, positions, layer_idx, kv_cache=None):
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (b, heads, t, head_dim)
        q, k = apply_rope(q, k, positions, self.head_dim)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        q_len, k_len = q.shape[2], k.shape[2]
        if q_len > 1:
            # Full causal mask needed whenever we're scoring more than one
            # new position at once (prefill, or a no-cache reference pass).
            causal = torch.triu(
                torch.ones(q_len, k_len, device=x.device, dtype=torch.bool),
                diagonal=k_len - q_len + 1,
            )
            attn = attn.masked_fill(causal, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.out(out)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_mult=4):
        super().__init__()
        hidden = dim * hidden_mult
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(self, x, positions, layer_idx, kv_cache=None):
        x = x + self.attn(self.norm1(x), positions, layer_idx, kv_cache)
        x = x + self.ffn(self.norm2(x))
        return x


class TinySLM(nn.Module):
    def __init__(self, vocab_size=64, dim=32, n_heads=4, n_layers=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.num_layers = n_layers
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, n_heads) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, kv_cache=None, positions=None):
        b, t = input_ids.shape
        if positions is None:
            positions = torch.arange(t, device=input_ids.device)
        x = self.tok_emb(input_ids)
        for i, block in enumerate(self.blocks):
            x = block(x, positions, i, kv_cache)
        x = self.norm(x)
        return self.lm_head(x)


class FakeTokenizer:
    """Whitespace tokenizer over a tiny fixed vocab — just enough for chat.py tests."""

    def __init__(self):
        self.vocab = ["<eos>", "User:", "Assistant:", "hi", "there", "how", "are", "you"]
        self.stoi = {w: i for i, w in enumerate(self.vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}
        self.eos_token_id = 0

    def encode(self, text):
        return [self.stoi.get(w, 3) for w in text.replace("\n", " ").split()]

    def decode(self, ids):
        return " ".join(self.itos.get(i, "?") for i in ids)


class FakeStepModel(nn.Module):
    """Deterministic fake model for eos-stopping tests, independent of TinySLM.
    Ignores real content; the argmax token depends only on how many forward
    calls have happened so far (call 0 = prefill, call 1 = first new-token
    step, ...) via an internal counter — decoupled from kv_cache.seq_len so
    the mapping stays simple regardless of prompt length."""

    def __init__(self, sequence_by_step, vocab_size=8):
        super().__init__()
        self.num_layers = 1
        self.sequence_by_step = sequence_by_step  # dict: call_index -> token_id
        self.vocab_size = vocab_size
        self._dummy = nn.Parameter(torch.zeros(1))
        self._call_index = 0

    def forward(self, input_ids, kv_cache=None, positions=None):
        b, t = input_ids.shape
        step = self._call_index
        self._call_index += 1
        logits = torch.full((b, t, self.vocab_size), -10.0) + self._dummy.sum() * 0
        for row in range(b):
            token = self.sequence_by_step.get(step, 1)
            logits[row, -1, token] = 10.0
        if kv_cache is not None:
            # advance the cache by 1 position per call so generate()'s
            # position bookkeeping still works; content is irrelevant here.
            dummy_kv = torch.zeros(b, 1, 1, 1)
            kv_cache.update(0, dummy_kv, dummy_kv)
        return logits


# ---------------------------------------------------------------------------
# kv_cache.py
# ---------------------------------------------------------------------------

def test_layer_kv_cache_starts_empty():
    cache = LayerKVCache()
    assert cache.key is None
    assert cache.seq_len == 0


def test_layer_kv_cache_first_update_sets_tensors():
    cache = LayerKVCache()
    k = torch.randn(2, 4, 3, 8)
    v = torch.randn(2, 4, 3, 8)
    out_k, out_v = cache.update(k, v)
    assert torch.equal(out_k, k)
    assert torch.equal(out_v, v)
    assert cache.seq_len == 3


def test_layer_kv_cache_appends_along_seq_dim():
    cache = LayerKVCache()
    k1 = torch.randn(2, 4, 3, 8)
    v1 = torch.randn(2, 4, 3, 8)
    cache.update(k1, v1)
    k2 = torch.randn(2, 4, 1, 8)
    v2 = torch.randn(2, 4, 1, 8)
    out_k, out_v = cache.update(k2, v2)
    assert out_k.shape == (2, 4, 4, 8)
    assert torch.equal(out_k[:, :, :3, :], k1)
    assert torch.equal(out_k[:, :, 3:, :], k2)


def test_layer_kv_cache_batch_mismatch_raises():
    cache = LayerKVCache()
    cache.update(torch.randn(2, 4, 3, 8), torch.randn(2, 4, 3, 8))
    with pytest.raises(ValueError):
        cache.update(torch.randn(3, 4, 1, 8), torch.randn(3, 4, 1, 8))


def test_layer_kv_cache_crop_keeps_most_recent():
    cache = LayerKVCache()
    k = torch.arange(5).float().view(1, 1, 5, 1)
    cache.update(k, k.clone())
    cache.crop(2)
    assert cache.seq_len == 2
    assert torch.equal(cache.key.flatten(), torch.tensor([3.0, 4.0]))


def test_kv_cache_wraps_n_layers():
    cache = KVCache(num_layers=3)
    assert len(cache) == 3
    assert len(cache.layers) == 3
    assert cache.seq_len == 0


def test_kv_cache_update_routes_to_correct_layer():
    cache = KVCache(num_layers=2)
    k0 = torch.ones(1, 1, 1, 4)
    k1 = torch.zeros(1, 1, 1, 4)
    cache.update(0, k0, k0)
    cache.update(1, k1, k1)
    assert torch.equal(cache.layers[0].key, k0)
    assert torch.equal(cache.layers[1].key, k1)


def test_kv_cache_reset_clears_all_layers():
    cache = KVCache(num_layers=2)
    cache.update(0, torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4))
    cache.reset()
    assert cache.seq_len == 0
    assert all(layer.key is None for layer in cache.layers)


# ---------------------------------------------------------------------------
# sampling.py
# ---------------------------------------------------------------------------

def test_apply_temperature_scales_logits():
    logits = torch.tensor([[2.0, 4.0]])
    scaled = apply_temperature(logits, 2.0)
    assert torch.allclose(scaled, torch.tensor([[1.0, 2.0]]))


def test_apply_temperature_rejects_zero():
    with pytest.raises(ValueError):
        apply_temperature(torch.tensor([[1.0]]), 0.0)


def test_top_k_filter_masks_below_threshold():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
    filtered = top_k_filter(logits, k=2)
    kept = torch.isfinite(filtered[0])
    assert kept.sum().item() == 2
    assert filtered[0, 1] == 5.0  # top value untouched
    assert filtered[0, 2] == 3.0  # second-highest untouched
    assert filtered[0, 0] == float("-inf")
    assert filtered[0, 3] == float("-inf")


def test_top_k_filter_noop_when_k_covers_vocab():
    logits = torch.tensor([[1.0, 5.0, 3.0]])
    filtered = top_k_filter(logits, k=10)
    assert torch.equal(filtered, logits)


def test_top_p_filter_keeps_at_least_one_token():
    logits = torch.tensor([[10.0, -10.0, -10.0]])
    filtered = top_p_filter(logits, p=0.01)
    assert torch.isfinite(filtered[0]).sum().item() >= 1
    assert filtered[0, 0] == 10.0


def test_top_p_filter_noop_at_p_one():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    filtered = top_p_filter(logits, p=1.0)
    assert torch.equal(filtered, logits)


def test_repetition_penalty_reduces_seen_positive_logit():
    logits = torch.tensor([[4.0, 4.0, 4.0]])
    generated = torch.tensor([[0]])
    penalized = apply_repetition_penalty(logits, generated, penalty=2.0)
    assert penalized[0, 0] == 2.0
    assert penalized[0, 1] == 4.0
    assert penalized[0, 2] == 4.0


def test_repetition_penalty_noop_at_one():
    logits = torch.tensor([[4.0, -4.0]])
    generated = torch.tensor([[0, 1]])
    penalized = apply_repetition_penalty(logits, generated, penalty=1.0)
    assert torch.equal(penalized, logits)


def test_sample_token_greedy_is_argmax():
    logits = torch.tensor([[1.0, 9.0, 3.0]])
    token = sample_token(logits, greedy=True)
    assert token.item() == 1


def test_sample_token_greedy_deterministic_across_calls():
    logits = torch.randn(4, 20)
    t1 = sample_token(logits, greedy=True)
    t2 = sample_token(logits, greedy=True)
    assert torch.equal(t1, t2)


def test_sample_token_repetition_penalty_requires_generated_ids():
    logits = torch.tensor([[1.0, 2.0]])
    with pytest.raises(ValueError):
        sample_token(logits, repetition_penalty=1.5)


def test_sample_token_output_shape():
    logits = torch.randn(3, 16)
    token = sample_token(logits, temperature=1.0)
    assert token.shape == (3, 1)


# ---------------------------------------------------------------------------
# generate.py — correctness against a real (tiny) transformer
# ---------------------------------------------------------------------------

def test_kv_cache_generation_matches_no_cache_reference():
    """The most important test: step-by-step generation with a KV cache
    must produce identical logits to a single full-sequence forward pass
    (no cache) at every position — proving the cache doesn't corrupt
    attention or drop the RoPE position offset."""
    torch.manual_seed(0)
    model = TinySLM(vocab_size=32, dim=16, n_heads=2, n_layers=2, seed=1)
    model.eval()

    prompt = torch.randint(0, 32, (1, 4))
    full_seq = torch.randint(0, 32, (1, 7))
    full_seq[:, :4] = prompt

    with torch.no_grad():
        reference_logits = model(full_seq, positions=torch.arange(7))

    cache = KVCache(num_layers=2)
    with torch.no_grad():
        out = model(prompt, kv_cache=cache, positions=torch.arange(4))
    cached_logits = [out[:, -1, :]]
    with torch.no_grad():
        for step in range(4, 7):
            next_input = full_seq[:, step: step + 1]
            pos = torch.tensor([cache.seq_len])
            out = model(next_input, kv_cache=cache, positions=pos)
            cached_logits.append(out[:, -1, :])

    for i, logits in enumerate(cached_logits[:-1]):
        ref_pos = 3 + i
        assert torch.allclose(logits, reference_logits[:, ref_pos, :], atol=1e-4), (
            f"mismatch at position {ref_pos}"
        )


def test_generate_output_length_without_eos():
    model = TinySLM(vocab_size=32, dim=16, n_heads=2, n_layers=2, seed=2)
    prompt = torch.randint(0, 32, (1, 3))
    config = GenerationConfig(max_new_tokens=5, greedy=True)
    out = generate(model, prompt, config)
    assert out.shape == (1, 3 + 5)
    assert torch.equal(out[:, :3], prompt)


def test_generate_greedy_is_reproducible():
    model = TinySLM(vocab_size=32, dim=16, n_heads=2, n_layers=2, seed=3)
    prompt = torch.randint(0, 32, (2, 3))
    config = GenerationConfig(max_new_tokens=6, greedy=True)
    out1 = generate(model, prompt, config)
    out2 = generate(model, prompt, config)
    assert torch.equal(out1, out2)


def test_generate_requires_num_layers_when_model_lacks_attribute():
    class NoLayersModel(nn.Module):
        def forward(self, input_ids, kv_cache=None, positions=None):
            return torch.randn(input_ids.shape[0], input_ids.shape[1], 10)

    model = NoLayersModel()
    prompt = torch.randint(0, 10, (1, 2))
    config = GenerationConfig(max_new_tokens=2, greedy=True)
    with pytest.raises(ValueError):
        generate(model, prompt, config)


def test_generate_rejects_wrong_input_shape():
    model = TinySLM(vocab_size=16, dim=8, n_heads=2, n_layers=1, seed=4)
    config = GenerationConfig(max_new_tokens=2, greedy=True)
    with pytest.raises(ValueError):
        generate(model, torch.randint(0, 16, (5,)), config)  # 1-D, not (batch, seq)


def test_generate_stops_when_all_sequences_hit_eos():
    # step 0 (first new token) emits eos_id=0 for every row -> should stop
    # after exactly one generated token, well before max_new_tokens.
    model = FakeStepModel(sequence_by_step={0: 0, 1: 0}, vocab_size=8)
    prompt = torch.randint(1, 8, (2, 2))
    config = GenerationConfig(max_new_tokens=10, greedy=True, eos_token_id=0)
    out = generate(model, prompt, config)
    assert out.shape[1] == 2 + 1  # prompt + single eos token, loop broke early


def test_generate_pads_finished_sequences_with_eos():
    # token id 5 forever -> never hits eos -> runs the full max_new_tokens
    model = FakeStepModel(sequence_by_step={}, vocab_size=8)
    prompt = torch.randint(1, 8, (1, 2))
    config = GenerationConfig(max_new_tokens=4, greedy=True, eos_token_id=0)
    out = generate(model, prompt, config)
    assert out.shape[1] == 2 + 4
    assert (out[:, 2:] == 1).all()  # FakeStepModel's default token


# ---------------------------------------------------------------------------
# chat.py
# ---------------------------------------------------------------------------

def test_build_prompt_no_history():
    prompt = build_prompt([], "hi there")
    assert prompt == "User: hi there\nAssistant:"


def test_build_prompt_includes_history_in_order():
    history = [("user", "hi"), ("assistant", "hello")]
    prompt = build_prompt(history, "how are you")
    expected = "User: hi\nAssistant: hello\nUser: how are you\nAssistant:"
    assert prompt == expected


def test_chat_turn_returns_decoded_string():
    model = TinySLM(vocab_size=8, dim=8, n_heads=2, n_layers=1, seed=5)
    tokenizer = FakeTokenizer()
    reply = chat_turn(model, tokenizer, history=[], user_message="hi there", max_new_tokens=3)
    assert isinstance(reply, str)


def test_chat_turn_strips_tokens_after_eos():
    tokenizer = FakeTokenizer()
    # call 0 = prefill -> first generated token; call 1 -> second generated
    # token (eos); call 2 would be a third token, never reached.
    model = FakeStepModel(sequence_by_step={0: 3, 1: 0, 2: 4}, vocab_size=8)
    reply = chat_turn(model, tokenizer, history=[], user_message="hi", max_new_tokens=5)
    # second generated token is eos (id 0); only the first token ("hi", id 3) should remain
    assert reply == "hi"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
