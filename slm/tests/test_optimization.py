"""
test_optimization.py

Tests for optimization/ (quantize.py, export.py, benchmark_inference.py).

TinySLM here is the same kind of test-only stand-in used in
test_generation.py (RoPE + RMSNorm + SwiGLU + causal attention,
matching model/'s architecture) — small enough to quantize, export,
and benchmark quickly in CI. Not the production model.
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.generate import GenerationConfig

from optimization.quantize import (
    quantize_tensor,
    dequantize_tensor,
    QuantizedLinear,
    quantize_model,
    model_size_bytes,
)
from optimization.export import export_checkpoint, load_checkpoint
from optimization.benchmark_inference import (
    benchmark_forward_pass,
    benchmark_generate,
    compare_quantized_vs_full,
)


# ---------------------------------------------------------------------------
# Test-support model (same family as test_generation.py's TinySLM)
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
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, k, positions, self.head_dim)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        q_len, k_len = q.shape[2], k.shape[2]
        if q_len > 1:
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
    def __init__(self, vocab_size=32, dim=16, n_heads=2, n_layers=2, seed=0):
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


# ---------------------------------------------------------------------------
# quantize.py
# ---------------------------------------------------------------------------

def test_quantize_tensor_roundtrip_error_is_small():
    torch.manual_seed(0)
    weight = torch.randn(8, 16)
    qweight, scale = quantize_tensor(weight)
    dequantized = dequantize_tensor(qweight, scale)
    max_err = (weight - dequantized).abs().max().item()
    # int8 with per-row symmetric scaling: error bounded by roughly
    # half a quantization step, well under 1% of the typical weight range here.
    assert max_err < 0.05


def test_quantize_tensor_dtype_and_shape():
    weight = torch.randn(4, 10)
    qweight, scale = quantize_tensor(weight)
    assert qweight.dtype == torch.int8
    assert qweight.shape == weight.shape
    assert scale.shape == (4, 1)


def test_quantize_tensor_handles_all_zero_row():
    weight = torch.zeros(2, 5)
    qweight, scale = quantize_tensor(weight)
    assert torch.equal(qweight, torch.zeros_like(qweight))
    assert torch.isfinite(scale).all()


def test_quantize_tensor_rejects_non_2d():
    with pytest.raises(ValueError):
        quantize_tensor(torch.randn(3, 4, 5))


def test_quantized_linear_matches_float_within_tolerance():
    torch.manual_seed(1)
    linear = nn.Linear(16, 8, bias=True)
    qlinear = QuantizedLinear.from_linear(linear)

    x = torch.randn(5, 16)
    with torch.no_grad():
        float_out = linear(x)
        quant_out = qlinear(x)

    max_diff = (float_out - quant_out).abs().max().item()
    assert max_diff < 0.1  # weight-only int8 introduces small but nonzero error


def test_quantized_linear_preserves_bias_exactly():
    linear = nn.Linear(4, 3, bias=True)
    qlinear = QuantizedLinear.from_linear(linear)
    assert torch.equal(qlinear.bias, linear.bias.data)


def test_quantize_model_replaces_all_linear_layers():
    model = TinySLM(vocab_size=16, dim=8, n_heads=2, n_layers=1, seed=2)
    linear_count_before = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    model, replaced = quantize_model(model)
    linear_count_after = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    assert linear_count_after == 0
    assert len(replaced) == linear_count_before


def test_quantize_model_respects_exclude_names():
    model = TinySLM(vocab_size=16, dim=8, n_heads=2, n_layers=1, seed=3)
    model, replaced = quantize_model(model, exclude_names=["lm_head"])
    assert isinstance(model.lm_head, nn.Linear)
    assert not any("lm_head" in name for name in replaced)
    # everything else should still have been converted
    assert not any(isinstance(m, nn.Linear) for name, m in model.named_modules() if "lm_head" not in name and name != "")


def test_quantize_model_output_shape_unchanged():
    model = TinySLM(vocab_size=16, dim=8, n_heads=2, n_layers=1, seed=4)
    input_ids = torch.randint(0, 16, (2, 5))
    with torch.no_grad():
        out_before = model(input_ids)
    quantize_model(model)
    with torch.no_grad():
        out_after = model(input_ids)
    assert out_before.shape == out_after.shape


def test_model_size_bytes_smaller_after_quantization():
    model_fp = TinySLM(vocab_size=32, dim=32, n_heads=4, n_layers=2, seed=5)
    size_before = model_size_bytes(model_fp)

    model_q = TinySLM(vocab_size=32, dim=32, n_heads=4, n_layers=2, seed=5)
    quantize_model(model_q, exclude_names=["lm_head"])
    size_after = model_size_bytes(model_q)

    assert size_after < size_before


# ---------------------------------------------------------------------------
# export.py
# ---------------------------------------------------------------------------

def test_export_and_load_checkpoint_roundtrip():
    model = TinySLM(vocab_size=16, dim=8, n_heads=2, n_layers=1, seed=6)
    config = {"vocab_size": 16, "dim": 8, "n_heads": 2, "n_layers": 1}

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = str(Path(tmp) / "checkpoint")
        manifest_path = export_checkpoint(model, config, ckpt_path, tokenizer_meta={"vocab_size": 16})

        assert Path(manifest_path).exists()
        assert Path(ckpt_path + ".pt").exists()

        fresh_model = TinySLM(vocab_size=16, dim=8, n_heads=2, n_layers=1, seed=999)
        loaded_model, manifest = load_checkpoint(fresh_model, ckpt_path)

        input_ids = torch.randint(0, 16, (1, 4))
        with torch.no_grad():
            out_original = model(input_ids)
            out_loaded = loaded_model(input_ids)
        assert torch.allclose(out_original, out_loaded, atol=1e-5)
        assert manifest["config"] == config
        assert manifest["tokenizer"] == {"vocab_size": 16}


def test_export_manifest_contains_expected_fields():
    model = TinySLM(vocab_size=8, dim=8, n_heads=2, n_layers=1, seed=7)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = str(Path(tmp) / "ckpt")
        manifest_path = export_checkpoint(model, {"dim": 8}, ckpt_path)
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert "checksum_sha256" in manifest
        assert "num_parameters" in manifest
        assert manifest["num_parameters"] == sum(p.nelement() for p in model.parameters())


def test_load_checkpoint_detects_corrupted_weights():
    model = TinySLM(vocab_size=8, dim=8, n_heads=2, n_layers=1, seed=8)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = str(Path(tmp) / "ckpt")
        export_checkpoint(model, {"dim": 8}, ckpt_path)

        # Corrupt the saved weights by overwriting with a different state dict.
        corrupt_model = TinySLM(vocab_size=8, dim=8, n_heads=2, n_layers=1, seed=999)
        torch.save(corrupt_model.state_dict(), ckpt_path + ".pt")

        fresh_model = TinySLM(vocab_size=8, dim=8, n_heads=2, n_layers=1, seed=999)
        with pytest.raises(ValueError):
            load_checkpoint(fresh_model, ckpt_path)


def test_load_checkpoint_can_skip_checksum_check():
    model = TinySLM(vocab_size=8, dim=8, n_heads=2, n_layers=1, seed=9)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = str(Path(tmp) / "ckpt")
        export_checkpoint(model, {"dim": 8}, ckpt_path)

        corrupt_model = TinySLM(vocab_size=8, dim=8, n_heads=2, n_layers=1, seed=999)
        torch.save(corrupt_model.state_dict(), ckpt_path + ".pt")

        fresh_model = TinySLM(vocab_size=8, dim=8, n_heads=2, n_layers=1, seed=999)
        # Should not raise when verification is explicitly disabled.
        loaded_model, manifest = load_checkpoint(fresh_model, ckpt_path, verify_checksum=False)
        assert loaded_model is fresh_model


# ---------------------------------------------------------------------------
# benchmark_inference.py
# ---------------------------------------------------------------------------

def test_benchmark_forward_pass_returns_expected_keys():
    model = TinySLM(vocab_size=16, dim=8, n_heads=2, n_layers=1, seed=10)
    input_ids = torch.randint(0, 16, (1, 6))
    stats = benchmark_forward_pass(model, input_ids, runs=2, warmup=1)
    assert set(["mean_latency_s", "std_latency_s", "min_latency_s", "runs"]).issubset(stats.keys())
    assert stats["mean_latency_s"] > 0
    assert stats["runs"] == 2


def test_benchmark_generate_returns_expected_keys_and_positive_rate():
    model = TinySLM(vocab_size=16, dim=8, n_heads=2, n_layers=1, seed=11)
    input_ids = torch.randint(0, 16, (1, 3))
    config = GenerationConfig(max_new_tokens=4, greedy=True)
    stats = benchmark_generate(model, input_ids, config, runs=2, warmup=1)
    assert set(["mean_latency_s", "tokens_per_sec", "mean_tokens_generated", "runs"]).issubset(stats.keys())
    assert stats["tokens_per_sec"] > 0
    assert stats["mean_tokens_generated"] == 4


def test_compare_quantized_vs_full_reports_size_reduction():
    model_fp = TinySLM(vocab_size=16, dim=16, n_heads=2, n_layers=2, seed=12)
    model_q = TinySLM(vocab_size=16, dim=16, n_heads=2, n_layers=2, seed=12)
    quantize_model(model_q, exclude_names=["lm_head"])

    input_ids = torch.randint(0, 16, (1, 3))
    config = GenerationConfig(max_new_tokens=3, greedy=True)

    result = compare_quantized_vs_full(model_fp, model_q, input_ids, config, runs=1)

    assert "speedup" in result
    assert result["quantized_size_bytes"] < result["full_precision_size_bytes"]
    assert 0 < result["size_reduction_ratio"] < 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
