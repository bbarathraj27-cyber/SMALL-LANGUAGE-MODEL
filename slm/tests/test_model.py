"""Unit tests for the model/ package.

Run with: python -m pytest tests/test_model.py -v
or directly: python tests/test_model.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.slm import SLM, SLMConfig
from model.rmsnorm import RMSNorm
from model.rope import RotaryEmbedding, apply_rotary_pos_emb
from model.swiglu import SwiGLUFFN, compute_swiglu_intermediate_size
from model.attention import CausalSelfAttention
from model.transformer_block import TransformerBlock
from model.embeddings import TokenEmbedding


def _tiny_config(**overrides) -> SLMConfig:
    defaults = dict(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=16,
    )
    defaults.update(overrides)
    return SLMConfig(**defaults)


def test_rmsnorm_output_shape_and_scale():
    norm = RMSNorm(hidden_size=16)
    x = torch.randn(2, 5, 16) * 10.0
    out = norm(x)
    assert out.shape == x.shape
    # RMS of output (before weight scaling, weight starts at 1) should be ~1
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)
    print("test_rmsnorm_output_shape_and_scale: PASSED")


def test_rope_shapes_and_rotation_changes_values():
    head_dim = 8
    rope = RotaryEmbedding(head_dim=head_dim, max_position_embeddings=32)
    cos, sin = rope(seq_len=5)
    assert cos.shape == (5, head_dim)
    assert sin.shape == (5, head_dim)

    q = torch.randn(1, 2, 5, head_dim)
    k = torch.randn(1, 2, 5, head_dim)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert not torch.allclose(q_rot, q)  # rotation should change values
    print("test_rope_shapes_and_rotation_changes_values: PASSED")


def test_rope_position_offset_matches_full_sequence():
    # cos/sin fetched incrementally (with offset) should match a slice
    # of the cos/sin fetched for the full sequence at once.
    head_dim = 8
    rope = RotaryEmbedding(head_dim=head_dim, max_position_embeddings=32)
    cos_full, sin_full = rope(seq_len=10, position_offset=0)
    cos_tail, sin_tail = rope(seq_len=3, position_offset=7)
    assert torch.allclose(cos_full[7:10], cos_tail)
    assert torch.allclose(sin_full[7:10], sin_tail)
    print("test_rope_position_offset_matches_full_sequence: PASSED")


def test_swiglu_intermediate_size_rounding():
    size = compute_swiglu_intermediate_size(hidden_size=768, multiple_of=64)
    assert size % 64 == 0
    assert size > 0
    print(f"test_swiglu_intermediate_size_rounding: PASSED (intermediate_size={size})")


def test_swiglu_forward_shape():
    ffn = SwiGLUFFN(hidden_size=32, intermediate_size=64)
    x = torch.randn(2, 5, 32)
    out = ffn(x)
    assert out.shape == x.shape
    print("test_swiglu_forward_shape: PASSED")


def test_token_embedding_rejects_out_of_range_ids():
    emb = TokenEmbedding(vocab_size=10, hidden_size=8)
    good_ids = torch.tensor([[0, 1, 9]])
    out = emb(good_ids)
    assert out.shape == (1, 3, 8)

    bad_ids = torch.tensor([[0, 1, 10]])  # 10 is out of range for vocab_size=10
    raised = False
    try:
        emb(bad_ids)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for out-of-range token id"
    print("test_token_embedding_rejects_out_of_range_ids: PASSED")


def test_attention_forward_shape():
    attn = CausalSelfAttention(hidden_size=32, num_heads=4, max_position_embeddings=16)
    x = torch.randn(2, 6, 32)
    out, present = attn(x, use_cache=False)
    assert out.shape == x.shape
    assert present is None
    print("test_attention_forward_shape: PASSED")


def test_attention_kv_cache_matches_full_forward():
    """The critical correctness test: generating token-by-token with a
    KV cache must produce identical output to a single full-sequence
    forward pass. If RoPE position offsets or cache concatenation are
    wrong, this test will catch it.
    """
    torch.manual_seed(0)
    attn = CausalSelfAttention(hidden_size=32, num_heads=4, max_position_embeddings=16)
    attn.eval()

    x = torch.randn(1, 5, 32)

    with torch.no_grad():
        full_out, _ = attn(x, use_cache=False)

        # Now replay incrementally: one token at a time using a growing cache.
        cache = None
        incremental_outputs = []
        for t in range(5):
            token = x[:, t : t + 1, :]
            out_t, cache = attn(token, past_key_value=cache, use_cache=True)
            incremental_outputs.append(out_t)
        incremental_out = torch.cat(incremental_outputs, dim=1)

    assert torch.allclose(full_out, incremental_out, atol=1e-4), (
        "KV-cache incremental attention diverged from full-sequence attention"
    )
    print("test_attention_kv_cache_matches_full_forward: PASSED")


def test_transformer_block_forward_shape():
    block = TransformerBlock(hidden_size=32, num_heads=4, max_position_embeddings=16)
    x = torch.randn(2, 6, 32)
    out, present = block(x, use_cache=False)
    assert out.shape == x.shape
    print("test_transformer_block_forward_shape: PASSED")


def test_slm_config_from_yaml(tmp_path=None):
    import tempfile

    yaml_content = """
vocab_size: 100
hidden_size: 16
num_layers: 2
num_heads: 2
max_position_embeddings: 32
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        path = f.name

    try:
        config = SLMConfig.from_yaml(path)
        assert config.vocab_size == 100
        assert config.hidden_size == 16
        assert config.num_layers == 2
    finally:
        os.remove(path)
    print("test_slm_config_from_yaml: PASSED")


def test_slm_config_rejects_unknown_keys():
    import tempfile

    yaml_content = "vocab_size: 100\nhidden_size: 16\nnum_layers: 2\nnum_heads: 2\ntotally_fake_key: 5\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        path = f.name

    raised = False
    try:
        SLMConfig.from_yaml(path)
    except ValueError:
        raised = True
    finally:
        os.remove(path)
    assert raised, "Expected ValueError for unknown config key"
    print("test_slm_config_rejects_unknown_keys: PASSED")


def test_slm_config_rejects_bad_head_division():
    raised = False
    try:
        SLMConfig(vocab_size=100, hidden_size=30, num_layers=2, num_heads=4)  # 30 not divisible by 4
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for hidden_size not divisible by num_heads"
    print("test_slm_config_rejects_bad_head_division: PASSED")


def test_slm_forward_shapes_and_loss():
    config = _tiny_config()
    model = SLM(config)
    model.eval()

    batch_size, seq_len = 2, 7
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    output = model(input_ids=input_ids, labels=labels)
    assert output["logits"].shape == (batch_size, seq_len, config.vocab_size)
    assert output["loss"] is not None
    assert output["loss"].dim() == 0  # scalar
    assert output["loss"].item() > 0
    print(f"test_slm_forward_shapes_and_loss: PASSED (loss={output['loss'].item():.4f})")


def test_slm_loss_ignores_masked_labels():
    config = _tiny_config()
    model = SLM(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 6))
    labels_all_ignored = torch.full((1, 6), -100, dtype=torch.long)

    with torch.no_grad():
        output = model(input_ids=input_ids, labels=labels_all_ignored)
    # cross_entropy over an all-ignored batch returns nan; verify it doesn't crash
    # and returns a tensor (nan is expected torch behavior here, not our bug).
    assert output["loss"].shape == ()
    print("test_slm_loss_ignores_masked_labels: PASSED")


def test_slm_gradient_flow():
    """Ensures gradients reach the embedding table, i.e. the whole
    network is actually connected end-to-end (would catch a disconnected
    residual stream or detached tensor bug).
    """
    config = _tiny_config()
    model = SLM(config)
    model.train()

    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    labels = torch.randint(0, config.vocab_size, (2, 5))

    output = model(input_ids=input_ids, labels=labels)
    output["loss"].backward()

    assert model.token_embedding.weight.grad is not None
    assert torch.any(model.token_embedding.weight.grad != 0)
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient reached parameter: {name}"
    print("test_slm_gradient_flow: PASSED")


def test_slm_weight_tying():
    config = _tiny_config(tie_word_embeddings=True)
    model = SLM(config)
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
    print("test_slm_weight_tying: PASSED")


def test_slm_no_weight_tying():
    config = _tiny_config(tie_word_embeddings=False)
    model = SLM(config)
    assert model.lm_head.weight.data_ptr() != model.token_embedding.weight.data_ptr()
    print("test_slm_no_weight_tying: PASSED")


def test_slm_kv_cache_generation_matches_full_forward():
    """End-to-end version of the attention-level cache test: generating
    through the *full* multi-layer model incrementally must match a
    single full-sequence forward pass.
    """
    torch.manual_seed(42)
    config = _tiny_config()
    model = SLM(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 6))

    with torch.no_grad():
        full_output = model(input_ids=input_ids, use_cache=False)
        full_logits = full_output["logits"]

        past_key_values = None
        incremental_logits = []
        for t in range(6):
            token = input_ids[:, t : t + 1]
            step_output = model(
                input_ids=token, past_key_values=past_key_values, use_cache=True
            )
            incremental_logits.append(step_output["logits"])
            past_key_values = step_output["past_key_values"]
        incremental_logits = torch.cat(incremental_logits, dim=1)

    assert torch.allclose(full_logits, incremental_logits, atol=1e-3), (
        "Full-model KV-cache generation diverged from full-sequence forward pass"
    )
    print("test_slm_kv_cache_generation_matches_full_forward: PASSED")


def test_slm_parameter_count_near_100m():
    """Sanity check that the actual recommended config.yaml produces a
    model in the ballpark of the ~100M parameter target from the spec.
    """
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "model.yaml"
    )
    config = SLMConfig.from_yaml(config_path)
    model = SLM(config)
    total_params = model.num_parameters()
    print(
        f"test_slm_parameter_count_near_100m: total_params={total_params:,} "
        f"({total_params / 1e6:.1f}M)"
    )
    # Allow a generous band since exact hyperparameters can be tuned;
    # this just guards against a gross architecture bug (e.g. an extra
    # order of magnitude).
    assert 50_000_000 <= total_params <= 200_000_000, (
        f"Parameter count {total_params:,} is far outside the expected "
        f"~100M range for the recommended config"
    )
    print("test_slm_parameter_count_near_100m: PASSED")


def test_invalid_config_raises():
    raised = False
    try:
        SLMConfig(vocab_size=-1)
    except ValueError:
        raised = True
    assert raised
    print("test_invalid_config_raises: PASSED")


def run_all_tests():
    test_fns = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    failures = []
    for fn in test_fns:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - test runner intentionally broad
            failures.append((fn.__name__, e))
            print(f"{fn.__name__}: FAILED -- {e}")

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} / {len(test_fns)} tests FAILED")
        for name, e in failures:
            print(f"  - {name}: {e}")
        sys.exit(1)
    else:
        print(f"All {len(test_fns)} tests PASSED")


if __name__ == "__main__":
    run_all_tests()
