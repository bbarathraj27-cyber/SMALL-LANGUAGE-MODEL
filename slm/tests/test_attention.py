"""Dedicated unit tests for model/attention.py.

Run with: python tests/test_attention.py

Note: test_model.py also exercises CausalSelfAttention indirectly as
part of full-model tests (test_slm_kv_cache_generation_matches_full_forward,
etc.). This file tests the attention module in isolation, with more
granular coverage of masking, causality, and cache behavior.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.attention import CausalSelfAttention


def test_attention_output_shape():
    attn = CausalSelfAttention(hidden_size=32, num_heads=4, max_position_embeddings=16)
    x = torch.randn(2, 6, 32)
    out, present = attn(x, use_cache=False)
    assert out.shape == x.shape
    assert present is None
    print("test_attention_output_shape: PASSED")


def test_attention_rejects_wrong_hidden_size():
    attn = CausalSelfAttention(hidden_size=32, num_heads=4, max_position_embeddings=16)
    x = torch.randn(2, 6, 64)  # wrong last dim
    raised = False
    try:
        attn(x)
    except ValueError:
        raised = True
    assert raised
    print("test_attention_rejects_wrong_hidden_size: PASSED")


def test_attention_rejects_bad_head_division():
    raised = False
    try:
        CausalSelfAttention(hidden_size=30, num_heads=4)  # 30 not divisible by 4
    except ValueError:
        raised = True
    assert raised
    print("test_attention_rejects_bad_head_division: PASSED")


def test_attention_is_causal_future_tokens_dont_affect_past_outputs():
    """The defining property of causal attention: output at position i
    must be identical whether or not tokens after position i exist in
    the input. If future tokens leak into past outputs, this indicates
    a masking bug that would let the model "cheat" during training.
    """
    torch.manual_seed(0)
    attn = CausalSelfAttention(hidden_size=32, num_heads=4, max_position_embeddings=16)
    attn.eval()

    full_seq = torch.randn(1, 8, 32)
    prefix_seq = full_seq[:, :4, :].clone()

    with torch.no_grad():
        full_out, _ = attn(full_seq, use_cache=False)
        prefix_out, _ = attn(prefix_seq, use_cache=False)

    # Outputs at positions 0-3 must match regardless of whether positions
    # 4-7 exist in the input.
    assert torch.allclose(full_out[:, :4, :], prefix_out, atol=1e-5), (
        "Future tokens leaked into earlier positions' outputs -- causal "
        "masking is broken"
    )
    print("test_attention_is_causal_future_tokens_dont_affect_past_outputs: PASSED")


def test_attention_different_batches_are_independent():
    """Verifies no cross-batch leakage: each item in a batch should
    produce the same output whether processed alone or alongside others.
    """
    torch.manual_seed(1)
    attn = CausalSelfAttention(hidden_size=16, num_heads=2, max_position_embeddings=16)
    attn.eval()

    x1 = torch.randn(1, 5, 16)
    x2 = torch.randn(1, 5, 16)
    batched = torch.cat([x1, x2], dim=0)

    with torch.no_grad():
        out1, _ = attn(x1, use_cache=False)
        out2, _ = attn(x2, use_cache=False)
        out_batched, _ = attn(batched, use_cache=False)

    assert torch.allclose(out1[0], out_batched[0], atol=1e-5)
    assert torch.allclose(out2[0], out_batched[1], atol=1e-5)
    print("test_attention_different_batches_are_independent: PASSED")


def test_attention_padding_mask_ignores_padded_positions():
    """A padding mask that marks trailing positions as padding should
    prevent real tokens from attending to them, and outputs at real
    positions should be unaffected by what padded positions contain.
    """
    torch.manual_seed(2)
    attn = CausalSelfAttention(hidden_size=16, num_heads=2, max_position_embeddings=16)
    attn.eval()

    real_tokens = torch.randn(1, 4, 16)
    pad_a = torch.randn(1, 2, 16)
    pad_b = torch.randn(1, 2, 16)  # different padding content

    seq_a = torch.cat([real_tokens, pad_a], dim=1)
    seq_b = torch.cat([real_tokens, pad_b], dim=1)

    # Mask: 1 for real tokens, -inf for padded positions (additive mask,
    # broadcastable to (batch, 1, seq_len, total_len)).
    mask = torch.zeros(1, 1, 6, 6)
    mask[:, :, :, 4:] = float("-inf")  # nobody may attend to padded positions

    with torch.no_grad():
        out_a, _ = attn(seq_a, attention_mask=mask, use_cache=False)
        out_b, _ = attn(seq_b, attention_mask=mask, use_cache=False)

    # Outputs at the real-token positions (0-3) should be identical
    # regardless of what garbage is in the padded positions, since the
    # mask prevents attending to them.
    assert torch.allclose(out_a[:, :4, :], out_b[:, :4, :], atol=1e-4), (
        "Padded position content leaked into real-token outputs -- "
        "padding mask is not being applied correctly"
    )
    print("test_attention_padding_mask_ignores_padded_positions: PASSED")


def test_attention_kv_cache_single_step_matches_full_forward():
    torch.manual_seed(3)
    attn = CausalSelfAttention(hidden_size=32, num_heads=4, max_position_embeddings=16)
    attn.eval()

    x = torch.randn(1, 5, 32)

    with torch.no_grad():
        full_out, _ = attn(x, use_cache=False)

        cache = None
        outputs = []
        for t in range(5):
            token = x[:, t : t + 1, :]
            out_t, cache = attn(token, past_key_value=cache, use_cache=True)
            outputs.append(out_t)
        incremental_out = torch.cat(outputs, dim=1)

    assert torch.allclose(full_out, incremental_out, atol=1e-4)
    print("test_attention_kv_cache_single_step_matches_full_forward: PASSED")


def test_attention_kv_cache_grows_correctly():
    attn = CausalSelfAttention(hidden_size=16, num_heads=2, max_position_embeddings=16)
    attn.eval()

    x1 = torch.randn(1, 3, 16)
    x2 = torch.randn(1, 1, 16)

    with torch.no_grad():
        _, cache1 = attn(x1, use_cache=True)
        key1, value1 = cache1
        assert key1.shape[2] == 3
        assert value1.shape[2] == 3

        _, cache2 = attn(x2, past_key_value=cache1, use_cache=True)
        key2, value2 = cache2
        assert key2.shape[2] == 4  # 3 cached + 1 new
        assert value2.shape[2] == 4
    print("test_attention_kv_cache_grows_correctly: PASSED")


def test_attention_no_cache_returned_when_use_cache_false():
    attn = CausalSelfAttention(hidden_size=16, num_heads=2, max_position_embeddings=16)
    x = torch.randn(1, 3, 16)
    _, present = attn(x, use_cache=False)
    assert present is None
    print("test_attention_no_cache_returned_when_use_cache_false: PASSED")


def test_attention_gradient_flows_through_all_projections():
    attn = CausalSelfAttention(hidden_size=16, num_heads=2, max_position_embeddings=16)
    attn.train()
    x = torch.randn(2, 4, 16, requires_grad=True)
    out, _ = attn(x, use_cache=False)
    loss = out.sum()
    loss.backward()

    for name, param in attn.named_parameters():
        assert param.grad is not None, f"No gradient reached {name}"
        assert torch.any(param.grad != 0), f"Gradient for {name} is all zeros"
    assert x.grad is not None
    print("test_attention_gradient_flows_through_all_projections: PASSED")


def test_attention_dropout_only_active_in_training_mode():
    """attn_dropout should have zero effect in eval mode (deterministic
    output across repeated calls), and should not crash in train mode.
    """
    attn = CausalSelfAttention(
        hidden_size=32, num_heads=4, max_position_embeddings=16, attn_dropout=0.5, resid_dropout=0.5
    )
    x = torch.randn(1, 5, 32)

    attn.eval()
    with torch.no_grad():
        out1, _ = attn(x, use_cache=False)
        out2, _ = attn(x, use_cache=False)
    assert torch.equal(out1, out2), "Eval mode should be deterministic (no dropout)"

    attn.train()
    out3, _ = attn(x, use_cache=False)  # should not crash with dropout active
    assert out3.shape == x.shape
    print("test_attention_dropout_only_active_in_training_mode: PASSED")


def run_all_tests():
    test_fns = [
        obj
        for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = []
    for fn in test_fns:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
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