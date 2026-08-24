"""Causal multi-head self-attention with rotary position embeddings.

Supports an optional key/value cache (past_key_value) so the same
module can be used both for full-sequence training/pretraining and for
efficient incremental decoding during inference (see
inference/kv_cache.py for the wrapper that manages cache lifetime
across a whole model).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rope import RotaryEmbedding, apply_rotary_pos_emb


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads. hidden_size must be
            divisible by num_heads.
        max_position_embeddings: Maximum sequence length supported,
            used to size the RoPE cache.
        rope_theta: Base frequency for rotary embeddings.
        attn_dropout: Dropout applied to attention weights.
        resid_dropout: Dropout applied to the attention output before
            the residual add (the residual add itself happens in
            TransformerBlock, not here).
        bias: Whether Q/K/V/O projections include a bias term.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if not (0.0 <= attn_dropout < 1.0):
            raise ValueError(f"attn_dropout must be in [0, 1), got {attn_dropout}")
        if not (0.0 <= resid_dropout < 1.0):
            raise ValueError(f"resid_dropout must be in [0, 1), got {resid_dropout}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_dropout_p = attn_dropout

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.resid_dropout = nn.Dropout(resid_dropout)

        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def _split_heads(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        # (batch, seq_len, hidden_size) -> (batch, num_heads, seq_len, head_dim)
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, hidden_size)
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Runs causal self-attention.

        Args:
            hidden_states: Input of shape (batch, seq_len, hidden_size).
            attention_mask: Optional additive mask of shape broadcastable
                to (batch, 1, seq_len, total_len), with 0 for positions
                to attend to and a large negative value (e.g. -inf) for
                positions to mask out. Used for padding masks; the
                causal mask itself is applied internally.
            past_key_value: Optional tuple of cached (key, value) tensors
                from previous decoding steps, each of shape
                (batch, num_heads, past_len, head_dim).
            use_cache: If True, returns the updated (key, value) cache
                for use in the next decoding step.

        Returns:
            Tuple of (attention_output, present_key_value). If
            use_cache is False, present_key_value is None.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected last dim {self.hidden_size}, got {hidden_size}"
            )

        query = self._split_heads(self.q_proj(hidden_states), batch_size, seq_len)
        key = self._split_heads(self.k_proj(hidden_states), batch_size, seq_len)
        value = self._split_heads(self.v_proj(hidden_states), batch_size, seq_len)

        position_offset = 0 if past_key_value is None else past_key_value[0].shape[2]
        cos, sin = self.rotary_emb(
            seq_len=seq_len, position_offset=position_offset, device=hidden_states.device
        )
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)

        present_key_value = (key, value) if use_cache else None

        total_len = key.shape[2]
        is_causal = attention_mask is None and total_len == seq_len

        if attention_mask is not None or not is_causal:
            # Build an explicit additive mask combining causal structure
            # with any provided padding mask, since scaled_dot_product_attention's
            # `is_causal` flag only supports the simple full-prefix case.
            causal_mask = torch.full(
                (seq_len, total_len), float("-inf"), device=hidden_states.device, dtype=query.dtype
            )
            causal_mask = torch.triu(causal_mask, diagonal=1 + position_offset)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, total_len)
            if attention_mask is not None:
                causal_mask = causal_mask + attention_mask
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=causal_mask,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=False,
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=True,
            )

        attn_output = self._merge_heads(attn_output, batch_size, seq_len)
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present_key_value
