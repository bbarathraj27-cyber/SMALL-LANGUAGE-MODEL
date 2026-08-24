"""A single pre-norm Transformer decoder block.

Uses the standard modern-LLM layout:

    x = x + Attention(RMSNorm(x))
    x = x + SwiGLU(RMSNorm(x))

Pre-normalization (norm applied before the sublayer rather than after)
gives more stable gradients at initialization than the original
post-norm Transformer, which matters for training stability without
extensive learning-rate warmup tuning.
"""

import torch
import torch.nn as nn

from model.attention import CausalSelfAttention
from model.rmsnorm import RMSNorm
from model.swiglu import SwiGLUFFN


class TransformerBlock(nn.Module):
    """One decoder layer: pre-norm attention + pre-norm SwiGLU FFN.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        intermediate_size: SwiGLU intermediate dimension. If None, it
            is derived automatically from hidden_size.
        max_position_embeddings: Maximum sequence length for RoPE.
        rope_theta: RoPE base frequency.
        norm_eps: Epsilon for RMSNorm layers.
        attn_dropout: Dropout probability inside attention weights.
        resid_dropout: Dropout probability applied to attention and
            FFN outputs before the residual add.
        bias: Whether linear layers include bias terms.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int | None = None,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-6,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.input_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.attention = CausalSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            bias=bias,
        )
        self.post_attention_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.ffn = SwiGLUFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=bias,
            dropout=resid_dropout,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Runs one decoder block.

        Args:
            hidden_states: Input of shape (batch, seq_len, hidden_size).
            attention_mask: Optional additive attention mask, see
                CausalSelfAttention.forward for shape details.
            past_key_value: Optional cached (key, value) from previous
                decoding steps.
            use_cache: If True, returns the updated key/value cache.

        Returns:
            Tuple of (output_hidden_states, present_key_value).
        """
        residual = hidden_states
        normed = self.input_norm(hidden_states)
        attn_output, present_key_value = self.attention(
            normed,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        normed = self.post_attention_norm(hidden_states)
        ffn_output = self.ffn(normed)
        hidden_states = residual + ffn_output

        return hidden_states, present_key_value
