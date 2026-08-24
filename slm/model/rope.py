"""Rotary Position Embedding (RoPE).

Implements the rotary embedding scheme used in GPT-NeoX / LLaMA style
models. Rather than adding a positional vector to the token embedding,
RoPE rotates pairs of dimensions within the query/key vectors by an
angle proportional to token position, which lets attention scores
naturally encode relative position and generalizes better to sequence
lengths beyond those seen during training.
"""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Precomputes rotary embedding cos/sin tables.

    Args:
        head_dim: Dimensionality of each attention head. Must be even,
            since RoPE operates on consecutive pairs of dimensions.
        max_position_embeddings: Maximum sequence length to precompute
            cos/sin caches for.
        base: The RoPE base (theta). 10000.0 is the standard default
            used by GPT-NeoX, LLaMA, and most modern decoder-only LLMs.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        if max_position_embeddings <= 0:
            raise ValueError(
                f"max_position_embeddings must be positive, got {max_position_embeddings}"
            )

        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        cos_cache, sin_cache = self._build_cache(max_position_embeddings)
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def _build_cache(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)  # (seq_len, head_dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        return emb.cos(), emb.sin()

    def _extend_cache_if_needed(self, seq_len: int, device: torch.device) -> None:
        if seq_len > self.cos_cache.shape[0]:
            cos_cache, sin_cache = self._build_cache(seq_len)
            self.cos_cache = cos_cache.to(device)
            self.sin_cache = sin_cache.to(device)
            self.max_position_embeddings = seq_len

    def forward(
        self, seq_len: int, position_offset: int = 0, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin tables for the requested positions.

        Args:
            seq_len: Number of new positions to fetch cos/sin for.
            position_offset: Starting position index. Used during
                incremental (KV-cached) decoding where new tokens are
                generated one at a time after an existing prefix.
            device: Device to place the returned tensors on.

        Returns:
            Tuple (cos, sin), each of shape (seq_len, head_dim).
        """
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if position_offset < 0:
            raise ValueError(f"position_offset must be non-negative, got {position_offset}")

        total_len = position_offset + seq_len
        self._extend_cache_if_needed(total_len, device if device is not None else self.cos_cache.device)

        cos = self.cos_cache[position_offset:total_len]
        sin = self.sin_cache[position_offset:total_len]
        if device is not None:
            cos = cos.to(device)
            sin = sin.to(device)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input, the standard RoPE helper."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim).
        k: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim).
        cos: Cosine table of shape (seq_len, head_dim).
        sin: Sine table of shape (seq_len, head_dim).

    Returns:
        Tuple of rotated (q, k) tensors, same shapes as inputs.
    """
    if q.shape[-1] != cos.shape[-1]:
        raise ValueError(
            f"q head_dim ({q.shape[-1]}) does not match cos/sin dim ({cos.shape[-1]})"
        )
    if q.shape[-2] != cos.shape[0]:
        raise ValueError(
            f"q seq_len ({q.shape[-2]}) does not match cos/sin seq_len ({cos.shape[0]})"
        )

    # Broadcast cos/sin over (batch, num_heads) dims: (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
    cos_b = cos.unsqueeze(0).unsqueeze(0)
    sin_b = sin.unsqueeze(0).unsqueeze(0)

    q_rotated = (q * cos_b) + (_rotate_half(q) * sin_b)
    k_rotated = (k * cos_b) + (_rotate_half(k) * sin_b)
    return q_rotated, k_rotated
