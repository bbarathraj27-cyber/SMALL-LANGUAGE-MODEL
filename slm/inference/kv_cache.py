"""
kv_cache.py

Key/Value cache for autoregressive generation.

During generation we run the model one new token at a time. Without a
cache, each step would require re-encoding the entire sequence so far
through every attention layer (O(n^2) work overall). A KV cache stores
the key and value projections computed for every past position, per
layer, so each new step only computes attention for the *new* token
against all past keys/values (O(n) work per step, O(n^2) total instead
of O(n^3)).

This module is architecture-agnostic: it only knows about tensor
shapes (batch, heads, seq_len, head_dim), not about RoPE, RMSNorm, or
any other model internals. model/attention.py is responsible for
calling update() with newly computed K/V and reading back the full
cached K/V it needs for attention.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch


@dataclass
class LayerKVCache:
    """K/V cache for a single transformer layer."""

    key: Optional[torch.Tensor] = None    # (batch, n_kv_heads, seq_len, head_dim)
    value: Optional[torch.Tensor] = None  # (batch, n_kv_heads, seq_len, head_dim)

    def update(self, new_key: torch.Tensor, new_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append newly computed key/value tensors to the cache and return
        the full cached key/value (past + new) for attention.

        new_key, new_value: (batch, n_kv_heads, new_seq_len, head_dim)
        """
        if self.key is None:
            self.key = new_key
            self.value = new_value
        else:
            if self.key.shape[0] != new_key.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: cache has {self.key.shape[0]}, "
                    f"new key has {new_key.shape[0]}"
                )
            self.key = torch.cat([self.key, new_key], dim=2)
            self.value = torch.cat([self.value, new_value], dim=2)
        return self.key, self.value

    @property
    def seq_len(self) -> int:
        return 0 if self.key is None else self.key.shape[2]

    def crop(self, max_len: int) -> None:
        """Drop cached entries beyond max_len positions (keeps the most recent)."""
        if self.key is not None and self.key.shape[2] > max_len:
            self.key = self.key[:, :, -max_len:, :]
            self.value = self.value[:, :, -max_len:, :]


@dataclass
class KVCache:
    """K/V cache for the full stack of transformer layers."""

    num_layers: int
    layers: List[LayerKVCache] = field(default_factory=list)

    def __post_init__(self):
        if not self.layers:
            self.layers = [LayerKVCache() for _ in range(self.num_layers)]

    def update(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor):
        return self.layers[layer_idx].update(new_key, new_value)

    @property
    def seq_len(self) -> int:
        return self.layers[0].seq_len if self.layers else 0

    def crop(self, max_len: int) -> None:
        for layer in self.layers:
            layer.crop(max_len)

    def reset(self) -> None:
        self.layers = [LayerKVCache() for _ in range(self.num_layers)]

    def __len__(self) -> int:
        return self.num_layers
