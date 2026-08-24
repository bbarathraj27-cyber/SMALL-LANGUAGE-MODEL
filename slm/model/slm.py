"""Top-level Small Language Model.

Assembles token embeddings, a stack of Transformer decoder blocks, a
final normalization layer, and a language-modeling head into a single
decoder-only causal language model. Also defines SLMConfig, the
dataclass used to load and validate configs/model.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any

import torch
import torch.nn as nn

from model.embeddings import TokenEmbedding
from model.rmsnorm import RMSNorm
from model.transformer_block import TransformerBlock
from model.lm_head import LMHead
from model.swiglu import compute_swiglu_intermediate_size


@dataclass
class SLMConfig:
    """Configuration for the SLM architecture.

    Mirrors the fields expected in configs/model.yaml. Field defaults
    correspond to the project's "recommended first version" spec:
    a ~100M parameter decoder-only Transformer.
    """

    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int | None = None  # auto-computed if None
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    initializer_range: float = 0.02
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    embed_dropout: float = 0.0
    bias: bool = False
    tie_word_embeddings: bool = True
    pad_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.max_position_embeddings <= 0:
            raise ValueError(
                f"max_position_embeddings must be positive, got {self.max_position_embeddings}"
            )
        if self.intermediate_size is None:
            self.intermediate_size = compute_swiglu_intermediate_size(self.hidden_size)
        elif self.intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be positive, got {self.intermediate_size}"
            )
        if self.pad_token_id is not None and not (0 <= self.pad_token_id < self.vocab_size):
            raise ValueError(
                f"pad_token_id ({self.pad_token_id}) out of range [0, {self.vocab_size})"
            )

    @classmethod
    def from_yaml(cls, path: str) -> "SLMConfig":
        """Loads a config from a YAML file.

        Args:
            path: Path to a YAML file containing config fields. Only
                keys matching SLMConfig fields are read; unknown keys
                raise an error to catch typos early.

        Returns:
            A validated SLMConfig instance.
        """
        import yaml

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a YAML mapping at top level in {path}, got {type(raw)}")

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        unknown_keys = set(raw.keys()) - valid_fields
        if unknown_keys:
            raise ValueError(
                f"Unknown config keys in {path}: {sorted(unknown_keys)}. "
                f"Valid keys are: {sorted(valid_fields)}"
            )

        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        """Returns the config as a plain dict, e.g. for checkpoint metadata."""
        return asdict(self)


class SLM(nn.Module):
    """Decoder-only causal language model.

    Args:
        config: An SLMConfig instance describing the architecture.
    """

    def __init__(self, config: SLMConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            initializer_range=config.initializer_range,
            pad_token_id=config.pad_token_id,
        )
        self.embed_dropout = nn.Dropout(config.embed_dropout)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    intermediate_size=config.intermediate_size,
                    max_position_embeddings=config.max_position_embeddings,
                    rope_theta=config.rope_theta,
                    norm_eps=config.norm_eps,
                    attn_dropout=config.attn_dropout,
                    resid_dropout=config.resid_dropout,
                    bias=config.bias,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        tied_weight = self.token_embedding.weight if config.tie_word_embeddings else None
        self.lm_head = LMHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            bias=config.bias,
            tied_weight=tied_weight,
        )

        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _scale_residual_projections(self) -> None:
        """Scales output-projection weights by 1/sqrt(2 * num_layers).

        This follows the GPT-2 / nanoGPT initialization scheme: since
        every layer adds a residual branch, scaling down the two
        projections that feed directly into the residual stream
        (attention output projection and FFN down projection) keeps
        activation variance from growing unboundedly with depth.
        """
        scale = 1.0 / (2 * self.config.num_layers) ** 0.5
        for layer in self.layers:
            with torch.no_grad():
                layer.attention.o_proj.weight.mul_(scale)
                layer.ffn.down_proj.weight.mul_(scale)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Counts total trainable parameters.

        Args:
            exclude_embeddings: If True, subtracts the token embedding
                parameter count (useful when weights are tied, to avoid
                double-counting the embedding/lm_head matrix).

        Returns:
            Number of trainable parameters.
        """
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if exclude_embeddings:
            total -= self.token_embedding.weight.numel()
        return total

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> dict[str, Any]:
        """Runs a forward pass through the model.

        Args:
            input_ids: LongTensor of shape (batch, seq_len).
            attention_mask: Optional padding mask of shape
                (batch, seq_len) with 1 for real tokens and 0 for
                padding. Internally converted to an additive mask.
            labels: Optional LongTensor of shape (batch, seq_len) for
                computing next-token-prediction cross-entropy loss.
                Positions with label value -100 are ignored in the loss
                (standard PyTorch convention for padding/ignored
                positions).
            past_key_values: Optional list (one entry per layer) of
                cached (key, value) tuples from previous decoding
                steps, for incremental generation.
            use_cache: If True, returns updated per-layer key/value
                caches for use in the next decoding step.

        Returns:
            Dict with keys:
                "logits": FloatTensor (batch, seq_len, vocab_size)
                "loss": scalar FloatTensor if labels was provided, else None
                "past_key_values": list of per-layer (key, value) tuples
                    if use_cache is True, else None
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2D (batch, seq_len), got shape {tuple(input_ids.shape)}"
            )

        batch_size, seq_len = input_ids.shape

        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError(
                f"past_key_values has {len(past_key_values)} entries, "
                f"expected {len(self.layers)} (one per layer)"
            )

        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.embed_dropout(hidden_states)

        additive_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"attention_mask shape {tuple(attention_mask.shape)} does not match "
                    f"input_ids shape {tuple(input_ids.shape)}"
                )
            # Convert (batch, seq_len) padding mask with 1=keep, 0=mask
            # into an additive mask of shape (batch, 1, 1, seq_len) that
            # broadcasts over heads and query positions.
            inverted = (1.0 - attention_mask.to(hidden_states.dtype))
            additive_mask = inverted.masked_fill(inverted.bool(), float("-inf"))
            additive_mask = additive_mask[:, None, None, :]

        present_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = (
            [] if use_cache else None
        )

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            hidden_states, present = layer(
                hidden_states,
                attention_mask=additive_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache:
                present_key_values.append(present)

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.shape != (batch_size, seq_len):
                raise ValueError(
                    f"labels shape {tuple(labels.shape)} does not match "
                    f"input_ids shape {tuple(input_ids.shape)}"
                )
            # Shift so that tokens < n predict token n (standard causal LM setup).
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "logits": logits,
            "loss": loss,
            "past_key_values": present_key_values,
        }
