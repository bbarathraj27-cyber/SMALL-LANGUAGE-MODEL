"""
generate.py

Autoregressive text generation using a KV cache for O(n) per-step cost.

Model contract: model(input_ids, kv_cache=..., positions=...) must
return logits of shape (batch, seq_len, vocab_size) and update the
passed-in KVCache in place (see model/attention.py and
inference/kv_cache.py). `positions` is the absolute position of each
token in input_ids, needed so RoPE stays correct after the prompt
"prefill" step (subsequent single-token steps aren't at position 0).

This file has no knowledge of tokenizer/text; it works entirely in
token-id space so it composes cleanly with tokenizer/tokenizer.py and
inference/chat.py.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from .kv_cache import KVCache
from .sampling import sample_token


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    greedy: bool = False
    repetition_penalty: float = 1.0
    eos_token_id: Optional[int] = None


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    config: GenerationConfig,
    num_layers: Optional[int] = None,
) -> torch.Tensor:
    """
    input_ids: (batch, prompt_len) token ids on the correct device.
    Returns: (batch, prompt_len + generated_len) token ids, including the
    original prompt.

    Stops early once every sequence in the batch has produced
    eos_token_id, but always returns a rectangular tensor: sequences
    that finish early are padded with eos_token_id for the remaining
    steps so the batch stays a single tensor (mask it out downstream
    using eos position, same as chat.py does).
    """
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be (batch, seq_len), got shape {tuple(input_ids.shape)}")

    model.eval()
    device = input_ids.device
    batch_size, prompt_len = input_ids.shape

    n_layers = num_layers if num_layers is not None else getattr(model, "num_layers", None)
    if n_layers is None:
        raise ValueError("num_layers must be passed (or model.num_layers must exist) to build a KV cache")

    cache = KVCache(num_layers=n_layers)
    generated = input_ids
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Prefill: run the full prompt through once, populating the cache for
    # every position so subsequent steps only process one new token.
    logits = model(input_ids, kv_cache=cache, positions=torch.arange(prompt_len, device=device))
    next_logits = logits[:, -1, :]

    for _ in range(config.max_new_tokens):
        next_token = sample_token(
            next_logits,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            greedy=config.greedy,
            generated_ids=generated,
            repetition_penalty=config.repetition_penalty,
        )

        if config.eos_token_id is not None:
            # Force already-finished sequences to keep emitting eos so the
            # batch tensor stays rectangular without corrupting live ones.
            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, config.eos_token_id),
                next_token,
            )
            finished = finished | (next_token.squeeze(-1) == config.eos_token_id)

        generated = torch.cat([generated, next_token], dim=1)

        if config.eos_token_id is not None and bool(finished.all()):
            break

        position = torch.tensor([cache.seq_len], device=device)
        logits = model(next_token, kv_cache=cache, positions=position)
        next_logits = logits[:, -1, :]

    return generated
