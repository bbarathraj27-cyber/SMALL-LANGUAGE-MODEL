"""
sampling.py

Token sampling strategies for autoregressive generation:
  - greedy decoding (argmax)
  - temperature scaling
  - top-k filtering
  - top-p (nucleus) filtering
  - repetition penalty

All filter/scale functions operate on raw logits of shape
(batch, vocab_size) for the *current* step (already sliced to the
last position) so they compose in any order before sampling.
"""

from typing import Optional
import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0; use greedy=True for argmax decoding instead")
    return logits / temperature


def apply_repetition_penalty(
    logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float
) -> torch.Tensor:
    """
    Penalize tokens that already appear in generated_ids (batch, seq_len).
    penalty > 1.0 discourages repeats; penalty == 1.0 is a no-op.
    Follows the standard convention: positive logits are divided by the
    penalty (pushed toward zero), negative logits are multiplied by it
    (pushed more negative) — either way the token becomes less likely.
    """
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    for b in range(logits.shape[0]):
        seen = torch.unique(generated_ids[b])
        seen = seen[(seen >= 0) & (seen < logits.shape[-1])]
        if seen.numel() == 0:
            continue
        scores = logits[b, seen]
        scores = torch.where(scores > 0, scores / penalty, scores * penalty)
        logits[b, seen] = scores
    return logits


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Mask out all but the top-k logits per row with -inf."""
    if k is None or k <= 0 or k >= logits.shape[-1]:
        return logits
    values, _ = torch.topk(logits, k, dim=-1)
    min_keep = values[:, -1].unsqueeze(-1)
    return torch.where(logits < min_keep, torch.full_like(logits, float("-inf")), logits)


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus filtering: keep the smallest set of highest-probability tokens
    whose cumulative probability exceeds p, mask the rest with -inf.
    Always keeps at least one token per row.
    """
    if p is None or p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)

    # Shift right so the first token that crosses the threshold is kept
    sorted_mask = cumulative > p
    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
    sorted_mask[:, 0] = False

    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(1, sorted_idx, sorted_mask)
    return logits.masked_fill(mask, float("-inf"))


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    greedy: bool = False,
    generated_ids: Optional[torch.Tensor] = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """
    logits: (batch, vocab_size) — already sliced to the current step.
    Returns: (batch, 1) sampled token ids.
    """
    if greedy:
        return torch.argmax(logits, dim=-1, keepdim=True)

    if repetition_penalty != 1.0:
        if generated_ids is None:
            raise ValueError("generated_ids is required when repetition_penalty != 1.0")
        logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    logits = apply_temperature(logits, temperature)

    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
