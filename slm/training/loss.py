"""
training/loss.py

Loss computation for causal language model pretraining and SFT.
Kept as a standalone module (rather than inlined in the model) so that
training/trainer.py, sft/train_sft.py and evaluation/perplexity.py can all
share the exact same loss definition -- avoiding subtle mismatches between
how "loss" is computed during training vs. evaluation.
"""

from typing import Optional

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def compute_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Cross-entropy loss for next-token prediction.

    Args:
        logits: (batch, seq_len, vocab_size) raw model outputs
        labels: (batch, seq_len) target token ids, with IGNORE_INDEX (-100)
                 at positions that should not contribute to the loss
                 (e.g. masked prompt tokens in SFT, or padding)
        label_smoothing: standard label smoothing factor, applied only to
                 non-ignored positions

    Returns:
        Scalar loss tensor (mean over non-ignored positions).

    Raises:
        ValueError: if shapes are inconsistent or labels contain out-of-range
                    token ids (other than IGNORE_INDEX).
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be (batch, seq_len, vocab_size), got shape {tuple(logits.shape)}")
    if labels.dim() != 2:
        raise ValueError(f"labels must be (batch, seq_len), got shape {tuple(labels.shape)}")
    if logits.shape[0] != labels.shape[0] or logits.shape[1] != labels.shape[1]:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} and labels shape {tuple(labels.shape)} "
            "must match on batch and sequence dimensions"
        )

    vocab_size = logits.shape[-1]
    valid_mask = labels != IGNORE_INDEX
    if valid_mask.any():
        valid_labels = labels[valid_mask]
        if valid_labels.min().item() < 0 or valid_labels.max().item() >= vocab_size:
            raise ValueError(
                f"labels contain token id(s) outside [0, {vocab_size}) "
                f"(min={valid_labels.min().item()}, max={valid_labels.max().item()})"
            )

    flat_logits = logits.reshape(-1, vocab_size)
    flat_labels = labels.reshape(-1)

    if not valid_mask.any():
        # Every position in this batch is masked out (can happen with a
        # degenerate SFT example) -- return a zero loss that still carries
        # gradient-compatible dtype/device so training doesn't crash.
        return flat_logits.sum() * 0.0

    loss = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=IGNORE_INDEX,
        label_smoothing=label_smoothing,
        reduction="mean",
    )
    return loss


def compute_perplexity(loss: torch.Tensor) -> float:
    """Convert a mean cross-entropy loss into perplexity. Clamped to avoid
    overflow from a runaway/uninitialized loss value."""
    if not torch.isfinite(loss):
        raise ValueError(f"Cannot compute perplexity from non-finite loss: {loss.item()}")
    clamped = torch.clamp(loss, max=20.0)  # exp(20) ~ 4.85e8, plenty for a sanity ceiling
    return torch.exp(clamped).item()
