"""
evaluation/perplexity.py

Computes corpus-level perplexity for a base (pretrained) model over a
pretraining-style token dataset (fixed-length blocks, no prompt masking --
every position contributes to the loss). This is distinct from
sft/evaluate_sft.py's response-only perplexity, which only scores the
response span of instruction-formatted examples.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.loss import compute_lm_loss, compute_perplexity


@torch.no_grad()
def compute_dataset_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Runs the model over `dataloader` in eval mode and returns aggregate loss
    and perplexity. Loss is token-count-weighted across batches so that a
    final partial batch (fewer tokens) doesn't skew the average.
    """
    if max_batches is not None and max_batches <= 0:
        raise ValueError(f"max_batches must be positive when provided, got {max_batches}")

    model.eval()
    model.to(device)

    total_loss_sum = 0.0
    total_token_count = 0
    batches_seen = 0

    for batch in dataloader:
        if max_batches is not None and batches_seen >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # model/slm.py's SLM.forward returns a dict with keys
        # "logits", "loss", "past_key_values" -- not a tuple.
        output = model(input_ids, labels=labels)
        logits = output["logits"]
        model_loss = output["loss"]
        loss = model_loss if model_loss is not None else compute_lm_loss(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss encountered during perplexity evaluation: {loss.item()}")

        # Count non-ignored label positions as the token weight for this batch.
        valid_tokens = int((labels != -100).sum().item())
        if valid_tokens == 0:
            continue

        total_loss_sum += loss.item() * valid_tokens
        total_token_count += valid_tokens
        batches_seen += 1

    if total_token_count == 0:
        raise ValueError("Dataloader produced zero scoreable tokens -- cannot compute perplexity")

    mean_loss = total_loss_sum / total_token_count
    perplexity = compute_perplexity(torch.tensor(mean_loss))

    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "num_batches": batches_seen,
        "num_tokens": total_token_count,
    }


def bits_per_byte(perplexity: float, avg_tokens_per_byte: float) -> float:
    """
    Converts perplexity (a per-token measure) into bits-per-byte, a
    tokenizer-independent metric useful for comparing models that use
    different vocabularies/tokenizers.

    avg_tokens_per_byte: measured ratio of (token count / raw UTF-8 byte
    count) for the evaluation corpus -- must be computed by the caller from
    the actual text, since it depends on the tokenizer.
    """
    if perplexity <= 0:
        raise ValueError(f"perplexity must be positive, got {perplexity}")
    if avg_tokens_per_byte <= 0:
        raise ValueError(f"avg_tokens_per_byte must be positive, got {avg_tokens_per_byte}")

    bits_per_token = math.log2(perplexity)
    return bits_per_token * avg_tokens_per_byte
