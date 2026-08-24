"""
training/optimizer.py

Builds the AdamW optimizer with the standard "no weight decay on norms/biases"
parameter grouping. Applying weight decay to RMSNorm weights or bias terms is
a common source of degraded training quality, so those parameters are
explicitly routed into a separate, zero-decay group.
"""

from typing import Iterable, Tuple

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
        raise ValueError(f"betas must each be in [0, 1), got {betas}")

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 1D parameters (biases, norm weights, embeddings-as-vectors) don't
        # benefit from weight decay and decaying them can hurt training.
        if param.dim() < 2 or "norm" in name.lower() or name.lower().endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    if not decay_params and not no_decay_params:
        raise ValueError("Model has no trainable parameters -- nothing to optimize")

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )
    return optimizer


def count_optimized_parameters(optimizer: torch.optim.Optimizer) -> int:
    """Total number of trainable scalar parameters registered with the optimizer."""
    total = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            total += p.numel()
    return total
