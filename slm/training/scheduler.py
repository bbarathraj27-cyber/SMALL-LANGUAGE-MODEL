"""
training/scheduler.py

Learning-rate scheduling: linear warmup followed by cosine (or linear) decay
down to a configurable minimum learning-rate floor. Implemented as a plain
function returning a LambdaLR multiplier rather than a custom class, so it's
trivial to unit-test the multiplier at any step without running real
optimizer steps.
"""

import math
from typing import Literal

import torch


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
    decay_type: Literal["cosine", "linear"] = "cosine",
) -> torch.optim.lr_scheduler.LambdaLR:
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if warmup_steps > total_steps:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) cannot exceed total_steps ({total_steps})"
        )
    if not (0.0 <= min_lr_ratio <= 1.0):
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")
    if decay_type not in ("cosine", "linear"):
        raise ValueError(f"decay_type must be 'cosine' or 'linear', got {decay_type!r}")

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps

        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / decay_steps)

        if decay_type == "cosine":
            decay_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:  # linear
            decay_factor = 1.0 - progress

        # Interpolate between 1.0 (peak lr) and min_lr_ratio (floor).
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
