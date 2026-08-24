"""Root Mean Square Layer Normalization (RMSNorm).

RMSNorm normalizes activations by their root-mean-square value only,
skipping the mean-centering step that standard LayerNorm performs.
It is cheaper to compute and empirically matches or exceeds LayerNorm
in transformer language models (used in LLaMA, Mistral, Qwen, T5).
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Applies RMS normalization over the last dimension of the input.

    Args:
        hidden_size: Size of the last dimension to normalize over.
        eps: Small constant added for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes the input and scales by the learned weight.

        Args:
            x: Tensor of shape (..., hidden_size).

        Returns:
            Tensor of the same shape as x.
        """
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected last dim {self.hidden_size}, got {x.shape[-1]}"
            )

        input_dtype = x.dtype
        # Compute the norm in float32 for numerical stability, then cast back.
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normalized = x_fp32 * torch.rsqrt(variance + self.eps)
        x_normalized = x_normalized.to(input_dtype)
        return self.weight * x_normalized
