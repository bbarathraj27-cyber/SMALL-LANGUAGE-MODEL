"""SwiGLU feed-forward network.

SwiGLU replaces the standard two-layer ReLU/GELU feed-forward block
with a gated linear unit using the SiLU (swish) activation. It
consistently outperforms plain FFNs at matched parameter counts and
is used in LLaMA, PaLM, and Mistral.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_swiglu_intermediate_size(hidden_size: int, multiple_of: int = 64) -> int:
    """Computes the SwiGLU intermediate (hidden) dimension.

    SwiGLU has three weight matrices instead of a standard FFN's two,
    so to keep the parameter count comparable to a 4x-hidden_size
    standard FFN, the intermediate size is scaled down by 2/3 and then
    rounded up to the nearest multiple of `multiple_of` for hardware
    efficiency (matches the LLaMA reference implementation).

    Args:
        hidden_size: The model's hidden dimension.
        multiple_of: Round the result up to the nearest multiple of
            this value.

    Returns:
        The intermediate size to use for the gate/up projections.
    """
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    if multiple_of <= 0:
        raise ValueError(f"multiple_of must be positive, got {multiple_of}")

    raw_intermediate = int(2 * (4 * hidden_size) / 3)
    rounded = multiple_of * ((raw_intermediate + multiple_of - 1) // multiple_of)
    return rounded


class SwiGLUFFN(nn.Module):
    """SwiGLU-gated feed-forward network.

    Computes: down_proj(SiLU(gate_proj(x)) * up_proj(x))

    Args:
        hidden_size: Input/output dimensionality.
        intermediate_size: Width of the gate/up projections. If None,
            it is computed automatically via
            `compute_swiglu_intermediate_size`.
        bias: Whether the linear layers include a bias term. Modern
            small LLMs typically disable bias for slightly better
            training stability and fewer parameters.
        dropout: Dropout probability applied to the output.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int | None = None,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if intermediate_size is not None and intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be positive, got {intermediate_size}"
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.hidden_size = hidden_size
        self.intermediate_size = (
            intermediate_size
            if intermediate_size is not None
            else compute_swiglu_intermediate_size(hidden_size)
        )

        self.gate_proj = nn.Linear(hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the SwiGLU transformation.

        Args:
            x: Tensor of shape (..., hidden_size).

        Returns:
            Tensor of the same shape as x.
        """
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected last dim {self.hidden_size}, got {x.shape[-1]}"
            )

        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        fused = gate * up
        out = self.down_proj(fused)
        return self.dropout(out)
