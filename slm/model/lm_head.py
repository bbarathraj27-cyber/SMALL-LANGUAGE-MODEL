"""Language modeling head.

Projects final hidden states to vocabulary-sized logits. Supports
weight tying with the token embedding matrix, a standard technique
(used in GPT-2, LLaMA-family small variants) that saves
vocab_size * hidden_size parameters by reusing the embedding table
for the output projection instead of learning a separate matrix.
"""

import torch
import torch.nn as nn


class LMHead(nn.Module):
    """Final linear projection from hidden states to vocabulary logits.

    Args:
        hidden_size: Model hidden dimension.
        vocab_size: Tokenizer vocabulary size.
        bias: Whether to include a bias term in the output projection.
        tied_weight: If provided, this parameter (typically the token
            embedding weight) is reused directly instead of allocating
            a new weight matrix, i.e. weight tying is enabled. Must
            have shape (vocab_size, hidden_size).
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
        tied_weight: nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tied = tied_weight is not None

        if self.tied:
            expected_shape = (vocab_size, hidden_size)
            if tuple(tied_weight.shape) != expected_shape:
                raise ValueError(
                    f"tied_weight shape {tuple(tied_weight.shape)} does not match "
                    f"expected {expected_shape}"
                )
            self.weight = tied_weight
        else:
            self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
            nn.init.normal_(self.weight, mean=0.0, std=0.02)

        self.bias = nn.Parameter(torch.zeros(vocab_size)) if bias else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Projects hidden states to vocabulary logits.

        Args:
            hidden_states: Tensor of shape (batch, seq_len, hidden_size).

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size).
        """
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected last dim {self.hidden_size}, got {hidden_states.shape[-1]}"
            )
        logits = torch.nn.functional.linear(hidden_states, self.weight, self.bias)
        return logits
