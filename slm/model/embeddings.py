"""Token embedding layer for the SLM.

Maps discrete token ids to dense vectors of size `hidden_size`. The
embedding weight matrix is also reused as the language-modeling head
when weight tying is enabled (see model/lm_head.py and model/slm.py).
"""

import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Learned token embedding table.

    Args:
        vocab_size: Number of tokens in the tokenizer vocabulary.
        hidden_size: Dimensionality of the model's hidden states.
        initializer_range: Standard deviation used for the normal
            initialization of the embedding weights.
        pad_token_id: If provided, the embedding row for this id is
            initialized to zeros and its gradient is not scaled by
            padding (kept trainable, matches common LLaMA/GPT practice
            of not freezing the pad row, but zero-initialized so it
            starts as a neutral vector).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        initializer_range: float = 0.02,
        pad_token_id: int | None = None,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id

        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        self._init_weights(initializer_range)

    def _init_weights(self, initializer_range: float) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=initializer_range)
        if self.pad_token_id is not None:
            with torch.no_grad():
                self.weight[self.pad_token_id].fill_(0.0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for a batch of token ids.

        Args:
            input_ids: LongTensor of shape (batch_size, seq_len) containing
                token ids in the range [0, vocab_size).

        Returns:
            FloatTensor of shape (batch_size, seq_len, hidden_size).
        """
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                f"input_ids must be an integer tensor, got dtype={input_ids.dtype}"
            )
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.vocab_size):
            raise ValueError(
                "input_ids contains values outside the valid range "
                f"[0, {self.vocab_size})"
            )
        return nn.functional.embedding(input_ids, self.weight, padding_idx=self.pad_token_id)

    def embedding_scale(self) -> float:
        """Standard sqrt(d_model) scale factor some architectures apply
        to embeddings before adding positional information. Exposed as a
        helper so callers can opt in/out explicitly rather than hiding
        the multiplication inside forward().
        """
        return math.sqrt(self.hidden_size)
