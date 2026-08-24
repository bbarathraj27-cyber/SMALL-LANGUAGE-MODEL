"""Phase 6: Tokenization.

Converts a cleaned, deduplicated, split JSONL corpus into a single
packed array of token ids, ready for sharding (preprocessing/shard.py)
and memory-mapped loading (dataset/pretraining_dataset.py).

This module *loads* an already-trained tokenizer (produced by
tokenizer/train_tokenizer.py as a tokenizer.json file) rather than
training one itself, keeping tokenizer training and corpus tokenization
as independently runnable/testable stages.

Packing strategy: documents are concatenated back-to-back with an
end-of-document token inserted between them (the standard approach
used by GPT-2/GPT-3 style pretraining, sometimes called "example
packing"). This avoids wasting compute on padding tokens during
pretraining, since fixed-length blocks are later cut across document
boundaries by the dataset loader rather than one block per document.
"""

from __future__ import annotations

import array
import os
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

try:
    from tokenizers import Tokenizer as _HFTokenizer
except ImportError as e:  # pragma: no cover - exercised only if dependency missing
    raise ImportError(
        "The 'tokenizers' package is required for preprocessing/tokenize.py. "
        "Install it with: pip install tokenizers"
    ) from e


def load_tokenizer(path: str) -> _HFTokenizer:
    """Loads a trained HuggingFace `tokenizers` tokenizer from a JSON file.

    Args:
        path: Path to a tokenizer.json file (as produced by
            tokenizer/train_tokenizer.py).

    Returns:
        A loaded `tokenizers.Tokenizer` instance.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Tokenizer file not found: {path}")
    return _HFTokenizer.from_file(path)


def _select_dtype(vocab_size: int) -> np.dtype:
    """Chooses the smallest unsigned integer dtype that can represent
    every token id in [0, vocab_size), to keep shard files compact.
    """
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if vocab_size <= 2**16:
        return np.dtype(np.uint16)
    if vocab_size <= 2**32:
        return np.dtype(np.uint32)
    raise ValueError(f"vocab_size {vocab_size} is too large for supported dtypes")


class CorpusTokenizer:
    """Tokenizes a JSONL corpus into a packed array of token ids.

    Args:
        tokenizer_path: Path to a trained tokenizer.json file.
        eos_token: The special token inserted between documents to mark
            document boundaries. Must already exist in the tokenizer's
            vocabulary (i.e. it must have been included as a special
            token when the tokenizer was trained).
    """

    def __init__(self, tokenizer_path: str, eos_token: str = "<|endoftext|>") -> None:
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.eos_token = eos_token

        eos_id = self.tokenizer.token_to_id(eos_token)
        if eos_id is None:
            raise ValueError(
                f"eos_token '{eos_token}' was not found in the tokenizer vocabulary. "
                "It must be added as a special token during tokenizer training."
            )
        self.eos_id = eos_id
        self.vocab_size = self.tokenizer.get_vocab_size()

        self.num_documents_processed = 0
        self.num_documents_skipped = 0

    def tokenize_records(
        self, records: Iterable[dict[str, Any]], text_field: str = "text"
    ) -> Iterator[list[int]]:
        """Tokenizes each record's text, yielding one token-id list per
        document with the EOS id appended at the end.

        Args:
            records: Iterable of dicts, each containing `text_field`.
            text_field: Key holding the text to tokenize.

        Yields:
            List of token ids for each non-empty document, terminated
            by the EOS token id.

        Raises:
            KeyError: If a record is missing `text_field`.
        """
        for record in records:
            if text_field not in record:
                raise KeyError(f"Record missing required field '{text_field}': {record}")
            text = record[text_field]
            if not text or not text.strip():
                self.num_documents_skipped += 1
                continue
            encoding = self.tokenizer.encode(text)
            ids = encoding.ids
            if not ids:
                self.num_documents_skipped += 1
                continue
            ids.append(self.eos_id)
            self.num_documents_processed += 1
            yield ids

    def tokenize_to_flat_array(
        self, records: Iterable[dict[str, Any]], text_field: str = "text"
    ) -> np.ndarray:
        """Tokenizes an entire corpus into one flat, packed token array.

        Args:
            records: Iterable of dicts, each containing `text_field`.
            text_field: Key holding the text to tokenize.

        Returns:
            1D numpy array of token ids (documents concatenated with
            EOS separators), using the smallest dtype that fits the
            tokenizer's vocabulary size.

        Raises:
            ValueError: If no documents produced any tokens (e.g. an
                empty corpus after filtering).
        """
        dtype = _select_dtype(self.vocab_size)
        buffer = array.array("I")  # unsigned int, widened to target dtype at the end

        for ids in self.tokenize_records(records, text_field=text_field):
            buffer.extend(ids)

        if len(buffer) == 0:
            raise ValueError(
                "No tokens were produced from the given records — corpus may be "
                "empty or every document may have been skipped."
            )

        return np.array(buffer, dtype=dtype)
