"""
tokenizer.py

Thin, dependency-isolating wrapper around a trained HuggingFace `tokenizers`
BPE tokenizer. Every other module (preprocessing/tokenize.py, dataset/*,
training/, inference/) should import SLMTokenizer from here rather than
touching the `tokenizers` library directly -- that keeps the tokenizer
backend swappable in one place.
"""

import os
from typing import List, Optional, Union

from tokenizers import Tokenizer

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


class SLMTokenizer:
    """Loads a trained tokenizer.json and exposes encode/decode helpers."""

    def __init__(self, tokenizer_path: str):
        if not os.path.isfile(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer file not found: {tokenizer_path}. "
                "Run tokenizer/train_tokenizer.py first to produce tokenizer.json."
            )

        try:
            self._tokenizer = Tokenizer.from_file(tokenizer_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from {tokenizer_path}: {e}") from e

        self._validate_special_tokens()

    def _validate_special_tokens(self) -> None:
        expected = {
            PAD_TOKEN: PAD_ID,
            BOS_TOKEN: BOS_ID,
            EOS_TOKEN: EOS_ID,
            UNK_TOKEN: UNK_ID,
        }
        for token, expected_id in expected.items():
            actual_id = self._tokenizer.token_to_id(token)
            if actual_id is None:
                raise ValueError(
                    f"Loaded tokenizer is missing required special token '{token}'."
                )
            if actual_id != expected_id:
                raise ValueError(
                    f"Special token '{token}' has id {actual_id}, expected {expected_id}. "
                    "Downstream modules hardcode these ids -- retrain the tokenizer "
                    "with tokenizer/train_tokenizer.py to fix this."
                )

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        if not isinstance(text, str):
            raise TypeError(f"encode() expects a str, got {type(text)}")

        ids = self._tokenizer.encode(text).ids

        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[List[int]]:
        if not isinstance(texts, list):
            raise TypeError(f"encode_batch() expects a list of str, got {type(texts)}")

        encodings = self._tokenizer.encode_batch(texts)
        results = []
        for enc in encodings:
            ids = enc.ids
            if add_bos:
                ids = [BOS_ID] + ids
            if add_eos:
                ids = ids + [EOS_ID]
            results.append(ids)
        return results

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        if not isinstance(ids, (list, tuple)):
            raise TypeError(f"decode() expects a list/tuple of ints, got {type(ids)}")

        clean_ids = [int(i) for i in ids]
        return self._tokenizer.decode(clean_ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(
        self, batch_ids: List[List[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        if not isinstance(batch_ids, list):
            raise TypeError(f"decode_batch() expects a list of lists, got {type(batch_ids)}")

        return self._tokenizer.decode_batch(
            [[int(i) for i in ids] for ids in batch_ids],
            skip_special_tokens=skip_special_tokens,
        )

    def token_to_id(self, token: str) -> Optional[int]:
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        return self._tokenizer.id_to_token(token_id)

    @classmethod
    def from_pretrained(cls, tokenizer_dir_or_file: str) -> "SLMTokenizer":
        """Convenience loader: accepts either a directory containing
        tokenizer.json or a direct path to the file."""
        if os.path.isdir(tokenizer_dir_or_file):
            path = os.path.join(tokenizer_dir_or_file, "tokenizer.json")
        else:
            path = tokenizer_dir_or_file
        return cls(path)
