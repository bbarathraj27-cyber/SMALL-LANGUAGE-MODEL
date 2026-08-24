"""Pretraining dataset.

Wraps a sharded, packed token corpus (produced by
preprocessing/tokenize.py + preprocessing/shard.py) as a PyTorch
Dataset that yields fixed-length blocks for causal language model
training. Shards are memory-mapped rather than loaded into RAM, so
corpora far larger than available memory can still be trained on.

Label convention: this dataset returns `labels` equal to the same
token sequence as `input_ids` (not pre-shifted). model.slm.SLM.forward
performs the next-token shift internally (predicting position i+1 from
position i), which is the standard HuggingFace-style convention and
keeps the shifting logic in exactly one place.
"""

from __future__ import annotations

import bisect
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.shard import load_manifest


_DTYPE_MAP = {
    "uint16": np.uint16,
    "uint32": np.uint32,
}


class PretrainingDataset(Dataset):
    """Fixed-length block dataset over a memory-mapped, sharded token corpus.

    Args:
        manifest_path: Path to a manifest.json (or its containing
            directory) as produced by preprocessing.shard.write_shards.
        block_size: Number of tokens per training example.
        stride: Step size (in tokens) between consecutive examples.
            Defaults to `block_size` (non-overlapping blocks), which is
            standard for pretraining on already-packed data. A smaller
            stride creates overlapping examples, trading more disk-read
            work per epoch for more training examples per epoch.

    Raises:
        FileNotFoundError: If the manifest or any referenced shard file
            is missing.
        ValueError: If block_size is not positive, or if the corpus has
            fewer than block_size tokens.
    """

    def __init__(self, manifest_path: str, block_size: int, stride: int | None = None) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.manifest = load_manifest(manifest_path)
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")

        dtype_name = self.manifest["dtype"]
        if dtype_name not in _DTYPE_MAP:
            raise ValueError(f"Unsupported manifest dtype: {dtype_name}")
        self.dtype = _DTYPE_MAP[dtype_name]

        manifest_dir = (
            manifest_path if os.path.isdir(manifest_path) else os.path.dirname(manifest_path)
        )

        self._shard_memmaps: list[np.memmap] = []
        self._shard_start_offsets: list[int] = []
        for shard_entry in self.manifest["shards"]:
            shard_path = os.path.join(manifest_dir, shard_entry["file"])
            mm = np.memmap(shard_path, dtype=self.dtype, mode="r")
            self._shard_memmaps.append(mm)
            self._shard_start_offsets.append(shard_entry["start_offset"])

        self.total_tokens = self.manifest["total_tokens"]
        if self.total_tokens < block_size:
            raise ValueError(
                f"Corpus has only {self.total_tokens} tokens, fewer than "
                f"block_size ({block_size}); cannot form a single example."
            )

        self._num_examples = (self.total_tokens - block_size) // self.stride + 1

    def __len__(self) -> int:
        return self._num_examples

    def _read_token_slice(self, global_start: int, length: int) -> np.ndarray:
        """Reads `length` tokens starting at `global_start`, transparently
        handling the case where the requested range spans a shard boundary.
        """
        global_end = global_start + length
        if global_end > self.total_tokens:
            raise IndexError(
                f"Requested range [{global_start}, {global_end}) exceeds "
                f"corpus length {self.total_tokens}"
            )

        # Find the shard containing global_start: the last shard whose
        # start_offset is <= global_start.
        shard_idx = bisect.bisect_right(self._shard_start_offsets, global_start) - 1
        shard_idx = max(shard_idx, 0)

        pieces = []
        remaining = length
        cursor = global_start
        while remaining > 0:
            shard_start = self._shard_start_offsets[shard_idx]
            shard_mm = self._shard_memmaps[shard_idx]
            local_start = cursor - shard_start
            available_in_shard = len(shard_mm) - local_start
            take = min(remaining, available_in_shard)
            if take <= 0:
                raise IndexError(
                    f"Could not read requested slice [{global_start}, {global_end}); "
                    f"ran out of shard data at shard {shard_idx}"
                )
            pieces.append(np.asarray(shard_mm[local_start : local_start + take]))
            cursor += take
            remaining -= take
            shard_idx += 1

        if len(pieces) == 1:
            return pieces[0]
        return np.concatenate(pieces)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Returns one fixed-length training example.

        Args:
            index: Example index in [0, len(self)).

        Returns:
            Dict with:
                "input_ids": LongTensor of shape (block_size,)
                "labels": LongTensor of shape (block_size,), identical
                    to input_ids (see module docstring on shift
                    convention).

        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= self._num_examples:
            raise IndexError(
                f"Index {index} out of range for dataset of length {self._num_examples}"
            )

        global_start = index * self.stride
        tokens = self._read_token_slice(global_start, self.block_size)
        input_ids = torch.from_numpy(tokens.astype(np.int64))
        labels = input_ids.clone()

        return {"input_ids": input_ids, "labels": labels}
