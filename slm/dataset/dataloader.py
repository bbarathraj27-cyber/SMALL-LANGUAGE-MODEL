"""DataLoader construction.

Provides two builder functions:

- build_pretraining_dataloader: wraps PretrainingDataset, which already
  returns fixed-length blocks, so the default collate (stacking) works
  as-is.
- build_instruction_dataloader: wraps InstructionDataset, which returns
  variable-length examples, using a custom collate function that pads
  each batch dynamically to the longest example *in that batch* rather
  than a fixed global max_length — this avoids wasting compute on
  padding tokens for batches that happen to contain only short examples.
"""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from dataset.pretraining_dataset import PretrainingDataset
from dataset.instruction_dataset import InstructionDataset


def collate_instruction_batch(
    batch: list[dict[str, torch.Tensor]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    """Pads a batch of variable-length instruction examples to a common length.

    Args:
        batch: List of per-example dicts as returned by
            InstructionDataset.__getitem__, each containing "input_ids",
            "labels", and "attention_mask" 1D tensors of possibly
            different lengths.
        pad_token_id: Token id used to pad "input_ids". "labels" are
            padded with -100 (ignored in the loss) and
            "attention_mask" is padded with 0 (marks padding positions
            as not-to-be-attended-to).

    Returns:
        Dict with "input_ids", "labels", "attention_mask", each a
        LongTensor of shape (batch_size, max_len_in_batch).

    Raises:
        ValueError: If batch is empty.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    input_ids_list = [example["input_ids"] for example in batch]
    labels_list = [example["labels"] for example in batch]
    attention_mask_list = [example["attention_mask"] for example in batch]

    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def build_pretraining_dataloader(
    dataset: PretrainingDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    seed: int = 42,
) -> DataLoader:
    """Builds a DataLoader over a PretrainingDataset.

    Args:
        dataset: A PretrainingDataset instance.
        batch_size: Number of examples per batch.
        shuffle: Whether to shuffle example order each epoch. Shuffling
            shuffles *block* order, not token order within a block, so
            document structure within each block is preserved.
        num_workers: Number of subprocess workers for data loading.
        drop_last: Whether to drop a final incomplete batch, which
            keeps every batch the same size (simplifies fixed-shape
            training loops and avoids awkward gradient-scale changes
            on a smaller last batch).
        seed: Seed for the shuffle generator, for reproducibility.

    Returns:
        A configured torch.utils.data.DataLoader.

    Raises:
        ValueError: If batch_size is not positive.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        generator=generator if shuffle else None,
        pin_memory=torch.cuda.is_available(),
    )


def build_pretrain_dataloader(
    shard_dir: str,
    block_size: int,
    batch_size: int,
    shuffle: bool = True,
    stride: int | None = None,
    num_workers: int = 0,
    drop_last: bool = True,
    seed: int = 42,
) -> DataLoader:
    """Convenience wrapper: builds a pretraining DataLoader directly from
    a shard directory, without the caller needing to construct a
    PretrainingDataset by hand first.

    This is the entry point production code (training/train.py,
    evaluation/evaluate.py) should use, since they only have a shard
    directory path on hand, not an already-built PretrainingDataset.

    Args:
        shard_dir: Directory containing manifest.json and shard .bin
            files, as produced by preprocessing.shard.write_shards.
        block_size: Number of tokens per training example. Should
            match the model's max_position_embeddings.
        batch_size: Number of examples per batch.
        shuffle: Whether to shuffle example order each epoch.
        stride: Step size between consecutive examples. Defaults to
            block_size (non-overlapping blocks).
        num_workers: Number of subprocess workers for data loading.
        drop_last: Whether to drop a final incomplete batch.
        seed: Seed for the shuffle generator, for reproducibility.

    Returns:
        A configured torch.utils.data.DataLoader.

    Raises:
        FileNotFoundError: If shard_dir or its manifest.json is missing.
        ValueError: If batch_size or block_size is not positive, or if
            the corpus has fewer tokens than block_size.
    """
    dataset = PretrainingDataset(shard_dir, block_size=block_size, stride=stride)
    return build_pretraining_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        seed=seed,
    )


def build_instruction_dataloader(
    dataset: InstructionDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """Builds a DataLoader over an InstructionDataset with dynamic padding.

    Args:
        dataset: An InstructionDataset instance.
        batch_size: Number of examples per batch.
        shuffle: Whether to shuffle example order each epoch.
        num_workers: Number of subprocess workers for data loading.
        drop_last: Whether to drop a final incomplete batch. Defaults
            to False for SFT, since instruction datasets are typically
            much smaller than pretraining corpora and dropping data is
            more costly relative to dataset size.

    Returns:
        A configured torch.utils.data.DataLoader with dynamic-padding
        collation.

    Raises:
        ValueError: If batch_size is not positive.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return collate_instruction_batch(batch, pad_token_id=dataset.pad_token_id)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=_collate,
        pin_memory=torch.cuda.is_available(),
    )
