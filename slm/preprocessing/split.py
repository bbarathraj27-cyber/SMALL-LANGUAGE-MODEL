"""Data splitting.

Splits a deduplicated corpus into train/validation/test sets with a
deterministic, seedable shuffle so results are reproducible across runs.
"""

from __future__ import annotations

import os
import random
from typing import Any

from preprocessing.collect import write_jsonl


def split_records(
    records: list[dict[str, Any]],
    train_ratio: float = 0.98,
    val_ratio: float = 0.01,
    test_ratio: float = 0.01,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Splits records into train/validation/test sets.

    Args:
        records: Full list of records to split.
        train_ratio: Fraction of records assigned to the training set.
        val_ratio: Fraction assigned to the validation set.
        test_ratio: Fraction assigned to the test set.
        seed: Random seed controlling the shuffle order, for
            reproducibility.

    Returns:
        Dict with keys "train", "validation", "test", each mapping to
        a list of records.

    Raises:
        ValueError: If records is empty, if any ratio is negative, or
            if the ratios do not sum to 1.0 (within floating-point
            tolerance).
    """
    if not records:
        raise ValueError("Cannot split an empty records list.")
    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Split ratios must be non-negative.")

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total_ratio} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )

    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = round(n * train_ratio)
    val_end = train_end + round(n * val_ratio)

    train_split = shuffled[:train_end]
    val_split = shuffled[train_end:val_end]
    test_split = shuffled[val_end:]

    return {"train": train_split, "validation": val_split, "test": test_split}


def write_splits(splits: dict[str, list[dict[str, Any]]], output_dir: str) -> dict[str, str]:
    """Writes each split to its own JSONL file under output_dir.

    Args:
        splits: Dict as returned by split_records (or any dict mapping
            split name -> list of records). Empty splits are skipped
            with no file written for them.
        output_dir: Directory to write "<split_name>.jsonl" files into.

    Returns:
        Dict mapping split name to the output file path actually written.

    Raises:
        ValueError: If splits is empty or every split is empty.
    """
    if not splits:
        raise ValueError("splits dict is empty; nothing to write.")

    written_paths: dict[str, str] = {}
    for split_name, split_records_list in splits.items():
        if not split_records_list:
            continue
        output_path = os.path.join(output_dir, f"{split_name}.jsonl")
        write_jsonl(split_records_list, output_path)
        written_paths[split_name] = output_path

    if not written_paths:
        raise ValueError("All splits were empty; no files were written.")

    return written_paths
