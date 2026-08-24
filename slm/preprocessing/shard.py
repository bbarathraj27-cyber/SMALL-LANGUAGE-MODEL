"""Sharding.

Splits one large packed token array into multiple fixed-size binary
shard files on disk, plus a manifest.json describing them. Sharding
keeps individual files small enough to handle comfortably (copy,
checksum, re-download) and lets the dataset loader memory-map shards
individually rather than needing the entire tokenized corpus resident
in RAM at once.
"""

from __future__ import annotations

import json
import os

import numpy as np


def write_shards(
    token_array: np.ndarray,
    output_dir: str,
    shard_size: int = 1_000_000,
    prefix: str = "shard",
) -> dict:
    """Writes a token array to disk as fixed-size binary shards.

    Args:
        token_array: 1D numpy array of token ids (typically produced by
            preprocessing.tokenize.CorpusTokenizer.tokenize_to_flat_array).
        output_dir: Directory to write shard files and manifest.json into.
        shard_size: Number of tokens per shard file. The final shard may
            contain fewer tokens if the total doesn't divide evenly.
        prefix: Filename prefix for shard files, e.g. "shard_00000.bin".

    Returns:
        The manifest dict that was written to output_dir/manifest.json.

    Raises:
        ValueError: If token_array is not 1D, is empty, or shard_size
            is not positive.
    """
    if token_array.ndim != 1:
        raise ValueError(f"token_array must be 1D, got shape {token_array.shape}")
    if token_array.size == 0:
        raise ValueError("token_array is empty; nothing to shard.")
    if shard_size <= 0:
        raise ValueError(f"shard_size must be positive, got {shard_size}")

    os.makedirs(output_dir, exist_ok=True)

    total_tokens = int(token_array.size)
    dtype_name = str(token_array.dtype)
    num_shards = (total_tokens + shard_size - 1) // shard_size

    shard_entries = []
    for shard_index in range(num_shards):
        start = shard_index * shard_size
        end = min(start + shard_size, total_tokens)
        shard_data = token_array[start:end]

        filename = f"{prefix}_{shard_index:05d}.bin"
        file_path = os.path.join(output_dir, filename)
        shard_data.tofile(file_path)

        shard_entries.append(
            {
                "file": filename,
                "num_tokens": int(shard_data.size),
                "start_offset": int(start),
            }
        )

    manifest = {
        "dtype": dtype_name,
        "total_tokens": total_tokens,
        "shard_size": shard_size,
        "num_shards": num_shards,
        "shards": shard_entries,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def load_manifest(path: str) -> dict:
    """Loads and validates a shard manifest.json.

    Args:
        path: Path to a manifest.json file, or to the directory
            containing it (in which case "manifest.json" is appended).

    Returns:
        The parsed manifest dict.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        ValueError: If the manifest is missing required keys or a
            referenced shard file does not exist alongside it.
    """
    if os.path.isdir(path):
        manifest_path = os.path.join(path, "manifest.json")
    else:
        manifest_path = path

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    required_keys = {"dtype", "total_tokens", "shard_size", "num_shards", "shards"}
    missing = required_keys - set(manifest.keys())
    if missing:
        raise ValueError(f"Manifest at {manifest_path} is missing keys: {missing}")

    manifest_dir = os.path.dirname(manifest_path)
    for shard_entry in manifest["shards"]:
        shard_path = os.path.join(manifest_dir, shard_entry["file"])
        if not os.path.isfile(shard_path):
            raise ValueError(
                f"Manifest references shard file that does not exist: {shard_path}"
            )

    return manifest
