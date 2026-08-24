#!/usr/bin/env python3
"""
scripts/tokenize_data.py

CLI wrapper for Phase 6: runs the trained tokenizer over the split
corpus (data/train/, data/validation/, data/test/) and writes
tokenized shards, via preprocessing/tokenize.py and
preprocessing/shard.py.

Usage:
    python scripts/tokenize_data.py --config configs/data.yaml
    python scripts/tokenize_data.py --config configs/data.yaml --shard-size 100000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tokenize the split corpus and write shards")
    parser.add_argument("--config", required=True, help="Path to configs/data.yaml")
    parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        help="Override tokens-per-shard from the config",
    )
    parser.add_argument(
        "--splits",
        default="train,validation,test",
        help="Comma-separated splits to tokenize (default: train,validation,test)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved plan without running it")
    return parser


def load_config(config_path: str) -> dict:
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_splits(splits_arg: str) -> list:
    valid = {"train", "validation", "test"}
    splits = [s.strip() for s in splits_arg.split(",") if s.strip()]
    unknown = [s for s in splits if s not in valid]
    if unknown:
        raise ValueError(f"Unknown split(s) {unknown}; valid splits are {sorted(valid)}")
    return splits


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    splits = resolve_splits(args.splits)
    config = load_config(args.config)

    if args.shard_size is not None:
        config["shard_size"] = args.shard_size

    if args.dry_run:
        print(f"[tokenize_data] config: {args.config}")
        print(f"[tokenize_data] splits: {splits}")
        print(f"[tokenize_data] shard_size: {config.get('shard_size')}")
        return

    # Lazy import: preprocessing/tokenize.py is expected to expose
    # tokenize_split(split_name, config) -> token id sequences, and
    # preprocessing/shard.py a write_shards(token_ids, split_name, config)
    # that writes fixed-size shard files. Adjust if yours differ.
    from preprocessing import tokenize as tokenize_mod, shard as shard_mod

    for split_name in splits:
        print(f"[tokenize_data] tokenizing split: {split_name}")
        token_ids = tokenize_mod.tokenize_split(split_name, config)
        shard_mod.write_shards(token_ids, split_name, config)
        print(f"[tokenize_data] wrote shards for: {split_name}")


if __name__ == "__main__":
    main()
