#!/usr/bin/env python3
"""
scripts/train_tokenizer.py

CLI wrapper around tokenizer/train_tokenizer.py (Phase 5). Trains a BPE
tokenizer over the cleaned/deduplicated corpus and writes
tokenizer.json, vocab.json, and merges.txt into tokenizer/.

Usage:
    python scripts/train_tokenizer.py --config configs/data.yaml
    python scripts/train_tokenizer.py --config configs/data.yaml --vocab-size 32000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the BPE tokenizer")
    parser.add_argument("--config", required=True, help="Path to configs/data.yaml")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Override the vocab size from the config (e.g. 32000)",
    )
    parser.add_argument(
        "--output-dir",
        default="tokenizer",
        help="Directory to write tokenizer.json / vocab.json / merges.txt (default: tokenizer/)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved settings without training")
    return parser


def load_config(config_path: str) -> dict:
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.vocab_size is not None:
        config["vocab_size"] = args.vocab_size

    if args.dry_run:
        print(f"[train_tokenizer] config: {args.config}")
        print(f"[train_tokenizer] vocab_size: {config.get('vocab_size')}")
        print(f"[train_tokenizer] output_dir: {args.output_dir}")
        return

    # Lazy import: tokenizer/train_tokenizer.py is expected to expose a
    # train(config, output_dir) function that writes the three artifact
    # files. Adjust this call if your actual signature differs.
    from tokenizer import train_tokenizer

    train_tokenizer.train(config, output_dir=args.output_dir)
    print(f"[train_tokenizer] wrote tokenizer.json / vocab.json / merges.txt to {args.output_dir}")


if __name__ == "__main__":
    main()
