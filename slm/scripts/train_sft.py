#!/usr/bin/env python3
"""
scripts/train_sft.py

CLI wrapper around sft/train_sft.py (Phase 15): fine-tunes a
pretrained checkpoint on instruction data, writing checkpoints under
checkpoints/sft/.

Usage:
    python scripts/train_sft.py --checkpoint checkpoints/pretraining/final \
        --data data/instructions.jsonl --training-config configs/training.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SFT on a pretrained checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path (no extension) to the pretrained checkpoint")
    parser.add_argument("--data", required=True, help="Path to the {prompt, response} JSONL instruction data")
    parser.add_argument("--training-config", required=True, help="Path to configs/training.yaml")
    parser.add_argument(
        "--output-dir",
        default="checkpoints/sft",
        help="Directory to write SFT checkpoints (default: checkpoints/sft)",
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

    training_config = load_config(args.training_config)
    training_config["output_dir"] = args.output_dir

    if args.dry_run:
        print(f"[train_sft] checkpoint: {args.checkpoint}")
        print(f"[train_sft] data: {args.data}")
        print(f"[train_sft] output_dir: {args.output_dir}")
        return

    # Lazy import: sft/prepare_data.py builds the prompt-masked examples,
    # sft/train_sft.py is expected to expose
    # run_sft(checkpoint_path, prepared_data, training_config). Adjust if
    # your actual signatures differ.
    from sft import prepare_data as sft_prepare, train_sft as sft_train

    prepared = sft_prepare.prepare(args.data)
    sft_train.run_sft(args.checkpoint, prepared, training_config)


if __name__ == "__main__":
    main()
