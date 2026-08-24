#!/usr/bin/env python3
"""
scripts/train.py

CLI wrapper around training/train.py (Phases 10-12): runs pretraining
from scratch using configs/model.yaml + configs/training.yaml,
writing checkpoints under checkpoints/pretraining/.

Usage:
    python scripts/train.py --model-config configs/model.yaml --training-config configs/training.yaml
    python scripts/train.py --model-config configs/model.yaml --training-config configs/training.yaml --resume checkpoints/pretraining/latest
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain the SLM from scratch")
    parser.add_argument("--model-config", required=True, help="Path to configs/model.yaml")
    parser.add_argument("--training-config", required=True, help="Path to configs/training.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint path (without extension) to resume from")
    parser.add_argument(
        "--output-dir",
        default="checkpoints/pretraining",
        help="Directory to write checkpoints (default: checkpoints/pretraining)",
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

    model_config = load_config(args.model_config)
    training_config = load_config(args.training_config)
    training_config["output_dir"] = args.output_dir
    if args.resume:
        training_config["resume_from"] = args.resume

    if args.dry_run:
        print(f"[train] model_config: {args.model_config}")
        print(f"[train] training_config: {args.training_config}")
        print(f"[train] output_dir: {args.output_dir}")
        print(f"[train] resume_from: {args.resume}")
        return

    # Lazy import: training/train.py is expected to expose
    # train(model_config, training_config) as its top-level entrypoint,
    # internally wiring up training/trainer.py, loss.py, optimizer.py,
    # scheduler.py, and checkpoint.py. Adjust if yours differs.
    from training import train as train_mod

    train_mod.train(model_config, training_config)


if __name__ == "__main__":
    main()
