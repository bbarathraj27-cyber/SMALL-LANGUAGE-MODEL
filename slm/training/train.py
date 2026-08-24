"""
training/train.py

CLI entrypoint that wires configs/model.yaml + configs/data.yaml +
configs/training.yaml together with model/slm.py and dataset/dataloader.py
to run pretraining end-to-end.

Usage:
    python training/train.py \
        --model-config configs/model.yaml \
        --training-config configs/training.yaml \
        --train-shards data/train \
        --tokenizer tokenizer/tokenizer.json \
        --resume
"""

import argparse
import logging
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.slm import SLM, SLMConfig
from dataset.dataloader import build_pretrain_dataloader
from training.trainer import Trainer, TrainerConfig
from training.optimizer import build_optimizer
from training.scheduler import build_lr_scheduler

logger = logging.getLogger("training.train")


def load_training_yaml(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Training config not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None or "pretraining" not in data:
        raise ValueError(
            f"Training config {path} must contain a top-level 'pretraining' section"
        )
    return data["pretraining"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the SLM.")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--train-shards", default="data/train")
    parser.add_argument("--val-shards", default="data/validation")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--resume-from", default=None, help="Resume from a specific checkpoint path")
    args = parser.parse_args()

    try:
        model_config = SLMConfig.from_yaml(args.model_config)
        train_cfg = load_training_yaml(args.training_config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Config error: {e}")
        sys.exit(1)

    required_keys = {
        "learning_rate", "max_steps", "warmup_steps", "batch_size",
        "gradient_accumulation_steps", "weight_decay", "max_grad_norm",
    }
    missing = required_keys - set(train_cfg.keys())
    if missing:
        logger.error(f"training.yaml 'pretraining' section is missing keys: {missing}")
        sys.exit(1)

    model = SLM(model_config)
    logger.info(f"Model initialized with {model.num_parameters():,} parameters")

    optimizer = build_optimizer(
        model,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = build_lr_scheduler(
        optimizer,
        warmup_steps=train_cfg["warmup_steps"],
        total_steps=train_cfg["max_steps"],
        min_lr_ratio=train_cfg.get("min_lr_ratio", 0.1),
        decay_type=train_cfg.get("decay_type", "cosine"),
    )

    train_dataloader = build_pretrain_dataloader(
        shard_dir=args.train_shards,
        block_size=model_config.max_position_embeddings,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
    )
    eval_dataloader = None
    if os.path.isdir(args.val_shards) and os.listdir(args.val_shards):
        eval_dataloader = build_pretrain_dataloader(
            shard_dir=args.val_shards,
            block_size=model_config.max_position_embeddings,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
        )

    trainer_config = TrainerConfig(
        max_steps=train_cfg["max_steps"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        max_grad_norm=train_cfg["max_grad_norm"],
        log_every=train_cfg.get("log_every", 10),
        eval_every=train_cfg.get("eval_every", 100),
        checkpoint_every=train_cfg.get("checkpoint_every", 500),
        checkpoint_dir=train_cfg.get("checkpoint_dir", "checkpoints/pretraining"),
        use_amp=train_cfg.get("use_amp", True),
        label_smoothing=train_cfg.get("label_smoothing", 0.0),
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=trainer_config,
    )

    if args.resume or args.resume_from:
        found = trainer.resume_from_checkpoint(args.resume_from)
        if not found:
            logger.info("No checkpoint found to resume from -- starting fresh")

    results = trainer.train()
    logger.info(
        f"Training complete: {results['final_step']} steps, "
        f"{results['total_time_seconds']:.1f}s total"
    )


if __name__ == "__main__":
    main()
