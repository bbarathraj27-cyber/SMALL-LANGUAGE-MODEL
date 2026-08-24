"""
sft/train_sft.py

CLI entrypoint for supervised fine-tuning. Loads a pretrained checkpoint,
fine-tunes on a prepared instruction dataset (see sft/prepare_data.py),
using the same Trainer as pretraining but with SFT-specific hyperparameters
from configs/training.yaml's 'sft' section (typically a lower learning
rate and shorter run than pretraining).

Usage:
    python sft/train_sft.py \
        --model-config configs/model.yaml \
        --training-config configs/training.yaml \
        --pretrained-checkpoint checkpoints/pretraining/checkpoint_step10000.pt \
        --train-data data/sft/prepared_train.jsonl \
        --val-data data/sft/prepared_val.jsonl
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Dict

import torch
import yaml
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.slm import SLM, SLMConfig
from training.trainer import Trainer, TrainerConfig
from training.optimizer import build_optimizer
from training.scheduler import build_lr_scheduler
from training.checkpoint import load_checkpoint

logger = logging.getLogger("sft.train_sft")

IGNORE_INDEX = -100


class PreparedInstructionDataset(Dataset):
    """Loads examples produced by sft/prepare_data.py (JSONL of
    {input_ids, labels})."""

    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Prepared SFT data not found: {path}")

        self.examples: List[Dict[str, List[int]]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "input_ids" not in record or "labels" not in record:
                    raise ValueError(f"Line {line_num} of {path} missing input_ids/labels")
                if len(record["input_ids"]) != len(record["labels"]):
                    raise ValueError(
                        f"Line {line_num} of {path}: input_ids and labels length mismatch"
                    )
                self.examples.append(record)

        if not self.examples:
            raise ValueError(f"No examples found in {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "labels": torch.tensor(ex["labels"], dtype=torch.long),
        }


def sft_collate_fn(batch: List[Dict[str, torch.Tensor]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """Dynamic padding: pads input_ids with pad_id and labels with
    IGNORE_INDEX so padded positions never contribute to the loss."""
    if not batch:
        raise ValueError("sft_collate_fn received an empty batch")

    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids_padded = []
    labels_padded = []
    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_amount = max_len - seq_len

        input_ids_padded.append(
            torch.cat([item["input_ids"], torch.full((pad_amount,), pad_id, dtype=torch.long)])
        )
        labels_padded.append(
            torch.cat([item["labels"], torch.full((pad_amount,), IGNORE_INDEX, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids_padded),
        "labels": torch.stack(labels_padded),
    }


def build_sft_dataloader(data_path: str, batch_size: int, pad_id: int, shuffle: bool) -> DataLoader:
    dataset = PreparedInstructionDataset(data_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: sft_collate_fn(batch, pad_id=pad_id),
    )


def load_sft_training_config(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Training config not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None or "sft" not in data:
        raise ValueError(f"Training config {path} must contain a top-level 'sft' section")
    return data["sft"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run supervised fine-tuning (SFT).")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", default=None)
    args = parser.parse_args()

    try:
        model_config = SLMConfig.from_yaml(args.model_config)
        sft_cfg = load_sft_training_config(args.training_config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Config error: {e}")
        sys.exit(1)

    required_keys = {
        "learning_rate", "max_steps", "warmup_steps", "batch_size",
        "gradient_accumulation_steps", "weight_decay", "max_grad_norm",
    }
    missing = required_keys - set(sft_cfg.keys())
    if missing:
        logger.error(f"training.yaml 'sft' section is missing keys: {missing}")
        sys.exit(1)

    model = SLM(model_config)
    try:
        load_checkpoint(args.pretrained_checkpoint, model)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Failed to load pretrained checkpoint: {e}")
        sys.exit(1)
    logger.info(f"Loaded pretrained weights from {args.pretrained_checkpoint}")

    optimizer = build_optimizer(
        model, learning_rate=sft_cfg["learning_rate"], weight_decay=sft_cfg["weight_decay"]
    )
    scheduler = build_lr_scheduler(
        optimizer,
        warmup_steps=sft_cfg["warmup_steps"],
        total_steps=sft_cfg["max_steps"],
        min_lr_ratio=sft_cfg.get("min_lr_ratio", 0.1),
        decay_type=sft_cfg.get("decay_type", "cosine"),
    )

    try:
        train_dataloader = build_sft_dataloader(
            args.train_data, batch_size=sft_cfg["batch_size"], pad_id=0, shuffle=True
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load SFT training data: {e}")
        sys.exit(1)

    eval_dataloader = None
    if args.val_data:
        eval_dataloader = build_sft_dataloader(
            args.val_data, batch_size=sft_cfg["batch_size"], pad_id=0, shuffle=False
        )

    trainer_config = TrainerConfig(
        max_steps=sft_cfg["max_steps"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        max_grad_norm=sft_cfg["max_grad_norm"],
        log_every=sft_cfg.get("log_every", 10),
        eval_every=sft_cfg.get("eval_every", 50),
        checkpoint_every=sft_cfg.get("checkpoint_every", 200),
        checkpoint_dir=sft_cfg.get("checkpoint_dir", "checkpoints/sft"),
        use_amp=sft_cfg.get("use_amp", True),
        label_smoothing=sft_cfg.get("label_smoothing", 0.0),
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=trainer_config,
    )

    results = trainer.train()
    logger.info(
        f"SFT complete: {results['final_step']} steps, {results['total_time_seconds']:.1f}s total"
    )


if __name__ == "__main__":
    main()
