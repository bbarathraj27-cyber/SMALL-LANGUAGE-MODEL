"""
training/trainer.py

The Trainer class ties together model, dataloader, optimizer, scheduler,
loss, and checkpointing into a single training loop. Supports gradient
accumulation, gradient clipping, mixed precision (when running on CUDA),
periodic checkpointing, and periodic evaluation.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.loss import compute_lm_loss, compute_perplexity
from training.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint

logger = logging.getLogger("training.trainer")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class TrainerConfig:
    max_steps: int
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    log_every: int = 10
    eval_every: int = 100
    checkpoint_every: int = 500
    checkpoint_dir: str = "checkpoints/pretraining"
    keep_last_n_checkpoints: int = 3
    use_amp: bool = True
    label_smoothing: float = 0.0

    def __post_init__(self):
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}"
            )
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {self.max_grad_norm}")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        train_dataloader: DataLoader,
        config: TrainerConfig,
        eval_dataloader: Optional[DataLoader] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.use_amp = config.use_amp and self.device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.global_step = 0
        self.epoch = 0

        self._history = []  # list of dicts: {step, loss, lr, perplexity}

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()}

    def _forward_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        with torch.autocast(device_type="cuda" if self.use_amp else "cpu", enabled=self.use_amp):
            logits, model_loss = self.model(input_ids, labels=labels)
            if model_loss is not None:
                loss = model_loss
            else:
                loss = compute_lm_loss(logits, labels, label_smoothing=self.config.label_smoothing)
        return loss

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Runs one micro-batch forward/backward. Does NOT step the optimizer --
        that happens once every gradient_accumulation_steps calls, from train()."""
        batch = self._move_batch_to_device(batch)
        loss = self._forward_loss(batch)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss encountered at step {self.global_step}: {loss.item()}. "
                "This usually indicates a learning-rate that's too high, a data corruption "
                "issue, or a numerical instability in the model."
            )

        scaled_loss = loss / self.config.gradient_accumulation_steps
        if self.use_amp:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return loss.item()

    def optimizer_step(self) -> float:
        """Clips gradients and steps the optimizer + scheduler. Returns the
        gradient norm actually applied (pre-clip value, for logging)."""
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )

        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.optimizer.zero_grad(set_to_none=True)

        return float(grad_norm)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.eval_dataloader is None:
            raise ValueError("evaluate() called but no eval_dataloader was provided")

        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        for batch in self.eval_dataloader:
            batch = self._move_batch_to_device(batch)
            loss = self._forward_loss(batch)
            total_loss += loss.item()
            total_batches += 1

        self.model.train()

        if total_batches == 0:
            raise ValueError("eval_dataloader produced zero batches")

        mean_loss = total_loss / total_batches
        perplexity = compute_perplexity(torch.tensor(mean_loss))
        return {"eval_loss": mean_loss, "eval_perplexity": perplexity}

    def train(self, on_log: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """Runs the training loop until config.max_steps is reached.
        Returns final training stats."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        accumulated_loss = 0.0
        micro_step = 0
        start_time = time.time()

        data_iter = iter(self.train_dataloader)

        while self.global_step < self.config.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            loss_value = self.train_step(batch)
            accumulated_loss += loss_value
            micro_step += 1

            if micro_step % self.config.gradient_accumulation_steps == 0:
                grad_norm = self.optimizer_step()
                self.global_step += 1

                mean_loss = accumulated_loss / self.config.gradient_accumulation_steps
                accumulated_loss = 0.0

                current_lr = self.optimizer.param_groups[0]["lr"]
                log_entry = {
                    "step": self.global_step,
                    "loss": mean_loss,
                    "lr": current_lr,
                    "grad_norm": grad_norm,
                    "epoch": self.epoch,
                }
                self._history.append(log_entry)

                if self.global_step % self.config.log_every == 0:
                    elapsed = time.time() - start_time
                    logger.info(
                        f"step {self.global_step}/{self.config.max_steps} | "
                        f"loss {mean_loss:.4f} | lr {current_lr:.2e} | "
                        f"grad_norm {grad_norm:.3f} | elapsed {elapsed:.1f}s"
                    )
                    if on_log is not None:
                        on_log(log_entry)

                if (
                    self.eval_dataloader is not None
                    and self.config.eval_every > 0
                    and self.global_step % self.config.eval_every == 0
                ):
                    eval_stats = self.evaluate()
                    logger.info(
                        f"step {self.global_step} eval | loss {eval_stats['eval_loss']:.4f} | "
                        f"perplexity {eval_stats['eval_perplexity']:.2f}"
                    )

                if (
                    self.config.checkpoint_every > 0
                    and self.global_step % self.config.checkpoint_every == 0
                ):
                    path = save_checkpoint(
                        checkpoint_dir=self.config.checkpoint_dir,
                        step=self.global_step,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=self.epoch,
                        keep_last_n=self.config.keep_last_n_checkpoints,
                    )
                    logger.info(f"Saved checkpoint: {path}")

                if self.global_step >= self.config.max_steps:
                    break

        total_time = time.time() - start_time
        return {
            "final_step": self.global_step,
            "final_epoch": self.epoch,
            "total_time_seconds": total_time,
            "history": self._history,
        }

    def resume_from_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        """Resumes from a specific checkpoint, or the latest one in
        config.checkpoint_dir if none is given. Returns True if a checkpoint
        was found and loaded, False if training should start from scratch."""
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(self.config.checkpoint_dir)
            if checkpoint_path is None:
                return False

        state = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device,
        )
        self.global_step = state["step"]
        self.epoch = state["epoch"]
        logger.info(f"Resumed from {checkpoint_path} at step {self.global_step}, epoch {self.epoch}")
        return True
