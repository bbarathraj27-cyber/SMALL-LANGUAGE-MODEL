"""
training/checkpoint.py

Save/load model + optimizer + scheduler + training-progress state, so a run
can be resumed exactly (not just the weights, but the optimizer momentum,
LR-schedule position, step count, and epoch).
"""

import os
import glob
from typing import Optional, Dict, Any

import torch


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    extra_state: Optional[Dict[str, Any]] = None,
    keep_last_n: int = 3,
) -> str:
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    state = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "extra_state": extra_state or {},
    }

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step{step}.pt")
    tmp_path = checkpoint_path + ".tmp"
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, checkpoint_path)  # atomic on POSIX -- avoids truncated files on crash
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Failed to save checkpoint to {checkpoint_path}: {e}") from e

    if keep_last_n > 0:
        _prune_old_checkpoints(checkpoint_dir, keep_last_n)

    return checkpoint_path


def _prune_old_checkpoints(checkpoint_dir: str, keep_last_n: int) -> None:
    checkpoints = _list_checkpoints(checkpoint_dir)
    if len(checkpoints) <= keep_last_n:
        return
    for _, path in checkpoints[:-keep_last_n]:
        try:
            os.remove(path)
        except OSError:
            pass  # best-effort cleanup; a failed delete shouldn't crash training


def _list_checkpoints(checkpoint_dir: str):
    """Returns list of (step, path) sorted ascending by step."""
    pattern = os.path.join(checkpoint_dir, "checkpoint_step*.pt")
    results = []
    for path in glob.glob(pattern):
        basename = os.path.basename(path)
        try:
            step_str = basename.replace("checkpoint_step", "").replace(".pt", "")
            step = int(step_str)
            results.append((step, path))
        except ValueError:
            continue  # skip files that don't match the expected naming pattern
    results.sort(key=lambda x: x[0])
    return results


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}") from e

    required_keys = {"step", "epoch", "model_state_dict", "optimizer_state_dict"}
    missing = required_keys - set(state.keys())
    if missing:
        raise ValueError(f"Checkpoint {checkpoint_path} is missing expected keys: {missing}")

    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and state["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    return {
        "step": state["step"],
        "epoch": state["epoch"],
        "extra_state": state.get("extra_state", {}),
    }


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Returns the path to the highest-step checkpoint in checkpoint_dir, or
    None if no checkpoints exist yet."""
    if not os.path.isdir(checkpoint_dir):
        return None
    checkpoints = _list_checkpoints(checkpoint_dir)
    if not checkpoints:
        return None
    return checkpoints[-1][1]
