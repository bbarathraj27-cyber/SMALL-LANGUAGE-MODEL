"""
export.py

Save/load a trained SLM checkpoint as a portable bundle: model weights
(`<path>.pt`, a torch state_dict) plus a JSON manifest (`<path>.json`)
recording the architecture config, tokenizer info, and a checksum of
the weights.

The checksum matters more than it looks: without it, a truncated
download or a manifest/weights pair from two different training runs
loads silently and fails in some confusing downstream way (garbage
generations, shape mismatches deep in a layer). Checking it up front
turns that into an immediate, clear error.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def _state_dict_checksum(state_dict: Dict[str, torch.Tensor]) -> str:
    """SHA-256 over every tensor's bytes, in a stable (sorted-key) order
    so the same weights always produce the same checksum regardless of
    state_dict insertion order."""
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        hasher.update(key.encode("utf-8"))
        tensor = state_dict[key]
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def export_checkpoint(
    model: nn.Module,
    config: Dict[str, Any],
    path: str,
    tokenizer_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Writes `<path>.pt` (state dict) and `<path>.json` (manifest).
    `config` should be the architecture config this model was built
    from (e.g. the parsed configs/model.yaml) so a fresh model can be
    reconstructed with matching shapes before load_checkpoint fills it in.

    Returns the manifest path.
    """
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    weights_path = path + ".pt"
    manifest_path = path + ".json"

    state_dict = model.state_dict()
    torch.save(state_dict, weights_path)

    manifest = {
        "config": config,
        "tokenizer": tokenizer_meta or {},
        "checksum_sha256": _state_dict_checksum(state_dict),
        "weights_file": os.path.basename(weights_path),
        "num_parameters": sum(p.nelement() for p in model.parameters()),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def load_checkpoint(
    model: nn.Module,
    path: str,
    strict: bool = True,
    verify_checksum: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Loads weights from `<path>.pt` into `model` in place, using the
    `<path>.json` manifest to verify integrity first.

    `model` must already be constructed with the right architecture
    (e.g. via model.slm.SLM(**manifest["config"]) if you've read the
    manifest ahead of time) — this function only fills in weights, it
    doesn't build the model for you.

    Returns (model, manifest).
    """
    path = str(path)
    weights_path = path + ".pt"
    manifest_path = path + ".json"

    with open(manifest_path) as f:
        manifest = json.load(f)

    state_dict = torch.load(weights_path, map_location="cpu")

    if verify_checksum:
        actual = _state_dict_checksum(state_dict)
        expected = manifest.get("checksum_sha256")
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch loading {weights_path}: "
                f"expected {expected}, got {actual}. The weights file may "
                f"be corrupted, truncated, or paired with the wrong manifest."
            )

    model.load_state_dict(state_dict, strict=strict)
    return model, manifest
