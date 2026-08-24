"""
quantize.py

Post-training weight-only quantization (int8) for trained SLM checkpoints.

Only Linear layer weights are quantized (the vast majority of a
transformer's parameters). Activations stay in full precision — this
is the same "weight-only" scheme used by tools like bitsandbytes and
llama.cpp's Q8_0: it trades a small amount of accuracy for roughly a
4x reduction in weight memory (fp32 -> int8), dequantizing on the fly
inside the forward pass. No separate calibration pass over activation
data is needed, which keeps it simple enough to run right after
training with nothing but the checkpoint itself.
"""

from typing import Any, Dict, Iterable, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_tensor(weight: torch.Tensor, num_bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-output-channel symmetric quantization.

    weight: (out_features, in_features)
    Returns (qweight int8, scale) where scale has shape (out_features, 1)
    and dequantized ~= qweight.float() * scale.

    Symmetric quantization always maps zero to zero, so there's no
    separate zero_point to store or apply — one less thing to get wrong
    when this is later ported to a lower-level runtime.
    """
    if weight.dim() != 2:
        raise ValueError(f"quantize_tensor expects a 2D weight, got shape {tuple(weight.shape)}")
    qmax = 2 ** (num_bits - 1) - 1  # 127 for int8

    max_per_row = weight.abs().amax(dim=1, keepdim=True)
    max_per_row = torch.clamp(max_per_row, min=1e-8)  # avoid div-by-zero for an all-zero row
    scale = max_per_row / qmax

    qweight = torch.clamp(torch.round(weight / scale), -qmax - 1, qmax).to(torch.int8)
    return qweight, scale


def dequantize_tensor(qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of quantize_tensor."""
    return qweight.to(torch.float32) * scale


class QuantizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear that stores its weight as int8
    and dequantizes on the fly during forward. Bias (if any) stays fp32
    since it's a tiny fraction of total parameters and not worth the
    precision loss."""

    def __init__(self, qweight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor = None):
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self.out_features, self.in_features = qweight.shape

    @classmethod
    def from_linear(cls, linear: nn.Linear, num_bits: int = 8) -> "QuantizedLinear":
        qweight, scale = quantize_tensor(linear.weight.data, num_bits=num_bits)
        bias = linear.bias.data.clone() if linear.bias is not None else None
        return cls(qweight, scale, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize_tensor(self.qweight, self.scale)
        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bits=8"


def quantize_model(
    model: nn.Module,
    num_bits: int = 8,
    exclude_names: Iterable[str] = (),
) -> Tuple[nn.Module, List[str]]:
    """
    Replace every nn.Linear submodule in `model` with a QuantizedLinear,
    in place, except modules whose dotted name contains any string in
    exclude_names (e.g. exclude_names=["lm_head"] to keep the output
    projection at full precision, since errors there directly skew
    every logit).

    Returns (model, replaced_names) so callers can confirm what changed.
    """
    replaced: List[str] = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and not any(ex in full_name for ex in exclude_names):
                setattr(module, child_name, QuantizedLinear.from_linear(child, num_bits=num_bits))
                replaced.append(full_name)
    return model, replaced


def model_size_bytes(model: nn.Module) -> int:
    """Approximate in-memory size of a model's parameters + buffers.
    Works for both full-precision models and quantized ones (buffers
    include the int8 qweight tensors, so quantized models correctly
    report a smaller total)."""
    total = 0
    for p in model.parameters():
        total += p.nelement() * p.element_size()
    for b in model.buffers():
        total += b.nelement() * b.element_size()
    return total
