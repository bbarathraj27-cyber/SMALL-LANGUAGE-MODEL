"""
benchmark_inference.py

Latency/throughput measurement for generation, plus a helper to
compare a full-precision model against a quantized one (see
quantize.py) on both speed and memory side by side.

Built on top of inference/generate.py rather than reimplementing a
generation loop, so whatever sampling settings you'd actually deploy
with are exactly what gets benchmarked.
"""

import statistics
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from inference.generate import GenerationConfig, generate
from .quantize import model_size_bytes


@torch.no_grad()
def benchmark_forward_pass(
    model: nn.Module,
    input_ids: torch.Tensor,
    runs: int = 5,
    warmup: int = 1,
) -> Dict[str, float]:
    """Times a single full-sequence forward pass (the 'prefill' cost,
    i.e. before any KV cache decode steps)."""
    model.eval()
    for _ in range(warmup):
        model(input_ids, positions=torch.arange(input_ids.shape[1]))

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        model(input_ids, positions=torch.arange(input_ids.shape[1]))
        times.append(time.perf_counter() - start)

    return {
        "mean_latency_s": statistics.mean(times),
        "std_latency_s": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_latency_s": min(times),
        "runs": runs,
    }


@torch.no_grad()
def benchmark_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    config: GenerationConfig,
    num_layers: Optional[int] = None,
    runs: int = 3,
    warmup: int = 1,
) -> Dict[str, float]:
    """Times end-to-end generation (prefill + all decode steps) and
    reports tokens/sec based on newly generated tokens only (the
    prompt tokens aren't 'generated' so they're excluded from the rate)."""
    model.eval()
    for _ in range(warmup):
        generate(model, input_ids, config, num_layers=num_layers)

    times = []
    generated_counts = []
    for _ in range(runs):
        start = time.perf_counter()
        out = generate(model, input_ids, config, num_layers=num_layers)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        generated_counts.append(out.shape[1] - input_ids.shape[1])

    mean_latency = statistics.mean(times)
    mean_generated = statistics.mean(generated_counts)
    tokens_per_sec = mean_generated / mean_latency if mean_latency > 0 else float("inf")

    return {
        "mean_latency_s": mean_latency,
        "std_latency_s": statistics.stdev(times) if len(times) > 1 else 0.0,
        "tokens_per_sec": tokens_per_sec,
        "mean_tokens_generated": mean_generated,
        "runs": runs,
    }


def compare_quantized_vs_full(
    model_fp: nn.Module,
    model_quantized: nn.Module,
    input_ids: torch.Tensor,
    config: GenerationConfig,
    num_layers: Optional[int] = None,
    runs: int = 3,
) -> Dict[str, Any]:
    """Runs benchmark_generate on both models and reports speedup plus
    memory savings from quantization side by side."""
    fp_stats = benchmark_generate(model_fp, input_ids, config, num_layers=num_layers, runs=runs)
    quant_stats = benchmark_generate(model_quantized, input_ids, config, num_layers=num_layers, runs=runs)

    fp_size = model_size_bytes(model_fp)
    quant_size = model_size_bytes(model_quantized)

    return {
        "full_precision": fp_stats,
        "quantized": quant_stats,
        "speedup": (
            fp_stats["mean_latency_s"] / quant_stats["mean_latency_s"]
            if quant_stats["mean_latency_s"] > 0 else float("inf")
        ),
        "full_precision_size_bytes": fp_size,
        "quantized_size_bytes": quant_size,
        "size_reduction_ratio": quant_size / fp_size if fp_size > 0 else 0.0,
    }
