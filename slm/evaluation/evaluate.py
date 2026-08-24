"""
evaluation/evaluate.py

CLI entrypoint for base-model evaluation (Phase 13 in the project plan).
Runs perplexity evaluation over a pretraining-format dataset and, if a
benchmark file is provided, also runs the multiple-choice benchmark.
Writes a combined JSON results file.

Usage:
    python evaluation/evaluate.py \
        --model-config configs/model.yaml \
        --checkpoint checkpoints/pretraining/checkpoint_step10000.pt \
        --eval-shards data/test \
        --tokenizer tokenizer/tokenizer.json \
        --benchmark data/benchmark/custom_tasks.jsonl \
        --output logs/evaluation/base_eval.json
"""

import argparse
import json
import logging
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.slm import SLM, SLMConfig
from tokenizer.tokenizer import SLMTokenizer
from training.checkpoint import load_checkpoint
from dataset.dataloader import build_pretrain_dataloader

from evaluation.perplexity import compute_dataset_perplexity
from evaluation.benchmark import run_benchmark

logger = logging.getLogger("evaluation.evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a base (pretrained) SLM checkpoint.")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-shards", required=True, help="Directory of tokenized eval shards")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--benchmark", default=None, help="Optional JSONL multiple-choice benchmark file")
    parser.add_argument("--output", default=None, help="Optional path to write JSON results")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    try:
        model_config = SLMConfig.from_yaml(args.model_config)
        model = SLM(model_config)
        load_checkpoint(args.checkpoint, model)
        tokenizer = SLMTokenizer(args.tokenizer)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Setup error: {e}")
        sys.exit(1)

    results = {}

    try:
        eval_dataloader = build_pretrain_dataloader(
            shard_dir=args.eval_shards,
            block_size=model_config.max_position_embeddings,
            batch_size=args.batch_size,
            shuffle=False,
        )
        ppl_results = compute_dataset_perplexity(
            model, eval_dataloader, max_batches=args.max_batches
        )
        results["perplexity"] = ppl_results
        logger.info(
            f"Perplexity: {ppl_results['perplexity']:.2f} "
            f"(loss {ppl_results['loss']:.4f}, {ppl_results['num_tokens']} tokens)"
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Perplexity evaluation failed: {e}")
        sys.exit(1)

    if args.benchmark:
        try:
            bench_results = run_benchmark(model, tokenizer, args.benchmark)
            results["benchmark"] = {
                "accuracy": bench_results["accuracy"],
                "correct": bench_results["correct"],
                "total": bench_results["total"],
            }
            logger.info(
                f"Benchmark accuracy: {bench_results['accuracy']:.2%} "
                f"({bench_results['correct']}/{bench_results['total']})"
            )
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            logger.error(f"Benchmark evaluation failed: {e}")
            sys.exit(1)

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
