#!/usr/bin/env python3
"""
scripts/evaluate.py

CLI wrapper around evaluation/evaluate.py (Phase 13): runs perplexity
(and an optional multiple-choice benchmark) on a checkpoint, writing a
combined JSON results file.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/pretraining/final --eval-data data/test
    python scripts/evaluate.py --checkpoint checkpoints/pretraining/final --eval-data data/test \
        --benchmark data/benchmark.jsonl --output logs/evaluation/results.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint: perplexity + optional benchmark")
    parser.add_argument("--checkpoint", required=True, help="Path (no extension) to the checkpoint to evaluate")
    parser.add_argument("--eval-data", required=True, help="Path to held-out pretraining-format eval data")
    parser.add_argument("--benchmark", default=None, help="Optional path to a multiple-choice benchmark JSONL")
    parser.add_argument(
        "--output",
        default="logs/evaluation/results.json",
        help="Where to write the combined results JSON (default: logs/evaluation/results.json)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved settings without evaluating")
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.dry_run:
        print(f"[evaluate] checkpoint: {args.checkpoint}")
        print(f"[evaluate] eval_data: {args.eval_data}")
        print(f"[evaluate] benchmark: {args.benchmark}")
        print(f"[evaluate] output: {args.output}")
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Lazy import: evaluation/evaluate.py already implements the CLI-facing
    # entrypoint for this (Phase 13) — this script just forwards args to it.
    # Expected signature: run_evaluation(checkpoint, eval_data, benchmark, output_path).
    # Adjust if yours differs.
    from evaluation import evaluate as evaluate_mod

    results = evaluate_mod.run_evaluation(
        checkpoint=args.checkpoint,
        eval_data=args.eval_data,
        benchmark=args.benchmark,
        output_path=args.output,
    )
    print(f"[evaluate] wrote results to {args.output}")
    print(results)


if __name__ == "__main__":
    main()
