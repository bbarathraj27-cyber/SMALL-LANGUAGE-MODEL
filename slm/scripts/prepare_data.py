#!/usr/bin/env python3
"""
scripts/prepare_data.py

Orchestrates the raw-data pipeline (Phases 2-4): collect -> clean ->
filter -> deduplicate -> split, driven by configs/data.yaml. This is
the *pretraining corpus* pipeline — not to be confused with
sft/prepare_data.py, which prepares instruction-tuning data instead.

Usage:
    python scripts/prepare_data.py --config configs/data.yaml
    python scripts/prepare_data.py --config configs/data.yaml --steps clean,filter
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ALL_STEPS = ["collect", "clean", "filter", "deduplicate", "split"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the pretraining data-prep pipeline")
    parser.add_argument("--config", required=True, help="Path to configs/data.yaml")
    parser.add_argument(
        "--steps",
        default=",".join(ALL_STEPS),
        help=f"Comma-separated subset of steps to run, in order. Default: all ({','.join(ALL_STEPS)})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved plan without running it")
    return parser


def load_config(config_path: str) -> dict:
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_steps(steps_arg: str) -> list:
    steps = [s.strip() for s in steps_arg.split(",") if s.strip()]
    unknown = [s for s in steps if s not in ALL_STEPS]
    if unknown:
        raise ValueError(f"Unknown step(s) {unknown}; valid steps are {ALL_STEPS}")
    return steps


def run_pipeline(config: dict, steps: list) -> None:
    """
    Runs each requested step by calling the matching function in
    preprocessing/<step>.py. Imported lazily here (rather than at module
    top-level) so `--dry-run` and argument parsing work even before
    preprocessing/ has real implementations behind these calls.

    Expected preprocessing/ signatures (adjust the calls below if yours differ):
        collect.collect(config)       -> writes data/raw/
        clean.clean(config)           -> reads data/raw/, writes data/cleaned/
        filter.filter_data(config)    -> filters data/cleaned/ in place or to a new dir
        deduplicate.deduplicate(config) -> writes data/deduplicated/
        split.split(config)           -> writes data/train/, data/validation/, data/test/
    """
    from preprocessing import collect, clean, filter as filter_mod, deduplicate, split

    step_fns = {
        "collect": lambda: collect.collect(config),
        "clean": lambda: clean.clean(config),
        "filter": lambda: filter_mod.filter_data(config),
        "deduplicate": lambda: deduplicate.deduplicate(config),
        "split": lambda: split.split(config),
    }

    for step in steps:
        print(f"[prepare_data] running step: {step}")
        step_fns[step]()
        print(f"[prepare_data] done: {step}")


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    steps = resolve_steps(args.steps)

    if args.dry_run:
        print(f"[prepare_data] config: {args.config}")
        print(f"[prepare_data] steps to run, in order: {steps}")
        return

    config = load_config(args.config)
    run_pipeline(config, steps)


if __name__ == "__main__":
    main()
