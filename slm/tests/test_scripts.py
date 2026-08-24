"""
test_scripts.py

Tests for scripts/*.py.

These scripts are thin CLI wrappers around modules built in earlier
phases (preprocessing/, tokenizer/, training/, sft/, evaluation/,
inference/, model/). Rather than depending on those modules' exact
internal function signatures (which this test file can't see), these
tests exercise the parts of each script that are self-contained:
argument parsing, config-loading, validation, and --dry-run output.
Each script's heavy lifting is behind a lazy import inside main(), so
none of that is triggered here.
"""

import argparse
import importlib
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import prepare_data as script_prepare_data
import train_tokenizer as script_train_tokenizer
import tokenize_data as script_tokenize_data
import train as script_train
import train_sft as script_train_sft
import evaluate as script_evaluate
import run_inference as script_run_inference


def write_yaml(tmp_path, name, content):
    path = Path(tmp_path) / name
    with open(path, "w") as f:
        yaml.safe_dump(content, f)
    return str(path)


# ---------------------------------------------------------------------------
# prepare_data.py
# ---------------------------------------------------------------------------

def test_prepare_data_default_steps_is_all_steps():
    args = script_prepare_data.build_parser().parse_args(["--config", "configs/data.yaml"])
    steps = script_prepare_data.resolve_steps(args.steps)
    assert steps == script_prepare_data.ALL_STEPS


def test_prepare_data_resolve_steps_subset_preserves_order():
    steps = script_prepare_data.resolve_steps("filter,clean")
    assert steps == ["filter", "clean"]


def test_prepare_data_resolve_steps_rejects_unknown_step():
    with pytest.raises(ValueError):
        script_prepare_data.resolve_steps("clean,not_a_real_step")


def test_prepare_data_dry_run_does_not_import_preprocessing(tmp_path, capsys):
    config_path = write_yaml(tmp_path, "data.yaml", {"raw_dir": "data/raw"})
    script_prepare_data.main(["--config", config_path, "--dry-run"])
    captured = capsys.readouterr()
    assert "prepare_data" in captured.out
    assert "preprocessing" not in sys.modules  # confirms the lazy import was skipped


# ---------------------------------------------------------------------------
# train_tokenizer.py
# ---------------------------------------------------------------------------

def test_train_tokenizer_vocab_size_override(tmp_path):
    config_path = write_yaml(tmp_path, "data.yaml", {"vocab_size": 16000})
    args = script_train_tokenizer.build_parser().parse_args(
        ["--config", config_path, "--vocab-size", "32000"]
    )
    config = script_train_tokenizer.load_config(args.config)
    assert config["vocab_size"] == 16000  # file itself unchanged
    assert args.vocab_size == 32000       # CLI override read separately


def test_train_tokenizer_default_output_dir():
    args = script_train_tokenizer.build_parser().parse_args(["--config", "configs/data.yaml"])
    assert args.output_dir == "tokenizer"


def test_train_tokenizer_dry_run_reports_effective_vocab_size(tmp_path, capsys):
    config_path = write_yaml(tmp_path, "data.yaml", {"vocab_size": 16000})
    script_train_tokenizer.main(["--config", config_path, "--vocab-size", "32000", "--dry-run"])
    captured = capsys.readouterr()
    assert "32000" in captured.out


# ---------------------------------------------------------------------------
# tokenize_data.py
# ---------------------------------------------------------------------------

def test_tokenize_data_default_splits():
    steps = script_tokenize_data.resolve_splits("train,validation,test")
    assert steps == ["train", "validation", "test"]


def test_tokenize_data_resolve_splits_rejects_unknown():
    with pytest.raises(ValueError):
        script_tokenize_data.resolve_splits("train,bogus")


def test_tokenize_data_shard_size_override_applied_to_config(tmp_path, capsys):
    config_path = write_yaml(tmp_path, "data.yaml", {"shard_size": 50000})
    script_tokenize_data.main(["--config", config_path, "--shard-size", "100000", "--dry-run"])
    captured = capsys.readouterr()
    assert "100000" in captured.out


# ---------------------------------------------------------------------------
# train.py
# ---------------------------------------------------------------------------

def test_train_requires_both_configs():
    parser = script_train.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--model-config", "configs/model.yaml"])  # missing --training-config


def test_train_dry_run_reports_resume_path(tmp_path, capsys):
    model_config = write_yaml(tmp_path, "model.yaml", {"dim": 512})
    training_config = write_yaml(tmp_path, "training.yaml", {"lr": 0.0003})
    script_train.main(
        [
            "--model-config", model_config,
            "--training-config", training_config,
            "--resume", "checkpoints/pretraining/step_1000",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert "checkpoints/pretraining/step_1000" in captured.out


def test_train_default_output_dir():
    args = script_train.build_parser().parse_args(
        ["--model-config", "configs/model.yaml", "--training-config", "configs/training.yaml"]
    )
    assert args.output_dir == "checkpoints/pretraining"


# ---------------------------------------------------------------------------
# train_sft.py
# ---------------------------------------------------------------------------

def test_train_sft_default_output_dir():
    args = script_train_sft.build_parser().parse_args(
        [
            "--checkpoint", "checkpoints/pretraining/final",
            "--data", "data/instructions.jsonl",
            "--training-config", "configs/training.yaml",
        ]
    )
    assert args.output_dir == "checkpoints/sft"


def test_train_sft_dry_run_does_not_import_sft_module(tmp_path, capsys):
    training_config = write_yaml(tmp_path, "training.yaml", {"lr": 0.0001})
    script_train_sft.main(
        [
            "--checkpoint", "checkpoints/pretraining/final",
            "--data", "data/instructions.jsonl",
            "--training-config", training_config,
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert "data/instructions.jsonl" in captured.out
    assert "sft.train_sft" not in sys.modules


# ---------------------------------------------------------------------------
# evaluate.py
# ---------------------------------------------------------------------------

def test_evaluate_default_output_path():
    args = script_evaluate.build_parser().parse_args(
        ["--checkpoint", "checkpoints/pretraining/final", "--eval-data", "data/test"]
    )
    assert args.output == "logs/evaluation/results.json"


def test_evaluate_dry_run_reports_benchmark_none_by_default(capsys):
    script_evaluate.main(
        ["--checkpoint", "checkpoints/pretraining/final", "--eval-data", "data/test", "--dry-run"]
    )
    captured = capsys.readouterr()
    assert "benchmark: None" in captured.out


def test_evaluate_creates_output_directory_before_running(tmp_path):
    # Point --output at a directory that doesn't exist yet; even though
    # the (lazy-imported) evaluation module will fail without a real
    # checkpoint, mkdir happens before that import, so we can verify it
    # directly by checking the directory exists after the ImportError/
    # AttributeError is raised past mkdir.
    output_path = Path(tmp_path) / "nested" / "results.json"
    with pytest.raises(Exception):
        script_evaluate.main(
            [
                "--checkpoint", "checkpoints/pretraining/final",
                "--eval-data", "data/test",
                "--output", str(output_path),
            ]
        )
    assert output_path.parent.exists()


# ---------------------------------------------------------------------------
# run_inference.py
# ---------------------------------------------------------------------------

def test_run_inference_requires_prompt_or_chat():
    args = script_run_inference.build_parser().parse_args(
        ["--checkpoint", "checkpoints/sft/final"]
    )
    with pytest.raises(ValueError):
        script_run_inference.validate_args(args)


def test_run_inference_rejects_prompt_and_chat_together():
    args = script_run_inference.build_parser().parse_args(
        ["--checkpoint", "checkpoints/sft/final", "--prompt", "hi", "--chat"]
    )
    with pytest.raises(ValueError):
        script_run_inference.validate_args(args)


def test_run_inference_accepts_prompt_only():
    args = script_run_inference.build_parser().parse_args(
        ["--checkpoint", "checkpoints/sft/final", "--prompt", "hi"]
    )
    script_run_inference.validate_args(args)  # should not raise


def test_run_inference_accepts_chat_only():
    args = script_run_inference.build_parser().parse_args(
        ["--checkpoint", "checkpoints/sft/final", "--chat"]
    )
    script_run_inference.validate_args(args)  # should not raise


def test_run_inference_default_sampling_settings():
    args = script_run_inference.build_parser().parse_args(
        ["--checkpoint", "checkpoints/sft/final", "--prompt", "hi"]
    )
    assert args.max_new_tokens == 128
    assert args.temperature == 0.8
    assert args.top_p == 0.95
    assert args.greedy is False
    assert args.device == "cpu"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
