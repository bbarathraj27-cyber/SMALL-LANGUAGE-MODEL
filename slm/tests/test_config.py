"""Validation tests for configs/*.yaml.

These don't test application logic (data.yaml and training.yaml aren't
consumed by any module yet — that lands with preprocessing scripts and
training/), but they do catch the two failure modes that matter most
for config files: invalid YAML syntax, and cross-file inconsistencies
(e.g. a dataset block_size that silently doesn't match the model's
max_position_embeddings, which would raise a shape error deep inside
training instead of at config-load time).

Run with: python tests/test_config_files.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml

from model.slm import SLMConfig

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def _load(name: str) -> dict:
    path = os.path.join(CONFIG_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_model_yaml_loads_via_slmconfig():
    config = SLMConfig.from_yaml(os.path.join(CONFIG_DIR, "model.yaml"))
    assert config.vocab_size > 0
    print(f"test_model_yaml_loads_via_slmconfig: PASSED (hidden_size={config.hidden_size})")


def test_data_yaml_parses_and_has_required_sections():
    data_config = _load("data.yaml")
    required_sections = {
        "paths",
        "collection",
        "cleaning",
        "filtering",
        "deduplication",
        "split",
        "tokenization",
        "pretraining_dataset",
        "instruction_dataset",
    }
    missing = required_sections - set(data_config.keys())
    assert not missing, f"data.yaml missing sections: {missing}"
    print("test_data_yaml_parses_and_has_required_sections: PASSED")


def test_data_yaml_split_ratios_sum_to_one():
    data_config = _load("data.yaml")
    split = data_config["split"]
    total = split["train_ratio"] + split["val_ratio"] + split["test_ratio"]
    assert abs(total - 1.0) < 1e-6, f"Split ratios sum to {total}, expected 1.0"
    print("test_data_yaml_split_ratios_sum_to_one: PASSED")


def test_data_yaml_block_size_matches_model_max_position_embeddings():
    data_config = _load("data.yaml")
    model_config = SLMConfig.from_yaml(os.path.join(CONFIG_DIR, "model.yaml"))
    block_size = data_config["pretraining_dataset"]["block_size"]
    assert block_size <= model_config.max_position_embeddings, (
        f"data.yaml block_size ({block_size}) exceeds model.yaml "
        f"max_position_embeddings ({model_config.max_position_embeddings}); "
        "RoPE cache would need to extend beyond the model's trained context."
    )
    print(
        "test_data_yaml_block_size_matches_model_max_position_embeddings: PASSED "
        f"(block_size={block_size}, max_position_embeddings={model_config.max_position_embeddings})"
    )


def test_data_yaml_tokenization_eos_matches_instruction_dataset_eos():
    data_config = _load("data.yaml")
    tok_eos = data_config["tokenization"]["eos_token"]
    sft_eos = data_config["instruction_dataset"]["eos_token"]
    assert tok_eos == sft_eos, (
        f"tokenization.eos_token ({tok_eos!r}) does not match "
        f"instruction_dataset.eos_token ({sft_eos!r}) — pretraining and SFT "
        "must use the same document/sequence terminator."
    )
    print("test_data_yaml_tokenization_eos_matches_instruction_dataset_eos: PASSED")


def test_training_yaml_parses_and_has_required_sections():
    training_config = _load("training.yaml")
    required_sections = {"seed", "device", "mixed_precision", "pretraining", "sft", "logging"}
    missing = required_sections - set(training_config.keys())
    assert not missing, f"training.yaml missing sections: {missing}"
    print("test_training_yaml_parses_and_has_required_sections: PASSED")


def test_training_yaml_pretraining_has_optimizer_and_scheduler():
    training_config = _load("training.yaml")
    pretraining = training_config["pretraining"]
    assert "optimizer" in pretraining and "learning_rate" in pretraining["optimizer"]
    assert "scheduler" in pretraining and "warmup_steps" in pretraining["scheduler"]
    assert pretraining["optimizer"]["learning_rate"] > 0
    print("test_training_yaml_pretraining_has_optimizer_and_scheduler: PASSED")


def test_training_yaml_sft_has_optimizer_and_scheduler():
    training_config = _load("training.yaml")
    sft = training_config["sft"]
    assert "optimizer" in sft and "learning_rate" in sft["optimizer"]
    assert "scheduler" in sft and "warmup_steps" in sft["scheduler"]
    print("test_training_yaml_sft_has_optimizer_and_scheduler: PASSED")


def test_training_yaml_mixed_precision_matches_model_precision_intent():
    training_config = _load("training.yaml")
    valid_precisions = {"no", "fp16", "bf16"}
    assert training_config["mixed_precision"] in valid_precisions
    print("test_training_yaml_mixed_precision_matches_model_precision_intent: PASSED")


def test_training_yaml_gradient_clipping_positive():
    training_config = _load("training.yaml")
    for phase in ("pretraining", "sft"):
        max_norm = training_config[phase]["gradient_clipping"]["max_norm"]
        assert max_norm > 0, f"{phase}.gradient_clipping.max_norm must be positive, got {max_norm}"
    print("test_training_yaml_gradient_clipping_positive: PASSED")


def run_all_tests():
    test_fns = [
        obj
        for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = []
    for fn in test_fns:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures.append((fn.__name__, e))
            print(f"{fn.__name__}: FAILED -- {e}")

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} / {len(test_fns)} tests FAILED")
        for name, e in failures:
            print(f"  - {name}: {e}")
        sys.exit(1)
    else:
        print(f"All {len(test_fns)} tests PASSED")


if __name__ == "__main__":
    run_all_tests()
