"""
train_tokenizer.py

Trains a Byte-Pair Encoding (BPE) tokenizer on a text corpus and saves it
in the formats consumed by the rest of the pipeline:
    tokenizer/tokenizer.json   -- full HuggingFace `tokenizers` artifact
    tokenizer/vocab.json       -- token -> id mapping
    tokenizer/merges.txt       -- BPE merge rules

Usage:
    python tokenizer/train_tokenizer.py \
        --input data/cleaned/train.txt \
        --vocab-size 32000 \
        --output-dir tokenizer/

The input can be a single file or a directory of .txt files (all files
are concatenated as the training corpus). Special tokens are fixed and
must match what preprocessing/tokenize.py and dataset/instruction_dataset.py
expect: <pad>, <bos>, <eos>, <unk>.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders

# Special tokens: fixed IDs so every downstream module can rely on them.
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


def collect_input_files(input_path: str) -> List[str]:
    """Resolve a file-or-directory input path into a list of text file paths."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if path.is_file():
        return [str(path)]

    if path.is_dir():
        files = sorted(str(p) for p in path.rglob("*.txt") if p.is_file())
        if not files:
            raise ValueError(f"No .txt files found under directory: {input_path}")
        return files

    raise ValueError(f"Input path is neither a file nor a directory: {input_path}")


def build_tokenizer(vocab_size: int) -> Tokenizer:
    """Construct an untrained BPE tokenizer with the pipeline this project uses."""
    if vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError(
            f"vocab_size ({vocab_size}) must be greater than the number of "
            f"special tokens ({len(SPECIAL_TOKENS)})"
        )

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def train(
    input_path: str,
    vocab_size: int,
    output_dir: str,
    min_frequency: int = 2,
) -> None:
    files = collect_input_files(input_path)
    print(f"Training BPE tokenizer on {len(files)} file(s), target vocab_size={vocab_size}")

    tokenizer = build_tokenizer(vocab_size)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train(files=files, trainer=trainer)

    # Sanity check: special tokens must land at the fixed ids we rely on downstream.
    for expected_id, token in enumerate(SPECIAL_TOKENS):
        actual_id = tokenizer.token_to_id(token)
        if actual_id != expected_id:
            raise RuntimeError(
                f"Special token '{token}' landed at id {actual_id}, expected {expected_id}. "
                "This will break every downstream module that hardcodes special-token ids."
            )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer_json_path = output_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))
    print(f"Saved: {tokenizer_json_path}")

    vocab = tokenizer.get_vocab()
    vocab_json_path = output_path / "vocab.json"
    with open(vocab_json_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2, sort_keys=False)
    print(f"Saved: {vocab_json_path}")

    # Extract merges from the trained model state for a standalone merges.txt,
    # in case downstream tools expect the classic GPT-2-style artifact.
    model_state = json.loads(tokenizer.to_str())
    merges = model_state.get("model", {}).get("merges", [])
    merges_txt_path = output_path / "merges.txt"
    with open(merges_txt_path, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for merge in merges:
            if isinstance(merge, list):
                f.write(f"{merge[0]} {merge[1]}\n")
            else:
                f.write(f"{merge}\n")
    print(f"Saved: {merges_txt_path}")

    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"Done. Actual vocab size: {actual_vocab_size}")
    if actual_vocab_size != vocab_size:
        print(
            f"WARNING: actual vocab size ({actual_vocab_size}) differs from "
            f"requested ({vocab_size}). This happens when the corpus is too "
            f"small to produce enough distinct merges. Update configs/model.yaml "
            f"and configs/data.yaml vocab_size to match, or provide more data."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer for the SLM project.")
    parser.add_argument("--input", required=True, help="Path to a .txt file or directory of .txt files")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Target vocabulary size")
    parser.add_argument("--output-dir", default="tokenizer", help="Directory to write tokenizer artifacts")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum pair frequency to merge")
    args = parser.parse_args()

    try:
        train(
            input_path=args.input,
            vocab_size=args.vocab_size,
            output_dir=args.output_dir,
            min_frequency=args.min_frequency,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
