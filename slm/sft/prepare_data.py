"""
sft/prepare_data.py

Converts raw instruction data (JSONL of {"prompt": ..., "response": ...}
records) into tokenized, prompt-masked examples ready for SFT training.

Each output example has:
    input_ids: [BOS] + prompt_tokens + response_tokens + [EOS]
    labels:    same length as input_ids, but every prompt-token position
               (and BOS) is set to IGNORE_INDEX (-100) so the loss only
               trains on the model's own generated response.

Usage:
    python sft/prepare_data.py \
        --input data/sft/raw_instructions.jsonl \
        --tokenizer tokenizer/tokenizer.json \
        --output data/sft/prepared.jsonl \
        --max-length 1024
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.tokenizer import SLMTokenizer, BOS_ID, EOS_ID

IGNORE_INDEX = -100


def load_raw_examples(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}") from e

            if "prompt" not in record or "response" not in record:
                raise ValueError(
                    f"Line {line_num} of {path} is missing 'prompt' or 'response' key: {record}"
                )
            if not isinstance(record["prompt"], str) or not isinstance(record["response"], str):
                raise ValueError(f"Line {line_num} of {path}: 'prompt' and 'response' must be strings")
            if not record["prompt"].strip() or not record["response"].strip():
                raise ValueError(f"Line {line_num} of {path}: 'prompt' or 'response' is empty")

            examples.append(record)

    if not examples:
        raise ValueError(f"No valid examples found in {path}")

    return examples


def prepare_example(
    tokenizer: SLMTokenizer,
    prompt: str,
    response: str,
    max_length: int,
) -> Optional[Dict[str, List[int]]]:
    """Tokenizes one prompt/response pair into a prompt-masked training
    example. Returns None if the example can't fit within max_length at all
    (prompt alone already exceeds the budget) -- callers should skip these."""
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    response_ids = tokenizer.encode(response, add_bos=False, add_eos=True)

    input_ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(response_ids)

    if len(input_ids) > max_length:
        # Truncate from the left of the response, keeping the full prompt --
        # truncating the prompt would remove the instruction the model is
        # supposed to follow.
        if len(prompt_ids) >= max_length:
            return None  # prompt alone doesn't fit; nothing useful to train on
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    if all(l == IGNORE_INDEX for l in labels):
        return None  # response was truncated away entirely; no training signal

    return {"input_ids": input_ids, "labels": labels}


def prepare_dataset(
    input_path: str,
    tokenizer_path: str,
    output_path: str,
    max_length: int = 1024,
) -> Dict[str, int]:
    tokenizer = SLMTokenizer(tokenizer_path)
    raw_examples = load_raw_examples(input_path)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    kept = 0
    skipped = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for record in raw_examples:
            prepared = prepare_example(tokenizer, record["prompt"], record["response"], max_length)
            if prepared is None:
                skipped += 1
                continue
            out_f.write(json.dumps(prepared) + "\n")
            kept += 1

    if kept == 0:
        raise ValueError(
            f"All {len(raw_examples)} examples were skipped (prompt too long for "
            f"max_length={max_length}). Increase --max-length or check your data."
        )

    return {"total": len(raw_examples), "kept": kept, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare instruction data for SFT.")
    parser.add_argument("--input", required=True, help="JSONL file of {prompt, response} records")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--output", required=True, help="Output JSONL path for prepared examples")
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    try:
        stats = prepare_dataset(args.input, args.tokenizer, args.output, args.max_length)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Prepared {stats['kept']}/{stats['total']} examples "
        f"({stats['skipped']} skipped) -> {args.output}"
    )


if __name__ == "__main__":
    main()
