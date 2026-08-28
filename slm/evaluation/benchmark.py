"""
evaluation/benchmark.py

A lightweight multiple-choice benchmark harness -- the "MMLU-lite / custom
task set" referenced in the project plan. Scores the model by comparing the
summed log-probability the model assigns to each candidate answer string
(appended to a shared question prompt), and picking the highest-scoring
candidate. This is the standard way to benchmark small/base models that
aren't yet instruction-tuned enough to reliably output "A", "B", "C", "D"
directly.

Benchmark format (JSONL), one record per question:
    {"question": "...", "choices": ["choice A text", "choice B text", ...], "answer_index": 2}
"""

import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from tokenizer.tokenizer import SLMTokenizer

IGNORE_INDEX = -100


def load_benchmark_file(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            for key in ("question", "choices", "answer_index"):
                if key not in record:
                    raise ValueError(f"Line {line_num} of {path} missing '{key}'")
            if not isinstance(record["choices"], list) or len(record["choices"]) < 2:
                raise ValueError(f"Line {line_num} of {path}: 'choices' must be a list of >= 2 items")
            if not (0 <= record["answer_index"] < len(record["choices"])):
                raise ValueError(
                    f"Line {line_num} of {path}: answer_index {record['answer_index']} "
                    f"out of range for {len(record['choices'])} choices"
                )

            items.append(record)

    if not items:
        raise ValueError(f"No valid benchmark items found in {path}")

    return items


@torch.no_grad()
def score_choice(
    model: nn.Module,
    tokenizer: SLMTokenizer,
    question: str,
    choice: str,
    device: str = "cpu",
    max_context: int = 512,
) -> float:
    """Returns the mean log-probability the model assigns to `choice`'s
    tokens, conditioned on `question`. Mean (not sum) so choices of
    different lengths are compared fairly."""
    question_ids = tokenizer.encode(question, add_bos=True, add_eos=False)
    choice_ids = tokenizer.encode(choice, add_bos=False, add_eos=False)

    if not choice_ids:
        raise ValueError(f"Choice '{choice}' tokenized to an empty sequence")

    input_ids = question_ids + choice_ids
    if len(input_ids) > max_context:
        # Truncate from the left of the question, keeping the full choice --
        # we need every choice token scored to compare fairly.
        overflow = len(input_ids) - max_context
        if overflow >= len(question_ids):
            raise ValueError(
                f"Choice '{choice}' alone exceeds max_context={max_context}; cannot score"
            )
        question_ids = question_ids[overflow:]
        input_ids = question_ids + choice_ids

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    # model/slm.py's SLM.forward returns a dict with keys "logits",
    # "loss", "past_key_values" -- not a tuple.
    output = model(input_tensor)  # (1, seq_len, vocab_size)
    logits = output["logits"]

    # logits[t] predicts token[t+1]; we want log-probs of the choice tokens,
    # so we look at logits at positions (len(question_ids)-1) .. (end-2).
    choice_start = len(question_ids)
    log_probs = torch.log_softmax(logits[0], dim=-1)

    total_log_prob = 0.0
    for i, token_id in enumerate(choice_ids):
        pred_position = choice_start + i - 1
        if pred_position < 0 or pred_position >= log_probs.size(0):
            raise RuntimeError(
                f"Position {pred_position} out of range for sequence length {log_probs.size(0)}"
            )
        total_log_prob += log_probs[pred_position, token_id].item()

    return total_log_prob / len(choice_ids)


@torch.no_grad()
def run_benchmark(
    model: nn.Module,
    tokenizer: SLMTokenizer,
    benchmark_path: str,
    device: str = "cpu",
    max_context: int = 512,
) -> Dict[str, float]:
    items = load_benchmark_file(benchmark_path)
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    per_item_results = []

    for item in items:
        scores = [
            score_choice(model, tokenizer, item["question"], choice, device, max_context)
            for choice in item["choices"]
        ]
        predicted_index = max(range(len(scores)), key=lambda i: scores[i])
        is_correct = predicted_index == item["answer_index"]

        correct += int(is_correct)
        total += 1
        per_item_results.append({
            "question": item["question"],
            "predicted_index": predicted_index,
            "answer_index": item["answer_index"],
            "correct": is_correct,
            "scores": scores,
        })

    if total == 0:
        raise ValueError(f"Benchmark {benchmark_path} produced zero scoreable items")

    accuracy = correct / total
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_item_results": per_item_results,
    }
