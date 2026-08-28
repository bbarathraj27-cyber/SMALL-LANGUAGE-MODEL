"""
sft/evaluate_sft.py

Evaluates a fine-tuned checkpoint: computes response-only perplexity on a
held-out prepared SFT set, and generates sample completions for a handful
of held-out prompts so a human can sanity-check output quality.

Usage:
    python sft/evaluate_sft.py \
        --model-config configs/model.yaml \
        --checkpoint checkpoints/sft/checkpoint_step2000.pt \
        --eval-data data/sft/prepared_val.jsonl \
        --tokenizer tokenizer/tokenizer.json \
        --num-samples 5
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.slm import SLM, SLMConfig
from tokenizer.tokenizer import SLMTokenizer, BOS_ID, EOS_ID
from training.loss import compute_lm_loss, compute_perplexity
from training.checkpoint import load_checkpoint
from sft.train_sft import PreparedInstructionDataset, sft_collate_fn

logger = logging.getLogger("sft.evaluate_sft")

IGNORE_INDEX = -100


@torch.no_grad()
def evaluate_perplexity(
    model: SLM,
    eval_data_path: str,
    batch_size: int = 8,
    device: str = "cpu",
) -> Dict[str, float]:
    dataset = PreparedInstructionDataset(eval_data_path)
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: sft_collate_fn(b, pad_id=0),
    )

    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        # model/slm.py's SLM.forward returns a dict with keys "logits",
        # "loss", "past_key_values" -- not a tuple.
        output = model(input_ids, labels=labels)
        logits = output["logits"]
        model_loss = output["loss"]
        loss = model_loss if model_loss is not None else compute_lm_loss(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite eval loss encountered: {loss.item()}")
        total_loss += loss.item()
        total_batches += 1

    if total_batches == 0:
        raise ValueError(f"Eval dataset at {eval_data_path} produced zero batches")

    mean_loss = total_loss / total_batches
    perplexity = compute_perplexity(torch.tensor(mean_loss))
    return {"eval_loss": mean_loss, "eval_perplexity": perplexity, "num_batches": total_batches}


@torch.no_grad()
def generate_greedy(
    model: SLM,
    tokenizer: SLMTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    device: str = "cpu",
) -> str:
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")

    model.eval()
    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    generated = list(input_ids)

    max_context = model.config.max_position_embeddings

    for _ in range(max_new_tokens):
        context = generated[-max_context:]
        input_tensor = torch.tensor([context], dtype=torch.long, device=device)
        # model/slm.py's SLM.forward returns a dict, not a tuple.
        output = model(input_tensor)
        logits = output["logits"]
        next_token_logits = logits[0, -1, :]
        next_token = int(torch.argmax(next_token_logits).item())
        generated.append(next_token)
        if next_token == EOS_ID:
            break

    response_ids = generated[len(input_ids):]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


def collect_sample_prompts(eval_data_path: str, tokenizer: SLMTokenizer, num_samples: int) -> List[str]:
    """Recovers prompt text from a small number of prepared examples, by
    decoding the input_ids positions where labels == IGNORE_INDEX (the
    prompt-masked region)."""
    if not os.path.isfile(eval_data_path):
        raise FileNotFoundError(f"Eval data not found: {eval_data_path}")

    prompts = []
    with open(eval_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= num_samples:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt_ids = [
                tok for tok, lab in zip(record["input_ids"], record["labels"])
                if lab == IGNORE_INDEX
            ]
            if not prompt_ids:
                continue
            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            if prompt_text.strip():
                prompts.append(prompt_text)

    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned SFT checkpoint.")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    try:
        model_config = SLMConfig.from_yaml(args.model_config)
        model = SLM(model_config)
        load_checkpoint(args.checkpoint, model)
        tokenizer = SLMTokenizer(args.tokenizer)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Setup error: {e}")
        sys.exit(1)

    try:
        metrics = evaluate_perplexity(model, args.eval_data, batch_size=args.batch_size)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Evaluation error: {e}")
        sys.exit(1)

    print(f"Eval loss: {metrics['eval_loss']:.4f}")
    print(f"Eval perplexity: {metrics['eval_perplexity']:.2f}")
    print(f"Batches evaluated: {metrics['num_batches']}")

    if args.num_samples > 0:
        prompts = collect_sample_prompts(args.eval_data, tokenizer, args.num_samples)
        print(f"\n--- Sample generations ({len(prompts)}) ---")
        for i, prompt in enumerate(prompts, start=1):
            response = generate_greedy(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
            print(f"\n[{i}] Prompt: {prompt}")
            print(f"    Response: {response}")


if __name__ == "__main__":
    main()
