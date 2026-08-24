#!/usr/bin/env python3
"""
scripts/run_inference.py

CLI wrapper around inference/generate.py and inference/chat.py
(Phase 18): loads a checkpoint (via optimization/export.py, so the
same integrity check runs here as everywhere else a checkpoint is
loaded) and either answers a single prompt or drops into the
interactive chat REPL.

Usage:
    python scripts/run_inference.py --checkpoint checkpoints/sft/final --prompt "Hello!"
    python scripts/run_inference.py --checkpoint checkpoints/sft/final --chat
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference against a trained checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path (no extension) to the checkpoint to load")
    parser.add_argument("--prompt", default=None, help="Single prompt to generate a reply for")
    parser.add_argument("--chat", action="store_true", help="Start the interactive chat REPL instead")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser


def validate_args(args) -> None:
    if not args.chat and not args.prompt:
        raise ValueError("Provide either --prompt \"...\" for a single reply, or --chat for the REPL")
    if args.chat and args.prompt:
        raise ValueError("--prompt and --chat are mutually exclusive; pick one")


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)

    # Lazy imports: this ties together model/slm.py (architecture),
    # tokenizer/tokenizer.py (encode/decode), and optimization/export.py
    # (checkpoint + manifest loading) with inference/. Adjust the model
    # construction call if your SLM class name/signature differs.
    from optimization.export import load_checkpoint
    from tokenizer.tokenizer import Tokenizer
    from model.slm import SLM
    from inference.generate import GenerationConfig, generate
    from inference.chat import chat_turn, run_cli

    with open(args.checkpoint + ".json") as f:
        import json

        manifest = json.load(f)
    model = SLM(**manifest["config"])
    model, manifest = load_checkpoint(model, args.checkpoint)
    model.to(args.device)
    model.eval()

    tokenizer = Tokenizer.load(manifest.get("tokenizer", {}))

    if args.chat:
        run_cli(model, tokenizer, device=args.device)
        return

    input_ids = torch.tensor([tokenizer.encode(args.prompt)], device=args.device)
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        greedy=args.greedy,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    output_ids = generate(model, input_ids, config)
    reply_ids = output_ids[0, input_ids.shape[1]:].tolist()
    print(tokenizer.decode(reply_ids).strip())


if __name__ == "__main__":
    main()
