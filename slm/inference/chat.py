"""
chat.py

Minimal interactive CLI for chatting with an SFT'd checkpoint.

Formats turns using the same plain "Role: text" transcript template
sft/prepare_data.py trains on, so behavior at inference time matches
what the model saw during fine-tuning — a template mismatch between
SFT and inference is the single most common cause of bad chat output.
"""

import argparse
import torch

from .generate import GenerationConfig, generate

PROMPT_TEMPLATE = "User: {user}\nAssistant:"
TURN_SEPARATOR = "\n"


def build_prompt(history: list, user_message: str) -> str:
    """
    history: list of (role, text) tuples for prior turns, role in
    {"user", "assistant"}. Returns the full text prompt to tokenize,
    ending right after "Assistant:" so generation continues the
    assistant's turn.
    """
    parts = []
    for role, text in history:
        label = "User" if role == "user" else "Assistant"
        parts.append(f"{label}: {text}")
    parts.append(f"User: {user_message}")
    parts.append("Assistant:")
    return TURN_SEPARATOR.join(parts)


@torch.no_grad()
def chat_turn(
    model,
    tokenizer,
    history: list,
    user_message: str,
    device: str = "cpu",
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> str:
    """Run one chat turn and return the assistant's decoded reply (stripped)."""
    prompt_text = build_prompt(history, user_message)
    ids = tokenizer.encode(prompt_text)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    output_ids = generate(model, input_ids, config)
    new_ids = output_ids[0, input_ids.shape[1]:].tolist()

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and eos_id in new_ids:
        new_ids = new_ids[: new_ids.index(eos_id)]

    return tokenizer.decode(new_ids).strip()


def run_cli(model, tokenizer, device: str = "cpu"):  # pragma: no cover - interactive loop
    """Interactive REPL. Not covered by tests (requires stdin)."""
    history = []
    print("SLM chat — type 'exit' to quit, 'reset' to clear history.")
    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_message.lower() == "exit":
            break
        if user_message.lower() == "reset":
            history = []
            print("(history cleared)")
            continue
        reply = chat_turn(model, tokenizer, history, user_message, device=device)
        print(f"Assistant: {reply}")
        history.append(("user", user_message))
        history.append(("assistant", reply))


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Chat with an SLM checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    raise SystemExit(
        "Wire this up to your checkpoint loader (see training/checkpoint.py), "
        "load a tokenizer, then call run_cli(model, tokenizer, device=args.device)."
    )
