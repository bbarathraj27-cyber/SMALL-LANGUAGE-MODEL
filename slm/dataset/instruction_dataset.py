"""Instruction (SFT) dataset.

Reads instruction/response pairs from a JSONL file and tokenizes them
for supervised fine-tuning. Each example follows an Alpaca-style
prompt template. Critically, label positions covering the *prompt*
are set to -100 (PyTorch's cross_entropy ignore_index), so the model
is only trained to predict the response tokens, not to reproduce the
instruction it was given — training on prompt tokens would waste
capacity teaching the model to "predict its own input" rather than to
follow instructions.

Examples are returned at variable length (truncated to max_length but
not padded); padding to a common length within each batch is handled
dynamically by dataset/dataloader.py's collate function for efficiency.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch.utils.data import Dataset

from preprocessing.collect import read_jsonl
from preprocessing.tokenize import load_tokenizer

_DEFAULT_TEMPLATE_WITH_INPUT = (
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
_DEFAULT_TEMPLATE_NO_INPUT = "### Instruction:\n{instruction}\n\n### Response:\n"


class InstructionDataset(Dataset):
    """Tokenized instruction-following dataset for SFT.

    Args:
        jsonl_path: Path to a JSONL file where each line has at least
            "instruction" and "output" fields, and optionally "input".
        tokenizer_path: Path to a trained tokenizer.json file.
        max_length: Maximum total sequence length (prompt + response).
            Examples longer than this are truncated from the end of
            the response (the prompt is never truncated, since a
            truncated instruction could silently change its meaning).
        eos_token: Special token appended after the response, teaching
            the model where to stop generating. Must exist in the
            tokenizer vocabulary.
        pad_token: Special token used for batch padding (applied later,
            in the dataloader's collate function). If not present in
            the tokenizer vocabulary, falls back to `eos_token` with a
            warning flag set on `self.used_fallback_pad`.
        prompt_template_with_input: Format string used when a record
            has a non-empty "input" field. Must contain
            {instruction} and {input} placeholders.
        prompt_template_no_input: Format string used when "input" is
            absent or empty. Must contain an {instruction} placeholder.

    Raises:
        FileNotFoundError: If jsonl_path or tokenizer_path don't exist.
        ValueError: If eos_token is not in the tokenizer vocabulary.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer_path: str,
        max_length: int = 1024,
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<pad>",
        prompt_template_with_input: str = _DEFAULT_TEMPLATE_WITH_INPUT,
        prompt_template_no_input: str = _DEFAULT_TEMPLATE_NO_INPUT,
    ) -> None:
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f"Instruction data file not found: {jsonl_path}")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        if "{instruction}" not in prompt_template_with_input or "{input}" not in prompt_template_with_input:
            raise ValueError(
                "prompt_template_with_input must contain {instruction} and {input} placeholders"
            )
        if "{instruction}" not in prompt_template_no_input:
            raise ValueError("prompt_template_no_input must contain an {instruction} placeholder")

        self.tokenizer = load_tokenizer(tokenizer_path)
        self.max_length = max_length
        self.prompt_template_with_input = prompt_template_with_input
        self.prompt_template_no_input = prompt_template_no_input

        eos_id = self.tokenizer.token_to_id(eos_token)
        if eos_id is None:
            raise ValueError(
                f"eos_token '{eos_token}' was not found in the tokenizer vocabulary."
            )
        self.eos_id = eos_id

        pad_id = self.tokenizer.token_to_id(pad_token)
        self.used_fallback_pad = pad_id is None
        self.pad_token_id = pad_id if pad_id is not None else eos_id

        self.records: list[dict[str, Any]] = []
        for line_number, record in enumerate(read_jsonl(jsonl_path), start=1):
            if "instruction" not in record:
                raise KeyError(
                    f"Record at line {line_number} of {jsonl_path} is missing "
                    f"required field 'instruction'"
                )
            if "output" not in record:
                raise KeyError(
                    f"Record at line {line_number} of {jsonl_path} is missing "
                    f"required field 'output'"
                )
            self.records.append(record)

        if not self.records:
            raise ValueError(f"No records found in {jsonl_path}")

    def __len__(self) -> int:
        return len(self.records)

    def _build_prompt(self, record: dict[str, Any]) -> str:
        instruction = str(record["instruction"]).strip()
        input_text = str(record.get("input", "")).strip()
        if input_text:
            return self.prompt_template_with_input.format(
                instruction=instruction, input=input_text
            )
        return self.prompt_template_no_input.format(instruction=instruction)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Tokenizes and masks one instruction/response example.

        Args:
            index: Example index in [0, len(self)).

        Returns:
            Dict with variable-length (unpadded) tensors:
                "input_ids": LongTensor (seq_len,) — prompt + response + eos.
                "labels": LongTensor (seq_len,) — same length as
                    input_ids, with prompt positions set to -100 and
                    response/eos positions equal to their token id.
                "attention_mask": LongTensor (seq_len,) of all 1s
                    (padding, if any, is added later by the collate
                    function and gets 0s there).

        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= len(self.records):
            raise IndexError(f"Index {index} out of range for dataset of length {len(self.records)}")

        record = self.records[index]
        prompt_text = self._build_prompt(record)
        response_text = str(record["output"]).strip()

        prompt_ids = self.tokenizer.encode(prompt_text).ids
        response_ids = self.tokenizer.encode(response_text).ids + [self.eos_id]

        # Truncate the response first if the combined sequence is too long;
        # never truncate the prompt, since a cut-off instruction can
        # silently change its meaning (e.g. "do NOT include X" truncated
        # to "do").
        available_for_response = self.max_length - len(prompt_ids)
        if available_for_response <= 0:
            # Prompt alone already exceeds max_length: truncate the prompt
            # as a last resort so __getitem__ still returns a valid example,
            # but this indicates max_length is too small for the dataset.
            prompt_ids = prompt_ids[: self.max_length]
            response_ids = []
        elif len(response_ids) > available_for_response:
            response_ids = response_ids[:available_for_response]

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + list(response_ids)

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        attention_mask_tensor = torch.ones(len(input_ids), dtype=torch.long)

        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
            "attention_mask": attention_mask_tensor,
        }
