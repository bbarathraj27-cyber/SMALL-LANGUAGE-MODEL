"""Quality filtering.

Rejects low-quality documents that survive cleaning but shouldn't make
it into the training corpus: too short/long, mostly symbols, mostly
repeated lines, or too few alphabetic characters to plausibly be
natural-language prose. These are the same broad heuristic categories
used in large-scale corpus filtering pipelines (e.g. CCNet, C4, RefinedWeb),
scaled down to something dependency-free and fast enough to run over a
local corpus.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Any

_WORD_RE = re.compile(r"\S+")


@dataclass
class FilterResult:
    """Outcome of running QualityFilter.evaluate on one document."""

    passed: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class QualityFilter:
    """Configurable heuristic quality filter for document text.

    Args:
        min_chars: Minimum character count (after cleaning) to keep.
        max_chars: Maximum character count to keep. None disables the
            upper bound.
        min_words: Minimum whitespace-delimited word count to keep.
        max_symbol_ratio: Maximum allowed fraction of characters that
            are neither alphanumeric nor whitespace. Documents above
            this ratio look like junk/markup rather than prose.
        min_alpha_ratio: Minimum required fraction of non-whitespace
            characters that are alphabetic. Filters out documents that
            are mostly numbers, symbols, or code-like content when
            targeting natural-language pretraining data.
        max_repeated_line_ratio: Maximum allowed fraction of lines
            (among lines with length > 0) that are exact duplicates of
            another line in the same document. High values here are a
            strong signal of scraped boilerplate (nav menus, repeated
            headers/footers).
    """

    min_chars: int = 50
    max_chars: int | None = 200_000
    min_words: int = 10
    max_symbol_ratio: float = 0.3
    min_alpha_ratio: float = 0.5
    max_repeated_line_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.min_chars < 0:
            raise ValueError(f"min_chars must be >= 0, got {self.min_chars}")
        if self.max_chars is not None and self.max_chars < self.min_chars:
            raise ValueError(
                f"max_chars ({self.max_chars}) must be >= min_chars ({self.min_chars})"
            )
        if self.min_words < 0:
            raise ValueError(f"min_words must be >= 0, got {self.min_words}")
        if not (0.0 <= self.max_symbol_ratio <= 1.0):
            raise ValueError(f"max_symbol_ratio must be in [0, 1], got {self.max_symbol_ratio}")
        if not (0.0 <= self.min_alpha_ratio <= 1.0):
            raise ValueError(f"min_alpha_ratio must be in [0, 1], got {self.min_alpha_ratio}")
        if not (0.0 <= self.max_repeated_line_ratio <= 1.0):
            raise ValueError(
                f"max_repeated_line_ratio must be in [0, 1], got {self.max_repeated_line_ratio}"
            )

    def evaluate(self, text: str) -> FilterResult:
        """Runs all configured checks against a document.

        Args:
            text: Document text to evaluate.

        Returns:
            FilterResult with `passed=True` only if every check passes.
            `reasons` lists every check that failed (not just the first),
            which is useful for diagnosing why a corpus is being
            aggressively filtered.

        Raises:
            TypeError: If text is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text)}")

        reasons: list[str] = []

        num_chars = len(text)
        if num_chars < self.min_chars:
            reasons.append(f"too_short ({num_chars} < {self.min_chars} chars)")
        if self.max_chars is not None and num_chars > self.max_chars:
            reasons.append(f"too_long ({num_chars} > {self.max_chars} chars)")

        words = _WORD_RE.findall(text)
        num_words = len(words)
        if num_words < self.min_words:
            reasons.append(f"too_few_words ({num_words} < {self.min_words})")

        non_space_chars = [ch for ch in text if not ch.isspace()]
        if non_space_chars:
            symbol_chars = sum(
                1 for ch in non_space_chars if ch not in string.ascii_letters
                and ch not in string.digits
                and not ch.isalpha()
            )
            symbol_ratio = symbol_chars / len(non_space_chars)
            if symbol_ratio > self.max_symbol_ratio:
                reasons.append(
                    f"too_many_symbols (ratio={symbol_ratio:.2f} > {self.max_symbol_ratio})"
                )

            alpha_chars = sum(1 for ch in non_space_chars if ch.isalpha())
            alpha_ratio = alpha_chars / len(non_space_chars)
            if alpha_ratio < self.min_alpha_ratio:
                reasons.append(
                    f"too_few_alpha_chars (ratio={alpha_ratio:.2f} < {self.min_alpha_ratio})"
                )

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if lines:
            unique_lines = len(set(lines))
            duplicate_ratio = 1.0 - (unique_lines / len(lines))
            if duplicate_ratio > self.max_repeated_line_ratio:
                reasons.append(
                    f"too_repetitive (duplicate_line_ratio={duplicate_ratio:.2f} "
                    f"> {self.max_repeated_line_ratio})"
                )

        return FilterResult(passed=(len(reasons) == 0), reasons=reasons)

    def filter_batch(
        self, records: list[dict[str, Any]], text_field: str = "text"
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Splits a list of records into kept and rejected groups.

        Args:
            records: List of dicts, each containing `text_field`.
            text_field: Key holding the text to evaluate.

        Returns:
            Tuple (kept, rejected). Rejected records are returned with
            an added "_reject_reasons" key for debugging/inspection;
            kept records are returned unmodified.

        Raises:
            KeyError: If a record is missing `text_field`.
        """
        kept: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for record in records:
            if text_field not in record:
                raise KeyError(f"Record missing required field '{text_field}': {record}")
            result = self.evaluate(record[text_field])
            if result.passed:
                kept.append(record)
            else:
                rejected_record = dict(record)
                rejected_record["_reject_reasons"] = result.reasons
                rejected.append(rejected_record)

        return kept, rejected
