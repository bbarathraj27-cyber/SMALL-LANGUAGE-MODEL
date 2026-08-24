"""Phase 3: Data cleaning.

Normalizes and strips noise from raw document text before it reaches
the quality filter and deduplication stages. Cleaning is intentionally
conservative: it removes clear noise (HTML tags, control characters,
excess whitespace) without attempting content judgments like "is this
spam" — that's the job of preprocessing/filter.py.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


@dataclass
class TextCleaner:
    """Configurable text cleaning pipeline.

    Args:
        strip_html: Remove HTML/XML-style tags (a coarse regex strip,
            not a full parser — sufficient for scraped boilerplate).
        normalize_unicode: Apply NFKC unicode normalization, which
            folds visually-identical characters (e.g. full-width vs
            half-width forms) into a consistent representation.
        remove_control_chars: Strip non-printable control characters
            while preserving newlines and tabs.
        collapse_whitespace: Collapse runs of spaces/tabs into one
            space and runs of 3+ blank lines into a single blank line.
        replace_urls_with: If not None, every URL is replaced with this
            string (e.g. "[URL]"). If None, URLs are left untouched.
        min_line_length: Lines shorter than this (after stripping) are
            dropped entirely. Helps remove nav-menu / boilerplate
            fragments common in scraped HTML-derived text. Set to 0 to
            disable.
        strip: Whether to strip leading/trailing whitespace from the
            final result.
    """

    strip_html: bool = True
    normalize_unicode: bool = True
    remove_control_chars: bool = True
    collapse_whitespace: bool = True
    replace_urls_with: str | None = None
    min_line_length: int = 0
    strip: bool = True

    def clean(self, text: str) -> str:
        """Applies the configured cleaning steps in order.

        Args:
            text: Raw input text.

        Returns:
            Cleaned text. May be an empty string if nothing survives
            (e.g. a document that was entirely HTML tags).

        Raises:
            TypeError: If text is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text)}")

        result = text

        if self.strip_html:
            result = _HTML_TAG_RE.sub(" ", result)

        if self.normalize_unicode:
            result = unicodedata.normalize("NFKC", result)

        if self.replace_urls_with is not None:
            result = _URL_RE.sub(self.replace_urls_with, result)

        if self.remove_control_chars:
            result = self._strip_control_chars(result)

        if self.min_line_length > 0:
            lines = result.split("\n")
            lines = [
                line for line in lines if len(line.strip()) >= self.min_line_length or line.strip() == ""
            ]
            result = "\n".join(lines)

        if self.collapse_whitespace:
            result = _MULTI_SPACE_RE.sub(" ", result)
            result = _MULTI_BLANK_LINE_RE.sub("\n\n", result)
            result = "\n".join(line.strip() for line in result.split("\n"))

        if self.strip:
            result = result.strip()

        return result

    @staticmethod
    def _strip_control_chars(text: str) -> str:
        """Removes unicode control characters (category "C*") except
        newline and tab, which are meaningful whitespace we want to keep.
        """
        return "".join(
            ch for ch in text if ch in ("\n", "\t") or unicodedata.category(ch)[0] != "C"
        )

    def clean_batch(
        self, records: list[dict[str, Any]], text_field: str = "text"
    ) -> list[dict[str, Any]]:
        """Cleans the text field of every record in a list.

        Args:
            records: List of dicts, each containing `text_field`.
            text_field: Key holding the text to clean.

        Returns:
            New list of records (originals are not mutated) with the
            text field replaced by its cleaned version. Records whose
            cleaned text is empty are dropped.

        Raises:
            KeyError: If a record is missing `text_field`.
        """
        cleaned_records = []
        for record in records:
            if text_field not in record:
                raise KeyError(f"Record missing required field '{text_field}': {record}")
            cleaned_text = self.clean(record[text_field])
            if not cleaned_text:
                continue
            new_record = dict(record)
            new_record[text_field] = cleaned_text
            cleaned_records.append(new_record)
        return cleaned_records
