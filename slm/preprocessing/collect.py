"""Phase 2: Data collection.

Gathers raw text from local files (plain text, JSONL, CSV) into a
unified record format:

    {"id": str, "text": str, "source": str}

This unified schema is what every downstream preprocessing stage
(clean, filter, deduplicate, split, tokenize) consumes and produces,
so each stage can be run independently and re-run without needing to
know where the data originally came from.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
from collections.abc import Iterator
from typing import Any


def _make_id(source: str, index: int, text: str) -> str:
    """Deterministic id from source + index + a content hash, so re-running
    collection on unchanged input files reproduces identical ids (useful
    for idempotent pipelines and for tracing a record back to its origin).
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    safe_source = os.path.basename(source).replace(" ", "_")
    return f"{safe_source}-{index:08d}-{digest}"


def write_jsonl(records: list[dict[str, Any]], output_path: str) -> None:
    """Writes a list of records to a JSONL file, one JSON object per line.

    Args:
        records: List of JSON-serializable dicts.
        output_path: Destination file path. Parent directories are
            created if they don't exist.

    Raises:
        ValueError: If records is empty.
        TypeError: If any record is not JSON-serializable.
    """
    if not records:
        raise ValueError("Cannot write an empty records list to JSONL.")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tmp_path = output_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for record in records:
                try:
                    line = json.dumps(record, ensure_ascii=False)
                except TypeError as e:
                    raise TypeError(
                        f"Record is not JSON-serializable: {record!r}"
                    ) from e
                f.write(line + "\n")
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    """Streams records from a JSONL file one line at a time.

    Args:
        path: Path to a JSONL file.

    Yields:
        Parsed dict for each non-empty line.

    Raises:
        FileNotFoundError: If path does not exist.
        json.JSONDecodeError: If a non-empty line is not valid JSON.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {line_number} of {path}: {e.msg}",
                    e.doc,
                    e.pos,
                )


class DataCollector:
    """Collects raw text from various local sources into unified records."""

    def collect_from_directory(
        self,
        directory: str,
        pattern: str = "*.txt",
        source_name: str | None = None,
        encoding: str = "utf-8",
    ) -> list[dict[str, Any]]:
        """Reads every file matching `pattern` in `directory` as one document.

        Args:
            directory: Directory to search.
            pattern: Glob pattern (e.g. "*.txt", "**/*.md"). Non-recursive
                unless pattern includes "**" and recursive matching is used.
            source_name: Label stored in each record's "source" field.
                Defaults to the directory path.
            encoding: Text encoding to use when reading files. Falls back
                to latin-1 with replacement if utf-8 decoding fails, so a
                handful of malformed files don't abort the whole collection.

        Returns:
            List of {"id", "text", "source"} records, one per file that
            contained non-empty text after stripping.

        Raises:
            NotADirectoryError: If `directory` does not exist.
        """
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Directory not found: {directory}")

        source_label = source_name or directory
        recursive = "**" in pattern
        search_pattern = os.path.join(directory, pattern)
        file_paths = sorted(glob.glob(search_pattern, recursive=recursive))

        records: list[dict[str, Any]] = []
        for index, file_path in enumerate(file_paths):
            if not os.path.isfile(file_path):
                continue
            text = self._read_text_file(file_path, encoding=encoding)
            text = text.strip()
            if not text:
                continue
            records.append(
                {
                    "id": _make_id(source_label, index, text),
                    "text": text,
                    "source": source_label,
                }
            )

        return records

    def collect_from_jsonl(
        self,
        path: str,
        text_field: str = "text",
        source_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Collects documents from an existing JSONL file.

        Args:
            path: Path to a JSONL file where each line has a text field.
            text_field: Key in each JSON object holding the document text.
            source_name: Label stored in each record's "source" field.
                Defaults to the file path.

        Returns:
            List of {"id", "text", "source"} records.

        Raises:
            FileNotFoundError: If path does not exist.
            KeyError: If a record is missing `text_field`.
        """
        source_label = source_name or path
        records: list[dict[str, Any]] = []
        for index, raw_record in enumerate(read_jsonl(path)):
            if text_field not in raw_record:
                raise KeyError(
                    f"Record at line {index + 1} of {path} is missing "
                    f"required field '{text_field}'"
                )
            text = str(raw_record[text_field]).strip()
            if not text:
                continue
            records.append(
                {
                    "id": _make_id(source_label, index, text),
                    "text": text,
                    "source": source_label,
                }
            )
        return records

    def collect_from_csv(
        self,
        path: str,
        text_column: str,
        delimiter: str = ",",
        source_name: str | None = None,
        encoding: str = "utf-8",
    ) -> list[dict[str, Any]]:
        """Collects documents from a CSV/TSV file.

        Args:
            path: Path to the CSV file. Must have a header row.
            text_column: Name of the column containing document text.
            delimiter: Field delimiter (use "\\t" for TSV).
            source_name: Label stored in each record's "source" field.
                Defaults to the file path.
            encoding: File encoding.

        Returns:
            List of {"id", "text", "source"} records.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If text_column is not present in the CSV header.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"CSV file not found: {path}")

        source_label = source_name or path
        records: list[dict[str, Any]] = []
        with open(path, "r", encoding=encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if reader.fieldnames is None or text_column not in reader.fieldnames:
                raise ValueError(
                    f"Column '{text_column}' not found in CSV header: "
                    f"{reader.fieldnames}"
                )
            for index, row in enumerate(reader):
                text = (row.get(text_column) or "").strip()
                if not text:
                    continue
                records.append(
                    {
                        "id": _make_id(source_label, index, text),
                        "text": text,
                        "source": source_label,
                    }
                )
        return records

    @staticmethod
    def _read_text_file(path: str, encoding: str = "utf-8") -> str:
        """Reads a text file, falling back to latin-1 on decode errors."""
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1", errors="replace") as f:
                return f.read()
