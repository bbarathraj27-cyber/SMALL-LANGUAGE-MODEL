"""Phase 4: Deduplication.

Two complementary techniques, mirroring what large-scale pretraining
corpora use (CCNet, GPT-3 paper, RefinedWeb):

1. Exact dedup: hash the normalized text and drop repeats. Cheap and
   catches identical documents (e.g. the same article scraped twice).

2. Near-duplicate dedup via MinHash + LSH banding: catches documents
   that are almost-but-not-quite identical (e.g. the same article with
   a different ad banner, or boilerplate-heavy pages that differ only
   in a sidebar). Computing exact pairwise similarity across a large
   corpus is O(n^2) and infeasible; MinHash approximates Jaccard
   similarity of shingle sets with fixed-size signatures, and LSH
   banding buckets similar documents together so we only need to
   compare candidates within the same bucket rather than every pair.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_hashing(text: str) -> str:
    """Lowercases and collapses whitespace so trivial formatting
    differences (extra spaces, capitalization) don't defeat exact dedup.
    """
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def _exact_hash(text: str) -> str:
    normalized = _normalize_for_hashing(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _word_shingles(text: str, shingle_size: int) -> set[str]:
    """Builds a set of overlapping word n-grams ("shingles") from text.

    Word-level shingles (rather than character-level) are used here
    because they are more robust to minor character-level edits while
    still sensitive to genuine content overlap, and are cheaper to
    compute over long documents.
    """
    words = _normalize_for_hashing(text).split(" ")
    words = [w for w in words if w]
    if len(words) < shingle_size:
        return {" ".join(words)} if words else set()
    return {
        " ".join(words[i : i + shingle_size])
        for i in range(len(words) - shingle_size + 1)
    }


def _hash_shingle(shingle: str, seed: int) -> int:
    """A seeded 64-bit hash of a shingle, used as one MinHash permutation."""
    h = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8, salt=seed.to_bytes(2, "little", signed=False) if seed < 65536 else b"\x00\x00")
    return int.from_bytes(h.digest(), "little")


def _minhash_signature(shingles: set[str], num_perm: int) -> tuple[int, ...]:
    """Computes a MinHash signature: for each of `num_perm` seeded hash
    functions, the minimum hash value over all shingles. Two documents
    with high Jaccard similarity between their shingle sets will, in
    expectation, agree on a large fraction of signature positions.
    """
    if not shingles:
        return tuple([0] * num_perm)
    signature = []
    for seed in range(num_perm):
        min_val = min(_hash_shingle(s, seed) for s in shingles)
        signature.append(min_val)
    return tuple(signature)


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


class Deduplicator:
    """Exact and near-duplicate document deduplication."""

    def exact_dedup(
        self, records: list[dict[str, Any]], text_field: str = "text"
    ) -> list[dict[str, Any]]:
        """Removes documents that are exact duplicates (after whitespace/case
        normalization) of an earlier document in the list.

        Args:
            records: List of dicts, each containing `text_field`.
            text_field: Key holding the text to compare.

        Returns:
            New list containing only the first occurrence of each
            unique normalized text. Order is preserved.

        Raises:
            KeyError: If a record is missing `text_field`.
        """
        seen_hashes: set[str] = set()
        kept: list[dict[str, Any]] = []
        for record in records:
            if text_field not in record:
                raise KeyError(f"Record missing required field '{text_field}': {record}")
            h = _exact_hash(record[text_field])
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            kept.append(record)
        return kept

    def near_dedup(
        self,
        records: list[dict[str, Any]],
        text_field: str = "text",
        shingle_size: int = 5,
        num_perm: int = 64,
        num_bands: int = 16,
        similarity_threshold: float = 0.8,
    ) -> list[dict[str, Any]]:
        """Removes near-duplicate documents using MinHash + LSH banding.

        Args:
            records: List of dicts, each containing `text_field`.
            text_field: Key holding the text to compare.
            shingle_size: Number of consecutive words per shingle.
            num_perm: MinHash signature length (number of hash
                permutations). Higher values give a more accurate
                similarity estimate at the cost of more compute.
            num_bands: Number of LSH bands the signature is split into.
                Must evenly divide num_perm. More bands increase recall
                (more candidate pairs found) at the cost of more false
                positives passed to the exact similarity check.
            similarity_threshold: Minimum Jaccard similarity (estimated
                via shared LSH bucket + verified exactly on shingle
                sets) for two documents to be considered duplicates.

        Returns:
            New list with near-duplicates removed, keeping the first
            occurrence of each near-duplicate cluster. Order is
            preserved among kept records.

        Raises:
            KeyError: If a record is missing `text_field`.
            ValueError: If num_bands does not evenly divide num_perm,
                or if similarity_threshold is not in [0, 1].
        """
        if num_perm % num_bands != 0:
            raise ValueError(
                f"num_bands ({num_bands}) must evenly divide num_perm ({num_perm})"
            )
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold must be in [0, 1], got {similarity_threshold}"
            )

        rows_per_band = num_perm // num_bands

        shingle_sets: list[set[str]] = []
        signatures: list[tuple[int, ...]] = []
        for record in records:
            if text_field not in record:
                raise KeyError(f"Record missing required field '{text_field}': {record}")
            shingles = _word_shingles(record[text_field], shingle_size)
            shingle_sets.append(shingles)
            signatures.append(_minhash_signature(shingles, num_perm))

        # LSH banding: bucket documents by (band_index, band_hash). Two
        # documents landing in the same bucket for ANY band are treated
        # as similarity candidates, then verified with exact Jaccard.
        buckets: dict[tuple[int, int], list[int]] = {}
        for doc_idx, signature in enumerate(signatures):
            for band_idx in range(num_bands):
                start = band_idx * rows_per_band
                band = signature[start : start + rows_per_band]
                band_key = (band_idx, hash(band))
                buckets.setdefault(band_key, []).append(doc_idx)

        is_duplicate = [False] * len(records)
        # Union-Find would be more efficient for large corpora; for this
        # project's scale, a straightforward candidate-pair verification
        # pass keeps the logic easy to audit and test.
        for bucket_doc_indices in buckets.values():
            if len(bucket_doc_indices) < 2:
                continue
            for i in range(len(bucket_doc_indices)):
                doc_i = bucket_doc_indices[i]
                if is_duplicate[doc_i]:
                    continue
                for j in range(i + 1, len(bucket_doc_indices)):
                    doc_j = bucket_doc_indices[j]
                    if is_duplicate[doc_j]:
                        continue
                    similarity = _jaccard_similarity(
                        shingle_sets[doc_i], shingle_sets[doc_j]
                    )
                    if similarity >= similarity_threshold:
                        # Keep the earlier-indexed document, drop the later one.
                        is_duplicate[doc_j] = True

        return [record for record, dup in zip(records, is_duplicate) if not dup]
