"""Unit tests for the preprocessing/ package.

Run with: python tests/test_preprocessing.py
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from preprocessing.collect import DataCollector, write_jsonl, read_jsonl
from preprocessing.clean import TextCleaner
from preprocessing.filter import QualityFilter
from preprocessing.deduplicate import Deduplicator
from preprocessing.split import split_records, write_splits
from preprocessing.tokenize import CorpusTokenizer, load_tokenizer
from preprocessing.shard import write_shards, load_manifest


def _make_tmp_dir() -> str:
    return tempfile.mkdtemp(prefix="slm_preprocess_test_")


def test_write_and_read_jsonl_roundtrip():
    tmp_dir = _make_tmp_dir()
    try:
        records = [{"id": "a", "text": "hello world", "source": "x"}]
        path = os.path.join(tmp_dir, "out.jsonl")
        write_jsonl(records, path)
        loaded = list(read_jsonl(path))
        assert loaded == records
        print("test_write_and_read_jsonl_roundtrip: PASSED")
    finally:
        shutil.rmtree(tmp_dir)


def test_write_jsonl_rejects_empty():
    raised = False
    try:
        write_jsonl([], "/tmp/should_not_be_created.jsonl")
    except ValueError:
        raised = True
    assert raised
    assert not os.path.exists("/tmp/should_not_be_created.jsonl")
    print("test_write_jsonl_rejects_empty: PASSED")


def test_collect_from_directory():
    tmp_dir = _make_tmp_dir()
    try:
        with open(os.path.join(tmp_dir, "doc1.txt"), "w") as f:
            f.write("This is document one.")
        with open(os.path.join(tmp_dir, "doc2.txt"), "w") as f:
            f.write("This is document two.")
        with open(os.path.join(tmp_dir, "empty.txt"), "w") as f:
            f.write("   ")  # whitespace only, should be skipped

        collector = DataCollector()
        records = collector.collect_from_directory(tmp_dir, pattern="*.txt")
        assert len(records) == 2, f"Expected 2 records, got {len(records)}"
        for r in records:
            assert "id" in r and "text" in r and "source" in r
        print("test_collect_from_directory: PASSED")
    finally:
        shutil.rmtree(tmp_dir)


def test_collect_from_jsonl():
    tmp_dir = _make_tmp_dir()
    try:
        raw_path = os.path.join(tmp_dir, "raw.jsonl")
        write_jsonl([{"body": "Some article text here."}, {"body": ""}], raw_path)

        collector = DataCollector()
        records = collector.collect_from_jsonl(raw_path, text_field="body")
        assert len(records) == 1  # empty body dropped
        print("test_collect_from_jsonl: PASSED")
    finally:
        shutil.rmtree(tmp_dir)


def test_collect_from_jsonl_missing_field_raises():
    tmp_dir = _make_tmp_dir()
    try:
        raw_path = os.path.join(tmp_dir, "raw.jsonl")
        write_jsonl([{"wrong_field": "text"}], raw_path)
        collector = DataCollector()
        raised = False
        try:
            collector.collect_from_jsonl(raw_path, text_field="body")
        except KeyError:
            raised = True
        assert raised
        print("test_collect_from_jsonl_missing_field_raises: PASSED")
    finally:
        shutil.rmtree(tmp_dir)


def test_collect_from_csv():
    tmp_dir = _make_tmp_dir()
    try:
        csv_path = os.path.join(tmp_dir, "data.csv")
        with open(csv_path, "w") as f:
            f.write("title,body\n")
            f.write("T1,First article body here.\n")
            f.write("T2,Second article body here.\n")

        collector = DataCollector()
        records = collector.collect_from_csv(csv_path, text_column="body")
        assert len(records) == 2
        print("test_collect_from_csv: PASSED")
    finally:
        shutil.rmtree(tmp_dir)


def test_clean_strips_html_and_collapses_whitespace():
    cleaner = TextCleaner()
    dirty = "<p>Hello   world</p>\n\n\n\n<b>Bold</b>   text"
    cleaned = cleaner.clean(dirty)
    assert "<p>" not in cleaned and "<b>" not in cleaned
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned
    print(f"test_clean_strips_html_and_collapses_whitespace: PASSED ({cleaned!r})")


def test_clean_normalizes_unicode():
    cleaner = TextCleaner()
    # Full-width 'Ａ' (U+FF21) should normalize to ASCII 'A' under NFKC.
    result = cleaner.clean("Ａ test")
    assert result.startswith("A"), f"Expected NFKC-normalized 'A', got {result!r}"
    print("test_clean_normalizes_unicode: PASSED")


def test_clean_removes_control_chars_but_keeps_newlines():
    cleaner = TextCleaner(collapse_whitespace=False)
    text = "line one\x00\x01\nline two"
    cleaned = cleaner.clean(text)
    assert "\x00" not in cleaned and "\x01" not in cleaned
    assert "\n" in cleaned
    print("test_clean_removes_control_chars_but_keeps_newlines: PASSED")


def test_clean_batch_drops_empty_results():
    cleaner = TextCleaner()
    records = [
        {"id": "1", "text": "Real content here."},
        {"id": "2", "text": "<p></p>"},  # becomes empty after HTML strip
    ]
    cleaned = cleaner.clean_batch(records)
    assert len(cleaned) == 1
    assert cleaned[0]["id"] == "1"
    print("test_clean_batch_drops_empty_results: PASSED")


def test_filter_rejects_too_short():
    f = QualityFilter(min_chars=100, min_words=1)
    result = f.evaluate("short")
    assert not result.passed
    assert any("too_short" in r for r in result.reasons)
    print("test_filter_rejects_too_short: PASSED")


def test_filter_rejects_symbol_heavy_text():
    f = QualityFilter(min_chars=1, min_words=1, max_symbol_ratio=0.2)
    result = f.evaluate("!!!@@@###$$$%%%^^^&&&***(((")
    assert not result.passed
    assert any("too_many_symbols" in r for r in result.reasons)
    print("test_filter_rejects_symbol_heavy_text: PASSED")


def test_filter_rejects_repetitive_text():
    f = QualityFilter(min_chars=1, min_words=1, max_repeated_line_ratio=0.3)
    text = "\n".join(["Click here to subscribe"] * 10 + ["Unique closing line."])
    result = f.evaluate(text)
    assert not result.passed
    assert any("too_repetitive" in r for r in result.reasons)
    print("test_filter_rejects_repetitive_text: PASSED")


def test_filter_accepts_good_text():
    f = QualityFilter()
    text = (
        "The quick brown fox jumps over the lazy dog. This is a perfectly "
        "reasonable paragraph of natural English prose with no obvious issues "
        "at all, and it should pass every configured quality check cleanly."
    )
    result = f.evaluate(text)
    assert result.passed, f"Expected pass, got reasons: {result.reasons}"
    print("test_filter_accepts_good_text: PASSED")


def test_filter_batch_splits_correctly():
    f = QualityFilter(min_chars=20, min_words=3)
    records = [
        {"id": "good", "text": "This is a long enough sentence to pass the filter easily."},
        {"id": "bad", "text": "no"},
    ]
    kept, rejected = f.filter_batch(records)
    assert len(kept) == 1 and kept[0]["id"] == "good"
    assert len(rejected) == 1 and rejected[0]["id"] == "bad"
    assert "_reject_reasons" in rejected[0]
    print("test_filter_batch_splits_correctly: PASSED")


def test_exact_dedup_removes_identical_documents():
    dedup = Deduplicator()
    records = [
        {"id": "1", "text": "The quick brown fox."},
        {"id": "2", "text": "The quick brown fox."},  # exact duplicate
        {"id": "3", "text": "  THE   quick BROWN fox.  "},  # dup after normalization
        {"id": "4", "text": "A completely different sentence."},
    ]
    result = dedup.exact_dedup(records)
    assert len(result) == 2, f"Expected 2 unique docs, got {len(result)}: {result}"
    ids = {r["id"] for r in result}
    assert ids == {"1", "4"}
    print("test_exact_dedup_removes_identical_documents: PASSED")


def test_near_dedup_removes_similar_documents():
    dedup = Deduplicator()
    base = (
        "artificial intelligence is transforming how software engineers "
        "build products and iterate quickly on new ideas every single day"
    )
    near_duplicate = base + " with additional tooling"  # one word changed at the end
    unrelated = (
        "the recipe calls for two cups of flour a pinch of salt and enough "
        "warm water to bring the dough together into a smooth elastic ball"
    )
    records = [
        {"id": "1", "text": base},
        {"id": "2", "text": near_duplicate},
        {"id": "3", "text": unrelated},
    ]
    result = dedup.near_dedup(
        records, shingle_size=4, num_perm=32, num_bands=16, similarity_threshold=0.5
    )
    ids = {r["id"] for r in result}
    assert "1" in ids, "First occurrence of near-duplicate pair should be kept"
    assert "2" not in ids, f"Near-duplicate should have been removed, kept ids: {ids}"
    assert "3" in ids, "Unrelated document should always be kept"
    print(f"test_near_dedup_removes_similar_documents: PASSED (kept={ids})")


def test_near_dedup_rejects_bad_band_config():
    dedup = Deduplicator()
    records = [{"id": "1", "text": "some text here"}]
    raised = False
    try:
        dedup.near_dedup(records, num_perm=64, num_bands=10)  # 64 % 10 != 0
    except ValueError:
        raised = True
    assert raised
    print("test_near_dedup_rejects_bad_band_config: PASSED")


def test_split_records_ratios_and_determinism():
    records = [{"id": str(i), "text": f"doc {i}"} for i in range(100)]
    splits_a = split_records(records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=7)
    splits_b = split_records(records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=7)

    assert len(splits_a["train"]) == 80
    assert len(splits_a["validation"]) == 10
    assert len(splits_a["test"]) == 10

    # Determinism: same seed -> identical split assignment.
    assert [r["id"] for r in splits_a["train"]] == [r["id"] for r in splits_b["train"]]

    # No overlap between splits.
    all_ids = (
        {r["id"] for r in splits_a["train"]}
        | {r["id"] for r in splits_a["validation"]}
        | {r["id"] for r in splits_a["test"]}
    )
    assert len(all_ids) == 100
    print("test_split_records_ratios_and_determinism: PASSED")


def test_split_records_rejects_bad_ratios():
    records = [{"id": "1", "text": "x"}]
    raised = False
    try:
        split_records(records, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)
    except ValueError:
        raised = True
    assert raised
    print("test_split_records_rejects_bad_ratios: PASSED")


def test_write_splits_creates_files():
    tmp_dir = _make_tmp_dir()
    try:
        records = [{"id": str(i), "text": f"doc {i}"} for i in range(10)]
        splits = split_records(records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=1)
        paths = write_splits(splits, tmp_dir)
        assert "train" in paths
        assert os.path.isfile(paths["train"])
        loaded_train = list(read_jsonl(paths["train"]))
        assert len(loaded_train) == 8
        print("test_write_splits_creates_files: PASSED")
    finally:
        shutil.rmtree(tmp_dir)


def _train_tiny_tokenizer(tmp_dir: str) -> str:
    """Trains a throwaway BPE tokenizer for testing preprocessing/tokenize.py,
    simulating the output of tokenizer/train_tokenizer.py (a separate,
    not-yet-built module) so this module can be tested independently.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=300,
        special_tokens=["<unk>", "<pad>", "<|endoftext|>"],
    )

    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "a small language model learns from text data",
        "transformers use attention to process sequences",
        "training requires tokenized data split into shards",
    ] * 20

    tokenizer.train_from_iterator(corpus, trainer=trainer)

    tokenizer_path = os.path.join(tmp_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    return tokenizer_path


def test_load_tokenizer_and_corpus_tokenizer_eos_resolution():
    tmp_dir = _make_tmp_dir()
    try:
        tokenizer_path = _train_tiny_tokenizer(tmp_dir)
        tok = load_tokenizer(tokenizer_path)
        assert tok.get_vocab_size() > 0

        corpus_tokenizer = CorpusTokenizer(tokenizer_path, eos_token="<|endoftext|>")
        assert corpus_tokenizer.eos_id is not None
        print("test_load_tokenizer_and_corpus_tokenizer_eos_resolution: PASSED")
    finally:
        shutil.rmtree(tmp_dir)


def test_corpus_tokenizer_rejects_missing_eos_token():
    tmp_dir = _make_tmp_dir()
    try:
        tokenizer_path = _train_tiny_tokenizer(tmp_dir)
        raised = False
        try:
            CorpusTokenizer(tokenizer_path, eos_token="<not_in_vocab>")
        except ValueError:
            raised = True
        assert raised
        print("test_corpus_tokenizer_rejects_missing_eos_token: PASSED")
    finally:
        shutil.rmtree(tmp_dir)


def test_tokenize_to_flat_array_inserts_eos_between_docs():
    tmp_dir = _make_tmp_dir()
    try:
        tokenizer_path = _train_tiny_tokenizer(tmp_dir)
        corpus_tokenizer = CorpusTokenizer(tokenizer_path)

        records = [
            {"text": "the quick brown fox"},
            {"text": "a small language model"},
        ]
        flat = corpus_tokenizer.tokenize_to_flat_array(records)
        assert flat.ndim == 1
        assert flat.dtype == np.uint16  # vocab_size=300 fits in uint16
        # EOS id should appear at least twice (once per document).
        eos_count = int(np.sum(flat == corpus_tokenizer.eos_id))
        assert eos_count == 2, f"Expected 2 EOS tokens, found {eos_count}"
        assert corpus_tokenizer.num_documents_processed == 2
        assert corpus_tokenizer.num_documents_skipped == 0
        print("test_tokenize_to_flat_array_inserts_eos_between_docs: PASSED")
    finally:
        shutil.rmtree(tmp_dir)


def test_tokenize_skips_empty_documents():
    tmp_dir = _make_tmp_dir()
    try:
        tokenizer_path = _train_tiny_tokenizer(tmp_dir)
        corpus_tokenizer = CorpusTokenizer(tokenizer_path)
        records = [{"text": "the quick brown fox"}, {"text": "   "}, {"text": ""}]
        flat = corpus_tokenizer.tokenize_to_flat_array(records)
        assert corpus_tokenizer.num_documents_processed == 1
        assert corpus_tokenizer.num_documents_skipped == 2
        print("test_tokenize_skips_empty_documents: PASSED")
    finally:
        shutil.rmtree(tmp_dir)


def test_write_shards_and_load_manifest_roundtrip():
    tmp_dir = _make_tmp_dir()
    try:
        token_array = np.arange(2500, dtype=np.uint16)
        manifest = write_shards(token_array, tmp_dir, shard_size=1000, prefix="shard")

        assert manifest["total_tokens"] == 2500
        assert manifest["num_shards"] == 3  # 1000, 1000, 500
        assert manifest["dtype"] == "uint16"

        loaded_manifest = load_manifest(tmp_dir)
        assert loaded_manifest == manifest

        # Verify actual file contents match the source array.
        reconstructed = np.concatenate(
            [
                np.fromfile(
                    os.path.join(tmp_dir, entry["file"]), dtype=np.uint16
                )
                for entry in loaded_manifest["shards"]
            ]
        )
        assert np.array_equal(reconstructed, token_array)
        print("test_write_shards_and_load_manifest_roundtrip: PASSED")
    finally:
        shutil.rmtree(tmp_dir)


def test_load_manifest_rejects_missing_shard_file():
    tmp_dir = _make_tmp_dir()
    try:
        manifest = {
            "dtype": "uint16",
            "total_tokens": 10,
            "shard_size": 10,
            "num_shards": 1,
            "shards": [{"file": "does_not_exist.bin", "num_tokens": 10, "start_offset": 0}],
        }
        manifest_path = os.path.join(tmp_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        raised = False
        try:
            load_manifest(manifest_path)
        except ValueError:
            raised = True
        assert raised
        print("test_load_manifest_rejects_missing_shard_file: PASSED")
    finally:
        shutil.rmtree(tmp_dir)


def test_end_to_end_pipeline():
    """Runs collect -> clean -> filter -> dedup -> split -> tokenize -> shard
    over a small synthetic corpus, verifying the full chain produces
    consistent, loadable output.
    """
    tmp_dir = _make_tmp_dir()
    try:
        raw_dir = os.path.join(tmp_dir, "raw")
        os.makedirs(raw_dir)
        sentences = [
            "the quick brown fox jumps over the lazy dog near the river bank",
            "a small language model learns useful patterns from text data",
            "transformers use self attention to process input sequences efficiently",
            "training requires tokenized data that has been split into shards",
            "the quick brown fox jumps over the lazy dog near the river bank",  # exact dup
        ]
        for i, sentence in enumerate(sentences):
            with open(os.path.join(raw_dir, f"doc{i}.txt"), "w") as f:
                f.write(sentence)

        collector = DataCollector()
        records = collector.collect_from_directory(raw_dir, pattern="*.txt")
        assert len(records) == 5

        cleaner = TextCleaner()
        cleaned = cleaner.clean_batch(records)
        assert len(cleaned) == 5

        quality_filter = QualityFilter(min_chars=10, min_words=3)
        kept, rejected = quality_filter.filter_batch(cleaned)
        assert len(kept) == 5

        dedup = Deduplicator()
        deduped = dedup.exact_dedup(kept)
        assert len(deduped) == 4  # one exact duplicate removed

        splits = split_records(deduped, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=3)
        split_dir = os.path.join(tmp_dir, "splits")
        paths = write_splits(splits, split_dir)
        assert "train" in paths

        tokenizer_path = _train_tiny_tokenizer(tmp_dir)
        corpus_tokenizer = CorpusTokenizer(tokenizer_path)
        train_records = list(read_jsonl(paths["train"]))
        flat_array = corpus_tokenizer.tokenize_to_flat_array(train_records)
        assert flat_array.size > 0

        shard_dir = os.path.join(tmp_dir, "shards")
        manifest = write_shards(flat_array, shard_dir, shard_size=100)
        assert manifest["total_tokens"] == flat_array.size

        print("test_end_to_end_pipeline: PASSED")
    finally:
        shutil.rmtree(tmp_dir)


def run_all_tests():
    test_fns = [
        obj
        for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = []
    for fn in test_fns:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures.append((fn.__name__, e))
            print(f"{fn.__name__}: FAILED -- {e}")

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} / {len(test_fns)} tests FAILED")
        for name, e in failures:
            print(f"  - {name}: {e}")
        sys.exit(1)
    else:
        print(f"All {len(test_fns)} tests PASSED")


if __name__ == "__main__":
    run_all_tests()
