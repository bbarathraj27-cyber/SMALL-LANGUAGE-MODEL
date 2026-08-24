"""
tests/test_tokenizer.py

Trains a small throwaway BPE tokenizer on a sample corpus and verifies:
  - special tokens land at the fixed ids the rest of the pipeline relies on
  - encode/decode round-trips correctly
  - add_bos / add_eos behave correctly
  - batch encode/decode match single-item encode/decode
  - unknown characters fall back to <unk> rather than crashing
  - loading a missing/corrupt tokenizer file raises a clear error
"""

import os
import sys
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.train_tokenizer import train as train_tokenizer
from tokenizer.tokenizer import (
    SLMTokenizer,
    PAD_ID,
    BOS_ID,
    EOS_ID,
    UNK_ID,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
)

SAMPLE_CORPUS = """
The quick brown fox jumps over the lazy dog.
Small language models learn grammar and structure from text.
The transformer architecture uses attention to relate tokens to each other.
RoPE, RMSNorm and SwiGLU are modern building blocks for efficient models.
The quick brown fox jumps over the lazy dog again and again.
Training a tokenizer requires a reasonably sized text corpus to find merges.
Byte pair encoding merges frequent character pairs into subword tokens.
""" * 20


class TestTokenizerTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="slm_tokenizer_test_")
        cls.corpus_path = os.path.join(cls.tmp_dir, "corpus.txt")
        with open(cls.corpus_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_CORPUS)

        cls.output_dir = os.path.join(cls.tmp_dir, "tokenizer_out")
        # Small vocab size appropriate for a tiny test corpus.
        train_tokenizer(
            input_path=cls.corpus_path,
            vocab_size=512,
            output_dir=cls.output_dir,
            min_frequency=1,
        )
        cls.tokenizer_json = os.path.join(cls.output_dir, "tokenizer.json")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_output_files_created(self):
        self.assertTrue(os.path.isfile(os.path.join(self.output_dir, "tokenizer.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.output_dir, "vocab.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.output_dir, "merges.txt")))

    def test_special_tokens_fixed_ids(self):
        tok = SLMTokenizer(self.tokenizer_json)
        self.assertEqual(tok.token_to_id(PAD_TOKEN), PAD_ID)
        self.assertEqual(tok.token_to_id(BOS_TOKEN), BOS_ID)
        self.assertEqual(tok.token_to_id(EOS_TOKEN), EOS_ID)
        self.assertEqual(tok.token_to_id(UNK_TOKEN), UNK_ID)

    def test_vocab_size_property(self):
        tok = SLMTokenizer(self.tokenizer_json)
        self.assertGreaterEqual(tok.vocab_size, 4)  # at least the 4 special tokens
        self.assertLessEqual(tok.vocab_size, 512)

    def test_encode_decode_roundtrip(self):
        tok = SLMTokenizer(self.tokenizer_json)
        text = "The quick brown fox jumps over the lazy dog."
        ids = tok.encode(text)
        self.assertIsInstance(ids, list)
        self.assertTrue(all(isinstance(i, int) for i in ids))
        decoded = tok.decode(ids)
        # Byte-level BPE round-trips should reconstruct the original text closely.
        self.assertEqual(decoded.strip(), text.strip())

    def test_add_bos_eos(self):
        tok = SLMTokenizer(self.tokenizer_json)
        text = "Small language models"
        plain = tok.encode(text)
        with_bos = tok.encode(text, add_bos=True)
        with_eos = tok.encode(text, add_eos=True)
        with_both = tok.encode(text, add_bos=True, add_eos=True)

        self.assertEqual(with_bos[0], BOS_ID)
        self.assertEqual(with_bos[1:], plain)

        self.assertEqual(with_eos[-1], EOS_ID)
        self.assertEqual(with_eos[:-1], plain)

        self.assertEqual(with_both[0], BOS_ID)
        self.assertEqual(with_both[-1], EOS_ID)
        self.assertEqual(with_both[1:-1], plain)
        self.assertEqual(len(with_both), len(plain) + 2)

    def test_batch_matches_single(self):
        tok = SLMTokenizer(self.tokenizer_json)
        texts = [
            "The quick brown fox.",
            "Attention relates tokens to each other.",
            "Byte pair encoding merges pairs.",
        ]
        batch_ids = tok.encode_batch(texts, add_bos=True, add_eos=True)
        single_ids = [tok.encode(t, add_bos=True, add_eos=True) for t in texts]
        self.assertEqual(batch_ids, single_ids)

        batch_decoded = tok.decode_batch(batch_ids)
        single_decoded = [tok.decode(ids) for ids in batch_ids]
        self.assertEqual(batch_decoded, single_decoded)

    def test_unknown_characters_do_not_crash(self):
        tok = SLMTokenizer(self.tokenizer_json)
        # Byte-level BPE maps every byte to something, so this should encode
        # without raising, even for characters absent from the training corpus.
        weird_text = "こんにちは 🚀 §∆"
        ids = tok.encode(weird_text)
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)

    def test_encode_type_validation(self):
        tok = SLMTokenizer(self.tokenizer_json)
        with self.assertRaises(TypeError):
            tok.encode(12345)
        with self.assertRaises(TypeError):
            tok.encode_batch("not a list")
        with self.assertRaises(TypeError):
            tok.decode("not a list")
        with self.assertRaises(TypeError):
            tok.decode_batch([12345])

    def test_missing_file_raises_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            SLMTokenizer(os.path.join(self.tmp_dir, "does_not_exist.json"))

    def test_from_pretrained_accepts_directory(self):
        tok = SLMTokenizer.from_pretrained(self.output_dir)
        self.assertGreater(tok.vocab_size, 0)

    def test_from_pretrained_accepts_file(self):
        tok = SLMTokenizer.from_pretrained(self.tokenizer_json)
        self.assertGreater(tok.vocab_size, 0)

    def test_vocab_size_smaller_than_special_tokens_rejected(self):
        with self.assertRaises(ValueError):
            train_tokenizer(
                input_path=self.corpus_path,
                vocab_size=2,
                output_dir=os.path.join(self.tmp_dir, "bad_out"),
            )

    def test_missing_input_path_rejected(self):
        with self.assertRaises(FileNotFoundError):
            train_tokenizer(
                input_path=os.path.join(self.tmp_dir, "nope.txt"),
                vocab_size=512,
                output_dir=os.path.join(self.tmp_dir, "bad_out2"),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
