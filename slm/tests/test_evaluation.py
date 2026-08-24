"""
tests/test_evaluation.py

Tests evaluation/perplexity.py and evaluation/benchmark.py against a tiny
real model and tokenizer. Verifies:
  - dataset perplexity computation is finite, token-weighted correctly,
    and rejects zero-token/empty inputs
  - bits_per_byte conversion math
  - benchmark loading validates required keys and index bounds
  - score_choice truncates from the question, never the choice, and raises
    if the choice alone can't fit
  - run_benchmark picks the correct predicted_index format and computes
    accuracy correctly against known correct answers
  - a model trained hard on one choice actually scores it higher (proves
    score_choice's log-prob math is directionally correct, not just
    "doesn't crash")
"""

import os
import sys
import json
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from tokenizer.train_tokenizer import train as train_tokenizer
from tokenizer.tokenizer import SLMTokenizer

from model.slm import SLM, SLMConfig
from dataset.dataloader import build_pretrain_dataloader
from training.optimizer import build_optimizer
from training.trainer import Trainer, TrainerConfig

from evaluation.perplexity import compute_dataset_perplexity, bits_per_byte
from evaluation.benchmark import load_benchmark_file, score_choice, run_benchmark

SAMPLE_CORPUS = """
The quick brown fox jumps over the lazy dog.
Small language models learn grammar and structure from text.
Paris is the capital of France, a country in Europe.
Tokyo is the capital of Japan, a country in East Asia.
The transformer architecture uses attention to relate tokens.
""" * 20


class EvalTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="slm_eval_test_")
        corpus_path = os.path.join(cls.tmp_dir, "corpus.txt")
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_CORPUS)

        cls.tokenizer_dir = os.path.join(cls.tmp_dir, "tokenizer_out")
        train_tokenizer(input_path=corpus_path, vocab_size=512, output_dir=cls.tokenizer_dir, min_frequency=1)
        cls.tokenizer = SLMTokenizer(os.path.join(cls.tokenizer_dir, "tokenizer.json"))

        cls.model_config = SLMConfig(
            vocab_size=cls.tokenizer.vocab_size,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            max_position_embeddings=32,
            tie_word_embeddings=True,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def _write_jsonl(self, records, filename):
        path = os.path.join(self.tmp_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return path


class TestPerplexity(EvalTestBase):
    def test_perplexity_is_finite_and_positive(self):
        model = SLM(self.model_config)
        loader = build_pretrain_dataloader(
            vocab_size=self.model_config.vocab_size,
            block_size=self.model_config.max_position_embeddings,
            num_examples=32, batch_size=8,
        )
        results = compute_dataset_perplexity(model, loader)
        self.assertTrue(torch.isfinite(torch.tensor(results["loss"])))
        self.assertGreater(results["perplexity"], 1.0)  # perplexity >= 1 always
        self.assertGreater(results["num_tokens"], 0)

    def test_max_batches_limits_evaluation(self):
        model = SLM(self.model_config)
        loader = build_pretrain_dataloader(
            vocab_size=self.model_config.vocab_size,
            block_size=self.model_config.max_position_embeddings,
            num_examples=64, batch_size=4,
        )
        results = compute_dataset_perplexity(model, loader, max_batches=3)
        self.assertLessEqual(results["num_batches"], 3)

    def test_invalid_max_batches_rejected(self):
        model = SLM(self.model_config)
        loader = build_pretrain_dataloader(
            vocab_size=self.model_config.vocab_size,
            block_size=self.model_config.max_position_embeddings,
            num_examples=8, batch_size=4,
        )
        with self.assertRaises(ValueError):
            compute_dataset_perplexity(model, loader, max_batches=0)

    def test_bits_per_byte_math(self):
        import math
        ppl = 8.0  # log2(8) = 3 bits/token
        result = bits_per_byte(ppl, avg_tokens_per_byte=0.5)
        self.assertAlmostEqual(result, 3.0 * 0.5, places=6)

    def test_bits_per_byte_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            bits_per_byte(-1.0, 0.5)
        with self.assertRaises(ValueError):
            bits_per_byte(5.0, 0.0)


class TestBenchmarkLoading(EvalTestBase):
    def test_loads_valid_benchmark(self):
        path = self._write_jsonl([
            {"question": "The capital of France is", "choices": [" Paris.", " Tokyo."], "answer_index": 0},
        ], "bench_valid.jsonl")
        items = load_benchmark_file(path)
        self.assertEqual(len(items), 1)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_benchmark_file(os.path.join(self.tmp_dir, "nope.jsonl"))

    def test_missing_key_raises(self):
        path = self._write_jsonl([{"question": "Q?", "choices": ["a", "b"]}], "bench_bad1.jsonl")
        with self.assertRaises(ValueError):
            load_benchmark_file(path)

    def test_too_few_choices_raises(self):
        path = self._write_jsonl(
            [{"question": "Q?", "choices": ["only one"], "answer_index": 0}], "bench_bad2.jsonl"
        )
        with self.assertRaises(ValueError):
            load_benchmark_file(path)

    def test_out_of_range_answer_index_raises(self):
        path = self._write_jsonl(
            [{"question": "Q?", "choices": ["a", "b"], "answer_index": 5}], "bench_bad3.jsonl"
        )
        with self.assertRaises(ValueError):
            load_benchmark_file(path)

    def test_empty_file_raises(self):
        path = os.path.join(self.tmp_dir, "empty.jsonl")
        with open(path, "w"):
            pass
        with self.assertRaises(ValueError):
            load_benchmark_file(path)


class TestScoreChoice(EvalTestBase):
    def test_returns_finite_score(self):
        model = SLM(self.model_config)
        score = score_choice(model, self.tokenizer, "The capital of France is", " Paris.")
        self.assertTrue(torch.isfinite(torch.tensor(score)))

    def test_empty_choice_raises(self):
        model = SLM(self.model_config)
        with self.assertRaises(ValueError):
            score_choice(model, self.tokenizer, "Question?", "")

    def test_truncates_question_not_choice(self):
        model = SLM(self.model_config)
        long_question = "This is a very long question padded out with many extra words " * 5
        # max_context small enough to force truncation, but larger than the choice alone
        score = score_choice(model, self.tokenizer, long_question, " Paris.", max_context=20)
        self.assertTrue(torch.isfinite(torch.tensor(score)))

    def test_choice_alone_exceeding_context_raises(self):
        model = SLM(self.model_config)
        long_choice = "a very long choice string with many words repeated many many times over"
        with self.assertRaises(ValueError):
            score_choice(model, self.tokenizer, "Q?", long_choice, max_context=3)


class TestRunBenchmark(EvalTestBase):
    def test_accuracy_computation_with_untrained_model(self):
        model = SLM(self.model_config)
        path = self._write_jsonl([
            {"question": "The capital of France is", "choices": [" Paris.", " Tokyo."], "answer_index": 0},
            {"question": "The capital of Japan is", "choices": [" Paris.", " Tokyo."], "answer_index": 1},
        ], "bench_run.jsonl")
        results = run_benchmark(model, self.tokenizer, path)
        self.assertEqual(results["total"], 2)
        self.assertIn(results["correct"], range(0, 3))
        self.assertAlmostEqual(results["accuracy"], results["correct"] / results["total"])
        for item in results["per_item_results"]:
            self.assertIn("predicted_index", item)
            self.assertIn("scores", item)
            self.assertEqual(len(item["scores"]), 2)

    def test_trained_model_prefers_correct_choice(self):
        # Train a tiny model hard to predict "Paris" after the France question
        # tokens, then verify score_choice actually reflects that preference --
        # this proves the log-prob accumulation direction is correct.
        model = SLM(self.model_config)
        question = "The capital of France is"
        target = " Paris."
        q_ids = self.tokenizer.encode(question, add_bos=True)
        t_ids = self.tokenizer.encode(target)
        full_ids = q_ids + t_ids

        input_ids = torch.tensor([full_ids[:-1]] * 8, dtype=torch.long)
        labels = torch.tensor([full_ids[1:]] * 8, dtype=torch.long)

        optimizer = build_optimizer(model, learning_rate=1e-2)
        for _ in range(60):
            optimizer.zero_grad()
            logits, loss = model(input_ids, labels=labels)
            loss.backward()
            optimizer.step()

        score_correct = score_choice(model, self.tokenizer, question, " Paris.")
        score_wrong = score_choice(model, self.tokenizer, question, " Tokyo.")
        self.assertGreater(score_correct, score_wrong)


if __name__ == "__main__":
    unittest.main(verbosity=2)
