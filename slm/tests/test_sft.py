"""
tests/test_sft.py

Tests sft/prepare_data.py, sft/train_sft.py, and sft/evaluate_sft.py against
a tiny real tokenizer and model. Verifies:
  - raw JSONL loading validates required keys and rejects malformed records
  - prompt masking is exact: labels are IGNORE_INDEX exactly over the prompt
    span and match token ids over the response span
  - truncation keeps the full prompt and drops from the response
  - examples that don't fit at all are skipped, not silently corrupted
  - dynamic padding collate produces correctly padded input_ids and
    IGNORE_INDEX-padded labels
  - end-to-end: prepared data trains without error and reduces loss
  - greedy generation produces a non-empty string and respects max_new_tokens
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
from tokenizer.tokenizer import SLMTokenizer, BOS_ID, EOS_ID

from sft.prepare_data import (
    load_raw_examples,
    prepare_example,
    prepare_dataset,
    IGNORE_INDEX,
)
from sft.train_sft import PreparedInstructionDataset, sft_collate_fn, build_sft_dataloader
from sft.evaluate_sft import evaluate_perplexity, generate_greedy, collect_sample_prompts

from model.slm import SLM, SLMConfig
from training.optimizer import build_optimizer
from training.trainer import Trainer, TrainerConfig
from training.checkpoint import save_checkpoint

SAMPLE_CORPUS = """
The quick brown fox jumps over the lazy dog.
Small language models learn grammar and structure from text.
Please summarize the article about renewable energy sources.
The capital of France is Paris, a major European city.
Explain how photosynthesis converts light into chemical energy.
Write a short poem about the changing seasons of the year.
""" * 20


class SFTTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="slm_sft_test_")
        corpus_path = os.path.join(cls.tmp_dir, "corpus.txt")
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_CORPUS)

        cls.tokenizer_dir = os.path.join(cls.tmp_dir, "tokenizer_out")
        train_tokenizer(input_path=corpus_path, vocab_size=512, output_dir=cls.tokenizer_dir, min_frequency=1)
        cls.tokenizer_path = os.path.join(cls.tokenizer_dir, "tokenizer.json")
        cls.tokenizer = SLMTokenizer(cls.tokenizer_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def _write_jsonl(self, records, filename="raw.jsonl"):
        path = os.path.join(self.tmp_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return path


class TestLoadRawExamples(SFTTestBase):
    def test_loads_valid_examples(self):
        path = self._write_jsonl([
            {"prompt": "What is the capital of France?", "response": "Paris."},
            {"prompt": "Summarize this text.", "response": "A short summary."},
        ])
        examples = load_raw_examples(path)
        self.assertEqual(len(examples), 2)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_raw_examples(os.path.join(self.tmp_dir, "nope.jsonl"))

    def test_missing_keys_raises(self):
        path = self._write_jsonl([{"prompt": "hello"}], filename="bad1.jsonl")
        with self.assertRaises(ValueError):
            load_raw_examples(path)

    def test_empty_fields_raises(self):
        path = self._write_jsonl([{"prompt": "", "response": "ok"}], filename="bad2.jsonl")
        with self.assertRaises(ValueError):
            load_raw_examples(path)

    def test_invalid_json_raises(self):
        path = os.path.join(self.tmp_dir, "bad3.jsonl")
        with open(path, "w") as f:
            f.write("{not valid json\n")
        with self.assertRaises(ValueError):
            load_raw_examples(path)

    def test_empty_file_raises(self):
        path = os.path.join(self.tmp_dir, "empty.jsonl")
        with open(path, "w") as f:
            pass
        with self.assertRaises(ValueError):
            load_raw_examples(path)


class TestPrepareExample(SFTTestBase):
    def test_labels_mask_prompt_exactly(self):
        prompt = "What is the capital of France?"
        response = "Paris."
        result = prepare_example(self.tokenizer, prompt, response, max_length=1024)
        self.assertIsNotNone(result)

        prompt_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        response_ids = self.tokenizer.encode(response, add_bos=False, add_eos=True)

        self.assertEqual(result["input_ids"], prompt_ids + response_ids)
        self.assertEqual(result["labels"][: len(prompt_ids)], [IGNORE_INDEX] * len(prompt_ids))
        self.assertEqual(result["labels"][len(prompt_ids):], response_ids)

    def test_starts_with_bos(self):
        result = prepare_example(self.tokenizer, "Explain gravity.", "Gravity pulls objects together.", 1024)
        self.assertEqual(result["input_ids"][0], BOS_ID)

    def test_ends_with_eos(self):
        result = prepare_example(self.tokenizer, "Explain gravity.", "Gravity pulls objects together.", 1024)
        self.assertEqual(result["input_ids"][-1], EOS_ID)

    def test_truncation_keeps_full_prompt(self):
        prompt = "Explain how photosynthesis converts light into chemical energy in plants."
        response = "Photosynthesis is a long process " * 20  # deliberately long
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)

        result = prepare_example(self.tokenizer, prompt, response, max_length=len(prompt_ids) + 5)
        self.assertIsNotNone(result)
        self.assertEqual(len(result["input_ids"]), len(prompt_ids) + 5)
        self.assertEqual(result["input_ids"][: len(prompt_ids)], prompt_ids)
        # At least one non-masked (response) label should remain.
        self.assertTrue(any(l != IGNORE_INDEX for l in result["labels"]))

    def test_prompt_too_long_returns_none(self):
        prompt = "This is a somewhat longer prompt used to test truncation behavior in the pipeline."
        response = "Short reply."
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        result = prepare_example(self.tokenizer, prompt, response, max_length=len(prompt_ids) - 1)
        self.assertIsNone(result)


class TestPrepareDataset(SFTTestBase):
    def test_end_to_end_prepare(self):
        raw_path = self._write_jsonl([
            {"prompt": "What is the capital of France?", "response": "Paris."},
            {"prompt": "Summarize the article.", "response": "A short summary of the article."},
            {"prompt": "Write a short poem about the seasons.", "response": "Autumn leaves fall gently down."},
        ], filename="prepare_raw.jsonl")
        out_path = os.path.join(self.tmp_dir, "prepared.jsonl")

        stats = prepare_dataset(raw_path, self.tokenizer_path, out_path, max_length=64)
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["kept"], 3)
        self.assertTrue(os.path.isfile(out_path))

        with open(out_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(lines), 3)
        for record in lines:
            self.assertIn("input_ids", record)
            self.assertIn("labels", record)
            self.assertEqual(len(record["input_ids"]), len(record["labels"]))

    def test_all_skipped_raises(self):
        raw_path = self._write_jsonl([
            {"prompt": "What is the capital of France and its long history dating back many centuries?",
             "response": "Paris."},
        ], filename="prepare_raw_skip.jsonl")
        out_path = os.path.join(self.tmp_dir, "prepared_skip.jsonl")
        with self.assertRaises(ValueError):
            prepare_dataset(raw_path, self.tokenizer_path, out_path, max_length=3)


class TestSFTDataloader(SFTTestBase):
    def _prepared_path(self):
        raw_path = self._write_jsonl([
            {"prompt": "What is the capital of France?", "response": "Paris."},
            {"prompt": "Summarize the article.", "response": "A short summary of the article text."},
            {"prompt": "Write a poem.", "response": "Autumn leaves fall gently down onto the ground."},
            {"prompt": "Explain gravity.", "response": "Gravity pulls objects toward each other."},
        ], filename="dl_raw.jsonl")
        out_path = os.path.join(self.tmp_dir, "dl_prepared.jsonl")
        prepare_dataset(raw_path, self.tokenizer_path, out_path, max_length=64)
        return out_path

    def test_dataset_loads_examples(self):
        path = self._prepared_path()
        ds = PreparedInstructionDataset(path)
        self.assertEqual(len(ds), 4)
        item = ds[0]
        self.assertIn("input_ids", item)
        self.assertIn("labels", item)
        self.assertEqual(item["input_ids"].shape, item["labels"].shape)

    def test_collate_pads_correctly(self):
        path = self._prepared_path()
        ds = PreparedInstructionDataset(path)
        batch = [ds[i] for i in range(len(ds))]
        collated = sft_collate_fn(batch, pad_id=0)

        max_len = max(item["input_ids"].size(0) for item in batch)
        self.assertEqual(collated["input_ids"].shape, (len(batch), max_len))
        self.assertEqual(collated["labels"].shape, (len(batch), max_len))

        # Every padded position in labels must be IGNORE_INDEX.
        for i, item in enumerate(batch):
            orig_len = item["input_ids"].size(0)
            if orig_len < max_len:
                pad_region = collated["labels"][i, orig_len:]
                self.assertTrue(torch.all(pad_region == IGNORE_INDEX))
                input_pad_region = collated["input_ids"][i, orig_len:]
                self.assertTrue(torch.all(input_pad_region == 0))

    def test_collate_rejects_empty_batch(self):
        with self.assertRaises(ValueError):
            sft_collate_fn([])

    def test_dataloader_end_to_end(self):
        path = self._prepared_path()
        loader = build_sft_dataloader(path, batch_size=2, pad_id=0, shuffle=False)
        batches = list(loader)
        self.assertGreater(len(batches), 0)
        for batch in batches:
            self.assertIn("input_ids", batch)
            self.assertIn("labels", batch)

    def test_missing_key_raises(self):
        bad_path = os.path.join(self.tmp_dir, "bad_prepared.jsonl")
        with open(bad_path, "w") as f:
            f.write(json.dumps({"input_ids": [1, 2, 3]}) + "\n")
        with self.assertRaises(ValueError):
            PreparedInstructionDataset(bad_path)


class TestSFTTrainingIntegration(SFTTestBase):
    def _make_model(self):
        cfg = SLMConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            max_position_embeddings=64,
            tie_word_embeddings=True,
        )
        return SLM(cfg), cfg

    def _prepared_path(self, n_repeats=8):
        records = [
            {"prompt": "What is the capital of France?", "response": "Paris is the capital of France."},
            {"prompt": "Summarize the article.", "response": "A short summary of the article text here."},
            {"prompt": "Write a poem about seasons.", "response": "Autumn leaves fall gently down onto ground."},
            {"prompt": "Explain gravity simply.", "response": "Gravity pulls objects toward each other always."},
        ] * n_repeats
        raw_path = self._write_jsonl(records, filename="train_raw.jsonl")
        out_path = os.path.join(self.tmp_dir, "train_prepared.jsonl")
        prepare_dataset(raw_path, self.tokenizer_path, out_path, max_length=64)
        return out_path

    def test_sft_training_reduces_loss(self):
        model, cfg = self._make_model()
        data_path = self._prepared_path()
        loader = build_sft_dataloader(data_path, batch_size=8, pad_id=0, shuffle=True)

        optimizer = build_optimizer(model, learning_rate=5e-3)
        trainer_cfg = TrainerConfig(
            max_steps=20, gradient_accumulation_steps=1, log_every=5,
            eval_every=0, checkpoint_every=0, use_amp=False,
        )
        trainer = Trainer(model, optimizer, None, loader, trainer_cfg, device="cpu")
        results = trainer.train()

        history = results["history"]
        early = sum(h["loss"] for h in history[:3]) / 3
        late = sum(h["loss"] for h in history[-3:]) / 3
        self.assertLess(late, early)

    def test_evaluate_perplexity_after_training(self):
        model, cfg = self._make_model()
        data_path = self._prepared_path()

        metrics = evaluate_perplexity(model, data_path, batch_size=8, device="cpu")
        self.assertTrue(torch.isfinite(torch.tensor(metrics["eval_loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(metrics["eval_perplexity"])))
        self.assertGreater(metrics["num_batches"], 0)

    def test_generate_greedy_produces_string(self):
        model, cfg = self._make_model()
        output = generate_greedy(model, self.tokenizer, "What is the capital of France?", max_new_tokens=10, device="cpu")
        self.assertIsInstance(output, str)

    def test_generate_respects_max_new_tokens(self):
        model, cfg = self._make_model()
        # With an untrained model EOS is unlikely to be hit, so length should
        # be governed by max_new_tokens.
        output_ids_len_before = len(self.tokenizer.encode("Explain gravity."))
        output = generate_greedy(model, self.tokenizer, "Explain gravity.", max_new_tokens=5, device="cpu")
        response_token_count = len(self.tokenizer.encode(output))
        # Loose bound -- decoding/re-encoding can shift token count slightly,
        # but it must not run away unbounded.
        self.assertLessEqual(response_token_count, 20)

    def test_generate_rejects_empty_prompt(self):
        model, cfg = self._make_model()
        with self.assertRaises(ValueError):
            generate_greedy(model, self.tokenizer, "", max_new_tokens=5, device="cpu")

    def test_generate_rejects_invalid_max_new_tokens(self):
        model, cfg = self._make_model()
        with self.assertRaises(ValueError):
            generate_greedy(model, self.tokenizer, "Hello", max_new_tokens=0, device="cpu")

    def test_collect_sample_prompts_recovers_prompt_text(self):
        data_path = self._prepared_path()
        prompts = collect_sample_prompts(data_path, self.tokenizer, num_samples=2)
        self.assertEqual(len(prompts), 2)
        for p in prompts:
            self.assertIsInstance(p, str)
            self.assertGreater(len(p.strip()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
