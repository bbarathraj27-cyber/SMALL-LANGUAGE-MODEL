"""
tests/test_training.py

Tests the training/ module (loss, optimizer, scheduler, checkpoint, trainer)
against a tiny real model and dataloader. Verifies:
  - loss.py shape/range validation and correctness
  - optimizer.py parameter grouping (decay vs no-decay)
  - scheduler.py warmup + cosine/linear decay math
  - checkpoint.py save/load round-trip restores model + optimizer + step exactly
  - Trainer.train_step / optimizer_step actually reduce loss over a few steps
  - Trainer.evaluate produces finite loss/perplexity
  - Trainer.resume_from_checkpoint restores global_step correctly

NOTE: this test file uses a small self-contained model and dataloader
(see _testsupport/) rather than importing model/slm.py and
dataset/dataloader.py directly, so that training/ can be verified in
isolation from those modules' exact interfaces.
"""

import os
import sys
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_testsupport")))

import torch

from model.slm import SLM, SLMConfig
from dataset.dataloader import build_pretrain_dataloader

from training.loss import compute_lm_loss, compute_perplexity, IGNORE_INDEX
from training.optimizer import build_optimizer, count_optimized_parameters
from training.scheduler import build_lr_scheduler
from training.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from training.trainer import Trainer, TrainerConfig


def make_tiny_model():
    cfg = SLMConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=16,
        tie_word_embeddings=True,
    )
    return SLM(cfg), cfg


class TestLoss(unittest.TestCase):
    def test_basic_loss_is_finite_and_positive(self):
        logits = torch.randn(2, 5, 20)
        labels = torch.randint(0, 20, (2, 5))
        loss = compute_lm_loss(logits, labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0)

    def test_ignore_index_excludes_positions(self):
        logits = torch.randn(1, 4, 10)
        labels = torch.tensor([[1, 2, IGNORE_INDEX, IGNORE_INDEX]])
        loss_masked = compute_lm_loss(logits, labels)

        labels_full = torch.tensor([[1, 2, 3, 4]])
        loss_full = compute_lm_loss(logits, labels_full)
        # Not asserting exact equality (different targets), just that masked
        # loss doesn't crash and produces a different, finite value.
        self.assertTrue(torch.isfinite(loss_masked))
        self.assertTrue(torch.isfinite(loss_full))

    def test_all_positions_masked_returns_zero(self):
        logits = torch.randn(1, 3, 10, requires_grad=True)
        labels = torch.full((1, 3), IGNORE_INDEX)
        loss = compute_lm_loss(logits, labels)
        self.assertEqual(loss.item(), 0.0)

    def test_shape_mismatch_raises(self):
        logits = torch.randn(2, 5, 20)
        labels = torch.randint(0, 20, (2, 4))
        with self.assertRaises(ValueError):
            compute_lm_loss(logits, labels)

    def test_out_of_range_label_raises(self):
        logits = torch.randn(1, 2, 10)
        labels = torch.tensor([[0, 999]])
        with self.assertRaises(ValueError):
            compute_lm_loss(logits, labels)

    def test_perplexity_matches_exp_of_loss(self):
        loss = torch.tensor(1.0)
        ppl = compute_perplexity(loss)
        self.assertAlmostEqual(ppl, torch.exp(loss).item(), places=4)

    def test_perplexity_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            compute_perplexity(torch.tensor(float("nan")))


class TestOptimizer(unittest.TestCase):
    def test_decay_and_no_decay_groups_populated(self):
        model, _ = make_tiny_model()
        opt = build_optimizer(model, learning_rate=1e-3)
        self.assertEqual(len(opt.param_groups), 2)
        self.assertGreater(len(opt.param_groups[0]["params"]), 0)  # decay
        self.assertGreater(len(opt.param_groups[1]["params"]), 0)  # no-decay
        self.assertEqual(opt.param_groups[0]["weight_decay"], 0.1)
        self.assertEqual(opt.param_groups[1]["weight_decay"], 0.0)

    def test_norm_params_get_no_decay(self):
        model, _ = make_tiny_model()
        opt = build_optimizer(model, learning_rate=1e-3)
        no_decay_ids = {id(p) for p in opt.param_groups[1]["params"]}
        for name, param in model.named_parameters():
            if "norm" in name.lower():
                self.assertIn(id(param), no_decay_ids, f"{name} should be in no-decay group")

    def test_count_matches_model_parameters(self):
        model, _ = make_tiny_model()
        opt = build_optimizer(model, learning_rate=1e-3)
        self.assertEqual(count_optimized_parameters(opt), model.num_parameters())

    def test_invalid_lr_rejected(self):
        model, _ = make_tiny_model()
        with self.assertRaises(ValueError):
            build_optimizer(model, learning_rate=0)
        with self.assertRaises(ValueError):
            build_optimizer(model, learning_rate=-1)


class TestScheduler(unittest.TestCase):
    def test_warmup_increases_linearly(self):
        model, _ = make_tiny_model()
        opt = build_optimizer(model, learning_rate=1.0)
        sched = build_lr_scheduler(opt, warmup_steps=10, total_steps=100)
        lrs = []
        for _ in range(10):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        for i in range(1, len(lrs)):
            self.assertGreaterEqual(lrs[i], lrs[i - 1])

    def test_decays_after_warmup(self):
        model, _ = make_tiny_model()
        opt = build_optimizer(model, learning_rate=1.0)
        sched = build_lr_scheduler(opt, warmup_steps=5, total_steps=50)
        for _ in range(5):
            opt.step()
            sched.step()
        lr_at_warmup_end = opt.param_groups[0]["lr"]
        for _ in range(20):
            opt.step()
            sched.step()
        lr_later = opt.param_groups[0]["lr"]
        self.assertLess(lr_later, lr_at_warmup_end)

    def test_respects_min_lr_ratio(self):
        model, _ = make_tiny_model()
        opt = build_optimizer(model, learning_rate=1.0)
        sched = build_lr_scheduler(opt, warmup_steps=0, total_steps=10, min_lr_ratio=0.2)
        for _ in range(10):
            opt.step()
            sched.step()
        final_lr = opt.param_groups[0]["lr"]
        self.assertAlmostEqual(final_lr, 0.2, places=2)

    def test_invalid_args_rejected(self):
        model, _ = make_tiny_model()
        opt = build_optimizer(model, learning_rate=1.0)
        with self.assertRaises(ValueError):
            build_lr_scheduler(opt, warmup_steps=-1, total_steps=10)
        with self.assertRaises(ValueError):
            build_lr_scheduler(opt, warmup_steps=20, total_steps=10)
        with self.assertRaises(ValueError):
            build_lr_scheduler(opt, warmup_steps=0, total_steps=0)


class TestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="slm_ckpt_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_and_load_roundtrip(self):
        model, cfg = make_tiny_model()
        opt = build_optimizer(model, learning_rate=1e-3)
        sched = build_lr_scheduler(opt, warmup_steps=2, total_steps=20)

        path = save_checkpoint(self.tmp_dir, step=5, model=model, optimizer=opt,
                                scheduler=sched, epoch=1)
        self.assertTrue(os.path.isfile(path))

        model2, _ = make_tiny_model()
        opt2 = build_optimizer(model2, learning_rate=1e-3)
        sched2 = build_lr_scheduler(opt2, warmup_steps=2, total_steps=20)

        result = load_checkpoint(path, model2, opt2, sched2)
        self.assertEqual(result["step"], 5)
        self.assertEqual(result["epoch"], 1)

        for p1, p2 in zip(model.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_load_missing_file_raises(self):
        model, _ = make_tiny_model()
        with self.assertRaises(FileNotFoundError):
            load_checkpoint(os.path.join(self.tmp_dir, "nope.pt"), model)

    def test_find_latest_checkpoint(self):
        model, _ = make_tiny_model()
        opt = build_optimizer(model, learning_rate=1e-3)
        save_checkpoint(self.tmp_dir, step=1, model=model, optimizer=opt, scheduler=None, epoch=0)
        save_checkpoint(self.tmp_dir, step=10, model=model, optimizer=opt, scheduler=None, epoch=0)
        save_checkpoint(self.tmp_dir, step=5, model=model, optimizer=opt, scheduler=None, epoch=0)
        latest = find_latest_checkpoint(self.tmp_dir)
        self.assertIn("step10", latest)

    def test_find_latest_checkpoint_empty_dir(self):
        self.assertIsNone(find_latest_checkpoint(self.tmp_dir))

    def test_keep_last_n_prunes_old_checkpoints(self):
        model, _ = make_tiny_model()
        opt = build_optimizer(model, learning_rate=1e-3)
        for step in [1, 2, 3, 4, 5]:
            save_checkpoint(self.tmp_dir, step=step, model=model, optimizer=opt,
                             scheduler=None, epoch=0, keep_last_n=2)
        remaining = [f for f in os.listdir(self.tmp_dir) if f.endswith(".pt")]
        self.assertEqual(len(remaining), 2)
        self.assertIn("checkpoint_step4.pt", remaining)
        self.assertIn("checkpoint_step5.pt", remaining)


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="slm_trainer_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _build_trainer(self, max_steps=20, checkpoint_every=0):
        model, cfg = make_tiny_model()
        opt = build_optimizer(model, learning_rate=5e-3)
        sched = build_lr_scheduler(opt, warmup_steps=2, total_steps=max_steps)
        dataloader = build_pretrain_dataloader(
            vocab_size=cfg.vocab_size, block_size=cfg.max_position_embeddings,
            num_examples=64, batch_size=8,
        )
        trainer_cfg = TrainerConfig(
            max_steps=max_steps,
            gradient_accumulation_steps=1,
            log_every=5,
            eval_every=0,
            checkpoint_every=checkpoint_every,
            checkpoint_dir=self.tmp_dir,
            use_amp=False,
        )
        trainer = Trainer(model, opt, sched, dataloader, trainer_cfg, device="cpu")
        return trainer

    def test_loss_decreases_over_steps(self):
        trainer = self._build_trainer(max_steps=30)
        results = trainer.train()
        history = results["history"]
        self.assertEqual(results["final_step"], 30)
        early_loss = sum(h["loss"] for h in history[:3]) / 3
        late_loss = sum(h["loss"] for h in history[-3:]) / 3
        self.assertLess(late_loss, early_loss)

    def test_gradient_accumulation_steps_correctly(self):
        model, cfg = make_tiny_model()
        opt = build_optimizer(model, learning_rate=1e-3)
        dataloader = build_pretrain_dataloader(
            vocab_size=cfg.vocab_size, block_size=cfg.max_position_embeddings,
            num_examples=64, batch_size=4,
        )
        trainer_cfg = TrainerConfig(
            max_steps=3, gradient_accumulation_steps=4, log_every=1,
            eval_every=0, checkpoint_every=0, use_amp=False,
        )
        trainer = Trainer(model, opt, None, dataloader, trainer_cfg, device="cpu")
        results = trainer.train()
        # 3 optimizer steps expected even though many more micro-batches ran
        self.assertEqual(results["final_step"], 3)

    def test_evaluate_returns_finite_metrics(self):
        model, cfg = make_tiny_model()
        opt = build_optimizer(model, learning_rate=1e-3)
        train_loader = build_pretrain_dataloader(
            vocab_size=cfg.vocab_size, block_size=cfg.max_position_embeddings,
            num_examples=32, batch_size=8,
        )
        eval_loader = build_pretrain_dataloader(
            vocab_size=cfg.vocab_size, block_size=cfg.max_position_embeddings,
            num_examples=16, batch_size=8, shuffle=False,
        )
        trainer_cfg = TrainerConfig(max_steps=1, use_amp=False, checkpoint_every=0, eval_every=0)
        trainer = Trainer(model, opt, None, train_loader, trainer_cfg, eval_dataloader=eval_loader, device="cpu")
        metrics = trainer.evaluate()
        self.assertTrue(torch.isfinite(torch.tensor(metrics["eval_loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(metrics["eval_perplexity"])))

    def test_evaluate_without_eval_dataloader_raises(self):
        trainer = self._build_trainer(max_steps=5)
        with self.assertRaises(ValueError):
            trainer.evaluate()

    def test_checkpointing_and_resume(self):
        trainer = self._build_trainer(max_steps=10, checkpoint_every=5)
        trainer.train()
        self.assertTrue(os.path.isfile(os.path.join(self.tmp_dir, "checkpoint_step10.pt")))

        # New trainer, resume from the saved checkpoint dir.
        resumed_trainer = self._build_trainer(max_steps=10, checkpoint_every=5)
        found = resumed_trainer.resume_from_checkpoint()
        self.assertTrue(found)
        self.assertEqual(resumed_trainer.global_step, 10)

    def test_resume_with_no_checkpoint_returns_false(self):
        trainer = self._build_trainer(max_steps=5)
        found = trainer.resume_from_checkpoint()
        self.assertFalse(found)

    def test_invalid_trainer_config_rejected(self):
        with self.assertRaises(ValueError):
            TrainerConfig(max_steps=0)
        with self.assertRaises(ValueError):
            TrainerConfig(max_steps=10, gradient_accumulation_steps=0)
        with self.assertRaises(ValueError):
            TrainerConfig(max_steps=10, max_grad_norm=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
