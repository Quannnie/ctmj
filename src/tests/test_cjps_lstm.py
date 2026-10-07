"""Tests for the LSTM pipeline under ``src/cjps_lstm/``.

The defects these pin down are the leakage family: the target event
sneaking into the input, preprocessing fitted outside the training rows,
padding advancing the recurrent state, and a customer straddling two
partitions. Each is a failure that trains cleanly and reports plausible
metrics while quietly measuring the wrong thing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from django.test import SimpleTestCase

from src.cjps_lstm import data as data_mod
from src.cjps_lstm import run as run_mod
from src.cjps_lstm import train as train_mod
from src.cjps_lstm.config import PAD_TOKEN, UNK_TOKEN, LSTMConfig
from src.cjps_lstm.model import JourneyDataset, JourneyLSTM


def _journeys(rows):
    """A journeys frame with every column ``build_events`` reads."""
    frame = pd.DataFrame(
        rows,
        columns=["UserID", "TIMESPSS", "type_touch", "Duration",
                 "DEVICE_TYPE", "purchase_own", "purchase_any"],
    )
    return frame


def _users(n: int) -> pd.DataFrame:
    """A minimal users table with all demographic columns populated."""
    rng = np.random.default_rng(7)
    cols = {c: rng.integers(1, 6, n) for c in data_mod.DEMOGRAPHIC_FEATURES}
    cols["UserID"] = np.arange(1, n + 1)
    return pd.DataFrame(cols)


class BuildEventsTests(SimpleTestCase):
    def test_consecutive_repeats_collapse_with_summed_duration(self):
        out = data_mod.build_events(_journeys([
            (1, "2024-01-01 10:00:00", 3, 10, "FIXED", 0, 0),
            (1, "2024-01-01 10:05:00", 3, 20, "FIXED", 0, 0),
            (1, "2024-01-01 12:00:00", 7, 40, "MOBILE", 1, 1),
        ]))
        self.assertEqual(len(out), 2)
        first = out.iloc[0]
        self.assertEqual(first["type_touch"], 3)
        self.assertEqual(first["duration"], 30)

    def test_zero_and_negative_durations_are_tracking_artefacts(self):
        out = data_mod.build_events(_journeys([
            (1, "2024-01-01 10:00:00", 3, 0, "FIXED", 0, 0),
            (1, "2024-01-01 10:05:00", 5, -4, "FIXED", 0, 0),
            (1, "2024-01-01 12:00:00", 7, 40, "FIXED", 0, 0),
        ]))
        self.assertEqual(out["type_touch"].tolist(), [7])

    def test_run_boundaries_do_not_cross_users(self):
        """Two users ending/starting on the same channel stay separate.

        A single global change-detector would merge user 1's last run with
        user 2's first run — one user's events would ride inside another's
        sequence.
        """
        out = data_mod.build_events(_journeys([
            (1, "2024-01-01 10:00:00", 3, 10, "FIXED", 0, 0),
            (2, "2024-01-01 10:00:00", 3, 10, "FIXED", 0, 0),
            (2, "2024-01-01 10:05:00", 3, 20, "FIXED", 0, 0),
        ]))
        self.assertEqual(len(out), 2)
        self.assertEqual(out["UserID"].tolist(), [1, 2])

    def test_first_event_gap_is_zero(self):
        out = data_mod.build_events(_journeys([
            (1, "2024-01-01 10:00:00", 3, 10, "FIXED", 0, 0),
            (1, "2024-01-01 12:00:00", 7, 10, "FIXED", 0, 0),
        ]))
        self.assertEqual(out.iloc[0]["gap_log"], 0.0)
        self.assertAlmostEqual(out.iloc[1]["gap_log"], np.log1p(2.0), places=5)


class BuildExamplesTests(SimpleTestCase):
    def _examples(self):
        events = data_mod.build_events(_journeys([
            (1, "2024-01-01 10:00:00", 1, 10, "FIXED", 0, 0),
            (1, "2024-01-02 10:00:00", 2, 10, "FIXED", 1, 0),
            (1, "2024-01-03 10:00:00", 3, 10, "MOBILE", 0, 0),
            (1, "2024-01-04 10:00:00", 7, 10, "FIXED", 0, 1),   # target event
        ]))
        return data_mod.build_examples(events, min_len=3)

    def test_target_is_the_last_event_and_never_enters_the_input(self):
        """Aggregates computed over the full journey would leak the answer:
        ``purchase_any_n`` would count the target event's own flags."""
        row = self._examples().iloc[0]
        self.assertEqual(row["target"], 7)
        self.assertEqual(row["seq_touch"], [1, 2, 3])
        # The target event carries purchase_any=1; the input does not.
        self.assertEqual(row["purchase_any_n"], 0.0)
        self.assertEqual(row["purchase_own_n"], 1.0)
        self.assertEqual(row["seq_len"], 3)
        self.assertEqual(row["n_unique_touch"], 3)

    def test_users_below_min_length_fail_loudly(self):
        """An empty examples frame has no columns; the old code logged
        ``frame["seq_len"]`` and died with a bare KeyError. The pipeline
        should say *why* there is nothing to learn from."""
        events = data_mod.build_events(_journeys([
            (1, "2024-01-01 10:00:00", 1, 10, "FIXED", 0, 0),
            (1, "2024-01-02 10:00:00", 2, 10, "FIXED", 0, 0),   # only 2 events
        ]))
        with self.assertRaises(ValueError) as ctx:
            data_mod.build_examples(events, min_len=3)
        self.assertIn("min_journey_length", str(ctx.exception))

    def test_input_is_chronological(self):
        row = self._examples().iloc[0]
        self.assertEqual(row["seq_touch"], sorted(row["seq_touch"]))
        self.assertEqual(row["seq_touch"][-1], 3)  # step2 = most recent input


class RareClassAndSplitTests(SimpleTestCase):
    def _frame(self, n_per_class=20):
        """Synthetic example rows: enough members per class to split."""
        rows = []
        for cls in (1, 2, 3):
            for i in range(n_per_class):
                rows.append({"UserID": len(rows), "target": cls,
                             "seq_touch": [1, 2], "seq_len": 2})
        return pd.DataFrame(rows)

    def test_rare_classes_are_removed_before_splitting(self):
        """A class with two members cannot appear in three partitions;
        keeping it makes the stratified split fail or place it nowhere."""
        frame = self._frame()
        frame.loc[len(frame)] = {"UserID": 999, "target": 9,
                                 "seq_touch": [1, 2], "seq_len": 2}
        out = data_mod.drop_rare_classes(frame, min_samples=6)
        self.assertNotIn(9, set(out["target"]))
        self.assertEqual(len(out), 60)

    def test_partitions_are_disjoint_and_cover_everything(self):
        frame = self._frame(n_per_class=40)
        split = data_mod.split_examples(frame, LSTMConfig())
        tr, va, te = set(split.train), set(split.val), set(split.test)
        self.assertEqual(len(tr & va) + len(tr & te) + len(va & te), 0)
        self.assertEqual(len(tr | va | te), len(frame))

    def test_split_is_stratified(self):
        frame = self._frame(n_per_class=40)
        split = data_mod.split_examples(frame, LSTMConfig())
        for idx in (split.train, split.val, split.test):
            counts = frame.iloc[list(idx)]["target"].value_counts()
            self.assertEqual(set(counts.index), {1, 2, 3})


class PaddingTests(SimpleTestCase):
    def test_padding_uses_zeros_and_lengths_are_real(self):
        ids, nums, lens = data_mod.pad_batch(
            [np.array([5, 6, 7]), np.array([9])],
            [np.ones((3, 5)), np.ones((1, 5)) * 2],
            max_len=4,
        )
        self.assertEqual(ids.shape, (2, 4))
        self.assertEqual(ids[0].tolist(), [5, 6, 7, 0])
        self.assertEqual(ids[1].tolist(), [9, 0, 0, 0])
        self.assertEqual(lens.tolist(), [3, 1])
        self.assertTrue((nums[1, 1:] == 0).all())

    def test_truncation_keeps_the_most_recent_events(self):
        """The tail of the journey is closest to the target; dropping the
        newest events would amputate the strongest signal."""
        ids, _, lens = data_mod.pad_batch(
            [np.arange(10)],
            [np.zeros((10, 5))],
            max_len=4,
        )
        self.assertEqual(ids[0].tolist(), [6, 7, 8, 9])
        self.assertEqual(lens[0], 4)


class VocabAndScalerTests(SimpleTestCase):
    def test_vocab_reserves_padding_and_unknown(self):
        vocab = data_mod.TouchpointVocab()
        self.assertEqual(vocab.pad, PAD_TOKEN)
        self.assertEqual(vocab.encode(999), UNK_TOKEN)
        self.assertEqual(vocab.encode(1), 1)
        self.assertEqual(vocab.decode(1), 1)

    def test_binary_channels_are_not_standardised(self):
        """Rescaling {0,1} flags only rewrites a distance that is already
        meaningful; the scaler must leave them alone."""
        rng = np.random.default_rng(0)
        num = np.column_stack([
            rng.normal(3, 2, 200), rng.normal(1, 1, 200),
            rng.integers(0, 2, 200), rng.integers(0, 2, 200), rng.integers(0, 2, 200),
        ]).astype(np.float32)
        mean, std = data_mod.fit_event_scaler([num])
        self.assertTrue((mean[2:] == 0).all() and (std[2:] == 1).all())
        self.assertNotEqual(std[0], 1.0)


class StaticEncoderTests(SimpleTestCase):
    def test_output_width_matches_feature_names(self):
        frame = _users(60)
        for col in data_mod.AGGREGATE_FEATURES:
            frame[col] = np.random.default_rng(1).normal(size=len(frame))
        frame["final_label"] = 0
        enc = data_mod.StaticEncoder(extra_nominal=("final_label",))
        out = enc.fit_transform(frame)
        self.assertEqual(out.shape[1], len(enc.feature_names_))
        self.assertEqual(out.dtype, np.float32)

    def test_transform_of_unseen_category_does_not_crash(self):
        """``handle_unknown='ignore'``: a category absent from training
        must become all-zeros, not an exception."""
        frame = _users(60)
        frame["GenderID"] = 1
        for col in data_mod.AGGREGATE_FEATURES:
            frame[col] = 0.0
        enc = data_mod.StaticEncoder().fit(frame)
        new = frame.head(3).copy()
        new["GenderID"] = 77
        out = enc.transform(new)
        self.assertEqual(out.shape[0], 3)
        self.assertFalse(np.isnan(out).any())


class ModelShapeTests(SimpleTestCase):
    def _forward(self, bidirectional=False):
        torch.manual_seed(0)
        model = JourneyLSTM(vocab_size=22, n_event_features=5, n_static=7,
                            n_classes=4, emb_dim=8, lstm_units=16,
                            dense_units=8, bidirectional=bidirectional)
        ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        num = torch.randn(2, 4, 5)
        lens = torch.tensor([3, 2])
        static = torch.randn(2, 7)
        return model(ids, num, lens, static)

    def test_output_shape_is_n_by_classes(self):
        self.assertEqual(self._forward().shape, (2, 4))
        self.assertEqual(self._forward(bidirectional=True).shape, (2, 4))

    def test_padding_does_not_advance_the_state(self):
        """If the padded steps fed the LSTM, a 3-event user padded to 10
        would be summarised by 7 steps of silence."""
        torch.manual_seed(0)
        model = JourneyLSTM(vocab_size=22, n_event_features=5, n_static=0,
                            n_classes=4, emb_dim=8, lstm_units=16,
                            dense_units=8)
        model.eval()
        ids = torch.tensor([[1, 2, 3]])
        num = torch.randn(1, 3, 5)
        base = model(ids, num, torch.tensor([3]), torch.zeros(1, 0))

        ids_pad = torch.tensor([[1, 2, 3, 0, 0, 0, 0]])
        num_pad = torch.cat([num, torch.zeros(1, 4, 5)], dim=1)
        padded = model(ids_pad, num_pad, torch.tensor([3]), torch.zeros(1, 0))
        self.assertTrue(torch.allclose(base, padded, atol=1e-6))


class TrainingTests(SimpleTestCase):
    def _dataset(self, n, n_classes=4, max_len=6, n_static=7, seed=0):
        rng = np.random.default_rng(seed)
        ids = rng.integers(1, 21, (n, max_len))
        num = rng.normal(size=(n, max_len, 5)).astype(np.float32)
        lens = np.full(n, max_len)
        static = rng.normal(size=(n, n_static)).astype(np.float32)
        labels = rng.integers(0, n_classes, n)
        return JourneyDataset(ids, num, lens, static, labels)

    def _config(self, **kw):
        import dataclasses
        return dataclasses.replace(
            LSTMConfig(), emb_dim=8, lstm_units=16, dense_units=8,
            max_epochs=3, patience=2, batch_size=32, **kw)

    def test_class_weights_rebalance_the_loss(self):
        labels = np.array([0] * 90 + [1] * 10)
        w = train_mod.balanced_class_weights(labels, 2)
        self.assertGreater(w[1], w[0])
        self.assertAlmostEqual(float(w.mean()), 1.0, places=5)

    def test_train_model_populates_history_and_restores_best(self):
        cfg = self._config()
        model = train_mod.build_model(cfg, 5, 7, 4, 22)
        history = train_mod.train_model(
            model, self._dataset(128, seed=1), self._dataset(64, seed=2), cfg)
        self.assertGreaterEqual(len(history.train_loss), 1)
        self.assertGreaterEqual(history.best_epoch, 0)
        self.assertEqual(len(history.train_loss), len(history.val_loss))

    def test_evaluate_returns_the_full_metric_bundle(self):
        cfg = self._config()
        model = train_mod.build_model(cfg, 5, 7, 4, 22)
        out = train_mod.evaluate(model, self._dataset(64), np.arange(4), 32)
        for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                    "f1_weighted", "balanced_accuracy", "top3_accuracy",
                    "roc_auc_macro", "pr_auc_macro", "confusion_matrix"):
            self.assertIn(key, out)
        self.assertEqual(out["confusion_matrix"].shape, (4, 4))
        self.assertGreaterEqual(out["top3_accuracy"], out["accuracy"])

    def test_permutation_importance_reports_every_block(self):
        cfg = self._config()
        model = train_mod.build_model(cfg, 5, 7, 4, 22)
        train_mod.train_model(model, self._dataset(128, seed=1),
                              self._dataset(64, seed=2), cfg)
        blocks = {"SEQ:touchpoints": ("seq_ids", 0),
                  "SEQ_NUM:gap_log": ("seq_num", 1),
                  "STATIC:Age": ("static", 0)}
        table = train_mod.permutation_importance(
            model, self._dataset(64, seed=3), blocks, 32, seed=0, repeats=2)
        self.assertEqual(len(table), 3)
        self.assertEqual(list(table.columns), ["feature", "importance", "std"])


class VariantDatasetTests(SimpleTestCase):
    def _tensors(self, n=20, n_classes=4, max_len=5, n_static=6):
        rng = np.random.default_rng(0)
        return {
            "ids": rng.integers(1, 21, (n, max_len)),
            "num": rng.normal(size=(n, max_len, 5)).astype(np.float32),
            "lens": np.full(n, max_len),
            "static": rng.normal(size=(n, n_static)).astype(np.float32),
            "y": rng.integers(0, n_classes, n),
            "classes": np.arange(n_classes),
        }

    def test_seq_only_blanks_numeric_and_static_inputs(self):
        """The ablation must zero the extra channels, not drop rows — the
        same label and the same journey are being scored either way."""
        tensors = self._tensors()
        ds = run_mod._variant_datasets(tensors, np.arange(10), seq_only=True)
        self.assertEqual(ds.static.shape[1], 0)
        self.assertTrue((ds.seq_num.numpy() == 0).all())
        np.testing.assert_array_equal(ds.seq_ids.numpy(), tensors["ids"][:10])
        np.testing.assert_array_equal(ds.labels.numpy(), tensors["y"][:10])

    def test_fused_variant_carries_the_static_vector(self):
        tensors = self._tensors()
        ds = run_mod._variant_datasets(tensors, np.arange(10), seq_only=False)
        self.assertEqual(ds.static.shape[1], 6)

    def test_step_columns_map_to_the_two_newest_input_events(self):
        frame = pd.DataFrame({"seq_touch": [[1, 2, 3], [4, 5], [6]]})
        steps = run_mod._last_two_touchpoints(frame)
        self.assertEqual(steps["step2"].tolist(), [3, 5, 6])
        self.assertEqual(steps["step3"].tolist(), [2, 4, 0])
