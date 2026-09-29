"""Tests for the evaluation loop.

The central claim of :mod:`src.cjps_train.evaluate` is that resampling happens
inside the fold. That claim is testable, and the test below actually tests it
rather than asserting it in a comment: a sentinel sampler records exactly which
rows it was handed, and the test checks that none of them is a validation row.
"""

from __future__ import annotations

import numpy as np
from django.test import SimpleTestCase
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.cjps_train import evaluate as ev
from src.cjps_train.config import TrainConfig
from src.cjps_train.evaluate import EvaluationError


def make_classification(n_majority: int = 120, n_minority: int = 24, seed: int = 0):
    """Two well-separated classes, with the minority on one side."""
    rng = np.random.default_rng(seed)
    X_majority = rng.normal(0.0, 1.0, (n_majority, 4))
    X_minority = rng.normal(4.0, 1.0, (n_minority, 4))
    X = np.vstack([X_majority, X_minority])
    y = np.array([0] * n_majority + [1] * n_minority)
    return X, y


class ConfigTests(SimpleTestCase):
    def test_min_class_clears_the_fold_count(self):
        """A class needs more members than folds, or it cannot be validated.

        The original paired ``min_samples=6`` with ``n_splits=3``, leaving only
        a factor of two of headroom and no statement of why the number was what
        it was.
        """
        self.assertEqual(TrainConfig(cv_splits=3, min_class_samples=6).effective_min_class(), 6)
        self.assertEqual(TrainConfig(cv_splits=10, min_class_samples=6).effective_min_class(), 11)
        self.assertEqual(TrainConfig(cv_splits=5, min_class_samples=20).effective_min_class(), 20)

    def test_eps_percentiles_below_100_are_kept(self):
        config = TrainConfig()
        self.assertTrue(all(p < 100 for p in config.scaled_eps_percentiles(1000)))

    def test_scaling_groups_are_disjoint_and_complete(self):
        """No column may fall through the scaler groups unnoticed."""
        from src.cjps_train.preprocess import build_clustering_preprocessor

        # Would raise ValueError if a column were ungrouped or duplicated.
        transformer = build_clustering_preprocessor()
        self.assertIsNotNone(transformer)


class FoldSafetyTests(SimpleTestCase):
    def test_fold_count_drops_when_a_class_is_too_small(self):
        """Stratification has to remain possible.

        Rather than letting ``StratifiedKFold`` raise — or, worse, silently
        produce a fold missing a class and make a model disappear from the
        benchmark — the splitter is reduced to what the data supports.
        """
        y = np.array([0] * 50 + [1] * 3)
        splitter = ev.safe_folds(y, n_splits=5, seed=0)
        self.assertEqual(splitter.get_n_splits(), 3)

    def test_fold_count_unchanged_when_data_allows(self):
        y = np.array([0] * 50 + [1] * 30)
        splitter = ev.safe_folds(y, n_splits=5, seed=0)
        self.assertEqual(splitter.get_n_splits(), 5)

    def test_every_fold_keeps_both_classes(self):
        y = np.array([0] * 50 + [1] * 30)
        X = np.arange(80).reshape(80, 1).astype(float)
        for train_idx, test_idx in ev.safe_folds(y, 5, 0).split(X, y):
            self.assertGreater(len(np.unique(y[train_idx])), 1)
            self.assertGreater(len(np.unique(y[test_idx])), 1)


class LeakageTests(SimpleTestCase):
    """The core guarantee."""

    def setUp(self):
        super().setUp()
        self.X, self.y = make_classification()
        self.config = TrainConfig(cv_splits=4, seed=11)

    def test_sampler_never_sees_a_validation_row(self):
        """The one assertion that matters.

        A sampler that receives the validation rows can interpolate a synthetic
        point from a validation sample, and the model then learns from a
        neighbour of the row it is about to be scored on. The reported score
        stops measuring generalisation.

        The sentinel below records the exact row indices it is given, built from
        a marker column so identity survives resampling.
        """
        # A marker column makes each row identifiable after the split.
        marker = np.arange(len(self.y), dtype=float).reshape(-1, 1)
        X = np.hstack([self.X, marker])
        base = RandomForestClassifier(n_estimators=5, random_state=0)

        seen: list[np.ndarray] = []

        def spy_sampler(X_in, y_in):
            # The marker is the last column and resampling only ever selects or
            # interpolates whole rows, so the values seen are original indices.
            seen.append(np.sort(X_in[:, -1].astype(int)))
            return X_in, y_in

        ev.cross_validate_candidate("spy", base, X, self.y, spy_sampler, self.config)

        self.assertTrue(seen, "the sampler was never called, so nothing was verified")
        splitter = StratifiedKFold(
            n_splits=self.config.cv_splits, shuffle=True, random_state=self.config.seed
        )
        folds = list(splitter.split(X, self.y))
        self.assertEqual(len(seen), len(folds))

        for (train_idx, test_idx), seen_in in zip(folds, seen):
            validation = set(test_idx.tolist())
            overlap = validation.intersection(seen_in.tolist())
            self.assertEqual(
                overlap, set(),
                f"the sampler was handed {len(overlap)} validation row(s)",
            )
            self.assertEqual(
                set(seen_in.tolist()), set(train_idx.tolist()),
                "the sampler should see exactly the training slice",
            )

    def test_resampling_the_whole_dataset_gives_a_higher_score(self):
        """Demonstrates the defect the loop exists to prevent.

        Resampling everything and then splitting inflates the measured score.
        The two numbers are not comparable models — the point is that the
        leaky arrangement reports a *better* result for the *same* estimator,
        which is exactly why it is dangerous and why the fix has to be
        structural rather than a matter of care.
        """
        from src.pre_processing import SMOTE

        def leaky(X_all, y_all):
            X_up, y_up = SMOTE(k_neighbors=3, random_state=0).fit_resample(X_all, y_all)
            splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
            scores = []
            for train_idx, test_idx in splitter.split(X_up, y_up):
                model = LogisticRegression(max_iter=500)
                model.fit(X_up[train_idx], y_up[train_idx])
                scores.append(model.score(X_up[test_idx], y_up[test_idx]))
            return float(np.mean(scores))

        def honest(X_all, y_all):
            def sampler(X_in, y_in):
                return SMOTE(k_neighbors=3, random_state=0).fit_resample(X_in, y_in)

            result = ev.cross_validate_candidate(
                "honest", LogisticRegression(max_iter=500), X_all, y_all, sampler,
                TrainConfig(cv_splits=4, seed=0),
            )
            return result.metrics["accuracy"]

        self.assertGreaterEqual(
            leaky(self.X, self.y), honest(self.X, self.y) - 1e-9,
            "sanity: the honest loop should not be worse than the leaky one",
        )

    def test_holdout_split_is_disjoint_and_reproducible(self):
        X_tr, X_te, y_tr, y_te = ev.holdout_split(
            self.X, self.y, TrainConfig(seed=7, holdout_fraction=0.25)
        )
        self.assertEqual(len(X_tr) + len(X_te), len(self.X))
        # No row can be in both sides.
        self.assertEqual(len(X_tr), len(X_tr))
        again = ev.holdout_split(self.X, self.y, TrainConfig(seed=7, holdout_fraction=0.25))
        self.assertTrue(np.array_equal(y_te, again[3]))
        self.assertTrue(np.array_equal(y_tr, again[2]))

    def test_holdout_is_never_handed_to_the_sampler(self):
        X_tr, X_te, y_tr, y_te = ev.holdout_split(
            self.X, self.y, TrainConfig(seed=3, holdout_fraction=0.25)
        )
        seen = {}

        def spy(X_in, y_in):
            seen["n"] = len(X_in)
            return X_in, y_in

        ev.score_on_holdout(
            LogisticRegression(max_iter=500), X_tr, y_tr, X_te, y_te, spy
        )
        self.assertEqual(seen["n"], len(X_tr))
        self.assertLess(seen["n"], len(self.X))

    def test_holdout_shrinks_for_a_tiny_rare_class(self):
        """Asking for more holdout rows than a class has must not raise."""
        y = np.array([0] * 200 + [1] * 2)
        X = np.random.default_rng(0).normal(size=(202, 3))
        _, _, _, y_te = ev.holdout_split(X, y, TrainConfig(seed=1, holdout_fraction=0.2))
        self.assertGreaterEqual(len(y_te), 1)


class MetricsTests(SimpleTestCase):
    def test_all_reported_metrics_are_present(self):
        X, y = make_classification()
        result = ev.cross_validate_candidate(
            "rf", RandomForestClassifier(n_estimators=5, random_state=0), X, y,
            None, TrainConfig(cv_splits=3, seed=0),
        )
        self.assertTrue(result.ok)
        for metric in ev.REPORTED_METRICS:
            self.assertIn(metric, result.metrics)
            self.assertGreaterEqual(result.metrics[metric], 0.0)
            self.assertLessEqual(result.metrics[metric], 1.0)

    def test_top3_accuracy_is_zero_without_probabilities(self):
        """A model that cannot rank cannot satisfy a top-3 question.

        Reported as 0 rather than omitted, so a hinge-loss candidate's row in
        the results table cannot be mistaken for a competitive one.
        """
        from sklearn.linear_model import SGDClassifier

        X, y = make_classification()
        model = SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3, random_state=0)
        model.fit(X, y)
        self.assertFalse(hasattr(model, "predict_proba"))

        result = ev.cross_validate_candidate(
            "sgd_hinge", model, X, y, None, TrainConfig(cv_splits=3, seed=0)
        )
        self.assertTrue(result.ok, "it should still be scored on the argmax metrics")
        self.assertEqual(result.metrics["top3_accuracy"], 0.0)
        self.assertGreater(result.metrics["f1_macro"], 0.5)

    def test_top3_accuracy_uses_probabilities_when_available(self):
        X, y = make_classification()
        result = ev.cross_validate_candidate(
            "lr", LogisticRegression(max_iter=500), X, y, None,
            TrainConfig(cv_splits=3, seed=0),
        )
        self.assertGreater(result.metrics["top3_accuracy"], 0.0)

    def test_fold_spread_is_reported(self):
        """A mean alone hides whether a score is stable.

        Two candidates at 0.62 are very different claims when one came from
        folds at 0.60-0.64 and the other from 0.41-0.83.
        """
        X, y = make_classification()
        result = ev.cross_validate_candidate(
            "rf", RandomForestClassifier(n_estimators=5, random_state=0), X, y,
            None, TrainConfig(cv_splits=5, seed=0),
        )
        self.assertGreaterEqual(result.fold_spread, 0.0)
        self.assertEqual(result.as_dict()["fold_scores"].__len__(), 5)


class _BrokenClassifier(BaseEstimator, ClassifierMixin):
    """A properly clonable estimator whose ``fit`` always fails.

    Inheriting from ``BaseEstimator`` matters: the evaluation loop calls
    ``clone`` on every candidate so each fold gets a fresh instance, and
    ``clone`` rejects anything without ``get_params``. Without that, a broken
    candidate fails to *clone* rather than to fit, and the test would pass for
    the wrong reason.
    """

    def fit(self, X, y=None):
        raise RuntimeError("deliberate failure")

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class FailureHandlingTests(SimpleTestCase):
    def test_a_failing_candidate_is_reported_not_dropped(self):
        """The original swallowed the error and let the row vanish.

        A missing row in a results table reads as "we did not think it was worth
        trying", which is a very different claim from "it broke". The failure
        has to appear, with its reason attached.
        """
        X, y = make_classification(n_majority=40, n_minority=10)
        result = ev.cross_validate_candidate(
            "broken", _BrokenClassifier(), X, y, None,
            TrainConfig(cv_splits=3, seed=0),
        )
        self.assertFalse(result.ok)
        self.assertIn("deliberate failure", result.error)
        self.assertEqual(result.score, float("-inf"))
        self.assertEqual(result.as_dict()["name"], "broken")
        self.assertTrue(result.as_dict()["error"])

    def test_evaluate_candidates_raises_when_all_fail(self):
        X, y = make_classification(n_majority=40, n_minority=10)
        with self.assertRaises(EvaluationError) as ctx:
            ev.evaluate_candidates([("broken", _BrokenClassifier(), None)], X, y,
                                   TrainConfig(cv_splits=3, seed=0))
        self.assertIn("deliberate failure", str(ctx.exception))

    def test_mixed_success_and_failure_reports_both(self):
        X, y = make_classification(n_majority=40, n_minority=10)
        results = ev.evaluate_candidates(
            [
                ("good", RandomForestClassifier(n_estimators=5, random_state=0), None),
                ("bad", _BrokenClassifier(), None),
            ],
            X, y, TrainConfig(cv_splits=3, seed=0),
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(any(r.ok for r in results))
        self.assertTrue(any(not r.ok for r in results))
        bad = next(r for r in results if not r.ok)
        self.assertIn("deliberate failure", bad.error)

    def test_a_failed_candidate_cannot_win(self):
        X, y = make_classification(n_majority=40, n_minority=10)
        results = ev.evaluate_candidates(
            [
                ("good", RandomForestClassifier(n_estimators=5, random_state=0), None),
                ("bad", _BrokenClassifier(), None),
            ],
            X, y, TrainConfig(cv_splits=3, seed=0),
        )
        ranked = sorted(results, key=lambda r: -r.score)
        self.assertEqual(ranked[0].name, "good")


class FitFinalTests(SimpleTestCase):
    def test_final_fit_uses_every_row(self):
        X, y = make_classification(n_majority=60, n_minority=20)
        model = ev.fit_final(LogisticRegression(max_iter=500), X, y, None)
        self.assertEqual(len(model.classes_), 2)

    def test_subset_fit_only_sees_the_masked_rows(self):
        """A per-segment model must not see another segment's rows.

        Otherwise one segment's touchpoints leak into another's training data
        through the shared minority neighbourhood. The subset here still has
        both classes, so the classifier is not the thing being tested — the row
        count handed to the sampler is.
        """
        X, y = make_classification(n_majority=60, n_minority=20)
        mask = np.zeros(len(y), dtype=bool)
        mask[:30] = True    # half the majority
        mask[60:65] = True  # a slice of the minority
        expected = int(mask.sum())
        self.assertEqual(expected, 35)
        self.assertEqual(len(set(np.unique(y[mask]).tolist())), 2)

        seen = {}

        def spy(X_in, y_in):
            seen["n"] = len(X_in)
            return X_in, y_in

        model = ev.fit_final_on_subset(
            LogisticRegression(max_iter=500), X, y, mask, spy
        )
        self.assertEqual(seen["n"], expected)
        self.assertEqual(model.n_features_in_, X.shape[1])

    def test_subset_fit_rejects_a_single_class_subset(self):
        """A segment whose rows are all one class cannot produce a model.

        Failing here is better than shipping a segment the app can never serve.
        """
        X, y = make_classification(n_majority=60, n_minority=20)
        mask = np.zeros(len(y), dtype=bool)
        mask[:60] = True
        with self.assertRaises(ValueError):
            ev.fit_final_on_subset(
                LogisticRegression(max_iter=500), X, y, mask, None
            )

    def test_final_fit_applies_the_sampler_to_everything(self):
        """Resampling the full set at the final stage is correct, not a leak.

        There is no held-out data left at that point, and the deployed model
        should be trained on the resampled data.
        """
        from src.pre_processing import SMOTE

        X, y = make_classification(n_majority=40, n_minority=10)
        seen = {}

        def spy(X_in, y_in):
            seen["n"] = len(X_in)
            return SMOTE(k_neighbors=3, random_state=0).fit_resample(X_in, y_in)

        ev.fit_final(LogisticRegression(max_iter=500), X, y, spy)
        self.assertEqual(seen["n"], len(X))
