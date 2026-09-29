"""Tests for model and resampler selection.

The original ran a benchmark, printed the table, and then fitted a hardcoded
``GradientBoostingClassifier`` against a hardcoded ``n_clusters=3``. These tests
check the opposite property: that the numbers reported actually determine the
artefact that gets written.
"""

from __future__ import annotations

import numpy as np
from django.test import SimpleTestCase
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

from src.cjps_train import select as sel
from src.cjps_train.config import TrainConfig
from src.cjps_train.evaluate import EvaluationError


def make_classification(n_majority: int = 90, n_minority: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0.0, 1.0, (n_majority, 5)),
                   rng.normal(3.0, 1.0, (n_minority, 5))])
    y = np.array([0] * n_majority + [1] * n_minority)
    return X, y


class _ConstantClassifier(BaseEstimator, ClassifierMixin):
    """Always predicts the majority class.

    Included to test that a well-behaved but useless model can lose to a real
    one. ``balanced_accuracy`` exists precisely to make that visible: plain
    accuracy would score it at 0.75.
    """

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


class _PerfectClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0] > 0


class RosterTests(SimpleTestCase):
    def test_every_candidate_carries_a_rationale(self):
        """A candidate nobody can justify should not be in the list.

        The manifest carries these notes, so a reader months later can see why
        a model was tried rather than having to guess.
        """
        for candidate in sel.default_candidates(0):
            with self.subTest(candidate=candidate.name):
                self.assertTrue(candidate.note.strip(), f"{candidate.name} has no note")

    def test_hinge_sgd_is_marked_as_unusable(self):
        """It cannot emit probabilities, so the app cannot render its output.

        Marking it lets the selection exclude it even if it happens to top the
        f1 table, which is a real possibility on a two-class fixture.
        """
        hinge = next(c for c in sel.default_candidates(0) if c.name == "SGD_hinge")
        self.assertFalse(hinge.supports_proba)
        self.assertIn("predict_proba", hinge.note)
        self.assertIsInstance(hinge.estimator, SGDClassifier)
        self.assertEqual(hinge.estimator.loss, "hinge")

    def test_every_candidate_that_needs_proba_has_it(self):
        for candidate in sel.default_candidates(0):
            if candidate.name == "SGD_hinge":
                continue
            with self.subTest(candidate=candidate.name):
                self.assertTrue(
                    candidate.supports_proba,
                    f"{candidate.name} is marked as unable to emit probabilities",
                )

    def test_svc_is_calibrated_for_probabilities(self):
        """SVC has no ``predict_proba`` unless it is calibrated.

        Left uncalibrated it would pass the benchmark on argmax metrics and then
        fail at inference with an ``AttributeError`` — the classic "trained fine,
        broke in production" outcome. ``SVC(probability=True)`` does supply them
        but is deprecated in scikit-learn 1.9 and removed in 1.11, so the
        explicit wrapper is used instead of a parameter with an expiry date.
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import SVC

        svc = next(c for c in sel.default_candidates(0) if c.name == "SVC_rbf")
        self.assertIsInstance(svc.estimator, CalibratedClassifierCV)
        # And the requirement is actually met, not just declared: fit it and
        # check the method the app calls really is there.
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=40, n_features=4, n_classes=2, random_state=0
        )
        svc.estimator.fit(X, y)
        self.assertTrue(hasattr(svc.estimator, "predict_proba"))
        probabilities = svc.estimator.predict_proba(X[:3])
        self.assertEqual(probabilities.shape, (3, 2))
        # The base SVC itself has none, which is why the wrapper is necessary.
        self.assertFalse(hasattr(SVC(), "predict_proba"))

    def test_mlp_uses_early_stopping(self):
        """The original ran it to 1000 iterations with warnings suppressed.

        A non-converged network then looked identical to a converged one,
        because the only signal was a warning nobody was allowed to see.
        """
        mlp = next(c for c in sel.default_candidates(0) if c.name == "MLP")
        self.assertTrue(mlp.estimator.early_stopping)
        self.assertLess(mlp.estimator.max_iter, 1000)

    def test_gradient_boosting_depth_is_reduced(self):
        """Depth 5 on a few thousand ordinal rows overfits.

        The original chose it by fiat; the reduction is the first thing the
        note has to justify.
        """
        gb = next(c for c in sel.default_candidates(0) if c.name == "GradientBoosting")
        self.assertEqual(gb.estimator.max_depth, 3)
        self.assertIn("max_depth", gb.note)

    def test_every_sampler_is_deterministic_for_a_given_seed(self):
        """Unseeded resampling makes a run unreproducible.

        Two independently constructed variants with the same run seed must
        produce byte-identical output, which is what makes a manifest enough to
        reproduce a run.
        """
        X, y = make_classification()
        first = dict(sel.sampler_variants(0, TrainConfig()))
        second = dict(sel.sampler_variants(0, TrainConfig()))

        for name, sampler in first.items():
            with self.subTest(sampler=name):
                if sampler is None:
                    continue
                a_X, a_y = sampler(X, y)
                b_X, b_y = second[name](X, y)
                self.assertTrue(np.array_equal(a_X, b_X))
                self.assertTrue(np.array_equal(a_y, b_y))

    def test_no_resampling_is_offered_as_a_baseline(self):
        """Without an unbalanced baseline, no resampler can be shown to help."""
        names = [n for n, _ in sel.sampler_variants(0, TrainConfig())]
        self.assertIn("none", names)


class SelectionMechanismTests(SimpleTestCase):
    def setUp(self):
        super().setUp()
        self.X, self.y = make_classification()
        self.config = TrainConfig(cv_splits=3, seed=0)

    def test_the_winner_is_the_highest_scoring_candidate(self):
        """The point of the whole module.

        The original's benchmark result had no bearing on the artefact, so this
        asserts the opposite: given a roster, the reported winner is the one
        whose score was highest.
        """
        roster = [
            sel.Candidate("weak", _ConstantClassifier(), "none"),
            sel.Candidate("strong", _PerfectClassifier(), "none"),
        ]
        selection = sel.select(self.X, self.y, self.config, candidates=roster,
                               include_sampler_search=False)
        self.assertEqual(selection.winner.name, "strong")
        self.assertEqual(selection.ranking[0][0], "strong")

    def test_balanced_accuracy_separates_a_useless_model(self):
        """A constant predictor is not 75% right in any useful sense.

        Plain accuracy on an imbalanced target rewards exactly this model, so
        the benchmark has to report something else as well.
        """
        roster = [
            sel.Candidate("constant", _ConstantClassifier(), "none"),
            sel.Candidate("real", LogisticRegression(max_iter=500), "none"),
        ]
        results = sel.select(
            self.X, self.y, self.config, candidates=roster,
            include_sampler_search=False,
        ).results
        by_name = {r.name: r for r in results}
        self.assertGreater(
            by_name["real"].metrics["balanced_accuracy"],
            by_name["constant"].metrics["balanced_accuracy"],
        )
        # And on plain accuracy the gap is much smaller — which is why accuracy
        # alone is not the selection criterion.
        self.assertGreater(
            by_name["real"].metrics["accuracy"],
            by_name["constant"].metrics["accuracy"],
        )

    def test_a_failure_cannot_be_selected(self):
        class Broken(BaseEstimator, ClassifierMixin):
            def fit(self, X, y):
                raise RuntimeError("cannot fit")

        roster = [
            sel.Candidate("broken", Broken(), "none"),
            sel.Candidate("real", LogisticRegression(max_iter=500), "none"),
        ]
        selection = sel.select(self.X, self.y, self.config, candidates=roster,
                               include_sampler_search=False)
        self.assertEqual(selection.winner.name, "real")
        self.assertTrue(selection.rejected)
        self.assertIn("broken", [n for n, _ in selection.rejected])

    def test_every_failure_is_recorded(self):
        class Broken(BaseEstimator, ClassifierMixin):
            def fit(self, X, y):
                raise RuntimeError("first")
                return self

        class Broken2(Broken):
            pass

        roster = [
            sel.Candidate("a", Broken(), "none"),
            sel.Candidate("b", Broken2(), "none"),
            sel.Candidate("real", LogisticRegression(max_iter=500), "none"),
        ]
        selection = sel.select(self.X, self.y, self.config, candidates=roster,
                               include_sampler_search=False)
        self.assertEqual(len(selection.rejected), 2)
        for name, error in selection.rejected:
            self.assertIn("first", error)
        # And they are not simply absent from the results table.
        self.assertEqual(len(selection.results), 3)

    def test_a_model_without_probabilities_is_not_selected(self):
        """Even if it tops the table, the app cannot render it.

        On a two-class fixture the argmax metrics are good enough for a
        hinge-loss model to win, and selecting it would produce an artefact that
        raises the first time somebody submits the form.
        """
        roster = [
            sel.Candidate("hinge", SGDClassifier(loss="hinge", max_iter=1000,
                                                tol=1e-3, random_state=0), "none",
                          supports_proba=False,
                          note="no probabilities"),
            sel.Candidate("real", LogisticRegression(max_iter=500), "none"),
        ]
        selection = sel.select(self.X, self.y, self.config, candidates=roster,
                               include_sampler_search=False)
        self.assertTrue(selection.winner.supports_proba)
        self.assertNotEqual(selection.winner.name, "hinge")

    def test_all_candidates_failing_raises(self):
        class Broken(BaseEstimator, ClassifierMixin):
            def fit(self, X, y):
                raise RuntimeError("nope")
                return self

        roster = [sel.Candidate("broken", Broken(), "none")]
        with self.assertRaises(EvaluationError):
            sel.select(self.X, self.y, self.config, candidates=roster,
                       include_sampler_search=False)

    def test_resampler_search_builds_a_full_grid(self):
        """Selection covers the (model, resampler) product, not one axis.

        SMOTE helps some models and not others on the same data, so fixing a
        resampler and varying the model would bake in an assumption the data
        was never asked about.
        """
        roster = [
            sel.Candidate("lr", LogisticRegression(max_iter=500), "none"),
            sel.Candidate("rf", RandomForestClassifier(n_estimators=10, random_state=0), "none"),
        ]
        samplers = [("none", None)]
        selection = sel.select(self.X, self.y, self.config, candidates=roster,
                               samplers=samplers, include_sampler_search=True)
        names = sorted(n for n, _ in selection.ranking)
        # The no-resampling row keeps the bare name, so one sampler over two
        # models is two rows, not four duplicated ones.
        self.assertEqual(len(selection.results), 2)
        self.assertEqual(names, ["lr", "rf"])

    def test_grid_covers_every_sampler(self):
        """A model is evaluated against each resampler, not just one."""
        roster = [sel.Candidate("lr", LogisticRegression(max_iter=500), "none")]
        samplers = [
            ("none", None),
            ("noop", lambda X, y: (X, y)),
        ]
        selection = sel.select(self.X, self.y, self.config, candidates=roster,
                               samplers=samplers, include_sampler_search=True)
        names = sorted(n for n, _ in selection.ranking)
        self.assertEqual(names, ["lr", "lr + noop"])

    def test_no_resampling_is_preferred_when_it_ties(self):
        """Synthesising points costs something and needs a clear win.

        A resampled candidate inside the fold-to-fold spread of the plain one
        is not distinguishable on this data, so the simpler option wins and the
        tie is logged rather than resolved by float comparison order.
        """
        roster = [
            sel.Candidate("only", LogisticRegression(max_iter=500), "none"),
        ]
        # A no-op sampler: produces the same data, so the two must tie exactly.
        identity = [("none", None), ("noop", lambda X, y: (X, y))]
        selection = sel.select(self.X, self.y, self.config, candidates=roster,
                               samplers=identity, include_sampler_search=True)
        scores = dict(selection.ranking)
        self.assertIn("only", scores)
        self.assertIn("only + noop", scores)
        self.assertAlmostEqual(scores["only"], scores["only + noop"], places=9)
        # The unsampled option is chosen, so the shipped model trains on the
        # real data.
        self.assertEqual(selection.winner.sampler_name, "none")
        self.assertIsNone(selection.winner.sampler)

    def test_selection_is_reproducible(self):
        roster = [
            sel.Candidate("lr", LogisticRegression(max_iter=500), "none"),
            sel.Candidate("rf", RandomForestClassifier(n_estimators=10, random_state=0), "none"),
        ]
        first = sel.select(self.X, self.y, self.config, candidates=roster,
                           include_sampler_search=False)
        second = sel.select(self.X, self.y, self.config, candidates=roster,
                            include_sampler_search=False)
        self.assertEqual(first.winner.name, second.winner.name)
        self.assertEqual(first.ranking, second.ranking)

    def test_manifest_records_the_full_grid(self):
        roster = [
            sel.Candidate("lr", LogisticRegression(max_iter=500), "none", note="linear"),
        ]
        selection = sel.select(self.X, self.y, self.config, candidates=roster,
                               include_sampler_search=False)
        payload = selection.as_dict()
        self.assertEqual(payload["winner"], "lr")
        self.assertEqual(payload["winner_note"], "linear")
        self.assertEqual(len(payload["ranking"]), 1)
        self.assertIn("fold_scores", payload["candidates"][0])
        self.assertIn("fold_spread", payload["candidates"][0])

    def test_winner_estimator_is_the_configured_one(self):
        """The selected object must be the one that gets fitted, not a copy."""
        estimator = LogisticRegression(max_iter=500)
        roster = [sel.Candidate("lr", estimator, "none")]
        selection = sel.select(self.X, self.y, self.config, candidates=roster,
                               include_sampler_search=False)
        self.assertIs(selection.winner.estimator, estimator)

    def test_winner_name_appears_in_its_own_ranking(self):
        """The manifest names the winner; the table has to contain it.

        An earlier version filed the no-resampling row under the bare candidate
        name but reconstructed the winner through ``with_sampler``, which
        appended a suffix. The manifest then recorded a winner — "lr + none" —
        that appears nowhere in the ranking it was chosen from, and the gap
        between the CV score and the holdout could not be computed.
        """
        roster = [sel.Candidate("lr", LogisticRegression(max_iter=500), "none")]
        samplers = [("none", None), ("noop", lambda X, y: (X, y))]
        selection = sel.select(self.X, self.y, self.config, candidates=roster,
                               samplers=samplers, include_sampler_search=True)
        ranked = [name for name, _ in selection.ranking]
        self.assertIn(selection.winner.name, ranked)
        result_names = [r.name for r in selection.results]
        self.assertIn(selection.winner.name, result_names)

    def test_candidate_name_matches_its_key(self):
        """``with_sampler(None, ...)`` must not rename the candidate."""
        base = sel.Candidate("lr", LogisticRegression(), "none")
        self.assertIs(base.with_sampler(None, "none"), base)
        self.assertEqual(base.with_sampler(None, "none").name, "lr")
        self.assertEqual(base.with_sampler(lambda X, y: (X, y), "SMOTE").name, "lr + SMOTE")
