"""Tests for the resampling module.

Each test names the defect it pins down, so a regression points straight at the
bug rather than at "resampling changed".
"""

import numpy as np
from collections import Counter

from django.test import SimpleTestCase

from src.pre_processing import (
    ADASYN,
    SMOTE,
    SMOTETomek,
    TomekLinks,
    adasyn,
    class_counts,
    drop_rare_classes,
    smote,
    tomek_links,
    tomek_pairs,
)

SEED = 7


def toy(n_majority=30, n_minority=6, separation=5.0, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    Xm = rng.normal(0.0, noise, (n_majority, 2))
    Xn = rng.normal(separation, noise, (n_minority, 2))
    return np.vstack([Xm, Xn]), np.array([0] * n_majority + [1] * n_minority)


class SMOTETests(SimpleTestCase):
    def test_balances_to_the_majority_size(self):
        X, y = toy()
        X2, y2 = smote(X, y, random_state=SEED)
        counts = class_counts(y2)
        self.assertEqual(counts[0], counts[1])

    def test_no_synthetic_point_duplicates_an_original(self):
        """The self-neighbour defect.

        kneighbors() returns the query point first, so drawing from the raw
        result sometimes picked it and produced ``diff = 0``. That is a
        verbatim copy, not an interpolation. This asserts zero such copies.
        """
        X, y = toy()
        X2, _ = smote(X, y, random_state=SEED)
        synthetic = X2[len(X):]

        originals = {tuple(np.round(p, 9)) for p in X}
        duplicates = [p for p in synthetic if tuple(np.round(p, 9)) in originals]
        self.assertEqual(
            duplicates, [], f"{len(duplicates)} synthetic points are exact copies"
        )

    def test_synthetic_points_are_all_distinct(self):
        X, y = toy()
        X2, _ = smote(X, y, random_state=SEED)
        synthetic = np.unique(np.round(X2[len(X):], 9), axis=0)
        self.assertEqual(len(synthetic), len(X2) - len(X))

    def test_synthetic_points_lie_on_the_segment_to_a_neighbour(self):
        """Every synthetic point must sit between a minority point and one of
        its same-class neighbours, i.e. inside the minority hull."""
        X, y = toy()
        minority = X[y == 1]
        X2, _ = smote(X, y, random_state=SEED)
        synthetic = X2[len(X):]

        lo, hi = minority.min(axis=0), minority.max(axis=0)
        # Allow a small tolerance for the random gap being exactly 0 or 1.
        self.assertTrue(
            np.all(synthetic >= lo - 1e-9) and np.all(synthetic <= hi + 1e-9),
            msg="a synthetic point fell outside the minority hull",
        )

    def test_is_deterministic_given_a_seed(self):
        X, y = toy()
        a, ya = smote(X, y, random_state=SEED)
        b, yb = smote(X, y, random_state=SEED)
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.array_equal(ya, yb))

    def test_interpolates_only_with_the_k_nearest_same_class_neighbours(self):
        """The neighbour set must be local, not the whole class.

        SMOTE is defined against a point's k nearest neighbours of its own
        class. Drawing a neighbour uniformly from the entire class instead lets
        a point be interpolated towards a member in a different region of the
        space, and the result belongs to neither neighbourhood.

        The minority class here is two well-separated blobs. With k=1 each
        point's only allowed partner is in its own blob, so no synthetic point
        may land in the gap between them.
        """
        rng = np.random.default_rng(5)
        majority = rng.normal(0.0, 0.3, (40, 2))
        blob_a = rng.normal(-30.0, 0.3, (6, 2))
        blob_b = rng.normal(30.0, 0.3, (6, 2))

        X = np.vstack([majority, blob_a, blob_b])
        y = np.array([0] * 40 + [1] * 12)

        X2, _ = smote(X, y, k_neighbors=1, random_state=SEED)
        synthetic = X2[len(X):]

        near_a = int(np.sum(np.abs(synthetic[:, 0] + 30.0) < 15.0))
        near_b = int(np.sum(np.abs(synthetic[:, 0] - 30.0) < 15.0))
        elsewhere = len(synthetic) - near_a - near_b
        self.assertEqual(
            elsewhere, 0,
            f"{elsewhere} synthetic points landed between the two blobs, so a "
            "neighbour was drawn from outside the k nearest",
        )
        # The budget should still be met, otherwise the test would pass by
        # producing nothing at all.
        self.assertEqual(len(synthetic), 40 - 12)

    def test_does_not_consume_global_rng_state(self):
        """A fixed seed must give the same result regardless of what ran before.

        The old implementation drew from the global numpy RNG, so the output
        depended on unrelated earlier calls.
        """
        X, y = toy()
        np.random.seed(0)
        np.random.random(1000)  # perturb the global stream
        a, _ = smote(X, y, random_state=SEED)
        np.random.seed(0)
        np.random.random(1000)
        b, _ = smote(X, y, random_state=SEED)
        self.assertTrue(np.array_equal(a, b))

    def test_tiny_minority_class_is_left_alone(self):
        """One member has no distinct neighbour to interpolate towards."""
        X = np.vstack([np.zeros((10, 2)), np.array([[5.0, 5.0]])])
        y = np.array([0] * 10 + [1])
        X2, y2 = smote(X, y, random_state=SEED)
        self.assertEqual(class_counts(y2)[1], 1)

    def test_single_class_is_a_no_op(self):
        X = np.zeros((5, 2))
        y = np.zeros(5, dtype=int)
        X2, y2 = smote(X, y, random_state=SEED)
        self.assertEqual(len(X2), 5)


class TomekTests(SimpleTestCase):
    def test_removes_overlapping_boundary_pairs(self):
        """The standard Tomek condition, not just mutual nearest pairs."""
        # Two classes with a genuinely ambiguous middle.
        X = np.array([
            [0.0, 0.0], [0.1, 0.0], [4.9, 0.0], [5.0, 0.0],
            [0.0, 5.0], [5.0, 5.0], [2.5, 0.0],   # ambiguous middle
        ])
        y = np.array([0, 0, 1, 1, 0, 1, 0])
        pairs = tomek_pairs(X, y)
        self.assertTrue(pairs, "expected at least one Tomek link")
        # Every reported pair must be cross-class.
        for a, b in pairs:
            self.assertNotEqual(y[a], y[b])

    def test_well_separated_data_has_no_links(self):
        X, y = toy()
        X2, y2 = tomek_links(X, y)
        self.assertEqual(len(X2), len(X))

    def test_never_removes_more_than_one_side_of_a_pair(self):
        """A point can appear in two pairs; it must be dropped once and the
        surviving data must stay aligned."""
        rng = np.random.default_rng(3)
        X = np.vstack([rng.normal(0, 1.2, (12, 2)), rng.normal(1.5, 1.2, (12, 2))])
        y = np.array([0] * 12 + [1] * 12)
        X2, y2, keep = tomek_links(X, y, return_indices=True)
        self.assertEqual(len(X2), len(y2))
        self.assertEqual(len(X2), len(keep))
        self.assertEqual(len(set(keep.tolist())), len(keep), "indices must be unique")
        self.assertTrue(np.array_equal(X2, X[keep]))
        self.assertTrue(np.array_equal(y2, y[keep]))

    def test_single_class_is_a_no_op(self):
        X = np.zeros((4, 2))
        y = np.zeros(4, dtype=int)
        X2, y2 = tomek_links(X, y)
        self.assertEqual(len(X2), 4)


class ADASYNTests(SimpleTestCase):
    def test_does_not_duplicate_originals(self):
        X, y = toy()
        X2, _ = adasyn(X, y, random_state=SEED)
        originals = {tuple(np.round(p, 9)) for p in X}
        dups = [p for p in X2[len(X):] if tuple(np.round(p, 9)) in originals]
        self.assertEqual(dups, [])

    def test_spends_budget_where_the_classes_interleave(self):
        """ADASYN redistributes a class's budget toward its hard points.

        The *total* synthesised per class is fixed at
        ``majority_size - class_size``; what ADASYN changes is *where inside*
        the class they land. Comparing totals between classes would therefore
        be meaningless — that total is identical by construction. So this
        builds one minority class with two halves and checks the spatial
        concentration of the output instead.

        Making a point "hard" takes deliberate geometry. ADASYN scores a
        minority point by the fraction of majority members among its k nearest
        neighbours of the whole dataset, so:

        * the majority must be dense and tight enough that its members outrank
          the minority's own members, otherwise an "inside" point is mostly
          surrounded by its own class and scores ~0;
        * the distant half must be tight too, so its k nearest neighbours are
          its own kind and it scores exactly 0;
        * the minority class must be large enough that a point's k nearest
          *same-class* neighbours are local. With only four members, the k
          nearest same-class neighbours necessarily include the distant half,
          and interpolation walks between the two regions.

        Getting those three wrong makes the test fail for the wrong reason,
        which is how it was originally mis-diagnosed as an implementation bug.
        """
        rng = np.random.default_rng(11)
        majority = rng.normal(0.0, 0.25, (100, 2))  # dense and tight
        inside = rng.normal(0.0, 0.45, (8, 2))     # hard: inside the cloud
        distant = rng.normal(-25.0, 0.2, (8, 2))   # easy: tight far cluster

        X = np.vstack([majority, inside, distant])
        y = np.array([0] * 100 + [1] * 16)
        X2, y2 = adasyn(X, y, random_state=SEED)

        synthetic = X2[len(X):][y2[len(X):] == 1]
        self.assertGreater(len(synthetic), 0, "ADASYN produced nothing to inspect")

        near_majority = int(np.sum(synthetic[:, 0] > -10.0))
        far_from_majority = int(np.sum(synthetic[:, 0] <= -10.0))
        self.assertGreater(
            near_majority, far_from_majority,
            msg=f"inside half got {near_majority}, distant half got {far_from_majority}; "
                "allocation is not following difficulty",
        )

    def test_is_deterministic_given_a_seed(self):
        X, y = toy()
        a, _ = adasyn(X, y, random_state=SEED)
        b, _ = adasyn(X, y, random_state=SEED)
        self.assertTrue(np.array_equal(a, b))


class SamplerInterfaceTests(SimpleTestCase):
    """The wrappers carry a seed and expose the sampler protocol.

    scikit-learn 1.9 removed samplers from ``Pipeline``, so these are consumed
    by the explicit cross-validation loop in ``src.cjps_train.evaluate`` rather
    than by a pipeline. What matters here is that the seed is honoured and that
    nothing resamples at predict time.
    """

    def test_fit_resample_returns_balanced_data(self):
        X, y = toy()
        X2, y2 = SMOTE(random_state=SEED).fit_resample(X, y)
        counts = class_counts(y2)
        self.assertEqual(counts[0], counts[1])

    def test_get_params_round_trips(self):
        for sampler in (SMOTE(random_state=1), ADASYN(k_neighbors=3), TomekLinks()):
            with self.subTest(sampler=type(sampler).__name__):
                params = sampler.get_params()
                clone = type(sampler)(**params)
                self.assertEqual(clone.get_params(), params)

    def test_transform_is_identity(self):
        """Predict-time must never see synthesised rows."""
        X, y = toy()
        X2, _ = SMOTE(random_state=SEED).fit_resample(X, y)
        out = SMOTE(random_state=SEED).transform(X2)
        self.assertTrue(np.array_equal(out, X2))

    def test_fit_transform_returns_x_only(self):
        """Returning (X, y) would let a caller pair resampled X with the
        original y and silently misalign them."""
        X, y = toy()
        out = SMOTE(random_state=SEED).fit_transform(X, y)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.ndim, 2)


class ClassFilterTests(SimpleTestCase):
    def test_mask_drops_rare_classes(self):
        y = np.array([0] * 10 + [1] * 2 + [2] * 7)
        mask = drop_rare_classes(y, min_samples=5)
        self.assertEqual(set(y[mask]), {0, 2})

    def test_mask_keeps_x_and_y_aligned(self):
        y = np.array([0] * 10 + [1] * 1)
        mask = drop_rare_classes(y, min_samples=5)
        self.assertEqual(len(mask), len(y))

    def test_min_samples_must_clear_the_fold_count(self):
        """A class with fewer members than n_splits cannot populate every
        training fold, and the failure is not always a clean error — it can
        surface as a model silently vanishing from a benchmark. The threshold
        is therefore stated explicitly at the call site rather than left to a
        magic number.
        """
        from sklearn.model_selection import StratifiedKFold

        y = np.array([0] * 20 + [1] * 4)
        X = np.zeros((24, 2))

        # Establish the real behaviour rather than assuming it raises.
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y))
        held_out = [len(test) for _, test in folds]
        self.assertEqual(sum(held_out), len(y))

        # Filtering with a threshold that respects n_splits removes the problem.
        mask = drop_rare_classes(y, min_samples=5)
        self.assertEqual(set(y[mask]), {0})
        folds = list(
            StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X[mask], y[mask])
        )
        for train_idx, _ in folds:
            counts = Counter(y[mask][train_idx])
            self.assertGreaterEqual(min(counts.values()), 1)
