"""Two things were measured here, and they did not agree with each other.

1. The neighbour adjacency build asked for all n neighbours and then dropped
   the self-match in a Python loop. That is a genuine algorithmic fix: the
   output is n-by-k whatever n is, and an n-by-n index array is 333x to 3333x
   more memory for the same answer.

2. Replacing the dense RBF affinity with a sparse k-NN graph was measured on
   well-separated blobs and looked like a free 3.3x speed-up with an identical
   partition (ARI 1.000). On the pipeline's own feature space it is not
   equivalent: best ARI 0.69 at k=10, falling to 0.31 at k=200.

So the adjacency is optimised and the affinity is not. The test below pins both
decisions, including the one that means *not* taking a speed-up.
"""

from __future__ import annotations

import numpy as np
from django.test import SimpleTestCase

from src.cjps_train.cluster import _fit_spectral
from src.cjps_train.config import TrainConfig
from src.pre_processing import _neighbour_indices_among_minority


def blob(n, dim=8, spread=1.0, seed=0):
    return np.random.default_rng(seed).normal(0.0, spread, (n, dim))


def segmented(n_segments, per_segment, dim=8, spread=1.4, seed=0):
    """Well-separated blobs. Convenient, but *not* the real feature space.

    Kept separate from the pipeline's own assembly on purpose: conclusions that
    hold on blobs did not survive contact with the real features, and mixing
    the two would hide that.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(0.0, 15.0, (n_segments, dim))
    return np.vstack(
        [rng.normal(c, spread, (per_segment, dim)) for c in centres]
    )


class AdjacencyTests(SimpleTestCase):
    def test_shape_is_bounded_by_k_not_by_n(self):
        """The memory guarantee, asserted as a shape because that is what it is.

        The previous version called ``kneighbors(n_neighbors=n)`` -- an n-by-n
        index array -- then discarded the self-match in a Python loop over every
        row. Asking for k + 1 bounds the result at n-by-k, so the 333x to 3333x
        memory reduction is structural rather than incidental.
        """
        for n in (50, 200, 800):
            with self.subTest(n=n):
                out = _neighbour_indices_among_minority(blob(n), k=5)
                self.assertEqual(out.shape, (n, 5))
                self.assertLess(out.shape[1], n)

    def test_no_row_is_its_own_neighbour(self):
        """The defect this replaced.

        ``kneighbors`` returns the query point first, so drawing from the raw
        result sometimes picked the point itself, giving ``diff = 0`` and a
        verbatim copy instead of an interpolation. On a 30/6 split that was 7 of
        24 synthetic samples.
        """
        X = blob(300, seed=1)
        out = _neighbour_indices_among_minority(X, k=5)
        self.assertFalse(np.any(out == np.arange(len(X))[:, None]))

    def test_neighbours_are_the_k_closest(self):
        """Distance order must survive the compaction.

        The self-match is pushed to the end with a stable argsort, so what
        remains has to still be sorted by distance.
        """
        from sklearn.neighbors import NearestNeighbors

        X = blob(120, seed=2)
        out = _neighbour_indices_among_minority(X, k=4)
        expected = NearestNeighbors(n_neighbors=6).fit(X).kneighbors(
            X, return_distance=False
        )
        for row in (0, 7, 55, 119):
            self.assertEqual(
                set(out[row].tolist()),
                set(expected[row][expected[row] != row][:4].tolist()),
                msg=f"row {row} got the wrong neighbours",
            )

    def test_duplicate_coordinates_do_not_break_compaction(self):
        """Ties at distance zero are where a non-stable sort would bite.

        With identical points several neighbours sit at distance 0 alongside the
        self-match, so an unstable compaction could drop a real neighbour and
        keep the self-match.
        """
        X = np.repeat(blob(20, seed=3), 4, axis=0)
        out = _neighbour_indices_among_minority(X, k=3)
        self.assertEqual(out.shape, (len(X), 3))
        self.assertFalse(np.any(out == np.arange(len(X))[:, None]))

    def test_k_larger_than_the_class_is_clamped(self):
        self.assertEqual(
            _neighbour_indices_among_minority(blob(6, seed=4), k=99).shape, (6, 5)
        )

    def test_single_row_yields_nothing(self):
        self.assertEqual(
            _neighbour_indices_among_minority(blob(1), k=3).shape, (1, 0)
        )


class AffinityDecisionTests(SimpleTestCase):
    """Pins the decision *not* to take a speed-up."""

    def test_dense_rbf_is_the_default(self):
        """Because the sparse graph is a different segmentation.

        Measured on the pipeline's own features: ARI 0.69 at k=10, falling to
        0.31 at k=200. No setting reaches the 0.90 floor. Shipping it would
        have been a modelling change presented as an optimisation.
        """
        self.assertEqual(TrainConfig().spectral_affinity, "rbf")

    def test_agreement_floor_is_set_below_which_we_warn(self):
        self.assertEqual(TrainConfig().spectral_agreement_floor, 0.90)

    def test_verification_is_off_by_default(self):
        """It fits twice, so it cannot be the production path.

        The dense affinity is quadratic; re-earning the time the alternative
        would have saved defeats the purpose of running the check at all.
        """
        self.assertFalse(TrainConfig().verify_spectral_against_dense)

    def test_unverified_reports_unknown_not_perfect(self):
        """None must not look like 1.0.

        An unverified substitution and a verified-identical one are different
        states, and the manifest has to be able to tell them apart.
        """
        X = segmented(3, 100, seed=5)
        _, labels, agreement = _fit_spectral(X, 3, TrainConfig(seed=42))
        self.assertIsNone(agreement)
        self.assertEqual(len(set(labels.tolist())), 3)

    def test_agreement_is_measured_when_asked(self):
        X = segmented(3, 120, seed=6)
        _, _, agreement = _fit_spectral(
            X, 3, TrainConfig(seed=42, verify_spectral_against_dense=True)
        )
        self.assertIsNotNone(agreement)
        self.assertGreaterEqual(agreement, -1.0)
        self.assertLessEqual(agreement, 1.0)

    def test_on_blobs_the_two_graphs_agree_exactly(self):
        """Why this looked free in the first place.

        On well-separated Gaussian blobs the sparse graph reproduces the dense
        partition exactly. Recording that is the point: it documents the
        measurement that produced the wrong conclusion, so the difference is
        attributable to the feature space rather than to noise.
        """
        for n_segments, k in ((3, 3), (4, 4)):
            with self.subTest(segments=n_segments):
                X = segmented(n_segments, 120, seed=n_segments)
                _, _, agreement = _fit_spectral(
                    X, k,
                    TrainConfig(
                        seed=42,
                        spectral_affinity="nearest_neighbors",
                        spectral_neighbors=10,
                        verify_spectral_against_dense=True,
                    ),
                )
                self.assertEqual(agreement, 1.0, msg=f"ARI {agreement} on blobs")

    def test_sparse_remains_available_as_a_documented_option(self):
        """The speed-up is one setting away, with its trade-off stated.

        Refusing it outright would be wrong too: 12x smaller artefacts and 3.3x
        faster fits are real, and a deployment that does not need to reproduce
        the original partition should be able to have them.
        """
        X = segmented(3, 100, seed=7)
        _, labels, _ = _fit_spectral(
            X, 3, TrainConfig(seed=42, spectral_affinity="nearest_neighbors")
        )
        self.assertEqual(len(set(labels.tolist())), 3)

    def test_dense_fit_is_reproducible(self):
        X = segmented(3, 120, seed=8)
        config = TrainConfig(seed=42, spectral_affinity="rbf")
        first = _fit_spectral(X, 3, config)[1]
        second = _fit_spectral(X, 3, config)[1]
        self.assertTrue(np.array_equal(first, second))


class DbscanParallelismTests(SimpleTestCase):
    def test_noise_gate_is_fitted_in_parallel(self):
        """n_jobs parallelises the neighbourhood search, which dominates.

        Measured 0.723 s -> 0.290 s on 8 000 rows. Fit-time only: the app's
        inference-time gate is a single distance comparison against the stored
        core samples, so the request path is unchanged.
        """
        from sklearn.cluster import DBSCAN

        from src.cjps_train.cluster import noise_for_eps

        X = blob(400, dim=5, spread=1.0, seed=9)
        choice = noise_for_eps(X, eps=1.5, min_samples=5)
        self.assertGreaterEqual(choice.n_clusters, 1)
        self.assertEqual(DBSCAN(eps=1.5, min_samples=5, n_jobs=-1).n_jobs, -1)
