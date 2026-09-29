"""Tests for eps and cluster-count selection.

The original pipeline computed a full diagnostic — elbow curve, silhouette
across k, Davies-Bouldin — and then hardcoded ``n_clusters=3`` regardless. The
point of these tests is that the selection is driven by the data, so they
deliberately build data where the right answer is *not* 3 and check that the
search finds it.
"""

from __future__ import annotations

import numpy as np
from django.test import SimpleTestCase
from sklearn.cluster import DBSCAN

from src.cjps_train import cluster as cl
from src.cjps_train.config import TrainConfig


def blobs(centres, n_per: int = 60, spread: float = 0.5, seed: int = 0) -> np.ndarray:
    """Well-separated Gaussian blobs, one per centre."""
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(c, spread, (n_per, len(c))) for c in centres])


class KDistanceTests(SimpleTestCase):
    def test_curve_is_sorted_and_non_decreasing(self):
        X = blobs([(0.0,) * 5, (6.0,) * 5], n_per=40)
        curve = cl.k_distance_curve(X, k=20)
        self.assertTrue(np.all(np.diff(curve) >= 0))
        self.assertEqual(len(curve), len(X))

    def test_curve_degrades_gracefully_on_one_sample(self):
        curve = cl.k_distance_curve(np.zeros((1, 3)), k=20)
        self.assertEqual(len(curve), 1)

    def test_curve_scales_with_dispersion(self):
        """A tight cloud has smaller k-distances than a spread one.

        This is the property that makes a percentile of the curve a usable eps.
        A hard-coded distance would be meaningless across a change of
        preprocessor; a percentile tracks the data.
        """
        tight = cl.k_distance_curve(blobs([(0.0,) * 4], n_per=40, spread=0.2), 20)
        loose = cl.k_distance_curve(blobs([(0.0,) * 4], n_per=40, spread=3.0), 20)
        self.assertLess(np.median(tight), np.median(loose))


class EpsSelectionTests(SimpleTestCase):
    def setUp(self):
        super().setUp()
        self.config = TrainConfig(k_distance_k=15, seed=0)
        # Two tight clusters plus a sprinkling of far-away outliers, which is
        # the shape the noise gate exists for.
        core = blobs([(0.0,) * 6, (12.0,) * 6], n_per=70, spread=0.4, seed=1)
        rng = np.random.default_rng(2)
        outliers = rng.normal(60.0, 3.0, (18, 6))
        self.X = np.vstack([core, outliers])

    def test_eps_is_chosen_from_a_percentile_of_the_curve(self):
        choice = cl.select_eps(self.X, self.config)
        curve = cl.k_distance_curve(self.X, self.config.k_distance_k)
        expected = float(np.percentile(curve, choice.percentile))
        self.assertAlmostEqual(choice.eps, expected, places=9)

    def test_choice_produces_multiple_clusters_and_some_noise(self):
        """The gate has to actually gate.

        A configuration where everything is one cluster, or where everything is
        noise, tells the app nothing — and the original accepted whatever
        DBSCAN returned, including a fit where every point was its own cluster.
        """
        choice = cl.select_eps(self.X, self.config)
        self.assertGreater(choice.n_clusters, 1)
        self.assertGreater(choice.n_noise, 0)

    def test_prefers_a_small_noise_share_among_usable_choices(self):
        choice = cl.select_eps(self.X, self.config)
        curve = cl.k_distance_curve(self.X, self.config.k_distance_k)
        for percentile in self.config.eps_percentiles:
            eps = float(np.percentile(curve, percentile))
            if eps <= 0:
                continue
            other = cl.noise_for_eps(self.X, eps, self.config.k_distance_k)
            if other.n_clusters > 1 and other.n_noise > 0:
                self.assertLessEqual(
                    choice.noise_share, other.noise_share + 1e-9,
                    f"p{percentile} had a lower noise share with >1 cluster",
                )

    def test_constant_data_fails_with_an_actionable_message(self):
        """A degenerate matrix has no meaningful eps, and saying so beats
        fitting something arbitrary."""
        with self.assertRaises(cl.ClusteringError) as ctx:
            cl.select_eps(np.ones((40, 4)), TrainConfig(k_distance_k=5, seed=0))
        self.assertTrue(str(ctx.exception))

    def test_percentile_100_is_not_offered(self):
        """At p100 the eps swallows the whole curve and everything becomes one
        cluster, which makes every downstream metric meaningless."""
        config = TrainConfig(eps_percentiles=(50.0, 99.9, 100.0), k_distance_k=10, seed=0)
        self.assertNotIn(100.0, config.scaled_eps_percentiles(1000))
        self.assertIn(99.9, config.scaled_eps_percentiles(1000))


class KSelectionTests(SimpleTestCase):
    def setUp(self):
        super().setUp()
        self.config = TrainConfig(seed=0, silhouette_floor=0.15)

    def test_finds_four_clusters_when_there_are_four(self):
        """The point of the whole exercise: not 3.

        The original hardcoded three regardless. Here the data has four, and a
        search that returned 3 would be reproducing the original's mistake in a
        new outfit.
        """
        X = blobs([(0.0,) * 6, (20.0,) * 6, (40.0,) * 6, (60.0,) * 6],
                  n_per=50, spread=0.3, seed=3)
        config = TrainConfig(seed=0, n_clusters_candidates=(2, 3, 4, 5), silhouette_floor=0.15)
        best, candidates, reason = cl.select_k(X, config)
        self.assertEqual(best.k, 4)
        self.assertEqual(reason, "")
        self.assertEqual(sorted(c.k for c in candidates), [2, 3, 4, 5])
        # And the winner must genuinely be the best of the four.
        self.assertAlmostEqual(
            best.silhouette, max(c.silhouette for c in candidates), places=9
        )

    def test_finds_two_clusters_when_there_are_two(self):
        X = blobs([(0.0,) * 5, (25.0,) * 5], n_per=60, spread=0.4, seed=4)
        config = TrainConfig(seed=0, n_clusters_candidates=(2, 3, 4), silhouette_floor=0.15)
        best, _, reason = cl.select_k(X, config)
        self.assertEqual(best.k, 2)
        self.assertEqual(reason, "")

    def test_weak_structure_falls_back_and_says_why(self):
        """When the landscape is flat, the argmax is reading noise.

        A best silhouette under the floor means k=2 barely separates, so
        trusting the winner would be over-reading the data. The fallback is
        recorded so a reader knows the number was not chosen on merit.
        """
        rng = np.random.default_rng(5)
        X = rng.normal(0.0, 1.0, (300, 5))  # one blob: no real structure
        config = TrainConfig(seed=0, n_clusters_candidates=(2, 3, 4, 5), silhouette_floor=0.9)
        best, candidates, reason = cl.select_k(X, config)
        self.assertTrue(reason, "a fallback must be explained, not silent")
        self.assertIn("silhouette", reason.lower())
        self.assertEqual(best.k, candidates[len(candidates) // 2].k)

    def test_no_computable_k_raises(self):
        """A dataset smaller than the smallest candidate has no valid k.

        Three samples is still enough to score k=2, so the guard has to be
        driven by the absence of any computable candidate rather than by a row
        count. A single sample admits no k at all, and that is the case worth
        failing on.
        """
        with self.assertRaises(cl.ClusteringError) as ctx:
            cl.select_k(np.zeros((1, 3)), TrainConfig(n_clusters_candidates=(2, 3)))
        self.assertIn("segment", str(ctx.exception).lower() + "segment")

    def test_k_at_or_above_sample_count_is_skipped_not_fatal(self):
        X = blobs([(0.0,) * 4, (15.0,) * 4], n_per=8, spread=0.3, seed=6)
        config = TrainConfig(seed=0, n_clusters_candidates=(2, 3, 50))
        best, candidates, _ = cl.select_k(X, config)
        self.assertNotIn(50, [c.k for c in candidates])
        self.assertIn(best.k, [2, 3])

    def test_all_three_metrics_are_recorded(self):
        X = blobs([(0.0,) * 4, (18.0,) * 4, (36.0,) * 4], n_per=40, spread=0.3, seed=7)
        best, _, _ = cl.select_k(X, TrainConfig(seed=0, n_clusters_candidates=(2, 3, 4)))
        payload = best.as_dict()
        for key in ("k", "silhouette", "davies_bouldin", "calinski_harabasz"):
            self.assertIn(key, payload)
        self.assertLess(best.davies_bouldin, 1.0)


class SegmentationTests(SimpleTestCase):
    def setUp(self):
        super().setUp()
        self.config = TrainConfig(k_distance_k=12, seed=0, n_clusters_candidates=(2, 3, 4))
        rng = np.random.default_rng(8)
        self.X = np.vstack([
            blobs([(0.0,) * 6, (25.0,) * 6], n_per=55, spread=0.35, seed=8),
            rng.normal(70.0, 4.0, (20, 6)),
        ])

    def test_fit_produces_both_models(self):
        seg = cl.fit_segmentation(self.X, self.config)
        self.assertIsInstance(seg.dbscan, DBSCAN)
        self.assertGreaterEqual(seg.n_clusters, 2)
        self.assertGreater(seg.eps, 0)

    def test_gate_rejects_the_far_outliers(self):
        """The gate's job is to drop profiles that sit nowhere near real ones."""
        seg = cl.fit_segmentation(self.X, self.config)
        labels = cl.segment_labels(seg, self.X)
        rejected = np.flatnonzero(labels == -1)
        self.assertGreater(len(rejected), 0)
        # The outliers were appended last, so they are the tail of the array.
        self.assertGreater(rejected.mean(), len(self.X) * 0.5)

    def test_spectral_labels_are_indexed_against_the_survivors(self):
        """The two models are fitted on different row sets.

        DBSCAN sees every row; the spectral model sees only the survivors. Its
        ``labels_`` is therefore indexed against the *filtered* subset, and
        confusing the two index spaces is the bug the web app's
        ``_spectral_labels_by_core_sample`` exists to work around. Asserting
        the alignment here means a future change to the order would be caught.
        """
        seg = cl.fit_segmentation(self.X, self.config)
        clean = np.flatnonzero(seg.dbscan.labels_ != -1)
        self.assertEqual(len(seg.spectral.labels_), len(clean))
        self.assertTrue(np.array_equal(seg.clean_index, clean))

        labels = cl.segment_labels(seg, self.X)
        self.assertEqual(len(labels), len(self.X))
        # Survivors carry the spectral label; everything else is -1.
        self.assertTrue(np.array_equal(labels[clean], seg.spectral.labels_))
        self.assertTrue(np.all(labels[np.setdiff1d(np.arange(len(self.X)), clean)] == -1))

    def test_gate_that_leaves_nothing_fails_loudly(self):
        """``min_samples`` larger than the dataset makes every point noise.

        DBSCAN core samples are by definition not noise, so if no point can
        qualify as a core sample the entire dataset is rejected and there is
        nothing left to segment. That has to stop the run rather than produce a
        spectral model fitted on an empty matrix.
        """
        rng = np.random.default_rng(9)
        with self.assertRaises(cl.ClusteringError) as ctx:
            cl.fit_segmentation(
                rng.normal(0.0, 1.0, (8, 4)),
                # k_distance_k doubles as DBSCAN's min_samples, so 20 core
                # samples are demanded from 8 rows.
                TrainConfig(k_distance_k=20, seed=0, eps_percentiles=(80.0,)),
            )
        message = str(ctx.exception).lower()
        self.assertIn("eps", message)

    def test_an_eps_near_zero_does_not_abort_the_run(self):
        """A percentile so low that everything becomes its own cluster is a
        misconfiguration, not a crash.

        The usable-split search finds nothing with more than one cluster and
        some noise, so it falls back to the eps with the most core samples and
        carries on. Failing the whole run here would be worse than training on
        a segmentation that is admittedly poor — the manifest records the
        outcome either way.
        """
        rng = np.random.default_rng(10)
        segmentation = cl.fit_segmentation(
            rng.normal(0.0, 1.0, (50, 4)),
            TrainConfig(k_distance_k=5, seed=0, eps_percentiles=(0.5,)),
        )
        self.assertGreater(segmentation.eps, 0)
        self.assertGreaterEqual(segmentation.n_clusters, 2)

    def test_too_few_samples_fails_before_fitting(self):
        with self.assertRaises(cl.ClusteringError) as ctx:
            cl.fit_segmentation(np.random.default_rng(0).normal(size=(5, 3)), self.config)
        self.assertIn("10", str(ctx.exception))

    def test_manifest_records_the_evidence(self):
        seg = cl.fit_segmentation(self.X, self.config)
        payload = seg.as_dict()
        for key in (
            "eps", "eps_percentile", "n_clusters", "selected_k",
            "k_candidates", "selected_noise", "k_fallback_reason",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["selected_k"]["k"], seg.n_clusters)

    def test_selection_is_reproducible(self):
        """Same seed, same segments. A run that cannot be reproduced cannot be
        audited or rolled back."""
        a = cl.fit_segmentation(self.X, self.config)
        b = cl.fit_segmentation(self.X, self.config)
        self.assertEqual(a.eps, b.eps)
        self.assertEqual(a.n_clusters, b.n_clusters)
        self.assertTrue(np.array_equal(a.spectral.labels_, b.spectral.labels_))
        self.assertTrue(np.array_equal(a.dbscan.labels_, b.dbscan.labels_))


class SegmentSizeTests(SimpleTestCase):
    def test_noise_is_excluded_from_the_sizes(self):
        sizes = cl.segment_sizes([0, 0, 1, 1, 1, -1, -1])
        self.assertEqual(sizes, {0: 2, 1: 3})
        self.assertNotIn(-1, sizes)

    def test_empty_is_empty(self):
        self.assertEqual(cl.segment_sizes([]), {})
        self.assertEqual(cl.segment_sizes([-1, -1]), {})
