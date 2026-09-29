"""The two-stage inference pipeline.

The two ``test_regression_*`` tests are the important ones: they fail against
the original ``predict_view`` and pass against :mod:`ctmj.services.predictor`.
"""

from __future__ import annotations

import numpy as np
from django.test import TestCase

from ctmj.services import formdata, predictor
from ctmj.services.registry import LoadState, ModelRegistry, RegistryStatus
from ctmj.tests import factories


class _FakeRegistry:
    """Minimal registry stand-in so pipeline tests need no settings access."""

    def __init__(self, artefacts):
        self._artefacts = artefacts

    def get(self, key):
        return self._artefacts.get(key)

    def ready(self):
        return all(v is not None for v in self._artefacts.values())

    @property
    def status(self):
        return RegistryStatus(state=LoadState.READY, message="ready")


class ClusterAssignmentTestCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        factories.seed_reference_data()
        cls.artefacts = factories.build_model_doubles()
        cls.registry = _FakeRegistry(cls.artefacts)

    def _validate(self, **overrides):
        return formdata.validate_prediction_form(factories.valid_post(**overrides))

    def test_returns_the_nearest_spectral_cluster_id(self):
        result = predictor.predict(self.registry, self._validate())
        self.assertIn(result.cluster_id, (0, 1, 2, -1))

    def test_regression_spectral_label_must_not_cross_index_spaces(self):
        """The segment id must come from the spectral labels of the core samples.

        ``dbscan.core_sample_indices_`` indexes the **full** training set while
        ``spectral.labels_`` indexes the **noise-filtered** subset (see
        ``src/main.py``). The original code indexed ``spectral.labels_`` with a
        full-set index, which reads the wrong row or overruns the array.

        This asserts the aligned lookup agrees with the ground-truth group of
        the nearest core sample.
        """
        spectral = self.artefacts["spectral"]
        frame, groups = factories.make_clustering_frame()
        scaled = self.artefacts["user_data_preprocessor"].transform(
            frame[list(formdata.CLUSTERING_FEATURES)]
        )

        dbscan = self.artefacts["dbscan"]
        # Ground truth: the group of the true core sample, remapped onto the
        # spectral label space. Blobs 0-2 became clusters 0-2; outliers are
        # excluded because they are noise.
        kept = np.flatnonzero(dbscan.labels_ != -1)
        positions = np.searchsorted(kept, dbscan.core_sample_indices_)

        from scipy.spatial.distance import cdist

        for row_idx in (0, 30, 60, 5, 47):
            cluster_id, _ = predictor.assign_cluster(scaled[row_idx : row_idx + 1], self.artefacts)
            nearest_core = int(
                np.argmin(cdist(scaled[row_idx : row_idx + 1], dbscan.components_)[0])
            )
            self.assertEqual(
                cluster_id,
                int(spectral.labels_[positions[nearest_core]]),
                f"row {row_idx}: segment id must match the nearest core sample's label",
            )
            # The buggy expression used the *unmapped* index.
            self.assertLess(
                int(spectral.labels_[nearest_core]) if nearest_core < len(spectral.labels_) else -1,
                len(spectral.labels_),
            )

    def test_regression_noise_gate_fires_beyond_eps(self):
        """DBSCAN core samples are never noise, so a label cannot detect it.

        A profile far outside every core neighbourhood must come back as -1.
        The original gate read ``labels_[core_sample_indices_[...]]``, which is
        always >= 0, so noise was unreachable.
        """
        import pandas as pd

        from ctmj.services.formdata import CLUSTERING_FEATURES

        # Every feature far outside the training range, hence far from any
        # core sample.
        far = pd.DataFrame([{col: 500.0 for col in CLUSTERING_FEATURES}])
        scaled = self.artefacts["user_data_preprocessor"].transform(far)

        cluster_id, distance = predictor.assign_cluster(scaled, self.artefacts)
        self.assertEqual(cluster_id, predictor.NOISE_LABEL)
        self.assertGreater(distance, self.artefacts["dbscan"].eps)

    def test_alignment_helper_rejects_mismatched_arrays(self):
        """A spectral model fitted on a different subset must not be trusted."""
        dbscan = self.artefacts["dbscan"]

        class _Truncated:
            """spectral.labels_ deliberately too short to index safely."""

            labels_ = np.array([0])

        self.assertIsNone(
            predictor._spectral_labels_by_core_sample(dbscan, _Truncated())
        )

    def test_nearby_profile_is_not_noise(self):
        frame, _ = factories.make_clustering_frame()
        scaled = self.artefacts["user_data_preprocessor"].transform(
            frame[list(formdata.CLUSTERING_FEATURES)].head(1)
        )
        cluster_id, distance = predictor.assign_cluster(scaled, self.artefacts)
        self.assertNotEqual(cluster_id, predictor.NOISE_LABEL)
        self.assertLessEqual(distance, self.artefacts["dbscan"].eps)

    def test_falls_back_to_dbscan_label_when_spectral_is_unusable(self):
        artefacts = dict(self.artefacts)
        artefacts["spectral"] = object()  # no components_ attribute
        frame, _ = factories.make_clustering_frame()
        scaled = artefacts["user_data_preprocessor"].transform(
            frame[list(formdata.CLUSTERING_FEATURES)].head(1)
        )
        cluster_id, _ = predictor.assign_cluster(scaled, artefacts)
        self.assertIn(cluster_id, (0, 1, 2))


class ChannelRankingTestCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        factories.seed_reference_data()
        cls.artefacts = factories.build_model_doubles()
        cls.registry = _FakeRegistry(cls.artefacts)

    def test_returns_top_three_sorted_descending(self):
        result = predictor.predict(
            self.registry,
            formdata.validate_prediction_form(factories.valid_post()),
        )
        self.assertEqual(len(result.channels), predictor.TOP_N)
        probabilities = [c.probability for c in result.channels]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))

    def test_probabilities_are_a_percentage(self):
        result = predictor.predict(
            self.registry,
            formdata.validate_prediction_form(factories.valid_post()),
        )
        for channel in result.channels:
            self.assertGreaterEqual(channel.probability, 0.0)
            self.assertLessEqual(channel.probability, 100.0)

    def test_channels_are_resolved_to_names_and_descriptions(self):
        result = predictor.predict(
            self.registry,
            formdata.validate_prediction_form(factories.valid_post()),
        )
        for channel in result.channels:
            self.assertNotIn("#", channel.name)
            self.assertTrue(channel.description)
            self.assertNotIn("Chưa có mô tả", channel.description)

    def test_distribution_covers_every_class(self):
        result = predictor.predict(
            self.registry,
            formdata.validate_prediction_form(factories.valid_post()),
        )
        expected = len(self.artefacts["gradient_boosting"].classes_)
        self.assertEqual(len(result.distribution), expected)

    def test_distribution_sums_to_one_hundred_percent(self):
        result = predictor.predict(
            self.registry,
            formdata.validate_prediction_form(factories.valid_post()),
        )
        total = sum(prob for _, prob in result.distribution)
        self.assertAlmostEqual(total, 100.0, places=4)

    def test_confidence_is_the_top_n_share(self):
        result = predictor.predict(
            self.registry,
            formdata.validate_prediction_form(factories.valid_post()),
        )
        self.assertAlmostEqual(
            result.confidence, sum(c.probability for c in result.channels), places=6
        )


class PredictionContractTestCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        factories.seed_reference_data()
        cls.artefacts = factories.build_model_doubles()

    def test_missing_artefacts_raise_a_user_facing_error(self):
        registry = _FakeRegistry({"dbscan": self.artefacts["dbscan"]})
        with self.assertRaises(predictor.PredictionError) as ctx:
            predictor.predict(
                registry, formdata.validate_prediction_form(factories.valid_post())
            )
        self.assertIn("chưa sẵn sàng", str(ctx.exception))

    def test_estimator_is_unwrapped_from_a_search_object(self):
        """A GridSearchCV-shaped model must still work."""
        wrapped = type(
            "Search",
            (),
            {"best_estimator_": self.artefacts["gradient_boosting"]},
        )()
        artefacts = dict(self.artefacts)
        artefacts["gradient_boosting"] = wrapped
        registry = _FakeRegistry(artefacts)
        result = predictor.predict(
            registry, formdata.validate_prediction_form(factories.valid_post())
        )
        self.assertEqual(len(result.channels), predictor.TOP_N)

    def test_cluster_metadata_is_attached(self):
        registry = _FakeRegistry(self.artefacts)
        result = predictor.predict(
            registry, formdata.validate_prediction_form(factories.valid_post())
        )
        self.assertTrue(result.cluster_name)
        self.assertTrue(result.cluster_description)
        self.assertTrue(result.cluster_label)

    def test_noise_result_sets_is_noise_and_uses_the_noise_label(self):
        from ctmj.services.reference_data import CLUSTER_LABELS

        registry = _FakeRegistry(self.artefacts)
        result = predictor.predict(registry, self._validate_noise_case())
        self.assertTrue(result.is_noise)
        self.assertEqual(result.cluster_label, CLUSTER_LABELS[-1])

    def _validate_noise_case(self):
        """Every numeric field pushed far out of the training distribution."""
        return formdata.validate_prediction_form(
            factories.valid_post(
                Age="118",
                BAS_huishoudgrootte="30",
                afg_kinderen_huishouden="28",
            )
        )
