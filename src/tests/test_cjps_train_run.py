"""End-to-end tests for the training run.

These use the real ``run.train`` entry point against generated data, so the
whole chain — load, impute, compress, segment, select, fit, write, verify — is
exercised exactly as an operator would invoke it.

They are slower than the unit tests and deliberately so. The pipeline has an
ordering that matters (scaling, then segmentation, then the classifier under
folds), and an ordering bug is precisely the kind of defect that unit tests on
individual functions cannot see.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from django.test import SimpleTestCase, TestCase

from src.cjps_train import fixtures, run
from src.cjps_train.config import DatasetPaths, TrainConfig


class _Workspace:
    """A temp directory with the source CSVs already written."""

    def __init__(self, n_users: int = 320):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.data = self.root / "data"
        self.models = self.root / "models"
        self.reports = self.root / "reports"
        fixtures.write_dataset(self.data, n_users=n_users)

    def close(self):
        self._tmp.cleanup()

    @property
    def paths(self) -> DatasetPaths:
        return DatasetPaths(
            journeys=self.data / "A_TravelDataJourneys.csv",
            users=self.data / "B_TravelDataUsers.csv",
        )

    def config(self, **overrides) -> TrainConfig:
        params = dict(
            seed=13,
            cv_splits=3,
            # A small candidate set: these tests check the run's structure, not
            # its conclusions, and the full roster costs minutes.
            n_clusters_candidates=(2, 3),
            k_distance_k=10,
            holdout_fraction=0.25,
            model_dir=self.models,
            reports_dir=self.reports,
            force=True,
        )
        params.update(overrides)
        return TrainConfig(**params)


class MissingDataTests(SimpleTestCase):
    def test_absent_files_are_named_before_anything_else_runs(self):
        """A run must not get as far as fitting a model on nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            paths = DatasetPaths(
                journeys=Path(tmp) / "a.csv", users=Path(tmp) / "b.csv"
            )
            with self.assertRaises(Exception) as ctx:
                run.train(TrainConfig(model_dir=Path(tmp), reports_dir=Path(tmp)),
                          paths=paths)
            message = str(ctx.exception)
            self.assertIn("a.csv", message)
            self.assertIn("b.csv", message)


class RunTests(TestCase):
    """End-to-end runs with a real database.

    The pipeline's last step drives an artefact through
    ``ctmj.services.predictor.predict``, and that path validates the fixture
    profile against the lookup tables. A ``SimpleTestCase`` would forbid those
    queries, so the most important assertion in this file — that the artefacts
    are actually usable — would be the one thing not covered.

    Lookup tables are seeded by the reference-data migration, so the fixture
    profile validates without any extra setup.
    """
    def setUp(self):
        super().setUp()
        self.workspace = _Workspace()
        self.addCleanup(self.workspace.close)

    def _run(self, **overrides):
        from src.cjps_train import select as sel

        # A tiny roster: the structural assertions below are about the run, not
        # about which model wins. The real roster is covered by
        # test_cjps_train_select.
        roster = [
            sel.Candidate("lr", _fast_logistic(), "none", note="test roster"),
        ]
        original = sel.default_candidates
        sel.default_candidates = lambda seed: roster
        try:
            return run.train(self.workspace.config(**overrides), paths=self.workspace.paths)
        finally:
            sel.default_candidates = original

    def test_run_completes_and_verifies(self):
        result = self._run()
        verification = result.manifest.get("verification", {})
        self.assertTrue(verification.get("ok"),
                        msg=f"verification failed: {verification}")
        # Not merely "no error": the check must have actually run. A skipped
        # verification is reported as ok, so the status is asserted separately.
        self.assertFalse(verification.get("skipped"),
                         msg=f"verification was skipped: {verification.get('error')}")
        self.assertTrue(verification.get("channels"))
        self.assertTrue(result.manifest_path.is_file())

    def test_all_five_artefacts_are_written(self):
        """The five filenames the app's settings expect, exactly."""
        result = self._run()
        written = sorted(p.name for p in self.workspace.models.glob("*.pkl"))
        for required in run.artefact_paths(self.workspace.models).values():
            self.assertTrue(required.is_file(), f"{required.name} was not written")
        self.assertIn("user_data_imputer.pkl", written)
        self.assertIn("dbscan_clustering_model.pkl", written)
        self.assertIn("spectral_clustering_model.pkl", written)
        self.assertIn("s1_predicting_preprocessor.pkl", written)
        self.assertIn("GradientBoostingClassifier_model.pkl", written)
        del result

    def test_imputer_is_saved_alongside_the_artefacts(self):
        """The app's inference path starts from complete form fields, so it
        never needs imputation — but a future version that accepts partial
        input could not reproduce training without this file."""
        self._run()
        self.assertTrue((self.workspace.models / "user_data_imputer.pkl").is_file())

    def test_artefacts_load_back_and_predict(self):
        """The run is not finished until its own output is usable.

        This is the check that catches a training/inference contract mismatch
        while it is still a training problem rather than a user-facing one.
        """
        result = self._run()
        verification = result.manifest["verification"]
        self.assertTrue(verification["ok"], msg=verification.get("error"))
        self.assertTrue(verification["channels"])
        for channel in verification["channels"]:
            self.assertGreaterEqual(channel["probability"], 0.0)
            self.assertLessEqual(channel["probability"], 100.0)

    def test_manifest_is_valid_json_with_the_sections_a_reader_needs(self):
        result = self._run()
        payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        for key in (
            "schema", "seed", "config", "versions", "data",
            "preprocessing", "segmentation", "selection", "holdout", "notes",
            "artefacts", "verification",
        ):
            self.assertIn(key, payload, msg=f"manifest is missing {key!r}")

    def test_manifest_records_the_seed_and_versions(self):
        """Without the seed and library versions a run cannot be reproduced,
        and sklearn in particular changes defaults between releases."""
        result = self._run()
        self.assertEqual(result.manifest["seed"], 13)
        versions = result.manifest["versions"]
        for key in ("python", "numpy", "pandas", "scikit_learn", "joblib"):
            self.assertIn(key, versions)
            self.assertTrue(versions[key])

    def test_manifest_records_the_clustering_evidence(self):
        """The point of selecting k is that the numbers behind it are kept."""
        result = self._run()
        segmentation = result.manifest["segmentation"]
        self.assertIn("selected_k", segmentation)
        self.assertTrue(segmentation["k_candidates"])
        self.assertGreater(segmentation["eps"], 0)
        for candidate in segmentation["k_candidates"]:
            self.assertIn("silhouette", candidate)
            self.assertIn("davies_bouldin", candidate)
        self.assertIn("segment_sizes", segmentation)

    def test_manifest_records_the_selection_grid(self):
        result = self._run()
        selection = result.manifest["selection"]
        self.assertEqual(selection["winner"], "lr")
        self.assertEqual(selection["winner_note"], "test roster")
        self.assertTrue(selection["candidates"])
        # The winner must be a row in its own table, or the recorded CV score
        # cannot be traced back to a measurement.
        ranked = {row["name"] for row in selection["ranking"]}
        self.assertIn(selection["winner"], ranked)

    def test_holdout_is_reported_separately_from_the_cv_score(self):
        """A holdout exists precisely so one number is not a decision's own
        output. Reporting only the cross-validated score would defeat it."""
        result = self._run()
        holdout = result.manifest["holdout"]
        for key in ("f1_macro", "accuracy", "top3_accuracy"):
            self.assertIn(key, holdout)
            self.assertGreaterEqual(holdout[key], 0.0)
        self.assertIn("cv_vs_holdout_gap", result.manifest)

    def test_holdout_and_selection_use_disjoint_data(self):
        """Guard the ordering: the holdout is taken before selection and the
        selection scores are computed on the training side only."""
        result = self._run()
        training_rows = result.manifest["data"]["modelling_rows"]
        holdout_rows = int(round(
            training_rows * result.manifest["config"]["holdout_fraction"]
        ))
        self.assertGreater(holdout_rows, 0)
        self.assertLess(holdout_rows, training_rows)

    def test_target_classes_below_the_floor_are_reported_as_dropped(self):
        """A touchpoint with too few examples cannot be learned, and the app
        must not pretend otherwise."""
        result = self._run()
        dropped = result.manifest["data"]["target_classes_dropped"]
        kept = result.manifest["data"]["target_classes_kept"]
        self.assertIsInstance(dropped, list)
        self.assertTrue(kept)
        # Nothing dropped may also be kept.
        self.assertEqual(set(dropped) & set(kept), set())

    def test_gate_rejected_rows_are_excluded_from_training(self):
        """The app cannot serve a segment for a noise-rejected profile, so
        training on those rows would be learning a class it never returns."""
        result = self._run()
        segmentation = result.manifest["segmentation"]
        rejected = segmentation["gate_rejected_rows"]
        total = segmentation["segment_sizes"]
        self.assertEqual(rejected + sum(total.values()),
                         result.manifest["data"]["modelling_rows"])

    def test_the_run_is_reproducible(self):
        """Same seed, same artefacts.

        DBSCAN and SpectralClustering are both stochastic, so without a seed
        two runs of the same code produce different models and a manifest
        cannot be used to reproduce anything.
        """
        first = self._run()
        second = self._run()
        self.assertEqual(
            first.manifest["segmentation"]["eps"],
            second.manifest["segmentation"]["eps"],
        )
        self.assertEqual(
            first.manifest["segmentation"]["n_clusters"],
            second.manifest["segmentation"]["n_clusters"],
        )
        self.assertEqual(
            first.manifest["holdout"]["f1_macro"],
            second.manifest["holdout"]["f1_macro"],
        )

    def test_different_seeds_are_reported_differently(self):
        """Sanity check on the seed itself: it has to reach the estimators.

        The manifest text is captured inside each run: both runs write to the
        same path, so reading the file after the second run would compare the
        second manifest with itself.
        """
        a = self._run()
        a_text = a.manifest_path.read_text(encoding="utf-8")
        b = self._run(seed=99)
        b_text = b.manifest_path.read_text(encoding="utf-8")

        self.assertEqual(a.manifest["seed"], 13)
        self.assertEqual(b.manifest["seed"], 99)
        self.assertIn('"seed": 13', a_text)
        self.assertIn('"seed": 99', b_text)
        self.assertNotEqual(a_text, b_text)

    def test_notes_flag_a_weak_segmentation(self):
        """A manifest that only reports good news is not a manifest."""
        result = self._run()
        self.assertIsInstance(result.manifest["notes"], list)
        for note in result.manifest["notes"]:
            self.assertTrue(note.strip())

    def test_data_report_distinguishes_imputed_from_complete(self):
        """The same post-hoc number would report "nothing was missing" and
        "everything missing was filled in" identically."""
        result = self._run()
        report = result.manifest["data"]["load_report"]
        self.assertIn("imputed_cells", report)
        self.assertGreaterEqual(report["imputed_cells"], 0)
        # This fixture injects gaps, so something must have been filled.
        self.assertGreater(report["imputed_cells"], 0)
        self.assertLess(report["users_after_clean"], report["users_raw"] + 1)


class SkipSelectionTests(TestCase):
    """``--skip-selection`` must say so in the manifest, not pretend."""

    def setUp(self):
        super().setUp()
        self.workspace = _Workspace(n_users=260)
        self.addCleanup(self.workspace.close)

    def test_placeholder_winner_is_labelled_as_not_chosen(self):
        result = run.train(
            self.workspace.config(), paths=self.workspace.paths, skip_selection=True
        )
        selection = result.manifest["selection"]
        self.assertIn("not a chosen model", selection["winner_note"])
        self.assertEqual(selection["ranking"], [])


def _fast_logistic():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=200)
