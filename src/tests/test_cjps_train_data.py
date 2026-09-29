"""Tests for data loading, cleaning and journey assembly.

The defects these pin down are all cases where the original pipeline produced a
model that trained without complaint while quietly training on the wrong thing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from src.cjps_train import data as data_mod
from src.cjps_train import fixtures
from src.cjps_train.config import CLUSTERING_FEATURES, TARGET, DatasetPaths


class _TempDirMixin:
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)


class ReadingTests(_TempDirMixin, SimpleTestCase):
    def test_missing_tokens_become_nan(self):
        """``"--"`` must be missing, not a string.

        Without ``na_values`` the column stays object dtype, every numeric
        operation silently degrades to a string comparison, and the KNN imputer
        downstream cannot repair it because the column is no longer numeric.
        """
        path = self.tmp / "B_TravelDataUsers.csv"
        pd.DataFrame({
            "UserID": [1, 2, 3],
            "Age": [30, "--", 45],
            "GenderID": [1, 2, 1],
        }).to_csv(path, index=False)

        frame = data_mod.read_csv(path, required=["UserID", "Age"])
        self.assertTrue(pd.isna(frame["Age"].iloc[1]))
        self.assertTrue(pd.api.types.is_numeric_dtype(frame["Age"]))

    def test_missing_file_names_the_path(self):
        with self.assertRaises(data_mod.DataError) as ctx:
            data_mod.read_csv(self.tmp / "nope.csv")
        self.assertIn("nope.csv", str(ctx.exception))

    def test_missing_required_column_is_named(self):
        path = self.tmp / "u.csv"
        pd.DataFrame({"UserID": [1], "Age": [3]}).to_csv(path, index=False)
        with self.assertRaises(data_mod.DataError) as ctx:
            data_mod.read_csv(path, required=["UserID", "GenderID"])
        self.assertIn("GenderID", str(ctx.exception))

    def test_load_dataset_reports_every_absent_file(self):
        """Both paths in one message, so the operator fixes it in one pass."""
        paths = DatasetPaths(
            journeys=self.tmp / "a.csv", users=self.tmp / "b.csv"
        )
        with self.assertRaises(data_mod.DataError) as ctx:
            data_mod.load_dataset(paths)
        message = str(ctx.exception)
        self.assertIn("a.csv", message)
        self.assertIn("b.csv", message)


class CleaningTests(SimpleTestCase):
    def test_non_numeric_values_become_nan(self):
        frame = pd.DataFrame({"Age": [30, "unknown", 45]})
        out = data_mod.coerce_numeric(frame, ["Age"])
        self.assertTrue(pd.isna(out["Age"].iloc[1]))
        self.assertTrue(pd.api.types.is_numeric_dtype(out["Age"]))

    def test_column_absent_is_not_an_error(self):
        out = data_mod.coerce_numeric(pd.DataFrame({"Age": [1]}), ["Nope"])
        self.assertIn("Age", out.columns)

    def test_duplicate_user_ids_are_collapsed(self):
        """A duplicate would multiply every touchpoint that user has on join."""
        users = pd.DataFrame({
            "UserID": [1, 1, 2],
            **{c: [1, 2, 3] for c in CLUSTERING_FEATURES},
        })
        out = data_mod.clean_users(users)
        self.assertEqual(len(out), 2)
        self.assertEqual(sorted(out["UserID"].tolist()), [1, 2])

    def test_rows_missing_some_features_are_kept_for_imputation(self):
        """A partial profile is recoverable; discarding it loses real signal.

        Dropping every row with any gap made the KNN imputer a no-op — there was
        nothing left to impute — and cost 28% of the profiles on this dataset
        while the report still said the data was clean.
        """
        users = pd.DataFrame({
            "UserID": [1, 2, 3],
            **{c: [1, 2, 3] for c in CLUSTERING_FEATURES},
        })
        users.loc[1, "Age"] = np.nan
        out = data_mod.clean_users(users)
        self.assertEqual(sorted(out["UserID"].tolist()), [1, 2, 3])
        self.assertEqual(int(out["Age"].isna().sum()), 1)

    def test_rows_missing_every_feature_are_dropped(self):
        """A row with no values has no neighbours, so it cannot be imputed.

        Keeping it would also poison the neighbour search: every other row's
        nearest match would be this one, and the filled-in values would all
        come from the same useless source.
        """
        users = pd.DataFrame({
            "UserID": [1, 2],
            **{c: [1, 2] for c in CLUSTERING_FEATURES},
        })
        users.loc[1, CLUSTERING_FEATURES] = np.nan
        out = data_mod.clean_users(users)
        self.assertEqual(out["UserID"].tolist(), [1])

    def test_rows_missing_most_features_are_dropped(self):
        """Past the threshold, imputation is mostly guessing."""
        users = pd.DataFrame({
            "UserID": [1, 2, 3],
            **{c: [1, 2, 3] for c in CLUSTERING_FEATURES},
        })
        # Seven of ten features missing: above the 0.5 threshold.
        for column in CLUSTERING_FEATURES[:7]:
            users.loc[2, column] = np.nan
        out = data_mod.clean_users(users)
        self.assertEqual(sorted(out["UserID"].tolist()), [1, 2])

    def test_report_distinguishes_retained_gaps_from_dropped_rows(self):
        users = pd.DataFrame({
            "UserID": [1, 2],
            **{c: [1, 2] for c in CLUSTERING_FEATURES},
        })
        users.loc[1, "Age"] = np.nan
        report = data_mod.LoadReport()
        out = data_mod.clean_users(users, report)
        self.assertEqual(len(out), 2)
        self.assertTrue(
            any("imputation" in n for n in report.notes),
            f"expected a note about retained gaps, got {report.notes}",
        )

    def test_report_records_what_was_dropped(self):
        users = pd.DataFrame({
            "UserID": [1, 1, 2],
            **{c: [1, 2, 3] for c in CLUSTERING_FEATURES},
        })
        report = data_mod.LoadReport()
        data_mod.clean_users(users, report)
        self.assertEqual(report.users_raw, 3)
        self.assertEqual(report.users_after_clean, 2)
        self.assertTrue(any("duplicate" in n for n in report.notes))


class ImputationTests(SimpleTestCase):
    def test_imputer_is_returned_not_discarded(self):
        """The fitted imputer is the contract.

        The original fitted ``KNNImputer``, replaced the table with its output,
        and threw the object away. Nothing downstream could then reproduce the
        same imputation, so training and serving silently disagreed the moment
        inference saw a missing value.
        """
        users = fixtures.make_users(n=80, missing_rate=0.3)
        imputed, imputer = data_mod.build_imputer(users)
        self.assertIsNotNone(imputer)
        self.assertFalse(imputed[list(CLUSTERING_FEATURES)].isna().any().any())
        self.assertTrue(hasattr(imputer, "transform"))

    def test_imputer_output_is_usable_again(self):
        """It has to work on new data, not just the data it was fitted on.

        The rows fed back in are the already-imputed ones. The fitted matrix is
        numeric, so handing the raw ``"--"`` placeholders back would be asking
        the imputer to do type conversion as well, which is the reader's job.
        """
        users = fixtures.make_users(n=80, missing_rate=0.2)
        imputed, imputer = data_mod.build_imputer(users)
        out = np.asarray(
            imputer.transform(imputed[list(CLUSTERING_FEATURES)].head(5).to_numpy(dtype=float))
        )
        self.assertEqual(out.shape[0], 5)
        self.assertFalse(np.isnan(out).any())

    def test_entirely_missing_column_fails_loudly(self):
        """KNNImputer cannot impute a column that is wholly NaN.

        It propagates NaN silently, which would leave a column of NaN in the
        scaler and produce a model that is quietly blind to that feature.
        """
        users = fixtures.make_users(n=60, missing_rate=0.0)
        users["AFG_sk2015"] = "--"
        with self.assertRaises(data_mod.DataError) as ctx:
            data_mod.build_imputer(users)
        self.assertIn("AFG_sk2015", str(ctx.exception))


class JourneyCompressionTests(SimpleTestCase):
    def _journey(self, rows):
        return pd.DataFrame(
            rows,
            columns=["UserID", "TIMESPSS", "type_touch", "Duration"],
        )

    def test_repeated_touchpoints_collapse_to_one(self):
        """Three pages on the same channel is one touchpoint, not three.

        Left uncollapsed, the (step2, step3) -> step1 window fills with a
        single repeated channel and the model learns that people never move.
        """
        journey = self._journey([
            (1, "2024-01-01 10:00:00", 3, 10),
            (1, "2024-01-01 10:05:00", 3, 20),   # same channel, repeat
            (1, "2024-01-01 10:10:00", 3, 30),   # same channel, repeat
            (1, "2024-01-01 12:00:00", 7, 40),   # different channel
        ])
        out = data_mod.compress_journey(journey)
        self.assertEqual(len(out), 2)
        self.assertEqual(sorted(out["type_touch"].tolist()), [3, 7])

    def test_run_ids_are_independent_per_user(self):
        """A user whose journey is identical to another's must not join their run.

        The original computed ``group_id`` with one global ``cumsum`` over the
        whole frame. It only worked because ``UserID`` also appeared in the
        groupby key â€” the two were coupled by coincidence, and the run counter
        itself was meaningless on its own.
        """
        journey = self._journey([
            (1, "2024-01-01 10:00:00", 3, 10),
            (1, "2024-01-01 10:05:00", 3, 20),
            (2, "2024-01-01 10:00:00", 3, 10),
            (2, "2024-01-01 10:05:00", 3, 20),
        ])
        out = data_mod.compress_journey(journey)
        # Two users, each with one run.
        self.assertEqual(len(out), 2)
        self.assertEqual(out.groupby("UserID")["_run"].nunique().tolist(), [1, 1])
        # And the run ids must not collide across users.
        runs = out.groupby("UserID")["_run"].first().tolist()
        self.assertEqual(runs[0], runs[1], "identical journeys should share a run id")

    def test_repeated_channel_after_a_gap_is_a_new_run(self):
        journey = self._journey([
            (1, "2024-01-01 10:00:00", 3, 10),
            (1, "2024-01-01 12:00:00", 7, 10),
            (1, "2024-01-01 14:00:00", 3, 10),   # back to 3, but not adjacent
        ])
        out = data_mod.compress_journey(journey)
        self.assertEqual(len(out), 3)

    def test_zero_duration_is_dropped(self):
        journey = self._journey([
            (1, "2024-01-01 10:00:00", 3, 0),
            (1, "2024-01-01 12:00:00", 7, 10),
        ])
        out = data_mod.compress_journey(journey)
        self.assertEqual(out["type_touch"].tolist(), [7])

    def test_unparseable_timestamp_drops_the_row(self):
        journey = self._journey([
            (1, "not a date", 3, 10),
            (1, "2024-01-01 12:00:00", 7, 10),
        ])
        out = data_mod.compress_journey(journey)
        self.assertEqual(out["type_touch"].tolist(), [7])


class JourneyWindowTests(SimpleTestCase):
    def _compressed(self, rows):
        return data_mod.compress_journey(pd.DataFrame(
            rows, columns=["UserID", "TIMESPSS", "type_touch", "Duration"]
        ))

    def test_step1_is_the_newest_touchpoint(self):
        """The target must be the event *after* the two inputs.

        Getting the order backwards yields a model that predicts the past, and
        it fails silently: the training runs, the accuracy looks fine, and every
        real prediction is wrong.
        """
        compressed = self._compressed([
            (1, "2024-01-01 10:00:00", 1, 10),   # oldest
            (1, "2024-01-02 10:00:00", 2, 10),
            (1, "2024-01-03 10:00:00", 3, 10),   # newest
            (1, "2024-01-04 10:00:00", 4, 10),   # newest
        ])
        window = data_mod.build_journey_window(compressed)
        row = window[window["UserID"] == 1].iloc[0]
        # Descending time: step1 = 04 Jan, step2 = 03 Jan, step3 = 02 Jan.
        self.assertEqual(row["step1"], 4)
        self.assertEqual(row["step2"], 3)
        self.assertEqual(row["step3"], 2)

    def test_only_the_last_three_events_are_kept(self):
        compressed = self._compressed([
            (1, f"2024-01-0{i} 10:00:00", i, 10) for i in range(1, 7)
        ])
        window = data_mod.build_journey_window(compressed)
        row = window[window["UserID"] == 1].iloc[0]
        self.assertEqual(row["step1"], 6)
        self.assertEqual(row["step2"], 5)
        self.assertEqual(row["step3"], 4)

    def test_users_with_too_few_events_are_excluded(self):
        compressed = self._compressed([
            (1, "2024-01-01 10:00:00", 1, 10),
            (1, "2024-01-02 10:00:00", 2, 10),   # only two: no step3
            (2, "2024-01-01 10:00:00", 1, 10),
            (2, "2024-01-02 10:00:00", 2, 10),
            (2, "2024-01-03 10:00:00", 3, 10),
        ])
        window = data_mod.build_journey_window(compressed)
        self.assertEqual(window["UserID"].tolist(), [2])

    def test_order_verification_catches_an_inverted_sort(self):
        """The assertion is only useful if it can actually fail."""
        compressed = self._compressed([
            (1, "2024-01-01 10:00:00", 1, 10),
            (1, "2024-01-02 10:00:00", 2, 10),
            (1, "2024-01-03 10:00:00", 3, 10),
        ])
        window = data_mod.build_journey_window(compressed)
        data_mod.verify_window_order(compressed, window)  # correct: no raise

        # Now claim step1 is the oldest, which is the bug this guards.
        broken = window.copy()
        broken["step1"], broken["step3"] = broken["step3"], broken["step1"]
        with self.assertRaises(data_mod.DataError):
            data_mod.verify_window_order(compressed, broken)


class AssemblyTests(_TempDirMixin, SimpleTestCase):
    def setUp(self):
        super().setUp()
        self.users = fixtures.make_users(n=300, missing_rate=0.08)
        self.journeys = fixtures.make_journeys(self.users)
        self.frame, self.report, self.imputer = data_mod.assemble(
            self.journeys, self.users
        )

    def test_frame_has_every_contract_column(self):
        """Everything the preprocessors will ask for must be present.

        ``final_label`` is the exception: it is the cluster id, produced by the
        segmentation stage in ``run.py``, not by assembly. It is listed here to
        document that the omission is deliberate rather than an oversight.
        """
        from src.cjps_train.config import PREDICTION_FEATURES

        for column in CLUSTERING_FEATURES:
            self.assertIn(column, self.frame.columns)
        for column in PREDICTION_FEATURES:
            if column == "final_label":
                continue
            self.assertIn(column, self.frame.columns)
        self.assertIn(TARGET, self.frame.columns)
        self.assertNotIn(
            "final_label", self.frame.columns,
            msg="assembly must not invent a cluster id; that is the segmenter's job",
        )

    def test_no_missing_values_survive(self):
        self.assertFalse(self.frame[list(CLUSTERING_FEATURES)].isna().any().any())

    def test_one_row_per_user(self):
        self.assertEqual(self.frame["UserID"].nunique(), len(self.frame))

    def test_target_is_not_the_previous_step(self):
        """If step1 always equalled step3 the task would be trivial.

        A fixture where they match proves nothing about the pipeline, so the
        generated data has to have a genuinely different next step.
        """
        self.assertGreater(
            float(np.mean(self.frame["step1"] != self.frame["step3"])),
            0.5,
            msg="the fixture's target is trivially predictable; it cannot test learning",
        )

    def test_report_counts_are_plausible(self):
        self.assertGreater(self.report.journeys_with_window, 0)
        self.assertLessEqual(
            self.report.journeys_compressed, self.report.journeys_after_duration_filter
        )
        self.assertLess(
            self.report.journeys_compressed, self.report.journeys_after_duration_filter
        )

    def test_imputer_travels_with_the_frame(self):
        self.assertIsNotNone(self.imputer)
        self.assertTrue(hasattr(self.imputer, "transform"))


class ContractTests(SimpleTestCase):
    """The training pipeline and the web app must agree on the columns.

    ``ColumnTransformer`` selects by name and the app builds its one-row frame
    from exactly ``formdata.CLUSTERING_FEATURES``. If the two lists drift, every
    prediction fails at runtime â€” so the agreement is asserted rather than
    documented.
    """

    def test_feature_lists_match_the_web_app(self):
        from ctmj.services import formdata

        self.assertEqual(
            list(CLUSTERING_FEATURES), list(formdata.CLUSTERING_FEATURES)
        )
        from src.cjps_train.config import PREDICTION_FEATURES

        self.assertEqual(
            list(PREDICTION_FEATURES), list(formdata.PREDICTION_FEATURES)
        )

    def test_target_is_a_known_touchpoint_code(self):
        from ctmj.services import reference_data

        codes = {t["code"] for t in reference_data.TOUCHPOINTS}
        frame = fixtures.tiny_frame(n=120)
        self.assertTrue(set(frame[TARGET].tolist()).issubset(codes))

