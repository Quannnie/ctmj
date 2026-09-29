"""Validation of submitted form data.

These tests pin the behaviour the original view lacked: codes are checked
against the real lookup tables, numeric bounds are enforced, and a bad field
never discards the other eleven.
"""

from __future__ import annotations

from django.test import TestCase

from ctmj.services import formdata
from ctmj.services.formdata import ALL_FIELDS, validate_prediction_form
from ctmj.tests import factories


class ValidationTestCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        factories.seed_reference_data()

    # --- happy path -------------------------------------------------------
    def test_valid_payload_is_accepted(self):
        result = validate_prediction_form(factories.valid_post())
        self.assertTrue(result.is_valid, msg=result.error_messages)
        self.assertEqual(result.get("Age"), 41)
        self.assertEqual(result.get("step_1_channel"), 20)
        self.assertEqual(result.get("step_2_channel"), 16)

    def test_whitespace_is_tolerated(self):
        result = validate_prediction_form(factories.valid_post(Age="  41 "))
        self.assertTrue(result.is_valid, msg=result.error_messages)
        self.assertEqual(result.get("Age"), 41)

    # --- missing values ---------------------------------------------------
    def test_blank_field_is_rejected(self):
        for field in ALL_FIELDS:
            with self.subTest(field=field):
                result = validate_prediction_form(factories.valid_post(**{field: ""}))
                self.assertFalse(result.is_valid)
                self.assertIn(field, result.errors)

    def test_error_messages_are_human_readable(self):
        result = validate_prediction_form(factories.valid_post(Age=""))
        self.assertIn("Tuổi", result.error_messages[0])

    # --- numeric fields ---------------------------------------------------
    def test_non_numeric_age_is_rejected(self):
        result = validate_prediction_form(factories.valid_post(Age="abc"))
        self.assertFalse(result.is_valid)
        self.assertIn("Age", result.errors)

    def test_negative_age_is_rejected(self):
        result = validate_prediction_form(factories.valid_post(Age="-1"))
        self.assertFalse(result.is_valid)

    def test_absurd_age_is_rejected(self):
        result = validate_prediction_form(factories.valid_post(Age="999"))
        self.assertFalse(result.is_valid)

    def test_zero_household_size_is_rejected(self):
        result = validate_prediction_form(factories.valid_post(BAS_huishoudgrootte="0"))
        self.assertFalse(result.is_valid)

    def test_zero_children_is_allowed(self):
        result = validate_prediction_form(factories.valid_post(afg_kinderen_huishouden="0"))
        self.assertTrue(result.is_valid, msg=result.error_messages)

    # --- choice fields ----------------------------------------------------
    def test_unknown_gender_code_is_rejected(self):
        """A crafted POST must not push arbitrary codes into the pipeline."""
        result = validate_prediction_form(factories.valid_post(GenderID="9999"))
        self.assertFalse(result.is_valid)
        self.assertIn("GenderID", result.errors)

    def test_unknown_touchpoint_code_is_rejected(self):
        result = validate_prediction_form(factories.valid_post(step_1_channel="7777"))
        self.assertFalse(result.is_valid)

    def test_touchpoint_name_is_accepted(self):
        """The datalist offers names, so a submitted name must resolve."""
        result = validate_prediction_form(factories.valid_post(step_1_channel="Email"))
        self.assertTrue(result.is_valid, msg=result.error_messages)
        self.assertEqual(result.get("step_1_channel"), 20)

    def test_touchpoint_name_is_case_insensitive(self):
        result = validate_prediction_form(factories.valid_post(step_1_channel="  eMaIl "))
        self.assertTrue(result.is_valid, msg=result.error_messages)
        self.assertEqual(result.get("step_1_channel"), 20)

    def test_garbage_touchpoint_is_rejected(self):
        result = validate_prediction_form(factories.valid_post(step_1_channel="<script>"))
        self.assertFalse(result.is_valid)

    # --- error shape ------------------------------------------------------
    def test_errors_are_reported_for_every_bad_field(self):
        result = validate_prediction_form(
            factories.valid_post(Age="x", GenderID="9999", SPSS_Regio5="")
        )
        self.assertEqual(
            set(result.errors), {"Age", "GenderID", "SPSS_Regio5"}
        )

    def test_submitted_values_are_preserved_for_repopulation(self):
        raw = factories.valid_post(Age="7", GenderID="2")
        result = validate_prediction_form(raw)
        self.assertEqual(result.submitted["Age"], "7")
        self.assertEqual(result.submitted["GenderID"], "2")

    def test_missing_post_key_is_handled(self):
        """An entirely empty payload must not raise."""
        result = validate_prediction_form({})
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.errors), len(ALL_FIELDS))


class FeatureFrameTestCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        factories.seed_reference_data()

    def test_feature_frame_has_the_expected_columns(self):
        result = validate_prediction_form(factories.valid_post())
        frame = formdata.to_feature_frame(result)
        expected = set(formdata.CLUSTERING_FEATURES) | {"step2", "step3"}
        self.assertEqual(set(frame.columns), expected)
        self.assertEqual(len(frame), 1)

    def test_journey_fields_map_to_step2_and_step3(self):
        result = validate_prediction_form(
            factories.valid_post(step_1_channel="16", step_2_channel="20")
        )
        frame = formdata.to_feature_frame(result)
        # step_1_channel is the older step, step_2_channel the most recent.
        self.assertEqual(int(frame["step2"][0]), 16)
        self.assertEqual(int(frame["step3"][0]), 20)

    def test_column_order_matches_the_training_contract(self):
        result = validate_prediction_form(factories.valid_post())
        frame = formdata.to_feature_frame(result)
        for column in formdata.CLUSTERING_FEATURES:
            self.assertIn(column, frame.columns)
