"""Reference-data integrity and ORM behaviour.

The lookup tables are the vocabulary the trained pipeline was fitted on. A
typo here does not raise — it silently produces a wrong label — so these tests
assert completeness and shape rather than just CRUD.
"""

from __future__ import annotations

from django.test import TestCase

from ctmj.models import (
    AFG_sk2015,
    BAS_bruto_jaarinkomen,
    BAS_werkzaamheid_resp,
    BAS_voltooide_opleiding8_resp,
    GenderID,
    PredictionRun,
    SPSS_Lifestage,
    SPSS_Regio5,
    cluster_info,
    type_touch,
)
from ctmj.services import formdata, reference_data
from ctmj.tests import factories

#: Codes the trained pipeline was fitted on. If one of these goes missing the
#: model cannot be fed a valid input at all.
REQUIRED_TOUCHPOINT_CODES = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22,
}
REQUIRED_CLUSTER_IDS = {-1, 0, 1, 2}


class ReferenceDataTestCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        factories.seed_reference_data()

    # --- completeness ----------------------------------------------------
    def test_every_table_is_populated(self):
        tables = [
            GenderID,
            BAS_werkzaamheid_resp,
            SPSS_Regio5,
            BAS_bruto_jaarinkomen,
            AFG_sk2015,
            BAS_voltooide_opleiding8_resp,
            SPSS_Lifestage,
            type_touch,
            cluster_info,
        ]
        for model in tables:
            with self.subTest(model=model.__name__):
                self.assertGreater(model.objects.count(), 0)

    def test_row_counts_match_the_source_of_truth(self):
        for model_name, _pk, rows in reference_data.REFERENCE_TABLES:
            with self.subTest(model=model_name):
                from django.apps import apps

                model = apps.get_model("ctmj", model_name)
                self.assertEqual(model.objects.count(), len(rows))

    def test_all_touchpoint_codes_required_by_the_model_are_present(self):
        present = set(type_touch.objects.values_list("code", flat=True))
        self.assertTrue(
            REQUIRED_TOUCHPOINT_CODES.issubset(present),
            msg=f"missing: {sorted(REQUIRED_TOUCHPOINT_CODES - present)}",
        )

    def test_all_cluster_ids_are_present(self):
        present = set(cluster_info.objects.values_list("cluster_id", flat=True))
        self.assertEqual(present, REQUIRED_CLUSTER_IDS)

    def test_every_touchpoint_has_a_description(self):
        for touch in type_touch.objects.all():
            with self.subTest(code=touch.code):
                self.assertTrue(touch.description)
                self.assertNotIn("#", touch.description)

    def test_every_cluster_has_a_name_and_description(self):
        for cluster in cluster_info.objects.all():
            with self.subTest(cluster=cluster.cluster_id):
                self.assertTrue(cluster.name)
                self.assertTrue(cluster.description)

    def test_no_label_contains_the_unicode_replacement_character(self):
        """Guards against the mojibake that affected src/dictionaries.py."""
        models = [
            (GenderID, "gender_name"),
            (BAS_werkzaamheid_resp, "name"),
            (SPSS_Regio5, "name"),
            (BAS_bruto_jaarinkomen, "name"),
            (AFG_sk2015, "name"),
            (BAS_voltooide_opleiding8_resp, "name"),
            (SPSS_Lifestage, "name"),
            (type_touch, "name"),
        ]
        for model, field in models:
            for obj in model.objects.all():
                with self.subTest(model=model.__name__, pk=obj.pk):
                    value = getattr(obj, field)
                    self.assertNotIn("\ufffd", value)
                    self.assertNotIn("?", value[:1])

    # --- consistency with the validator ----------------------------------
    def test_every_cluster_id_has_a_human_label(self):
        for cluster_id in REQUIRED_CLUSTER_IDS:
            with self.subTest(cluster=cluster_id):
                self.assertIn(cluster_id, reference_data.CLUSTER_LABELS)

    def test_income_labels_avoid_raw_euro_glyphs(self):
        """Plain text only — the UI is Vietnamese and must render everywhere."""
        for row in reference_data.INCOMES:
            with self.subTest(code=row["code"]):
                self.assertNotIn("\u20ac", row["name"])


class OcrudTestCase(TestCase):
    """The original suite covered plain CRUD; keep that coverage."""

    def test_gender_create_read_update_delete(self):
        obj = GenderID.objects.create(gender_code=99, gender_name="Test")
        self.assertEqual(GenderID.objects.get(gender_code=99).gender_name, "Test")
        obj.gender_name = "Updated"
        obj.save()
        self.assertEqual(GenderID.objects.get(gender_code=99).gender_name, "Updated")
        obj.delete()
        self.assertFalse(GenderID.objects.filter(gender_code=99).exists())

    def test_touchpoint_create_read_update_delete(self):
        obj = type_touch.objects.create(code=99, name="Test", description="d")
        self.assertEqual(type_touch.objects.get(code=99).name, "Test")
        obj.name = "Updated"
        obj.save()
        self.assertEqual(type_touch.objects.get(code=99).name, "Updated")
        obj.delete()
        self.assertFalse(type_touch.objects.filter(code=99).exists())

    def test_cluster_info_create_read_update_delete(self):
        obj = cluster_info.objects.create(cluster_id=9, name="n", description="d")
        self.assertEqual(cluster_info.objects.get(cluster_id=9).name, "n")
        obj.delete()
        self.assertFalse(cluster_info.objects.filter(cluster_id=9).exists())

    def test_default_ordering_is_by_code(self):
        factories.seed_reference_data()
        codes = list(type_touch.objects.values_list("code", flat=True))
        self.assertEqual(codes, sorted(codes))

    def test_str_is_informative(self):
        factories.seed_reference_data()
        touch = type_touch.objects.get(code=20)
        self.assertIn("20", str(touch))
        self.assertIn(touch.name, str(touch))


class PredictionRunTestCase(TestCase):
    def test_record_round_trip(self):
        run = PredictionRun.objects.create(
            cluster_id=1,
            top_channel=20,
            top_probability=41.25,
            age=42,
            household_size=4,
            children=2,
            step_previous=16,
            step_last=20,
        )
        self.assertEqual(PredictionRun.objects.get(pk=run.pk).top_probability, 41.25)
        self.assertIn("41.2", str(run))

    def test_newest_first_ordering(self):
        for i in range(3):
            PredictionRun.objects.create(
                cluster_id=0,
                top_channel=1,
                top_probability=float(i),
                age=30,
                household_size=2,
                children=0,
                step_previous=1,
                step_last=2,
            )
        ordered = list(PredictionRun.objects.values_list("pk", flat=True))
        self.assertEqual(ordered, sorted(ordered, reverse=True))


class ContractTestCase(TestCase):
    """The pipeline's feature lists must line up with the models."""

    def test_clustering_features_are_unique(self):
        self.assertEqual(
            len(formdata.CLUSTERING_FEATURES), len(set(formdata.CLUSTERING_FEATURES))
        )

    def test_prediction_features_are_unique(self):
        self.assertEqual(
            len(formdata.PREDICTION_FEATURES), len(set(formdata.PREDICTION_FEATURES))
        )

    def test_clustering_features_exclude_the_journey_steps(self):
        for step in ("step2", "step3", "step1", "final_label"):
            self.assertNotIn(step, formdata.CLUSTERING_FEATURES)

    def test_prediction_features_include_the_cluster_label(self):
        self.assertIn("final_label", formdata.PREDICTION_FEATURES)

    def test_every_numeric_field_is_validated(self):
        self.assertEqual(
            set(formdata.NUMERIC_FIELDS), {"Age", "BAS_huishoudgrootte", "afg_kinderen_huishouden"}
        )

    def test_every_field_has_a_label(self):
        for field in formdata.ALL_FIELDS:
            self.assertIn(field, formdata.FIELD_LABELS)

    def test_every_field_is_classified(self):
        """No field may fall through unvalidated."""
        for field in formdata.ALL_FIELDS:
            self.assertTrue(
                field in formdata.NUMERIC_FIELDS or field in formdata.CHOICE_FIELDS,
                msg=f"{field} is neither numeric nor a choice",
            )
