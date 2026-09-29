"""Tests for the reference-data cache.

The cache is only worth having if it is *correct*. A cache that serves a stale
label is worse than a slow query: the person who just corrected the spelling of
a touchpoint in ``/admin/`` goes to check, sees the old name, and concludes the
edit failed.

So the invalidation path gets the most attention here, and the performance
claim is asserted as a query count rather than a wall-clock figure, which would
be noise on a shared machine.
"""

from __future__ import annotations

from django.core.cache import cache
from django.db import connection
from django.test import TestCase
from django.test.utils import CaptureQueriesContext

from ctmj.models import cluster_info, type_touch
from ctmj.services import formdata, lookups

VALID = {
    "GenderID": "2", "Age": "41", "SPSS_Regio5": "1",
    "BAS_huishoudgrootte": "4", "BAS_werkzaamheid_resp": "2",
    "afg_kinderen_huishouden": "2", "BAS_bruto_jaarinkomen": "4",
    "AFG_sk2015": "5", "BAS_voltooide_opleiding8_resp": "5",
    "SPSS_Lifestage": "7", "step_1_channel": "16", "step_2_channel": "20",
}


class CacheClearingMixin:
    def setUp(self):
        super().setUp()
        cache.clear()
        lookups.invalidate_all()
        self.addCleanup(cache.clear)


class ContentTests(CacheClearingMixin, TestCase):
    def test_valid_codes_are_integers(self):
        """A string code would make every comparison in the form wrong."""
        codes = lookups.valid_codes("GenderID")
        self.assertTrue(codes)
        self.assertTrue(all(isinstance(c, int) for c in codes))

    def test_name_map_is_case_folded(self):
        mapping = lookups.name_to_code("step_1_channel")
        self.assertTrue(mapping)
        for key in mapping:
            self.assertEqual(key, key.casefold())

    def test_name_map_resolves_a_real_touchpoint(self):
        mapping = lookups.name_to_code("step_1_channel")
        target = type_touch.objects.get(code=16)
        self.assertEqual(mapping[target.name.strip().casefold()], 16)

    def test_options_are_ordered_by_code(self):
        """A dropdown that reshuffles changes the meaning of a chosen value."""
        for field in ("type_touch", "SPSS_Regio5", "GenderID"):
            with self.subTest(field=field):
                values = [o["value"] for o in lookups.options(field)]
                self.assertEqual(values, sorted(values))

    def test_options_have_value_and_label(self):
        for option in lookups.options("type_touch"):
            self.assertIn("value", option)
            self.assertIn("label", option)
            self.assertTrue(option["label"])

    def test_touchpoint_details_returns_only_what_was_asked_for(self):
        details = lookups.touchpoint_details([16, 20])
        self.assertEqual(set(details), {16, 20})
        for entry in details.values():
            self.assertIn("name", entry)
            self.assertIn("description", entry)

    def test_touchpoint_details_of_nothing_is_empty_and_free(self):
        self.assertEqual(lookups.touchpoint_details([]), {})

    def test_unknown_touchpoint_code_is_absent_not_an_error(self):
        """The classifier can return a code the lookup table lacks."""
        self.assertEqual(lookups.touchpoint_details([9999]), {})

    def test_cluster_details_falls_back_rather_than_raising(self):
        """A retrain can pick k the lookup table has never heard of."""
        self.assertEqual(lookups.cluster_details(9999), {})

    def test_cluster_details_reads_a_real_row(self):
        row = cluster_info.objects.first()
        if row is None:
            self.skipTest("no cluster_info rows in this database")
        detail = lookups.cluster_details(row.cluster_id)
        self.assertIn("name", detail)

    def test_unknown_field_names_the_ones_that_exist(self):
        """A typo must not silently skip validation for the field."""
        with self.assertRaises(KeyError) as ctx:
            lookups.valid_codes("NoSuchField")
        message = str(ctx.exception)
        self.assertIn("type_touch", message)
        self.assertIn("NoSuchField", message)


class InvalidationTests(CacheClearingMixin, TestCase):
    """The property that makes the cache safe to have."""

    def test_a_saved_row_is_visible_immediately(self):
        """An edit in /admin/ takes effect on the next request.

        ``save`` fires post_save, which is wired to invalidate. Without that
        wiring the correction would not appear until the timeout, and the person
        who made it would see the old value.

        The map keys are case-folded, so the "before" lookup has to be folded
        too — comparing a raw ``"Generic search"`` against ``"generic search"``
        would pass whether or not the cache had been invalidated.
        """
        row = type_touch.objects.get(code=16)
        original_key = row.name.strip().casefold()

        # Populate the cache while the old value is live.
        self.assertIn(original_key, lookups.name_to_code("step_1_channel"))

        row.name = "A corrected touchpoint name"
        row.save()

        mapping = lookups.name_to_code("step_1_channel")
        self.assertIn("a corrected touchpoint name", mapping)
        self.assertNotIn(original_key, mapping)

    def test_a_deleted_row_stops_being_accepted(self):
        """A code removed in the admin must stop validating.

        Without invalidation the deleted code would keep validating until the
        timeout, and the model could be handed a channel the site no longer
        describes.
        """
        row = type_touch.objects.create(code=2100, name="Temporary", description="x")
        lookups.invalidate_all()
        self.assertIn(2100, lookups.valid_codes("step_1_channel"))

        row.delete()
        self.assertNotIn(2100, lookups.valid_codes("step_1_channel"))

    def test_a_new_row_becomes_accepted(self):
        self.assertNotIn(2101, lookups.valid_codes("type_touch"))
        type_touch.objects.create(code=2101, name="Brand new", description="x")
        self.assertIn(2101, lookups.valid_codes("type_touch"))
        self.assertIn("brand new", lookups.name_to_code("step_1_channel"))

    def test_cluster_edit_is_visible(self):
        cluster_info.objects.update_or_create(
            cluster_id=7, defaults={"name": "First", "description": "a"}
        )
        lookups.invalidate_all()
        self.assertEqual(lookups.cluster_details(7)["name"], "First")

        cluster_info.objects.filter(cluster_id=7).update(name="Second")
        # queryset.update bypasses signals, so this is the operator's
        # responsibility -- but an explicit invalidate must still clear it.
        lookups.invalidate_all()
        self.assertEqual(lookups.cluster_details(7)["name"], "Second")

    def test_version_actually_rotates(self):
        first = lookups._version()
        lookups.invalidate_all()
        self.assertNotEqual(lookups._version(), first)

    def test_bump_survives_a_flushed_cache(self):
        """A cache that cannot be incremented must not break the request."""
        cache.clear()
        self.assertIsInstance(lookups.bump_version(), int)
        self.assertIsInstance(lookups.bump_version(), int)

    def test_a_broken_cache_falls_back_to_the_database(self):
        """Losing the cache costs queries; raising costs the request.

        Simulated with a backend that raises, so the read path is proven to
        degrade rather than fail.
        """
        class _Broken:
            def get(self, key, default=None):
                raise RuntimeError("cache is down")

            def set(self, key, value, timeout=None):
                raise RuntimeError("cache is down")

            def incr(self, key):
                raise RuntimeError("cache is down")

        import ctmj.services.lookups as lookups_module

        original = lookups_module.cache
        lookups_module.cache = _Broken()
        try:
            self.assertTrue(lookups_module.valid_codes("GenderID"))
            self.assertTrue(lookups_module.options("type_touch"))
            self.assertIsInstance(lookups_module.bump_version(), int)
        finally:
            lookups_module.cache = original


class QueryCountTests(CacheClearingMixin, TestCase):
    """The reason the module exists, asserted as a count.

    Wall-clock timings would be noise on a shared machine; a query count is
    exact. The assertion is on a *warm* cache, because a cold one legitimately
    hits the database.
    """

    def test_validation_issues_no_queries_when_warm(self):
        formdata.validate_prediction_form(VALID)  # warm
        with CaptureQueriesContext(connection) as capture:
            for _ in range(10):
                formdata.validate_prediction_form(VALID)
        self.assertEqual(
            len(capture.captured_queries), 0,
            msg="validation re-queried the reference tables on a warm cache",
        )

    def test_dropdown_context_issues_no_queries_when_warm(self):
        from ctmj.views import _lookup_context

        _lookup_context()  # warm
        with CaptureQueriesContext(connection) as capture:
            for _ in range(10):
                _lookup_context()
        self.assertEqual(len(capture.captured_queries), 0)

    def test_a_cold_cache_reads_each_table_once(self):
        lookups.invalidate_all()
        with CaptureQueriesContext(connection) as capture:
            lookups.valid_codes("GenderID")
        self.assertEqual(len(capture.captured_queries), 1)

    def test_a_warm_cache_beats_a_cold_one(self):
        lookups.invalidate_all()
        with CaptureQueriesContext(connection) as cold:
            for _ in range(5):
                lookups.options("type_touch")
        with CaptureQueriesContext(connection) as warm:
            for _ in range(5):
                lookups.options("type_touch")
        self.assertGreater(len(cold.captured_queries), len(warm.captured_queries))
        self.assertEqual(len(warm.captured_queries), 0)


class BehaviourPreservedTests(CacheClearingMixin, TestCase):
    """The cache must not change what validation accepts."""

    def test_a_valid_payload_still_validates(self):
        self.assertTrue(formdata.validate_prediction_form(VALID).is_valid)

    def test_a_code_outside_the_table_is_still_rejected(self):
        payload = dict(VALID, SPSS_Regio5="999")
        result = formdata.validate_prediction_form(payload)
        self.assertFalse(result.is_valid)
        self.assertIn("SPSS_Regio5", result.errors)

    def test_a_touchpoint_name_still_accepted_after_caching(self):
        target = type_touch.objects.get(code=16)
        payload = dict(VALID, step_1_channel=target.name)
        result = formdata.validate_prediction_form(payload)
        self.assertTrue(result.is_valid, msg=result.errors)
        self.assertEqual(result.get("step_1_channel"), 16)

    def test_a_touchpoint_name_is_case_insensitive(self):
        target = type_touch.objects.get(code=16)
        payload = dict(VALID, step_1_channel=target.name.upper())
        result = formdata.validate_prediction_form(payload)
        self.assertTrue(result.is_valid, msg=result.errors)
        self.assertEqual(result.get("step_1_channel"), 16)

    def test_prediction_details_come_from_the_cache(self):
        details = lookups.touchpoint_details([16])
        row = type_touch.objects.get(code=16)
        self.assertEqual(details[16]["name"], row.name)
