"""HTTP behaviour, including every failure state the old view lacked.

Covers the 503 path (models not loaded) and the 400 path (invalid input) that
the original implementation collapsed into a single 200 response carrying a
stringified Python exception.
"""

from __future__ import annotations

from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from ctmj.apps import CtmjConfig
from ctmj.services import predictor
from ctmj.services.registry import LoadState, RegistryStatus
from ctmj.tests import factories


class ViewTestCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        factories.seed_reference_data()

    def setUp(self):
        # Every test starts from an empty registry; individual tests opt in.
        self.original = CtmjConfig.registry
        self.addCleanup(self._restore)

    def _restore(self):
        CtmjConfig.registry = self.original

    def _offline_registry(self):
        from ctmj.services.registry import ModelRegistry

        registry = ModelRegistry(
            {
                "MODEL_SOURCE": "none",
                "MODEL_DIR": ".",
                "HF_REPO": "",
                "MODEL_FILES": {"dbscan": "x.pkl"},
            }
        )
        CtmjConfig.registry = registry
        return registry

    def _ready_registry(self):
        from ctmj.services.registry import ModelRegistry

        registry = ModelRegistry(
            {
                "MODEL_SOURCE": "none",
                "MODEL_DIR": ".",
                "HF_REPO": "",
                "MODEL_FILES": {"dbscan": "x.pkl"},
            }
        )
        registry._status = RegistryStatus(state=LoadState.READY, message="ready")
        CtmjConfig.registry = registry
        factories.install_models(registry)
        return registry


class HomeViewTests(ViewTestCase):
    def test_home_renders(self):
        self._offline_registry()
        response = self.client.get(reverse("home"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "home.html")
        self.assertContains(response, "CJPS")

    def test_home_reports_the_touchpoint_count_from_the_database(self):
        self._offline_registry()
        response = self.client.get(reverse("home"))
        self.assertEqual(response.context["touchpoint_count"], 20)

    def test_home_shows_model_status(self):
        self._offline_registry()
        response = self.client.get(reverse("home"))
        self.assertIn("registry_status", response.context)
        self.assertContains(response, "Mô hình", status_code=200)

    def test_home_marks_the_current_nav_item(self):
        self._offline_registry()
        response = self.client.get(reverse("home"))
        self.assertContains(response, 'aria-current="page"')


class PredictFormTests(ViewTestCase):
    def test_get_renders_the_form(self):
        self._offline_registry()
        response = self.client.get(reverse("predict"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "predict.html")
        self.assertContains(response, 'name="Age"')
        self.assertContains(response, 'name="step_1_channel"')

    def test_every_dropdown_is_populated(self):
        self._offline_registry()
        response = self.client.get(reverse("predict"))
        for key in ("genders", "jobs", "regions", "incomes",
                    "social_classes", "educations", "lifestages", "touchpoints"):
            with self.subTest(context=key):
                self.assertTrue(response.context[key])

    def test_form_carries_a_csrf_token(self):
        self._offline_registry()
        response = self.client.get(reverse("predict"))
        self.assertContains(response, "csrfmiddlewaretoken")

    def test_offline_notice_is_shown(self):
        self._offline_registry()
        response = self.client.get(reverse("predict"))
        self.assertContains(response, "Mô hình chưa sẵn sàng")


class PredictSubmissionTests(ViewTestCase):
    def test_valid_submission_returns_results(self):
        self._ready_registry()
        response = self.client.post(reverse("predict"), factories.valid_post())
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.context)
        self.assertContains(response, "Kết quả")

    def test_results_list_three_channels(self):
        self._ready_registry()
        response = self.client.post(reverse("predict"), factories.valid_post())
        prediction = response.context["prediction"]
        self.assertEqual(len(prediction.channels), 3)

    def test_results_show_probabilities(self):
        self._ready_registry()
        response = self.client.post(reverse("predict"), factories.valid_post())
        self.assertContains(response, "data-count-to")
        self.assertContains(response, "meter__fill")

    def test_results_show_the_segment_metadata(self):
        from django.utils.html import escape

        self._ready_registry()
        response = self.client.post(reverse("predict"), factories.valid_post())
        prediction = response.context["prediction"]
        # Cluster names contain '&', which the template escapes.
        self.assertContains(response, escape(prediction.cluster_name))
        self.assertContains(response, escape(prediction.cluster_label))

    def test_animated_values_use_a_dot_decimal_separator(self):
        """vi-VN localises floats to a comma, which breaks parseFloat and CSS.

        ``data-count-to="15,50"`` would animate to 15% and
        ``width: 15,50%`` is invalid, leaving the bars permanently empty.
        """
        self._ready_registry()
        response = self.client.post(reverse("predict"), factories.valid_post())
        content = response.content.decode()

        import re

        for value in re.findall(r'data-count-to="([^"]+)"', content):
            with self.subTest(value=value):
                self.assertNotIn(",", value)
                float(value)  # must parse as a float

        for value in re.findall(r'data-fill="([^"]+)"', content):
            with self.subTest(value=value):
                self.assertNotIn(",", value)

        for value in re.findall(r"--final: ([^;]+);", content):
            with self.subTest(value=value):
                self.assertNotIn(",", value)
    def test_results_are_focusable_for_assistive_tech(self):
        self._ready_registry()
        response = self.client.post(reverse("predict"), factories.valid_post())
        self.assertContains(response, 'id="prediction-results"')
        self.assertContains(response, 'tabindex="-1"')


class PredictValidationTests(ViewTestCase):
    def test_invalid_submission_returns_400(self):
        self._ready_registry()
        response = self.client.post(reverse("predict"), factories.valid_post(Age="abc"))
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("prediction", response.context)

    def test_error_summary_is_rendered(self):
        self._ready_registry()
        response = self.client.post(reverse("predict"), factories.valid_post(Age="abc"))
        self.assertTrue(response.context["error_summary"])
        self.assertContains(response, "data-error-summary", status_code=400)

    def test_error_summary_focuses_for_assistive_tech(self):
        self._ready_registry()
        response = self.client.post(reverse("predict"), factories.valid_post(Age="abc"))
        self.assertContains(response, 'role="alert"', status_code=400)

    def test_field_error_is_announced_via_aria_describedby(self):
        self._ready_registry()
        response = self.client.post(reverse("predict"), factories.valid_post(Age="abc"))
        self.assertContains(response, 'aria-invalid="true"', status_code=400)
        self.assertContains(response, 'id="e-Age"', status_code=400)

    def test_other_fields_are_repopulated_after_an_error(self):
        """One bad field must not discard the other eleven."""
        self._ready_registry()
        response = self.client.post(
            reverse("predict"), factories.valid_post(Age="abc", GenderID="2")
        )
        self.assertEqual(response.context["submitted"]["GenderID"], "2")
        self.assertContains(response, 'id="f-GenderID"', status_code=400)
        self.assertContains(response, 'id="e-Age"', status_code=400)

    def test_dropdowns_stay_selected_after_an_error(self):
        """A rejected submission must not silently reset the user's choices.

        POST values are strings and lookup codes are integers, so a naive
        comparison never matches and every select falls back to its first
        option — the user would fix one field and unknowingly resubmit a
        different household profile.
        """
        self._ready_registry()
        response = self.client.post(
            reverse("predict"),
            factories.valid_post(Age="abc", GenderID="2", SPSS_Regio5="3", AFG_sk2015="4"),
        )
        content = response.content.decode()
        self.assertIn('<option value="2" selected>', content)
        self.assertIn('<option value="3" selected>', content)
        self.assertIn('<option value="4" selected>', content)

    def test_dropdowns_stay_selected_after_a_successful_prediction(self):
        self._ready_registry()
        response = self.client.post(reverse("predict"), factories.valid_post())
        content = response.content.decode()
        self.assertIn('<option value="1" selected>', content)  # GenderID
        self.assertIn('<option value="7" selected>', content)  # SPSS_Lifestage

    def test_out_of_range_code_is_rejected(self):
        self._ready_registry()
        response = self.client.post(reverse("predict"), factories.valid_post(GenderID="4242"))
        self.assertEqual(response.status_code, 400)
        self.assertIn("GenderID", response.context["errors"])

    def test_empty_post_is_rejected_without_a_crash(self):
        self._ready_registry()
        response = self.client.post(reverse("predict"), {})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(response.context["errors"]), 12)


class PredictUnavailableTests(ViewTestCase):
    def test_offline_submission_returns_503(self):
        """The old view returned 200 with a red alert for this case."""
        self._offline_registry()
        response = self.client.post(reverse("predict"), factories.valid_post())
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("prediction", response.context)
        self.assertTrue(response.context["error_summary"])

    def test_pipeline_failure_returns_500_with_a_readable_message(self):
        from unittest import mock

        self._ready_registry()
        with mock.patch.object(
            predictor, "predict", side_effect=predictor.PredictionError("Mô hình lỗi")
        ):
            response = self.client.post(reverse("predict"), factories.valid_post())
        self.assertEqual(response.status_code, 500)
        self.assertContains(response, "Mô hình lỗi", status_code=500)

    def test_unexpected_exception_does_not_leak_internals(self):
        from unittest import mock

        self._ready_registry()
        with mock.patch.object(
            predictor, "predict", side_effect=RuntimeError("secret /etc/passwd")
        ):
            response = self.client.post(reverse("predict"), factories.valid_post())
        self.assertEqual(response.status_code, 500)
        self.assertNotContains(response, "passwd", status_code=500)

    def test_get_still_works_while_offline(self):
        self._offline_registry()
        self.assertEqual(self.client.get(reverse("predict")).status_code, 200)


class RegistryWiringTests(SimpleTestCase):
    """The views read ``CtmjConfig.registry``; ``ready()`` must populate it.

    Regression guard: assigning ``self.registry`` inside ``ready()`` creates an
    *instance* attribute, leaving the class attribute ``None`` — so every view
    silently behaved as if no model existed, with no error anywhere.
    """

    def test_ready_populates_the_class_level_registry(self):
        from django.apps import apps

        from ctmj.apps import CtmjConfig

        config = apps.get_app_config("ctmj")
        self.assertIsInstance(config, CtmjConfig)
        self.assertIsNotNone(
            CtmjConfig.registry,
            "CtmjConfig.registry is None; views will never see the models",
        )

    def test_views_and_the_registry_agree_on_the_same_object(self):
        from ctmj.apps import CtmjConfig
        from ctmj.views import _registry

        self.assertIs(_registry(), CtmjConfig.registry)

    def test_registry_exposes_a_serialisable_status(self):
        from ctmj.views import _registry

        status = _registry().status
        self.assertIn(status.state.value, {"idle", "loading", "ready", "failed", "disabled"})
        self.assertIsInstance(status.as_dict(), dict)


class HealthTests(ViewTestCase):
    def test_health_reports_degraded_when_offline(self):
        self._offline_registry()
        response = self.client.get(reverse("health"))
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "degraded")

    def test_health_reports_ok_when_ready(self):
        self._ready_registry()
        response = self.client.get(reverse("health"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["models"]["state"], "ready")


class NotFoundTests(TestCase):
    def test_custom_404_view_renders(self):
        response = self.client.get("/definitely-not-a-page/", status=404)
        self.assertEqual(response.status_code, 404)
        self.assertTemplateUsed(response, "404.html")
        self.assertContains(response, "Không tìm thấy trang", status_code=404)

    def test_404_offers_a_way_back(self):
        response = self.client.get("/definitely-not-a-page/", status=404)
        self.assertContains(response, reverse("home"), status_code=404)
        self.assertContains(response, reverse("predict"), status_code=404)


class TemplateContractTests(ViewTestCase):
    """Templates must degrade gracefully and stay accessible."""

    def test_base_has_a_skip_link(self):
        self._offline_registry()
        response = self.client.get(reverse("home"))
        self.assertContains(response, 'class="skip-link"')
        self.assertContains(response, 'id="main"')

    def test_base_has_a_favicon_and_meta_description(self):
        self._offline_registry()
        response = self.client.get(reverse("home"))
        self.assertContains(response, 'rel="icon"')
        self.assertContains(response, 'name="description"')

    def test_no_javascript_alert_dialogs(self):
        """window.alert is a design and accessibility failure."""
        self._ready_registry()
        response = self.client.post(reverse("predict"), factories.valid_post())
        self.assertNotContains(response, "alert(")
        self.assertNotContains(response, "window.alert")

    def test_static_javascript_is_referenced_and_exists(self):
        """The old base.html loaded a script that was never committed."""
        from pathlib import Path

        from django.conf import settings as django_settings

        self._offline_registry()
        response = self.client.get(reverse("home"))
        self.assertContains(response, "js/app.js")
        path = Path(django_settings.BASE_DIR) / "static" / "js" / "app.js"
        self.assertTrue(path.is_file(), "static/js/app.js is referenced but missing")

    def test_static_css_is_referenced_and_exists(self):
        from pathlib import Path

        from django.conf import settings as django_settings

        self._offline_registry()
        response = self.client.get(reverse("home"))
        self.assertContains(response, "css/main.css")
        path = Path(django_settings.BASE_DIR) / "static" / "css" / "main.css"
        self.assertTrue(path.is_file(), "static/css/main.css is referenced but missing")

    def test_error_summary_links_to_each_field(self):
        self._ready_registry()
        response = self.client.post(reverse("predict"), factories.valid_post(Age="abc"))
        content = response.content.decode()
        self.assertIn("Tuổi", content)
        self.assertIn("e-Age", content)

    def test_404_page_also_carries_the_skip_link(self):
        response = self.client.get("/definitely-not-a-page/", status=404)
        self.assertContains(response, 'class="skip-link"', status_code=404)

    def test_template_comments_do_not_leak_into_the_page(self):
        """Django's ``{# ... #}`` only matches within a single line.

        A multi-line ``{# ... #}`` block is therefore *not* a comment: Django
        consumes up to the first ``#}`` and emits the remaining lines as page
        text, so a section banner appears at the top of the site. Long comments
        must use ``{% comment %}`` instead.
        """
        self._offline_registry()
        for url in (reverse("home"), reverse("predict")):
            with self.subTest(url=url):
                content = self.client.get(url).content.decode()
                self.assertNotIn("{#", content)
                self.assertNotIn("#}", content)
                self.assertNotIn("{% comment", content)
                self.assertNotIn("{% endcomment", content)

    def test_no_unrendered_django_tags(self):
        self._ready_registry()
        for url in (reverse("home"), reverse("predict")):
            with self.subTest(url=url):
                content = self.client.get(url).content.decode()
                self.assertNotIn("{{", content)
                self.assertNotIn("{%", content)
