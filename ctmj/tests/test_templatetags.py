"""Template tag and filter behaviour.

These cover the traps that make per-field forms silently misbehave: Django's
dot syntax resolving to a *literal* key, and `vi` locale rendering floats with
a comma that `parseFloat` and CSS both misread.
"""

from __future__ import annotations

from django.template import Context, Template
from django.test import SimpleTestCase

from ctmj.templatetags import ctmj_tags


def render(template: str, **context) -> str:
    return Template("{% load ctmj_tags %}" + template).render(Context(context))


class FieldErrorTests(SimpleTestCase):
    ERRORS = {"Age": "Vui lòng nhập một số nguyên hợp lệ.", "GenderID": "Thiếu."}

    def test_returns_the_message_for_the_field(self):
        self.assertEqual(
            ctmj_tags.field_error(self.ERRORS, "Age"),
            "Vui lòng nhập một số nguyên hợp lệ.",
        )

    def test_returns_empty_for_a_valid_field(self):
        self.assertEqual(ctmj_tags.field_error(self.ERRORS, "step_1_channel"), "")

    def test_tolerates_none(self):
        self.assertEqual(ctmj_tags.field_error(None, "Age"), "")

    def test_django_dot_syntax_would_have_failed(self):
        """Documents why the filter exists.

        ``{{ errors.name }}`` looks up the literal key ``"name"``, so it can
        never resolve a field whose name is only available as a variable.
        """
        self.assertEqual(render("[{{ errors.name }}]", errors=self.ERRORS), "[]")
        self.assertEqual(
            render("[{{ errors|field_error:name }}]", errors=self.ERRORS, name="Age"),
            "[Vui lòng nhập một số nguyên hợp lệ.]",
        )

    def test_usable_inside_an_include_with_a_name_argument(self):
        out = render(
            "{% with name='Age' %}{{ errors|field_error:name }}{% endwith %}",
            errors=self.ERRORS,
        )
        self.assertIn("số nguyên", out)


class FieldValueTests(SimpleTestCase):
    def test_returns_the_submitted_value(self):
        self.assertEqual(ctmj_tags.field_value({"Age": "41"}, "Age"), "41")

    def test_returns_empty_when_absent(self):
        self.assertEqual(ctmj_tags.field_value({"Age": "41"}, "SPSS_Regio5"), "")

    def test_tolerates_none(self):
        self.assertEqual(ctmj_tags.field_value(None, "Age"), "")


class IsSelectedTests(SimpleTestCase):
    """POST data is text, lookup codes are integers — they must still match."""

    def test_string_and_int_compare_equal(self):
        self.assertTrue(ctmj_tags.is_selected("2", 2))

    def test_int_and_string_compare_equal(self):
        self.assertTrue(ctmj_tags.is_selected(2, "2"))

    def test_non_matching_values(self):
        self.assertFalse(ctmj_tags.is_selected("2", 3))

    def test_empty_submission_never_selects(self):
        self.assertFalse(ctmj_tags.is_selected("", 1))
        self.assertFalse(ctmj_tags.is_selected(None, 1))

    def test_works_through_the_template_chain(self):
        out = render(
            "{% for o in options %}"
            "{% if submitted|field_value:field|is_selected:o.value %}[x]{% endif %}"
            "{% endfor %}",
            submitted={"GenderID": "2"},
            field="GenderID",
            options=[{"value": 1}, {"value": 2}],
        )
        self.assertEqual(out, "[x]")


class PercentageTests(SimpleTestCase):
    """Machine-consumed numbers must use a dot, never the vi-VN comma."""

    def test_uses_a_dot_decimal_separator(self):
        self.assertEqual(ctmj_tags.percentage(15.5), "15.50")
        self.assertNotIn(",", ctmj_tags.percentage(15.5))

    def test_rounds_to_two_places(self):
        self.assertEqual(ctmj_tags.percentage(20.104), "20.10")

    def test_handles_zero(self):
        self.assertEqual(ctmj_tags.percentage(0), "0.00")

    def test_handles_non_numeric_input(self):
        self.assertEqual(ctmj_tags.percentage("n/a"), "0.00")
        self.assertEqual(ctmj_tags.percentage(None), "0.00")

    def test_decimal_supports_variable_precision(self):
        self.assertEqual(ctmj_tags.decimal(1.23456, 3), "1.235")
        self.assertEqual(ctmj_tags.decimal(1.23456, 1), "1.2")

    def test_decimal_handles_non_numeric_input(self):
        self.assertEqual(ctmj_tags.decimal("", 3), "")

    def test_filter_chain_in_a_template(self):
        out = render("{{ v|percentage }}", v=13.479)
        self.assertEqual(out, "13.48")
        self.assertNotIn(",", out)
