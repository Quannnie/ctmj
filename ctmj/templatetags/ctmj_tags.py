"""Template tags and filters used by the prediction form.

Django's dot syntax resolves to a *literal* key, so ``{{ errors.name }}``
inside a partial included with ``name='Age'`` looks up ``errors["name"]`` and
finds nothing. These filters take the field name as an argument instead, which
is what makes per-field error rendering possible.
"""

from __future__ import annotations

from typing import Any, Mapping

from django import template

register = template.Library()


@register.filter
def field_error(errors: Mapping[str, str] | None, field_name: str) -> str:
    """Return the validation message for ``field_name``, or an empty string."""
    if not errors:
        return ""
    return errors.get(field_name, "") or ""


@register.filter
def field_value(values: Mapping[str, str] | None, field_name: str) -> str:
    """Return the raw submitted value for ``field_name``, or an empty string.

    Used to repopulate inputs after a failed submission.
    """
    if not values:
        return ""
    return values.get(field_name, "") or ""


@register.filter
def is_selected(submitted: Any, option: Any) -> bool:
    """True when ``option`` equals the submitted value for this field.

    Both sides are compared as strings because POST data is always text while
    lookup-table codes are integers; comparing them directly would never match
    and a resubmitted select would silently reset to the first option.
    """
    if submitted in (None, ""):
        return False
    return str(submitted) == str(option)


@register.filter
def percentage(value: Any) -> str:
    """Format a 0-100 float with a dot decimal separator.

    Django localises floats with a comma under ``vi-VN``, which is correct on
    screen but breaks ``parseFloat`` in JavaScript and ``width`` in CSS. Values
    consumed by script or style must be built here, in Python.
    """
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "0.00"


@register.filter
def decimal(value: Any, places: int = 3) -> str:
    """Like :func:`percentage` but for arbitrary precision (e.g. distances)."""
    try:
        return f"{float(value):.{int(places)}f}"
    except (TypeError, ValueError):
        return ""


@register.filter
def get_item(mapping: Mapping[str, Any] | None, key: Any) -> str:
    """Look ``key`` up in ``mapping``, returning ``""`` when absent.

    Needed because the model registry reports a *reason per artefact*: a 403,
    an absent file and a scikit-learn version clash all land in
    ``status.missing``, and each needs a different remedy. Rendering that
    reason next to the filename it belongs to means indexing a dict from a
    template, and Django's dot syntax only does literal keys on the object it
    is given — not a variable lookup into a mapping.

    A missing key yields ``""`` rather than raising, because the caller is
    iterating a *different* list (``missing``) than the one the mapping was
    built from, and a file can legitimately be absent with no recorded error.
    """
    if not mapping:
        return ""
    try:
        return mapping.get(key) or ""
    except AttributeError:
        return ""
