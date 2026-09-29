"""Seed every lookup table from :mod:`ctmj.services.reference_data`.

This is what makes a fresh clone usable with nothing but ``migrate``: the
dropdown vocabularies and segment descriptions are reference data, not user
data, so they belong in the migration graph rather than in a committed
SQLite binary.

Re-running is safe. The operation is idempotent (update-or-create), so it can be
re-applied after the reference data changes — either by
``manage.py migrate ctmj 0011`` or via ``manage.py seed_reference_data``.
"""

from django.db import migrations

from ctmj.services.reference_data import REFERENCE_TABLES


def seed(apps, schema_editor):
    for model_name, pk_field, rows in REFERENCE_TABLES:
        model = apps.get_model("ctmj", model_name)
        pk = pk_field
        for row in rows:
            model.objects.update_or_create(
                **{pk: row[pk]},
                defaults={k: v for k, v in row.items() if k != pk},
            )


def unseed(apps, schema_editor):
    """Remove only the rows this migration owns, leaving user data intact."""
    for model_name, pk_field, rows in REFERENCE_TABLES:
        model = apps.get_model("ctmj", model_name)
        model.objects.filter(**{f"{pk_field}__in": [row[pk_field] for row in rows]}).delete()


class Migration(migrations.Migration):
    dependencies = [("ctmj", "0010_prediction_run")]

    operations = [migrations.RunPython(seed, unseed)]
