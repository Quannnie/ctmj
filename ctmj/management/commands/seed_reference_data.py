"""Re-sync the lookup tables from :mod:`ctmj.services.reference_data`.

The data migration ``0011_seed_reference_data`` already populates a new
database. This command exists for the other cases: syncing an existing database
after the reference data changes, and repairing a database whose lookup tables
were emptied by hand.
"""

from __future__ import annotations

from django.core.management.base import BaseCommand
from django.db import transaction

from ctmj.models import (
    AFG_sk2015,
    BAS_bruto_jaarinkomen,
    BAS_werkzaamheid_resp,
    BAS_voltooide_opleiding8_resp,
    GenderID,
    SPSS_Lifestage,
    SPSS_Regio5,
    cluster_info,
    type_touch,
)
from ctmj.services.reference_data import REFERENCE_TABLES


class Command(BaseCommand):
    help = "Populate or refresh the reference/lookup tables."

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Report what would change without writing to the database.",
        )

    @transaction.atomic
    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        # Resolve through the historical registry so this keeps working if a
        # model is renamed later.
        live_models = {
            "GenderID": GenderID,
            "BAS_werkzaamheid_resp": BAS_werkzaamheid_resp,
            "SPSS_Regio5": SPSS_Regio5,
            "BAS_bruto_jaarinkomen": BAS_bruto_jaarinkomen,
            "AFG_sk2015": AFG_sk2015,
            "BAS_voltooide_opleiding8_resp": BAS_voltooide_opleiding8_resp,
            "SPSS_Lifestage": SPSS_Lifestage,
            "type_touch": type_touch,
            "cluster_info": cluster_info,
        }

        total_created = total_updated = 0
        for model_name, pk_field, rows in REFERENCE_TABLES:
            model = live_models[model_name]
            created = updated = 0
            for row in rows:
                pk_value = row[pk_field]
                defaults = {k: v for k, v in row.items() if k != pk_field}
                if dry_run:
                    if model.objects.filter(**{pk_field: pk_value}).exists():
                        updated += 1
                    else:
                        created += 1
                    continue
                _obj, was_created = model.objects.update_or_create(
                    **{pk_field: pk_value}, defaults=defaults
                )
                created += int(was_created)
                updated += int(not was_created)

            total_created += created
            total_updated += updated
            self.stdout.write(
                f"{model_name:32s} {created:3d} created  {updated:3d} updated"
            )

        verb = "would sync" if dry_run else "synced"
        self.stdout.write(
            self.style.SUCCESS(
                f"{verb}: {total_created} created, {total_updated} updated."
            )
        )

        if not dry_run:
            # The lookup tables are cached in memory. ``update_or_create`` does
            # fire post_save, but the cache is bumped explicitly so a bulk
            # rewrite, a raw SQL fix, or a future change to this loop cannot
            # leave the site serving pre-seed labels.
            from ctmj.services.lookups import invalidate_all

            invalidate_all()
            self.stdout.write("Reference-data cache invalidated.")
