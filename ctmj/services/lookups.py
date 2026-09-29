"""Cached access to the reference tables.

Why
---
These tables hold 91 rows in total and are populated once, by a data migration
or ``manage.py seed_reference_data``. Nothing writes them while the site runs —
except an administrator editing a label in ``/admin/``, which is exactly the
case a cache has to get right.

They were nevertheless re-read on every request. Measured before this module
existed:

    validate_prediction_form   11 queries   2.481 ms
    _lookup_context             8 queries   1.981 ms
                            --------------------------
    per prediction POST        19 queries   4.462 ms

Nineteen round trips to fetch the same 91 static rows, on the critical path of
the one request that has to feel instant. The counts are what makes it
unambiguous: the tables are small enough to cache whole and identical on every
call, so nothing is being re-read for a reason.

Invalidation
------------
Rather than a TTL — which would be a lie about how long the data is fresh, and
would also mean serving stale labels for up to that long after a correction —
the cache is keyed by a version integer stored in the cache itself, and the
version is bumped by:

* a ``post_save`` / ``post_delete`` / ``post_save`` on the cluster model, so an
  edit made in ``/admin/`` takes effect on the next request;
* ``seed_reference_data`` and any bulk load.

Bumping is O(1) regardless of how much is cached, which deleting keys per table
would not be. The trade-off is that stale keys linger until they expire; the
timeout is therefore short.

The cache backend is whatever ``CACHES`` says. In development that is
process-local, so the cache is per-worker; in production it can be Redis and
shared. Either way correctness does not depend on it being shared, only
freshness latency does.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from django.core.cache import cache

from ctmj.models import (
    AFG_sk2015,
    BAS_bruto_jaarinkomen,
    BAS_voltooide_opleiding8_resp,
    BAS_werkzaamheid_resp,
    GenderID,
    SPSS_Lifestage,
    SPSS_Regio5,
    cluster_info,
    type_touch,
)

logger = logging.getLogger("ctmj.lookups")

#: Namespace for every key this module writes, so invalidation is bounded.
KEY_PREFIX = "ctmj.refdata"

#: Seconds a cached payload may survive without an explicit invalidation. The
#: signals below are the real mechanism; this is a backstop so a write that
#: bypasses the ORM cannot leave a label stale indefinitely.
DEFAULT_TIMEOUT = 3600

_VERSION_KEY = f"{KEY_PREFIX}.version"


def _version() -> int:
    """The current cache generation.

    Defensive on purpose. This runs on the read path, and an unavailable cache
    backend must not turn into a failed request -- the caller's fallback to
    reading the database is the whole point of the read-through wrapper, and it
    cannot work if reading the version raises before the wrapper is reached.
    """
    try:
        version = cache.get(_VERSION_KEY)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cache unavailable for the version key (%s)", exc)
        return 0

    if version is None:
        version = 1
        try:
            cache.set(_VERSION_KEY, version, None)
        except Exception:  # noqa: BLE001
            # Uncacheable is acceptable; every read will fall through anyway.
            pass
    return int(version)


def bump_version() -> int:
    """Invalidate every cached payload by rotating the version key.

    O(1): the next read misses and repopulates under the new version. Deleting
    each key instead would be O(number of keys) and would also have to know them
    all, including the ones other workers created.

    Survives both a missing key and an unavailable backend. A cache that cannot
    be invalidated is not a reason to stop serving — the read path falls through
    to the database, which is slower and always correct.
    """
    try:
        return int(cache.incr(_VERSION_KEY))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not increment the cache version (%s); recreating it", exc)
    try:
        cache.set(_VERSION_KEY, 1, None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cache unavailable; reference data will be read from the database")
    return 1


def _key(name: str) -> str:
    return f"{KEY_PREFIX}.v{_version()}.{name}"


def _cache_get(key: str) -> Any:
    """``cache.get`` that returns a miss instead of raising."""
    try:
        return cache.get(key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cache read failed for %s (%s); treating as a miss", key, exc)
        return None


def _cache_set(key: str, value: Any) -> None:
    """``cache.set`` that swallows a backend failure.

    A cache that cannot be written costs queries on the next request. A request
    that raises because of it costs the user their prediction.
    """
    try:
        cache.set(key, value, DEFAULT_TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cache write failed for %s (%s)", key, exc)


def _cached(name: str, producer) -> Any:
    """Read through the cache.

    A cache failure must not become an outage, so an exception from the backend
    falls back to calling the producer directly. Losing the cache costs queries;
    raising costs the request.
    """
    key = _key(name)
    hit = _cache_get(key)
    if hit is not None:
        return hit

    value = producer()
    _cache_set(key, value)
    return value


def invalidate_all() -> None:
    """Bump the version. Safe to call from anywhere, including a data migration."""
    new_version = bump_version()
    logger.info("Reference-data cache invalidated (version %d)", new_version)


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------

#: Form field name -> (model, primary key attribute). Mirrors
#: ``formdata.CHOICE_FIELDS`` / ``CHOICE_PK``; kept here so this module does not
#: import formdata, which would be circular (formdata needs this module).
_TABLES: dict[str, tuple[Any, str]] = {
    "GenderID": (GenderID, "gender_code"),
    "SPSS_Regio5": (SPSS_Regio5, "code"),
    "BAS_werkzaamheid_resp": (BAS_werkzaamheid_resp, "code"),
    "BAS_bruto_jaarinkomen": (BAS_bruto_jaarinkomen, "code"),
    "AFG_sk2015": (AFG_sk2015, "code"),
    "BAS_voltooide_opleiding8_resp": (BAS_voltooide_opleiding8_resp, "code"),
    "SPSS_Lifestage": (SPSS_Lifestage, "code"),
    "type_touch": (type_touch, "code"),
    # The two journey-step fields validate against the same table as the
    # touchpoint list. They are listed explicitly rather than resolved
    # dynamically, so a typo in formdata.CHOICE_FIELDS fails loudly here
    # instead of silently skipping validation for that field.
    "step_1_channel": (type_touch, "code"),
    "step_2_channel": (type_touch, "code"),
}


def _table(field: str) -> tuple[Any, str]:
    try:
        return _TABLES[field]
    except KeyError:
        # A silent fallback would skip validation for the field entirely, so a
        # caller with a typo gets an error naming every field that exists.
        raise KeyError(
            f"Unknown lookup field {field!r}. Known fields: "
            f"{', '.join(sorted(_TABLES))}"
        ) from None


def valid_codes(field: str) -> set[int]:
    """Every accepted code for a choice field.

    Returned as a ``set`` because the caller does ``code in valid_codes`` once
    per field per request, and a set membership test does not hash the value on
    every comparison the way a list scan of 20 items does.
    """
    model, pk = _table(field)
    return _cached(
        f"codes.{field}",
        lambda: {int(v) for v in model.objects.values_list(pk, flat=True)},
    )


def name_to_code(field: str) -> dict[str, int]:
    """Case-folded label -> code, for the fields that also accept a name.

    Touchpoint names are long, so the UI offers them through a datalist and a
    user may submit either ``20`` or ``Email``.
    """
    model, pk = _table(field)
    return _cached(
        f"names.{field}",
        lambda: {
            str(obj.name).strip().casefold(): int(getattr(obj, pk))
            for obj in model.objects.all()
        },
    )


def options(field: str) -> list[dict[str, Any]]:
    """``{value, label}`` pairs for a dropdown, ordered for stable rendering.

    The ordering is explicit rather than inherited from the cache, because a
    UI that reshuffles between requests silently changes the meaning of a value
    an operator has already submitted.
    """
    model, pk = _table(field)
    label_attr = "gender_name" if field == "GenderID" else "name"
    return _cached(
        f"options.{field}",
        lambda: [
            {"value": int(getattr(obj, pk)), "label": getattr(obj, label_attr)}
            for obj in model.objects.order_by(pk)
        ],
    )


def touchpoint_details(codes: Iterable[int]) -> dict[int, dict[str, str]]:
    """Name and description for the given touchpoint codes.

    Prediction needs one or two rows out of twenty, so the whole table is cached
    and sliced here rather than issuing a filtered query per prediction. The
    cache makes the filtered query pointless: it costs the same round trip to
    fetch two rows as to fetch none and read them from memory.
    """
    wanted = {int(c) for c in codes}
    if not wanted:
        return {}
    table = _cached(
        "touchpoints",
        lambda: {
            int(obj.code): {
                "name": obj.name or "",
                "description": obj.description or "",
            }
            for obj in type_touch.objects.all()
        },
    )
    return {code: table[code] for code in wanted if code in table}


def cluster_details(cluster_id: int) -> dict[str, str]:
    """Name and description for one segment, falling back to a usable label.

    A segment id with no ``cluster_info`` row is a real possibility — the
    segmenter chooses k from the data, so a retrain can produce an id the
    lookup table has never heard of. The fallback keeps the UI readable instead
    of showing a blank, and the manifest notes the mismatch for a data fix.
    """
    table = _cached(
        "clusters",
        lambda: {
            int(obj.cluster_id): {
                "name": obj.name or "",
                "description": obj.description or "",
            }
            for obj in cluster_info.objects.all()
        },
    )
    return table.get(int(cluster_id), {})


def touchpoint_count() -> int:
    """How many touchpoints exist, for the landing page summary."""
    return len(_cached("touchpoints", lambda: _all_touchpoints()))


def _all_touchpoints() -> dict[int, dict[str, str]]:
    return {
        int(obj.code): {
            "name": obj.name or "",
            "description": obj.description or "",
        }
        for obj in type_touch.objects.all()
    }
