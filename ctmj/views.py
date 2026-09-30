"""HTTP layer.

Deliberately thin: parse and validate the request, call
:mod:`ctmj.services.predictor`, render. No model loading, no dataframe
construction and no database fallbacks live here.
"""

from __future__ import annotations

import logging
from typing import Any

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from ctmj.apps import CtmjConfig
from ctmj.services import formdata, predictor, reference_data

logger = logging.getLogger("ctmj.views")


def _registry():
    return CtmjConfig.registry


def _options(field: str) -> list[dict[str, Any]]:
    """Normalise a lookup table into uniform ``{value, label}`` pairs.

    The templates then never need to know the real model or which attribute
    holds the code, so a column rename touches one place instead of every
    template. Read through the cache: this used to be eight queries per
    request, re-fetching the same 91 static rows each time.
    """
    from ctmj.services import lookups

    return lookups.options(field)


def _lookup_context() -> dict[str, list[dict[str, Any]]]:
    """Every dropdown population, ordered for stable rendering.

    Ordering is explicit so the UI never reshuffles between requests, which
    would silently change the meaning of an already-submitted value.
    """
    return {
        "genders": _options("GenderID"),
        "jobs": _options("BAS_werkzaamheid_resp"),
        "regions": _options("SPSS_Regio5"),
        "incomes": _options("BAS_bruto_jaarinkomen"),
        "social_classes": _options("AFG_sk2015"),
        "educations": _options("BAS_voltooide_opleiding8_resp"),
        "lifestages": _options("SPSS_Lifestage"),
        "touchpoints": _options("type_touch"),
    }


def _error_items(errors: dict[str, str]) -> list[dict[str, str]]:
    """Turn the validator's field-keyed errors into linkable rows.

    ``error_summary`` is a flat list of strings, which is the right shape for
    an aria-live region but useless for navigation: a user told "2 fields need
    fixing" still has to hunt for them. Each row here carries the field label,
    the message and the id of the inline error, so the summary can link
    straight to the control that is wrong.

    Only produced for validation failures. The 503 and 500 paths describe a
    problem with the server rather than with a field, and have nothing to link
    to.
    """
    return [
        {
            "name": name,
            "label": formdata.FIELD_LABELS.get(name, name),
            "message": message,
            "anchor": f"e-{name}",
        }
        for name, message in errors.items()
    ]


@require_GET
def home_view(request: HttpRequest) -> HttpResponse:
    """Overview console: the system, its segments and its model state.

    The segments come from :mod:`reference_data` rather than from a count. The
    previous revision rendered the number ``3`` and nothing else, which told an
    analyst nothing they could act on -- the actual segment names and their
    descriptions are the part of this page worth reading.
    """
    from ctmj.services import lookups

    registry = _registry()

    # The noise entry is a real DBSCAN outcome rather than one of the
    # behavioural segments, so it is kept out of the segment grid and given
    # its own row instead of being silently folded into the count.
    segments = [
        {
            "cluster_id": row["cluster_id"],
            "label": reference_data.CLUSTER_LABELS.get(row["cluster_id"], ""),
            "name": row["name"],
            "description": row["description"],
        }
        for row in reference_data.CLUSTERS
        if row["cluster_id"] != predictor.NOISE_LABEL
    ]
    noise = next(
        (
            {
                "cluster_id": row["cluster_id"],
                "label": reference_data.CLUSTER_LABELS.get(row["cluster_id"], "Nhiễu"),
                "name": row["name"],
                "description": row["description"],
            }
            for row in reference_data.CLUSTERS
            if row["cluster_id"] == predictor.NOISE_LABEL
        ),
        None,
    )

    return render(
        request,
        "home.html",
        {
            "touchpoint_count": lookups.touchpoint_count(),
            "cluster_count": len(segments),
            "segments": segments,
            "noise_segment": noise,
            "registry_status": registry.status if registry else None,
        },
    )


def predict_view(request: HttpRequest) -> HttpResponse:
    """Render the prediction form, and run inference on POST.

    On a failed submission the form is redrawn with field-level errors and the
    submitted values intact, so one bad field never forces the user to retype
    the other eleven.
    """
    registry = _registry()
    context = _lookup_context()
    context["registry_status"] = registry.status if registry else None

    if request.method == "GET":
        return render(request, "predict.html", context)

    validated = formdata.validate_prediction_form(request.POST)
    context["submitted"] = validated.submitted
    context["errors"] = validated.errors

    if not validated.is_valid:
        context["error_summary"] = validated.error_messages
        context["error_items"] = _error_items(validated.errors)
        logger.info("Rejected prediction form: %s", validated.error_messages)
        return render(request, "predict.html", context, status=400)

    status = registry.status
    if registry is None or not registry.ready():
        message = status.message if status else "Bộ mô hình chưa sẵn sàng."
        context["error_summary"] = [message]
        logger.warning("Prediction requested while model state is %s", status.state if status else "none")
        return render(request, "predict.html", context, status=503)

    try:
        context["prediction"] = predictor.predict(registry, validated)
    except predictor.PredictionError as exc:
        logger.warning("Prediction failed: %s", exc)
        context["error_summary"] = [str(exc)]
        return render(request, "predict.html", context, status=500)
    except Exception:
        logger.exception("Unexpected error during prediction")
        context["error_summary"] = ["Hệ thống gặp lỗi khi dự báo. Vui lòng thử lại sau."]
        return render(request, "predict.html", context, status=500)

    return render(request, "predict.html", context)


def page_not_found(request: HttpRequest, exception: Exception | None = None) -> HttpResponse:
    """Branded 404. Wired through ``handler404`` so it applies in production."""
    return render(request, "404.html", {"request_path": request.path}, status=404)


@require_GET
def health_view(request: HttpRequest) -> HttpResponse:
    """Liveness probe that also reports model readiness.

    Returns 503 while artefacts are missing or still loading, so an
    orchestrator can hold traffic back instead of serving a broken predictor.
    """
    registry = _registry()
    status = registry.status if registry else None
    ready = bool(registry and registry.ready())
    return JsonResponse(
        {
            "status": "ok" if ready else "degraded",
            "models": status.as_dict() if status else None,
        },
        status=200 if ready else 503,
    )
