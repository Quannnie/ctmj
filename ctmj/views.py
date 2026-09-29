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
from ctmj.models import (
    AFG_sk2015,
    BAS_bruto_jaarinkomen,
    BAS_werkzaamheid_resp,
    BAS_voltooide_opleiding8_resp,
    GenderID,
    SPSS_Lifestage,
    SPSS_Regio5,
    type_touch,
)
from ctmj.services import formdata, predictor, reference_data

logger = logging.getLogger("ctmj.views")


def _registry():
    return CtmjConfig.registry


def _options(model: Any, pk: str, label: str) -> list[dict[str, Any]]:
    """Normalise a lookup queryset into uniform ``{value, label}`` pairs.

    The templates then never need to know the real model or which attribute
    holds the code, so a column rename touches the view instead of every
    template.
    """
    return [
        {"value": getattr(obj, pk), "label": getattr(obj, label)}
        for obj in model.objects.order_by(pk)
    ]


def _lookup_context() -> dict[str, list[dict[str, Any]]]:
    """Every dropdown population, ordered for stable rendering.

    Ordering is explicit so the UI never reshuffles between requests, which
    would silently change the meaning of an already-submitted value.
    """
    return {
        "genders": _options(GenderID, "gender_code", "gender_name"),
        "jobs": _options(BAS_werkzaamheid_resp, "code", "name"),
        "regions": _options(SPSS_Regio5, "code", "name"),
        "incomes": _options(BAS_bruto_jaarinkomen, "code", "name"),
        "social_classes": _options(AFG_sk2015, "code", "name"),
        "educations": _options(BAS_voltooide_opleiding8_resp, "code", "name"),
        "lifestages": _options(SPSS_Lifestage, "code", "name"),
        "touchpoints": _options(type_touch, "code", "name"),
    }


@require_GET
def home_view(request: HttpRequest) -> HttpResponse:
    """Landing page with an honest summary of the model pipeline."""
    registry = _registry()
    return render(
        request,
        "home.html",
        {
            "touchpoint_count": type_touch.objects.count(),
            "cluster_count": max(len(reference_data.CLUSTERS) - 1, 0),
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
