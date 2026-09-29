"""Request parsing and validation for the prediction form.

The original view validated only that the twelve fields were non-empty, then
called ``int()`` on each. Two problems followed: a non-numeric value produced a
raw Python exception string shown to the user, and a crafted POST could push
arbitrary codes (e.g. ``GenderID=9999``) straight into the scikit-learn
pipeline, where they silently produce nonsense probabilities.

This module validates against the real lookup tables and returns
field-level errors so the template can render them inline next to the input,
plus a summary the UI can focus on submit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

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

#: Free-text numeric inputs: (min, max, inclusive-min, inclusive-max).
#: Upper bounds are deliberately generous — they exist to catch typos, not to
#: encode domain rules the model was never trained with.
NUMERIC_FIELDS: dict[str, tuple[int, int]] = {
    "Age": (0, 120),
    "BAS_huishoudgrootte": (1, 30),  # household size
    "afg_kinderen_huishouden": (0, 30),  # children in household
}

#: Dropdowns whose valid codes come from a lookup table.
CHOICE_FIELDS: dict[str, Any] = {
    "GenderID": GenderID,
    "SPSS_Regio5": SPSS_Regio5,
    "BAS_werkzaamheid_resp": BAS_werkzaamheid_resp,
    "BAS_bruto_jaarinkomen": BAS_bruto_jaarinkomen,
    "AFG_sk2015": AFG_sk2015,
    "BAS_voltooide_opleiding8_resp": BAS_voltooide_opleiding8_resp,
    "SPSS_Lifestage": SPSS_Lifestage,
    "step_1_channel": type_touch,
    "step_2_channel": type_touch,
}

#: Primary-key attribute on each lookup model.
CHOICE_PK: dict[str, str] = {
    "GenderID": "gender_code",
    "SPSS_Regio5": "code",
    "BAS_werkzaamheid_resp": "code",
    "BAS_bruto_jaarinkomen": "code",
    "AFG_sk2015": "code",
    "BAS_voltooide_opleiding8_resp": "code",
    "SPSS_Lifestage": "code",
    "step_1_channel": "code",
    "step_2_channel": "code",
}

#: Touchpoint inputs are free text (datalist) so long names stay searchable.
#: For those fields the validator also accepts the label, not just the code.
NAME_ACCEPTING_FIELDS: frozenset[str] = frozenset({"step_1_channel", "step_2_channel"})

#: User-facing field labels, reused by the template's error summary.
FIELD_LABELS: dict[str, str] = {
    "GenderID": "Giới tính",
    "Age": "Tuổi",
    "SPSS_Regio5": "Vùng",
    "BAS_huishoudgrootte": "Quy mô hộ gia đình",
    "BAS_werkzaamheid_resp": "Tình trạng việc làm",
    "BAS_bruto_jaarinkomen": "Tổng thu nhập hộ",
    "afg_kinderen_huishouden": "Số trẻ em",
    "AFG_sk2015": "Tầng lớp xã hội",
    "BAS_voltooide_opleiding8_resp": "Trình độ học vấn",
    "SPSS_Lifestage": "Giai đoạn cuộc sống",
    "step_1_channel": "Kênh trước kênh cuối",
    "step_2_channel": "Kênh tương tác cuối cùng",
}

ALL_FIELDS: tuple[str, ...] = tuple(FIELD_LABELS)

#: Maps the form's journey inputs onto the column names the trained
#: GradientBoosting pipeline expects. ``step1`` is the *target* — the next
#: touchpoint being predicted — so it is deliberately absent here.
JOURNEY_COLUMN_MAP: dict[str, str] = {
    "step_1_channel": "step2",
    "step_2_channel": "step3",
}


@dataclass
class ValidationResult:
    """Outcome of parsing a submitted form."""

    values: dict[str, int] = field(default_factory=dict)
    #: Raw strings as submitted, so the template can repopulate inputs.
    submitted: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def error_messages(self) -> list[str]:
        return [f"{FIELD_LABELS.get(name, name)}: {msg}" for name, msg in self.errors.items()]

    def get(self, name: str) -> int | None:
        return self.values.get(name)


def _as_int(raw: str) -> int | None:
    """Parse an integer, tolerating whitespace, thousands separators and '30+'."""
    text = raw.strip().replace(" ", "").replace(" ", "")
    if text.endswith("+"):
        text = text[:-1]
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _accepting_map(name: str, model: Any, pk: str) -> dict[str, int]:
    """Build a case-insensitive ``token -> code`` map for a lookup table.

    Touchpoint names are long, so the UI offers them through a datalist. This
    lets a user submit either ``20`` or ``Email`` and get the same result,
    instead of a validation error for a value they can plainly see listed.
    """
    if name not in NAME_ACCEPTING_FIELDS:
        return {}
    return {
        str(obj.name).strip().casefold(): getattr(obj, pk)
        for obj in model.objects.all()
    }


def validate_prediction_form(post: Mapping[str, str]) -> ValidationResult:
    """Validate a POST payload against the lookup tables and numeric bounds."""
    result = ValidationResult()

    valid_codes: dict[str, set[int]] = {}
    by_name: dict[str, dict[str, int]] = {}
    for name, model in CHOICE_FIELDS.items():
        pk = CHOICE_PK[name]
        valid_codes[name] = set(model.objects.values_list(pk, flat=True))
        by_name[name] = _accepting_map(name, model, pk)

    for name in ALL_FIELDS:
        raw = (post.get(name) or "").strip()
        result.submitted[name] = raw

        if not raw:
            result.errors[name] = "Vui lòng chọn hoặc nhập giá trị này."
            continue

        if name in NUMERIC_FIELDS:
            value = _as_int(raw)
            low, high = NUMERIC_FIELDS[name]
            if value is None:
                result.errors[name] = "Vui lòng nhập một số nguyên hợp lệ."
                continue
            if not (low <= value <= high):
                result.errors[name] = f"Giá trị phải nằm trong khoảng {low} đến {high}."
                continue
            result.values[name] = value
            continue

        value = _as_int(raw)
        if value is None:
            # Maybe the user submitted the touchpoint's name.
            resolved = by_name[name].get(raw.casefold())
            if resolved is not None:
                result.values[name] = resolved
            else:
                result.errors[name] = "Giá trị không có trong danh mục."
            continue

        if value not in valid_codes[name]:
            result.errors[name] = "Giá trị không có trong danh mục."
            continue
        result.values[name] = value

    return result


def to_feature_frame(result: ValidationResult):
    """Build the clustering-stage DataFrame from validated values.

    Raises ``KeyError`` if ``result`` is not valid; callers must check
    :attr:`ValidationResult.is_valid` first.
    """
    import pandas as pd

    v = result.values
    return pd.DataFrame(
        [
            {
                # --- clustering stage ---
                "BAS_huishoudgrootte": v["BAS_huishoudgrootte"],
                "BAS_werkzaamheid_resp": v["BAS_werkzaamheid_resp"],
                "afg_kinderen_huishouden": v["afg_kinderen_huishouden"],
                "BAS_voltooide_opleiding8_resp": v["BAS_voltooide_opleiding8_resp"],
                "SPSS_Lifestage": v["SPSS_Lifestage"],
                "GenderID": v["GenderID"],
                "SPSS_Regio5": v["SPSS_Regio5"],
                "BAS_bruto_jaarinkomen": v["BAS_bruto_jaarinkomen"],
                "AFG_sk2015": v["AFG_sk2015"],
                "Age": v["Age"],
                # --- prediction stage ---
                "step2": v["step_1_channel"],
                "step3": v["step_2_channel"],
            }
        ]
    )


#: Column order the fitted ColumnTransformer was trained on. ColumnTransformer
#: selects by name, but keeping the order explicit makes the contract readable
#: and guards against silent reordering.
CLUSTERING_FEATURES: tuple[str, ...] = (
    "BAS_huishoudgrootte",
    "BAS_werkzaamheid_resp",
    "afg_kinderen_huishouden",
    "BAS_voltooide_opleiding8_resp",
    "SPSS_Lifestage",
    "GenderID",
    "SPSS_Regio5",
    "BAS_bruto_jaarinkomen",
    "AFG_sk2015",
    "Age",
)

PREDICTION_FEATURES: tuple[str, ...] = (
    "BAS_werkzaamheid_resp",
    "afg_kinderen_huishouden",
    "GenderID",
    "SPSS_Regio5",
    "final_label",
    "step2",
    "step3",
    "Age",
)
