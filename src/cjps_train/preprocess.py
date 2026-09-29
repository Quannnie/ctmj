"""Preprocessors for the two stages.

Kept in one module because the two ColumnTransformers are a matched pair: they
read overlapping columns from the same rows, and a change to one that is not
mirrored in the other shows up only as a wrong prediction, never as an error.

**Scaling groups are chosen by measurement type, not by convenience.** Age and
household size are roughly continuous and skewed, so a RobustScaler keeps the
median behaviour without letting a handful of large households dominate. Codes
with a meaningful zero (children, income band, region) are bounded ordinals, so
MinMaxScaler keeps them interpretable on [0, 1]. The original assigned these by
whichever group was nearest in the list.

**The imputer travels with the artefacts.** The app's inference path starts from
twelve form fields, which are always complete, so it never needs imputation.
Training data does. If the fitted imputer is not saved, a future version of the
app that starts accepting partial input would have to guess, and the two
pipelines would quietly disagree. Saving it costs one file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.cjps_train.config import CLUSTERING_FEATURES, PREDICTION_FEATURES

logger = logging.getLogger("cjps_train.preprocess")

#: Stage 1 groups, by measurement type.
CLUSTER_ROBUST: tuple[str, ...] = (
    "BAS_werkzaamheid_resp",
    "afg_kinderen_huishouden",
    "BAS_bruto_jaarinkomen",
)
CLUSTER_MINMAX: tuple[str, ...] = (
    "GenderID",
    "SPSS_Regio5",
    "BAS_voltooide_opleiding8_resp",
    "SPSS_Lifestage",
    "AFG_sk2015",
)
CLUSTER_STANDARD: tuple[str, ...] = (
    "Age",
    "BAS_huishoudgrootte",
)

#: Stage 2 groups.
PREDICT_ROBUST: tuple[str, ...] = ("BAS_werkzaamheid_resp", "afg_kinderen_huishouden")
PREDICT_MINMAX: tuple[str, ...] = (
    "GenderID",
    "SPSS_Regio5",
    "final_label",
    "step2",
    "step3",
)
PREDICT_STANDARD: tuple[str, ...] = ("Age",)


@dataclass
class PreprocessorBundle:
    """The fitted transformers plus the groups they were built from."""

    clustering: ColumnTransformer
    predicting: ColumnTransformer
    user_imputer: object | None = None
    clustering_groups: dict[str, tuple[str, ...]] = None  # type: ignore[assignment]
    predicting_groups: dict[str, tuple[str, ...]] = None  # type: ignore[assignment]

    def as_dict(self) -> dict:
        return {
            "clustering_groups": {k: list(v) for k, v in (self.clustering_groups or {}).items()},
            "predicting_groups": {k: list(v) for k, v in (self.predicting_groups or {}).items()},
            "clustering_columns": list(CLUSTERING_FEATURES),
            "predicting_columns": list(PREDICTION_FEATURES),
            "imputer_saved": self.user_imputer is not None,
        }


def _assert_partitioned(columns, groups: dict[str, tuple[str, ...]], label: str) -> None:
    """Every column must appear in exactly one group.

    ``remainder='drop'`` means a column that falls through the cracks is not
    reported — it is silently absent from the output, so the model trains on
    fewer features than the metrics were computed over and nothing fails. This
    check turns that silence into an error at build time.
    """
    seen: list[str] = []
    for group in groups.values():
        seen.extend(group)
    duplicates = {c for c in seen if seen.count(c) > 1}
    if duplicates:
        raise ValueError(f"{label}: column(s) in more than one scaling group: {sorted(duplicates)}")
    missing = sorted(set(columns) - set(seen))
    extra = sorted(set(seen) - set(columns))
    if missing:
        raise ValueError(f"{label}: column(s) in no scaling group: {missing}")
    if extra:
        raise ValueError(f"{label}: scaling group(s) reference unknown column(s): {extra}")


def build_clustering_preprocessor() -> ColumnTransformer:
    """Stage-1 transformer over the ten clustering columns.

    A ``SimpleImputer`` sits in front of the scalers. The user table is imputed
    once with KNN before it reaches here, so this only ever fires as a safety
    net for a column that became NaN during assembly — and it makes that
    condition non-fatal instead of propagating a confusing downstream error.
    """
    groups = {
        "rob": CLUSTER_ROBUST,
        "mms": CLUSTER_MINMAX,
        "std": CLUSTER_STANDARD,
    }
    _assert_partitioned(CLUSTERING_FEATURES, groups, "clustering preprocessor")

    return ColumnTransformer(
        transformers=[
            (
                "impute",
                SimpleImputer(strategy="median"),
                list(CLUSTERING_FEATURES),
            ),
            ("rob", RobustScaler(), list(CLUSTER_ROBUST)),
            ("mms", MinMaxScaler(), list(CLUSTER_MINMAX)),
            ("std", StandardScaler(), list(CLUSTER_STANDARD)),
        ],
        remainder="drop",
        # ColumnTransformer drops sparse columns by default; keeping them
        # explicit documents that the choice is deliberate.
        sparse_threshold=0.0,
    )


def build_predicting_preprocessor() -> ColumnTransformer:
    """Stage-2 transformer over the eight prediction columns.

    ``final_label`` is scaled alongside the other codes rather than being passed
    through. Segment ids are ordinal but arbitrary — label 2 is not twice label
    1 — so MinMaxScaler at least keeps the classifier's splits on a comparable
    numeric scale instead of treating a raw cluster id as a magnitude.
    """
    groups = {
        "rob": PREDICT_ROBUST,
        "mms": PREDICT_MINMAX,
        "std": PREDICT_STANDARD,
    }
    _assert_partitioned(PREDICTION_FEATURES, groups, "prediction preprocessor")

    return ColumnTransformer(
        transformers=[
            (
                "impute",
                SimpleImputer(strategy="median"),
                list(PREDICTION_FEATURES),
            ),
            ("rob", RobustScaler(), list(PREDICT_ROBUST)),
            ("mms", MinMaxScaler(), list(PREDICT_MINMAX)),
            ("std", StandardScaler(), list(PREDICT_STANDARD)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def fit_clustering_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    """Fit the stage-1 transformer. Safe to call before segmentation exists."""
    missing = [c for c in CLUSTERING_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Modelling frame is missing clustering column(s): {missing}")
    transformer = build_clustering_preprocessor()
    transformer.fit(frame[list(CLUSTERING_FEATURES)])
    X = transformer.transform(frame[list(CLUSTERING_FEATURES)])
    logger.info("Clustering matrix: %s", np.asarray(X).shape)
    return transformer


def fit_predicting_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    """Fit the stage-2 transformer.

    Separate from the clustering fit because ``PREDICTION_FEATURES`` contains
    ``final_label``, which is the cluster id and therefore does not exist until
    the segmentation stage has run. Fitting both in one call would demand the
    answer to a question the caller has not asked yet.
    """
    missing = [c for c in PREDICTION_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Modelling frame is missing prediction column(s): {missing}")
    transformer = build_predicting_preprocessor()
    transformer.fit(frame[list(PREDICTION_FEATURES)])
    return transformer


def fit_preprocessors(
    frame: pd.DataFrame,
    user_imputer: object | None = None,
) -> PreprocessorBundle:
    """Fit both transformers on a frame that already carries ``final_label``.

    Convenience path for tests and for a frame assembled in one go. The
    training run calls the two functions separately, because the real order is
    clustering → segmentation → ``final_label`` → prediction.

    Both are fitted on the *full* frame rather than a training split. That is
    deliberate and is the one place a scaler is allowed to see everything: a
    scaler has no fitted parameters that can encode a label, so fitting it on
    all rows leaks no target information. The same argument does **not** extend
    to the classifier, which is why that one is selected and scored under folds.
    """
    return PreprocessorBundle(
        clustering=fit_clustering_preprocessor(frame),
        predicting=fit_predicting_preprocessor(frame),
        user_imputer=user_imputer,
        clustering_groups={
            "rob": CLUSTER_ROBUST,
            "mms": CLUSTER_MINMAX,
            "std": CLUSTER_STANDARD,
        },
        predicting_groups={
            "rob": PREDICT_ROBUST,
            "mms": PREDICT_MINMAX,
            "std": PREDICT_STANDARD,
        },
    )


def as_matrix(bundle: PreprocessorBundle, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Scaled clustering matrix and scaled prediction matrix for a frame."""
    X_cluster = bundle.clustering.transform(frame[list(CLUSTERING_FEATURES)])
    X_predict = bundle.predicting.transform(frame[list(PREDICTION_FEATURES)])
    return np.asarray(X_cluster, dtype=float), np.asarray(X_predict, dtype=float)
