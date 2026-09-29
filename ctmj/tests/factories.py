"""Shared fixtures: reference data and fitted model doubles.

The model doubles are *real* scikit-learn estimators fitted on tiny synthetic
data, not mocks. That matters: the bugs this suite guards against are precisely
the ones a hand-rolled fake would hide — the difference between
``SpectralClustering.labels_`` (one entry per training sample) and
``components_`` (one row per cluster) is a property of the real estimator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from django.apps import apps
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from ctmj.services import reference_data
from ctmj.services.formdata import CLUSTERING_FEATURES, PREDICTION_FEATURES

#: Seed shared by every fixture so runs are reproducible.
SEED = 4242


def seed_reference_data() -> None:
    """Populate every lookup table. Idempotent."""
    live = {
        "GenderID": apps.get_model("ctmj", "GenderID"),
        "BAS_werkzaamheid_resp": apps.get_model("ctmj", "BAS_werkzaamheid_resp"),
        "SPSS_Regio5": apps.get_model("ctmj", "SPSS_Regio5"),
        "BAS_bruto_jaarinkomen": apps.get_model("ctmj", "BAS_bruto_jaarinkomen"),
        "AFG_sk2015": apps.get_model("ctmj", "AFG_sk2015"),
        "BAS_voltooide_opleiding8_resp": apps.get_model(
            "ctmj", "BAS_voltooide_opleiding8_resp"
        ),
        "SPSS_Lifestage": apps.get_model("ctmj", "SPSS_Lifestage"),
        "type_touch": apps.get_model("ctmj", "type_touch"),
        "cluster_info": apps.get_model("ctmj", "cluster_info"),
    }
    for model_name, pk_field, rows in reference_data.REFERENCE_TABLES:
        model = live[model_name]
        for row in rows:
            model.objects.update_or_create(
                **{pk_field: row[pk_field]},
                defaults={k: v for k, v in row.items() if k != pk_field},
            )


# ---------------------------------------------------------------------------
# Model doubles
# ---------------------------------------------------------------------------

#: Three tight clusters whose centres are *realistic* household profiles, so a
#: normal submission lands inside a cluster rather than in the noise branch.
#: Column order matches CLUSTERING_FEATURES:
#:   BAS_huishoudgrootte, BAS_werkzaamheid_resp, afg_kinderen_huishouden,
#:   BAS_voltooide_opleiding8_resp, SPSS_Lifestage, GenderID, SPSS_Regio5,
#:   BAS_bruto_jaarinkomen, AFG_sk2015, Age
_CLUSTER_CENTRES = np.array(
    [
        # cluster 0 — empty nesters / mature singles
        [2.0, 7.0, 0.0, 3.0, 4.0, 1.0, 2.0, 3.0, 3.0, 66.0],
        # cluster 1 — mature & established families
        [4.0, 2.0, 2.0, 5.0, 7.0, 2.0, 1.0, 4.0, 2.0, 42.0],
        # cluster 2 — young singles / couples
        [1.0, 2.0, 0.0, 6.0, 1.0, 2.0, 3.0, 2.0, 2.0, 27.0],
    ]
)

#: Per-feature jitter. Age and income vary far more than a gender code does, so
#: a single scalar would blur the clusters.
_JITTER = np.array([0.4, 0.6, 0.4, 0.6, 0.5, 0.3, 0.6, 0.7, 0.5, 4.0])

#: Numeric value range used to clamp the scattered outliers.
_VALUE_RANGE = np.array([30.0, 11.0, 30.0, 10.0, 10.0, 2.0, 5.0, 8.0, 5.0, 120.0])

#: Number of points per tight cluster.
PER_CLUSTER = 24

#: Number of scattered "outlier" points. Kept small relative to the clusters so
#: each one is genuinely isolated and DBSCAN labels it noise.
N_OUTLIERS = 12

#: Tuned so the three blobs separate cleanly and every scattered point falls
#: outside every core neighbourhood.
EPS = 0.6
MIN_SAMPLES = 4


def make_clustering_frame() -> tuple[pd.DataFrame, np.ndarray]:
    """Return a raw clustering DataFrame and the true group of each row.

    Group 0-2 are the tight clusters; group 3 marks the scattered outliers that
    DBSCAN should label -1.
    """
    rng = np.random.default_rng(SEED)
    frames = []
    groups = []

    for group_idx, centre in enumerate(_CLUSTER_CENTRES):
        block = centre + rng.normal(scale=1.0, size=(PER_CLUSTER, centre.size)) * _JITTER
        block = np.clip(block, 0, _VALUE_RANGE)
        frames.append(pd.DataFrame(block, columns=list(CLUSTERING_FEATURES)))
        groups.extend([group_idx] * PER_CLUSTER)

    # Scattered across the whole value box: far from each other and from every
    # cluster, so DBSCAN labels them noise.
    outliers = rng.uniform(0, 1, size=(N_OUTLIERS, len(CLUSTERING_FEATURES))) * _VALUE_RANGE
    frames.append(pd.DataFrame(outliers, columns=list(CLUSTERING_FEATURES)))
    groups.extend([3] * N_OUTLIERS)

    return pd.concat(frames, ignore_index=True), np.array(groups)


def build_model_doubles() -> dict[str, object]:
    """Fit a complete, self-consistent set of the five artefacts.

    Returns a dict keyed exactly as ``settings.CTMJ['MODEL_FILES']`` values, so
    a test can install it into a :class:`ModelRegistry` directly.
    """
    frame, _groups = make_clustering_frame()

    # --- clustering preprocessor ----------------------------------------
    user_pre = ColumnTransformer(
        transformers=[
            ("std", StandardScaler(), ["Age", "BAS_huishoudgrootte"]),
            ("rob", RobustScaler(), ["afg_kinderen_huishouden", "BAS_bruto_jaarinkomen"]),
            (
                "mms",
                MinMaxScaler(),
                [
                    "GenderID",
                    "SPSS_Regio5",
                    "BAS_werkzaamheid_resp",
                    "BAS_voltooide_opleiding8_resp",
                    "SPSS_Lifestage",
                    "AFG_sk2015",
                ],
            ),
        ],
        remainder="drop",
    )
    scaled = user_pre.fit_transform(frame[list(CLUSTERING_FEATURES)])

    # --- DBSCAN: three clusters plus scattered noise ---------------------
    dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(scaled)
    assert set(dbscan.labels_) == {0, 1, 2, -1}, (
        f"fixture must yield 3 clusters + noise, got {sorted(set(dbscan.labels_))}"
    )

    # --- Spectral over the non-noise rows --------------------------------
    keep = dbscan.labels_ != -1
    spectral = SpectralClustering(n_clusters=3, affinity="rbf", gamma=1.0, random_state=42)
    spectral.fit_predict(scaled[keep])
    # scikit-learn >= 1.6 dropped SpectralClustering.components_; the app must
    # work either way, so assert only what every version guarantees.
    assert len(spectral.labels_) == int(keep.sum())

    # --- prediction stage -------------------------------------------------
    rng = np.random.default_rng(SEED)
    n = int(keep.sum())
    predict_frame = pd.DataFrame(
        {
            "BAS_werkzaamheid_resp": rng.integers(1, 11, n),
            "afg_kinderen_huishouden": rng.integers(0, 4, n),
            "GenderID": rng.integers(1, 3, n),
            "SPSS_Regio5": rng.integers(1, 6, n),
            "final_label": spectral.labels_,
            "step2": rng.integers(1, 23, n),
            "step3": rng.integers(1, 23, n),
            "Age": rng.integers(18, 80, n),
        }
    )
    predict_pre = ColumnTransformer(
        transformers=[
            ("rob", RobustScaler(), ["BAS_werkzaamheid_resp", "afg_kinderen_huishouden"]),
            (
                "mms",
                MinMaxScaler(),
                ["GenderID", "SPSS_Regio5", "final_label", "step2", "step3"],
            ),
            ("std", StandardScaler(), ["Age"]),
        ],
        remainder="drop",
    )
    X_predict = predict_pre.fit_transform(predict_frame[list(PREDICTION_FEATURES)])

    # A classifier covering every real touchpoint code so the ranking stage
    # has the full label space.
    codes = sorted(
        apps.get_model("ctmj", "type_touch").objects.values_list("code", flat=True)
    ) or [1, 2, 3, 4, 5]
    y = np.array([codes[i % len(codes)] for i in range(n)])
    classifier = GradientBoostingClassifier(n_estimators=20, max_depth=3, random_state=42)
    classifier.fit(X_predict, y)

    return {
        "user_data_preprocessor": user_pre,
        "dbscan": dbscan,
        "spectral": spectral,
        "predicting_preprocessor": predict_pre,
        "gradient_boosting": classifier,
    }


def install_models(registry, artefacts: dict[str, object] | None = None) -> dict[str, object]:
    """Put a working artefact set into ``registry`` and mark it ready."""
    artefacts = artefacts if artefacts is not None else build_model_doubles()
    registry._models.clear()
    registry._models.update(artefacts)
    from ctmj.services.registry import LoadState

    registry._status.state = LoadState.READY
    registry._status.message = "ready"
    registry._status.loaded = sorted(artefacts)
    registry._status.missing = []
    registry._status.error = ""
    return artefacts


def valid_post(**overrides: str) -> dict[str, str]:
    """A POST payload that passes validation. Override any field."""
    payload = {
        "GenderID": "1",
        "Age": "41",
        "SPSS_Regio5": "1",
        "BAS_huishoudgrootte": "4",
        "BAS_werkzaamheid_resp": "2",
        "BAS_bruto_jaarinkomen": "4",
        "afg_kinderen_huishouden": "2",
        "AFG_sk2015": "2",
        "BAS_voltooide_opleiding8_resp": "5",
        "SPSS_Lifestage": "7",
        "step_1_channel": "20",
        "step_2_channel": "16",
    }
    payload.update(overrides)
    return payload
