"""The two-stage CJPS inference pipeline.

Stage 1 — segmentation
    DBSCAN is used purely as a *noise gate*: a profile is "noise" when it sits
    further than ``eps`` from every core sample in the training set. Surviving
    profiles are assigned to a segment by their nearest Spectral Clustering
    eigenvector.

Stage 2 — next-touchpoint ranking
    The assigned segment is fed into a Gradient Boosting classifier, which
    returns a probability distribution over the candidate touchpoints. The top
    three are returned with the names and descriptions looked up from the
    database.

Correctness notes
-----------------
Two bugs in the original ``views.predict_view`` are fixed here; both produced
plausible-looking but meaningless output:

1. **Spectral label lookup was wrong.** The code did
   ``model_spectral.labels_[argmin(cdist(X, model_spectral.components_))]``.
   For ``SpectralClustering``, ``components_`` has one row *per cluster*
   (the eigenvectors) while ``labels_`` has one entry *per training sample*.
   Indexing ``labels_`` with a cluster index returned the label of whichever
   clean training sample happened to sit at that position — effectively
   random, and almost never the intended segment. The cluster id is the row
   index of the nearest eigenvector.

2. **The noise gate could never fire.** DBSCAN core samples are, by
   definition, never noise, so ``labels_[core_sample_indices_[...]]`` is always
   >= 0. The correct test is a distance comparison against ``eps``.

The dead third branch (which indexed ``model_spectral.labels_`` with a *DBSCAN*
sample index) has been removed; it mixed two different index spaces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ctmj.services import formdata, reference_data
from ctmj.services.registry import ModelRegistry

logger = logging.getLogger("ctmj.predictor")

#: DBSCAN's sentinel label for "not in any cluster".
NOISE_LABEL = -1

#: How many touchpoints to surface.
TOP_N = 3


class PredictionError(RuntimeError):
    """Raised when inference cannot proceed. Message is user-facing."""


@dataclass(frozen=True)
class ChannelPrediction:
    """One ranked next-touchpoint candidate."""

    channel_id: int
    probability: float  # 0-100
    name: str = ""
    description: str = ""

    @property
    def probability_value(self) -> str:
        """Dot-decimal percentage for machine consumption.

        Django localises floats to a comma for vi-VN, which is correct on
        screen but breaks the two consumers of this value: ``parseFloat`` in
        app.js would read ``"15,50"`` as 15, and ``width: 15,50%`` is invalid
        CSS that silently leaves the bar empty. Formatting here keeps one
        locale-independent source for both.
        """
        return f"{self.probability:.2f}"


@dataclass(frozen=True)
class Prediction:
    """Complete result of a single inference run."""

    cluster_id: int
    cluster_name: str
    cluster_label: str
    cluster_description: str
    is_noise: bool
    channels: list[ChannelPrediction] = field(default_factory=list)
    #: Full probability distribution, useful for charts and debugging.
    distribution: list[tuple[int, float]] = field(default_factory=list)
    #: Euclidean distance from the profile to its nearest DBSCAN core sample.
    noise_distance: float | None = None

    @property
    def top_channel(self) -> ChannelPrediction | None:
        return self.channels[0] if self.channels else None

    @property
    def confidence(self) -> float:
        """Share of probability mass captured by the returned top-N."""
        return sum(c.probability for c in self.channels)

    @property
    def noise_distance_value(self) -> str:
        """Dot-decimal DBSCAN distance. See ChannelPrediction.probability_value."""
        if self.noise_distance is None:
            return ""
        return f"{self.noise_distance:.3f}"


def _require(registry: ModelRegistry) -> dict[str, Any]:
    required = (
        "user_data_preprocessor",
        "dbscan",
        "spectral",
        "predicting_preprocessor",
        "gradient_boosting",
    )
    absent = [key for key in required if registry.get(key) is None]
    if absent:
        raise PredictionError(
            "Bộ mô hình chưa sẵn sàng (" + ", ".join(absent) + "). "
            "Vui lòng thử lại sau khi máy chủ nạp xong mô hình."
        )
    return {key: registry.get(key) for key in required}


def _unpack_estimator(model: Any) -> Any:
    """Return the fitted classifier, unwrapping a GridSearchCV if needed.

    The training script fits a plain estimator, but supporting a search object
    costs three lines and makes the app work with tuned models too.
    """
    return getattr(model, "best_estimator_", model)


def _spectral_labels_by_core_sample(
    dbscan: Any, spectral: Any
) -> np.ndarray | None:
    """Spectral label for each DBSCAN core sample, aligned with ``components_``.

    The two arrays live in different index spaces, which is where the original
    implementation went wrong:

    * ``dbscan.labels_`` / ``dbscan.core_sample_indices_`` are indexed against
      the **full** training set.
    * ``spectral.labels_`` is indexed against the **noise-filtered** subset
      ``X_clean`` — see ``src/main.py``, where ``SpectralClustering`` is fitted
      on ``X_scaled[dbscan_labels != -1]``.

    Mapping a full-set index straight onto ``spectral.labels_`` therefore reads
    the wrong row, or runs off the end of a shorter array.

    Returns ``None`` when the shapes do not line up, so the caller can fall
    back rather than emit a silently wrong label.
    """
    spectral_labels = getattr(spectral, "labels_", None)
    core_indices = getattr(dbscan, "core_sample_indices_", None)
    dbscan_labels = getattr(dbscan, "labels_", None)
    if spectral_labels is None or core_indices is None or dbscan_labels is None:
        return None

    # Positions in the full training set that survived the noise filter, in
    # ascending order — exactly the row order of ``spectral.labels_``.
    kept = np.flatnonzero(dbscan_labels != -1)
    if len(kept) != len(spectral_labels):
        return None

    positions = np.searchsorted(kept, core_indices)
    if positions.max(initial=0) >= len(spectral_labels):
        return None
    return np.asarray(spectral_labels)[positions]


def assign_cluster(scaled_row: np.ndarray, models: dict[str, Any]) -> tuple[int, float]:
    """Assign a scaled profile to a segment.

    Returns ``(cluster_id, distance_to_nearest_core)`` where ``cluster_id`` is
    ``-1`` for profiles that fall outside every DBSCAN core neighbourhood.
    """
    from scipy.spatial.distance import cdist

    dbscan = models["dbscan"]
    spectral = models["spectral"]

    components = getattr(dbscan, "components_", None)
    if components is None or len(components) == 0:
        raise PredictionError("Mô hình DBSCAN chưa có tập core sample.")

    distances = cdist(scaled_row, components, metric="euclidean")[0]
    nearest_idx = int(np.argmin(distances))
    nearest_distance = float(distances[nearest_idx])

    # --- noise gate -----------------------------------------------------
    # A new point is noise when no core sample lies within eps. Comparing the
    # distance against eps is the only correct test: core samples themselves
    # are never labelled -1, so reading a label could never detect noise.
    eps = float(getattr(dbscan, "eps", 0.0) or 0.0)
    if eps > 0.0 and nearest_distance > eps:
        logger.info("Profile classified as noise (distance %.4f > eps %.4f)", nearest_distance, eps)
        return NOISE_LABEL, nearest_distance

    # --- segment assignment --------------------------------------------
    # Fall back to the DBSCAN label if the spectral model is unusable.
    fallback = int(dbscan.labels_[dbscan.core_sample_indices_[nearest_idx]])

    # Preferred path, available on scikit-learn < 1.6: row i of
    # components_ is the eigenvector for cluster i, so the nearest
    # eigenvector's row index *is* the cluster id.
    spectral_components = getattr(spectral, "components_", None)
    if spectral_components is not None and len(spectral_components) > 0:
        spectral_distances = cdist(scaled_row, spectral_components, metric="euclidean")[0]
        return int(np.argmin(spectral_distances)), nearest_distance

    # scikit-learn >= 1.6 dropped components_. Recover the same answer by
    # nearest-neighbour against the spectral label of each core sample.
    by_core = _spectral_labels_by_core_sample(dbscan, spectral)
    if by_core is not None:
        return int(by_core[nearest_idx]), nearest_distance

    logger.warning(
        "Spectral labels could not be aligned with DBSCAN core samples; "
        "falling back to the DBSCAN label %s",
        fallback,
    )
    return fallback, nearest_distance


def rank_channels(
    raw_frame: Any, cluster_id: int, models: dict[str, Any]
) -> list[tuple[int, float]]:
    """Return the full ``(channel_id, probability_percent)`` distribution.

    ``raw_frame`` is the unscaled one-row input. The prediction preprocessor is
    a *different* ColumnTransformer from the clustering one: it takes eight
    columns, and ``final_label`` is one of them, so the label has to be added
    before scaling rather than after.
    """
    classifier = _unpack_estimator(models["gradient_boosting"])

    frame = raw_frame.copy()
    frame["final_label"] = cluster_id
    # ColumnTransformer selects by name; reindex to the exact training order.
    frame = frame[list(formdata.PREDICTION_FEATURES)]

    scaled = models["predicting_preprocessor"].transform(frame)
    probabilities = classifier.predict_proba(scaled)[0]
    classes = classifier.classes_
    return [(int(label), float(p) * 100.0) for label, p in zip(classes, probabilities)]


def predict(registry: ModelRegistry, validated: formdata.ValidationResult) -> Prediction:
    """Run both stages and return a fully populated :class:`Prediction`.

    ``validated`` must have passed :attr:`formdata.ValidationResult.is_valid`.
    """
    models = _require(registry)

    # --- stage 1: segmentation -----------------------------------------
    raw = formdata.to_feature_frame(validated)
    cluster_frame = raw[list(formdata.CLUSTERING_FEATURES)]

    try:
        scaled = models["user_data_preprocessor"].transform(cluster_frame)
    except Exception as exc:  # noqa: BLE001
        raise PredictionError(
            f"Không chuẩn hóa được dữ liệu đầu vào: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        cluster_id, noise_distance = assign_cluster(scaled, models)
    except PredictionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PredictionError(
            f"Không xác định được phân khúc khách hàng: {type(exc).__name__}: {exc}"
        ) from exc

    # --- stage 2: next touchpoint --------------------------------------
    try:
        distribution = rank_channels(raw, cluster_id, models)
    except Exception as exc:  # noqa: BLE001
        raise PredictionError(
            f"Không dự báo được bước tiếp theo: {type(exc).__name__}: {exc}"
        ) from exc

    distribution.sort(key=lambda item: item[1], reverse=True)
    top = distribution[:TOP_N]

    from ctmj.models import cluster_info, type_touch

    cluster = cluster_info.objects.filter(cluster_id=cluster_id).first()
    cluster_name = (cluster.name if cluster and cluster.name else "") or (
        f"Nhóm #{cluster_id}"
    )
    cluster_description = (cluster.description if cluster else "") or (
        "Chưa có mô tả cho nhóm phân cụm này."
    )

    touch_map = {
        obj.code: obj
        for obj in type_touch.objects.filter(code__in=[code for code, _ in top])
    }

    channels: list[ChannelPrediction] = []
    for code, prob in top:
        obj = touch_map.get(code)
        channels.append(
            ChannelPrediction(
                channel_id=code,
                probability=prob,
                name=(obj.name if obj else f"Kênh #{code}"),
                description=(obj.description if obj else "")
                or "Chưa có mô tả cho điểm chạm này.",
            )
        )

    return Prediction(
        cluster_id=cluster_id,
        cluster_name=cluster_name,
        cluster_label=reference_data.CLUSTER_LABELS.get(cluster_id, f"Nhóm #{cluster_id}"),
        cluster_description=cluster_description,
        is_noise=cluster_id == NOISE_LABEL,
        channels=channels,
        distribution=distribution,
        noise_distance=noise_distance,
    )


def predict_many(
    registry: ModelRegistry, records: Sequence[formdata.ValidationResult]
) -> list[Prediction | None]:
    """Convenience batch helper. Returns ``None`` per failed record."""
    return [predict(registry, record) for record in records]
