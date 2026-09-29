"""Segmentation: noise detection and cluster-count selection.

The original pipeline computed a full clustering diagnostic — elbow curve,
silhouette across k, Davies-Bouldin — and then never used any of it::

    visualizer_all_metrics(X_scaled)        # prints elbow / silhouette / DBI
    dbscan = DBSCAN(eps=find_optimal_eps(X_scaled))
    X_clean = X_scaled[dbscan.labels_ != -1]
    spectral = SpectralClustering(n_clusters=3)   # hard-coded

So the number of segments was a constant that happened to sit inside the range
the diagnostic had just examined. This module makes the diagnostic load-bearing:
``select_k`` returns the k that the metrics actually support, and the chosen
value is written to the manifest alongside the scores that chose it.

The two stages play different roles and are kept apart deliberately.

**DBSCAN as a noise gate, not as the segmenter.** It answers "is this profile
near anything real?" and nothing else. Its ``labels_`` are never used as the
app's segment ids, because cluster ids from DBSCAN are not stable and its
clusters are defined by density connectivity rather than by the segment shape
the business cares about.

**Spectral clustering as the segmenter.** It partitions the survivors into k
groups. Its ``labels_`` are what the app serves, so k must be chosen from the
data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from src.cjps_train.config import TrainConfig

logger = logging.getLogger("cjps_train.cluster")


class ClusteringError(RuntimeError):
    """Raised when the data cannot support a segmentation at all."""


@dataclass
class NoiseChoice:
    """A candidate eps and what DBSCAN made of it."""

    percentile: float
    eps: float
    n_clusters: int
    n_noise: int
    noise_share: float
    core_sample_count: int

    def as_dict(self) -> dict:
        return {
            "percentile": self.percentile,
            "eps": round(self.eps, 6),
            "dbscan_clusters": self.n_clusters,
            "noise": self.n_noise,
            "noise_share": round(self.noise_share, 6),
            "core_samples": self.core_sample_count,
        }


@dataclass
class KChoice:
    """Silhouette and Davies-Bouldin for one candidate k."""

    k: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float

    def as_dict(self) -> dict:
        return {
            "k": self.k,
            "silhouette": round(self.silhouette, 6),
            "davies_bouldin": round(self.davies_bouldin, 6),
            "calinski_harabasz": round(self.calinski_harabasz, 4),
        }


@dataclass
class Segmentation:
    """The fitted stage-1 models plus the evidence behind their parameters."""

    dbscan: DBSCAN
    spectral: SpectralClustering
    eps: float
    eps_percentile: float
    noise_choice: NoiseChoice
    k_choice: KChoice
    k_candidates: list[KChoice] = field(default_factory=list)
    noise_candidates: list[NoiseChoice] = field(default_factory=list)
    k_fallback_reason: str = ""
    clean_index: np.ndarray | None = None

    @property
    def n_clusters(self) -> int:
        return int(self.k_choice.k)

    def as_dict(self) -> dict:
        return {
            "eps": round(self.eps, 6),
            "eps_percentile": self.eps_percentile,
            "n_clusters": self.n_clusters,
            "selected_k": self.k_choice.as_dict(),
            "k_candidates": [c.as_dict() for c in self.k_candidates],
            "selected_noise": self.noise_choice.as_dict(),
            "noise_candidates": [c.as_dict() for c in self.noise_candidates],
            "k_fallback_reason": self.k_fallback_reason,
        }


# ---------------------------------------------------------------------------
# DBSCAN eps
# ---------------------------------------------------------------------------


def k_distance_curve(X: np.ndarray, k: int) -> np.ndarray:
    """Distance to each point's k-th nearest neighbour, sorted ascending.

    The shape of this curve is what makes DBSCAN's eps scale-free: in a space
    of ten standardised features, an absolute distance threshold is arbitrary,
    but "the point where the curve bends" is not.
    """
    n = len(X)
    if n < 2:
        return np.zeros(1)
    k_eff = max(1, min(k, n - 1))
    neighbours = NearestNeighbors(n_neighbors=k_eff + 1).fit(X)
    distances, _ = neighbours.kneighbors(X)
    # Column -1 is the k-th neighbour; column 0 is the point itself at 0.
    return np.sort(distances[:, -1])


def noise_for_eps(
    X: np.ndarray, eps: float, min_samples: int
) -> NoiseChoice:
    """Fit DBSCAN at one eps and summarise the outcome."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    unique = set(int(v) for v in np.unique(labels))
    noise = int(np.sum(labels == -1))
    return NoiseChoice(
        percentile=float("nan"),
        eps=eps,
        n_clusters=len(unique - {-1}),
        n_noise=noise,
        noise_share=noise / len(labels) if len(labels) else 0.0,
        core_sample_count=int(len(getattr(model, "core_sample_indices_", []))),
    )


def select_eps(
    X: np.ndarray, config: TrainConfig, min_samples: int | None = None
) -> NoiseChoice:
    """Pick eps by searching k-distance percentiles and keeping a usable split.

    "Usable" means the gate actually gates: both a non-trivial number of
    clusters and a noise share that is neither zero (the gate is doing nothing)
    nor near-total (nothing survives and there is no model to fit). The original
    searched a fixed grid of raw distances and accepted whatever DBSCAN returned,
    including a configuration where every point became its own cluster.
    """
    k = config.k_distance_k
    min_samples = min_samples or k
    curve = k_distance_curve(X, k)

    results: list[NoiseChoice] = []
    for percentile in config.scaled_eps_percentiles(len(X)):
        if percentile >= 100:
            continue
        eps = float(np.percentile(curve, percentile))
        if eps <= 0:
            continue
        choice = noise_for_eps(X, eps, min_samples)
        choice.percentile = percentile
        results.append(choice)
        logger.info(
            "  eps p%-5.1f = %.4f -> %d cluster(s), %.1f%% noise, %d core",
            percentile,
            eps,
            choice.n_clusters,
            100 * choice.noise_share,
            choice.core_sample_count,
        )

    if not results:
        raise ClusteringError(
            "Không tìm được eps phù hợp. Dữ liệu clustering có thể hằng số "
            "(tất cả các dòng giống nhau) — kiểm tra lại bước chuẩn hóa."
        )

    # Prefer the smallest noise share that still yields more than one cluster.
    # DBSCAN's raw output is order-dependent on eps: too small and every point
    # is its own cluster (the segmenter then has nothing to partition), too
    # large and everything merges into one.
    usable = [r for r in results if r.n_clusters > 1 and r.n_noise > 0]
    if usable:
        best = min(usable, key=lambda r: r.noise_share)
    else:
        best = max(results, key=lambda r: r.core_sample_count)
        logger.warning(
            "No percentile produced both multiple clusters and some noise; "
            "falling back to the eps with the most core samples (%.1f%% noise)",
            100 * best.noise_share,
        )

    logger.info(
        "Chosen eps %.4f (k-distance p%.1f): %d cluster(s), %.1f%% noise",
        best.eps,
        best.percentile,
        best.n_clusters,
        100 * best.noise_share,
    )
    return NoiseChoice(
        percentile=best.percentile,
        eps=best.eps,
        n_clusters=best.n_clusters,
        n_noise=best.n_noise,
        noise_share=best.noise_share,
        core_sample_count=best.core_sample_count,
    )


# ---------------------------------------------------------------------------
# Cluster count
# ---------------------------------------------------------------------------


def evaluate_k(
    X: np.ndarray, k: int, seed: int
) -> KChoice | None:
    """Silhouette, Davies-Bouldin and Calinski-Harabasz for one k.

    Returns ``None`` when k is not computable (too few samples, degenerate
    partition) rather than raising, so one bad candidate does not abort the
    search.
    """
    n = len(X)
    if k < 2 or k >= n:
        return None

    # KMeans is used as the probe rather than the spectral model because it is
    # orders of magnitude cheaper, and the internal-validity metrics are
    # partition-quality measures: they rank partitions of the same data. The
    # k they select is then fitted with the spectral model for the artefacts.
    # Using the expensive model here would make the search unusable.
    try:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    except Exception as exc:  # noqa: BLE001
        logger.warning("k=%d failed to fit: %s", k, exc)
        return None

    if len(set(int(v) for v in np.unique(labels))) < 2:
        # A single-cluster partition has no meaningful silhouette; sklearn
        # returns 0.0, which is not a score to select on.
        return None

    try:
        sil = float(silhouette_score(X, labels))
    except ValueError as exc:  # pragma: no cover - guarded above
        logger.warning("k=%d silhouette failed: %s", k, exc)
        return None

    return KChoice(
        k=k,
        silhouette=sil,
        davies_bouldin=float(davies_bouldin_score(X, labels)),
        calinski_harabasz=float(calinski_harabasz_score(X, labels)),
    )


def select_k(
    X: np.ndarray, config: TrainConfig
) -> tuple[KChoice, list[KChoice], str]:
    """Choose k from internal-validity metrics, with a documented fallback.

    Three metrics are computed because each fails differently. Silhouette is the
    primary signal, Davies-Bouldin lower-is-better is the tie-break (it is less
    sensitive to cluster size than silhouette), and Calinski-Harabasz
    higher-is-better is recorded for the record. The original printed all three
    and selected none.

    When the best silhouette is below ``silhouette_floor`` the structure is
    weak — a flat landscape where the winner barely beats k=2. Trusting the argmax
    there would be reading noise, so the midpoint candidate is used and the
    reason is recorded.
    """
    candidates: list[KChoice] = []
    for k in config.n_clusters_candidates:
        choice = evaluate_k(X, k, config.seed)
        if choice is not None:
            candidates.append(choice)
        else:
            logger.info("  k=%d not computable on %d samples; skipped", k, len(X))

    if not candidates:
        raise ClusteringError(
            f"Không đánh giá được số cụm nào trên {len(X)} mẫu. "
            "Dữ liệu sau bước lọc nhiễu còn quá ít điểm để phân đoạn."
        )

    # Primary: silhouette. Tie-break: lowest Davies-Bouldin.
    best = sorted(candidates, key=lambda c: (-c.silhouette, c.davies_bouldin))[0]

    reason = ""
    if best.silhouette < config.silhouette_floor:
        midpoint = candidates[len(candidates) // 2]
        reason = (
            f"Best silhouette {best.silhouette:.3f} is below the "
            f"{config.silhouette_floor:.2f} floor, so the partition structure is "
            f"weak. Falling back to k={midpoint.k} rather than trusting an "
            f"argmax that barely separates from k={candidates[0].k}."
        )
        logger.warning(reason)
        best = midpoint

    logger.info(
        "Selected k=%d (silhouette %.3f, DBI %.3f) from %s",
        best.k,
        best.silhouette,
        best.davies_bouldin,
        ", ".join(f"k={c.k}:{c.silhouette:.3f}" for c in candidates),
    )
    return best, candidates, reason


# ---------------------------------------------------------------------------
# The two stages together
# ---------------------------------------------------------------------------


def fit_segmentation(
    X_scaled: np.ndarray, config: TrainConfig
) -> Segmentation:
    """Fit the noise gate then the segmenter, and keep the evidence.

    The two models are fitted on different data — DBSCAN on everything, the
    spectral model on the survivors only — and that asymmetry is deliberate.
    The gate has to see every profile in order to judge it, while the segmenter
    should not be shaped by the outliers the gate just removed. The
    ``clean_index`` field records which rows the segmenter saw, because the
    spectral labels it produces are indexed against that subset and nowhere
    else. Mixing the two index spaces is the bug the web app's
    ``_spectral_labels_by_core_sample`` exists to avoid.
    """
    n = len(X_scaled)
    if n < 10:
        raise ClusteringError(
            f"Chỉ có {n} hồ sơ để phân cụm; cần tối thiểu 10. "
            "Kiểm tra lại bước lọc nhiễu hoặc điều chỉnh eps."
        )

    noise_choice = select_eps(X_scaled, config)
    dbscan = DBSCAN(eps=noise_choice.eps, min_samples=config.k_distance_k)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    clean_index = np.flatnonzero(dbscan_labels != -1)
    if len(clean_index) < 10:
        raise ClusteringError(
            f"eps={noise_choice.eps:.4f} loại bỏ toàn bộ dữ liệu còn lại "
            f"({len(clean_index)} mẫu). Nới khoảng percentile của eps."
        )

    X_clean = X_scaled[clean_index]
    logger.info(
        "Noise gate kept %d of %d profiles (%.1f%% noise)",
        len(clean_index),
        n,
        100 * (1 - len(clean_index) / n),
    )

    k_choice, k_candidates, fallback_reason = select_k(X_clean, config)

    spectral = SpectralClustering(
        n_clusters=k_choice.k,
        affinity="rbf",
        gamma=1.0,
        assign_labels="kmeans",
        random_state=config.seed,
        n_jobs=-1,
    )
    # Fit on the survivors only. ``labels_`` is indexed against X_clean.
    spectral_labels = spectral.fit_predict(X_clean)
    found = len(set(int(v) for v in np.unique(spectral_labels)))
    if found < 2:
        raise ClusteringError(
            f"Spectral clustering with k={k_choice.k} trả về {found} cụm. "
            "Tăng gamma hoặc giảm k."
        )
    logger.info("Spectral clustering produced %d segment(s)", found)

    return Segmentation(
        dbscan=dbscan,
        spectral=spectral,
        eps=float(noise_choice.eps),
        eps_percentile=float(noise_choice.percentile),
        noise_choice=noise_choice,
        k_choice=k_choice,
        k_candidates=k_candidates,
        noise_candidates=[],
        k_fallback_reason=fallback_reason,
        clean_index=clean_index,
    )


def segment_labels(segmentation: Segmentation, X_scaled: np.ndarray) -> np.ndarray:
    """Cluster id for every row of ``X_scaled``, with ``-1`` for gate-rejected rows.

    Mirrors what the app does at inference, on training data rather than a
    single form submission. Used to build the target column and to report the
    segment sizes that the classifier will be trained on.
    """
    labels = np.full(len(X_scaled), -1, dtype=int)
    clean_index = segmentation.clean_index
    if clean_index is None or len(clean_index) == 0:
        return labels
    labels[clean_index] = segmentation.spectral.labels_
    return labels


def segment_sizes(labels: Sequence[int]) -> dict[int, int]:
    """Rows per segment, for the manifest."""
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts) if int(v) >= 0}
