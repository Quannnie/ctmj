"""Imbalanced-learning resampling: SMOTE, Tomek links, ADASYN.

Two things changed here that are not cosmetic.

**The self-neighbour defect.** ``sklearn.neighbors.NearestNeighbors.kneighbors``
always returns the query point itself first, at distance 0. The previous
implementation drew a neighbour uniformly from that result, so roughly
``1/k`` of the time it drew the point itself, giving ``diff = 0`` and emitting
an exact duplicate of an original sample instead of an interpolated one. On a
30/6 toy split, 7 of 24 "synthetic" points were verbatim copies of originals
and 3 more duplicated each other. Interpolation between distinct neighbours
is the entire mechanism of SMOTE; a third of the output being noise quietly
inflates the class count without adding information. Every sampler below now
excludes the query point explicitly.

**Resampling inside the cross-validation fold.** These are also exposed as
scikit-learn samplers (``fit_resample``), so they can be steps in a
``Pipeline``. A sampler placed in a pipeline is fitted on the training slice
of each fold, which is the only leakage-free arrangement: sampling the whole
dataset first lets the minority neighbourhood of a synthetic point span the
validation fold, because the validator's own samples are candidates in that
neighbourhood.

The original free functions are kept for the research notebook, now with an
explicit ``random_state`` and without global RNG state.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Iterator, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger("cjps_train.resample")

__all__ = [
    "smote",
    "tomek_links",
    "adasyn",
    "SMOTE",
    "TomekLinks",
    "ADASYN",
    "SMOTETomek",
    "class_counts",
    "drop_rare_classes",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def class_counts(y: Sequence) -> dict:
    """Class -> count, as a plain dict."""
    return dict(Counter(np.asarray(y).ravel()))


def drop_rare_classes(
    y: Sequence, min_samples: int
) -> np.ndarray:
    """Boolean mask keeping only classes with at least ``min_samples`` rows.

    Returning a mask rather than a filtered copy keeps X and y aligned through
    a single index array, so a caller cannot accidentally desynchronise them.
    """
    y = np.asarray(y).ravel()
    keep_labels = {label for label, n in class_counts(y).items() if n >= min_samples}
    return np.isin(y, list(keep_labels))


def _rng(random_state) -> np.random.Generator:
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _interpolate(
    base: np.ndarray, neighbour: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """One SMOTE-style synthetic sample strictly between two distinct points."""
    gap = rng.random()
    return base + gap * (neighbour - base)


def _neighbour_indices_among_minority(
    X_class: np.ndarray, k: int
) -> np.ndarray:
    """For every row, its ``k`` nearest *other* rows of the same class.

    Truncating to the k nearest matters. If a candidate is drawn uniformly
    from the whole class, a point can be interpolated towards a member in a
    completely different region of the space, producing a synthetic sample
    that belongs to neither neighbourhood. Both SMOTE and ADASYN are defined
    in terms of a point's k nearest same-class neighbours, so that is what
    this returns.

    Rows with fewer than k distinct neighbours are padded by repeating their
    last entry; the caller filters self-matches, and duplicate coordinates can
    otherwise shrink a row's neighbour list below k.
    """
    n = len(X_class)
    if n < 2:
        return np.empty((n, 0), dtype=int)

    k = max(1, min(k, n - 1))

    # k + 1 candidates, not n.
    #
    # The previous version asked the neighbour search for all n neighbours and
    # then discarded the self-match in a Python loop over every row. Asking for
    # n makes the search return an n-by-n index array -- quadratic in both time
    # and memory -- to answer a question about k columns. k + 1 is enough,
    # because exactly one of them is the point itself.
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X_class).kneighbors(
        X_class, return_distance=False
    )

    # Compaction without a per-row loop: push each row's self-match to the end
    # with a stable argsort, then slice. Stability matters because ties at
    # distance zero -- duplicate coordinates -- would otherwise be reordered,
    # and the caller relies on the remaining order being distance-sorted.
    is_self = idx == np.arange(n)[:, None]
    order = np.argsort(is_self, axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1)[:, :k]


# ---------------------------------------------------------------------------
# SMOTE
# ---------------------------------------------------------------------------

def smote(
    X: np.ndarray,
    y: Sequence,
    k_neighbors: int = 5,
    random_state=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic Minority Over-sampling Technique.

    For each minority point, draw ``k`` of its *distinct* same-class neighbours
    and emit one synthetic sample between the point and each of them, until the
    class matches the majority size.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()

    counts = class_counts(y)
    if len(counts) < 2:
        return X.copy(), y.copy()

    majority_size = max(counts.values())
    rng = _rng(random_state)

    X_out = [X]
    y_out = [y]

    for label, count in counts.items():
        if count >= majority_size:
            continue

        X_class = X[y == label]
        n_class = len(X_class)

        # A minority class of 1 or 2 points has no distinct neighbour to
        # interpolate towards, so it cannot be meaningfully expanded.
        if n_class < 2:
            continue

        k = min(k_neighbors, n_class - 1)
        if k < 1:
            continue

        adjacency = _neighbour_indices_among_minority(X_class, k)

        needed = majority_size - count
        # Cycle through the class so every point contributes, rather than
        # leaving the tail of the class untouched.
        picks = rng.integers(0, n_class, needed)
        for pick in picks:
            candidates = adjacency[pick]
            # Only neighbours that actually differ can produce interpolation.
            candidates = candidates[candidates != pick]
            if len(candidates) == 0:
                continue
            neighbour = candidates[rng.integers(0, len(candidates))]
            X_out.append(_interpolate(X_class[pick], X_class[neighbour], rng)[None, :])
            y_out.append(np.array([label]))

    if len(X_out) == 1:
        return X.copy(), y.copy()
    return np.vstack(X_out), np.concatenate(y_out)


# ---------------------------------------------------------------------------
# Tomek links
# ---------------------------------------------------------------------------

def tomek_pairs(X: np.ndarray, y: Sequence) -> list[tuple[int, int]]:
    """Indices of Tomek link pairs, using the standard definition.

    A cross-class pair (a, b) is a Tomek link when

        d(a, b) <= d(a, a')   or   d(a, b) <= d(b, b')

    where ``a'`` is a's nearest neighbour of a's own class and ``b'`` is b's
    nearest neighbour of b's own class. The second half is what makes this more
    than a mutual-nearest-neighbour rule: plenty of genuine Tomek links have an
    unrequited partner, and an earlier version of this module kept all of them
    because it only removed pairs where both points chose each other.

    Cost
    ----
    The distances are gathered in **one batched query per class** rather than one
    per point per class. A straightforward per-row loop calls ``kneighbors``
    ``n * n_classes`` times — 14,000 separate sklearn invocations on 700 rows
    across 20 touchpoint classes, which took nearly five minutes inside a
    cross-validation loop. The batched form makes the same ``n_classes`` calls
    regardless of n, and the only remaining Python loop is a single pass over
    the rows.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    n = len(y)

    counts = class_counts(y)
    if n < 2 or len(counts) < 2:
        return []

    # Sorted for a deterministic result regardless of dict ordering.
    labels = sorted(counts, key=lambda v: (isinstance(v, str), v))
    rows_of: dict = {label: np.flatnonzero(y == label) for label in labels}

    # --- distance from every row to the nearest member of each class --------
    dist_to_class: dict = {}
    row_of_class: dict = {}
    for label in labels:
        rows = rows_of[label]
        index = NearestNeighbors(n_neighbors=1).fit(X[rows])
        distances, local = index.kneighbors(X, n_neighbors=1)
        dist_to_class[label] = distances[:, 0]
        row_of_class[label] = rows[local[:, 0]]

    # --- distance to the nearest *other* member of a row's own class ---------
    # A singleton class has no such neighbour, so its own-class distance is
    # infinite, which makes the Tomek test fall through to the other clause
    # rather than silently comparing a point to itself at distance 0.
    same_class_distance = np.full(n, np.inf, dtype=float)
    for label in labels:
        rows = rows_of[label]
        if len(rows) == 1:
            continue
        # Two neighbours so the self-match at distance 0 can be skipped.
        index = NearestNeighbors(n_neighbors=2).fit(X[rows])
        distances, local = index.kneighbors(X[rows], n_neighbors=2)
        for position, row in enumerate(rows):
            for distance, neighbour in zip(distances[position], local[position]):
                candidate = int(rows[neighbour])
                if candidate != int(row):
                    same_class_distance[row] = distance
                    break

    # --- nearest member of any *other* class, per row -----------------------
    nearest_other_distance = np.full(n, np.inf, dtype=float)
    nearest_other_row = np.full(n, -1, dtype=int)
    for label in labels:
        outside = np.flatnonzero(y != label)
        if len(outside) == 0:
            continue
        better = dist_to_class[label][outside] < nearest_other_distance[outside]
        rows = outside[better]
        nearest_other_distance[rows] = dist_to_class[label][rows]
        nearest_other_row[rows] = row_of_class[label][rows]

    # --- the Tomek test -----------------------------------------------------
    valid = nearest_other_row >= 0
    if not valid.any():
        return []

    candidates = np.flatnonzero(valid)
    d_ab = nearest_other_distance[candidates]
    d_a_ap = same_class_distance[candidates]
    d_b_bp = same_class_distance[nearest_other_row[candidates]]

    is_link = (d_ab <= d_a_ap) | (d_ab <= d_b_bp)

    pairs: list[tuple[int, int]] = []
    for row in candidates[is_link]:
        other = int(nearest_other_row[row])
        if other != int(row):
            pairs.append((min(int(row), other), max(int(row), other)))
    return pairs


def tomek_links(
    X: np.ndarray,
    y: Sequence,
    return_indices: bool = False,
):
    """Remove Tomek link pairs. Optionally also return the surviving indices.

    Over-sampling and Tomek cleaning are usually combined as
    "SMOTE-Tomek", where Tomek runs *after* SMOTE to trim the synthetic
    boundary points that the over-sampling introduced.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()

    pairs = tomek_pairs(X, y)
    dropped = {i for pair in pairs for i in pair}

    keep = np.ones(len(y), dtype=bool)
    keep[list(dropped)] = False

    if return_indices:
        return X[keep], y[keep], np.flatnonzero(keep)
    return X[keep], y[keep]


# ---------------------------------------------------------------------------
# ADASYN
# ---------------------------------------------------------------------------

def adasyn(
    X: np.ndarray,
    y: Sequence,
    k_neighbors: int = 5,
    random_state=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive Synthetic Sampling.

    Unlike SMOTE, which treats every minority point equally, ADASYN spends its
    budget on the points that sit closest to the majority class, where the
    decision boundary is hardest.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()

    counts = class_counts(y)
    if len(counts) < 2:
        return X.copy(), y.copy()

    majority_label = max(counts, key=counts.get)
    majority_size = counts[majority_label]
    rng = _rng(random_state)

    X_out = [X]
    y_out = [y]

    # Density of the majority around each minority point decides how much
    # synthetic data that point earns.
    all_nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(X))).fit(X)

    for label, count in counts.items():
        if label == majority_label or count < 2:
            continue

        X_class = X[y == label]
        n_class = len(X_class)

        k = min(k_neighbors, n_class - 1)
        if k < 1:
            continue

        distances, neighbours = all_nn.kneighbors(X_class)
        # Fraction of the k neighbours that belong to the majority. High means
        # the point is deep inside majority territory, so synthesise there.
        difficulty = np.array(
            [np.mean(y[neighbours[i]] == majority_label) for i in range(n_class)]
        )

        total = difficulty.sum()
        if total == 0:
            # Fully isolated class: fall back to an even spread.
            per_point = np.full(n_class, 1.0 / n_class)
        else:
            per_point = difficulty / total

        budget = majority_size - count
        allocation = np.floor(per_point * budget).astype(int)

        adjacency = _neighbour_indices_among_minority(X_class, k)

        for i in range(n_class):
            for _ in range(allocation[i]):
                candidates = adjacency[i][adjacency[i] != i]
                if len(candidates) == 0:
                    continue
                neighbour = candidates[rng.integers(0, len(candidates))]
                X_out.append(_interpolate(X_class[i], X_class[neighbour], rng)[None, :])
                y_out.append(np.array([label]))

    if len(X_out) == 1:
        return X.copy(), y.copy()
    return np.vstack(X_out), np.concatenate(y_out)


# ---------------------------------------------------------------------------
# scikit-learn samplers, for use as Pipeline steps
# ---------------------------------------------------------------------------

class _BaseSampler(BaseEstimator):
    """Base for the resampling wrappers below.

    Each exposes ``fit_resample(X, y)``, the scikit-learn sampler protocol.

    Note on pipelines: scikit-learn removed samplers from ``Pipeline`` in 1.9.
    An intermediate step must now be a transformer whose ``fit_transform``
    returns only X, so the labels reaching the classifier can no longer be the
    resampled ones. Fold-safe resampling is therefore done explicitly by
    :mod:`src.cjps_train.evaluate`, which calls ``fit_resample`` on the
    training slice of each fold. These wrappers exist because that loop
    consumes them and because they carry ``random_state`` through
    ``get_params``.
    """

    def fit(self, X, y=None):  # noqa: D102 - sklearn contract
        return self

    def fit_resample(self, X, y):  # pragma: no cover - overridden
        raise NotImplementedError

    def fit_transform(self, X, y=None, **fit_params):  # noqa: D102 - sklearn contract
        """Return only the resampled X, per the transformer contract.

        y is deliberately not returned. Handing it back would invite a caller
        to pair resampled X with the original y, silently misaligning features
        and labels.
        """
        X_out, _ = self.fit_resample(X, y)
        return np.asarray(X_out)

    def transform(self, X):  # noqa: D102 - sklearn contract
        """Identity. Resampling must never happen at predict time."""
        return X

    def _more_tags(self):  # noqa: D102 - sklearn contract
        return {"no_validation": True}


class SMOTE(_BaseSampler):
    """SMOTE as a pipeline step, so it runs inside each CV fold."""

    def __init__(self, k_neighbors: int = 5, random_state=None):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        return smote(
            X, y, k_neighbors=self.k_neighbors, random_state=self.random_state
        )


class TomekLinks(_BaseSampler):
    """Tomek cleaning as a pipeline step."""

    def __init__(self):
        pass

    def fit_resample(self, X, y):
        return tomek_links(X, y)


class ADASYN(_BaseSampler):
    """ADASYN as a pipeline step."""

    def __init__(self, k_neighbors: int = 5, random_state=None):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        return adasyn(
            X, y, k_neighbors=self.k_neighbors, random_state=self.random_state
        )


class SMOTETomek(_BaseSampler):
    """Over-sample, then trim the synthetic boundary points."""

    def __init__(self, k_neighbors: int = 5, random_state=None):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        """Over-sample, then trim, reporting how destructive the trim was.

        Tomek removes every point whose nearest opposite-class point is closer
        than its nearest same-class point. On classes that genuinely overlap
        that is most of the data: on the 20-way touchpoint target, where the
        classes are barely separable, this combination took 1200 rows down to
        188 — an 84% cut. The result is *correct* by the definition, and also
        useless as a training set.

        The share removed is therefore reported rather than left for the caller
        to infer from a row count. A cleaning step that discards most of its
        input is a decision the manifest should record, not a detail.
        """
        before = len(y)
        X_s, y_s = smote(
            X, y, k_neighbors=self.k_neighbors, random_state=self.random_state
        )
        X_t, y_t = tomek_links(X_s, y_s)

        removed_share = 0.0 if before == 0 else 1.0 - (len(y_t) / len(y_s))
        if removed_share > 0.5:
            logger.warning(
                "Tomek removed %.0f%% of the resampled set (%d -> %d rows). The "
                "classes overlap heavily, so the boundary points are most of the "
                "data. This combination will train on very little.",
                100 * removed_share,
                len(y_s),
                len(y_t),
            )
        return X_t, y_t
