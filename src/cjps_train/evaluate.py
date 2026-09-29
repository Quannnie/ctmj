"""Leakage-free evaluation.

The defect this module exists to prevent
----------------------------------------
The original pipeline applied SMOTE/ADASYN to the **entire dataset** and only
then ran ``cross_validate`` over the result::

    X_sm, y_sm = smote(X_filtered, y_filtered)      # <- whole dataset
    for variant in variants:
        train_multi_model(variant, X_sm, y_sm)     # <- folds drawn afterwards

That is backwards, and it inflates every number it reports. SMOTE synthesises a
minority point by interpolating between a real minority sample and one of its
k nearest same-class neighbours. Once the whole dataset has been resampled, a
validation row is one of those neighbours, and the model is trained on a
synthetic sample derived from the very row it will later be scored on. The
reported F1 measures how well the model memorises its own training data.

The fix is ordering, not a different algorithm: the split comes first, and
resampling happens on the training slice only. Since scikit-learn 1.9 no longer
accepts samplers as ``Pipeline`` steps — an intermediate step must be a
transformer whose ``fit_transform`` returns only X, so the classifier would see
the original labels — the loop is written out here rather than delegated.

A second, quieter problem: the original wrapped ``cross_validate`` in a
``try/except`` that printed and continued, so a model that failed to fit simply
vanished from the results table. A missing row reads as "we did not think it was
worth trying", which is a very different claim from "it broke". Failures are
recorded here and surfaced.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.cjps_train.config import TrainConfig

logger = logging.getLogger("cjps_train.evaluate")


#: Metrics reported for every candidate. ``f1_macro`` is the selection metric
#: because the app ranks touchpoints: a model that is right about the two
#: frequent channels and wrong about the other eighteen is useless for ranking
#: no matter how good its accuracy looks.
REPORTED_METRICS: tuple[str, ...] = (
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "recall_macro",
    "accuracy",
    "balanced_accuracy",
    "top3_accuracy",
)


class EvaluationError(RuntimeError):
    """Raised when no candidate can be evaluated at all."""


@dataclass
class FoldResult:
    """Metrics from a single fold."""

    metrics: dict[str, float]
    n_train: int
    n_test: int
    classes: list = field(default_factory=list)


@dataclass
class CandidateResult:
    """Everything known about one candidate, successful or not."""

    name: str
    ok: bool
    metrics: dict[str, float] = field(default_factory=dict)
    fold_metrics: list[dict[str, float]] = field(default_factory=list)
    seconds: float = 0.0
    error: str = ""
    n_train: int = 0
    n_test: int = 0

    @property
    def score(self) -> float:
        """Primary score, or -inf when the candidate failed.

        Failed candidates sort last instead of being dropped, so the results
        table is a complete account of what was tried.
        """
        return float(self.metrics.get("f1_macro", float("-inf")))

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "ok": self.ok,
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
            "fold_scores": [round(m.get("f1_macro", float("nan")), 6) for m in self.fold_metrics],
            "fold_spread": self.fold_spread,
            "seconds": round(self.seconds, 3),
            "error": self.error,
            "n_train": self.n_train,
            "n_test": self.n_test,
        }

    @property
    def fold_spread(self) -> float:
        """Standard deviation of the per-fold selection score.

        Reported because a 0.62 mean from folds at 0.41 and 0.83 is a very
        different claim from 0.62 from five folds at 0.61, and the mean alone
        hides the difference.
        """
        scores = [m["f1_macro"] for m in self.fold_metrics if "f1_macro" in m]
        if len(scores) < 2:
            return 0.0
        return float(np.std(scores))


def safe_folds(y: Sequence, n_splits: int, seed: int) -> StratifiedKFold:
    """Build a stratified splitter, reducing ``n_splits`` if the data demands it.

    ``StratifiedKFold`` requires at least ``n_splits`` members in every class.
    Rather than letting it raise — or, worse, silently produce a fold missing a
    class — the fold count is lowered to what the rarest class can support.
    """
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    smallest = int(counts.min()) if len(counts) else 0
    usable = max(2, min(n_splits, smallest))
    if usable < n_splits:
        logger.warning(
            "Rarest class has %d member(s); reducing n_splits from %d to %d",
            smallest,
            n_splits,
            usable,
        )
    return StratifiedKFold(n_splits=usable, shuffle=True, random_state=seed)


def top_k_hit_rate(
    y_true: np.ndarray,
    y_proba: np.ndarray | None,
    classes: np.ndarray,
    k: int = 3,
) -> float:
    """Share of rows whose true label is among the ``k`` highest-scoring ones.

    Computed directly rather than through ``top_k_accuracy_score``, which
    insists that ``y_true`` and the score matrix agree on the label set. They
    routinely do not: a validation fold that happens to contain only one of the
    two classes breaks that contract, and the metric raises — taking the whole
    candidate's evaluation down with it, over a labelling subtlety in a metric
    the app does not even use directly.

    The mapping is positional: column *i* of ``y_proba`` corresponds to
    ``classes[i]``, which is the contract every scikit-learn classifier
    follows.
    """
    if y_proba is None or not len(y_proba) or not len(classes):
        return 0.0
    k = min(k, y_proba.shape[1], len(classes))
    if k < 1:
        return 0.0

    # Columns of the best k classes for each row, then back to label values.
    top_columns = np.argsort(-np.asarray(y_proba, dtype=float), axis=1)[:, :k]
    top_labels = np.asarray(classes)[top_columns]

    truth = np.asarray(y_true).reshape(-1, 1)
    return float(np.mean(np.any(top_labels == truth, axis=1)))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    classes: np.ndarray,
) -> dict[str, float]:
    """All reported metrics for one fold, in one place so they stay comparable."""
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        # Top-3 accuracy: the app returns three ranked candidates, so whether the
        # true touchpoint is among the three shown is the metric that matters to
        # a user. It is not recoverable from any of the others.
        "top3_accuracy": top_k_hit_rate(y_true, y_proba, classes, k=3),
    }


def cross_validate_candidate(
    name: str,
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    sampler: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None,
    config: TrainConfig,
) -> CandidateResult:
    """Evaluate one candidate with resampling confined to each training fold.

    ``sampler`` is applied to ``X[train_idx]`` and nothing else. That single
    line is the whole point of this module.
    """
    started = time.perf_counter()
    result = CandidateResult(name=name, ok=False)

    try:
        splitter = safe_folds(y, config.cv_splits, config.seed)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            if sampler is not None:
                # The boundary that matters: only the training slice is
                # resampled, so no validation row can reach a minority
                # neighbourhood and then be scored against a point built from
                # itself.
                X_train, y_train = sampler(X_train, y_train)

            model = clone(estimator)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            y_proba = None
            if hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)
                except (AttributeError, NotImplementedError):
                    # Hinge-loss SGD has no probabilities. It can still be
                    # scored on the argmax metrics, so this is not a failure.
                    y_proba = None

            classes = getattr(model, "classes_", np.unique(y_train))
            result.fold_metrics.append(compute_metrics(y_test, y_pred, y_proba, classes))
            result.n_train += len(train_idx)
            result.n_test += len(test_idx)
            logger.debug("%s fold %d: f1_macro=%.4f", name, fold, result.fold_metrics[-1]["f1_macro"])

        result.metrics = {
            key: float(np.mean([m[key] for m in result.fold_metrics]))
            for key in REPORTED_METRICS
        }
        result.ok = True
    except Exception as exc:  # noqa: BLE001
        # Recorded, not swallowed. A candidate that cannot be evaluated must
        # appear in the results table with its error attached.
        result.error = f"{type(exc).__name__}: {exc}"
        logger.warning("Candidate %s failed: %s", name, result.error)

    result.seconds = time.perf_counter() - started
    return result


def evaluate_candidates(
    candidates: Sequence[tuple[str, object, object | None]],
    X: np.ndarray,
    y: np.ndarray,
    config: TrainConfig,
) -> list[CandidateResult]:
    """Evaluate ``(name, estimator, sampler)`` triples.

    ``sampler`` may be ``None`` to evaluate the model on the raw data, which is
    itself a useful comparison: it shows what the resampling actually bought.
    """
    results: list[CandidateResult] = []
    for name, estimator, sampler in candidates:
        outcome = cross_validate_candidate(name, estimator, X, y, sampler, config)
        if outcome.ok:
            logger.info(
                "%-34s f1_macro=%.4f  top3=%.4f  (%.1fs)",
                name,
                outcome.metrics["f1_macro"],
                outcome.metrics["top3_accuracy"],
                outcome.seconds,
            )
        results.append(outcome)

    if not any(r.ok for r in results):
        errors = "; ".join(f"{r.name}: {r.error}" for r in results)
        raise EvaluationError(
            "Không mô hình nào đánh giá được. Lỗi chi tiết: " + errors
        )
    return results


def holdout_split(
    X: np.ndarray, y: np.ndarray, config: TrainConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/holdout split, performed exactly once.

    The holdout is taken before any resampling, tuning or model selection, and
    is touched exactly once at the end. Its purpose is a number that no
    decision in the pipeline could have influenced; the cross-validated scores
    cannot serve that purpose because they are what selection optimises.
    """
    counts = np.unique(y, return_counts=True)[1]
    smallest = int(counts.min())
    test_size = config.holdout_fraction
    # Never ask for more holdout rows than the rarest class can give, or the
    # split fails outright on an imbalanced target.
    if smallest * test_size < 1:
        test_size = 1.0 / smallest
        logger.warning(
            "Rarest class has only %d member(s); lowering holdout to %.3f",
            smallest,
            test_size,
        )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=config.seed, stratify=y
    )
    logger.info(
        "Holdout split: %d train / %d test (%.1f%%), seed=%d",
        len(X_tr),
        len(X_te),
        100 * len(X_te) / len(y),
        config.seed,
    )
    return X_tr, X_te, y_tr, y_te


def score_on_holdout(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sampler: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None,
) -> dict[str, float]:
    """Fit on the training split and score the holdout once.

    Even here the sampler sees only ``X_train``. Resampling the holdout would
    fabricate test points and the resulting score would be meaningless.
    """
    X_fit, y_fit = (sampler(X_train, y_train) if sampler is not None else (X_train, y_train))
    model = clone(estimator)
    model.fit(X_fit, y_fit)

    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
        except (AttributeError, NotImplementedError):
            y_proba = None
    classes = getattr(model, "classes_", np.unique(y_fit))
    return compute_metrics(y_test, y_pred, y_proba, classes)


def fit_final(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    sampler: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None,
):
    """Fit the shipped model on everything available.

    The sampler is applied to the *full* training set here, which is correct and
    not a leak: there is no held-out data left to protect at this point, and
    resampling is what the deployed model should have been trained on.
    """
    X_fit, y_fit = (sampler(X, y) if sampler is not None else (X, y))
    model = clone(estimator)
    model.fit(X_fit, y_fit)
    return model


def fit_final_on_subset(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    row_mask: np.ndarray,
    sampler: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None,
):
    """Fit on a subset of rows, resampling only within the subset.

    Used for per-segment models: the rows outside the segment must not
    contribute to the minority neighbourhood, or one segment's touchpoints
    would leak into another's training data.
    """
    X_sub = X[row_mask]
    y_sub = y[row_mask]
    X_fit, y_fit = (sampler(X_sub, y_sub) if sampler is not None else (X_sub, y_sub))
    model = clone(estimator)
    model.fit(X_fit, y_fit)
    return model
