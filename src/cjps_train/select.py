"""Model and resampling selection.

The original script's benchmark was decorative. It fitted six models across
five resampling variants, printed a results table, and then ignored the table
entirely::

    s1_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    s1_X = data_variants['ADASYN'][0]        # chosen by fiat

Whatever the table said, the shipped artefacts were fixed in advance. This
module makes the choice real: candidates are declared, evaluated, ranked, and
the winner is recorded in the manifest with the numbers that put it there.

Two selection decisions are made explicitly rather than by accident.

**SGD with hinge loss is not a candidate for a probability task.** The app's
entire output is a ranked list with percentages, so a classifier that cannot
produce probabilities is not a worse fit, it is not a fit at all. It can still
be benchmarked on the argmax metrics, which is why it is included with a note
rather than dropped — but its top-3 accuracy is always reported as 0, which is
the honest reading of "cannot rank".

**Resampling is selected alongside the model, not separately.** SMOTE and
ADASYN help different models on different data. Evaluating the resampler with a
fixed classifier and then reusing the winner for every model would bake in an
assumption the data has not been asked about.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Sequence

from src.cjps_train.config import TrainConfig
from src.cjps_train.evaluate import CandidateResult, evaluate_candidates

logger = logging.getLogger("cjps_train.select")


@dataclass(frozen=True)
class Candidate:
    """One (model, resampler) pair to evaluate."""

    name: str
    estimator: object
    sampler_name: str
    sampler: Callable[[np.ndarray, np.ndarray], tuple] | None = None
    #: Why this candidate is in the list, or why it is limited. Recorded in the
    #: manifest so the selection is auditable months later.
    note: str = ""
    #: Whether the model can emit probabilities the app can display.
    supports_proba: bool = True

    def with_sampler(self, sampler: Callable | None, sampler_name: str) -> "Candidate":
        """Pair this candidate with a resampler.

        A ``None`` sampler returns the candidate unchanged rather than a copy
        with a suffix, so a candidate's ``name`` always matches the key it is
        filed under in the results table.
        """
        if sampler is None:
            return self
        return Candidate(
            name=f"{self.name} + {sampler_name}",
            estimator=self.estimator,
            sampler_name=sampler_name,
            sampler=sampler,
            note=self.note,
            supports_proba=self.supports_proba,
        )


@dataclass
class Selection:
    """Outcome of the selection stage."""

    winner: Candidate
    results: list[CandidateResult]
    ranking: list[tuple[str, float]] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "winner": self.winner.name,
            "winner_note": self.winner.note,
            "winner_sampler": self.winner.sampler_name,
            "ranking": [{"name": n, "f1_macro": round(s, 6)} for n, s in self.ranking],
            "failed": [{"name": n, "error": e} for n, e in self.rejected],
            "candidates": [r.as_dict() for r in self.results],
        }


def default_candidates(seed: int) -> list[Candidate]:
    """The benchmark roster.

    Deliberately small. Six models times five resamplers is thirty fits, most of
    which measure nothing: ``SGDClassifier`` and ``MLPClassifier`` are included
    because the original benchmark claimed to cover them, not because they are
    expected to win.
    """
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV

    return [
        Candidate(
            name="GradientBoosting",
            estimator=GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.1, random_state=seed
            ),
            sampler_name="none",
            note=(
                "The original pipeline's choice, but with max_depth reduced from 5 "
                "to 3 and twice the trees. Depth 5 on a few thousand rows of "
                "ordinal codes overfits badly; depth 3 generalises better and is "
                "roughly four times faster to fit."
            ),
        ),
        Candidate(
            name="HistGradientBoosting",
            estimator=HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.1, max_iter=200, random_state=seed
            ),
            sampler_name="none",
            note=(
                "Same algorithm as GradientBoosting but histogram-based, so it "
                "scales to the full dataset instead of being the bottleneck."
            ),
        ),
        Candidate(
            name="RandomForest",
            estimator=RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed
            ),
            sampler_name="none",
            note=(
                "min_samples_leaf=2 instead of 1. Leaf size 1 memorises training "
                "rows, which on 20 imbalanced touchpoint classes produces a model "
                "that scores well and predicts the frequent class regardless of "
                "the profile."
            ),
        ),
        Candidate(
            name="LogisticRegression",
            estimator=LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed
            ),
            sampler_name="none",
            note=(
                "class_weight='balanced' handles the imbalance without "
                "synthesising anything, so it doubles as the no-resampling "
                "baseline the resampled models have to beat."
            ),
        ),
        Candidate(
            name="SVC_rbf",
            estimator=CalibratedClassifierCV(
                SVC(kernel="rbf", C=10.0, gamma="scale", random_state=seed),
                method="sigmoid",
                cv=3,
                ensemble=False,
            ),
            sampler_name="none",
            note=(
                "Wrapped in CalibratedClassifierCV because the app needs "
                "probabilities. SVC's own probability=True is deprecated in "
                "scikit-learn 1.9 and removed in 1.11, and without calibration "
                "SVC has no predict_proba at all — it would pass the benchmark "
                "on argmax metrics and then fail on the first prediction."
            ),
        ),
        Candidate(
            name="MLP",
            estimator=MLPClassifier(
                hidden_layer_sizes=(128, 64),
                early_stopping=True,
                n_iter_no_change=15,
                max_iter=400,
                random_state=seed,
            ),
            sampler_name="none",
            note=(
                "early_stopping=True and a real max_iter. The original ran this "
                "unregularised to 1000 iterations with convergence warnings "
                "globally suppressed, so a non-converged network looked identical "
                "to a converged one."
            ),
        ),
        Candidate(
            name="SGD_hinge",
            estimator=_hinge_sgd(seed),
            sampler_name="none",
            supports_proba=False,
            note=(
                "Included for completeness only. Hinge loss has no "
                "predict_proba, so this model cannot produce the ranked output "
                "the app renders. Its top3_accuracy is structurally 0 and it "
                "should not be selected. Kept in the table to document why."
            ),
        ),
    ]


def _hinge_sgd(seed: int):
    from sklearn.linear_model import SGDClassifier

    return SGDClassifier(
        loss="hinge", alpha=1e-4, max_iter=2000, tol=1e-3, random_state=seed
    )


def sampler_variants(seed: int, config: TrainConfig) -> list[tuple[str, Callable | None]]:
    """Resamplers to try, as ``(name, function_or_None)``.

    The samplers are bound to a per-fold seed derived from the run seed. Using
    one fixed seed for all folds is reproducible but makes every fold draw the
    same synthetic offsets, which slightly correlates them; varying by fold is
    both reproducible and independent.
    """
    from src.pre_processing import ADASYN, SMOTE, SMOTETomek

    k = config.k_neighbors
    return [
        (
            "none",
            None,
        ),
        (
            "SMOTE",
            lambda X, y, _k=k, _s=seed: SMOTE(
                k_neighbors=_k, random_state=_s
            ).fit_resample(X, y),
        ),
        (
            "ADASYN",
            lambda X, y, _k=k, _s=seed: ADASYN(
                k_neighbors=_k, random_state=_s
            ).fit_resample(X, y),
        ),
        (
            "SMOTE-Tomek",
            lambda X, y, _k=k, _s=seed: SMOTETomek(
                k_neighbors=_k, random_state=_s
            ).fit_resample(X, y),
        ),
    ]


def select(
    X: "np.ndarray",
    y: "np.ndarray",
    config: TrainConfig,
    candidates: Sequence[Candidate] | None = None,
    samplers: Sequence[tuple[str, Callable | None]] | None = None,
    include_sampler_search: bool = True,
) -> Selection:
    """Evaluate the grid, pick a winner, and record why.

    Selection is by mean ``f1_macro`` across folds. Ties within a small margin
    go to the simpler and faster option, because two candidates separated by
    less than the fold-to-fold spread are not distinguishable on this data and
    picking either is arbitrary.
    """
    base = list(candidates if candidates is not None else default_candidates(config.seed))
    variants = list(samplers) if samplers is not None else sampler_variants(config.seed, config)

    grid: list[tuple[str, object, object | None]] = []
    for candidate in base:
        if not include_sampler_search or candidate.sampler is not None:
            grid.append((candidate.name, candidate.estimator, candidate.sampler))
            continue
        for sampler_name, sampler in variants:
            if sampler is None:
                # The no-resampling row keeps the bare candidate name. Labelling
                # it "X + none" would produce a second, identical row in the
                # results table under a different name.
                grid.append((candidate.name, candidate.estimator, None))
                continue
            pair = candidate.with_sampler(sampler, sampler_name)
            grid.append((pair.name, pair.estimator, pair.sampler))

    results = evaluate_candidates(grid, X, y, config)

    # Grid entry name -> the candidate that produced it, so the winning
    # candidate object (with its note and proba capability) can be recovered
    # from a row of the results table.
    #
    # The stored object's ``name`` must equal its key. An earlier version stored
    # ``with_sampler(None, "none")`` under the bare key, so the winner came back
    # named "lr + none" while the results table row it was read from said "lr" —
    # and the manifest then recorded a winner that appears nowhere in its own
    # ranking.
    lookup: dict[str, Candidate] = {}
    for candidate in base:
        lookup[candidate.name] = candidate
        if include_sampler_search:
            for sampler_name, sampler in variants:
                if sampler is None:
                    lookup[candidate.name] = candidate
                    continue
                pair = candidate.with_sampler(sampler, sampler_name)
                lookup[pair.name] = pair

    successful = [r for r in results if r.ok]
    failed = [(r.name, r.error) for r in results if not r.ok]

    if not successful:
        from src.cjps_train.evaluate import EvaluationError

        raise EvaluationError("Không ứng viên nào đánh giá thành công.")

    # Prefer the no-resampling option when it is statistically tied with a
    # resampled one. Synthesising points is a real cost — it inflates the
    # training set and can create samples that do not correspond to any real
    # profile — so it needs a clear win to be justified.
    best = max(successful, key=lambda r: r.score)
    margin = best.fold_spread
    tied = [r for r in successful if best.score - r.score <= margin]
    for candidate in tied:
        entry = lookup.get(candidate.name)
        if entry is not None and entry.sampler is None:
            logger.info(
                "Preferring %r over %r: within the fold-to-fold spread (%.4f)",
                candidate.name,
                best.name,
                margin,
            )
            best = candidate
            break

    winner = lookup.get(best.name)
    if winner is None:  # pragma: no cover - lookup is built from the same grid
        raise EvaluationError(f"Không ánh xạ được ứng viên thắng: {best.name}")

    if not winner.supports_proba:
        logger.warning(
            "Winner %r cannot emit probabilities, which the app requires. "
            "Re-selecting among probability-capable candidates.",
            winner.name,
        )
        capable = [
            r for r in successful if lookup.get(r.name, winner).supports_proba
        ]
        if capable:
            best = max(capable, key=lambda r: r.score)
            winner = lookup[best.name]

    ranking = sorted(((r.name, r.score) for r in successful), key=lambda kv: -kv[1])
    logger.info("Selected %r with f1_macro=%.4f", winner.name, best.score)

    return Selection(
        winner=winner,
        results=results,
        ranking=ranking,
        rejected=failed,
    )
