"""Configuration for the CJPS training pipeline.

Two things this module fixes relative to ``src/main.py``:

**Paths are absolute.** The original script resolved ``"../resources/model"``
against the current working directory, so ``joblib.dump`` only found its
destination when the script happened to be launched from ``src/``. Every path
here is derived from this file's own location, so the pipeline writes to the
same place regardless of where it is invoked from.

**The data location is configuration, not a constant.** The original hardcoded
``D:\\MASTER\\2.1. ADM\\...``, which is one person's machine. ``DatasetPaths``
is filled from the environment or the CLI instead.

The feature lists are duplicated from ``ctmj.services.formdata`` on purpose
rather than imported. ``src/`` must stay runnable without Django configured —
the training pipeline is a standalone tool — so the contract is restated here
and :mod:`src.cjps_train.contracts` asserts the two copies still agree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Repository root, i.e. the directory containing manage.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Where the training run writes artefacts, matching the app's default MODEL_DIR.
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"

#: Where the run record and per-run reports land.
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "resources" / "training"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

#: Every stochastic step in the pipeline draws from this. Recording it in the
#: manifest is what makes a run reproducible; the original script relied on the
#: global numpy RNG, so two runs of the same code produced different models.
SEED = 20260929


# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------
# These names are load-bearing. ``ColumnTransformer`` selects by column name, so
# a rename here silently produces a transformer that drops the renamed column
# and predicts from fewer features than the metrics claimed.

#: Columns fed to the stage-1 clustering preprocessor. Mirrors
#: ``ctmj.services.formdata.CLUSTERING_FEATURES`` exactly.
#:
#: Kept identical to the app on purpose. ``ColumnTransformer`` selects by
#: column name, so training on an extra derived column (a one-hot ``GenderID``,
#: say) would make inference fail outright: the app builds a one-row frame from
#: the ten form fields, and the transformer would raise on the missing column.
#: ``GenderID`` needs no dummy either way, since MinMaxScaler maps {1, 2} to
#: {0, 1}, which is already a correct binary encoding.
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

#: Columns fed to the stage-2 prediction preprocessor. ``final_label`` is the
#: cluster id and must be present at inference time, which is why the app adds
#: it to the one-row frame before transforming.
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

#: Target column: the touchpoint the model must predict.
TARGET = "step1"

#: Journey columns the app form maps onto. ``step1`` is the most recent observed
#: touchpoint, ``step2`` the one before it, ``step3`` the one before that.
JOURNEY_TARGET = "step1"
JOURNEY_INPUTS: tuple[str, ...] = ("step2", "step3")

#: User-profile columns carried through from the user table.
USER_COLUMNS: tuple[str, ...] = (
    "UserID",
    "GenderID",
    "Age",
    "SPSS_Regio5",
    "BAS_huishoudgrootte",
    "BAS_werkzaamheid_resp",
    "afg_kinderen_huishouden",
    "BAS_bruto_jaarinkomen",
    "AFG_sk2015",
    "BAS_voltooide_opleiding8_resp",
    "SPSS_Lifestage",
)

#: The placeholder the source data uses for "not answered". Treating it as a
#: number instead of missing would make it the most common "age" in the dataset.
MISSING_TOKENS: tuple[str, ...] = ("--", "-", "", "NA", "N/A", "null", "None")


# ---------------------------------------------------------------------------
# Data locations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetPaths:
    """Where to read the two source tables from."""

    journeys: Path
    users: Path

    @classmethod
    def from_env(cls) -> "DatasetPaths":
        """Read ``CJPS_DATA_DIR`` (or the two individual variables).

        Falls back to a ``data/`` directory inside the repository so a fresh
        clone has one obvious place to drop the CSVs.
        """
        base = Path(os.environ.get("CJPS_DATA_DIR", PROJECT_ROOT / "data"))
        journeys = Path(
            os.environ.get("CJPS_JOURNEYS_CSV", base / "A_TravelDataJourneys.csv")
        )
        users = Path(os.environ.get("CJPS_USERS_CSV", base / "B_TravelDataUsers.csv"))
        return cls(journeys=journeys, users=users)

    def missing(self) -> list[Path]:
        """Paths that do not exist, so callers can report all of them at once."""
        return [p for p in (self.journeys, self.users) if not p.is_file()]

    def describe(self) -> str:
        return f"journeys={self.journeys} users={self.users}"


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    """Knobs for a training run. Every field is recorded in the manifest."""

    seed: int = SEED

    # --- resampling -----------------------------------------------------
    #: k for SMOTE/ADASYN neighbour selection. Must stay below the smallest
    #: minority class size, which :func:`effective_min_class` guarantees.
    k_neighbors: int = 5
    #: A class needs at least this many members to survive filtering *and* to
    #: populate every CV fold, so the real threshold is derived from n_splits.
    min_class_samples: int = 6

    # --- evaluation -----------------------------------------------------
    #: Inner folds for model selection.
    cv_splits: int = 5
    #: Share of rows held out and scored exactly once, never used for selection.
    holdout_fraction: float = 0.2
    #: The app ranks touchpoints, so a model's usefulness is top-weighted.
    #: F1 is the selection criterion; accuracy would reward a model that is
    #: right about frequent channels and useless for ranking.
    selection_metric: str = "f1_macro"

    # --- clustering -----------------------------------------------------
    #: k-distance neighbours used by the DBSCAN elbow heuristic.
    k_distance_k: int = 20
    #: Candidate eps values, expressed as percentiles of the sorted k-distance
    #: curve. Searching percentiles rather than raw distances keeps the
    #: heuristic scale-free, which matters because absolute distances change
    #: whenever the preprocessor changes.
    eps_percentiles: tuple[float, ...] = (70.0, 75.0, 80.0, 85.0, 90.0)
    #: Candidate cluster counts for the spectral stage.
    n_clusters_candidates: tuple[int, ...] = (2, 3, 4, 5)
    #: Below this silhouette the k-selection falls back to the midpoint
    #: candidate rather than trusting a meaningless maximum.
    silhouette_floor: float = 0.15

    # --- spectral stage -------------------------------------------------
    #: Graph used by the segmenter.
    #:
    #: ``"rbf"`` is a dense n-by-n affinity: 0.45 GiB and a 465 MB artefact at
    #: the real dataset's 7 807 rows, quadratic in both time and space.
    #: ``"nearest_neighbors"`` is a sparse k-NN graph, orders of magnitude
    #: smaller, and much faster.
    #:
    #: The default is ``"rbf"`` and that is a measured decision, not a
    #: preference. On well-separated Gaussian blobs the two agree exactly
    #: (ARI 1.000 at k=10, verified over three, four and five segments). On the
    #: pipeline's own feature space -- twenty scaled ordinal codes, which is a
    #: different neighbourhood geometry -- they do not:
    #:
    #:     k     10     15     20     30     50     80    120    200
    #:     ARI  0.69   0.67   0.66   0.65   0.61   0.59   0.50   0.31
    #:
    #: No setting reaches the agreement floor, and agreement falls as k grows.
    #: The sparse graph is a *different* segmentation, not a faster route to the
    #: same one -- and it scores a higher silhouette (+0.34 against +0.26), which
    #: is precisely why substituting it on the strength of its internal metric
    #: would be wrong. Set this to ``"nearest_neighbors"`` to trade the
    #: segmentation for a 12x smaller artefact, and enable the verification below
    #: so the change is recorded rather than assumed.
    spectral_affinity: str = "rbf"
    #: Neighbours per point when ``spectral_affinity`` is ``"nearest_neighbors"``.
    spectral_neighbors: int = 10
    #: Only used when ``spectral_affinity`` is ``"rbf"``.
    spectral_gamma: float = 1.0
    #: Fit the other graph as well and report the adjusted Rand index between
    #: the two partitions. It fits twice, so it cannot be the production path;
    #: it is a one-off check that a substitution is sound for this data.
    verify_spectral_against_dense: bool = False
    #: Below this, the run warns that the two graphs are not interchangeable
    #: here.
    spectral_agreement_floor: float = 0.90

    # --- stage 2 granularity -------------------------------------------
    #: Train one classifier per segment, or one pooled model over all of them.
    #: ``pooled`` is the default: with three segments and imbalanced touchpoints
    #: inside each, per-segment models starve the classifier.
    per_cluster_models: bool = False
    #: Skip a segment whose target distribution cannot support a fold.
    min_rows_per_cluster: int = 30

    # --- output ---------------------------------------------------------
    model_dir: Path = DEFAULT_MODEL_DIR
    reports_dir: Path = DEFAULT_REPORTS_DIR
    #: Overwrite existing artefacts instead of refusing.
    force: bool = False

    def effective_min_class(self) -> int:
        """Class-size floor that keeps stratified folds valid.

        A class with fewer members than ``cv_splits`` cannot appear in every
        training fold, and scikit-learn's ``StratifiedKFold`` does not always
        raise when that happens — it can silently produce a fold with a
        missing class, which then makes the model vanish from a benchmark. The
        original script hardcoded ``min_samples = 6`` alongside ``n_splits = 3``,
        leaving only a factor of two of headroom. Taking the max of the two
        rules states the constraint once.
        """
        return max(self.min_class_samples, self.cv_splits + 1)

    def scaled_eps_percentiles(self, n_samples: int) -> tuple[float, ...]:
        """Drop percentiles that would collapse the sample into one cluster.

        A percentile near 100 sits above the whole k-distance curve, so
        everything becomes a single cluster and the downstream metrics become
        meaningless. With small samples the upper percentiles are unreachable.
        """
        return tuple(p for p in self.eps_percentiles if p < 100.0)
