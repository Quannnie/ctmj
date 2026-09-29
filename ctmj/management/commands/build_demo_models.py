"""Build a coherent set of demo model artefacts.

The trained models live in a private Hugging Face repository, so a fresh clone
has no way to exercise the prediction path. This command synthesises a small
dataset with real cluster structure, fits the exact pipeline the production
artefacts implement, and writes the five joblib files into the local model
directory. The app then runs end to end with ``CTMJ_MODEL_SOURCE=local``.

The output is *demo* data. The numbers are plausible, not meaningful; never cite
them. Rebuild the real artefacts with ``src/main.py``.

Artefact contract (must match ctmj/services/formdata.py):
    user_data_preprocessor  ColumnTransformer over the 10 clustering columns
    dbscan                   DBSCAN fitted on the scaled clustering matrix
    spectral                 SpectralClustering(n_clusters=3)
    predicting_preprocessor  ColumnTransformer over the 8 prediction columns
    gradient_boosting        classifier with classes_ / predict_proba
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

#: RNG seed. Fixed so the demo artefacts are byte-stable across rebuilds.
SEED = 20260929

#: Number of synthetic profiles per segment.
PER_SEGMENT = 700

#: k for the k-distance elbow heuristic used to pick DBSCAN's eps.
K_DISTANCE_K = 20

#: Percentile of the k-distance distribution used as eps. The real pipeline
#: (``src/main.py::find_optimal_eps``) searches a grid; for demo data a single
#: high percentile is stable and, unlike a hard-coded constant, survives a
#: change to the preprocessor.
EPS_PERCENTILE = 82.0

CLUSTERING_COLUMNS = [
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
]

PREDICTION_COLUMNS = [
    "BAS_werkzaamheid_resp",
    "afg_kinderen_huishouden",
    "GenderID",
    "SPSS_Regio5",
    "final_label",
    "step2",
    "step3",
    "Age",
]

#: Per-segment sampling ranges: (age, household size, children, work status,
#: education, lifestage, income, social class).
SEGMENT_PROFILES = {
    0: dict(  # Empty nesters & mature singles
        age=(58, 78), household=(1, 2), children=(0, 0), work=(7, 7),
        education=(2, 5), lifestage=(2, 4), income=(2, 5), klass=(2, 4),
    ),
    1: dict(  # Mature & established families
        age=(34, 56), household=(3, 5), children=(1, 2), work=(2, 9),
        education=(3, 7), lifestage=(6, 7), income=(3, 6), klass=(2, 4),
    ),
    2: dict(  # Young singles & young couples
        age=(19, 35), household=(1, 2), children=(0, 0), work=(2, 9),
        education=(3, 8), lifestage=(1, 3), income=(2, 4), klass=(2, 5),
    ),
}

#: Touchpoint codes present in the reference data.
TOUCHPOINTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22]

#: Realistic next-step bias: a session that just searched tends to continue
#: searching or move to an app. Weights are per (segment, last channel) so the
#: demo produces differentiated rather than uniform probabilities.
CHANNEL_FAMILIES = {
    "search": [3, 6, 9, 12, 15, 16],
    "web": [1, 4, 7, 10, 13],
    "app": [2, 5, 8, 14],
    "paid": [18, 19, 21, 22],
    "direct": [20],
}


def _family_of(code: int) -> str:
    for family, codes in CHANNEL_FAMILIES.items():
        if code in codes:
            return family
    return "web"


def _synthesise(rng: np.random.Generator) -> pd.DataFrame:
    """Build a synthetic household panel with genuine segment structure."""
    frames = []
    for label, spec in SEGMENT_PROFILES.items():
        n = PER_SEGMENT
        frames.append(
            pd.DataFrame(
                {
                    "Age": rng.integers(spec["age"][0], spec["age"][1] + 1, n),
                    "BAS_huishoudgrootte": rng.integers(spec["household"][0], spec["household"][1] + 1, n),
                    "afg_kinderen_huishouden": rng.integers(spec["children"][0], spec["children"][1] + 1, n),
                    "BAS_werkzaamheid_resp": rng.integers(spec["work"][0], spec["work"][1] + 1, n),
                    "BAS_voltooide_opleiding8_resp": rng.integers(spec["education"][0], spec["education"][1] + 1, n),
                    "SPSS_Lifestage": rng.integers(spec["lifestage"][0], spec["lifestage"][1] + 1, n),
                    "BAS_bruto_jaarinkomen": rng.integers(spec["income"][0], spec["income"][1] + 1, n),
                    "AFG_sk2015": rng.integers(spec["klass"][0], spec["klass"][1] + 1, n),
                    "GenderID": rng.choice([1, 2], n, p=[0.42, 0.58]),
                    "SPSS_Regio5": rng.integers(1, 6, n),
                    "_segment": label,
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)

    # A little overlap between segments, so DBSCAN's noise gate has real work
    # to do and does not trivially separate everything.
    overlap = rng.choice(len(panel), size=int(len(panel) * 0.06), replace=False)
    panel.loc[overlap, "Age"] = rng.integers(25, 70, len(overlap))
    panel.loc[overlap, "afg_kinderen_huishouden"] = rng.integers(0, 3, len(overlap))

    # Two prior journey steps, then a target that depends on the segment and
    # on the channel just visited.
    panel["step2"] = [int(rng.choice(TOUCHPOINTS)) for _ in range(len(panel))]
    panel["step3"] = [int(rng.choice(TOUCHPOINTS)) for _ in range(len(panel))]

    def next_channel(row) -> int:
        family = _family_of(int(row["step3"]))
        if rng.random() < 0.55:
            return int(rng.choice(CHANNEL_FAMILIES[family]))
        if rng.random() < 0.5:
            return int(rng.choice(CHANNEL_FAMILIES["paid"]))
        return int(rng.choice(TOUCHPOINTS))

    panel["step1"] = [next_channel(row) for _, row in panel.iterrows()]
    return panel


class Command(BaseCommand):
    help = "Generate demo model artefacts so the prediction flow runs offline."

    def add_arguments(self, parser):
        parser.add_argument(
            "--output",
            type=Path,
            default=None,
            help="Destination directory (default: CTMJ_MODEL_DIR from settings).",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Overwrite existing artefacts.",
        )

    def handle(self, *args, **options):
        import joblib
        from sklearn.cluster import DBSCAN, SpectralClustering
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

        out_dir = Path(options["output"] or settings.CTMJ["MODEL_DIR"])
        out_dir.mkdir(parents=True, exist_ok=True)

        files = settings.CTMJ["MODEL_FILES"]
        existing = [name for name in files.values() if (out_dir / name).exists()]
        if existing and not options["force"]:
            raise CommandError(
                f"{out_dir} already contains {', '.join(existing)}. "
                "Pass --force to overwrite."
            )

        rng = np.random.default_rng(SEED)
        self.stdout.write("Synthesising demo household panel...")
        panel = _synthesise(rng)
        self.stdout.write(f"  {len(panel):,} synthetic profiles")

        # --- stage 1 preprocessor -----------------------------------------
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
        X_cluster = user_pre.fit_transform(panel[CLUSTERING_COLUMNS])

        # --- stage 1: noise gate ------------------------------------------
        # eps comes from the k-distance elbow rather than a magic number:
        # with ten scaled features, absolute distances are meaningless
        # without reference to the local density.
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=K_DISTANCE_K).fit(X_cluster)
        k_distances, _ = nbrs.kneighbors(X_cluster)
        kth = np.sort(k_distances[:, -1])
        eps = float(np.percentile(kth, EPS_PERCENTILE))
        self.stdout.write(f"  eps from k-distance p{EPS_PERCENTILE:.0f}: {eps:.4f}")

        dbscan = DBSCAN(eps=eps, min_samples=K_DISTANCE_K)
        dbscan_labels = dbscan.fit_predict(X_cluster)
        noise_share = (dbscan_labels == -1).mean()
        self.stdout.write(f"  DBSCAN noise share: {noise_share:.1%}")

        keep = dbscan_labels != -1
        if keep.sum() < 50:
            raise CommandError(
                "Too few non-noise samples to fit the spectral model; "
                "the demo eps is misconfigured for this data."
            )

        spectral = SpectralClustering(
            n_clusters=3, affinity="rbf", gamma=1.0, random_state=42
        )
        spectral_labels = spectral.fit_predict(X_cluster[keep])
        self.stdout.write("  Spectral clusters: 3")

        panel = panel[keep].copy()
        panel["final_label"] = spectral_labels

        # --- stage 2 preprocessor -----------------------------------------
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
        X_predict = predict_pre.fit_transform(panel[PREDICTION_COLUMNS])

        # --- stage 2: classifier ------------------------------------------
        y = panel["step1"].values
        # The pipeline filters classes with fewer than 6 samples before
        # training; mirror that so classes_ matches production behaviour.
        counts = pd.Series(y).value_counts()
        valid = set(counts[counts >= 6].index)
        mask = np.isin(y, list(valid))
        classifier = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )
        classifier.fit(X_predict[mask], y[mask])
        self.stdout.write(
            f"  Classifier: {len(classifier.classes_)} classes, "
            f"train accuracy {classifier.score(X_predict[mask], y[mask]):.3f}"
        )

        artefacts = {
            files["user_data_preprocessor"]: user_pre,
            files["dbscan"]: dbscan,
            files["spectral"]: spectral,
            files["predicting_preprocessor"]: predict_pre,
            files["gradient_boosting"]: classifier,
        }
        for name, obj in artefacts.items():
            joblib.dump(obj, out_dir / name)

        self.stdout.write(
            self.style.SUCCESS(f"Wrote {len(artefacts)} artefacts to {out_dir}")
        )
        self.stdout.write(
            "Set CTMJ_MODEL_SOURCE=local (or auto) and run: python manage.py runserver"
        )
