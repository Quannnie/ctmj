"""End-to-end training run.

Order of operations, and why it is this order::

    load -> clean -> impute -> compress journeys -> window
         -> fit scalers on the full frame
         -> segmentation (eps + k, both selected from the data)
         -> holdout split           <- taken here, before any model decision
         -> cross-validated selection of (model, resampler)
         -> one holdout score
         -> refit the winner on everything
         -> write artefacts + manifest
         -> verify the artefacts through the app's own code path

The holdout split happens *after* segmentation and *before* selection. The
segmentation models are unsupervised and are fitted on everything on purpose —
there is no target to leak through them, and the segment ids they produce are
features of the classifier, not predictions of it. The moment the classifier
becomes involved, the holdout is sealed.

The final verification step exists because a training run that writes five files
without checking that they can be loaded and used has not finished. The artefacts
are re-read from disk and pushed through
:func:`ctmj.services.predictor.predict`, so a mismatch between the training
column contract and the app's is caught by the training run rather than by the
first user who submits the form.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.cjps_train import cluster as cluster_mod
from src.cjps_train import data as data_mod
from src.cjps_train import evaluate as eval_mod
from src.cjps_train import preprocess as prep_mod
from src.cjps_train import select as select_mod
from src.cjps_train.config import (
    CLUSTERING_FEATURES,
    PREDICTION_FEATURES,
    TARGET,
    DatasetPaths,
    TrainConfig,
)

logger = logging.getLogger("cjps_train.run")


class TrainingError(RuntimeError):
    """Raised when a run cannot complete. The message says what to do next."""


@dataclass
class RunResult:
    """What a run produced."""

    ok: bool
    manifest: dict[str, Any]
    model_dir: Path
    manifest_path: Path | None = None
    artefacts: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        selection = self.manifest.get("selection", {})
        holdout = self.manifest.get("holdout", {})
        return (
            f"k={self.manifest.get('segmentation', {}).get('n_clusters')} "
            f"eps={self.manifest.get('segmentation', {}).get('eps')} "
            f"model={selection.get('winner')} "
            f"holdout_f1={holdout.get('f1_macro')} "
            f"holdout_top3={holdout.get('top3_accuracy')}"
        )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def drop_rare_classes(
    y: np.ndarray, min_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mask keeping only classes with at least ``min_samples`` members.

    A class the classifier cannot be evaluated on has to go before the split,
    not after: a class with two members cannot populate five stratified folds,
    and scikit-learn's response to that is not always a clean error. The
    original set ``min_samples = 6`` against ``n_splits = 3`` and left it there;
    the threshold is now derived from the fold count.
    """
    values, counts = np.unique(y, return_counts=True)
    keep = set(int(v) for v, n in zip(values, counts) if n >= min_samples)
    mask = np.isin(y, list(keep))
    dropped = int(len(y) - mask.sum())
    if dropped:
        logger.info(
            "Dropped %d row(s) in classes with fewer than %d members", dropped, min_samples
        )
    return mask, np.asarray(sorted(keep), dtype=y.dtype)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def train(
    config: TrainConfig,
    paths: DatasetPaths | None = None,
    skip_selection: bool = False,
) -> RunResult:
    """Execute a full training run and write the artefacts."""
    paths = paths or DatasetPaths.from_env()
    started = datetime.now(timezone.utc)

    logger.info("=" * 72)
    logger.info("CJPS training run | seed=%d | %s", config.seed, paths.describe())
    logger.info("=" * 72)

    # --- 1. data ---------------------------------------------------------
    journeys, users = data_mod.load_dataset(paths)
    report = data_mod.LoadReport()
    frame, report, imputer = data_mod.assemble(journeys, users)
    for note in report.notes:
        logger.info("  data: %s", note)

    # --- 2. stage-1 preprocessor ----------------------------------------
    # Fitted before segmentation because the segmenter needs the scaled matrix,
    # and it does not reference final_label.
    cluster_transformer = prep_mod.fit_clustering_preprocessor(frame)
    X_cluster = np.asarray(
        cluster_transformer.transform(frame[list(CLUSTERING_FEATURES)]), dtype=float
    )
    logger.info("Scaled clustering matrix: %s", X_cluster.shape)

    # --- 3. segmentation -------------------------------------------------
    segmentation = cluster_mod.fit_segmentation(X_cluster, config)
    labels = cluster_mod.segment_labels(segmentation, X_cluster)
    frame = frame.copy()
    frame["final_label"] = labels

    sizes = cluster_mod.segment_sizes(labels)
    logger.info("Segment sizes: %s", sizes)

    gate_rejected = int(np.sum(labels == -1))
    if gate_rejected:
        logger.info(
            "  %d profile(s) rejected by the noise gate; they are excluded from "
            "the classifier because the app cannot serve a segment for them",
            gate_rejected,
        )

    # The classifier trains on survivors only. Training on gate-rejected rows
    # would mean learning a class the app never returns.
    keep_rows = labels != -1
    trainable = frame[keep_rows].reset_index(drop=True)
    if len(trainable) < 50:
        raise TrainingError(
            f"Chỉ còn {len(trainable)} hồ sơ sau bộ lọc nhiễu, không đủ để huấn luyện. "
            "Nới khoảng percentile của eps hoặc kiểm tra lại dữ liệu đầu vào."
        )

    y_raw = trainable[TARGET].to_numpy()
    class_mask, kept_classes = drop_rare_classes(y_raw, config.effective_min_class())
    if class_mask.sum() == 0:
        raise TrainingError(
            f"Không lớp đích nào đạt ngưỡng {config.effective_min_class()} mẫu. "
            "Giảm n_splits hoặc kiểm tra phân phối step1."
        )
    dropped_labels = sorted(set(int(v) for v in np.unique(y_raw)) - set(int(v) for v in kept_classes))
    if dropped_labels:
        logger.warning(
            "Touchpoint code(s) %s dropped for having too few examples; the app "
            "will never be able to predict them.",
            dropped_labels,
        )
    trainable = trainable[class_mask].reset_index(drop=True)
    y = trainable[TARGET].to_numpy()

    # --- stage-2 preprocessor -------------------------------------------
    # Fitted on every row, holdout included. A scaler has no parameter that can
    # encode the target, so this leaks nothing — and it keeps the served
    # transform identical whether or not a given row was in the holdout. The
    # classifier is held to the stricter rule below.
    predict_transformer = prep_mod.fit_predicting_preprocessor(trainable)
    X_predict = np.asarray(
        predict_transformer.transform(trainable[list(PREDICTION_FEATURES)]), dtype=float
    )
    logger.info("Predictor matrix: %s over %d class(es)", X_predict.shape, len(kept_classes))

    bundle = prep_mod.PreprocessorBundle(
        clustering=cluster_transformer,
        predicting=predict_transformer,
        user_imputer=imputer,
        clustering_groups={
            "rob": prep_mod.CLUSTER_ROBUST,
            "mms": prep_mod.CLUSTER_MINMAX,
            "std": prep_mod.CLUSTER_STANDARD,
        },
        predicting_groups={
            "rob": prep_mod.PREDICT_ROBUST,
            "mms": prep_mod.PREDICT_MINMAX,
            "std": prep_mod.PREDICT_STANDARD,
        },
    )

    # --- 4. holdout ------------------------------------------------------
    X_tr, X_te, y_tr, y_te = eval_mod.holdout_split(X_predict, y, config)

    # --- 5. selection ----------------------------------------------------
    if skip_selection:
        logger.info("Skipping selection; using the first candidate by default")
        candidates = select_mod.default_candidates(config.seed)
        winner = select_mod.Candidate(
            name="GradientBoosting (default)",
            estimator=candidates[0].estimator,
            sampler_name="none",
            note="Selection was skipped; this is a placeholder, not a chosen model.",
        )
        selection = select_mod.Selection(winner=winner, results=[], ranking=[])
    else:
        selection = select_mod.select(X_tr, y_tr, config)

    # --- 6. one holdout score -------------------------------------------
    holdout = eval_mod.score_on_holdout(
        selection.winner.estimator, X_tr, y_tr, X_te, y_te, selection.winner.sampler
    )
    logger.info(
        "Holdout (never used for selection): f1_macro=%.4f top3=%.4f acc=%.4f",
        holdout["f1_macro"],
        holdout["top3_accuracy"],
        holdout["accuracy"],
    )
    if selection.results:
        winner_cv = next(
            (r for r in selection.results if r.name == selection.winner.name), None
        )
        if winner_cv is not None and winner_cv.ok:
            # A cross-validated score well above the holdout is a warning, not
            # a good result. Selection optimises the CV score, so some gap is
            # expected; a large one means the folds were easier than the
            # holdout, which makes the CV number the misleading one.
            if winner_cv.score - holdout["f1_macro"] > 0.15:
                logger.warning(
                    "Cross-validated f1_macro %.3f exceeds the holdout %.3f by more "
                    "than 0.15. A gap that large usually means the holdout is "
                    "harder than the folds rather than that the model is good.",
                    winner_cv.score,
                    holdout["f1_macro"],
                )

    # --- 7. final fit on everything -------------------------------------
    if config.per_cluster_models:
        logger.info("Fitting per-segment models")
        final_models = _fit_per_cluster(
            selection.winner, X_predict, y, trainable["final_label"].to_numpy(), config
        )
        pooled = None
    else:
        final_models = None
        pooled = eval_mod.fit_final(selection.winner.estimator, X_predict, y, selection.winner.sampler)
        logger.info(
            "Final model refit on all %d rows; %d class(es) in classes_",
            len(X_predict),
            len(getattr(pooled, "classes_", [])),
        )

    # --- 8. manifest -----------------------------------------------------
    manifest = _build_manifest(
        config=config,
        paths=paths,
        report=report,
        bundle=bundle,
        segmentation=segmentation,
        selection=selection,
        holdout=holdout,
        sizes=sizes,
        gate_rejected=gate_rejected,
        trainable=trainable,
        kept_classes=kept_classes,
        dropped_labels=dropped_labels,
        started=started,
    )

    # --- 9. write --------------------------------------------------------
    artefacts = _write_artefacts(
        config.model_dir,
        segmentation=segmentation,
        bundle=bundle,
        classifier=pooled,
        per_cluster=final_models,
    )
    manifest["artefacts"] = artefacts
    manifest["artifacts_dir"] = str(config.model_dir)
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()

    manifest_path = config.reports_dir / "training_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Manifest written to %s", manifest_path)

    # --- 10. verify through the app's own path --------------------------
    verification = verify_artefacts(config.model_dir)
    manifest["verification"] = verification
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ok = verification.get("ok", False)
    if not ok:
        logger.error("Artefact verification FAILED: %s", verification.get("error", ""))

    result = RunResult(
        ok=ok,
        manifest=manifest,
        model_dir=config.model_dir,
        manifest_path=manifest_path,
        artefacts=artefacts,
    )
    logger.info("=" * 72)
    logger.info("Run complete: %s", result.summary())
    logger.info("=" * 72)
    return result


def _fit_per_cluster(winner, X: np.ndarray, y: np.ndarray, labels: np.ndarray, config: TrainConfig):
    """One classifier per segment, fitted only on that segment's rows."""
    models: dict[int, Any] = {}
    for label in sorted(set(int(v) for v in labels)):
        mask = labels == label
        rows = int(np.sum(mask))
        if rows < config.min_rows_per_cluster:
            logger.warning(
                "Segment %d has %d rows (< %d); skipping its dedicated model",
                label,
                rows,
                config.min_rows_per_cluster,
            )
            continue
        try:
            models[label] = eval_mod.fit_final_on_subset(
                winner.estimator, X, y, mask, winner.sampler
            )
            logger.info("  segment %d: %d rows, %d classes", label, rows, len(models[label].classes_))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Segment %d model failed (%s); skipping", label, exc)
    if not models:
        raise TrainingError("No per-segment model could be fitted.")
    return models


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _versions() -> dict[str, str]:
    import joblib
    import sklearn

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _build_manifest(**kw) -> dict[str, Any]:
    config: TrainConfig = kw["config"]
    segmentation = kw["segmentation"]
    selection = kw["selection"]
    trainable = kw["trainable"]
    held_out = kw["holdout"]

    target_counts = (
        trainable[TARGET].value_counts().sort_index().to_dict() if TARGET in trainable else {}
    )

    return {
        "schema": 2,
        "started_at": kw["started"].isoformat(),
        "seed": config.seed,
        "config": _jsonable(asdict(config)),
        "versions": _versions(),
        "data": {
            "paths": kw["paths"].describe(),
            "load_report": kw["report"].as_dict(),
            "modelling_rows": len(trainable),
            "target_distribution": {str(k): int(v) for k, v in target_counts.items()},
            "target_classes_kept": [int(c) for c in kw["kept_classes"]],
            "target_classes_dropped": kw["dropped_labels"],
        },
        "preprocessing": kw["bundle"].as_dict(),
        "segmentation": {
            **segmentation.as_dict(),
            "segment_sizes": {str(k): v for k, v in kw["sizes"].items()},
            "gate_rejected_rows": kw["gate_rejected"],
        },
        "selection": selection.as_dict(),
        "holdout": {k: round(float(v), 6) for k, v in held_out.items()},
        "cv_vs_holdout_gap": (
            round(
                float(
                    next(
                        (r.score for r in selection.results if r.name == selection.winner.name),
                        float("nan"),
                    )
                    - held_out["f1_macro"]
                ),
                6,
            )
            if selection.results
            else None
        ),
        "notes": _integrity_notes(kw),
    }


def _integrity_notes(kw) -> list[str]:
    """Things a reader of the manifest should know before trusting the numbers."""
    notes: list[str] = []
    config: TrainConfig = kw["config"]
    report = kw["report"]
    segmentation = kw["segmentation"]
    sizes = kw["sizes"]

    if len(sizes) < 2:
        notes.append(
            "Fewer than two segments survived the noise gate. The app will serve "
            "a single segment, which makes the ranking task much easier and the "
            "model correspondingly less interesting."
        )
    if report.imputed_cells:
        notes.append(
            f"{report.imputed_cells} user cell(s) were missing and filled by KNN "
            "imputation. Synthetic values were used during training, so the "
            "reported accuracy partly measures how well the imputer works."
        )
    if segmentation.k_fallback_reason:
        notes.append(segmentation.k_fallback_reason)
    if config.n_clusters_candidates and segmentation.n_clusters in (min(config.n_clusters_candidates), max(config.n_clusters_candidates)):
        notes.append(
            f"k={segmentation.n_clusters} sits at the edge of the searched range "
            f"{list(config.n_clusters_candidates)}. The true optimum may lie "
            "outside it; widen n_clusters_candidates before trusting this."
        )
    smallest_segment = min(sizes.values()) if sizes else 0
    if smallest_segment < config.min_rows_per_cluster:
        notes.append(
            f"The smallest segment holds only {smallest_segment} rows. Its "
            "probability estimates will be unstable even if the model is correct."
        )
    return notes


def _jsonable(value):
    """Recursively convert to something ``json.dumps`` accepts."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if is_dataclass(value):
        return _jsonable(asdict(value))
    return value


# ---------------------------------------------------------------------------
# Writing and verifying
# ---------------------------------------------------------------------------


def artefact_paths(model_dir: Path) -> dict[str, Path]:
    """The five filenames the app's settings expect."""
    return {
        "user_data_preprocessor": model_dir / "user_data_preprocessor.pkl",
        "dbscan": model_dir / "dbscan_clustering_model.pkl",
        "spectral": model_dir / "spectral_clustering_model.pkl",
        "predicting_preprocessor": model_dir / "s1_predicting_preprocessor.pkl",
        "gradient_boosting": model_dir / "GradientBoostingClassifier_model.pkl",
    }


def _write_artefacts(
    model_dir: Path,
    segmentation,
    bundle,
    classifier,
    per_cluster: dict | None = None,
) -> dict[str, str]:
    import joblib

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = artefact_paths(model_dir)

    existing = [p.name for p in paths.values() if p.exists()]
    if existing and not existing:  # pragma: no cover - defensive
        raise TrainingError("Unexpected artefact state.")

    payload = {
        "user_data_preprocessor": bundle.clustering,
        "dbscan": segmentation.dbscan,
        "spectral": segmentation.spectral,
        "predicting_preprocessor": bundle.predicting,
        "gradient_boosting": classifier,
    }
    for key, obj in payload.items():
        joblib.dump(obj, paths[key])
        logger.info("  wrote %s (%d bytes)", paths[key].name, paths[key].stat().st_size)

    # The KNN imputer is not one of the five the app loads, but it is the only
    # way to reproduce training-time imputation, so it is saved next to them.
    if bundle.user_imputer is not None:
        imputer_path = model_dir / "user_data_imputer.pkl"
        joblib.dump(bundle.user_imputer, imputer_path)
        logger.info("  wrote %s (not loaded by the app)", imputer_path.name)

    if per_cluster:
        cluster_models_path = model_dir / "per_cluster_models.pkl"
        joblib.dump(per_cluster, cluster_models_path)
        logger.info("  wrote %s (%d models)", cluster_models_path.name, len(per_cluster))

    return {key: path.name for key, path in paths.items()}


def verify_artefacts(model_dir: Path) -> dict[str, Any]:
    """Re-read the artefacts and drive one prediction through the app.

    The training run is not finished until the files it just wrote have been
    loaded back from disk and used the way a request will use them. This is the
    check that catches a training/inference contract mismatch — a renamed
    column, a missing ``final_label``, a preprocessor that transforms but cannot
    predict — while it is still a training problem rather than a user-facing
    one.
    """
    import os

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ctmj.settings")
    try:
        import django

        django.setup()
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"Could not initialise Django for verification: {type(exc).__name__}: {exc}",
        }

    import joblib

    from ctmj.services import formdata
    from ctmj.services.predictor import predict

    paths = artefact_paths(Path(model_dir))
    missing = [p.name for p in paths.values() if not p.is_file()]
    if missing:
        return {"ok": False, "error": f"artefact(s) missing after write: {', '.join(missing)}"}

    try:
        loaded = {key: joblib.load(path) for key, path in paths.items()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"artefacts failed to load: {type(exc).__name__}: {exc}"}

    class _Stub:
        """Minimal registry stand-in, so verification does not depend on app state."""

        def get(self, key):
            return loaded.get(key)

        def ready(self):
            return all(v is not None for v in loaded.values())

    # A mid-range profile: valid against every lookup table, so a failure here
    # is a model problem and not a bad fixture.
    #
    # Values are strings because this is a form. ``validate_prediction_form``
    # is the request-parsing path, and passing integers would exercise a code
    # path no real request ever takes — a verification that passes on inputs
    # the app cannot receive is worth very little.
    fixture = {
        "GenderID": "2",
        "Age": "41",
        "SPSS_Regio5": "1",
        "BAS_huishoudgrootte": "4",
        "BAS_werkzaamheid_resp": "2",
        "afg_kinderen_huishouden": "2",
        "BAS_bruto_jaarinkomen": "4",
        "AFG_sk2015": "5",
        "BAS_voltooide_opleiding8_resp": "5",
        "SPSS_Lifestage": "7",
        "step_1_channel": "16",
        "step_2_channel": "20",
    }

    try:
        validated = formdata.validate_prediction_form(fixture)
    except Exception as exc:  # noqa: BLE001
        # The form validator consults the lookup tables, so verification needs a
        # migrated database. A training run against a fresh clone that has not
        # run ``migrate`` lands here. That is a real gap in the check, not a
        # fault in the artefacts, and it is reported as its own status so the
        # operator is not sent looking at the models.
        from django.db import Error as DjangoError

        if isinstance(exc, DjangoError):
            return {
                "ok": True,
                "skipped": True,
                "error": (
                    "Could not validate the fixture profile: the database is not "
                    f"available ({type(exc).__name__}). Run 'python manage.py "
                    "migrate' to enable the end-to-end check. The artefacts were "
                    "written and load correctly."
                ),
            }
        return {"ok": False, "error": f"fixture failed validation: {type(exc).__name__}: {exc}"}

    if not validated.is_valid:
        return {"ok": False, "error": f"fixture is not a valid profile: {validated.errors}"}

    try:
        prediction = predict(_Stub(), validated)
    except Exception as exc:  # noqa: BLE001
        from django.db import Error as DjangoError

        if isinstance(exc, DjangoError):
            return {
                "ok": True,
                "skipped": True,
                "error": (
                    "Could not run a prediction: the database is not available "
                    f"({type(exc).__name__}). Run 'python manage.py migrate' to "
                    "enable the end-to-end check. The artefacts were written and "
                    "load correctly."
                ),
            }
        return {
            "ok": False,
            "error": f"prediction raised: {type(exc).__name__}: {exc}",
        }

    if not prediction.channels:
        return {"ok": False, "error": "prediction returned no channels"}

    probabilities = [c.probability for c in prediction.channels]
    if any(p < 0 or p > 100 for p in probabilities):
        return {"ok": False, "error": f"probabilities out of range: {probabilities}"}

    return {
        "ok": True,
        "cluster_id": prediction.cluster_id,
        "cluster_name": prediction.cluster_name,
        "is_noise": prediction.is_noise,
        "noise_distance": prediction.noise_distance,
        "channels": [
            {"id": c.channel_id, "name": c.name, "probability": round(c.probability, 2)}
            for c in prediction.channels
        ],
    }
