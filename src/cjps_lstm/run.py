"""End-to-end LSTM pipeline run.

Order of operations::

    load -> clean users -> compress journeys -> per-user examples
         -> merge demographics -> drop rare targets
         -> split (stratified, one row per user)
         -> impute demographics (fit on train) -> segmentation -> final_label
         -> encode static features (fit on train) -> scale event channels
         -> pad sequences (max_len = train P95)
         -> baselines: majority, GBM (app contract), GBM + engineered
         -> LSTM touchpoints-only -> LSTM fused
         -> random-search tuning on validation
         -> final fit on train+val -> one test evaluation
         -> permutation importance + error analysis on test
         -> write artefacts, figures, metrics, report

The test partition is touched exactly once, by the final model. Tuning and
early stopping see the validation partition only; the baselines are
selected on validation as well so the comparison is fair.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.cjps_lstm import data as data_mod
from src.cjps_lstm.config import LSTMConfig, SEARCH_SPACE, data_paths
from src.cjps_lstm.model import JourneyDataset
from src.cjps_lstm import train as train_mod

logger = logging.getLogger("cjps_lstm.run")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RunArtifacts:
    """Paths and numbers a run produced, for the CLI summary."""

    output_dir: Path
    metrics: dict = field(default_factory=dict)
    comparison: pd.DataFrame | None = None
    importance: pd.DataFrame | None = None
    tuning: pd.DataFrame | None = None
    history: object | None = None


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def prepare(config: LSTMConfig, data_dir: Path | None = None):
    """Everything up to tensors. Returns a dict of prepared objects."""
    journeys_path, users_path = data_paths(data_dir)

    journeys = data_mod.read_csv(
        journeys_path, required=[data_mod.USER_ID, data_mod.TIME, data_mod.TOUCH, data_mod.DURATION]
    )
    users = data_mod.read_csv(users_path, required=[data_mod.USER_ID])
    n_users_raw = len(users)
    users_clean = data_mod.clean_users(users)

    events = data_mod.build_events(journeys)
    examples = data_mod.build_examples(events, config.min_journey_length)

    # --- data fusion ----------------------------------------------------
    # Inner join on UserID: journey users without a demographic profile are
    # dropped (they carry no static signal); duplicated UserIDs were already
    # collapsed inside clean_users, so the join cannot fan out.
    n_before = len(examples)
    frame = examples.merge(users_clean, on=data_mod.USER_ID, how="inner")
    logger.info(
        "Fusion: %d example(s) joined to %d profiles -> %d rows (%d dropped: no demographics)",
        n_before, len(users_clean), len(frame), n_before - len(frame),
    )

    frame = data_mod.drop_rare_classes(frame, config.min_class_samples)
    split = data_mod.split_examples(frame, config)

    # --- imputation, fitted on train only --------------------------------
    from sklearn.impute import KNNImputer

    demo_cols = [c for c in data_mod.DEMOGRAPHIC_FEATURES if c in frame.columns]
    imputer = KNNImputer(n_neighbors=5)
    frame[demo_cols] = frame[demo_cols].apply(pd.to_numeric, errors="coerce")
    imputer.fit(frame.iloc[split.train][demo_cols])
    frame[demo_cols] = imputer.transform(frame[demo_cols])
    logger.info("KNN imputer fit on %d train profiles, applied to all %d", len(split.train), len(frame))

    # --- segmentation feature -------------------------------------------
    frame["final_label"] = -1
    segmentation = None
    if config.use_segments:
        segmentation = _fit_segmentation(frame, config)
        if segmentation is not None:
            frame["final_label"] = segmentation

    return {
        "frame": frame,
        "split": split,
        "events": events,
        "n_users_raw": n_users_raw,
        "n_users_clean": len(users_clean),
        "imputer": imputer,
    }


def _fit_segmentation(frame: pd.DataFrame, config: LSTMConfig) -> np.ndarray | None:
    """Reuse the stage-1 segmentation to produce ``final_label`` per user.

    Unsupervised, so fitting on all profiles leaks nothing about the
    target — the same argument the production pipeline documents. The
    noise-gate label -1 is kept as a real category rather than dropping the
    users, because for the LSTM "unusual profile" is itself a signal.
    """
    try:
        from src.cjps_train import cluster as cluster_mod
        from src.cjps_train.config import CLUSTERING_FEATURES, TrainConfig
        from src.cjps_train import preprocess as prep_mod
    except ImportError:  # pragma: no cover
        logger.warning("cjps_train not importable; skipping segmentation")
        return None

    cols = [c for c in CLUSTERING_FEATURES if c in frame.columns]
    if len(cols) != len(CLUSTERING_FEATURES):
        logger.warning("Missing clustering columns %s; skipping segmentation",
                       sorted(set(CLUSTERING_FEATURES) - set(cols)))
        return None

    started = time.perf_counter()
    transformer = prep_mod.build_clustering_preprocessor()
    X = np.asarray(transformer.fit_transform(frame[list(CLUSTERING_FEATURES)]), dtype=float)
    seg = cluster_mod.fit_segmentation(X, TrainConfig(seed=config.seed))
    labels = cluster_mod.segment_labels(seg, X)
    logger.info(
        "Segmentation done in %.1fs: eps=%.4f k=%d sizes=%s",
        time.perf_counter() - started, seg.eps, seg.n_clusters,
        cluster_mod.segment_sizes(labels),
    )
    return labels


# ---------------------------------------------------------------------------
# Tensors
# ---------------------------------------------------------------------------


def build_tensors(frame: pd.DataFrame, split: data_mod.Split, config: LSTMConfig):
    """Encode + pad all partitions. Returns the datasets and helpers."""
    vocab = data_mod.TouchpointVocab()

    ids_l, num_l = [], []
    for _, row in frame.iterrows():
        ids, num = data_mod.encode_example(row, vocab)
        ids_l.append(ids)
        num_l.append(num)

    # max_len from the TRAIN input-length distribution — P95 covers 95% of
    # journeys fully while capping the 1 622-event outlier that would triple
    # every batch's compute for the sake of a handful of users.
    if config.max_len is not None:
        max_len = config.max_len
    else:
        max_len = int(np.percentile(frame.iloc[split.train]["seq_len"], config.max_len_percentile))
        max_len = max(max_len, config.min_journey_length - 1)
    logger.info("max_len=%d (train P%d)", max_len, int(config.max_len_percentile))

    # Event-channel scaler fitted on train events only.
    mean, std = data_mod.fit_event_scaler([num_l[i] for i in split.train])
    num_l = [data_mod.apply_event_scaler(n, mean, std) for n in num_l]

    ids_pad, num_pad, lens = data_mod.pad_batch(ids_l, num_l, max_len)

    # Feature selection on the aggregate block, decided on train rows only.
    kept_aggregates = data_mod.select_aggregate_features(frame.iloc[split.train])

    # Static encoding, fit on train rows only.
    encoder = data_mod.StaticEncoder(extra_nominal=("final_label",),
                                     aggregate_features=tuple(kept_aggregates))
    train_static = encoder.fit_transform(frame.iloc[split.train])
    static_full = np.zeros((len(frame), train_static.shape[1]), dtype=np.float32)
    static_full[split.train] = train_static
    for idx in (split.val, split.test):
        static_full[idx] = encoder.transform(frame.iloc[idx])

    classes = np.sort(frame["target"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = frame["target"].map(class_to_idx).to_numpy()

    def make_ds(idx) -> JourneyDataset:
        return JourneyDataset(ids_pad[idx], num_pad[idx], lens[idx], static_full[idx], y[idx])

    return {
        "vocab": vocab,
        "encoder": encoder,
        "max_len": max_len,
        "classes": classes,
        "y": y,
        "ids": ids_pad,
        "num": num_pad,
        "lens": lens,
        "static": static_full,
        "train": make_ds(split.train),
        "val": make_ds(split.val),
        "test": make_ds(split.test),
    }


def make_seq_only_dataset(ds: JourneyDataset, n_classes: int) -> JourneyDataset:
    """The same partition with event numerics and static features blanked.

    This is the ablation baseline: it isolates what the touchpoint sequence
    alone contributes, without retraining a differently-shaped network.
    """
    return JourneyDataset(
        ds.seq_ids.numpy(),
        np.zeros_like(ds.seq_num.numpy()),
        ds.lengths.numpy(),
        np.zeros((len(ds), 0), dtype=np.float32),
        ds.labels.numpy(),
    )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def _last_two_touchpoints(frame: pd.DataFrame) -> pd.DataFrame:
    """step2/step3 = the two most recent INPUT touchpoints.

    The input sequence is chronological, so the last element of ``seq_touch``
    is step2 and the second-to-last is step3 — matching the app's contract.
    """
    out = pd.DataFrame(index=frame.index)
    out["step2"] = frame["seq_touch"].apply(lambda s: s[-1] if len(s) else 0)
    out["step3"] = frame["seq_touch"].apply(lambda s: s[-2] if len(s) > 1 else 0)
    return out


def run_gbm_baseline(frame, split, tensors, config, feature_set: str):
    """GradientBoosting baseline selected on validation, scored on test.

    ``feature_set='original'`` reproduces the app's contract: step2, step3,
    the demographic predictors and final_label, scaled exactly as
    ``cjps_train.preprocess`` scales them. ``'engineered'`` uses the full
    fused static vector plus step2/step3, to isolate what feature
    engineering buys the strongest tabular model.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    steps = _last_two_touchpoints(frame)
    work = frame.copy()
    work["step2"], work["step3"] = steps["step2"], steps["step3"]

    if feature_set == "original":
        cols = ["BAS_werkzaamheid_resp", "afg_kinderen_huishouden", "GenderID",
                "SPSS_Regio5", "final_label", "step2", "step3", "Age"]
        pre = ColumnTransformer(
            transformers=[
                ("imp", SimpleImputer(strategy="median"), cols),
                ("rob", RobustScaler(), ["BAS_werkzaamheid_resp", "afg_kinderen_huishouden"]),
                ("mms", MinMaxScaler(), ["GenderID", "SPSS_Regio5", "final_label", "step2", "step3"]),
                ("std", StandardScaler(), ["Age"]),
            ], remainder="drop", sparse_threshold=0.0)
        # ColumnTransformer concatenates every branch's output: 8 imputed
        # passthrough columns + 8 scaled = 16, the same matrix shape the
        # production preprocessor produces.
        train_X = np.asarray(pre.fit_transform(work.iloc[split.train][cols]), dtype=float)
        X = np.zeros((len(work), train_X.shape[1]), dtype=float)
        X[split.train] = train_X
        for idx in (split.val, split.test):
            X[idx] = np.asarray(pre.transform(work.iloc[idx][cols]), dtype=float)
    else:
        X = np.hstack([tensors["static"], steps.to_numpy(dtype=float)])

    y = tensors["y"]
    best, best_f1 = None, -1.0
    for params in ({"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
                   {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5},
                   {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 5}):
        m = GradientBoostingClassifier(random_state=config.seed, **params)
        m.fit(X[split.train], y[split.train])
        f1 = float(f1_score(y[split.val], m.predict(X[split.val]),
                            average="macro", zero_division=0))
        if f1 > best_f1:
            best, best_f1 = m, f1
    logger.info("GBM %s: best val f1_macro=%.4f params=%s", feature_set, best_f1, best.get_params())

    proba = best.predict_proba(X[split.test])
    pred = best.predict(X[split.test])
    return best, _tabular_metrics(y[split.test], pred, proba, best.classes_)


def _tabular_metrics(y_true, y_pred, y_proba, classes) -> dict:
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 f1_score, precision_score, recall_score,
                                 roc_auc_score)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc_macro": float("nan"), "pr_auc_macro": float("nan"), "top3_accuracy": float("nan"),
    }
    if y_proba is not None:
        from sklearn.metrics import top_k_accuracy_score
        out["top3_accuracy"] = float(top_k_accuracy_score(
            y_true, y_proba, k=min(3, y_proba.shape[1]), labels=np.arange(len(classes))))
        present = np.unique(y_true)
        keep = [i for i, c in enumerate(classes) if c in present]
        y_bin = np.zeros((len(y_true), len(classes)))
        for i, c in enumerate(classes):
            y_bin[:, i] = (y_true == c)
        try:
            out["roc_auc_macro"] = float(roc_auc_score(y_bin[:, keep], y_proba[:, keep], average="macro"))
            out["pr_auc_macro"] = float(np.mean([average_precision_score(y_bin[:, i], y_proba[:, i]) for i in keep]))
        except ValueError:
            pass
    return out


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------


def tune(frame, split, tensors, config) -> tuple[dict, pd.DataFrame]:
    """Random search over SEARCH_SPACE, scored on validation macro-F1.

    Every trial trains on the train partition and is early-stopped on the
    validation partition. The test partition is not present in any trial —
    this function never receives a test tensor.
    """
    import dataclasses

    rng = np.random.RandomState(config.seed)
    rows, best_params, best_f1 = [], None, -1.0

    def _py(v):
        # np.choice returns numpy scalars; dataclasses.replace and torch
        # want plain Python types.
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        return v

    for trial in range(config.n_tune_trials):
        params = {k: _py(rng.choice(list(v))) for k, v in SEARCH_SPACE.items()}

        trial_cfg = dataclasses.replace(config, max_epochs=config.tune_max_epochs,
                                        patience=5, **params)
        weights = train_mod.balanced_class_weights(
            tensors["train"].labels.numpy(), len(tensors["classes"])
        ) if trial_cfg.class_weighted else None

        model = train_mod.build_model(
            trial_cfg,
            n_event_features=tensors["train"].seq_num.shape[2],
            n_static=tensors["train"].static.shape[1],
            n_classes=len(tensors["classes"]),
            vocab_size=tensors["vocab"].size,
        )
        t0 = time.perf_counter()
        hist = train_mod.train_model(model, tensors["train"], tensors["val"], trial_cfg, weights)
        metrics = train_mod.evaluate(
            model, tensors["val"], np.arange(len(tensors["classes"])), trial_cfg.batch_size
        )
        rows.append({"trial": trial, **params,
                     "val_f1_macro": metrics["f1_macro"],
                     "val_loss": float(min(hist.val_loss)),
                     "epochs": len(hist.train_loss),
                     "seconds": round(time.perf_counter() - t0, 1)})
        logger.info("trial %2d  f1=%.4f  %s", trial, metrics["f1_macro"], params)
        if metrics["f1_macro"] > best_f1:
            best_f1, best_params = metrics["f1_macro"], dict(params)

    table = pd.DataFrame(rows)
    logger.info("Tuning done. Best val f1_macro=%.4f params=%s", best_f1, best_params)
    return best_params, table


# ---------------------------------------------------------------------------
# Feature importance wiring
# ---------------------------------------------------------------------------


def importance_blocks(encoder: data_mod.StaticEncoder) -> dict:
    """Named blocks for permutation importance.

    Three granularities at once: the whole sequence channel (does journey
    order matter), each per-event numeric channel (does cadence/duration/
    device matter), and each individual static feature.
    """
    blocks: dict[str, tuple[str, int | slice]] = {
        "SEQ:touchpoints": ("seq_ids", 0),
    }
    for i, name in enumerate(data_mod.EVENT_NUMERIC):
        blocks[f"SEQ_NUM:{name}"] = ("seq_num", i)
    for i, name in enumerate(encoder.feature_names_):
        blocks[f"STATIC:{name}"] = ("static", i)
    return blocks


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------


def error_analysis(frame, split, eval_out, classes) -> dict:
    """Break test errors down by the dimensions the brief asks about.

    Returns per-group accuracy tables. Only features that actually exist in
    the data are used; each table is written to CSV by the caller.
    """
    test_frame = frame.iloc[split.test].copy()
    test_frame["pred_idx"] = eval_out["y_pred"]
    test_frame["correct"] = eval_out["y_pred"] == eval_out["y_true"]

    tables = {}

    # Journey length — binned on the observed distribution.
    test_frame["len_bucket"] = pd.cut(
        test_frame["seq_len"],
        bins=[0, 5, 15, 40, 100, np.inf],
        labels=["2-5", "6-15", "16-40", "41-100", "100+"],
    )
    tables["by_journey_length"] = (
        test_frame.groupby("len_bucket", observed=True)
        .agg(n=("correct", "size"), accuracy=("correct", "mean"))
        .reset_index()
    )

    # Gender — only when the column survived preprocessing.
    if "GenderID" in test_frame.columns:
        tables["by_gender"] = (
            test_frame.groupby("GenderID")
            .agg(n=("correct", "size"), accuracy=("correct", "mean"))
            .reset_index()
        )

    # Age bands and customer segment.
    if "Age" in test_frame.columns:
        test_frame["age_band"] = pd.cut(
            test_frame["Age"], bins=[0, 30, 45, 60, 120], labels=["<30", "30-45", "45-60", "60+"]
        )
        tables["by_age"] = (
            test_frame.groupby("age_band", observed=True)
            .agg(n=("correct", "size"), accuracy=("correct", "mean"))
            .reset_index()
        )
    if "final_label" in test_frame.columns:
        tables["by_segment"] = (
            test_frame.groupby("final_label")
            .agg(n=("correct", "size"), accuracy=("correct", "mean"))
            .reset_index()
        )
    if "SPSS_Lifestage" in test_frame.columns:
        tables["by_lifestage"] = (
            test_frame.groupby("SPSS_Lifestage")
            .agg(n=("correct", "size"), accuracy=("correct", "mean"))
            .reset_index()
        )
    if "n_sessions" in test_frame.columns:
        test_frame["interaction_band"] = pd.cut(
            test_frame["n_sessions"], bins=[0, 1, 3, 8, np.inf],
            labels=["1", "2-3", "4-8", "9+"]
        )
        tables["by_sessions"] = (
            test_frame.groupby("interaction_band", observed=True)
            .agg(n=("correct", "size"), accuracy=("correct", "mean"))
            .reset_index()
        )

    # Per-class precision/recall — which touchpoints the model keeps missing.
    from sklearn.metrics import classification_report
    target_names = [str(c) for c in classes]
    report = classification_report(
        eval_out["y_true"], eval_out["y_pred"],
        labels=list(range(len(classes))), target_names=target_names,
        output_dict=True, zero_division=0,
    )
    tables["per_class"] = pd.DataFrame(report).T.reset_index().rename(columns={"index": "class"})

    # Compare journeys the model got right vs wrong on sequence stats.
    err = test_frame.groupby("correct")[["seq_len", "n_unique_touch", "gap_mean_h"]].mean()
    tables["error_profile"] = err.reset_index()

    # Where do the confusions concentrate?
    cm = eval_out["confusion_matrix"]
    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": int(classes[i]), "pred": int(classes[j]),
                              "count": int(cm[i, j])})
    tables["top_confusions"] = (
        pd.DataFrame(pairs).sort_values("count", ascending=False).head(20)
        if pairs else pd.DataFrame(columns=["true", "pred", "count"])
    )
    return tables


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _save_figures(outdir: Path, history, eval_out, classes, importance, comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # Learning curves.
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(history.train_loss, label="train")
    ax[0].plot(history.val_loss, label="validation")
    ax[0].axvline(history.best_epoch, ls="--", c="grey", alpha=0.7,
                  label=f"best epoch {history.best_epoch}")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history.train_f1, label="train")
    ax[1].plot(history.val_f1, label="validation")
    ax[1].axvline(history.best_epoch, ls="--", c="grey", alpha=0.7)
    ax[1].set_title("Macro-F1"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.suptitle("Final LSTM — learning curves")
    fig.tight_layout()
    fig.savefig(figdir / "learning_curves.png", dpi=130)
    plt.close(fig)

    # Confusion matrix.
    cm = eval_out["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_xticks(range(len(classes)), classes, rotation=90)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted touchpoint"); ax.set_ylabel("true touchpoint")
    ax.set_title("Test-set confusion matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(figdir / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    # Feature importance (top 20).
    top = importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#4c9a2a")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("macro-F1 drop when permuted (test)")
    ax.set_title("Permutation importance — top 20")
    fig.tight_layout()
    fig.savefig(figdir / "feature_importance.png", dpi=130)
    plt.close(fig)

    # Model comparison.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(comparison))
    ax.bar(x - 0.2, comparison["precision_macro"], 0.2, label="precision")
    ax.bar(x, comparison["recall_macro"], 0.2, label="recall")
    ax.bar(x + 0.2, comparison["f1_macro"], 0.2, label="f1")
    ax.set_xticks(x, comparison["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1); ax.legend(); ax.set_title("Model comparison (test set)")
    fig.tight_layout()
    fig.savefig(figdir / "model_comparison.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run(config: LSTMConfig, data_dir: Path | None = None) -> RunArtifacts:
    import joblib
    import torch

    outdir = Path(config.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    prep = prepare(config, data_dir)
    frame, split = prep["frame"], prep["split"]
    tensors = build_tensors(frame, split, config)
    classes = tensors["classes"]
    n_classes = len(classes)
    n_event_feat = tensors["train"].seq_num.shape[2]
    n_static = tensors["train"].static.shape[1]
    logger.info("Tensors: %d classes | %d event feats | %d static feats | max_len=%d",
                n_classes, n_event_feat, n_static, tensors["max_len"])

    weights = train_mod.balanced_class_weights(
        tensors["train"].labels.numpy(), n_classes
    ) if config.class_weighted else None

    # --- baselines -------------------------------------------------------
    comparison_rows = []

    majority = int(np.bincount(tensors["y"][split.train]).argmax())
    maj_pred = np.full(len(split.test), majority)
    comparison_rows.append({"model": "Majority class", "features": "—",
                            **_tabular_metrics(tensors["y"][split.test], maj_pred, None, classes)})

    for fs, label in (("original", "GBM — app contract"), ("engineered", "GBM — engineered features")):
        _, m = run_gbm_baseline(frame, split, tensors, config, fs)
        comparison_rows.append({"model": label,
                                "features": "step2+step3+demo+segment" if fs == "original"
                                            else "static vector + step2/step3",
                                **m})

    # --- LSTM: sequence-only ablation ------------------------------------
    ds_train_seq = make_seq_only_dataset(tensors["train"], n_classes)
    ds_val_seq = make_seq_only_dataset(tensors["val"], n_classes)
    ds_test_seq = make_seq_only_dataset(tensors["test"], n_classes)
    m_seq = train_mod.build_model(config, n_event_feat, 0, n_classes, tensors["vocab"].size)
    train_mod.train_model(m_seq, ds_train_seq, ds_val_seq, config, weights)
    ev = train_mod.evaluate(m_seq, ds_test_seq, np.arange(n_classes), config.batch_size)
    comparison_rows.append({"model": "LSTM — touchpoints only", "features": "sequence",
                            **{k: v for k, v in ev.items()
                               if k not in ("confusion_matrix", "y_true", "y_pred", "y_proba")}})

    # --- LSTM: full fused input ------------------------------------------
    m_full = train_mod.build_model(config, n_event_feat, n_static, n_classes,
                                   tensors["vocab"].size)
    train_mod.train_model(m_full, tensors["train"], tensors["val"], config, weights)
    ev = train_mod.evaluate(m_full, tensors["test"], np.arange(n_classes), config.batch_size)
    comparison_rows.append({"model": "LSTM — fused", "features": "sequence + static",
                            **{k: v for k, v in ev.items()
                               if k not in ("confusion_matrix", "y_true", "y_pred", "y_proba")}})

    # --- tuning -----------------------------------------------------------
    best_params, tuning_table = tune(frame, split, tensors, config)
    tuning_table.to_csv(outdir / "tuning_results.csv", index=False)

    # --- final model: best params, refit on train+val --------------------
    final = final_stage(frame, split, tensors, config, best_params,
                        seq_datasets=(ds_train_seq, ds_val_seq, ds_test_seq),
                        outdir=outdir)
    eval_test = final["eval_test"]
    final_model = final["model"]
    history = final["history"]
    best_cfg = final["best_cfg"]
    comparison_rows.extend(final["comparison_rows"])

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(outdir / "model_comparison.csv", index=False)
    logger.info("\n%s", comparison.to_string(index=False))

    # --- importance (post-hoc analysis on test; no selection reads it) ---
    blocks = importance_blocks(tensors["encoder"])
    if not final["use_static"]:
        # The winning variant carries no static vector — keep only the
        # sequence blocks, where "importance" is still meaningful.
        blocks = {k: v for k, v in blocks.items() if not k.startswith("STATIC:")}
    imp = train_mod.permutation_importance(
        final_model, final["test_ds"], blocks,
        best_cfg.batch_size, config.seed, config.importance_repeats,
    )
    imp.to_csv(outdir / "feature_importance.csv", index=False)

    # --- error analysis ----------------------------------------------------
    tables = error_analysis(frame, split, eval_test, classes)
    for name, tbl in tables.items():
        tbl.to_csv(outdir / f"error_{name}.csv", index=False)

    # --- artefacts ---------------------------------------------------------
    torch.save({"state_dict": final_model.state_dict(),
                "config": asdict(best_cfg),
                "use_static": final["use_static"],
                "classes": classes.tolist(),
                "max_len": tensors["max_len"]},
               outdir / "best_lstm.pt")
    joblib.dump({"encoder": tensors["encoder"], "vocab": tensors["vocab"],
                 "imputer": prep["imputer"]}, outdir / "preprocessors.pkl")

    _save_figures(outdir, history, eval_test, classes, imp, comparison)

    metrics = {
        "test": {k: v for k, v in eval_test.items()
                 if k not in ("confusion_matrix", "y_true", "y_pred", "y_proba")},
        "best_params": best_params,
        "input_variant": final["variant_label"],
        "val_variant_f1": {str(k): v for k, v in final["val_variant_f1"].items()},
        "max_len": int(tensors["max_len"]),
        "n_classes": n_classes,
        "classes": classes.tolist(),
        "split": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        "best_epoch": int(history.best_epoch),
        "epochs_run": len(history.train_loss),
    }
    (outdir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    return RunArtifacts(output_dir=outdir, metrics=metrics, comparison=comparison,
                        importance=imp, tuning=tuning_table, history=history)


def _variant_datasets(tensors, idx, seq_only: bool):
    """JourneyDataset for an arbitrary row subset of one input variant."""
    from src.cjps_lstm.model import JourneyDataset
    n = len(idx)
    static = (np.zeros((n, 0), dtype=np.float32) if seq_only
              else tensors["static"][idx])
    num = (np.zeros_like(tensors["num"][idx]) if seq_only
           else tensors["num"][idx])
    return JourneyDataset(tensors["ids"][idx], num, tensors["lens"][idx],
                          static, tensors["y"][idx])


def final_stage(frame, split, tensors, config, best_params, seq_datasets, outdir=None):
    """Model selection by input variant on validation, then one test eval.

    The earlier ablation showed the static block can hurt the LSTM, so which
    input the final model gets is itself a hyperparameter — decided on the
    validation partition like every other selection, never on test.

    Returns the fitted final model, its history, the test evaluation, and
    enough metadata for artefacts and the report.
    """
    import dataclasses

    import numpy as np
    from sklearn.model_selection import train_test_split

    from src.cjps_lstm import train as train_mod

    best_cfg = dataclasses.replace(config, **best_params)
    classes = tensors["classes"]
    n_classes = len(classes)
    n_event_feat = tensors["train"].seq_num.shape[2]

    weights = train_mod.balanced_class_weights(
        tensors["train"].labels.numpy(), n_classes
    ) if config.class_weighted else None

    # --- variant selection on validation -----------------------------------
    candidates = {}
    for seq_only, label in ((False, "sequence + static"), (True, "sequence only")):
        tr = _variant_datasets(tensors, split.train, seq_only)
        vl = _variant_datasets(tensors, split.val, seq_only)
        m = train_mod.build_model(best_cfg, n_event_feat, tr.static.shape[1],
                                  n_classes, tensors["vocab"].size)
        train_mod.train_model(m, tr, vl, best_cfg, weights)
        ev = train_mod.evaluate(m, vl, np.arange(n_classes), best_cfg.batch_size)
        candidates[seq_only] = ev["f1_macro"]
        logger.info("variant %-18s val f1_macro=%.4f", label, ev["f1_macro"])
    use_seq_only = candidates[True] > candidates[False]
    variant_label = "sequence only" if use_seq_only else "sequence + static"
    logger.info("Selected input variant: %s", variant_label)

    # --- refit the winner on train+val ------------------------------------
    # Early stopping still needs a held-out slice; a 12% stratified inner
    # split of the pool supplies it. The test partition stays sealed.
    pool = np.concatenate([split.train, split.val])
    y_pool = tensors["y"][pool]
    fit_idx, es_idx = train_test_split(pool, test_size=0.12,
                                       random_state=config.seed, stratify=y_pool)
    ds_fit = _variant_datasets(tensors, fit_idx, use_seq_only)
    ds_es = _variant_datasets(tensors, es_idx, use_seq_only)
    test_ds = _variant_datasets(tensors, split.test, use_seq_only)

    model = train_mod.build_model(best_cfg, n_event_feat,
                                  ds_fit.static.shape[1], n_classes,
                                  tensors["vocab"].size)
    final_weights = train_mod.balanced_class_weights(
        tensors["y"][fit_idx], n_classes) if config.class_weighted else None
    history = train_mod.train_model(model, ds_fit, ds_es, best_cfg, final_weights)

    eval_test = train_mod.evaluate(model, test_ds, np.arange(n_classes),
                                   best_cfg.batch_size)

    rows = [{
        "model": f"LSTM — tuned ({variant_label})",
        "features": variant_label + ", tuned",
        **{k: v for k, v in eval_test.items()
           if k not in ("confusion_matrix", "y_true", "y_pred", "y_proba")},
    }]
    return {
        "model": model, "history": history, "eval_test": eval_test,
        "best_cfg": best_cfg, "use_static": not use_seq_only,
        "test_ds": test_ds, "variant_label": variant_label,
        "val_variant_f1": candidates, "comparison_rows": rows,
    }
