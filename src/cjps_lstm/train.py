"""Training loop, evaluation and permutation importance for JourneyLSTM.

Early stopping monitors **validation loss** (not accuracy): for a
14-class imbalanced problem, loss is the smoother and more sensitive
signal, and ``restore_best`` keeps the weights from the best epoch rather
than the last one — which is the entire point of the callback.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from torch import nn
from torch.utils.data import DataLoader

from src.cjps_lstm.model import JourneyDataset, JourneyLSTM

logger = logging.getLogger("cjps_lstm.train")


@dataclass
class History:
    """Per-epoch training record, for learning curves."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_f1: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)
    best_epoch: int = -1
    stopped_at: int = -1


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(ds: JourneyDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, generator=gen, drop_last=False
    )


def _predict_proba(model: JourneyLSTM, loader: DataLoader) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for ids, num, lens, static, _ in loader:
            out.append(torch.softmax(model(ids, num, lens, static), dim=1).numpy())
    return np.vstack(out) if out else np.empty((0, 0))


def _epoch(model, loader, criterion, optimizer=None) -> tuple[float, np.ndarray, np.ndarray]:
    """One pass. ``optimizer=None`` runs evaluation-only."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, n = 0.0, 0
    probs, labels = [], []
    with torch.set_grad_enabled(training):
        for ids, num, lens, static, y in loader:
            logits = model(ids, num, lens, static)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)
            probs.append(torch.softmax(logits.detach(), dim=1).numpy())
            labels.append(y.numpy())
    return (
        total_loss / max(n, 1),
        np.vstack(probs) if probs else np.empty((0, 0)),
        np.concatenate(labels) if labels else np.empty(0, dtype=int),
    )


def train_model(
    model: JourneyLSTM,
    train_ds: JourneyDataset,
    val_ds: JourneyDataset,
    config,
    class_weights: np.ndarray | None = None,
) -> History:
    """Fit with early stopping on validation loss + restore-best-weights.

    ``ReduceLROnPlateau`` halves the learning rate when the monitored loss
    stalls, which extends the useful tail of training without a manual
    schedule.
    """
    set_seed(config.seed)
    weights = (
        torch.as_tensor(class_weights, dtype=torch.float32)
        if class_weights is not None
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    train_ld = make_loader(train_ds, config.batch_size, True, config.seed)
    val_ld = make_loader(val_ds, config.batch_size, False, config.seed)

    history = History()
    best_val = np.inf
    best_state = None
    bad_epochs = 0

    for epoch in range(config.max_epochs):
        tr_loss, tr_prob, tr_y = _epoch(model, train_ld, criterion, optimizer)
        vl_loss, vl_prob, vl_y = _epoch(model, val_ld, criterion)
        scheduler.step(vl_loss)

        tr_f1 = f1_score(tr_y, tr_prob.argmax(1), average="macro", zero_division=0)
        vl_f1 = f1_score(vl_y, vl_prob.argmax(1), average="macro", zero_division=0)
        history.train_loss.append(tr_loss)
        history.val_loss.append(vl_loss)
        history.train_f1.append(float(tr_f1))
        history.val_f1.append(float(vl_f1))

        if vl_loss < best_val - 1e-4:
            best_val = vl_loss
            best_state = copy.deepcopy(model.state_dict())
            history.best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                history.stopped_at = epoch
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if history.stopped_at < 0:
        history.stopped_at = len(history.train_loss) - 1
    logger.info(
        "Trained %d epoch(s); best val_loss %.4f at epoch %d",
        len(history.train_loss), best_val, history.best_epoch,
    )
    return history


def evaluate(model: JourneyLSTM, ds: JourneyDataset, class_labels: np.ndarray, batch_size: int = 512) -> dict:
    """Full metric bundle on one partition.

    ROC-AUC and PR-AUC are computed one-vs-rest, macro-averaged over the
    classes actually present — the multiclass analogue of the binary
    metrics, and PR-AUC is included because the target is imbalanced.
    """
    loader = make_loader(ds, batch_size, False, 0)
    proba = _predict_proba(model, loader)
    y_true = ds.labels.numpy()
    y_pred = proba.argmax(1)

    present = np.unique(y_true)
    y_bin = np.zeros((len(y_true), len(class_labels)), dtype=int)
    for i, c in enumerate(class_labels):
        y_bin[:, i] = (y_true == c)
    keep_cols = [i for i, c in enumerate(class_labels) if c in present]

    try:
        roc_auc = float(roc_auc_score(y_bin[:, keep_cols], proba[:, keep_cols], average="macro"))
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = float(
            np.mean(
                [
                    average_precision_score(y_bin[:, i], proba[:, i])
                    for i in keep_cols
                ]
            )
        )
    except ValueError:
        pr_auc = float("nan")

    k = min(3, proba.shape[1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "top3_accuracy": float(
            top_k_accuracy_score(y_true, proba, k=k, labels=np.arange(proba.shape[1]))
        ),
        "roc_auc_macro": roc_auc,
        "pr_auc_macro": pr_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(len(class_labels))),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": proba,
    }


def permutation_importance(
    model: JourneyLSTM,
    ds: JourneyDataset,
    feature_blocks: dict[str, tuple[str, int | slice]],
    batch_size: int,
    seed: int,
    repeats: int = 3,
) -> pd.DataFrame:
    """Permutation importance over named feature blocks.

    ``feature_blocks`` maps a name to ``(where, index)``: ``where`` is one
    of ``"seq_ids"`` (permute whole token sequences across samples),
    ``"seq_num"`` (permute one per-event channel), ``"static"`` (permute
    one static column). Importance is the drop in macro-F1, averaged over
    ``repeats`` shuffles.

    Evaluated on the VALIDATION partition so the test set stays sealed.
    """
    import pandas as pd

    rng = np.random.RandomState(seed)
    base = evaluate(model, ds, np.arange(ds.labels.max() + 1), batch_size)["f1_macro"]

    rows = []
    for name, (where, idx) in feature_blocks.items():
        drops = []
        for rep in range(repeats):
            ids = ds.seq_ids.clone()
            num = ds.seq_num.clone()
            static = ds.static.clone()
            perm = torch.as_tensor(rng.permutation(len(ds)), dtype=torch.long)

            if where == "seq_ids":
                # Whole-sequence permutation: tests whether the order/content
                # of the journey matters at all.
                ids = ids[perm]
            elif where == "seq_num":
                num[:, :, idx] = num[perm][:, :, idx]
            elif where == "static":
                static[:, idx] = static[perm][:, idx]
            else:  # pragma: no cover - defensive
                raise ValueError(f"unknown block location {where!r}")

            shuffled = JourneyDataset(ids, num, ds.lengths, static, ds.labels.numpy())
            f1 = evaluate(model, shuffled, np.arange(ds.labels.max() + 1), batch_size)["f1_macro"]
            drops.append(base - f1)

        rows.append(
            {"feature": name, "importance": float(np.mean(drops)), "std": float(np.std(drops))}
        )
        logger.info("importance %-28s %.4f ± %.4f", name, rows[-1]["importance"], rows[-1]["std"])

    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def balanced_class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """``n / (k * count)`` weights, computed on the training labels only.

    Handles imbalance inside the loss instead of resampling: SMOTE-style
    synthesis does not apply to variable-length sequences, and class
    weights achieve the same recall protection without fabricating data.
    """
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = len(labels) / (n_classes * counts)
    return w / w.mean()  # normalised so the loss scale stays readable


def build_model(config, n_event_features: int, n_static: int, n_classes: int, vocab_size: int) -> JourneyLSTM:
    set_seed(config.seed)
    return JourneyLSTM(
        vocab_size=vocab_size,
        n_event_features=n_event_features,
        n_static=n_static,
        n_classes=n_classes,
        emb_dim=config.emb_dim,
        lstm_units=config.lstm_units,
        lstm_layers=config.lstm_layers,
        dense_units=config.dense_units,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
    )
