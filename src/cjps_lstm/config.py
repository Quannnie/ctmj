"""Configuration for the LSTM pipeline.

Same discipline as :mod:`src.cjps_train.config`: paths are absolute,
hyperparameters live in one dataclass, and everything that can change the
result is recorded in the run manifest.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Everything the run writes: figures, metrics, the best checkpoint, the
#: fitted preprocessors and the report.
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resources" / "lstm"

SEED = 20260929

#: Touchpoint codes the source vocabulary defines. The mapping lives in
#: ``src.dictionaries.dictionary.type_touch``; codes 11 and 17 exist in the
#: dictionary's gaps and simply never appear. Keeping the vocabulary a fixed
#: contract — rather than fitting it on the training split — is safe here
#: because the codes are domain knowledge, not statistics of the target.
TOUCHPOINT_CODES: tuple[int, ...] = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22
)

PAD_TOKEN = 0          # embedding index 0 is reserved for padding
UNK_TOKEN = len(TOUCHPOINT_CODES) + 1   # last index is the unknown bucket
VOCAB_SIZE = len(TOUCHPOINT_CODES) + 2  # + PAD + UNK


@dataclass(frozen=True)
class LSTMConfig:
    """Every knob a run exposes. Serialised into the manifest."""

    seed: int = SEED

    # --- sequence construction ------------------------------------------
    #: A user needs at least this many distinct (compressed) touchpoints —
    #: two for the input sequence, one for the target.
    min_journey_length: int = 3
    #: Longest input sequence kept. ``None`` = take the percentile below of
    #: the observed input lengths, computed on the TRAIN split only.
    max_len: int | None = None
    max_len_percentile: float = 95.0

    # --- split -----------------------------------------------------------
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    #: A target class needs at least this many members to survive; below it
    #: stratified three-way splitting cannot place a row in every partition.
    min_class_samples: int = 6

    # --- architecture ----------------------------------------------------
    emb_dim: int = 32
    lstm_units: int = 64
    lstm_layers: int = 1
    dropout: float = 0.2
    dense_units: int = 64
    bidirectional: bool = False

    # --- optimisation ----------------------------------------------------
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 60
    #: Early stopping patience on validation loss.
    patience: int = 7
    #: Balanced cross-entropy weights, fitted on the training labels only.
    class_weighted: bool = True

    # --- segmentation feature -------------------------------------------
    #: When true, fit the stage-1 DBSCAN + Spectral segmentation and expose
    #: ``final_label`` as a static feature. Unsupervised, so it is fitted on
    #: all users without leaking the target.
    use_segments: bool = True

    # --- tuning ----------------------------------------------------------
    #: Random-search trials for hyperparameter tuning on the validation set.
    n_tune_trials: int = 12
    #: Cap on epochs inside a tuning trial — trials get less patience budget
    #: than the final fit.
    tune_max_epochs: int = 30

    # --- permutation importance ------------------------------------------
    #: Repetitions per feature when permuting; means are reported.
    importance_repeats: int = 3

    output_dir: Path = DEFAULT_OUTPUT_DIR


#: Random-search space for tuning. Bounds are deliberate, not guessed:
#: embedding and LSTM width bracket a wide range around the defaults, the
#: learning rate spans an order of magnitude, and sequence length compares
#: the P90/P95/P99 choices directly rather than assuming longer is better.
SEARCH_SPACE: dict[str, list] = {
    "emb_dim": [16, 32, 48, 64],
    "lstm_units": [32, 64, 128],
    "lstm_layers": [1, 2],
    "dropout": [0.1, 0.2, 0.3, 0.4],
    "dense_units": [32, 64, 128],
    "lr": [1e-3, 5e-4, 3e-4],
    "batch_size": [64, 128, 256],
    "bidirectional": [False, True],
}


def data_paths(data_dir: Path | None = None) -> tuple[Path, Path]:
    """Resolve the two source CSVs, mirroring ``cjps_train`` conventions."""
    base = Path(data_dir or os.environ.get("CJPS_DATA_DIR", PROJECT_ROOT / "data"))
    journeys = Path(os.environ.get("CJPS_JOURNEYS_CSV", base / "A_TravelDataJourneys.csv"))
    users = Path(os.environ.get("CJPS_USERS_CSV", base / "B_TravelDataUsers.csv"))
    return journeys, users
