"""Data assembly for the LSTM pipeline.

The unit of prediction is one row per **user**: a chronological sequence of
touchpoint events (the input) plus the user's final touchpoint (the target).
That is the same prediction contract as the shipped app — ``step1`` is the
most recent touchpoint, predicted from the history before it — so the LSTM
and the GradientBoosting baseline answer the same question and their metrics
are comparable.

Three design decisions matter and are easy to get wrong:

**The target event never enters the input.** Every aggregate feature
(journey length, purchase counts, durations, ...) is computed over the input
events only. Computing them over the full journey would let the model read
the answer off its own features.

**Consecutive repeats are compressed.** A user refreshing the same channel
produces many identical rows; feeding them verbatim teaches the LSTM that
the most likely continuation of any touchpoint is itself. Runs of the same
``type_touch`` are collapsed to one event with summed duration, matching
what ``cjps_train.data.compress_journey`` does for the tabular model.

**Everything fitted is fitted on the training users.** The KNN imputer, the
scalers and the one-hot encoders see only training rows. The existing
pipeline fits the imputer on the full user table — acceptable there because
it is unsupervised, but the stricter rule costs nothing and removes the
ambiguity entirely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from src.cjps_train.data import clean_users, read_csv
from src.cjps_lstm.config import PAD_TOKEN, TOUCHPOINT_CODES, UNK_TOKEN, LSTMConfig

logger = logging.getLogger("cjps_lstm.data")

USER_ID = "UserID"
TIME = "TIMESPSS"
TOUCH = "type_touch"
DURATION = "Duration"

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
# Demographics, split by measurement type. The names come from the user
# table; ``RESP_GEM_GROOTTE`` (municipality size) is included here — the
# tabular pipeline ignored it, but it is a genuine ordinal demographic and
# the fusion step has no reason to drop available signal.

#: Numeric, roughly continuous → standardised.
DEMO_NUMERIC: tuple[str, ...] = ("Age", "BAS_huishoudgrootte")

#: Ordinal codes with a meaningful order → MinMax keeps the order on [0, 1].
DEMO_ORDINAL: tuple[str, ...] = (
    "BAS_bruto_jaarinkomen",        # income bands, 1..8 ascending
    "BAS_voltooide_opleiding8_resp",  # education level, 1..8 ascending
    "AFG_sk2015",                    # social class
    "SPSS_Lifestage",                # household life stage progression
    "RESP_GEM_GROOTTE",              # municipality size class
)

#: Nominal codes, no ordering → one-hot. Cardinality is small (2–9), so the
#: sparse expansion costs a handful of columns; an embedding would add
#: parameters without buying anything at this vocabulary size.
DEMO_NOMINAL: tuple[str, ...] = ("GenderID", "SPSS_Regio5", "BAS_werkzaamheid_resp")

#: Count-like, skewed → robust scaling.
DEMO_SKEWED: tuple[str, ...] = ("afg_kinderen_huishouden",)

DEMOGRAPHIC_FEATURES = DEMO_NUMERIC + DEMO_ORDINAL + DEMO_NOMINAL + DEMO_SKEWED

#: Per-event numeric channels fed to the LSTM alongside the touchpoint
#: embedding. All are leakage-free: they describe the event itself.
EVENT_NUMERIC: tuple[str, ...] = (
    "duration_log",   # log1p of summed run duration, standardised
    "gap_log",        # log1p of hours since the previous event, standardised
    "device_mobile",  # 1 = MOBILE panel, 0 = FIXED
    "purchase_own",   # own-brand purchase flag on the event
    "purchase_any",   # any-brand purchase flag on the event
)

#: Behavioural aggregates over the input sequence — the "customer journey
#: summary" block of the fusion step. Computed without the target event.
AGGREGATE_FEATURES: tuple[str, ...] = (
    "seq_len",            # number of input events (journey length)
    "n_unique_touch",     # distinct touchpoints visited
    "duration_total",     # summed dwell time across the input journey
    "duration_mean",      # mean dwell per event
    "duration_max",       # longest single dwell
    "gap_mean_h",         # mean hours between consecutive events
    "gap_std_h",          # irregularity of visit cadence
    "span_days",          # days between first and last input event
    "events_per_day",     # interaction frequency
    "n_sessions",         # visits separated by >30 min count as new sessions
    "purchase_own_n",     # own-brand purchases in the input journey
    "purchase_any_n",     # any-brand purchases in the input journey
    "mobile_share",       # share of events on the mobile panel
    "last_hour",          # hour of day of the latest input event
    "last_dow",           # day of week of the latest input event
    "last_is_weekend",    # weekend flag of the latest input event
)

#: Seconds gap that separates two sessions. Standard web-analytics
#: convention.
SESSION_GAP_SECONDS = 30 * 60


# ---------------------------------------------------------------------------
# Event table
# ---------------------------------------------------------------------------


def build_events(journeys: pd.DataFrame) -> pd.DataFrame:
    """Compress the raw click log into distinct touchpoint events per user.

    One output row per *run* of consecutive identical ``type_touch`` values
    within a user, carrying the run's start time, summed duration and the
    purchase/device flags. Mirrors ``cjps_train.data.compress_journey`` but
    keeps the aggregated per-run values the sequence model consumes.
    """
    frame = journeys.copy()

    frame[USER_ID] = pd.to_numeric(frame[USER_ID], errors="coerce")
    frame[TIME] = pd.to_datetime(frame[TIME], errors="coerce")
    frame[TOUCH] = pd.to_numeric(frame[TOUCH], errors="coerce")
    frame[DURATION] = pd.to_numeric(frame[DURATION], errors="coerce")
    frame = frame.dropna(subset=[USER_ID, TIME, TOUCH])
    # A zero/negative duration is a tracking artefact, not a visit.
    frame = frame[frame[DURATION].fillna(0) > 0]

    frame[USER_ID] = frame[USER_ID].astype("int64")
    frame[TOUCH] = frame[TOUCH].astype(int)
    frame = frame.sort_values([USER_ID, TIME], kind="mergesort").reset_index(drop=True)

    # Run boundary = user change or touchpoint change; the cumsum is grouped
    # per user so the counter restarts for each journey.
    changed = (frame[TOUCH] != frame[TOUCH].shift()) | (frame[USER_ID] != frame[USER_ID].shift())
    frame["_run"] = changed.groupby(frame[USER_ID]).cumsum()

    events = (
        frame.groupby([USER_ID, "_run"], sort=False)
        .agg(
            ts=(TIME, "min"),
            duration=(DURATION, "sum"),
            type_touch=(TOUCH, "first"),
            device_mobile=("DEVICE_TYPE", lambda s: int((s == "MOBILE").any())),
            purchase_own=("purchase_own", "max"),
            purchase_any=("purchase_any", "max"),
        )
        .reset_index()
        .sort_values([USER_ID, "ts"], kind="mergesort")
        .reset_index(drop=True)
    )

    # Per-event engineered channels.
    events["duration_log"] = np.log1p(events["duration"])
    prev_ts = events.groupby(USER_ID)["ts"].shift()
    events["gap_log"] = np.log1p(
        (events["ts"] - prev_ts).dt.total_seconds().fillna(0).clip(lower=0) / 3600.0
    )
    logger.info(
        "Events: %d raw rows -> %d compressed events over %d users",
        len(frame), len(events), events[USER_ID].nunique(),
    )
    return events


# ---------------------------------------------------------------------------
# Per-user examples
# ---------------------------------------------------------------------------


def build_examples(events: pd.DataFrame, min_len: int) -> pd.DataFrame:
    """Turn the event table into one row per user.

    Columns:
      - ``target``: ``type_touch`` of the user's most recent event
      - ``seq_*``: parallel lists describing the input events (all but last)
      - aggregate features over the input events only
      - ``last_ts``: timestamp of the final input event (the "now" of the
        prediction), kept for error analysis and diagnostics
    """
    records = []
    for user_id, g in events.groupby(USER_ID, sort=True):
        n = len(g)
        if n < min_len:
            continue
        inp = g.iloc[:-1]
        target = int(g.iloc[-1][TOUCH])

        ts = inp["ts"]
        gaps_h = ts.diff().dt.total_seconds().fillna(0) / 3600.0
        span_days = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0
        n_inp = len(inp)

        records.append(
            {
                USER_ID: int(user_id),
                "target": target,
                "seq_touch": inp[TOUCH].astype(int).tolist(),
                "seq_duration_log": inp["duration_log"].tolist(),
                "seq_gap_log": inp["gap_log"].tolist(),
                "seq_device": inp["device_mobile"].tolist(),
                "seq_pown": inp["purchase_own"].tolist(),
                "seq_pany": inp["purchase_any"].tolist(),
                "last_ts": ts.iloc[-1],
                # --- aggregates over the INPUT events only -----------------
                "seq_len": n_inp,
                "n_unique_touch": inp[TOUCH].nunique(),
                "duration_total": float(inp["duration"].sum()),
                "duration_mean": float(inp["duration"].mean()),
                "duration_max": float(inp["duration"].max()),
                "gap_mean_h": float(gaps_h.mean()),
                "gap_std_h": float(gaps_h.std()) if n_inp > 1 else 0.0,
                "span_days": float(span_days),
                "events_per_day": float(n_inp / max(span_days, 1.0)),
                "n_sessions": int((gaps_h > SESSION_GAP_SECONDS / 3600.0).sum() + 1),
                "purchase_own_n": float(inp["purchase_own"].sum()),
                "purchase_any_n": float(inp["purchase_any"].sum()),
                "mobile_share": float(inp["device_mobile"].mean()),
                "last_hour": float(ts.iloc[-1].hour),
                "last_dow": float(ts.iloc[-1].dayofweek),
                "last_is_weekend": float(ts.iloc[-1].dayofweek >= 5),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        # from_records on [] yields a columnless frame; nothing downstream
        # can run without the schema, so fail loudly at the source.
        raise ValueError(
            f"no user has >= {min_len} compressed events; lower "
            "min_journey_length or check the journeys table"
        )
    logger.info(
        "Examples: %d users with >= %d events; input length P50=%d P95=%d max=%d",
        len(frame), min_len,
        int(frame["seq_len"].median()),
        int(np.percentile(frame["seq_len"], 95)),
        int(frame["seq_len"].max()),
    )
    return frame


# ---------------------------------------------------------------------------
# Encoders — everything fitted on the training split only
# ---------------------------------------------------------------------------


def select_aggregate_features(frame: pd.DataFrame, threshold: float = 0.9) -> list[str]:
    """Drop redundant aggregate features by correlation, on the train split.

    Several aggregates are near-duplicates by construction — ``duration_total``
    is ``seq_len × duration_mean`` up to rounding, ``n_unique_touch`` grows
    with ``seq_len``. Keeping both gives the network two copies of the same
    signal and the importance analysis a false sense of "this feature does
    nothing". A greedy pass drops the later column of any pair above
    ``threshold``, so listed order doubles as a preference order.
    """
    cols = list(AGGREGATE_FEATURES)
    corr = frame[cols].corr().abs()
    dropped: set[str] = set()
    for i, a in enumerate(cols):
        if a in dropped:
            continue
        for b in cols[i + 1:]:
            if b not in dropped and corr.loc[a, b] > threshold:
                dropped.add(b)
                logger.info("feature selection: dropping %r (|r|=%.2f with %r)", b, corr.loc[a, b], a)
    keep = [c for c in cols if c not in dropped]
    logger.info("feature selection: %d/%d aggregate features kept", len(keep), len(cols))
    return keep


class StaticEncoder:
    """Encodes the fused per-user static feature block.

    Combines the demographic groups (numeric / ordinal / nominal / skewed)
    with the behavioural aggregates. Fitted on training users only; the
    fitted transformers are what must travel with the model.
    """

    def __init__(self, extra_nominal: tuple[str, ...] = (),
                 aggregate_features: tuple[str, ...] | None = None):
        self.nominal_cols = list(DEMO_NOMINAL) + list(extra_nominal)
        self.aggregate_cols = list(aggregate_features or AGGREGATE_FEATURES)
        self.numeric_scaler = StandardScaler()
        self.ordinal_scaler = MinMaxScaler()
        self.skewed_scaler = RobustScaler()
        self.aggregate_scaler = RobustScaler()
        self.nominal_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
        self.feature_names_: list[str] = []

    def _split(self, frame: pd.DataFrame):
        demo = frame.copy()
        for col in DEMOGRAPHIC_FEATURES:
            demo[col] = pd.to_numeric(demo[col], errors="coerce")
        for col in self.nominal_cols:
            if col not in DEMO_NOMINAL:
                demo[col] = pd.to_numeric(demo[col], errors="coerce")
        agg = frame[self.aggregate_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        return demo, agg

    def fit(self, frame: pd.DataFrame) -> "StaticEncoder":
        demo, agg = self._split(frame)
        self.numeric_scaler.fit(demo[list(DEMO_NUMERIC)])
        self.ordinal_scaler.fit(demo[list(DEMO_ORDINAL)])
        self.skewed_scaler.fit(demo[list(DEMO_SKEWED)])
        self.nominal_encoder.fit(demo[self.nominal_cols].fillna(-1))
        self.aggregate_scaler.fit(agg)

        nominal_names = list(
            self.nominal_encoder.get_feature_names_out(self.nominal_cols)
        )
        self.feature_names_ = (
            list(DEMO_NUMERIC)
            + list(DEMO_ORDINAL)
            + nominal_names
            + list(DEMO_SKEWED)
            + self.aggregate_cols
        )
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        demo, agg = self._split(frame)
        parts = [
            self.numeric_scaler.transform(demo[list(DEMO_NUMERIC)]),
            self.ordinal_scaler.transform(demo[list(DEMO_ORDINAL)]),
            self.nominal_encoder.transform(demo[self.nominal_cols].fillna(-1)),
            self.skewed_scaler.transform(demo[list(DEMO_SKEWED)]),
            self.aggregate_scaler.transform(agg.fillna(0)),
        ]
        return np.hstack(parts).astype(np.float32)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)


class TouchpointVocab:
    """Fixed vocabulary over the known touchpoint codes.

    Index 0 is the padding token, ``UNK_TOKEN`` catches codes outside the
    dictionary (none exist in the data, but the bucket keeps the contract
    honest rather than crashing on a surprise).
    """

    def __init__(self):
        self._map = {code: i + 1 for i, code in enumerate(TOUCHPOINT_CODES)}
        self.size = len(TOUCHPOINT_CODES) + 2
        self.pad = PAD_TOKEN
        self.unk = UNK_TOKEN

    def encode(self, code) -> int:
        return self._map.get(int(code), UNK_TOKEN)

    def encode_seq(self, codes) -> list[int]:
        return [self.encode(c) for c in codes]

    # Reverse mapping for reporting.
    def decode(self, idx: int) -> int:
        if idx == PAD_TOKEN or idx == UNK_TOKEN:
            return -1
        return TOUCHPOINT_CODES[idx - 1]


def fit_event_scaler(num_arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over the TRAIN input events only.

    ``duration_log`` and ``gap_log`` are log-compressed but still skewed;
    standardising them keeps the LSTM's input scale comparable to the
    embedding outputs. Binary channels are passed through unchanged —
    scaling a {0,1} flag only rescales a distance that is already correct.
    """
    stacked = np.concatenate(num_arrays, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std == 0] = 1.0
    # Only continuous channels are scaled; binary flags (device, purchases)
    # are left alone. Their channel indices are the last three.
    binary = {2, 3, 4}
    for i in range(stacked.shape[1]):
        if i in binary:
            mean[i] = 0.0
            std[i] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_event_scaler(num: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (num - mean) / std


# ---------------------------------------------------------------------------
# Sequence padding
# ---------------------------------------------------------------------------


def encode_example(row: pd.Series, vocab: TouchpointVocab) -> tuple[np.ndarray, np.ndarray]:
    """One user's input sequence as ``(token_ids, numeric_events)`` arrays."""
    ids = np.asarray(vocab.encode_seq(row["seq_touch"]), dtype=np.int64)
    numeric = np.stack(
        [
            np.asarray(row["seq_duration_log"], dtype=np.float32),
            np.asarray(row["seq_gap_log"], dtype=np.float32),
            np.asarray(row["seq_device"], dtype=np.float32),
            np.asarray(row["seq_pown"], dtype=np.float32),
            np.asarray(row["seq_pany"], dtype=np.float32),
        ],
        axis=1,
    )
    return ids, numeric


def pad_batch(
    ids_list: list[np.ndarray],
    num_list: list[np.ndarray],
    max_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad/truncate to ``max_len`` — keep the MOST RECENT events.

    The tail of the journey is the most predictive part (it is closest to
    the target), so truncation drops from the front. Returns
    ``(ids, numerics, lengths)`` with shapes ``(N, L)``, ``(N, L, F)``,
    ``(N,)``.
    """
    n = len(ids_list)
    n_feat = num_list[0].shape[1]
    ids = np.zeros((n, max_len), dtype=np.int64)
    nums = np.zeros((n, max_len, n_feat), dtype=np.float32)
    lens = np.zeros(n, dtype=np.int64)
    for i, (tok, num) in enumerate(zip(ids_list, num_list)):
        take = min(len(tok), max_len)
        ids[i, :take] = tok[-take:]
        nums[i, :take] = num[-take:]
        lens[i] = take
    return ids, nums, lens


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


@dataclass
class Split:
    """Indices of the three partitions. One row per user, so a row-level
    stratified split is automatically a user-level split: no customer can
    appear in two partitions because each contributes exactly one example."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def split_examples(frame: pd.DataFrame, config: LSTMConfig) -> Split:
    """Stratified 70/15/15 split on the target label.

    A temporal split was considered and rejected: journeys overlap heavily in
    calendar time (each user's target is simply their last observed event,
    scattered over 17 months), so a global time cut would truncate training
    histories arbitrarily while buying no leakage protection — the inputs
    and the target already belong to the same user by construction.
    """
    idx = np.arange(len(frame))
    y = frame["target"].to_numpy()

    holdout = config.val_fraction + config.test_fraction
    train_idx, hold_idx = train_test_split(
        idx, test_size=holdout, random_state=config.seed, stratify=y
    )
    val_rel = config.val_fraction / holdout
    val_idx, test_idx = train_test_split(
        hold_idx, test_size=1 - val_rel, random_state=config.seed, stratify=y[hold_idx]
    )
    logger.info(
        "Split: %d train / %d val / %d test", len(train_idx), len(val_idx), len(test_idx)
    )
    return Split(train=train_idx, val=val_idx, test=test_idx)


def drop_rare_classes(frame: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    """Remove targets too rare to populate all three partitions."""
    counts = frame["target"].value_counts()
    keep = set(counts[counts >= min_samples].index)
    dropped = sorted(set(counts.index) - keep)
    if dropped:
        logger.warning(
            "Dropping %d row(s): target class(es) %s have < %d members",
            int((~frame["target"].isin(keep)).sum()), dropped, min_samples,
        )
    return frame[frame["target"].isin(keep)].reset_index(drop=True)
