"""Loading, cleaning and feature assembly for the CJPS training run.

This is where most of the original pipeline's silent failure modes lived.

**Missing values were strings.** The source data uses ``"--"`` for "not
answered". ``pd.read_csv`` without ``na_values`` keeps it as a string, so a
column of mostly ``"--"`` reads back as object dtype and every later
``fit_transform`` either raises or, worse, succeeds by treating a lexicographic
comparison as numeric. Worse, a ``KNNImputer`` fitted afterwards cannot repair
it, because the column is no longer numeric.

**A journey was not a sequence, it was a bag.** Consecutive visits to the same
touchpoint have to collapse to one row before "the last three touchpoints" means
anything. The original code did this, but with a ``group_id`` computed from a
*global* ``cumsum`` over the whole frame rather than per user. It happened to be
correct only because ``UserID`` was part of the grouping key — a coupling that
would break the moment anyone reused the column.

**Row order decided what "step 1" meant.** The pivot took whichever three rows
came last in the frame, so the target was the most recent touchpoint *in
arbitrary order* rather than in time order.

**The imputer was fitted but never saved.** ``KNNImputer`` ran over the user
table, then the imputed frame was thrown away and only the numeric result was
used. Nothing persisted it, so inference had no way to apply the same
imputation — the training and serving pipelines simply disagreed.

Everything here is a pure function of its inputs, so each step is testable
without touching the filesystem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.cjps_train.config import (
    CLUSTERING_FEATURES,
    MISSING_TOKENS,
    TARGET,
    USER_COLUMNS,
    DatasetPaths,
)

logger = logging.getLogger("cjps_train.data")

#: Column that distinguishes one visit from the next within a user's journey.
TOUCHPOINT_COLUMN = "type_touch"
#: Timestamp column, snake case in the source file.
TIME_COLUMN = "TIMESPSS"
DURATION_COLUMN = "Duration"
USER_ID_COLUMN = "UserID"

#: A journey must have at least this many distinct touchpoints to yield a
#: (step2, step3) -> step1 triple. Two would only give one input.
MIN_JOURNEY_LENGTH = 3

#: How many most-recent distinct touchpoints the model uses.
JOURNEY_WINDOW = 3


class DataError(RuntimeError):
    """Raised when the source data cannot support training. Message is explicit."""


@dataclass
class LoadReport:
    """Row counts at each stage, so a suspicious drop is visible rather than
    discovered later as a mysteriously small training set."""

    users_raw: int = 0
    users_after_clean: int = 0
    journeys_raw: int = 0
    journeys_after_duration_filter: int = 0
    journeys_compressed: int = 0
    journeys_with_window: int = 0
    dropped_unknown_user: int = 0
    imputed_cells: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "users_raw": self.users_raw,
            "users_after_clean": self.users_after_clean,
            "journeys_raw": self.journeys_raw,
            "journeys_after_duration_filter": self.journeys_after_duration_filter,
            "journeys_compressed": self.journeys_compressed,
            "journeys_with_window": self.journeys_with_window,
            "dropped_unknown_user": self.dropped_unknown_user,
            "imputed_cells": self.imputed_cells,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def read_csv(path: Path, *, required: Iterable[str] = ()) -> pd.DataFrame:
    """Read a source CSV, treating the dataset's missing markers as NaN.

    ``na_values`` is not optional here. Without it, ``"--"`` survives as a
    string and the column's dtype is object, which defeats every numeric
    operation that follows.
    """
    if not path.is_file():
        raise DataError(
            f"Không tìm thấy tệp dữ liệu: {path}\n"
            "Đặt A_TravelDataJourneys.csv và B_TravelDataUsers.csv vào thư mục "
            "data/ hoặc trỏ CTMJ_DATA_DIR sang vị trí khác."
        )

    frame = pd.read_csv(
        path,
        na_values=list(MISSING_TOKENS),
        keep_default_na=True,
        low_memory=False,
    )
    frame.columns = [str(c).strip() for c in frame.columns]

    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise DataError(
            f"{path.name} thiếu cột bắt buộc: {', '.join(missing)}. "
            f"Cột hiện có: {', '.join(frame.columns[:15])}"
        )
    logger.info("Read %s: %d rows x %d columns", path.name, len(frame), frame.shape[1])
    return frame


def load_dataset(paths: DatasetPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read both source tables, or explain precisely what is missing."""
    absent = paths.missing()
    if absent:
        listing = "\n  ".join(str(p) for p in absent)
        raise DataError(
            "Thiếu tệp dữ liệu nguồn:\n  " + listing + "\n"
            "Đặt chúng vào data/ hoặc đặt biến môi trường CJPS_DATA_DIR / "
            "CJPS_JOURNEYS_CSV / CJPS_USERS_CSV."
        )

    users = read_csv(
        paths.users,
        required=[USER_ID_COLUMN, *CLUSTERING_FEATURES],
    )
    journeys = read_csv(
        paths.journeys,
        required=[USER_ID_COLUMN, TIME_COLUMN, TOUCHPOINT_COLUMN, DURATION_COLUMN],
    )
    return journeys, users


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def coerce_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Force the given columns to numeric, reporting what could not be coerced.

    Silent coercion failures are the problem: ``pd.to_numeric`` turns
    ``"unknown"`` into NaN without complaint, so a column that was meant to be
    an age and is actually free text disappears into missing values and the
    imputer quietly papers over it.
    """
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            continue
        before = out[column].isna().sum()
        out[column] = pd.to_numeric(out[column], errors="coerce")
        lost = int(out[column].isna().sum() - before)
        if lost:
            logger.warning(
                "Column %s: %d value(s) could not be read as numbers and became NaN",
                column,
                lost,
            )
    return out


#: A profile row is discarded when at least this fraction of its clustering
#: features are missing. Below it, KNN imputation can still say something
#: useful; above it, most of the "neighbours" that would fill the gaps are
#: themselves mostly guesses, and keeping the row adds noise rather than data.
MAX_MISSING_FRACTION = 0.5


def clean_users(users: pd.DataFrame, report: LoadReport | None = None) -> pd.DataFrame:
    """Normalise the user table: coerce dtypes, drop unusable rows and duplicates.

    Duplicate ``UserID`` rows are collapsed by taking the first, because the
    journey frame is joined on this key and a duplicate would multiply every
    touchpoint that user has.

    Rows with *some* missing features are deliberately **kept**. Dropping them
    instead would make the KNN imputer a no-op — there would be nothing left to
    impute — and on this dataset it cost 28% of the profiles while the manifest
    reported ``imputed_cells: 0``, which reads as "no missing data" rather than
    "all of it was thrown away". Only rows missing most of their features are
    dropped.
    """
    frame = users.copy()
    if report is not None:
        report.users_raw = len(frame)

    feature_cols = [c for c in CLUSTERING_FEATURES if c in frame.columns]
    frame = coerce_numeric(frame, feature_cols)

    frame[USER_ID_COLUMN] = pd.to_numeric(frame[USER_ID_COLUMN], errors="coerce")
    before = len(frame)
    frame = frame.dropna(subset=[USER_ID_COLUMN])
    dropped_id = before - len(frame)

    frame[USER_ID_COLUMN] = frame[USER_ID_COLUMN].astype("int64")
    duplicated = int(frame[USER_ID_COLUMN].duplicated().sum())
    if duplicated:
        logger.warning("Dropping %d duplicate UserID row(s)", duplicated)
        frame = frame.drop_duplicates(subset=[USER_ID_COLUMN], keep="first")

    # A row with no feature values at all has no neighbours, so imputation
    # cannot help it and it would poison the neighbour search for others.
    all_missing = frame[feature_cols].isna().all(axis=1)
    n_all_missing = int(all_missing.sum())
    if n_all_missing:
        frame = frame[~all_missing]

    if feature_cols:
        missing_fraction = frame[feature_cols].isna().mean(axis=1)
        too_sparse = missing_fraction > MAX_MISSING_FRACTION
        n_sparse = int(too_sparse.sum())
        if n_sparse:
            logger.info(
                "Dropping %d profile(s) missing more than %.0f%% of their features",
                n_sparse,
                100 * MAX_MISSING_FRACTION,
            )
            frame = frame[~too_sparse]

    if report is not None:
        report.users_after_clean = len(frame)
        if dropped_id:
            report.notes.append(f"{dropped_id} user row(s) dropped for a missing UserID")
        if duplicated:
            report.notes.append(f"{duplicated} duplicate UserID(s) collapsed")
        kept_with_gaps = int(frame[feature_cols].isna().any(axis=1).sum()) if feature_cols else 0
        if kept_with_gaps:
            report.notes.append(
                f"{kept_with_gaps} profile(s) retained with missing features and "
                "filled by KNN imputation"
            )
        logger.info(
            "Users after cleaning: %d (from %d), %d still has gaps to impute",
            len(frame),
            report.users_raw,
            kept_with_gaps,
        )
    else:
        logger.info("Users after cleaning: %d", len(frame))
    return frame


# ---------------------------------------------------------------------------
# Journey assembly
# ---------------------------------------------------------------------------


def compress_journey(journeys: pd.DataFrame, report: LoadReport | None = None) -> pd.DataFrame:
    """Collapse consecutive visits to the same touchpoint into one row.

    A user who browsed three accommodation pages in a row has had one touchpoint
    three times, not three touchpoints. Keeping all three would fill the
    ``(step2, step3) -> step1`` window with a single channel and teach the model
    that people stay put.

    The run identifier is computed **per user**. The original used a single
    global ``cumsum``, which only worked because ``UserID`` also appeared in the
    groupby key — the two were coupled by coincidence rather than by design.
    """
    frame = journeys.copy()
    if report is not None:
        report.journeys_raw = len(frame)

    frame[USER_ID_COLUMN] = pd.to_numeric(frame[USER_ID_COLUMN], errors="coerce")
    frame = frame.dropna(subset=[USER_ID_COLUMN])
    frame[USER_ID_COLUMN] = frame[USER_ID_COLUMN].astype("int64")

    frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN], errors="coerce")
    frame = frame.dropna(subset=[TIME_COLUMN])

    frame[TOUCHPOINT_COLUMN] = pd.to_numeric(frame[TOUCHPOINT_COLUMN], errors="coerce")
    frame = frame.dropna(subset=[TOUCHPOINT_COLUMN])
    frame[TOUCHPOINT_COLUMN] = frame[TOUCHPOINT_COLUMN].astype(int)

    if DURATION_COLUMN in frame.columns:
        frame[DURATION_COLUMN] = pd.to_numeric(frame[DURATION_COLUMN], errors="coerce")
        # A zero or negative duration is a tracking artefact, not a visit.
        frame = frame[frame[DURATION_COLUMN].fillna(0) > 0]

    if report is not None:
        report.journeys_after_duration_filter = len(frame)

    frame = frame.sort_values([USER_ID_COLUMN, TIME_COLUMN], kind="mergesort").reset_index(drop=True)

    # A run breaks when the user changes or the touchpoint changes. Computing
    # this within each user group is what makes ``run`` restart per user.
    changed = frame[TOUCHPOINT_COLUMN] != frame[TOUCHPOINT_COLUMN].shift()
    changed.iloc[0] = True
    new_user = frame[USER_ID_COLUMN] != frame[USER_ID_COLUMN].shift()
    run_starts = (changed | new_user).groupby(frame[USER_ID_COLUMN]).cumsum()
    frame["_run"] = run_starts

    keep_time = frame.groupby([USER_ID_COLUMN, "_run"], sort=False)[TIME_COLUMN].min()
    frame = frame.drop_duplicates(subset=[USER_ID_COLUMN, "_run"], keep="first")
    frame = frame.set_index([USER_ID_COLUMN, "_run"])
    frame[TIME_COLUMN] = keep_time
    frame = frame.reset_index()

    if report is not None:
        report.journeys_compressed = len(frame)
        before = report.journeys_after_duration_filter
    else:
        before = len(frame)
    logger.info(
        "Journey rows %s -> %d after compressing consecutive repeats",
        before,
        len(frame),
    )
    return frame


def build_journey_window(
    compressed: pd.DataFrame, report: LoadReport | None = None
) -> pd.DataFrame:
    """One row per user holding the last three distinct touchpoints, in time order.

    Naming follows the app, which is the part that matters: ``step1`` is the
    most recent touchpoint, ``step2`` the one before it, ``step3`` the one
    before that. The target is ``step1`` and the inputs are ``step2``/``step3``,
    so the model is trained to predict the step that follows two observed ones.

    Getting this backwards is easy and produces a model that looks fine, so the
    ordering is asserted rather than assumed: :func:`verify_window_order`
    re-derives the ordering from the timestamps.
    """
    ordered = compressed.sort_values(
        [USER_ID_COLUMN, TIME_COLUMN], ascending=[True, False], kind="mergesort"
    )

    # Rank each user's events newest-first and keep the top three, so step 1 is
    # the most recent. ``groupby().head()`` alone preserves frame order, which
    # is only the right order because of the sort above; the explicit rank makes
    # the numbering independent of that.
    ranked = ordered.groupby(USER_ID_COLUMN, sort=False).cumcount() + 1
    recent = ordered[ranked <= JOURNEY_WINDOW].copy()
    recent["step"] = ranked[ranked <= JOURNEY_WINDOW].to_numpy()

    window = recent.pivot(index=USER_ID_COLUMN, columns="step", values=TOUCHPOINT_COLUMN)
    window.columns = [f"step{int(c)}" for c in window.columns]
    window = window.reset_index()

    required = ["step1", "step2", "step3"]
    present = [c for c in required if c in window.columns]
    missing = [c for c in required if c not in window.columns]
    if missing:
        raise DataError(
            f"Không tạo được cửa sổ hành trình: thiếu {', '.join(missing)}. "
            f"Đã tạo được {', '.join(present)}."
        )
    window = window.dropna(subset=required)

    if report is not None:
        report.journeys_with_window = len(window)
    logger.info("Users with a full %d-step window: %d", JOURNEY_WINDOW, len(window))
    return window


def verify_window_order(
    compressed: pd.DataFrame, window: pd.DataFrame, sample: int = 500
) -> None:
    """Confirm each window row matches the events it claims to be.

    This has to be a genuinely independent check, or it is decoration. An
    earlier version re-derived the timestamps from ``compressed`` and compared
    those against each other — which is always true, because it derived them
    with the same descending sort it was checking. It could not fail, and would
    not have caught an inverted sort.

    What is actually verified here is the window's *contents*: the touchpoint
    codes it claims are step 1, 2 and 3 are compared against an independent
    reconstruction that ranks each user's events with ``nlargest`` rather than
    with the ``cumcount`` path used to build the window. A sort inversion, an
    off-by-one in the ranking, or a stale window all show up as a mismatch.
    """
    ordered = compressed.sort_values(
        [USER_ID_COLUMN, TIME_COLUMN], ascending=[True, False], kind="mergesort"
    )

    # Independent reconstruction: the three newest events per user, by
    # timestamp, with the step number derived from a list index rather than a
    # cumulative count.
    newest = (
        ordered.groupby(USER_ID_COLUMN, sort=False)[TIME_COLUMN]
        .apply(lambda s: list(s.nlargest(JOURNEY_WINDOW).index))
    )
    # One row per user, with the three expected codes in three columns. Building
    # a list of single-key dicts instead would produce one row per position and
    # a join on nothing.
    rows = []
    for user_id, index_list in newest.items():
        record = {USER_ID_COLUMN: user_id}
        for position, row_index in enumerate(index_list, start=1):
            record[f"_expect{position}"] = int(ordered.loc[row_index, TOUCHPOINT_COLUMN])
        rows.append(record)
    expected = pd.DataFrame(rows)

    merged = window.merge(expected, on=USER_ID_COLUMN, how="inner")
    if merged.empty:
        return
    merged = merged.head(sample)

    mismatch = (
        (merged["step1"] != merged["_expect1"])
        | (merged["step2"] != merged["_expect2"])
        | (merged["step3"] != merged["_expect3"])
    )
    bad = merged[mismatch]
    if len(bad):
        example = bad.iloc[0]
        raise DataError(
            "Nội dung cửa sổ hành trình không khớp với các sự kiện gốc: "
            f"{len(bad)}/{len(merged)} hồ sơ sai. "
            f"Ví dụ UserID={example[USER_ID_COLUMN]}: "
            f"step1={example['step1']} nhưng sự kiện mới nhất là {example['_expect1']}. "
            "Thứ tự bước đang bị đảo hoặc cửa sổ được dựng từ dữ liệu cũ."
        )


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------


def build_imputer(users: pd.DataFrame, n_neighbors: int = 5) -> tuple[pd.DataFrame, object]:
    """Fit a KNN imputer on the user table and return the frame plus the imputer.

    The imputer is **returned, not discarded**. The original fitted it, replaced
    the table with the result, and never saved the fitted object, so nothing
    downstream — including the web app — could apply the same imputation. The
    trained pair is the contract; the imputed frame alone is not.
    """
    from sklearn.impute import KNNImputer

    feature_cols = [c for c in CLUSTERING_FEATURES if c in users.columns]
    if not feature_cols:
        raise DataError("Bảng người dùng không có cột đặc trưng nào để nội suy.")

    # Coerce before converting to float. A frame that still holds the dataset's
    # "--" placeholder would raise from ``to_numpy(dtype=float)`` with a message
    # about string conversion, which says nothing about where the problem came
    # from. Running the coercion here means this function is safe to call
    # directly, not only after ``clean_users``.
    frame = coerce_numeric(users, feature_cols)

    # Check for wholly-missing columns *before* fitting. KNNImputer silently
    # drops all-NaN columns from its output, so a column that cannot be imputed
    # does not survive as NaN — it makes the returned array narrower than the
    # input, and the assignment back into the frame fails with "Columns must be
    # same length as key". That error names neither the column nor the cause, so
    # the check has to happen here rather than after the fit.
    dead = [c for c in feature_cols if frame[c].isna().all()]
    if dead:
        raise DataError(
            "Cột không thể nội suy được vì toàn bộ giá trị đều thiếu: "
            f"{', '.join(dead)}. "
            "Kiểm tra lại tên cột trong tệp nguồn — thường là cột bị đổi tên."
        )

    n_neighbors = max(1, min(n_neighbors, len(users)))
    imputer = KNNImputer(n_neighbors=n_neighbors)
    array = imputer.fit_transform(frame[feature_cols].to_numpy(dtype=float))
    if array.shape[1] != len(feature_cols):
        # Belt and braces: KNNImputer's own dropped-column behaviour, caught
        # before it turns into a shape mismatch during assignment.
        raise DataError(
            f"KNNImputer trả về {array.shape[1]} cột cho {len(feature_cols)} cột "
            "đầu vào; có cột toàn số NaN đã bị loại bỏ."
        )

    imputed = frame.copy()
    imputed[feature_cols] = array
    imputed[feature_cols] = imputed[feature_cols].round(6)

    still_missing = int(imputed[feature_cols].isna().sum().sum())
    if still_missing:
        bad = [c for c in feature_cols if imputed[c].isna().any()]
        raise DataError(f"Cột vẫn còn giá trị thiếu sau nội suy: {', '.join(bad)}.")

    logger.info("Imputed %d user feature column(s) with KNN k=%d", len(feature_cols), n_neighbors)
    return imputed, imputer


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble(
    journeys: pd.DataFrame, users: pd.DataFrame
) -> tuple[pd.DataFrame, LoadReport, object]:
    """Run the full preparation and return the modelling frame.

    Returns ``(frame, report, imputer)``. The imputer travels with the data
    because the serving pipeline must reproduce the same imputation.
    """
    report = LoadReport()

    users_clean = clean_users(users, report)
    # Counted before imputation, so the manifest distinguishes "no missing data"
    # from "everything missing was filled in" — two very different claims that
    # the same post-hoc number would report identically.
    report.imputed_cells = int(users_clean[list(CLUSTERING_FEATURES)].isna().sum().sum())
    users_imputed, imputer = build_imputer(users_clean)

    compressed = compress_journey(journeys, report)
    window = build_journey_window(compressed, report)
    verify_window_order(compressed, window)

    known = set(users_imputed[USER_ID_COLUMN].tolist())
    before = len(window)
    window = window[window[USER_ID_COLUMN].isin(known)].copy()
    report.dropped_unknown_user = before - len(window)
    if report.dropped_unknown_user:
        report.notes.append(
            f"{report.dropped_unknown_user} user(s) in the journey table are absent "
            "from the user table and were dropped"
        )

    frame = window.merge(users_imputed, on=USER_ID_COLUMN, how="inner")

    required = set(CLUSTERING_FEATURES) | {"step1", "step2", "step3"}
    absent = sorted(c for c in required if c not in frame.columns)
    if absent:
        raise DataError(
            "Bảng dữ liệu cuối thiếu cột: " + ", ".join(absent) + ". "
            "Kiểm tra lại tên cột trong tệp nguồn."
        )

    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise DataError(
            "Không còn dòng nào sau khi ghép dữ liệu hành trình với hồ sơ người dùng. "
            "Kiểm tra lại ràng buộc Duration > 0 và số bước tối thiểu."
        )

    report.notes.append(
        f"target {TARGET!r} distribution: "
        f"{frame[TARGET].nunique()} distinct touchpoint(s) over {len(frame)} rows"
    )
    logger.info("Modelling frame: %d rows x %d columns", len(frame), frame.shape[1])
    return frame, report, imputer
