"""Synthetic source data for exercising the training pipeline end to end.

The real dataset is a private Figshare download. These fixtures reproduce its
*shape* — the same two tables, the same column names, the same quirks — so the
pipeline can be tested without network access or the original data:

* a users table with ``"--"`` standing in for unanswered questions, so the
  missing-value path is actually exercised;
* a journeys table with a mix of repeated and distinct touchpoints, so the
  compression step has something to collapse and the next-step target is not
  trivially the previous channel;
* three genuine behavioural segments, so k-selection has real structure to find
  rather than a single blob it must flail against.

The generator is seeded, so every test sees identical data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 4242

#: Touchpoint codes present in the reference data.
TOUCHPOINTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22]

#: Channel families, so the next step is not uniformly random.
FAMILIES = {
    "search": [3, 6, 9, 12, 15, 16],
    "web": [1, 4, 7, 10, 13],
    "app": [2, 5, 8, 14],
    "paid": [18, 19, 21, 22],
    "direct": [20],
}

#: Per-segment profile ranges. Deliberately overlapping on Age so clustering has
#: to work for its answer rather than reading a single column.
SEGMENT_PROFILES = {
    0: dict(age=(30, 72), household=(1, 2), children=(0, 0), work=(6, 9),
            education=(2, 5), lifestage=(2, 5), income=(2, 5), klass=(2, 4), regio=(1, 5)),
    1: dict(age=(32, 58), household=(3, 5), children=(1, 3), work=(2, 9),
            education=(3, 7), lifestage=(6, 7), income=(3, 6), klass=(2, 5), regio=(1, 5)),
    2: dict(age=(19, 38), household=(1, 3), children=(0, 1), work=(2, 9),
            education=(3, 8), lifestage=(1, 3), income=(2, 4), klass=(2, 5), regio=(1, 5)),
}


def _family_of(code: int) -> str:
    for name, codes in FAMILIES.items():
        if code in codes:
            return name
    return "web"


def _pick(rng: np.random.Generator, low: int, high: int) -> int:
    """Inclusive integer in ``[low, high]``, tolerating a single-value range.

    Several segments are pinned to a constant (no children, for instance), and
    ``rng.integers(0, 0)`` raises. Writing the range this way keeps the profile
    table declarative instead of forcing every pinned column to be spelled
    specially.
    """
    return int(rng.integers(int(low), int(high) + 1))


def make_users(n: int = 600, seed: int = SEED, missing_rate: float = 0.06) -> pd.DataFrame:
    """A users table with realistic gaps."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        segment = i % 3
        spec = SEGMENT_PROFILES[segment]
        rows.append({
            "UserID": 100000 + i,
            "GenderID": int(rng.choice([1, 2])),
            "Age": _pick(rng, *spec["age"]),
            "SPSS_Regio5": _pick(rng, *spec["regio"]),
            "BAS_huishoudgrootte": _pick(rng, *spec["household"]),
            "BAS_werkzaamheid_resp": _pick(rng, *spec["work"]),
            "afg_kinderen_huishouden": _pick(rng, *spec["children"]),
            "BAS_bruto_jaarinkomen": _pick(rng, *spec["income"]),
            "AFG_sk2015": _pick(rng, *spec["klass"]),
            "BAS_voltooide_opleiding8_resp": _pick(rng, *spec["education"]),
            "SPSS_Lifestage": _pick(rng, *spec["lifestage"]),
        })
    frame = pd.DataFrame(rows)

    # Blank out a slice of the answerable columns, the way the real file does.
    #
    # The columns have to be widened to object first: pandas refuses to assign
    # the string "--" into an int64 column, and the real file stores exactly
    # that string. Writing the frame as all-numeric would produce a fixture
    # that never exercises the missing-value path, which is the path most worth
    # testing here.
    if missing_rate:
        answerable = [
            "BAS_werkzaamheid_resp",
            "BAS_bruto_jaarinkomen",
            "AFG_sk2015",
            "BAS_voltooide_opleiding8_resp",
            "SPSS_Lifestage",
        ]
        for column in answerable:
            hit = rng.random(len(frame)) < missing_rate
            frame[column] = frame[column].astype(object)
            frame.loc[hit, column] = "--"
    return frame


def make_journeys(
    users: pd.DataFrame,
    seed: int = SEED,
    avg_events: int = 14,
    repeat_probability: float = 0.45,
) -> pd.DataFrame:
    """A journeys table whose last three events depend on segment and channel.

    The target is the *next* event after two observed ones, so the relationship
    is learnable: a user who just did a search tends to search or open an app
    next, and a retired household behaves differently from a family with
    children. That is the signal the classifier is supposed to find, and without
    it a fixture would only prove the code runs.
    """
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2024-01-01")
    rows = []

    for _, user in users.iterrows():
        user_id = int(user["UserID"])
        segment = (user_id - 100000) % 3

        n_events = int(rng.integers(max(4, avg_events - 6), avg_events + 7))
        channel = int(rng.choice(TOUCHPOINTS))
        timestamp = base

        for _ in range(n_events):
            # Repeat the current channel sometimes, so compression has work.
            if rng.random() > repeat_probability:
                if rng.random() < 0.4:
                    channel = next_touchpoint(rng, segment, channel)
                else:
                    channel = int(rng.choice([c for c in TOUCHPOINTS if c != channel]))

            timestamp = timestamp + pd.Timedelta(minutes=int(rng.integers(3, 600)))
            rows.append({
                "UserID": user_id,
                "TIMESPSS": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "type_touch": channel,
                "Duration": int(rng.integers(1, 1800)),
                "PurchaseID": int(rng.integers(1, 5000)),
                "DEVICE_TYPE": int(rng.choice([1, 2, 3])),
            })

    return pd.DataFrame(rows)


def write_dataset(
    directory: Path,
    n_users: int = 600,
    seed: int = SEED,
    missing_rate: float = 0.06,
) -> tuple[Path, Path]:
    """Write both CSVs and return their paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    users = make_users(n=n_users, seed=seed, missing_rate=missing_rate)
    journeys = make_journeys(users, seed=seed)

    journeys_path = directory / "A_TravelDataJourneys.csv"
    users_path = directory / "B_TravelDataUsers.csv"
    journeys.to_csv(journeys_path, index=False)
    users.to_csv(users_path, index=False)
    return journeys_path, users_path


#: Channels a segment is likely to move to next. Narrower than a whole family,
#: so the next-step task is genuinely learnable rather than near-random: a
#: family has four to six members, and spreading the signal across all of them
#: puts the achievable macro-F1 close to the 1/20 chance baseline, which would
#: make an end-to-end test unable to distinguish learning from noise.
SEGMENT_NEXT_CHANNELS = {
    0: [3, 6, 12, 15],     # older households keep searching
    1: [1, 4, 7, 10, 13],  # families browse comparison sites
    2: [2, 5, 8, 14, 20],  # younger households use the apps
}

#: Probability the next touchpoint comes from the segment's own set.
SEGMENT_FIDELITY = 0.62


def next_touchpoint(
    rng: np.random.Generator, segment: int, step3: int
) -> int:
    """The touchpoint a user is likely to reach next.

    Two overlapping influences, because the features available at prediction
    time are the profile and the two previous channels:

    * the segment's own preferred set, which the model can read off
      ``final_label`` and the profile columns;
    * continuity with the channel just visited, which it can read off
      ``step3``.

    The remainder is uniform, so the task is learnable but not deterministic —
    a fixture where the target is a function of the inputs would only prove
    that the code runs.
    """
    roll = rng.random()
    if roll < SEGMENT_FIDELITY:
        return int(rng.choice(SEGMENT_NEXT_CHANNELS[segment]))
    if roll < SEGMENT_FIDELITY + 0.25:
        return int(rng.choice(FAMILIES[_family_of(int(step3))]))
    return int(rng.choice(TOUCHPOINTS))


def tiny_frame(n: int = 240, seed: int = SEED) -> pd.DataFrame:
    """A small already-assembled modelling frame, for tests that skip loading.

    Columns match what :func:`src.cjps_train.data.assemble` produces, including
    the ``step1/2/3`` window and a ``final_label``.
    """
    rng = np.random.default_rng(seed)
    users = make_users(n=n, seed=seed, missing_rate=0.0)
    # Two clusters is enough for the fixture; the real run selects k.
    labels = np.arange(n) % 2

    step2 = [int(rng.choice(TOUCHPOINTS)) for _ in range(n)]
    step3 = [int(rng.choice(TOUCHPOINTS)) for _ in range(n)]
    step1 = [
        next_touchpoint(rng, i % 3, s3) for i, s3 in enumerate(step3)
    ]

    frame = users.copy()
    frame["step2"] = step2
    frame["step3"] = step3
    frame["step1"] = step1
    frame["final_label"] = labels
    return frame
