"""Command line entry point for the CJPS training pipeline.

    python -m src.cjps_train.cli --data-dir data --force
    python -m src.cjps_train.cli --dry-run     # data checks only, writes nothing

Every option exists because it changes a decision the pipeline otherwise makes
silently. The defaults are the ones the manifest records, so a run with no
arguments is fully described by its output file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.cjps_train.config import (
    DEFAULT_MODEL_DIR,
    DEFAULT_REPORTS_DIR,
    DatasetPaths,
    TrainConfig,
)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cjps_train.cli",
        description="Train the CJPS segmentation + next-touchpoint models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--data-dir", type=Path, default=None,
                      help="Directory holding the two source CSVs.")
    data.add_argument("--journeys-csv", type=Path, default=None)
    data.add_argument("--users-csv", type=Path, default=None)
    data.add_argument("--dry-run", action="store_true",
                      help="Load and validate the data, then stop. Writes nothing.")

    out = parser.add_argument_group("output")
    out.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                     help="Where to write the five joblib artefacts.")
    out.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR,
                     help="Where to write the training manifest.")
    out.add_argument("--force", action="store_true",
                     help="Overwrite existing artefacts.")

    tune = parser.add_argument_group("evaluation")
    tune.add_argument("--seed", type=int, default=None,
                      help="RNG seed. Recorded in the manifest.")
    tune.add_argument("--cv-splits", type=int, default=None)
    tune.add_argument("--holdout", type=float, default=None, dest="holdout_fraction",
                      help="Fraction of rows reserved and scored exactly once.")
    tune.add_argument("--min-class-samples", type=int, default=None,
                      help="Floor for a target class to survive. Raised to at "
                           "least cv_splits + 1 regardless.")
    tune.add_argument("--k-neighbors", type=int, default=None,
                      help="SMOTE/ADASYN neighbour count.")
    tune.add_argument("--per-cluster-models", action="store_true", default=None,
                      help="Train one classifier per segment instead of one pooled model.")
    tune.add_argument("--skip-selection", action="store_true",
                      help="Skip the benchmark and use a default model. The "
                           "manifest records that no model was actually chosen.")

    seg = parser.add_argument_group("segmentation")
    seg.add_argument("--k-distance-k", type=int, default=None,
                     help="k for the DBSCAN elbow heuristic.")
    seg.add_argument("--eps-percentiles", type=float, nargs="+", default=None,
                     help="Percentiles of the k-distance curve to search for eps.")
    seg.add_argument("--n-clusters", type=int, nargs="+", default=None,
                     dest="n_clusters_candidates",
                     help="Candidate cluster counts to score.")
    seg.add_argument("--silhouette-floor", type=float, default=None,
                     help="Below this best-silhouette, fall back to the midpoint k.")

    verbosity = parser.add_argument_group("verbosity")
    verbosity.add_argument("-v", "--verbose", action="store_true")
    verbosity.add_argument("-q", "--quiet", action="store_true")

    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Build a config, applying only the options the user actually passed.

    Defaults live on :class:`TrainConfig` rather than on the parser, so the
    manifest and the code cannot drift apart. ``None`` here means "not
    specified", which is different from "specified as the default".
    """
    kwargs: dict = {}
    for name in (
        "seed",
        "cv_splits",
        "holdout_fraction",
        "min_class_samples",
        "k_neighbors",
        "k_distance_k",
        "silhouette_floor",
    ):
        value = getattr(args, name, None)
        if value is not None:
            kwargs[name] = value

    if args.eps_percentiles:
        kwargs["eps_percentiles"] = tuple(args.eps_percentiles)
    if args.n_clusters_candidates:
        kwargs["n_clusters_candidates"] = tuple(args.n_clusters_candidates)
    if args.per_cluster_models is not None:
        kwargs["per_cluster_models"] = bool(args.per_cluster_models)

    kwargs["model_dir"] = args.model_dir
    kwargs["reports_dir"] = args.reports_dir
    kwargs["force"] = args.force
    return TrainConfig(**kwargs)


def paths_from_args(args: argparse.Namespace) -> DatasetPaths:
    if args.data_dir or args.journeys_csv or args.users_csv:
        return DatasetPaths(
            journeys=args.journeys_csv or (args.data_dir / "A_TravelDataJourneys.csv"),
            users=args.users_csv or (args.data_dir / "B_TravelDataUsers.csv"),
        )
    return DatasetPaths.from_env()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    logging.basicConfig(level=level, format=LOG_FORMAT, stream=sys.stdout)

    config = config_from_args(args)
    paths = paths_from_args(args)

    # Import late so ``--help`` stays fast and does not pull in sklearn.
    from src.cjps_train import data as data_mod
    from src.cjps_train.cluster import ClusteringError
    from src.cjps_train.data import DataError
    from src.cjps_train.evaluate import EvaluationError
    from src.cjps_train.run import TrainingError, train

    # Everything this pipeline raises on purpose. Each message already says what
    # to fix, so they are reported rather than dumped as a traceback.
    expected = (DataError, ClusteringError, EvaluationError, TrainingError)

    if args.dry_run:
        logging.getLogger("cjps_train").info("Dry run: validating data only")
        try:
            journeys, users = data_mod.load_dataset(paths)
            report = data_mod.LoadReport()
            frame, report, _ = data_mod.assemble(journeys, users)
        except (DataError, ClusteringError) as exc:
            print(f"\nData check failed:\n{exc}", file=sys.stderr)
            return 2
        print("\nData looks usable.")
        print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
        print(f"\nmodelling frame: {frame.shape[0]} rows x {frame.shape[1]} columns")
        return 0

    try:
        result = train(config, paths=paths, skip_selection=args.skip_selection)
    except expected as exc:
        print(f"\nTraining failed:\n{exc}", file=sys.stderr)
        return 2

    print()
    print(result.summary())
    if result.manifest_path:
        print(f"manifest: {result.manifest_path}")

    verification = result.manifest.get("verification", {})
    if verification.get("ok") and verification.get("skipped"):
        print(f"verification: {verification.get('error')}")
    elif verification.get("ok"):
        print("verification: artefact(s) loaded and produced a prediction")
        for channel in verification.get("channels", []):
            print(f"  {channel['name']:<38} {channel['probability']:6.2f}%")
    else:
        print(f"verification FAILED: {verification.get('error')}", file=sys.stderr)
        return 1

    notes = result.manifest.get("notes", [])
    if notes:
        print("\nRead before trusting these numbers:")
        for note in notes:
            print(f"  - {note}")

    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
