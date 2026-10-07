"""Command line entry point for the LSTM pipeline.

    python -m src.cjps_lstm.cli --data-dir data
    python -m src.cjps_lstm.cli --data-dir data --no-segments --trials 8

Mirrors ``src.cjps_train.cli``: defaults live in ``LSTMConfig`` so the
manifest, not the parser, describes the run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.cjps_lstm.config import DEFAULT_OUTPUT_DIR, LSTMConfig

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.cjps_lstm.cli",
        description="LSTM pipeline for next-touchpoint prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Directory holding A_TravelDataJourneys.csv and B_TravelDataUsers.csv.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-len", type=int, default=None,
                   help="Sequence cap. Default: P%d of train input lengths." % 95)
    p.add_argument("--trials", type=int, default=None, dest="n_tune_trials",
                   help="Random-search tuning trials.")
    p.add_argument("--epochs", type=int, default=None, dest="max_epochs")
    p.add_argument("--no-segments", action="store_true",
                   help="Skip the DBSCAN+Spectral segmentation feature.")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: 2 trials, 10 epochs, no segmentation.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT, stream=sys.stdout,
    )

    kwargs: dict = {"output_dir": args.output_dir}
    for name in ("seed", "max_len", "n_tune_trials", "max_epochs"):
        value = getattr(args, name, None)
        if value is not None:
            kwargs[name] = value
    if args.no_segments:
        kwargs["use_segments"] = False
    if args.quick:
        kwargs.update(n_tune_trials=2, max_epochs=10, tune_max_epochs=10,
                      use_segments=False)
    config = LSTMConfig(**kwargs)

    from src.cjps_lstm.run import run
    artifacts = run(config, data_dir=args.data_dir)

    print()
    print(artifacts.comparison.to_string(index=False))
    print(f"\noutputs: {artifacts.output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
