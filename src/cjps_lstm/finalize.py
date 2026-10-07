"""Re-run only the final stage of the pipeline on an existing output dir.

The expensive parts of a run — data prep, the GBM baselines and the
12-trial tuning search — are deterministic given the seed and already live
in ``resources/lstm/``. This module reuses that state to redo only what a
change to :func:`src.cjps_lstm.run.final_stage` affects: the input-variant
selection, the train+val refit, the test evaluation, the importance and
error-analysis tables, and the artefacts.

    python -m src.cjps_lstm.finalize --data-dir data
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.cjps_lstm.config import LSTMConfig, data_paths
from src.cjps_lstm import run as run_mod
from src.cjps_lstm import train as train_mod

logger = logging.getLogger("cjps_lstm.finalize")


def finalize(config: LSTMConfig, data_dir: Path | None = None) -> dict:
    import joblib
    import torch

    outdir = Path(config.output_dir)
    metrics_path = outdir / "metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError(f"{metrics_path} not found — run the full pipeline first.")

    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    best_params = saved["best_params"]

    prep = run_mod.prepare(config, data_dir)
    frame, split = prep["frame"], prep["split"]

    # Sanity check: deterministic prep must reproduce the recorded splits.
    recorded = saved.get("split", {})
    if recorded and (len(split.train) != recorded.get("train")
                     or len(split.val) != recorded.get("val")
                     or len(split.test) != recorded.get("test")):
        raise RuntimeError(
            f"Split drift: recorded {recorded} vs rebuilt "
            f"{len(split.train)}/{len(split.val)}/{len(split.test)}. "
            "Prep is not deterministic — do not reuse the old artefacts."
        )

    tensors = run_mod.build_tensors(frame, split, config)
    classes = tensors["classes"]
    n_classes = len(classes)
    n_event_feat = tensors["train"].seq_num.shape[2]
    n_static = tensors["train"].static.shape[1]

    ds_train_seq = run_mod.make_seq_only_dataset(tensors["train"], n_classes)
    ds_val_seq = run_mod.make_seq_only_dataset(tensors["val"], n_classes)
    ds_test_seq = run_mod.make_seq_only_dataset(tensors["test"], n_classes)

    final = run_mod.final_stage(
        frame, split, tensors, config, best_params,
        seq_datasets=(ds_train_seq, ds_val_seq, ds_test_seq), outdir=outdir,
    )
    eval_test = final["eval_test"]
    final_model = final["model"]
    history = final["history"]
    best_cfg = final["best_cfg"]

    # Refresh the comparison: keep the non-LSTM-final rows, append the new one.
    comparison = pd.read_csv(outdir / "model_comparison.csv")
    comparison = comparison[~comparison["model"].str.startswith("LSTM — tuned")]
    comparison = pd.concat([comparison, pd.DataFrame(final["comparison_rows"])],
                           ignore_index=True)
    comparison.to_csv(outdir / "model_comparison.csv", index=False)
    logger.info("\n%s", comparison.to_string(index=False))

    blocks = run_mod.importance_blocks(tensors["encoder"])
    if not final["use_static"]:
        blocks = {k: v for k, v in blocks.items() if not k.startswith("STATIC:")}
    imp = train_mod.permutation_importance(
        final_model, final["test_ds"], blocks,
        best_cfg.batch_size, config.seed, config.importance_repeats,
    )
    imp.to_csv(outdir / "feature_importance.csv", index=False)

    tables = run_mod.error_analysis(frame, split, eval_test, classes)
    for name, tbl in tables.items():
        tbl.to_csv(outdir / f"error_{name}.csv", index=False)

    torch.save({"state_dict": final_model.state_dict(),
                "config": asdict(best_cfg),
                "use_static": final["use_static"],
                "classes": classes.tolist(),
                "max_len": tensors["max_len"]},
               outdir / "best_lstm.pt")
    joblib.dump({"encoder": tensors["encoder"], "vocab": tensors["vocab"],
                 "imputer": prep["imputer"]}, outdir / "preprocessors.pkl")

    run_mod._save_figures(outdir, history, eval_test, classes, imp, comparison)

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
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    logger.info("Finalized. Test metrics: %s", metrics["test"])
    return metrics


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.cjps_lstm.finalize")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
                        stream=sys.stdout)
    config = LSTMConfig()
    if args.output_dir:
        config = dataclasses.replace(config, output_dir=args.output_dir)
    finalize(config, data_dir=args.data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
