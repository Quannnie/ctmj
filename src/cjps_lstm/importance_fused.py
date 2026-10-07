"""Supplemental: grouped permutation importance for the FUSED variant.

The final (winning) model is sequence-only, so its static channels carry no
signal by construction and its importance table would be uninformative about
demographics. This script retrains the default fused variant — the same
model that appears as "LSTM — fused" in model_comparison.csv — and permutes
feature groups on the validation set, where the static block is live.

    python -m src.cjps_lstm.importance_fused --data-dir data
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.cjps_lstm.config import LSTMConfig
from src.cjps_lstm import run as run_mod
from src.cjps_lstm import train as train_mod

logger = logging.getLogger("cjps_lstm.importance_fused")

#: Roll the one-hot and code-level features back up to the fields a reader
#: recognises. "What matters" is more actionable at this granularity.
GROUP_OF = {
    "Age": "demographic", "BAS_huishoudgrootte": "demographic",
    "BAS_bruto_jaarinkomen": "demographic",
    "BAS_voltooide_opleiding8_resp": "demographic",
    "AFG_sk2015": "demographic", "SPSS_Lifestage": "demographic",
    "RESP_GEM_GROOTTE": "demographic", "GenderID": "demographic",
    "SPSS_Regio5": "demographic", "BAS_werkzaamheid_resp": "demographic",
    "final_label": "segment",
    "duration_log": "temporal", "gap_log": "temporal",
    "device_mobile": "journey", "purchase_own": "journey", "purchase_any": "journey",
}


def group_of(name: str) -> str:
    if name.startswith("SEQ:"):
        return "sequence (touchpoint order)"
    if name.startswith("SEQ_NUM:"):
        return "temporal"
    if name.startswith("STATIC:"):
        base = name.split(":", 1)[1]
        # One-hot names look like "GenderID_1.0" — strip the category suffix.
        for prefix in sorted(GROUP_OF, key=len, reverse=True):
            if base == prefix or base.startswith(prefix + "_"):
                return GROUP_OF[prefix]
        return "behavioural aggregate"
    return "other"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.cjps_lstm.importance_fused")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
                        stream=sys.stdout)

    config = LSTMConfig()
    if args.output_dir:
        config = dataclasses.replace(config, output_dir=args.output_dir)
    outdir = Path(config.output_dir)

    prep = run_mod.prepare(config, args.data_dir)
    frame, split = prep["frame"], prep["split"]
    tensors = run_mod.build_tensors(frame, split, config)
    classes = tensors["classes"]
    n_classes = len(classes)
    n_event_feat = tensors["train"].seq_num.shape[2]
    n_static = tensors["train"].static.shape[1]

    weights = train_mod.balanced_class_weights(
        tensors["train"].labels.numpy(), n_classes) if config.class_weighted else None
    model = train_mod.build_model(config, n_event_feat, n_static, n_classes,
                                  tensors["vocab"].size)
    train_mod.train_model(model, tensors["train"], tensors["val"], config, weights)

    blocks = run_mod.importance_blocks(tensors["encoder"])
    imp = train_mod.permutation_importance(
        model, tensors["val"], blocks, config.batch_size,
        config.seed, config.importance_repeats,
    )
    imp["group"] = imp["feature"].map(group_of)
    imp.to_csv(outdir / "feature_importance_fused.csv", index=False)

    grouped = (imp.groupby("group")["importance"].sum()
               .sort_values(ascending=False).reset_index())
    grouped.to_csv(outdir / "feature_importance_grouped.csv", index=False)
    logger.info("\n%s", grouped.to_string(index=False))
    print(grouped.to_string(index=False))
    print(imp.head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
