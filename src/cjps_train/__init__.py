"""Leakage-free, reproducible training for the CJPS models.

Replaces ``src/main.py``, which remains available as the original research
script. What changed is not the modelling intent but three things that made the
reported numbers untrustworthy:

**Resampling now happens inside the cross-validation fold.** The original
resampled the whole dataset and then drew folds from the result, so every
validation row was reachable from the minority neighbourhood that produced the
synthetic samples the model trained on. See :mod:`src.cjps_train.evaluate`.

**There is a holdout set.** Nothing was ever scored on data that had not
influenced a decision. One stratified split is taken before model selection and
scored exactly once, at the end.

**Model and cluster count are actually selected.** The original ran a benchmark,
printed it, and then fitted a hardcoded ``GradientBoostingClassifier`` against a
hardcoded ``n_clusters=3`` regardless of what the numbers said. Here the
benchmark picks, and the manifest records the scores behind the choice.

Module map
----------
``config``      paths, seeds, the feature contract shared with the web app
``data``        loading, cleaning, journey assembly, imputation
``preprocess``  the two ColumnTransformers
``cluster``     DBSCAN eps and the cluster count, both chosen from the data
``evaluate``    the leakage-free cross-validation loop and the holdout
``select``      the candidate roster and the selection rule
``run``         orchestration, manifest, artefact writing and verification
``cli``         command line entry point
"""

from src.cjps_train.config import TrainConfig, DatasetPaths

__all__ = ["TrainConfig", "DatasetPaths", "train"]


def train(*args, **kwargs):
    """Lazy re-export so ``import src.cjps_train`` stays cheap."""
    from src.cjps_train.run import train as _train

    return _train(*args, **kwargs)
