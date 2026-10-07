"""LSTM pipeline for next-touchpoint prediction.

Companion to ``src/cjps_train/``: that package ships the production
DBSCAN + Spectral + GradientBoosting artefacts the web app serves. This
package asks a different question of the same two source tables — can a
sequence model over the *whole* journey beat a classifier that sees only
the last two touchpoints? — and answers it with a leakage-safe
train/validate/test protocol.

Run it with::

    python -m src.cjps_lstm.cli --data-dir data
"""
