# LSTM pipeline — report

Sequence-model experiment on the same contract the production pipeline serves:
given a customer's journey history, predict the touchpoint of their **most
recent event** (`step1`). Everything below is computed from
`data/A_TravelDataJourneys.csv` + `data/B_TravelDataUsers.csv` by
`python -m src.cjps_lstm.cli --data-dir data`; nothing is hand-written.

Companion notebook: `src/notebook/lstm_pipeline.ipynb`.

---

## 1. Data and EDA

| table | rows | columns | role |
|---|---:|---:|---|
| `A_TravelDataJourneys.csv` | 2 456 414 | 10 | click log: `UserID`, `TIMESPSS`, `type_touch`, `Duration`, `DEVICE_TYPE`, `purchase_own`, `purchase_any`, `PurchaseID`, panel flags |
| `B_TravelDataUsers.csv` | 9 678 | 12 | profile: `UserID` + 11 demographic attributes |

Findings that shaped the pipeline:

- **Target** = `type_touch` of a user's latest event; inputs are all earlier
  events. 20 possible codes; 13 survive the ≥6-members-per-class filter needed
  for a three-way stratified split.
- **Imbalance is severe**: classes 1 and 7 hold ~328/~315 of 927 test rows
  (majority baseline accuracy 0.354); classes 3, 8, 9, 14 have ≤6 test members.
- **`Duration` is dirty**: 141 065 missing, 75 365 ≤ 0 (tracking artefacts —
  dropped), capped at 720 s.
- **Duplicates**: 20 075 exact duplicate rows; consecutive same-channel runs
  compress 2 239 984 usable rows to 303 086 events (8.1×) over 9 638 users.
- **Journey length is heavy-tailed**: compressed-length P50 = 9, P95 = 133,
  max = 1 622 → `max_len` is set from the train P95 (168), not the max.
- **Demographics are sparse**: ~1 603 of 9 678 profiles miss most fields;
  2 031 miss education. KNN imputation (fit on train only) rather than
  row-dropping, which would cost ~17% of fused examples.
- **Join**: 7 243 users have ≥3 compressed events; 6 180 survive the inner
  join to profiles + rare-class filter → 4 326 / 927 / 927 train/val/test,
  one row per user.

## 2. Feature engineering

Two families, `who they are` vs `what they did`:

- **Event channels** (fed at every timestep): touchpoint id → learned
  embedding; `duration_log`, `gap_log` (hours since previous event, log1p);
  `device_mobile`, `purchase_own`, `purchase_any` (binary passthrough).
- **Per-user aggregates** over *input events only* (the target event never
  contributes — that is the leakage guard): `seq_len`, `n_unique_touch`,
  duration/gap statistics, `span_days`, `events_per_day`, `n_sessions`
  (>30 min gap), purchase counts, `mobile_share`, last-event
  hour/day-of-week/weekend.
- **Feature selection**: correlation filter at |r|>0.9 on train aggregates —
  dropped `n_sessions` (r=0.93 with `seq_len`), `duration_max` (r=0.99 with
  `duration_mean`), `gap_std_h` (r=0.94 with `gap_mean_h`); 13/16 kept.

## 3. Encoding (justified per type)

| group | variables | encoding | why |
|---|---|---|---|
| ordinal | income band, education, social class, lifestage, municipality size | MinMax on codes | order is real; preserve it |
| continuous | `Age`, household size | StandardScaler | roughly continuous |
| skewed count | children in household | RobustScaler | median/IQR shrugs off outliers |
| nominal, low card. | `GenderID`, `SPSS_Regio5`, employment, `final_label` | one-hot | ≤10 categories each; an embedding buys nothing |
| nominal, event id | `type_touch` (20 codes) | **learned embedding**, dim tuned 16–64, best 32 | shared metric space for the LSTM; PAD=0, UNK=last index |
| binary | device, purchase flags | passthrough | already correct scale |

No label encoding of unordered data, no target encoding anywhere.

## 4. Fusion, sequence, split

Inner join on `UserID` (unique in the user table → no fan-out). Sequence =
compressed events in chronological order, last event removed to become the
target. Truncation keeps the **most recent** 168 events; `pack_padded_sequence`
guarantees padded steps never touch the hidden state. Split: stratified
70/15/15 on the target — user-level by construction (one row per customer).
A global temporal split was rejected: each user's input and target share one
clock, so time cuts buy no leakage protection and only truncate histories.

## 5. Models and selection discipline

- Baselines: majority class; **GBM on the app's contract** (step2+step3 +
  demographics + segment, the production feature set); **GBM on engineered
  features** (fused static vector + step2/step3). Both GBMs pick their config
  on validation.
- LSTM variants at default config: **touchpoints-only** ablation and
  **fused** (sequence + static).
- Random search (12 trials) over embedding dim, LSTM units/layers, dropout,
  dense width, lr, batch size, bidirectionality — scored on **validation
  macro-F1** only.
- **Input-variant selection is a validation decision**: the tuned config was
  trained both ways; sequence-only won (0.2832 vs 0.2687 fused), then was
  refit on train+val (12% inner split for early stopping) and scored on test
  exactly once.
- Early stopping on val loss (patience 7, restore best) + `ReduceLROnPlateau`
  (×0.5, patience 3); balanced class weights inside the loss instead of
  resampling — SMOTE does not apply to variable-length sequences.

## 6. Test results

| model | features | acc | P-macro | R-macro | F1-macro | F1-w | top-3 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority | — | .354 | .027 | .077 | .040 | .185 | — | — | — |
| GBM | app contract | .621 | .337 | .264 | .273 | .574 | **.888** | .734 | .257 |
| GBM | engineered | .616 | .273 | .215 | .223 | .562 | .882 | **.774** | .246 |
| LSTM | touchpoints only | .497 | .253 | **.354** | .261 | .535 | .732 | .790 | **.267** |
| LSTM | fused | .457 | .213 | .256 | .210 | .507 | .743 | .787 | .238 |
| **LSTM tuned (seq only)** | sequence | .423 | .224 | .309 | .223 | .478 | .710 | .772 | .247 |

Winner's config: emb 32, BiLSTM 64×1, dropout 0.3, dense 64, lr 1e-3, batch
128; early stop at epoch 17, best epoch 9.

Reading the gap honestly: the GBM dominates top-1/top-3 (the contract it was
designed for — the last two touchpoints are extremely predictive here). The
LSTM's edge is **macro recall** (.354 vs .264 at default config) — it finds
more of the rare touchpoints the GBM ignores — but the tuned model traded
some of that for precision and lost F1 on test, a normal val→test wobble on
a 13-class problem where the selection margin was ~0.01.

## 7. Feature importance (permutation, macro-F1 drop)

Sequence-only winner (`feature_importance.csv`): `SEQ:touchpoints` = **0.1701
± 0.004**; the (zeroed) numeric channels are structurally 0.

Fused variant, grouped (`feature_importance_grouped.csv`, measured on val):

| group | importance |
|---|---:|
| sequence (touchpoint order) | **0.2125** |
| journey (device/purchase flags) | 0.0063 |
| segment | −0.0013 |
| behavioural aggregate | −0.0069 |
| demographic | −0.0089 |
| temporal channels | −0.0252 |

The order/identity of touchpoints carries essentially *all* the signal;
every static block is noise within ±0.01 — consistent with fusion hurting
the model. Per-feature top entries: `seq_len` 0.018, `purchase_any_n` 0.0066,
`duration_log` 0.0055; every demographic ≤ 0.004.

## 8. Error analysis

- **Per class** (`error_per_class.csv`): 1 and 7 are handled well
  (F1 .558/.590); 9, 14 score exactly 0; 3, 8, 15 near 0 — a support problem
  (3–10 test members), not a systematic misread.
- **Top confusions** concentrate between the dominant channels and the
  mid-tier ones: 1→13 (46), 7→4 (35), 7→13 (32), 1→4 (25), 7→16 (23).
- **Journey length**: flat ~0.40–0.43 up to 100 events, then **0.52 at
  100+** — long histories give the LSTM real context to exploit.
- **Age gradient is the strongest demographic split**: <30 → 0.23, 30–45 →
  0.40, 45–60 → 0.44, 60+ → 0.47. Young customers' journeys are less
  predictable, or underrepresented (n=91).
- **Segment**: noise (−1) 0.29 vs segment 0 → 0.43 — unusual profiles are
  harder, as expected. Gender is nearly flat (0.41 vs 0.43).
- **Interaction count**: single-session users worst (0.34), then flat.
- **Error profile**: wrong predictions have *more* unique touchpoints
  (4.87 vs 4.00) and *shorter* journeys (40.6 vs 48.7 events) — diverse,
  short histories are the hard cases.

## 9. Leakage audit

- One row per user → customer cannot straddle partitions.
- All fitted transformers (KNN imputer, scalers, one-hot, event scaler) see
  `split.train` only; `max_len`, class weights, feature selection, tuning and
  early stopping likewise. Test is read once, post-selection.
- Aggregates exclude the target event; the `final_label` segmenter is
  unsupervised (and turned out near-degenerate: k=2 gave {6073, 2} — the
  feature is there but carries ~no signal).
- Legacy `src/main.py` resampled before CV and reported CV-only numbers —
  those ~0.79 F1 figures are not comparable to anything here.

## 10. Conclusion

The most effective feature engineering was the *sequence itself*: compressing
consecutive identical touchpoints into distinct events and feeding their
order to the network. Permutation importance is unambiguous — permuting
touchpoint order costs 0.17–0.21 macro-F1 while every static group sums to
roughly zero. The engineered aggregates, demographics, temporal channels and
the segment label each contributed noise rather than signal to this model.

The encoding strategy that survived is deliberately boring: learned
embeddings for the 20 touchpoint codes (the only high-cardinality nominal
input), one-hot for low-cardinality nominals, MinMax for ordinal codes,
standard/robust scaling for continuous and skewed counts, passthrough for
binary flags. No blanket label encoding, no target encoding — and the
feature-selection filter removed three redundant aggregates before they
could double-count signal.

Data fusion did **not** improve the model. On validation the fused variant
scored 0.2687 against 0.2832 for sequence-only, and on test the untuned
fused model dropped to 0.210 F1 vs 0.261. The static block is near-uninformative
for this target — demographics describe *who* the customer is, but "which
channel did they just use" is almost entirely a *what* question. Fusion was
implemented, measured, and then honestly rejected by the selection rule
rather than kept because it was built.

LSTM is *suitable but not superior* for this data. It trains stably, respects
padding via packed sequences, and produces the best macro-recall of any model
tested — evidence it does learn sequence structure. But on the headline
contract it loses to the GBM (0.22–0.26 F1 vs 0.27; 0.71–0.74 vs 0.89 top-3).
The likely cause is that this dataset's predictive mass sits in the last two
touchpoints and the two dominant classes, which a tabular model on
step2/step3 captures directly; sequence depth mainly helps on the 100+-event
journeys where the LSTM beats its own average (0.52).

The hyperparameters that mattered most were bidirectionality and capacity:
the only two trials clearing 0.25 val F1 were bidirectional, and 2-layer
128-unit configs were uniformly worse (0.16–0.25) than 1-layer 64-unit ones —
the dataset is too small to feed deep recurrent stacks. Learning rate 1e-3
beat 3e-4/5e-4 consistently.

The most important feature is touchpoint order/identity (importance 0.17,
an order of magnitude above everything else). Final test metrics for the
selected tuned model: accuracy 0.423, precision-macro 0.224, recall-macro
0.309, F1-macro 0.223, F1-weighted 0.478, top-3 0.710, ROC-AUC 0.772,
PR-AUC 0.247.

The unresolved problem is minority-class blindness: macro-F1 is half of
weighted-F1 because classes 9, 14, 15 are effectively never predicted, and
the model's recall comes from spreading predictions across mid-tier classes
(high recall, low precision on classes 2/5/10/13). Secondary issues: young
customers (<30, accuracy 0.23) and noise-segment users are poorly served,
and the segmentation itself degenerated to k=2 with a two-member cluster.

Plausible causes: the rare classes have 3–14 test members — irreducible
without more data or class merging; journeys under ~40 events carry too
little context; and the reward structure (plain cross-entropy + balanced
weights) still favours hedging toward frequent channels.

Next experiments, in order of expected value: **(1)** focal loss or
class-balanced-softmax to push probability mass onto rare classes; **(2)**
merge/group the <10-member classes or predict channel *families* (the
dataset's natural hierarchy) before fine-grained codes; **(3)** attention
over the sequence or a compact Transformer — the overfitting signature
(train F1 0.34 vs val 0.28 at best epoch) says capacity is being spent on
memorising long prefixes rather than attending to salient events; **(4)**
sequence windowing/session-level hierarchies so medium journeys keep their
context; **(5)** ranking-oriented training (the app consumes top-3 — optimise
a top-k objective, where the LSTM's recall advantage may pay off); **(6)**
revisit or drop the degenerate segmentation. The GBM remains the stronger
production baseline; the LSTM is a valid, leakage-aware sequence benchmark
whose residual value is rare-class recall and long-journey handling.

## 11. Reproduce

```bash
pip install -r requirements-lstm.txt        # adds torch on top of requirements.txt
python -m src.cjps_lstm.cli --data-dir data # full run (~40 min on CPU)
python -m src.cjps_lstm.cli --data-dir data --quick   # smoke: 2 trials, 10 epochs
python -m src.cjps_lstm.finalize --data-dir data      # re-runs only the final stage
python -m src.cjps_lstm.importance_fused --data-dir data  # grouped importance for the fused variant
python manage.py test src.tests.test_cjps_lstm        # 25 tests, ~2 s
```

Artefacts: `best_lstm.pt` (winner checkpoint + config), `preprocessors.pkl`
(imputer, encoders, vocab), `metrics.json`, `model_comparison.csv`,
`tuning_results.csv`, `feature_importance{,_fused,_grouped}.csv`,
`error_*.csv`, `figures/{learning_curves,confusion_matrix,feature_importance,model_comparison}.png`.
