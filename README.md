# CJPS — Customer Journey Prediction System

Django web application that segments customer profiles and predicts the next
three marketing touchpoints in a travel customer's journey.

Two stages, both running in memory:

1. **Segmentation** — DBSCAN rejects profiles that sit outside every core
   neighbourhood (the *noise* cluster), then Spectral Clustering assigns the
   rest to one of three segments.
2. **Next-touchpoint ranking** — a Gradient Boosting classifier scores all 20
   touchpoints and the top three are returned with names and descriptions.

---

## Quick start

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt

python manage.py migrate          # creates the DB and seeds all reference data
python manage.py build_demo_models   # writes a working demo model set to ./models
python manage.py createsuperuser
python manage.py runserver
```

Open http://127.0.0.1:8000/.

`build_demo_models` fits the real pipeline on synthetic data so the prediction
flow works offline. Its output is **demo data** — plausible, not meaningful.
Rebuild the genuine artefacts from `src/main.py`.

### Running the tests

```bash
python manage.py test
```

The suite needs no network access and no model artefacts: it fits small real
scikit-learn estimators as fixtures (`ctmj/tests/factories.py`).

---

## Configuration

All settings are environment-driven; see `.env.example` for the full list. Copy
it to `.env` for local overrides.

| Variable | Default | Purpose |
|---|---|---|
| `DJANGO_SECRET_KEY` | ephemeral in DEBUG | **Required** when `DJANGO_DEBUG=False` |
| `DJANGO_DEBUG` | `True` | Set `False` in production |
| `DJANGO_ALLOWED_HOSTS` | `127.0.0.1,localhost` | Comma-separated |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | empty | e.g. `https://cjps.example.com` |
| `CTMJ_MODEL_SOURCE` | `auto` | `auto` \| `local` \| `hf` \| `none` |
| `CTMJ_MODEL_DIR` | `./models` | Local artefact directory |
| `CTMJ_HF_REPO` | `quanghuynh0122/datamining_models` | Hugging Face repository |
| `HF_TOKEN` | empty | Read token, only for `CTMJ_MODEL_SOURCE=hf` |
| `DJANGO_LOG_LEVEL` | `INFO` | Application log level |

`auto` prefers `./models` when all five artefacts are present, otherwise falls
back to the Hub if `HF_TOKEN` is set, otherwise runs with predictions disabled
and explains why in the UI. **There is no interactive prompt anywhere** — see
[Why model loading changed](#why-model-loading-changed).

### Model artefacts

Five joblib files, named by `CTMJ["MODEL_FILES"]` in `ctmj/settings.py`:

| Key | File | Fitted on |
|---|---|---|
| `user_data_preprocessor` | `user_data_preprocessor.pkl` | 10 clustering columns |
| `dbscan` | `dbscan_clustering_model.pkl` | scaled clustering matrix |
| `spectral` | `spectral_clustering_model.pkl` | DBSCAN non-noise rows |
| `predicting_preprocessor` | `s1_predicting_preprocessor.pkl` | 8 prediction columns |
| `gradient_boosting` | `GradientBoostingClassifier_model.pkl` | scaled prediction matrix |

The column contract lives in `ctmj/services/formdata.py` and must match
`src/main.py`. Changing a name there without retraining breaks inference.

### Operations

- `GET /health/` — readiness probe. Returns **200** when the artefacts are
  loaded, **503** while loading or missing, with a JSON breakdown. Point your
  orchestrator or load balancer here.
- `GET /admin/` — browse the reference tables and the prediction audit log.

---

## Project layout

```
ctmj/
  settings.py            env-driven, refuses to boot insecurely in production
  apps.py                builds the registry, defers loading for CLI commands
  models.py              reference/lookup tables + PredictionRun audit log
  views.py               thin HTTP layer only
  urls.py                /, /predict/, /health/, handler404
  services/
    reference_data.py    single source of truth for every dropdown
    registry.py          thread-safe, failure-tolerant artefact loader
    formdata.py          validation against the real lookup tables
    predictor.py         the two-stage inference pipeline
  templatetags/
    ctmj_tags.py         per-field errors and locale-safe number formatting
  management/commands/
    seed_reference_data.py   re-sync lookup tables
    build_demo_models.py     generate offline demo artefacts
  migrations/0011_seed_reference_data.py   fresh clone -> migrate -> works
  tests/                 121 tests, no network, no artefacts required
src/                     research pipeline (EDA, preprocessing, training)
static/css/main.css      design tokens + components, no framework
static/js/app.js         progressive enhancement only
templates/               base, home, predict, 404, partials/
```

Reference data lives in a **data migration**, not a committed SQLite file, so a
fresh clone needs nothing but `migrate`. `db.sqlite3` is git-ignored.

---

## Design

Editorial data-journalism: warm paper ground, one oxblood accent, Fraunces
against Inter, asymmetric editorial spreads, `tabular-nums` on every figure.

Deliberately absent: gradients, glassmorphism, centred three-card feature rows,
`window.alert`. Every value is a CSS custom property in `:root`, so the palette
retunes from one block.

Accessibility is handled rather than assumed: a skip link, visible focus rings,
inline field errors wired through `aria-describedby`, an error summary that
receives focus on submit, `role="img"` probability tracks with text equivalents,
decorative icons marked `aria-hidden`, and full `prefers-reduced-motion`
support.

The form and results work with JavaScript disabled. `app.js` only adds an
animated submit state, animated probability counters, and stepper buttons.

---

## Notes for maintainers

### Why model loading changed

The original `apps.py` called `input()` from `AppConfig.ready()` to ask for a
Hugging Face token, guarded by `RUN_MAIN == 'true' or not DEBUG`. That guard is
false during ordinary local development, so models were never loaded under
`DEBUG` and *every* prediction failed. In production `input()` hung or crashed
on EOF. The feature was dead in both modes.

`ModelRegistry` now reads credentials from the environment, loads on a worker
thread so the first response is never delayed, and records failure as state
rather than raising. `ready()` assigns the registry through the **class**, not
`self` — an instance attribute would leave `CtmjConfig.registry` (the name the
views read) as `None`.

### Two index-space bugs in the cluster pipeline

Both produced plausible-looking but meaningless segments, and neither raised.
`ctmj/services/predictor.py` documents them in full; the regression tests are
`test_regression_*` in `ctmj/tests/test_predictor.py`.

1. **Spectral label lookup crossed two index spaces.**
   `dbscan.core_sample_indices_` indexes the *full* training set, while
   `spectral.labels_` indexes the *noise-filtered* subset (`src/main.py` fits
   `SpectralClustering` on `X_scaled[dbscan_labels != -1]`). The old code used a
   full-set index against `labels_`. On scikit-learn ≥ 1.6
   `SpectralClustering.components_` no longer exists, so that branch was
   unreachable and the broken fallback branch was always taken.

2. **The noise gate could never fire.** DBSCAN core samples are never noise, so
   `labels_[core_sample_indices_[...]]` is always ≥ 0. The correct test is a
   distance comparison against `eps`.

### Locale and number formatting

`LANGUAGE_CODE` is `vi`, where Django renders floats with a comma. That is
correct on screen (`41,2%`) but breaks machine consumers: `parseFloat("15,50")`
returns 15, and `width: 15,50%` is invalid CSS that silently leaves the
progress bars empty. Values consumed by script or style are therefore
pre-formatted in Python (`ChannelPrediction.probability_value`) rather than in
a template filter — `unlocalize` is a no-op on the strings `floatformat`
returns. `test_animated_values_use_a_dot_decimal_separator` guards this.

### Template comments

Django's `{# ... #}` only matches within a single line. A multi-line `{# ... #}`
block is **not** a comment: Django consumes up to the first `#}` and emits the
remaining lines as page text. Use `{% comment %}`. Guarded by
`test_template_comments_do_not_leak_into_the_page`.

### The 404 page

The branded 404 activates only when `DJANGO_DEBUG=False`, because Django renders
its own technical 404 under `DEBUG`. The test suite runs with `DEBUG` forced off,
so it is covered there.

---

## Data

All customer data in this repository is synthetic. `src/` contains the research
pipeline (EDA, preprocessing, model benchmarking, training); its outputs under
`resources/profiling/` are git-ignored because they are regenerable and large.

The research code retains some `print()` calls and interactive
`asking_window()` prompts. It is a notebook-driven pipeline and is not wired into
the web app.
