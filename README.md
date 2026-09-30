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

294 tests, about 90 seconds. No network access and no model artefacts required:
the app tests fit small real scikit-learn estimators as fixtures
(`ctmj/tests/factories.py`) and the training tests generate their own source
CSVs (`src/cjps_train/fixtures.py`).

The default `manage.py test` found only 143 of those. `src/` has no
`__init__.py`, so it imports as a namespace package and `unittest`'s directory
discovery skips it — quietly, because the modules themselves import fine.
`ctmj/testrunner.py` now discovers both roots explicitly and **fails** if either
collects nothing, so a moved or renamed test directory cannot present as a
passing run. To run one root: `python manage.py test ctmj` or
`python manage.py test src`.

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
| `CTMJ_HF_REPO` | `quanghuynh0122/datamining_models` | Hugging Face **storage bucket** |
| `HF_TOKEN` | empty | Optional. The bucket is public; a token is only needed for a private one |
| `DJANGO_LOG_LEVEL` | `INFO` | Application log level |

`auto` prefers `./models` when all five artefacts are present, otherwise falls
back to the Hub if `HF_TOKEN` is set, otherwise runs with predictions disabled
and explains why in the UI. **There is no interactive prompt anywhere** — see
[Why model loading changed](#why-model-loading-changed).

`CTMJ_HF_REPO` must be `{owner}/{bucket}`. A bare name has no owner and the URL
would name the wrong thing; the registry rejects it at startup rather than
letting a 404 discover it.

> **Rotate any HF token pasted into a chat, terminal log or issue.** `HF_TOKEN`
> is read from the environment and is never written to the repository, but a
> token shared in plain text should be treated as compromised. Check a token
> without exposing it: `curl -H "Authorization: Bearer $HF_TOKEN"
> https://huggingface.co/api/whoami-v2` — an expired one answers
> `{"error":"User Access Token \"...\" is expired"}`.

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
`src/cjps_train/config.py`. Changing a name there without retraining breaks
inference — which is why a contract test asserts the two lists are identical,
and why a training run re-reads its own artefacts and drives them through
`predict()` before reporting success.

#### Reading artefacts from Hugging Face

`CTMJ_HF_REPO` may name a **storage bucket** or a **model repository**. They are
addressed differently, and a loader that knows only one fails on the other with
a 404 that names neither the token nor the file:

| | list | download |
|---|---|---|
| bucket | `GET /api/buckets/{id}/tree` | `GET /buckets/{id}/resolve/{filename}` |
| repository | `GET /api/models/{id}/tree/{rev}` | `GET /{id}/resolve/{rev}/{filename}` |

The bucket has **no revision segment**; a repository has one, defaulting to
`main`. Nothing in the request says which applies, so `ctmj/services/hub.py`
probes once and remembers the answer, with the bucket tried first.

`huggingface_hub` cannot address a bucket at all: the id validator accepts
`repo_name` or `namespace/repo_name`, no `repo_type` fits, and `HfApi.repo_info`
raises `HFValidationError` before making a request. The old
`fsspec.filesystem("hf")` path therefore could not have read a bucket under any
circumstances. The HTTP client is spelled out in `hub.py`, and both address
kinds go through the same code path.

Buckets are **read-only mirrors of external cloud storage** — no upload endpoint
exists (eight candidate routes answer 404), the page has no upload control, and
its state carries `canReadRepoContent` with no write flag. Every "upload" string
on a bucket page is the `uploadedAt` column. Publishing artefacts means creating
a model repository.

> **Three corrections, each of which cost a debugging cycle.** The `buckets/`
> segment was twice taken for a Google Cloud Storage convention and "fixed"
> away — it is the bucket namespace. The failure was blamed on the token,
> because `FileNotFoundError: repository not found` is what a bad token looks
> like and also what a wrong address looks like; the bucket was public
> throughout. And a 401 from an *anonymous* request means private-or-absent,
> not "bad token" — the Hub answers 401 rather than 404 when no credential is
> sent, so the probe only reports a credential problem when one was actually
> sent. `status.missing` carries the reason per artefact in a separate `errors`
> field, so a 403, an absent file and a version clash stop being
> indistinguishable.

Set `CTMJ_TEST_HF_NETWORK=1` to run the opt-in tests in
`ctmj/tests/test_hf_path.py` that hit the live Hub.

#### The two artefact sets

| | `buckets/quanghuynh0122/datamining_models` | `quanghuynh0122/ctmj-demo-models` |
|---|---|---|
| Kind | storage bucket, public | model repository, private |
| Contents | 9 files, 471 MB | 7 files, 28 MB |
| Trained on | the real journey dataset | synthetic fixture, 700 profiles |
| Use | the only quotable numbers | proving the loading path works |

The real bucket also carries four
`s2_predicting_preprocessor_cluster_*.pkl` — the per-segment preprocessors
`src/main.py` fitted and that the pooled design in `src/cjps_train/` replaced.
Its `spectral_clustering_model.pkl` is 465 MB on its own, because
`SpectralClustering` persists its n-by-n affinity matrix (7 807 rows here).

Point the app at either:

```bash
# real artefacts (public bucket, no token needed)
export CTMJ_MODEL_SOURCE=hf
export CTMJ_HF_REPO=quanghuynh0122/datamining_models

# demo artefacts (private repository, token required)
export CTMJ_HF_REPO=quanghuynh0122/ctmj-demo-models
export HF_TOKEN=...
```

#### The trained artefacts and scikit-learn

On scikit-learn 1.9, **four of the five real artefacts load and the classifier
does not**:

```
AttributeError: Can't get attribute '__pyx_unpickle_CyHalfMultinomialLoss'
on <module 'sklearn._loss.loss'>
```

The pickle was written by a Cython build that emitted `__pyx_unpickle_*`
symbols; scikit-learn 1.9 emits none. Registering a top-level `_loss` alias in
`sys.modules` bridges the module name but not the missing symbol, so this is
not shimmable — the artefact has to be either loaded under the scikit-learn
version that wrote it, or retrained. The preprocessors load with an
`InconsistentVersionWarning` (written by 1.8.0) and work; the spectral model has
no `components_`, which is the scikit-learn 1.6 removal that
`ctmj/services/predictor.py` already compensates for.

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
  testrunner.py        discovers both test roots; fails if one is empty
  tests/               152 tests, no network, no artefacts required
src/
  cjps_train/          the production training pipeline (see below)
  main.py              the original research script, kept for reference
  pre_processing.py    SMOTE / Tomek / ADASYN, fold-safe wrappers
  eda.py, dataset/     profiling helpers and the Figshare loader
  tests/               142 tests for everything under src/
static/css/main.css      design tokens + components, no framework
static/js/app.js         progressive enhancement only
templates/               base, home, predict, 404, partials/
```

Reference data lives in a **data migration**, not a committed SQLite file, so a
fresh clone needs nothing but `migrate`. `db.sqlite3` is git-ignored.

---

## Performance

### Reference data is cached

The lookup tables hold 91 rows between them and are populated once, by a data
migration or `manage.py seed_reference_data`. They were nevertheless re-read on
every request: `validate_prediction_form` issued 11 queries and the dropdown
context another 8, all for the same static rows, on the critical path of the
one request that has to feel instant.

Measured end to end through the WSGI handler, 30 requests per case, rotating the
cache before every request in the "before" column to reproduce the old
behaviour exactly:

| Request | Before | After | |
|---|---|---|---|
| `GET /` | 1.42 ms, 1 query | 1.02 ms, 0 queries | −29% |
| `GET /predict/` | 6.59 ms, 8 queries | 4.35 ms, 0 queries | −34% |
| `POST /predict/` | 9.71 ms, 19 queries | 4.61 ms, 0 queries | **−53%, 2.1×** |

Validation and dropdown population together: 4.462 ms → 0.333 ms.

`ctmj/services/lookups.py` owns the cache. Invalidation is a version integer
rotated on write, so it is O(1) regardless of how much is cached, and it is
triggered by `post_save` / `post_delete` on the reference models — an edit in
`/admin/` takes effect on the next request. `seed_reference_data` invalidates
explicitly, because `bulk_create` and `queryset.update` bypass signals.

A cache is only worth having if it is correct, so a failure of the cache
backend degrades to reading the database rather than raising: losing the cache
costs queries, raising costs the request. The signalled invalidation is
registered *before* the model-loading deferral check in `apps.py`, because every
process that can write a lookup row — a test, a management command, a worker —
is one that defers loading.

### What is left

The remaining 4.3 ms on `GET /predict/` is template rendering: eight dropdowns
of `<option>` elements, plus the form. There are no database queries left on the
request path. A profile of `POST /predict/` shows `predict_view` accounting for
essentially all of it, with no single dominant callee — the cost is spread
across Django's template engine rather than concentrated anywhere worth
attacking.

The inference path itself was measured and left alone. `assign_cluster` costs
1.2 ms, almost all of it the noise gate's `cdist` against 7 275 stored core
samples. That is a single query against 7 275 points, where brute force is
optimal — a KD-tree would be *slower* for one point. It is not worth
optimising, and "optimising" it would be a regression.

---

## Training performance

### What was changed

**The neighbour adjacency is no longer quadratic.** `kneighbors` was called
with `n_neighbors=n` — an n-by-n index array — and then the self-match was
dropped in a Python loop over every row. To get k neighbours it now asks for
`k + 1` and compacts with a stable argsort, so the result is n-by-k whatever
n is:

| n | old index array | new | |
|---|---|---|---|
| 2 000 | 30.5 MiB | 0.09 MiB | 333× |
| 8 000 | 488.3 MiB | 0.37 MiB | 1 333× |
| 20 000 | 3 051.8 MiB | 0.92 MiB | 3 333× |

This runs per minority class, per fold, per candidate in the benchmark grid, so
it is squarely in the training hot path.

**DBSCAN fits in parallel.** `n_jobs=-1` on the noise gate: 0.723 s → 0.290 s
on 8 000 rows. Fit-time only — inference compares one profile against the
stored core samples.

### What was measured and then *not* changed

The segmenter's dense RBF affinity is 0.45 GiB and a 465 MB artefact at the
real dataset's 7 807 rows, and it is quadratic. A sparse k-nearest-neighbour
graph is 3.3× faster, 532× less memory, and makes the artefact 12× smaller
(73 MiB → 6 MiB end to end).

It was very nearly shipped as the default. On well-separated Gaussian blobs the
two produce **identical** partitions — ARI 1.000 at k=10, checked across three,
four and five segments. On the pipeline's own feature space, twenty scaled
ordinal codes, they do not:

| k | 10 | 15 | 20 | 30 | 50 | 80 | 120 | 200 |
|---|---|---|---|---|---|---|---|---|
| ARI vs dense | 0.69 | 0.67 | 0.66 | 0.65 | 0.61 | 0.59 | 0.50 | 0.31 |

Nothing reaches the agreement floor, and agreement falls as k grows. The sparse
graph is a *different* segmentation, not a faster route to the same one.

It is also not obviously worse — its silhouette is +0.34 against the dense
+0.26 on the same rows. That is the trap: a higher internal metric on a
different partition is an argument for the different partition, not evidence
that it is the same one.

So `spectral_affinity` defaults to `"rbf"`, and the sparse option is available
for a deployment that does not need to reproduce the original partition.
`--verify-spectral` fits both and writes the adjusted Rand index to the
manifest, so the trade is recorded rather than assumed. The run that produced
the table above warned and recorded ARI −0.045 on the fixture, where the
partition sizes differ visibly: 1986/995 dense against 2433/548 sparse.

One measurement worth recording because it was wrong on the first pass: the
obvious explanation for the dense path's cost — that computing the affinity is
expensive — is false. `pairwise_distances` is 2% of the fit. The
eigendecomposition is the cost, and it is expensive because the matrix is dense.

---

## Design

**"Signal" — a graphite instrument with one lime accent.**

The brief for this pass was a reference to [OpenStitch](https://github.com/heldernoid/openstitch),
a local-first developer tool. What that project actually shares with this one is
not a palette but a *register*: dense, monospace-forward, precise, dark by
default, built for someone reading numbers rather than browsing. So the system
was rebuilt on the Linear/Vercel convention — a token layer, one accent,
hairline separation instead of stacked cards, monospace reserved for anything
numeric.

Dials: `DESIGN_VARIANCE 6` (offset, not symmetric) / `MOTION_INTENSITY 4`
(CSS-level, purposeful) / `VISUAL_DENSITY 7` (instrument, not marketing).

### What changed, and why

**The overview stopped being a landing page.** Two earlier revisions failed the
same way: a display hero and generous whitespace for what is a tool. It now
leads with the one number an operator needs on arrival — is the model loaded, and
how much of it — and everything else is small. The single large element on the
page is that state word. The other four grid/layout families (fact strip,
segment cards, stage flow, definition list) are each used once, because
repeating a layout family three times on one page is what makes a page read as a
template.

**The segments are named instead of counted.** The page previously rendered the
number `3` and nothing else. It now lists each segment's label, its English
name and its description, straight from `reference_data.CLUSTERS`. An analyst
cannot act on a count; they can act on a segment.

**The artefact table only appears when there is something to explain.** The
loader records a per-file *reason*, because a 403, an absent file and a
scikit-learn version clash all surface as `missing` and each needs a different
remedy. A healthy set therefore gets a compact chip list, and the table
reserves itself for failures.

**The prediction page became a two-column console.** The form sits left and the
result in a sticky right rail, so the profile stays on screen next to the answer
it produced. Previously a submit scrolled the page and the analyst had to
re-read a form they had already filled correctly. Below 1080px the rail drops
under the form, because two 190px columns of selects are not a layout anyone can
use on a laptop in a train.

**The form is two columns, explicitly.** Not `auto-fit` over a min-width, which
looks tidier in a mockup and worse in use: at this column width it lands on
three, leaving the tenth field alone on the last row and squeezing the income
bands into ~190px.

**The error summary links to the field.** `error_summary` is a flat list of
strings, which is right for an `aria-live` region and useless for navigation.
`error_items` adds the label, the message and the id of the inline error, so the
summary can jump straight to the control that is wrong.

**Dark and light both ship.** Dark is the default because the reference is a dark
instrument and a researcher stares at this for hours; light is a full peer, not
an afterthought, and follows `prefers-color-scheme` until the user overrides it.
The accent keeps its fill value in both modes (a lime button carries near-black
text at 11.6:1 light and 13.1:1 dark) while its *text* role darkens, because
lime on white is 1.9:1 and unusable as type. A per-mode token value, which is
what real systems do, rather than shipping one hex that fails in one mode.

### Colour

One accent: a signal lime. The subject of this tool is *measurement*, so a single
high-luminance hue is reserved for measured values — lime means "this is a
number the model produced", nothing else. Everything else is a neutral graphite
ramp. Two semantic colours sit alongside it (danger, warning); they are
functional rather than decorative, because a form error that is not red is not an
error, and they are never used for emphasis.

The previous revision used a warm-paper palette (`#f7f4ee` and family). That
family is the single most recognisable default in AI-generated design, and it
made a data tool look like a brochure. Replaced, not adjusted.

Contrast is measured, not eyeballed. Every ink value clears WCAG AA against its
own surface: `--ink` 15.4:1, `--ink-2` 8.4:1, `--ink-3` 5.2:1 on `--bg` in dark;
18.1:1 / 8.0:1 / 5.0:1 in light. The placeholder was previously drawn at 0.75
alpha, which put it near 2.8:1 — now full strength.

### Type

Self-hosted **Geist + Geist Mono**, latin / latin-ext / **vietnamese** subsets,
97 KiB across six files. Previously linked from `fonts.googleapis.com`, which
meant three things this project cannot accept: the interface is entirely
Vietnamese and depends on the `U+1EA0-1EF9` subset, so a font CDN that serves a
different set renders the whole app with fallback diacritics; a research tool
should not hand every visitor's IP to a third party on page load; and the
cross-origin stylesheet is render-blocking.

All six files are **variable** fonts (`wght` 100–900, verified with fontTools) —
Google returns one URL per subset for every requested weight, so downloading a
file per weight would have downloaded the same bytes four times. Cyrillic and
greek are omitted; no string in the interface or the reference data uses those
scripts.

Only the vietnamese subset is preloaded, and that is deliberate. The other five
are same-origin and are discovered a few milliseconds later from the stylesheet.
The Vietnamese subset is the one whose absence is *visible*: with
`font-display: swap` a missing Vietnamese file means every heading renders in a
fallback and then reflows, and Vietnamese stacks combining diacritics so the swap
shifts the line.

`tools/fetch_fonts.py` regenerates the stylesheet; `static/css/fonts.css` is
generated output and should not be hand-edited.

### Shape

One radius, 3px, on every surface and control. Fully-round pills are used only
for status badges — a rule, not a mixture. Near-square reads as instrument; a
16px radius reads as consumer app.

### Motion, and one bug worth recording

Scroll-reveal is the only page-level animation, plus the result bars and the
probability counters. The first version of this pass **shipped content loss**:
`IntersectionObserver` only fires for elements that scroll into view and the
sweep backstop only ran on a scroll or resize event, so a page that is never
scrolled — a full-page screenshot, a print, a headless render, a reader who
reaches the bottom by keyboard — left every section below the fold at `opacity: 0`.
A blank half-page. It is now bounded by a 2s safety net that reveals anything
still pending, and `@media print` forces reveals visible, because print has no
viewport. Screenshots and print are how this was caught; neither shows up in a
test suite.

The probability bars carry their final width in a `--final` custom property
server-side, so **with JavaScript disabled they are already correct** — verified:
62px / 50px / 42px for 20.1% / 16.0% / 13.5%, identical to the animated result,
with no layout shift. `app.js` only moves them from zero to that value, skips the
movement entirely under `prefers-reduced-motion`, and adds a completion meter
over the twelve required fields plus `Ctrl/Cmd+Enter` to submit and `/` to jump
to the first field that still needs an answer.

One design decision was measured and then **withdrawn**: a reference hairline at
the cumulative top-three position, drawn on each of the three meter tracks. The
three bars are separate channels, not slices of one bar, so a mark at 49.6% on
each of them touched none of them and read as a rendering artefact. It is now a
single stacked bar above the meters that genuinely *is* stacked, with the
remainder and the candidate count in the legend beside it.

Deliberately absent: gradients on surfaces, glassmorphism, centred three-card
feature rows, `window.alert`, decorative photography. Every value is a CSS
custom property in `:root`, so the palette retunes from one block. Every
multi-column layout states its sub-768px fallback in the same rule.

Accessibility is handled rather than assumed: a skip link, visible focus rings
throughout, inline field errors wired through `aria-describedby`, an error
summary that receives focus on submit, `role="img"` probability tracks with text
equivalents, decorative icons marked `aria-hidden`, and full
`prefers-reduced-motion` support.

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

## Training

`src/cjps_train/` is the production training pipeline. `src/main.py` is the
original research script and is kept for reference; it still contains its
`print()` calls and interactive `asking_window()` prompts.

```bash
# Validate the data without training anything
python -m src.cjps_train.cli --data-dir data --dry-run

# Full run: trains, writes the five artefacts, verifies them
python -m src.cjps_train.cli --data-dir data --force
```

Point it at the source CSVs with `--data-dir`, or set `CJPS_DATA_DIR`
(`CJPS_JOURNEYS_CSV` / `CJPS_USERS_CSV` override individually). It writes to
`models/` and `resources/training/training_manifest.json`; both are
configurable. Every run writes a manifest recording the seed, library versions,
data row counts, the full clustering diagnostic, the whole benchmark table, and
the holdout score.

### What changed, and why it matters

The numbers the original pipeline reported could not be trusted. Four defects,
each of which produces a model that trains without complaint while measuring
the wrong thing:

**Resampling ran before the cross-validation split.** SMOTE and ADASYN were
applied to the whole dataset, and folds were drawn from the resampled result.
SMOTE synthesises a minority point by interpolating between a real sample and one
of its k nearest same-class neighbours — so once the whole dataset had been
resampled, a validation row was a candidate neighbour, and the model trained on
a synthetic point derived from the row it was about to be scored against. Every
reported F1 was measuring memorisation. Resampling now happens on the training
slice of each fold, and a test asserts the sampler never sees a validation row.

**There was no holdout set.** Nothing was ever scored on data that had not
already influenced a decision. A stratified split is now taken before model
selection and scored exactly once, at the end.

**The benchmark was decorative.** Six models across five resamplers were fitted
and a table printed, then the code fitted a hardcoded
`GradientBoostingClassifier(n_estimators=100, max_depth=5)` against a hardcoded
`n_clusters=3` regardless of what the table said. The roster is now evaluated,
ranked, and the winner is what gets written — with the numbers that put it
there in the manifest. `eps` and the cluster count are likewise chosen by
searching, not assumed.

**A failing model vanished.** `cross_validate` was wrapped in a `try/except`
that printed and continued, so a model that could not fit simply disappeared
from the results. A missing row reads as "we did not think it was worth trying",
which is a very different claim from "it broke". Failures are now recorded with
their error and sorted last.

Also fixed along the way: `--` was being read as a string rather than as missing
data, which defeated every numeric operation downstream; rows with partial
profiles were dropped outright, making the KNN imputer a no-op while the report
claimed the data was clean; the fitted imputer was discarded instead of saved, so
training and serving could never agree; `SGDClassifier(loss='hinge')` was
benchmarked for a task that requires probabilities; `MLPClassifier` ran
unregularised with convergence warnings globally suppressed; output paths were
relative to the working directory; input paths were hardcoded to one machine.

### Reading the manifest

`resources/training/training_manifest.json` is the record of a run. The
`notes` array is the part to read first — it flags a segmentation with fewer
than two segments, a `k` sitting at the edge of the searched range, a smallest
segment too thin for stable probabilities, and a cross-validated score more than
0.15 above the holdout. A run is not finished until `verification.ok` is true:
the artefacts are re-read from disk and driven through
`ctmj.services.predictor.predict` before the run reports success, so a
training/inference contract mismatch surfaces as a training failure rather than
as a broken form.
