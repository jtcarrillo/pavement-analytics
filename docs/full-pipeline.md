# Full original pipeline with local model training

Use `pavement full` for the original workflow followed by a target sweep. It
executes the original preparation calculations and replaces the retired
DataRobot service with this package's local training, evaluation and reporting.
The separate `study` command uses a different inspection-level population and
lookback exposure contract; it is not required by `full`.

## Install and run

```bash
python -m pip install -e '.[original]'
pavement full "/path/to/private/data" --open
```

Only the ERA5 acquisition stage uses a remote service. Configure the normal CDS
credentials and dataset access on your workstation. DataRobot credentials,
configuration, SDK and network access are not used. Models, predictions, data
and reports remain local in a new private workspace under `~/pavement-runs/`.
The command displays local input/mapping paths, stage progress and basic counts.
Detailed diagnostics and raw subprocess output stay in private logs.
`--open` opens the local progress page when the run starts, then opens the
consolidated dashboard after success. Use `--progress` to open only the progress
page. Every full run writes `progress.html` and `progress.json` in its workspace,
even when no browser is opened.

The original input and auxiliary files are still required. Setup discovers
`pavement-full.json`, then `pipeline.json`, then `pipeline.example.json`, looking
beside the input and in its parent for each name. A fallback profile can live
at `~/.config/pavement-analytics/full.json`. An explicit profile wins:

```bash
pavement full "/path/to/private/data" --config "/path/to/pipeline.example.json" --open
```

Existing profile paths remain relative to that profile. The data argument
overrides its old `input_dir`; output goes into the new private workspace,
without overwriting the old run. The full command enables climate, distress,
PCI and Shapley even if the old profile had skip flags. Those overrides are
audited. Legacy rate-target lists are replaced by the deduct-family sweep.
Legacy `datarobot` settings are recorded as ignored, including `train`.

## Required private inputs

| Input | How it is resolved |
|---|---|
| Inspection exports | The supplied CSV, TSV, XLSX or XLSM file/folder; first Excel worksheet |
| Location mapping | Profile `mapping`, `--mapping`, or discovered `data_file_mapping.json`, `data/data_file_mapping.json`, or `mapping.json` |
| Location CSV directory | Profile `csv_dir`, `--csv-dir`, or the mapping directory |
| CDV curve profile | Profile `cdv_curves`, `--cdv-curves`, or discovered `cdv_curves.json`, `dr-multitarget/cdv_curves.json`, or `data/cdv_curves.json` |
| ERA5 cache | Profile `cache_dir`, `--cache-dir`, or a sibling `original-era5-cache` beside run workspaces |

Discovery uses the profile directory, data directory and data's parent. The
location JSON maps inspection filenames to CSV paths relative to `csv_dir`:

```json
{
  "inspection_export.xlsx": "locations.csv"
}
```

Location CSVs contain `LOCATION(DEGREES)`. Source keys containing a path are
normalized to a filename only when it matches exactly one selected input;
colliding aliases or ambiguous filenames are rejected. Absolute location CSV
paths are accepted only within the configured `csv_dir` and are normalized to
relative paths. Location paths cannot use parent traversal or symlinks. Setup reports
each invalid entry's reason before `INVALID_LOCATION_MAPPING` and saves the
entry decisions under `mapping_entries` in `full_setup_report.json`.
Referenced location CSVs are excluded from inspection discovery. Locations and
the CDV profile are never guessed.
The original digitized CDV table remains a private input and is not distributed
with this public package.

The command snapshots raw exports, normalizes recognized headers and inspection
dates, and privately copies the selected mapping, location files and profiles.
Original files are unchanged. Ambiguous date order requires `--date-format`.
`full_setup_report.json` records hashes, original-to-normalized source mappings,
date/header decisions, missing inputs and forced full-stage settings.
Before processing, setup checks the selected mapping, location directory, CDV
profile and any supplied deduct-curve or family profile. It lists every missing,
wrong-type or unreadable path before `FULL_SETUP_INPUT_PATHS_INVALID`; the
`input_checks` report entries include each role, path and status. Correct the
selected paths in the private profile or CLI options; setup does not search for
replacements after these checks fail.

Keep the ERA5 cache outside the input directory, the run workspace and all Git
checkouts. Safe cache paths in an existing profile are reused. If an inherited
profile points inside a checkout, overlaps the inputs/run/source tree, uses a
symlink, or names a file, setup selects the external default instead. It records
the requested path, selected path and reason under `climate_cache` in
`full_setup_report.json`. The old cache and profile are left untouched; cached
weather is not copied automatically.

An explicit `--cache-dir` takes precedence and must itself be safe. To override
a legacy cache setting immediately:

```bash
pavement full "/path/to/private/data" --cache-dir "$HOME/pavement-cache/era5" --open
```

The cache boundary checks are unchanged. To reuse an existing cache inside an
old checkout, copy it to a **new** destination outside every Git checkout, input
directory and run workspace, then select that copy explicitly. For example,
with generic private paths and an unused destination:

```bash
mkdir -p "/private/examples/pavement-cache"
cp -R "/private/examples/legacy-checkout/cache/era5" "/private/examples/pavement-cache/era5"
pavement full "/private/examples/inspection-data" --cache-dir "/private/examples/pavement-cache/era5" --open
```

This leaves the old cache in place. Use ordinary directories, not symlink aliases.

After correcting a setup failure, start a fresh run; `--resume` requires
completed setup. A compact private profile in the **parent** of the data folder can be:

```json
{
  "mapping": "metadata/data_file_mapping.json",
  "csv_dir": "metadata",
  "cdv_curves": "profiles/cdv_curves.json",
  "cache_dir": "cache/original-era5",
  "climate_start_year": 1980,
  "climate_end_year": 2025,
  "construction_year_col": "Original Construction Year_start",
  "dr_cv_reps": 5,
  "dr_holdout_pct": 20
}
```

The two `dr_` validation keys are accepted for compatibility with existing
profiles; they now configure **local** cross-validation and holdout. They do
not contact the retired service. The full comparison requires a holdout greater
than zero and below 50 percent. Climate years remain configurable; the original
default end year is the previous calendar year. Review coverage warnings for
inspection intervals outside the configured window.

## Stages and outputs

Progress is enabled by default for `pavement full`. Before the numerical stages,
it lists the selected configuration, mapping JSON, location CSV directory and
ERA5 cache. Each original input file is shown with its row count and its mapped
location CSV, or an explicit unmapped/missing-file status. A mapped file means
the referenced CSV exists; coordinate validity is checked by the location stage.

Each stage prints Started and then Completed, Skipped, Failed or Interrupted,
with elapsed seconds and relevant counts. Verified stages print Reused on
resume. Summaries include retained/rejected inspections, pairs, valid/missing
locations, climate coverage, quarantined pairs, PCI agreement, modeling rows,
features, validation groups and model outcome counts. The local sweep also
prints the material and original deduct-family target as each model task starts
and finishes. Failure output identifies the phase and its detailed log; success
prints the dashboard, models and run-report paths.

The same readable output is appended to `logs/progress.log` in the run workspace.
Use `--quiet` for fixed terminal status/error codes while retaining this saved
progress log:

```bash
pavement full "/path/to/private/data" --quiet
```

Progress includes local file paths and aggregate results; keep captured terminal
output with the private run. Raw data rows, credentials, complete configuration
objects and exception details are not forwarded to the terminal. Changing this
display option does not change models or invalidate a verified resume.

### Live progress and existing runs

The local `progress.html` page refreshes while the run is active and shows the
current stage, elapsed time, selected inputs, recent progress messages and time
since the latest signal. It reads no remote resources and needs no HTTP server.
The corresponding `progress.json` retains structured display state.

During ERA5 processing, progress distinguishes location loading, cache reuse,
requests, parsing and feature calculations. Location and inspection-row counters
show how much work has finished. The parent process emits a heartbeat about
every 15 seconds while a subprocess is running, even if the child has printed
nothing. A heartbeat establishes that the child process is still active; it
does not prove that the remote CDS job is advancing.

The request phase covers both the CDS queue and the subsequent download.
The display therefore says "waiting for CDS / downloading" rather than claiming
a precise queue position, transfer percentage or completion time. After weather
loading finishes, calculating climate features for inspection intervals can
also take time; this has its own row counter.

For a pipeline already running, open a second terminal and monitor it without
restarting it:

```bash
# Latest run under ~/pavement-runs
pavement monitor --open

# A particular run workspace
pavement monitor "/path/to/existing/private/run" --open
```

Keep the monitor command running to follow changes. Ctrl+C stops the monitor
only; the separate pipeline process keeps running. Add `--once` to write one
snapshot and exit instead of following the run.

For a run started by an older version, the monitor uses its saved checkpoints
and private stage logs. It can show recorded phase/location activity and log
freshness, but cannot prove the original child is still alive or reconstruct
row counts that were never recorded. Detailed counters and new heartbeats
require a run started with the updated code. Updating the installed code does
not add instrumentation to an already-running process. Code changes also
require a new run under the existing resume fingerprint rules; a new run may
still reuse its external ERA5 cache.

A missing-location warning counts inspection pairs without usable coordinates;
it does not identify a stalled CDS request. Those pairs cannot receive their
location's climate features until the private mapping/location inputs are fixed.
The progress display does not invent coordinates or silently repair mappings.

All stages are recorded in `pipeline/run_report.json` and `run_report.md` with
status, timing, counts, warnings and artifact checksums. Empty surface populations
are explicit. A failed stage is not reported as completed.

| Stage | Work performed |
|---|---|
| `01_split` | Original surface classification, eligibility checks and rejection accounting |
| `02_pairs_<surface>` | Original consecutive-inspection pairing, equal-PCI handling, increases/resets and interval diagnostics |
| `03_location_rate_<surface>` | Original location matching and deterioration-rate calculation |
| `04_climate_<surface>` | Original ERA5-Land worker, cache, grid snapping/fallback, all original feature outputs and coverage audit |
| `05_targets_<surface>` | Distress validity/quarantine, pooling, densities, severity indices, prevalence and family totals |
| `05_fit_curves` | Original optional deduct-curve fitting, or an explicit not-selected record |
| `06_pci_<surface>` | PCI reconstruction, export-gap checks, family PCI/leave-one-out/Shapley results and agreement metrics |
| `07_modeling_<surface>` | Original predictor filtering with complete per-column audit and per-target eligibility |
| `07_partitions` | One shared validation plan and partitioned modeling copies |
| `08_model_backend` | Select the local engine and record retirement of the old service |
| `09_local_models` | Separate local candidate searches, predictions, feature importance and climate ablation per target |
| `10_report` | Consolidated offline dashboard, target rankings, scores, predictions and importance CSVs |
| `11_qualification` | Verify completed-stage hashes, source inputs, shared groups, original targets/weights and reloaded model predictions; record pending release evidence |

Curve fitting remains an original configuration choice: use a supplied private
`deduct_curves` file, or enable `fit_deduct_curves`/`--fit-deduct-curves`; they
are mutually exclusive. This fitted deduct profile is separate from the
required CDV correction profile. The fitting stage requires enough valid
quantity/deduct samples. When fitting on the run population, reconstructed
curve-dependent outputs stay in the analysis tables but are excluded from
model predictors so held-out observations cannot influence a learned feature.

## Preserved definitions

The original pairing population and interval weights are retained. These are
end-inspection deduct-family totals, named `fam_<family> deduct_end`; they are
not annualized changes. The original 11 family definitions are retained unless
a private `families` profile replaces them. Recorded deterioration rates are
still calculated as an original stage output, but are not the sweep's targets.

The ERA5 worker emits its original **37 output fields**: interval and source
metadata plus thermal, PG/LTPP, cooling, rainflow and construction-aging results.
Its local-solar-day handling, units, normal windows, interval inclusivity,
snapping and fallback behavior are preserved. There is no replacement 365-day
lookback. Construction-aging results require the configured construction-year
field; missing values remain missing with warnings. Other climate features may
also be unavailable when their required history is missing.

These original climate calculations combine inspection-interval, historical-PG
and construction-aging windows. In particular, forward cooling events may use
samples after the interval endpoint. This is a **retrospective association
study**, not an evaluation of inputs available for a future forecast. Feature
importance does not establish that climate caused the observed distress.

The original structural-zero treatment of exported blank distress cells is
retained. Malformed values are quarantined. Simultaneous positive ungraded and
graded channels for one distress code are also quarantined to avoid an unknown
double count. A family with no deduct channels in a source export stays missing
for that source. Partially exported families retain their observed sums and are
flagged; these may represent different coverage across sources. Review
`01_split/source_family_schema.csv`, `05_targets/<surface>_family_target_coverage.csv`,
export-gap checks and target exclusion audits before interpreting results.

The original numeric calculations have generated-input parity checks against
the recovered implementation. Deliberate integration changes are limited to
privacy/boundary guards, robust header/date setup, explicit blank/malformed
handling, ambiguity quarantine, target-specific availability, retained constant
targets, shared validation, and local training/reporting. Those checks do not
constitute standards certification or validation on operational data.

## Local models and interpretation

Observed-level `full` and `study` searches enable the R7 comparison/runtime path
automatically. Primary CV-MAE selection and all original preparation stages
are preserved. The library call to `search_levels` without `model_options`
retains its established in-process execution path.

Each material/family task compares a weighted median baseline, two ElasticNet
settings, two random-forest settings and four ExtraTrees settings. ExtraTrees
uses 200 trees, leaf sizes 2/10, maximum depth 12/unlimited and one worker.
It needs no additional dependency. The four configurations add 40 training
fits per target with five CV folds and two feature sets, plus selected refits.
Installing `.[original,xgboost]` adds eight optional CPU XGBoost settings:
depth 3/6 and learning rate 0.05/0.1, each with a fixed 300-round budget or
20-round patience. XGBoost is version-bounded to `>=3.1.3,<3.2`.

Stopped candidates reserve 20% of the current training fold's groups for
stopping. Preprocessing is fitted on the remaining fitting groups and applied
unchanged to stopping inputs. The chosen iteration count is then used to refit
fresh preprocessing and the model on that entire training fold. Final
development fitting repeats this same nested policy. Outer CV evaluation
groups and holdout groups never supply stopping data. Both fitting and stopping
receive their matching interval weights. Selection still uses mean CV MAE.

`leaderboard.csv` and the consolidated model comparisons include `cv_iterations`,
`cv_training`, `final_iterations` and `final_training`. The training JSON fields
record fitting/stopping/refit groups, row and weight totals, actual model
iterations and probe iterations. `candidate_audit.json` retains these details
for every attempted fit, including unsuccessful candidates. Group identities
stay in private exports. The eight XGBoost settings add 80 CV model fits and
40 stopping probes per five-fold target, plus selected refits. Unavailable,
unsupported or unloadable backends have explicit statuses; base models continue.

The optional `lightgbm` extra (`>=4.6.0,<4.7`) adds eight more CPU configurations:
15/31 leaves, minimum leaf observations 10/30, learning rate 0.05, and up to
300 rounds, with fixed or grouped-stopping variants. They reuse the same
training-only split/refit policy and interval weights. Fresh native callbacks
use patience 20 and weighted MAE. CPU fits use `n_jobs=1`, deterministic mode
and column-wise histograms. Inputs follow the existing numeric/one-hot path;
native categorical support is not introduced.

LightGBM audits retain raw feature order, neutral encoded column names,
transformed feature names, category and infrequent-category mappings, for both
the stopping probe and final refit. These mappings remain private. Native
LightGBM can finish below its requested round limit when no split is useful;
`iterations` records the actual count and `requested_iterations` the limit used
for final fitting. Probe evaluation rounds and best retained model rounds are
separate. The eight configurations add 80 CV model fits and 40 stopping probes
per five-fold target, plus selected refits.

Four additive spline candidates are included in the base installation: 5/8
uniform knots, cubic univariate bases and weighted Ridge alpha 0.1/1.0. Each
numeric predictor has its own curve and a separate linear missingness indicator.
Constant/all-missing numeric predictors skip spline expansion. Categorical
terms use one-hot encoding; reviewed subgrade, branch-use, construction and
code fields stay categorical even when represented by numbers. Declared pandas
categoricals and booleans are also preserved. No interaction basis is added.

Spline bounds, medians, knots, categories and coefficients are learned within
each training fold. Values beyond training support use constant extrapolation.
Private training details record roles, bounds, knot vectors and feature blocks.
The saved estimator's `predict_components(X)` returns an intercept and feature
terms for its unclipped margin; `predict_margin(X)` reconstructs that sum.
`predict(X)` then applies the shared nonnegative floor, which can alter strict
additivity at the boundary. These terms describe predictive associations and
can share effects when predictors are correlated. This is additive spline
regression with an L2 coefficient penalty, not a pyGAM smoothing implementation.
It adds 40 CV fits per five-fold target, plus selected refits. Closed-form Ridge
fitting has no boosting iteration count; its training policy is recorded as
`closed_form_weighted_ridge` with null iterations.

Four base-install occurrence/severity candidates pair weighted logistic
occurrence (C 0.1/1.0, lbfgs, tolerance 0.0001, at most 1,000 iterations) with
positive-only ExtraTrees means (200 trees, depth 12, leaf size 2/10, one worker).
Occurrence preprocessing learns from all current training rows; severity
preprocessing learns only from positive training rows. Both stages receive
their corresponding interval weights. Mixed-class classifier fitting needs
at least five rows and two groups in each class. Severity fitting needs ten
positive rows and two positive groups. Otherwise, training-weighted prevalence
and positive means supply explicit fallbacks. No positives means zero prediction;
all positives means probability one; a constant positive magnitude stays constant.
The global development-constant task shortcut still evaluates the median alone.

`predict_occurrence_severity(X)` returns `p_positive`, `positive_mean` and their
product `prediction`, plus training-only constant references. The product
estimates the overall conditional mean, without a probability threshold. It is
not the mixture median and can lose MAE to a zero baseline. Primary selection
continues to use mean development-fold MAE. Independently for each feature set,
the best complete two-stage CV candidate is refitted on development data and
saved as `two_stage_all_model.joblib` or `two_stage_nonclimate_model.joblib`,
even when another family wins overall. Failed diagnostic refits remain visible
and the next complete CV candidate is tried, without changing primary selection.

Private `two_stage_predictions.csv` files retain component OOF rows for each
attempted candidate and holdout rows for that one within-family diagnostic.
`two_stage_metrics.csv` reports pooled saved-weight Brier score/skill, clipped
log loss, trapezoidal PR-AUC, average precision, overall mean-prediction
MAE/RMSE and positive-only severity errors. Single-class ranking metrics and
errors without positive evaluation rows are undefined with explicit statuses.
Mean RMSE skill uses each fitting partition's weighted overall mean baseline;
Brier uses its weighted prevalence. These pooled metrics differ from the
mean-fold MAE used for selection. The dashboard and consolidated CSVs include
the diagnostics. R7 adds family and component explanations as described below.
The four candidates add 40 CV
fits per five-fold target, each containing up to two native fits, plus refits.

Imputation, numeric scaling and
categorical encoding are learned within training folds. Model selection uses
cross-validation; the selected model is fitted on development groups only and
evaluated on reserved holdout groups. Saved models do not train on holdout.
Original `delta_days` interval weights are used for training and reported errors.
Dependency versions, including optional backend presence/version, now contribute
to the full-run fingerprint and are recorded in its report. Installing or
changing XGBoost requires a new run; its external ERA5 cache can still be reused.

All targets reuse the same group assignments. Declared source/airfield groups
sharing identical supplied coordinates are joined before splitting; this does
not guarantee independent ERA5 cells after nearby-grid snapping. Configure a
stable actual group identifier when several exports belong to the same site.
Targets retain their own valid observations rather than requiring complete
cases across every family. Missing, constant, sparse or unsupported tasks remain
visible with reasons instead of stopping the whole sweep.

The predictor audit excludes endpoint outcomes, other targets, identity fields
and derived deltas. Prior inspection condition and reviewed original start-state
features remain available. `keep_distress` and `keep_latlon` retain their original
explicit meanings. Counts fitted from the entire run are not silently presented
as training-fold preprocessing.

The dashboard ranks baseline-relative holdout MAE skill and shows R-squared,
MAE/RMSE, prevalence, group support and prediction plots. Raw MAE is not used to
rank differently scaled targets. The same model search without climate measures
its added value, and repeated holdout permutations rank individual features and
the climate block. Correlated features can share importance; repeat variability
is not a sampling confidence interval. A many-target ranking is exploratory and
needs confirmation on independent data.

The final dashboard and consolidated CSVs are in `pipeline/10_report/`.
The dashboard links to `pipeline/11_qualification/qualification.html`, which
summarizes integrity and saved-model checks, target support, fitting platform,
dependency versions, climate acquisition counts and remaining release gates.
This final stage is automatic; no additional data-path argument is required.
Separate model bundles, CV scores, full predictions and importance tables are
in `pipeline/09_local_models/materials/<surface>/<target>/`. `target_map.csv`
preserves the mapping from safe directory names to original target column names.
No plotting/modeling service is needed to view the report.

## Resume and diagnostics

```bash
pavement full --workspace "/path/to/existing/private/run" --resume --open
```

Resume verifies original files, snapshots, relevant settings, code and completed
artifacts. Changed data/settings/code require a new run. Failed stages retry;
completed original stages and local models are reused. Failed model/report
attempts are preserved privately before retrying. An interrupted original
climate stage can also reuse its weather cache.

Use `full_result.json` for the dashboard location after success. Review
`pipeline/run_report.md`, `pipeline/warnings.csv`, individual stage logs and
`logs/progress.log` and `logs/runtime.log` for diagnostics. Missing private curve/location inputs fail
setup explicitly. Live CDS access still requires acceptance testing on the
deployment workstation; automated tests substitute generated weather and never
download private-location climate data or contact a training service.

## Model runtime and task checkpoints

The default model path uses one spawned process and one native estimator thread.
It runs candidate/fold work sequentially, records wall time and worker CPU time,
and enforces a 300-second deadline for each fit or prediction operation. A timed
out worker is terminated, its candidate receives an explicit status, and the
next eligible model can run in a fresh worker. Native diagnostics stay in the
private task's `model_worker.log`; progress shows fixed model IDs, feature sets,
folds and fit-call counts. File/output failures abort the task.

A private full-pipeline JSON profile, or study options profile, accepts:

```json
{
  "local_model_options": {
    "comparisons": true,
    "isolated": true,
    "fit_timeout_seconds": 300,
    "prediction_timeout_seconds": 300,
    "max_fit_calls": null,
    "candidate_ids": null,
    "cache_dir": null
  }
}
```

Merge this object into the existing profile. `candidate_ids` can be a list of
implemented, available registry IDs, while the median baseline is retained.
`max_fit_calls` rejects a sweep whose conservative fit-call upper bound exceeds
the supplied positive integer. The pre-fit `model_fit_budget.json` includes
target counts, outer fits, worst-case selection/refit fallbacks and native
stopping/two-stage fits. It is an upper bound, not a claim every fit ran.
The per-task `fit_budget.json` and `model_runtime.csv` retain planned and actual
work. Wall timing includes transfer/startup overhead; worker CPU timing covers
the native operation. Refits reused from primary/R6 diagnostics are labeled
with their source and original fit cost. Explicit `isolated: false` is a debug
mode and disables hard process deadlines/native thread enforcement.

Every complete CV candidate is refitted on development data, serialized into
`candidate_models/<feature_set>__<candidate>.joblib`, reloaded, and checked
against its saved holdout predictions. Model paths in CSVs are relative to the
material/target task directory. Existing primary model and prediction files
remain available. For each family, all-features and nonclimate representatives
are chosen independently by development CV MAE; holdout outcomes do not choose
representatives. Failed refits, unavailable backends, partial CV and constant
shortcuts retain explicit statuses. Importance failures have separate statuses.

Family importance uses the saved fitted pipeline and unchanged holdout weights.
Five repeated joint permutations cover individual predictors, the whole climate
block, and connected training-only numeric Spearman blocks at absolute
correlation at least 0.8. The CSV records each block's member columns. Two-stage
models additionally measure occurrence Brier loss and positive-only severity
MAE. Support/improvement flags are heuristics for exploratory predictive reliance;
repeat SD is permutation variability, not a confidence interval. Correlated
predictors can share importance, and none of these comparisons establish climate
causation. The all/nonclimate comparison includes independent model selection.

| Private output | Contents |
|---|---|
| `candidate_comparisons.csv` | Every CV attempt, final status, parameters, time, support, scores and model path |
| `candidate_predictions.csv` | Holdout predictions and original weights for every completed candidate |
| `family_summary.csv` | Development-selected representatives, holdout skill, climate ablation and explanation status |
| `family_importance.csv` | Family and component predictor/block reliance, repeat SD and support flags |
| `model_runtime.csv` | Candidate/fold/operation statuses and timings |

Each output exists per task and is consolidated into the final report. Old
reports without these sidecars remain readable. The dashboard has family
comparison tables and separate predictor/component rankings; complete CSVs
retain the rows omitted by the compact display.

Ctrl+C stops the active worker. Only complete targets are reusable; partial
CV/fits are retained privately for diagnosis and are rerun. Full runs keep
content-keyed tasks under `pipeline/09_model_tasks`, independent of the
`09_local_models` stage's atomic publication. Use the existing workspace with
`pavement --workspace "/path/to/run" full --resume --open` to resume. Earlier
original-stage input/config/code/dependency guards still apply.

An explicit model `cache_dir` is resolved relative to its private profile and
must be outside version control and the public source tree. Study runs can
reuse this cache across new workspaces. Task keys include data/order/dtypes,
shared plans, features, target, weights, seed, candidate definitions, options,
registry/code hashes, Python/platform and dependency versions. Every artifact's
checksum and the exact output set are verified before reuse/copy. Changed or
missing artifacts reject reuse; changing inputs/settings creates a new key.
One foreground writer per workspace/cache is supported. No concurrent fitting
mode is enabled by default.
