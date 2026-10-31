# Regression model expansion roadmap

Prepared 2026-10-31. Scope: ExtraTrees, XGBoost, LightGBM, Gaussian additive
spline models, and a two-stage occurrence/severity model for the existing
deduct-family sweep. Each target remains a separate regression task for each
pavement surface. DataRobot's proprietary blueprints are not reproduced exactly.

The base observed-level search contains a weighted-median baseline, two Elastic
Net, two Random Forest, four ExtraTrees, four additive spline and four
occurrence/severity configurations. R0 provides the reproducible dataset and
benchmark runner. Installing each optional XGBoost or
LightGBM extra adds eight configurations. Build 1 completes R1 with a versioned
registry and common evaluation/failure contract. Build 2 completes R2 with
ExtraTrees; Build 3 completes R3 with fixed-budget and grouped-stopping XGBoost.
Build 4 completes R4 with encoded, weighted LightGBM and native grouped stopping.
Build 5 completes R5 with weighted additive spline regression.
Build 6 completes R6 with weighted occurrence/positive-severity regression and
separate component diagnostics. Build 7 completes R7 with comprehensive model-family
comparisons/explanations, process deadlines, progress and verified task checkpoints.

**Roadmap progress: 88.9% — 8 of 9 phase gates complete (R0 through R7).** Phases R0
through R8 have equal weight for this count. This is accepted implementation
coverage, not an estimate of elapsed engineering effort. Build publication,
merge status and private deployment acceptance are reported separately.
New algorithm families delivered: 5 of 5. The next build is R8, full-pipeline
release qualification and private deployment acceptance.

## Delivery sequence and dependencies

| Phase | Deliverable | Depends on | Completion gate | Current status |
|---|---|---|---|---|
| R0 | Synthetic data, truth contracts and benchmark runner | Existing local search | Reproducible files, isolated groups and existing-model smoke run | Complete — test kit |
| R1 | Candidate registry and common evaluation contract | R0 | Existing candidates retain their selection/prediction behavior | Complete — Build 1 |
| R2 | ExtraTrees | R1 | Weighted fitting, mixed inputs, serialization and nonlinear fixtures pass | Complete — Build 2 |
| R3 | XGBoost | R1, R2 evaluation pattern | Optional dependency and nested stopping tests pass | Complete — Build 3 |
| R4 | LightGBM | R1, R3 stopping pattern | Weighted validation, encoded category and installation diagnostics pass | Complete — Build 4 |
| R5 | Gaussian additive spline model | R1 | Additive shapes, weights and extrapolation checks pass | Complete — Build 5 |
| R6 | Two-stage occurrence/severity model | R1 and one accepted severity estimator | Zero/positive/sparse training cases and prediction semantics pass | Complete — Build 6 |
| R7 | Consolidated comparisons, explanations and runtime controls | R2–R6 | Every task reports candidates, failures, scores and importance consistently | Complete — Build 7 |
| R8 | Full-pipeline release and private pilot | R7 | Existing workflow regression tests and private acceptance pass | Planned |

ExtraTrees comes first because it uses the existing scikit-learn dependency.
XGBoost and LightGBM follow with explicit dependency and stopping rules. R5 can
be developed once R1 is stable. The two-stage wrapper starts with logistic
occurrence and an accepted positive-severity estimator; it need not wait for
every boosting implementation. Record completion by accepted gate, not by files
written or models merely added to a menu.

## R0 — A common, offline test population

The supplied standard profile has 4,320 model-stage observations: 60 groups,
nine sections per group and eight consecutive intervals per section. Each group
contains PCC, APC and AC_AAC_ACC. There are 540 section histories and 1,440 rows
per surface. Group identifiers and section identifiers are invented. All weather
and pavement predictors are synthetic; no real installation names, coordinates,
CDV tables, inspection records or ERA5 downloads are required.

The shared plan assigns 48 groups to development, 12 to holdout, and uses five
development folds. Every row of a group stays in its assigned partition. Labels
missing for a target are excluded after this assignment. The smaller `smoke`
profile has 180 rows, 15 groups and three folds. `scale` has 17,280 rows and 120
groups. Use smoke for plumbing, standard for model acceptance, and scale for
bounded runtime/memory checks. These sizes are software fixtures, not a statement
about the sample size needed for an operational study.

| Synthetic target | Relationship | Principal test |
|---|---|---|
| `level_linear` | Linear aging, wetness and traffic | Elastic Net reference; sample-weight and feature checks |
| `level_smooth` | Smooth additive climate curves | GAM shapes; tree approximation of smooth functions |
| `level_threshold` | Thresholds plus interacting conditions | ExtraTrees/XGBoost/LightGBM splits |
| `level_interaction` | Continuous products of climate/pavement variables | Tree interaction learning; additive model limitations |
| `level_zero_inflated` | Mostly zeros, with feature-dependent occurrence and severity | Two-stage output, probability and sparse-class behavior |
| `level_nonclimate` | Pavement features only | Climate ablation should show no planted advantage |
| `level_no_signal` | Independent random outcome | Baseline can legitimately win; importance should not be promoted |
| `level_constant_zero` | All zeros | Explicit constant status and zero predictions |
| `level_constant_positive` | Constant seven | Constant status and absent R-squared denominator |

Other fixtures cover missing numeric values, an entirely missing column, a
categorical level seen only in holdout, missing target labels, invalid interval
weights, duplicated row identifiers, no positive training outcomes and no
holdout labels. A separate leakage challenge contains a group token and copied
endpoint outcomes. Those columns never enter the normal predictor allowlist.

The oracle file contains conditional means, occurrence probabilities, positive
means and the two-stage conditional median. The contracts list planted drivers
and correlated proxies. Oracle columns and other target columns must never be
automatically added to predictors. Correlated climate proxies can share
permutation importance, so test driver blocks rather than demanding a fixed
single-feature ordering. The generated formulas test implementation; they do not
establish physical relationships in actual pavements.

Delivered commands, after installing this branch:

```bash
python -m pavement_analytics.synthetic_benchmark create \
  --output "$HOME/pavement-model-tests/data" --profile standard
python -m pavement_analytics.synthetic_benchmark validate \
  --workspace "$HOME/pavement-model-tests/data"
python -m pavement_analytics.synthetic_benchmark smoke \
  --workspace "$HOME/pavement-model-tests/data" \
  --output "$HOME/pavement-model-tests/reference-run"
```

The smoke command tests the currently installed candidate function. Its default
population is PCC with linear, smooth and zero-inflated targets. Add
`--all-targets --all-materials` for the complete existing-candidate sweep. Output
directories must be new and outside Git checkouts. The dataset is already at
the model stage: it is not an input to `pavement full`, which has upstream PAVER,
location, ERA5 and private curve requirements. Upstream integration remains
covered by the repository's separate generated pipeline tests.

## R1 — Candidate registry and evaluation contract

### Delivered in Build 1

- Build 1's registry declared the implemented IDs `median`, `enet_0.01`,
  `enet_0.1`, `rf_5` and `rf_20`, model families, fixed parameters, dependency
  availability, sample-weight support, prediction meaning and worker limits.
  The public `study_modeling.candidates(seed)` adapter remains usable.
- The original four learning configurations and preprocessing class locations
  are retained. The separate legacy rate-model grid is unchanged.
- `candidate_audit.json` records each candidate/feature-set attempt. The model
  comparisons CSV retains failed, unavailable and unsupported candidates.
  Partial-fold results are never ranked as a complete CV score. Raw exceptions
  go to the private task's `candidate_errors.log`; shared status fields contain
  fixed codes. Failure of the median baseline still fails the task.
- A candidate that fails final development refitting or prediction falls back
  to the next eligible CV result, without using holdout labels to choose it.
- Summaries and saved models record the registry version, family, parameters
  and dependency versions. Existing prediction keys and model bundle format
  remain compatible. Reports show candidate failure/unavailability counts.
- The benchmark can require a subset, always retaining the median baseline:

```bash
python -m pavement_analytics.synthetic_benchmark smoke \
  --workspace "$HOME/pavement-model-tests/data" \
  --output "$HOME/pavement-model-tests/registry-check" \
  --candidate enet_0.01
```

Repeat `--candidate` for multiple implemented IDs. Unknown, duplicate,
unavailable or unweighted explicitly requested candidates fail clearly rather
than disappearing from the search. Optional unavailable entries are recorded
and supported candidates can continue.

Acceptance tests compare every original factory and the complete old grid
against frozen pre-registry definitions. They check saved prediction parity,
holdout-label invariance, incomplete CV failure, dependency/constructor/fit/
prediction/refit errors, feature guards and consolidated failure reporting.
All tests run locally while hosted CI is disabled.

**Implementation.** Extract candidate construction from
`study_modeling.candidates` into a small registry. Each entry supplies a stable
ID, family, parameters, required package, weight support, prediction meaning,
resource limits and availability status. Keep `candidates(seed)` as a compatible
adapter initially. Do not change the separate legacy rate search in
`modeling.py` accidentally. `full_modeling.run_full_models` should continue to
call the same observed-level evaluation path.

All estimators must implement cloneable `fit(X, y, sample_weight)` and
`predict(X)` behavior, be serializable, and return finite one-dimensional
nonnegative predictions. Fit numeric/categorical roles, imputation, scaling,
encoding, knots and all feature selection within each training fold. Keep
identifiers, dates, interval weights, endpoint outcomes, other target columns
and oracle values outside the feature set. Preserve constant-target fallbacks.

The full-pipeline primary score remains mean inspection-interval-weighted MAE
over the existing grouped folds. Also report weighted RMSE, R-squared, median
baseline skill and climate-added skill. Preserve the existing `study` command's
equal-total-group weighting; do not silently substitute full-pipeline weights.
Use one plan for every candidate, target, surface and all/nonclimate comparison.
The holdout cannot choose hyperparameters, feature sets, stopping iterations,
probability thresholds or explanation settings.

**Gate.** Compare the refactored old candidates with the original search on the
same seed/plan, with equal scores, selected IDs and predictions within declared
floating-point tolerances. Mutate only holdout labels and show unchanged
development fits and CV selection. Verify that missing dependency, failed fit,
unsupported configuration and successful fit are distinct outcomes. A failed
candidate must not silently vanish from the report. An explicitly required
unavailable model fails preflight; optional search candidates may be recorded
as unavailable while supported candidates continue.

## R2 — ExtraTrees

### Delivered in Build 2 (version 0.5.2)

The default observed-level searches in `full` and `study` include these stable
IDs, using the existing training-fold preprocessing and nonnegative wrapper:

| Candidate ID | Minimum leaf observations | Maximum depth |
|---|---|---|
| `et_leaf2_depth12` | 2 | 12 |
| `et_leaf2_depthnone` | 2 | Unlimited |
| `et_leaf10_depth12` | 10 | 12 |
| `et_leaf10_depthnone` | 10 | Unlimited |

Every configuration uses 200 trees, squared-error fitting with supplied sample
weights, all encoded features at each split, no bootstrap and one worker.
Selection remains mean weighted CV MAE. No additional package is required.
Registry version 2 records these settings in audits, summaries and saved models.

The larger default pool can change selected models and numerical results. The
original four learning configurations retain their fit, score and prediction
parity when explicitly selected. The separate legacy rate grid is unchanged.
With five folds and both feature sets, ExtraTrees adds 40 CV fits per target,
plus any selected refits. Use a new run after updating the source; existing
resume fingerprints continue to reject code changes.

### Frozen acceptance protocol

Build 2 acceptance is frozen before execution: calibration seeds 20261011,
20261012 and 20261013, followed by verification seed 20261031. Use the standard
PCC population, its original five grouped folds and interval-day weights.
For each calibration seed, the four-candidate ExtraTrees search must reduce
development CV MAE by at least 20% against the weighted median on both threshold
and interaction targets. These gates concern planted signal, not superiority
to every competing estimator. Do not change the grid or gates after inspecting
verification results.

The standard verification covers linear, threshold, interaction, no-signal and
stress inputs, plus constant/missing-label controls. Every registered ExtraTrees
configuration is fitted and round-trip serialized independently of selection.
The scale resource check fits all four configurations sequentially on the
development portion of the scale PCC fixture (5,760 total observations).
Before measurement, the per-standard-task and complete scale-fit budgets are
120 seconds wall time, 120 seconds process CPU time and 768 MiB peak process
RSS on the recorded Linux environment, with one estimator worker. These are
acceptance budgets for these fixtures, not a runtime promise for private data.

**Tests.** Run linear, threshold, interaction, no-signal and stress fixtures.
Check supplied weights affect the fitted loss, saved models reproduce saved
predictions, and correlated/noise features are not misreported as physical
causes. Compare appropriate nonlinear targets with the median baseline; no test
should require ExtraTrees to beat every other algorithm.

**Gate.** Shared functional tests pass, nonlinear signal is learnable on
predeclared calibration seeds, and CPU/memory fit within the recorded budget.

### Acceptance evidence

- All 374 local regression tests passed. The six calibration seed/target cases
  cleared the frozen 20% development-CV improvement gate. Analytical single-leaf
  tests verify that every configuration actually uses supplied weights.
- Mixed numeric/categorical inputs, missing predictors, an entirely missing
  column, unseen holdout categories and missing target labels pass. Tests prove
  training-only imputation/encoding, cloneability, serialization for every
  configuration, grouped fold identity and holdout-label mutation invariance.
  Constant and unsupported target populations retain explicit statuses.
- Verification seed 20261031 completed all nine standard PCC targets and a
  separate stress search without candidate failures or unavailable candidates.
  ExtraTrees was selected on smooth and interaction targets. ElasticNet and
  Random Forest remained selected on other signals; the median remained selected
  on no-signal and zero-inflated targets. Reloaded model predictions match saved
  predictions within absolute/relative tolerance `1e-12`; direct configuration
  round trips use absolute tolerance `1e-12`.
- The standard tasks took at most 23.83 seconds wall / 23.82 seconds CPU. The
  stress task took 21.36 seconds wall / CPU. Their peak process RSS was
  177.80 MiB. The four scale fits, prediction and serialization checks took
  9.04 seconds wall / CPU, with 346.62 MiB cumulative peak process RSS.
  All resource gates passed. Measurements used Linux x86_64, Python 3.12.14,
  scikit-learn 1.8.0, NumPy 2.3.5, pandas 2.2.3 and SciPy 1.17.0.

Reproduce functional and calibration checks locally:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_extra_trees.py' -v
```

The existing benchmark `smoke --all-targets` command exercises the expanded
default pool on a generated standard bundle. To isolate ExtraTrees, repeat
`--candidate` with the four IDs above. The shared weighted median remains in
each search. These are synthetic model-stage checks; live ERA5, deployment
macOS validation and private-pilot acceptance remain in R8. Hosted CI is disabled.

## R3 — XGBoost

### Delivered in Build 3 (version 0.5.3)

The optional `xgboost` extra supplies these eight stable candidate IDs:

| Fixed-budget ID | Grouped-stopping ID | Depth | Learning rate |
|---|---|---|---|
| `xgb_depth3_lr0.05` | `xgb_depth3_lr0.05_es` | 3 | 0.05 |
| `xgb_depth3_lr0.1` | `xgb_depth3_lr0.1_es` | 3 | 0.1 |
| `xgb_depth6_lr0.05` | `xgb_depth6_lr0.05_es` | 6 | 0.05 |
| `xgb_depth6_lr0.1` | `xgb_depth6_lr0.1_es` | 6 | 0.1 |

All use CPU histogram trees, squared-error fitting, MAE for stopping, a maximum
of 300 rounds, one estimator worker, and full row/feature sampling. Stopped
variants use patience 20 and a 20% group stopping fraction. The package range
is `>=3.1.3,<3.2`, with prereleases rejected; admission uses 3.1.3. Installing
the extra automatically includes supported candidates in `full` and `study`.
Base installations retain their original eight learning configurations;
unavailable/unsupported XGBoost candidates are recorded rather than silently
discarded. Explicit required-candidate selection fails preflight if unusable.

`GroupedXGBoostRegressor` receives group identities separately from predictor
columns. Each stopped outer fit learns imputation/scaling/encoding only on its
inner fitting groups, transforms its stopping groups with that same pipeline,
and forwards the corresponding weights to both native fitting and evaluation.
It then fits fresh preprocessing and a fixed-count model on the entire current
training population using `best_iteration + 1`. Final development fitting uses
this same nested policy. No outer-validation or holdout group participates in
stopping. A stopped estimator called without grouping fails explicitly.

Registry version 3 records fixed parameters and supported dependency bounds.
Candidate audits and both task/consolidated leaderboards record actual CV/final
iterations plus fitting, stopping and refit identities, rows and weight totals.
Probe iterations are separate from the final model count. Saved models retain
both all-feature and nonclimate final training details. These private identities
are not displayed in dashboard HTML; old report inputs still load.

Full-run fingerprints now include model dependency versions, including optional
backend presence/version; the private run report records that environment.
Installing or changing a backend therefore requires a new run. The original
four-model and ExtraTrees definitions and the separate legacy rate grid are
unchanged. Default winners and scores can change with the larger candidate pool.

### Frozen acceptance protocol

Build 3 acceptance is frozen before measurement. Keep R2's calibration seeds
20261011/20261012/20261013 and verification seed 20261031 so comparisons share
the same invented populations. Use four 300-round fixed configurations
(depth 3/6, learning rate 0.05/0.1) and four matching stopped configurations
(patience 20, stopping fraction 0.2), CPU histogram trees and one worker.
Both the best fixed and best stopped configurations must reduce development
CV MAE by at least 20% against the weighted median on threshold and interaction
targets for every calibration seed. Do not tune on verification outcomes.
Standard verification covers all nine PCC targets and stress inputs; direct
checks cover every configuration even if it is never selected. The scale
check fits all eight sequentially on scale PCC development rows. The resource
budgets, declared before execution, are 180 seconds wall / process CPU per
standard task or complete scale check, and 1,024 MiB peak process RSS on the
recorded Linux environment. Real-data and macOS acceptance remain separate.

**Tests.** Threshold/interaction/zero-heavy signals, weighted evaluation sets,
missing inputs, deterministic CPU runs, transformed feature alignment,
unsupported version and dependency absence. Test stopping-set identity and the
holdout-label mutation invariant.

**Gate.** Both fixed-budget and stopping configurations satisfy common contracts;
the leaderboard records actual iterations and all stopping-group identities.

### Acceptance evidence

- All 386 local regression tests passed, including the original full-pipeline,
  report, registry and ExtraTrees checks. Every fixed/stopped configuration
  passes analytical weight forwarding, mixed/missing/unseen input handling,
  deterministic repetition, cloning and serialization. The six calibration
  seed/target cases passed the frozen CV gate independently for fixed and
  stopped grids.
- Native-fit interception verifies that the stopping table is transformed
  with fitting-only preprocessing and receives its exact evaluation weights.
  The weighted native stopping metric matches an independent MAE calculation.
  Recorded groups exclude outer-validation and holdout populations. Changing
  only holdout labels leaves selected candidates, CV scores, training audits
  and predictions unchanged. A flat-signal check confirms that stopping can
  shorten the final model to one round.
- Missing and unsupported dependencies have distinct recorded statuses;
  explicit unusable requests fail before creating outputs. Base searches
  continue without XGBoost. Constant, missing-label and positive-only-holdout
  controls terminate with explicit statuses. A dependency-presence change
  invalidates an otherwise unchanged full-pipeline checkpoint.
- Verification seed 20261031 completed all nine standard PCC targets and an
  XGBoost-only stress search without candidate failures or unavailable models.
  XGBoost was selected on smooth, threshold and interaction targets; ElasticNet
  remained selected on linear/nonclimate targets, and the median on no-signal
  and zero-heavy targets. The stopped XGBoost configuration was selected in the
  family-only stress search. Every saved model reproduced saved predictions
  within absolute/relative tolerance `1e-12`; direct configuration round trips
  use absolute tolerance `1e-12`.
- Standard tasks took at most 53.27 seconds wall / 63.31 seconds process CPU,
  with 216.63 MiB peak process RSS. Stress took 18.70 seconds wall / 29.14
  seconds CPU. The eight scale fits plus prediction/serialization checks took
  5.24 seconds wall / 7.38 seconds CPU, with 250.50 MiB cumulative peak RSS.
  All declared resource gates passed. Measurements used Linux x86_64,
  Python 3.12.14, XGBoost 3.1.3, scikit-learn 1.8.0, NumPy 2.3.5, pandas 2.2.3,
  SciPy 1.17.0 and packaging 26.3. Deployment macOS and real-data acceptance
  remain in R8; hosted CI is disabled.

Reproduce functional, boundary and calibration checks after installing the extra:

```bash
python -m pip install -e '.[xgboost]'
PYTHONPATH=src python -m unittest discover -s tests -p 'test_xgboost.py' -v
```

The existing benchmark can isolate required configurations with, for example,
`--candidate xgb_depth3_lr0.05 --candidate xgb_depth3_lr0.05_es`. Its shared
weighted median remains present. Add `--all-targets` for the complete target
sweep on an existing generated standard bundle. No benchmark outputs are
stored in the source repository.

## R4 — LightGBM

### Delivered in Build 4 (version 0.5.4)

The optional `lightgbm` extra adds eight stable candidate IDs:

| Fixed-budget ID | Grouped-stopping ID | Leaves | Minimum leaf observations |
|---|---|---|---|
| `lgbm_leaves15_min10` | `lgbm_leaves15_min10_es` | 15 | 10 |
| `lgbm_leaves15_min30` | `lgbm_leaves15_min30_es` | 15 | 30 |
| `lgbm_leaves31_min10` | `lgbm_leaves31_min10_es` | 31 | 10 |
| `lgbm_leaves31_min30` | `lgbm_leaves31_min30_es` | 31 | 30 |

All use learning rate 0.05, up to 300 rounds, squared-error regression,
weighted MAE stopping, CPU `gbdt`, one estimator worker, deterministic mode
and forced column-wise histograms. Stopped variants use fresh native callbacks
with patience 20 and the existing 20% training-group split. The package
range is `>=4.6.0,<4.7`; admission uses 4.6.0. Installed supported candidates
join `full` and `study` automatically, with no new private configuration keys.

The wrapper reuses the original numeric/one-hot preprocessing and R3's grouped
splitter. It transforms stopping inputs with fitting-only preprocessing, passes
matching fitting/evaluation weights, then refits fresh preprocessing and a
fixed-count model on all current training rows. Outer CV and holdout groups
never supply stopping data. LightGBM's best-iteration count is already one-based;
the wrapper uses it directly, independently of XGBoost's zero-based convention.

Neutral encoded column names avoid restrictions on quoted raw category labels.
Private training audits record raw feature order, encoded/transformed order,
category mappings and infrequent-category mappings separately for the probe and
final refit. The saved preprocessing retains its fitted encoders. Prediction
uses the same encoded columns; native categorical support is not introduced.

Actual native rounds can be below the requested budget even without a stopping
callback when no useful split remains. The common leaderboard's iteration fields
contain actual fitted counts; `requested_iterations`, probe evaluation rounds
and probe retained-model rounds are separate fields in training details.
Registry version 4 and environment provenance include the optional backend.
Missing, unsupported and unloadable backends remain distinct outcomes; explicitly
required unusable candidates fail preflight. Existing model definitions and
the legacy rate grid remain unchanged. The expanded pool can change winners.

### Frozen acceptance protocol

Build 4 acceptance is frozen before measurement: calibration seeds
20261011/20261012/20261013 and verification seed 20261031, with the same standard
PCC population, grouped plan and interval weights as R2/R3. Four configurations
use 15/31 leaves, minimum leaf observations 10/30, learning rate 0.05 and a
300-round budget. Four matching stopped configurations use patience 20 and a
20% training-group stopping fraction. Each uses deterministic CPU training and
one worker. Both the best fixed and best stopped grids must improve development
CV MAE at least 20% against the median on threshold/interaction targets for
every calibration seed; no tuning on verification results is permitted.
Final verification covers all nine PCC targets and a LightGBM-only stress
search. All eight configurations receive direct serialization and scale checks.
The predeclared limits are 180 seconds wall / process CPU per standard task or
complete scale check, and 1,024 MiB peak process RSS on the recorded Linux
environment. Deployment macOS runtime acceptance remains an R8 release check.

**Tests.** Threshold/interaction targets, stress schema, evaluation weights,
rare target values, serialization and installation/dependency diagnostics.

**Gate.** Weighted fixed-budget and early-stopping tests pass; native-category
variants, if introduced, have independent schema and unseen-category coverage.

### Functional acceptance evidence

- All 400 local regression tests passed, including the existing pipeline,
  reports, original-model parity, ExtraTrees and XGBoost suites. All 14 new
  LightGBM admission tests pass. The six calibration seed/target cases clear
  the frozen development-CV gate independently for fixed and stopped grids.
- Every configuration passes analytical sample-weight forwarding, deterministic
  repetition, cloning, mixed/missing/unseen inputs and serialization. Native
  fitting interception verifies exact stopping transformations, evaluation
  weights and weighted MAE. Holdout-label mutation leaves CV selection,
  training audits and predictions unchanged; recorded groups exclude evaluation
  populations.
- Quoted and high-cardinality categories preserve explicit encoded order and
  infrequent-category mappings. No native categorical variant is enabled.
  Tests distinguish requested 300-round fitting from a one-round native model
  when no split is possible, and verify the one-based stopping convention.
- Constant, missing-label, positive-only-holdout and one-positive-group controls
  terminate predictably. Missing, unsupported and native-load failure cases
  remain visible, with raw loader diagnostics kept privately. The optional
  LightGBM version contributes to full-run checkpoint invalidation.
- Linux runtime fitting was tested with LightGBM 4.6.0. Pip successfully resolved
  and downloaded its macOS 12+ ARM64 wheel for Python 3.12. This checks package
  availability, not native macOS loading/fitting. Deployment commands and OpenMP
  guidance are in `docs/deployment.md`; macOS runtime qualification remains R8.

Final verification seed 20261031 completed all nine standard PCC targets and
a separate LightGBM-only stress search without candidate failures or unavailable
models. All saved predictions match reloaded predictions within absolute/relative
tolerance `1e-12`; direct configuration round trips use absolute tolerance
`1e-12`. LightGBM was selected on smooth, threshold and interaction targets;
ElasticNet remained selected on linear and nonclimate targets, and the median
on zero-heavy and constant targets. The no-signal control selected a stopped
LightGBM with only 0.34% holdout MAE skill, negative holdout R-squared and
`importance_reliable=false`; that chance variation is not signal recovery.
The family-only stress search selected a stopped LightGBM configuration.

Standard tasks took at most 69.68 seconds wall / 79.45 seconds process CPU,
with 241.39 MiB peak RSS. Stress took 14.44 seconds wall / CPU. The eight scale
fits plus prediction/serialization checks took 4.55 seconds wall / CPU, with
266.20 MiB cumulative peak RSS. All predeclared resource gates passed.
Measurements used Linux x86_64, Python 3.12.14, LightGBM 4.6.0, XGBoost 3.1.3,
scikit-learn 1.8.0, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0 and packaging 26.3.
Live ERA5 and private-pilot acceptance remain in R8.

Reproduce the LightGBM functional and calibration checks:

```bash
python -m pip install -e '.[lightgbm]'
PYTHONPATH=src python -m unittest discover -s tests -p 'test_lightgbm.py' -v
```

The benchmark supports required selectors such as
`--candidate lgbm_leaves15_min10 --candidate lgbm_leaves15_min10_es`, always
retaining its shared weighted median. `--all-targets` exercises all nine target
patterns on a generated standard bundle. Hosted CI remains disabled.

## R5 — Gaussian additive spline model

### Delivered in Build 5 (version 0.5.5)

The base installation adds four stable IDs:

| Candidate ID | Uniform knots | Ridge alpha |
|---|---|---|
| `spline_knots5_alpha0.1` | 5 | 0.1 |
| `spline_knots5_alpha1.0` | 5 | 1.0 |
| `spline_knots8_alpha0.1` | 8 | 0.1 |
| `spline_knots8_alpha1.0` | 8 | 1.0 |

`AdditiveSplineRegressor` fits independent cubic B-spline blocks for numeric
predictors, additive one-hot terms for categoricals and separate linear numeric
missingness indicators, then applies weighted Ridge to their concatenation.
The coefficient penalty is L2, not a curvature/smoothing penalty. This is
Gaussian identity-link additive spline regression, not proprietary DataRobot
GAM reproduction or a pyGAM backend. No additional package is required.

Every fold learns feature roles, median imputations, numeric support, knots,
category mappings and coefficients on its training rows. Reviewed subgrade,
branch-use, construction and code fields, explicit categorical declarations,
pandas categorical types and booleans remain categorical even when numerically
coded. Other numeric-looking fields follow the existing training-only parser;
unreviewed numeric category codes require explicit typing/declaration. Constant
and all-missing numeric columns skip expansion but retain their missing term.
Uniform spline knots and constant extrapolation avoid unbounded tail curves.

The private saved model offers `predict_components(X)`, returning an intercept
and per-feature terms, and `predict_margin(X)` for their sum. `predict(X)` clips
the margin at zero; this final floor can alter additivity at the boundary.
Components are not causal effects or unique allocations among correlated
features. The pure additive margin contains no interaction bases.

Registry version 5 records the four configurations. Private training details
include feature roles, median values, training bounds, full knot vectors,
category/infrequent-category mappings and component block positions. They also
record fitting groups and weights. The policy is `closed_form_weighted_ridge`
with null iterations, rather than an invented boosting count. The existing
model definitions and separate legacy rate grid are unchanged; a larger default
pool can change the winner. Four settings add 40 CV fits per five-fold target,
plus selected refits. Source fingerprints require a new run after this update.

### Frozen acceptance protocol

Build 5 acceptance is frozen before measurement: use the R2–R4 calibration
seeds 20261011/20261012/20261013 and verification seed 20261031, with the same
standard PCC population, grouped plan and interval weights. The grid is
5/8 uniformly spaced knots, cubic bases, Ridge alpha 0.1/1.0 and constant
extrapolation. For each calibration seed, the spline grid must improve
development CV MAE by at least 20% against the median on linear and smooth
targets. A fixed `spline_knots8_alpha0.1` development fit must recover each
centered planted smooth driver curve (aging, temperature range, wetness) with
grid RMSE at most 1.0, without using holdout outcomes. Centering removes the
arbitrary additive intercept allocation. The pure-interaction limitation is
checked algebraically rather than requiring a universal model winner.
Final verification covers all nine PCC targets and a spline-only stress search;
every configuration is independently serialized and scale-tested. Resource
limits are declared as 180 seconds wall / process CPU per standard task or
complete four-configuration scale check, with 1,024 MiB peak process RSS on the
recorded Linux environment. No grid or gate changes follow verification results.

**Tests.** Recover the planted smooth shapes from `level_smooth`, check linear
reference performance, all-missing/constant predictors, boundary/extrapolation
behavior, sample-weight forwarding and saved prediction parity. Check that an
additive model's weaker performance on pure interactions is reported honestly.

**Gate.** Shape and prediction tests pass across fixed calibration seeds.
Any later pyGAM variant must separately prove weighted fitting, cloning,
serialization and platform support before registry admission.

### Functional acceptance evidence

- All 411 local regression tests passed, including the existing pipeline,
  reports, original-model parity, tree and boosting suites. All 11 new spline
  tests pass. The six calibration seed/target cases clear the frozen 20%
  development-CV improvement gate, and all nine centered driver curves meet
  the declared RMSE-at-most-1.0 requirement using the fixed development fit.
- Every configuration passes analytical sample-weight forwarding, cloning,
  deterministic repetition, missing/unseen inputs and serialization. Separate
  tests prove that medians and knot support remain fixed when new inputs arrive,
  coded categorical fields stay one-hot, and missing indicators stay linear.
  Constant and all-missing numeric columns skip spline expansion.
- Extrapolation beyond either training boundary equals the corresponding
  boundary prediction. Feature terms plus intercept reconstruct the margin
  within `1e-12`; nonnegative prediction clips that margin. An independent
  product fixture has a true cross difference of 40, while the fitted additive
  margin has zero cross difference, demonstrating its interaction limitation.
- Changing holdout labels leaves selected candidates, CV errors, training
  audits and predictions unchanged. Fitting-group identities match the shared
  plan, with no stopping populations. Constant, missing-label and
  positive-only-holdout controls retain explicit statuses. Invalid weights or
  unsupported spline settings fail explicitly.

Verification seed 20261031 completed all nine standard PCC targets and a
spline-only stress search without candidate failures or unavailable models.
Every saved prediction matches its reloaded model within absolute/relative
tolerance `1e-12`; direct configuration round trips use absolute tolerance
`1e-12`. The full search selected additive splines on the smooth target,
ElasticNet on linear/nonclimate targets, LightGBM on threshold/interaction
targets and the median on zero-heavy/constant targets. The no-signal control's
small LightGBM MAE gain remains flagged unreliable with negative holdout
R-squared. The family-only stress search selected additive splines.

Standard tasks took at most 71.86 seconds wall / 82.49 seconds process CPU,
with 249.03 MiB peak process RSS. The spline-only stress search took 3.52 seconds
wall / 4.31 seconds CPU. The four scale fits plus prediction/serialization took
0.67 seconds wall / 2.33 seconds CPU, with 274.97 MiB cumulative peak RSS.
All declared resource gates passed. Measurements used Linux x86_64,
Python 3.12.14, scikit-learn 1.8.0, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0,
packaging 26.3, XGBoost 3.1.3 and LightGBM 4.6.0.

Reproduce spline functional and calibration checks with the base installation:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_spline_model.py' -v
```

The existing benchmark supports selectors such as
`--candidate spline_knots5_alpha0.1 --candidate spline_knots8_alpha0.1`, retaining
the shared weighted median. Add `--all-targets` for all nine generated target
patterns. These checks use invented model-stage data; macOS runtime, live ERA5
and real-data release acceptance remain in R8. Hosted CI is disabled.

## R6 — Two-stage occurrence and positive severity

### Delivered in Build 6 (version 0.5.6)

The base installation adds four stable IDs:

| Candidate ID | Logistic C | Positive ExtraTrees leaf size |
|---|---|---|
| `two_stage_c0.1_leaf2` | 0.1 | 2 |
| `two_stage_c0.1_leaf10` | 0.1 | 10 |
| `two_stage_c1.0_leaf2` | 1.0 | 2 |
| `two_stage_c1.0_leaf10` | 1.0 | 10 |

`OccurrenceSeverityRegressor` uses a cloneable weighted logistic classifier
on all current training rows and an accepted squared-error ExtraTrees regressor
on positive rows only. The stages learn separate preprocessing from those
respective training subsets. Logistic fitting uses lbfgs, tolerance 0.0001 and
at most 1,000 iterations. Severity uses 200 trees, depth 12, no bootstrap,
all features and one worker. No probability calibration or thresholding is added.
Both stages receive their matching positive interval-day or group-balanced
weights. Required group context enforces the frozen class/group support rules.
All-zero, all-positive, constant-positive and sparse-class/positive-group
policies have explicit recorded statuses; missing classes are never supplied
by evaluation labels. The whole-task development-constant shortcut remains
the existing median-only evaluation, with empty component exports.

The saved estimator returns probability, expected positive magnitude and their
product through `predict_occurrence_severity(X)`; `predict(X)` returns that
product. Every component must be finite, nonnegative, correctly shaped and
consistent with the product. Positive-class probability is extracted by class
identity, including reversed native class order. Training audits retain row,
group and weight support for both stages, fallbacks and classifier iterations.

Primary selection remains complete development-CV MAE with baseline-first ties.
Each feature set also selects its best complete two-stage CV candidate for a
separate development-only diagnostic refit and component holdout evaluation.
It is saved as `two_stage_all_model.joblib` or
`two_stage_nonclimate_model.joblib`, even when the median wins primary selection.
An unsuccessful diagnostic refit/prediction records a fixed failure code and
tries the next complete within-family CV candidate; raw details stay private.
Serialization/output failures remain fatal. No holdout labels choose a model.

Task-level and consolidated `two_stage_predictions.csv` retain saved weights,
all successful attempted OOF component rows, and holdout components for that
one selected family diagnostic. `two_stage_metrics.csv` reports overall mean
MAE/RMSE and RMSE skill, occurrence Brier score/skill and clipped log loss,
trapezoidal PR-AUC and separate average precision, plus positive-only severity
errors. Constant references are the corresponding fitting partition's weighted
prevalence, weighted positive mean and their product (weighted overall mean).
Metrics pool the saved evaluation weights; they do not replace mean-fold CV
MAE. PR metrics are undefined for single-class evaluation, and severity errors
are undefined without positive evaluation rows. Partial CV attempts are labeled
and cannot enter selection. The offline dashboard explains the mean semantics
and shows the within-family diagnostic without changing primary importance.
R7 below adds family comparisons, component importance and runtime controls.

### Frozen acceptance protocol

Build 6 acceptance is frozen before measurement. Keep calibration seeds
20261011/20261012/20261013 and verification seed 20261031, standard PCC rows,
the shared grouped plan and interval-day weights. The four configurations use
logistic occurrence C 0.1/1.0 and accepted depth-12 ExtraTrees severity with
200 trees and leaf size 2/10. Mixed-class fitting needs at least five rows and
two groups in each class; severity fitting needs ten positive rows and two
positive groups. Sparse cases use weighted training prevalence/positive means.
Neither evaluation labels nor a probability threshold choose these fallbacks.

A fixed `two_stage_c1.0_leaf10` reference must show pooled development-OOF
Brier skill and conditional-mean RMSE skill of at least 1% against fold-training
constant baselines on the zero-inflated target, for every calibration seed.
Its probability oracle RMSE must be at most 0.15 and overall conditional-mean
oracle RMSE at most 3.5. OOF oracles only check software recovery; they are not
predictors or selection inputs. MAE remains the production selection metric,
and a median baseline may win. An independent uniform-mixture calculation checks
mean/median loss semantics without fitting or evaluating on a preferred holdout.

Final verification covers all nine PCC targets and a family-only zero-inflated
stress search. Every configuration is serialized and scale-tested on a
zero-inflated population, exercising both stages. Predeclared resource limits
are 180 seconds wall/process CPU per standard task or complete four-model scale
check, with 1,024 MiB peak process RSS on the recorded Linux environment.
No grid or gate changes follow verification outcomes.

**Implementation.** Fit an occurrence classifier to `I(y>0)` using all training
rows and weights. Fit the positive-severity regressor only where the training
target is positive, with corresponding weights. The first candidate can use
logistic occurrence plus an already accepted severity estimator. More complex
occurrence/severity pairs should be separate, budgeted candidates.

Record `p_positive`, positive magnitude and the final point prediction. If no
training values are positive, use an explicit zero fallback without fitting a
severity model. If all are positive, use probability one. Very few positives or
a fold without a class must produce a documented fallback/status; holdout
labels cannot supply the missing class. Probability calibration, if used,
requires a training-only grouped procedure.

**Prediction meaning matters.** Probability times conditional positive mean
estimates `E[Y|X]`. This minimizes population squared-error loss, not generally
absolute-error loss. With a zero atom, the conditional median is zero whenever
`p_positive <= 0.5`. For the fixture's uniform positive severity with mean `s`,
the conditional median otherwise equals `s * (1.4 - 0.4/p_positive)`. Both
oracles are supplied. For a general positive distribution, an MAE-oriented
point prediction requires the positive quantile `1 - 0.5/p_positive`, not a
probability-scaled severity median. A quantile extension needs its own tests.

Keep the current primary MAE selection for comparability and allow a baseline
to win. Report conditional-mean RMSE, occurrence Brier/log loss and PR-AUC when
defined, and positive-only severity error separately. Do not claim failure just
because an expected-value predictor loses MAE to zero. A different production
selection metric is a separately reviewed task and must not be mixed into one
leaderboard without labeling it.

**Tests.** Zero-inflated and positive-only targets; all-zero, all-positive,
training-constant/holdout-positive, one-positive-group and missing-label cases.
Test class-index extraction, weights in both stages, rare-class fallbacks,
pipeline cloning/serialization and nonnegative output. Use the same grouped
splits for both components and all competing single-stage regressors.

**Gate.** Probability/mean/median semantics are documented and independently
verified. Every sparse-class case terminates predictably without target leakage.

### Functional acceptance evidence

- All 427 local regression tests passed in 533.572 seconds, including 15 new
  two-stage tests and a full-report checkpoint test for the component exports.
  The fixed reference clears every declared probability/mean recovery and skill
  gate on all three calibration seeds; oracle values never enter predictors or
  model selection. The independent uniform-mixture calculation confirms that
  the mean minimizes squared loss while the median can have lower absolute loss.
- Native fitting receives all binary occurrence rows/weights and only positive
  severity rows/weights. Analytical constant-input optima, reversed class order,
  invalid components, sparse row/group support, repeated fitting, missing/unseen
  predictors and serialization pass. Zero/all-positive and global constant or
  missing-label cases retain the documented fallbacks and task statuses.
- Shared fold identities match training audits. Changing holdout labels leaves
  primary/diagnostic selection, CV scores, training audits and predictions
  unchanged. Partial CV prediction failures cannot win; failed diagnostic refits
  preserve the primary model and private error boundary. Consolidation retains
  all component rows without displaying group identities in HTML. Altering
  either consolidated component CSV is detected by full-run resume checks.

Verification seed 20261031 completed all nine standard PCC targets and a
two-stage-only zero-inflated stress search with no candidate failures or
unavailable models. The standard grid contains 33 candidates including the
median, both optional boosting families and all four two-stage configurations.
Every saved primary/diagnostic prediction matches its reloaded estimator within
absolute/relative tolerance `1e-12`; direct component round trips are identical.
All four scale fits exercise both native stages and pass serialization.

On the invented zero-heavy verification target, the median remains the primary
MAE choice. The within-family diagnostic `two_stage_c1.0_leaf10` has weighted
holdout Brier score 0.1817, Brier skill 17.52%, overall-mean RMSE 11.9131
and RMSE skill 10.35% against its corresponding constant training references.
Positive-only severity RMSE is 6.2329.
These distinct loss comparisons are retained without forcing an MAE win or
changing production selection. They are software-fixture results.

Standard tasks took at most 88.75 seconds wall / 100.25 seconds process CPU,
with 270.68 MiB peak process RSS. The family-only stress search took
10.35 seconds wall / 38.98 seconds CPU.
The four scale fits plus prediction/serialization took 3.56 seconds
wall / 6.05 seconds CPU, with 279.55 MiB cumulative peak RSS.
All predeclared resource gates passed. Measurements used Linux x86_64,
Python 3.12.14, scikit-learn 1.8.0, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0,
packaging 26.3, XGBoost 3.1.3 and LightGBM 4.6.0.

Reproduce two-stage functional and calibration checks with the base installation:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_two_stage.py' -v
```

The existing benchmark accepts repeated selectors such as
`--candidate two_stage_c0.1_leaf2 --candidate two_stage_c1.0_leaf10`, retaining
the weighted median. Add `--all-targets` for all nine generated patterns.
Hosted CI remains disabled. Actual macOS runtime, live ERA5 and private-data
acceptance remain R8 release checks; comprehensive family explanations and
runtime controls are next in R7.

## R7 — Comparisons, explanation and bounded execution

### Delivered in Build 7 (version 0.5.7)

Observed-level full/study runs enable the comparison/runtime path by default.
Each completed CV candidate receives a development-fitted artifact, a reload
prediction check and weighted holdout evaluation. New candidate comparison,
prediction, family summary, importance and timing sidecars retain all attempts
and consolidate into the offline report. Family representatives use the lowest
complete development-CV MAE independently for all/nonclimate features. Holdout
scores never choose the representative. The primary model/metrics and legacy
in-process library path remain compatible.

Family explanations use saved fitted pipelines, the original evaluation
weights, five permutations, the whole climate block and training-only numeric
Spearman blocks at absolute correlation at least 0.8. Two-stage explanations
also measure occurrence Brier loss and positive-only severity MAE. Heuristic
support flags are labeled exploratory; permutation SD is not a confidence
interval. Constant and unsupported families retain explicit statuses. Failed
refits and failed/timed-out explanations are visible without fabricated scores.

The default worker uses spawn and one native thread, with configurable
fit/prediction deadlines. Timed-out operations terminate the worker; later work
can start a fresh one. Native diagnostics stay private. Budgets include worst-case
refits and native stopping/two-stage fits. Candidate/fold events show active
work in the console and progress page. Ctrl+C cleans the worker and preserves
completed task checkpoints. Complete tasks are content-keyed, checksum-verified
and copied atomically into the final model stage; partial work is never reused.
Keys cover data/plan/order/dtypes, target/features/weights, seed, registry/grid,
options, source, Python/platform and dependencies. Full-stage fingerprint guards
remain in place. Full resume reuses completed targets; study can share an
external task cache across new workspaces. Study preprocessing resume is not
introduced by this model-stage change.

### Frozen acceptance protocol

Build 7 acceptance is frozen before measurement. The estimator grids and primary
mean-fold CV MAE rule remain unchanged. Functional checks use calibration seeds
20261011/20261012/20261013 for shared-plan and old/new selection/prediction parity,
without using holdout outcomes to choose family representatives or settings.
Final seed 20261031 covers standard PCC threshold, zero-inflated and no-signal
tasks with all 33 installed candidates, a seven-family stress search, and a
scale search with one predeclared representative per learned family. Each saved
candidate must reproduce its holdout predictions within absolute/relative
`1e-12`. Every family must have separate all/nonclimate CV selections or an
explicit failure/unavailability/constant status. Correlated blocks use training
absolute Spearman at least 0.8; permutation importance uses five fixed repeats.
No bootstrap confidence intervals or operational accuracy claims are added.
Resource gates are 300 seconds wall/process CPU per complete standard/stress
task or scale search, and 1,024 MiB for the sum of parent and peak worker RSS on
the recorded Linux environment. Timeout/cancellation, worker restart, task-key
invalidation, tampering, copy integrity and full/study setup/fingerprint checks
must pass before marking R7 complete. Default full/study work uses one spawned
worker, one native thread and 300-second fit/prediction deadlines.

A separate controlled one-vs-two-task experiment uses the standard PCC linear
and smooth targets with fixed ElasticNet/ExtraTrees selectors. It records wall,
CPU and summed peak-memory costs for sequential versus concurrent independent
tasks. This measures the tradeoff; concurrent fitting is not enabled in the
released full/study runner. The default remains one worker per sequential task.

Extend reports to show model family and parameters, candidate availability,
failure status, fitting time, iterations, feature count, row/group coverage,
CV and holdout scores, and baseline-relative skill. Preserve existing output
fields for readers. Record every attempted candidate, including unavailable
ones; do not advertise planned model families as executed.

For every accepted family, select an all-features model and a nonclimate model
using development CV independently. Holdout permutation importance should use
the saved fitted pipeline and unchanged evaluation weights. Include the whole
climate block and correlated-driver blocks. Raw permutation-repeat SD remains
permutation variability, not sampling confidence. If bootstrap uncertainty is
added, resample complete groups and state which comparisons it estimates.

The runner preserves primary model selection and adds comprehensive
per-candidate prediction/serialization reports. Losing selection does not hide
a family from direct evaluation. The legacy smoke command still exercises
installed registry entries; full/study and the R7 library options enable the
complete comparison/runtime path.

Precompute the fit budget as targets × surfaces × feature sets × outer folds
× configurations, plus refits/stopping fits. Start with one fit worker and one
estimator thread, then measure controlled parallelism. Add candidate/fold
progress events, cancellation and isolated resumable task outputs. Resume keys
must include data/plan hashes, candidate parameters, seed, feature rules and
dependency versions. Preserve the original full-pipeline code-change guards.

**Gate.** A model that raises or times out is visible in the result. Cancellation
keeps completed artifacts intact. Old reports still load. Importance on no-signal
and constant targets is not presented as reliable explanatory evidence.

### Functional acceptance evidence

- All 449 local regression tests passed in 859.245 seconds.
  The 18 new comparison/runtime/checkpoint tests include three frozen calibration
  seeds for old/new selection/prediction parity, independent family choices,
  holdout-label mutation, weighted component importance, native fit/prediction
  deadlines, worker restart, cancellation and checkpoint integrity/path guards.
- Full integration preserves original stages, target masks, weights and groups.
  An interrupted sweep fits only the unfinished target on retry. Full/study
  options resolve and validate; changed model settings invalidate the original
  fingerprint. New report CSVs participate in full checkpoint verification.
  Old reports and earlier diagnostic labels remain readable.
- Final verification completed three all-candidate standard tasks, a seven-family
  stress search and scale search without candidate/comparison failures or
  unavailable models. All 230
  candidate artifacts reproduce holdout predictions within absolute/relative
  tolerance 1e-12. Standard reports consolidate 198 candidate comparisons and
  48 independent family/feature-set selections. Saved-model climate-block
  importance is independently reconstructed using the original weights.
- The no-signal control's primary importance remains unsupported.
  31 family/component rows pass a
  numerical support heuristic despite no planted signal. Small score gains
  do not establish reliability: new exports explicitly name this flag
  heuristic_support, and every explanation is labeled exploratory.
  Estimator grids, loss rules and thresholds were not tuned to make the control
  pass. Constant controls do not produce learned feature explanations.

Measured Linux costs include refits, serialization, explanation and parity
checks. Memory conservatively sums parent and peak child/worker RSS; peaks
need not be simultaneous.

| Search / target | Candidate comparisons | Family selections | Wall seconds | CPU seconds | Peak sum MiB |
|---|---|---|---|---|---|
| standard / level_threshold | 66 | 16 | 140.70 | 164.57 | 574.35 |
| standard / level_zero_inflated | 66 | 16 | 103.65 | 114.46 | 648.23 |
| standard / level_no_signal | 66 | 16 | 104.58 | 115.12 | 648.23 |
| stress / level_zero_inflated | 16 | 16 | 34.61 | 37.41 | 648.23 |
| scale / level_threshold | 16 | 16 | 77.94 | 81.21 | 648.23 |

Every declared 300-second / 1,024-MiB resource gate passed. Each standard task
records 396 worker fit calls against a conservative upper bound of 471; the
seven-family searches record 96 against 115. Native stopping/two-stage fits
have their own larger upper bound. Defaults use one worker/native thread.

A separate fixed-selector experiment measured sequential independent tasks at
23.60 seconds wall / 24.15
summed operation CPU seconds and 286.76 MiB peak.
Two concurrent tasks took 13.55 seconds wall /
25.93 operation CPU seconds, with conservative
572.94 MiB peak sum. Startup and memory costs differ;
this is not a universal speedup claim. Concurrent fitting is not enabled in the
released runner.

Measurements used Linux x86_64, Python 3.12.14, scikit-learn 1.8.0, NumPy 2.3.5,
pandas 2.2.3, SciPy 1.17.0, threadpoolctl 3.6.0, packaging 26.3, XGBoost 3.1.3
and LightGBM 4.6.0. Functional checks can be reproduced with:

    PYTHONPATH=src python -m unittest discover -s tests -p 'test_model_comparison.py' -v

This is invented model-stage data. Actual macOS execution, live ERA5 and
private-data release acceptance remain R8. Hosted CI remains disabled.

## R8 — Release and real-data acceptance

Run existing repository checks, the complete fixture case matrix and selected
standard/scale runs. Record dependency versions and test OS/Python combinations,
including the deployment macOS environment. Verify feature guards, weights,
saved model parity, baseline selection, shared splits and climate ablation for
every registered family. Optional packages should not break the existing base
installation. New configuration keys must be accepted by full setup, original
pipeline defaults/validation, fingerprinting and report readers together.

Freeze a candidate grid and acceptance rules using three predeclared calibration
seeds before the final synthetic verification seed. Do not keep changing the
synthetic holdout or grid until a preferred algorithm wins. Numerical tolerances
and bounded runtime/memory limits are documented with the measured environment;
they are not universal performance promises.

For the private pilot, verify location/climate coverage and target support,
review leakage and the intended retrospective/prospective question, compare
every candidate under the same private grouped plan, and inspect sparse/constant
families. Real-data feature effects are predictive associations. Synthetic
oracle recovery does not establish operational accuracy or climate causation.
Promote a new family only after its common gate and relevant model-specific
gate pass, and record which datasets and versions were actually tested.

## Artifact and acceptance checklist

The generated directory includes clean/stress tables, one partition plan,
an explicit feature allowlist and dictionary, per-target truth contracts,
row-level oracles, adversarial controls and file hashes. It is deterministic
within the recorded environment and refuses populated or versioned outputs.
The downloader/cache/live ERA5 path is deliberately outside this model-stage
fixture; a successful fixture run does not mean those stages were executed.

Minimum admission evidence per new family:

1. Instantiate, clone, fit with weights, predict finite nonnegative values,
   serialize/reload and reproduce predictions.
2. Preserve grouped train/stopping/outer-CV/holdout separation and prove the
   holdout-label mutation invariant.
3. Handle mixed inputs, missingness, unseen categories, absent/constant labels
   and insufficient support with explicit results.
4. Report all attempts and compare with the same weighted-median baseline.
5. Recover appropriate planted signal patterns across fixed calibration seeds;
   compare controls without requiring every algorithm to win.
6. Run full-pipeline adapter/report regression checks and document deployment
   dependencies and observed resource cost.

## Primary implementation references

API documentation is a design reference; pin and test actual dependency versions
when implementing each phase.

- [scikit-learn ExtraTreesRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
- [XGBoost scikit-learn estimator guide](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html)
- [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
- [LightGBM LGBMRegressor](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html)
- [scikit-learn SplineTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html)
- [scikit-learn Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
