# Regression model expansion roadmap

Prepared 2026-10-31. Scope: ExtraTrees, XGBoost, LightGBM, Gaussian additive
spline models, and a two-stage occurrence/severity model for the existing
deduct-family sweep. Each target remains a separate regression task for each
pavement surface. DataRobot's proprietary blueprints are not reproduced exactly.

The observed-level search contains a weighted-median baseline, two Elastic
Net, two Random Forest and four ExtraTrees configurations. R0 provides the
reproducible dataset and benchmark runner. Installing the optional XGBoost
extra adds eight configurations. Build 1 completes R1 with a versioned
registry and common evaluation/failure contract. Build 2 completes R2 with
ExtraTrees; Build 3 completes R3 with fixed-budget and grouped-stopping XGBoost.
**LightGBM, additive splines and two-stage models remain planned.**

**Roadmap progress: 44.4% — 4 of 9 phase gates complete (R0 through R3).** Phases R0
through R8 have equal weight for this count. This is accepted implementation
coverage, not an estimate of elapsed engineering effort. Build publication,
merge status and private deployment acceptance are reported separately.
New algorithm families delivered: 2 of 5. The next build is R4, LightGBM.

## Delivery sequence and dependencies

| Phase | Deliverable | Depends on | Completion gate | Current status |
|---|---|---|---|---|
| R0 | Synthetic data, truth contracts and benchmark runner | Existing local search | Reproducible files, isolated groups and existing-model smoke run | Complete — test kit |
| R1 | Candidate registry and common evaluation contract | R0 | Existing candidates retain their selection/prediction behavior | Complete — Build 1 |
| R2 | ExtraTrees | R1 | Weighted fitting, mixed inputs, serialization and nonlinear fixtures pass | Complete — Build 2 |
| R3 | XGBoost | R1, R2 evaluation pattern | Optional dependency and nested stopping tests pass | Complete — Build 3 |
| R4 | LightGBM | R1, R3 stopping pattern | Weighted validation, category and platform checks pass | Planned |
| R5 | Gaussian additive spline model | R1 | Additive shapes, weights and extrapolation checks pass | Planned |
| R6 | Two-stage occurrence/severity model | R1 and one accepted severity estimator | Zero/positive/sparse training cases and prediction semantics pass | Planned |
| R7 | Consolidated comparisons, explanations and runtime controls | R2–R6 | Every task reports candidates, failures, scores and importance consistently | Planned |
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

**Implementation.** Add a separate optional `lightgbm` extra and validate its
installation on supported macOS/Linux environments. Begin with the same encoded
feature path as the existing estimators. Native categorical support is a later
candidate variant, not an implicit schema change. Proposed first grid: 15/31
leaves, minimum leaf observations 10/30, learning rate 0.05, 300 trees.

Reuse R3's training-only stopping design, adapted to the pinned LightGBM API.
Record feature order and any categorical mapping explicitly. Unknown holdout
categories and missing-value sentinels must be handled consistently. Do not
assume callback/validation parameter names remain identical across versions.

**Tests.** Threshold/interaction targets, stress schema, evaluation weights,
rare target values, serialization and installation/dependency diagnostics.

**Gate.** Weighted fixed-budget and early-stopping tests pass; native-category
variants, if introduced, have independent schema and unseen-category coverage.

## R5 — Gaussian additive spline model

**Implementation.** Start with a Gaussian identity-link additive regression:
univariate B-spline bases from `SplineTransformer` plus weighted Ridge fitting.
Label this explicitly as additive spline regression; it is not a byte-for-byte
replacement for a proprietary DataRobot GAM or a pyGAM smoothing implementation.
Proposed grid: 5/8 knots, degree 3, Ridge alpha 0.1/1.0, constant extrapolation.
Use additive one-hot terms for categoricals and separate missing indicators.
Avoid applying spline expansion to identifiers or arbitrary category codes.

**Tests.** Recover the planted smooth shapes from `level_smooth`, check linear
reference performance, all-missing/constant predictors, boundary/extrapolation
behavior, sample-weight forwarding and saved prediction parity. Check that an
additive model's weaker performance on pure interactions is reported honestly.

**Gate.** Shape and prediction tests pass across fixed calibration seeds.
Any later pyGAM variant must separately prove weighted fitting, cloning,
serialization and platform support before registry admission.

## R6 — Two-stage occurrence and positive severity

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

## R7 — Comparisons, explanation and bounded execution

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

The runner currently selects the champion and computes its permutation
importance. R1 supplies a required-candidate selector and candidate-level CV
status records. R7 still adds comprehensive per-candidate prediction and
serialization reports, so new families cannot escape direct acceptance tests
merely by losing selection. The smoke command exercises implemented registry
entries only.

Precompute the fit budget as targets × surfaces × feature sets × outer folds
× configurations, plus refits/stopping fits. Start with one fit worker and one
estimator thread, then measure controlled parallelism. Add candidate/fold
progress events, cancellation and isolated resumable task outputs. Resume keys
must include data/plan hashes, candidate parameters, seed, feature rules and
dependency versions. Preserve the original full-pipeline code-change guards.

**Gate.** A model that raises or times out is visible in the result. Cancellation
keeps completed artifacts intact. Old reports still load. Importance on no-signal
and constant targets is not presented as reliable explanatory evidence.

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
