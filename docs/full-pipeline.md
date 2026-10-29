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
The command prints fixed status/error codes; detailed logs are private.

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
location JSON retains its original source-filename to location-CSV mapping;
location CSVs contain `LOCATION(DEGREES)`. Referenced location CSVs are excluded
from inspection discovery. Locations and the CDV profile are never guessed.
The original digitized CDV table remains a private input and is not distributed
with this public package.

The command snapshots raw exports, normalizes recognized headers and inspection
dates, and privately copies the selected mapping, location files and profiles.
Original files are unchanged. Ambiguous date order requires `--date-format`.
`full_setup_report.json` records hashes, original-to-normalized source mappings,
date/header decisions, missing inputs and forced full-stage settings.

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

This starts a fresh run after a setup failure; `--resume` requires completed
setup. A compact private profile in the **parent** of the data folder can be:

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

Each material/family task compares a weighted median baseline, two ElasticNet
settings and two random-forest settings. Imputation, numeric scaling and
categorical encoding are learned within training folds. Model selection uses
cross-validation; the selected model is fitted on development groups only and
evaluated on reserved holdout groups. Saved models do not train on holdout.
Original `delta_days` interval weights are used for training and reported errors.

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
`logs/runtime.log` for diagnostics. Missing private curve/location inputs fail
setup explicitly. Live CDS access still requires acceptance testing on the
deployment workstation; automated tests substitute generated weather and never
download private-location climate data or contact a training service.
