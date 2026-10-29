# Deduct totals, climate, and predictor comparison

`pavement study` runs the complete observed-deduct analysis from one inspection
file or folder. It snapshots inputs, normalizes headers/dates, audits individual
inspections, attaches locations, obtains cached ERA5-Land exposures, constructs
the deduct targets, fits each material/target task, and consolidates results.
The older `pavement analyze` command still analyzes annual changes using paired
inspections; use `study` for observed deduct values and climate predictors.

## One-time setup

Install the package and its climate client in your workstation environment:

```bash
python -m pip install -e '.[climate]'
```

Set up the CDS account/API credentials as described by the
[official CDS instructions](https://cds.climate.copernicus.eu/how-to-api), and
accept the dataset terms in your account for
[ERA5-Land hourly time-series](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-timeseries).
Keep credentials in the normal CDS client configuration, never this repository.
This application never writes credentials into its run configuration.

Locations can come from `latitude`/`longitude` columns, or from the earlier
filename-to-location-file mapping. With raw PAVER exports, retain this private
layout (or use the profile below):

```text
private-data/
  inspections.csv
  data_file_mapping.json
  location.csv
```

The mapping JSON maps the exact inspection basename to the location CSV:

```json
{"inspections.csv": "location.csv"}
```

Each location CSV needs a `LOCATION(DEGREES)` column containing one unambiguous
decimal latitude/longitude pair, optionally with hemisphere letters. Repeated
identical coordinates are accepted. Referenced location CSVs are excluded from
inspection-file discovery. No geocoding or location guessing is performed.
The mapping is discovered in the input folder or its parent. Location filenames
are relative to the mapping folder unless a profile supplies `csv_dir`.

If those files live elsewhere, save `pavement-study.json` beside the data or in
its parent. Alternatively use `~/.config/pavement-analytics/study.json` or pass
`--options /path/to/profile.json`. The first matching profile is used, without
merging multiple files. Relative paths resolve from that profile's directory.

```json
{
  "blank_deducts": "zero",
  "climate": {
    "mode": "era5",
    "mapping": "metadata/data_file_mapping.json",
    "csv_dir": "metadata",
    "cache_dir": "cache/era5",
    "lookback_days": 365
  }
}
```

The cache must be outside the inspection input directory, the new run directory,
and every Git checkout. If omitted it defaults to `~/pavement-runs/era5-cache`
(or a sibling of an explicit run workspace). Place a profile with `cache/era5`
in the **parent** of the data folder, so its cache is a sibling of the inputs.
Mapping files and location CSVs are snapshotted privately for provenance.

After that one-time setup, run:

```bash
pavement study "/path/to/private/data" --open
```

Without a profile, pass `--blank-deducts zero` when the export uses blanks for
absent distress. Otherwise blanks remain unknown. Add `--by-code` to study one
summed deduct per distress code instead of the default families. This does not
create separate low/medium/high-severity targets. A private `deduct_families`
mapping can define other nonoverlapping code groups. The default family map is
documented in [family-analysis.md](family-analysis.md).

No live climate download occurs during public CI. On the first workstation run,
CDS downloads can take substantial time; requests are split into at most one
calendar year and skip gaps outside needed exposure windows. Subsequent identical
location/date chunks reuse verified cached responses. Download errors stop the study and
leave a private audit. There is no silent climate-free fallback.

## Target and population

Each observation is one inspection. A family target is the **sum of its exported
deduct channels at that inspection**, without annualization, interval weighting,
or dependence on a preceding inspection. Construction records, first inspections,
PCI increases and short intervals are eligible. Exact repeated inspections
collapse; conflicting records for the same group/section/date are excluded.

Malformed/negative/over-100 channel values and overlapping positive ungraded
and graded representations invalidate the affected family. Other families can
remain usable. An all-zero exported distress inventory with valid PCI below
99.5 is an export gap, not confirmed zero damage. A missing code/severity column
is unobserved, not zero; pooled totals may cover different exported channels.
Review `observation_audit.json` before interpreting cross-source comparisons.
The family names are analytical groupings, not established causal diagnoses.

## Climate and other predictors

The ERA5-Land time-series API provides hourly 2 m air temperature, skin
temperature and de-accumulated total precipitation. Temperatures are converted
from Kelvin to Celsius and precipitation from metres to millimetres. Features
use `[inspection_date - lookback_days, inspection_date)`; the inspection day is
excluded. Days are UTC, and all required hours/variables must be present.
Incomplete windows and missing locations are excluded from models and audited.

The 25 historical features cover:

- Air mean/minimum/maximum, average daily minima/maxima and daily temperature range.
- Freezing hours, days spanning freezing, and hours above 30 degrees Celsius.
- Skin mean/minimum/maximum, daily range and degree-hours above 40 degrees Celsius.
- Maximum 3/6/12-hour cooling, 6-hour cooling events exceeding 10 degrees Celsius
  with at least 24 hours between peaks, and maximum cooling excursion.
- Rainflow thermal cycles of at least 10/20 degrees Celsius and squared thermal
  ranges. The squared-range measure is an exposure proxy, not calibrated damage.
- Total precipitation, maximum daily precipitation and days with at least 1 mm.

Whole-history pavement-grade estimates and construction-age thermal integrals
from the earlier interval worker are **not** reused: their windows do not match
this inspection-level historical exposure contract. The new study is not an
exact numerical reproduction of that worker. ERA5 is a gridded reanalysis;
its skin temperature is not a direct pavement-temperature measurement. A one-year
lookback is configurable and does not represent lifetime cumulative exposure.

Reviewed available metadata are admitted automatically: years since major/global/
localized maintenance, subgrade category, inspected branch use, area, slabs,
construction flag, allowable gross loads/passes, and ACN/PCN ratios. A profile's
`features` list can restrict these reviewed fields. The private covariate audit
lists every retained and dropped column. PCI/SCI/FOD, raw distress quantities,
other deduct targets, identifiers, coordinates and dates are excluded from
model predictors. These exclusions prevent direct outcome leakage; they do not
remove all confounding in observational data.

For climate columns already computed under a verified historical contract:

```json
{"climate": {"mode": "existing", "features": ["historical_air_mean_c"]}}
```

This explicit mode performs no downloads. It requires finite numeric values in
every declared climate field and flags historical timing as unverified. Merely
naming a column cannot prove its temporal provenance. Use the ERA5 mode when
the application should generate the historical exposure itself.

## Shared evaluation and ranking

One seeded group assignment is frozen before filtering by material or target:
20% of groups are reserved for holdout and the rest form five CV folds. Profile
keys `seed`, `folds`, and `holdout_fraction` override these defaults. Declared
network/group identity is used when available. Groups sharing an ERA5 grid
location are joined transitively before splitting. If only section identity is
available with precomputed climate, site-independent performance is not proven.

Each task compares a group-weighted median baseline, two ElasticNet settings,
and two random-forest settings. Imputation, scaling and categorical encoding
are fitted within each training fold. CV selects the model; the selected model
is fitted on development groups only. The saved model never trains on holdout.
Each group has equal total weight within each evaluation set. Missing targets
do not trigger a new split. Insufficient group/fold coverage and constant targets
get explicit statuses instead of fabricated accuracy or importance.

The dashboard ranks targets by holdout `MAE skill = 1 - MAE / baseline_MAE`:
positive values beat the baseline, zero matches it, and negative values are
worse. A zero baseline error makes this ratio undefined. Weighted R-squared,
MAE, RMSE, nonzero prevalence, and row/group support are shown alongside it.
Targets with fewer than 5% nonzero observations receive a descriptive mostly-zero flag.
Raw MAE is not comparable across differently scaled deduct families. Even skill
and R-squared can differ because of target prevalence and eligible populations.
Sweeping many targets makes this an exploratory ranking; confirm selected
findings on new data before claiming general performance.

Each target also fits the same candidate set without climate. The report shows
how much climate changes holdout error beyond the other predictors. Five repeated
holdout permutations measure each predictor's increase in MAE, and a joint
permutation measures the whole climate block. Repeat standard deviations measure
permutation variability, not sampling confidence intervals. Correlated features
can share importance, so individual rankings are not unique causal effects.
The report labels importance from weak models as exploratory.

## Private outputs

Every invocation creates a fresh private run under `~/pavement-runs/` unless
`--workspace` supplies a new empty destination. `results/study_status.json`
distinguishes running, failed and completed runs. This mode does not resume
model fitting; reruns reuse the climate cache and create new model artifacts.

| Output | Purpose |
|---|---|
| `dashboard.html` | Offline rankings, prediction plots, importance and coverage |
| `target_summary.csv` | One comparable summary per material/target task |
| `feature_importance.csv` | Consolidated predictor and climate-block importance |
| `model_leaderboard.csv` | CV comparisons for all candidate models |
| `predictions.csv` | Full holdout predictions for all modeled targets |
| `partitions.csv` | Shared fixed group/holdout/fold assignments |
| `materials/<material>/<target>/` | Separate models, scores, predictions and audits |
| `climate_rows.csv`, `climate_audit.json` | Historical windows, coverage, request/cache provenance |
| `observation_audit.json`, `covariate_audit.csv` | Target eligibility, source coverage and predictor admission |
| `manifest.json` | Code/environment/configuration and output checksums |

The dashboard makes no network requests. Only the ERA5 acquisition stage uses
CDS; it sends location coordinates, dates and weather-variable requests, never
inspection records or deduct values. Detailed failures remain in
`logs/runtime.log`. Keep the entire run, mapping, cache and credentials private.
