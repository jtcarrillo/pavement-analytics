# Analyze deduct families from a data path

After the one-time installation in the README, run on an authorized private
workstation with a local file or a directory of exports:

```bash
pavement analyze "/path/to/private/data" --open
```

Activate the environment used for installation in each new terminal. If the
`pavement` command is not found, the equivalent command is
`python -m pavement_analytics analyze "/path/to/private/data" --open` using that
environment's Python. See [installation troubleshooting](deployment.md#installation-and-test-troubleshooting).

No environment variables or hand-written run configuration are required for
`analyze`. The command explicitly requests local private execution; it still
refuses execution in GitHub Actions. A private repository already checked out
behind the firewall can be the input source. This command does not fetch a remote
repository; use the existing pinned `sync` workflow when fetching is required.

CSV, TSV and XLSX are supported. A directory means all supported files directly
inside it, excluding temporary Excel lock files; subdirectories are not searched.
Supply a folder containing only the intended exports, or supply one file.
XLSX uses the first worksheet and cached formula values. Original files are never
modified. Symlinks and inputs in the public code tree are rejected.

## Automatic setup

Common spellings of section ID, inspection date, PCI, surface type and group ID
are recognized, including `SectionUniqueID`, `Date`, `PCI`, and
`Inspected Surface Type`. Ambiguous mappings fail with a fixed error code instead
of guessing. Text identities retain leading zeros. Dates default to ISO
`YYYY-MM-DD`; native Excel dates are converted automatically. Other CSV date
formats require a private options override.

Every recognized two-digit distress code and L/M/H deduct column is included.
The following editable analytical groupings are defaults, not standardized causal
attributions. Family names describe hypotheses rather than diagnoses.

| Family | Codes |
| --- | --- |
| pcc_freeze_thaw | 64, 70, 74, 75 |
| pcc_thermal | 61, 63, 65 |
| pcc_moisture | 69, 71, 76 |
| pcc_load | 62, 72 |
| pcc_other | 66, 67, 68, 73 |
| ac_thermal_crack | 43, 47, 48 |
| ac_aging | 52, 57 |
| ac_high_temp | 42, 53, 54 |
| ac_load | 41, 45 |
| ac_subgrade | 44, 56 |
| ac_other | 46, 49, 50, 51, 55 |

Unmapped codes receive individual `deduct_NN` families. Use `--by-code` to group
every observed code individually instead. All configured families get a task
status for each observed material; absent families are not invented zero targets.

For each family, the target is `(sum_end - sum_start) * 365 / delta_days`, in
deduct points per year. L/M/H are summed. Sums cover columns actually exported by
each source; confirm equivalent family inventories before comparing sources.
The private setup report records the exact columns. Blank or invalid exported
values are not assumed zero. Export gaps exclude deduct targets but retain the
PCI-rate task. Columns structurally absent from another file do not invalidate
the current source. These sums are not corrected PCI contributions or Shapley
attributions, and no correction curves are required for this workflow.

The default predictor is **starting PCI only**. Mean and condition baselines,
ElasticNet and RandomForest are compared using grouped cross-validation. This
is a reproducible starting-condition analysis, not a climate or causal model.
Additional numeric start-state predictors must be explicitly declared privately;
identifiers, locations, future measurements and raw distress inventories remain
blocked. A more complex candidate must beat the condition baseline consistently
or the baseline is retained (`no_model`).

A recognized group column defines holdout isolation. Without one, section IDs
define groups and the setup report warns that broader group isolation has not
been established. Equal section IDs across sources remain in the same split.
Pairing remains scoped to source file, section, group and surface. Five CV folds
and a 20% group holdout require at least seven eligible groups per target.
Insufficient groups are reported and do not trigger a random row split. Target
missingness can change eligible groups, so target holdouts need not be identical.

## Results

Each invocation creates a new private workspace under
`~/pavement-runs/run-<timestamp>-<id>/`. Use `--workspace /path/to/new/workspace`
to choose a new empty destination. `--open` opens the local dashboard after
success; without it, open `results/dashboard.html` in the generated workspace.
Console diagnostics contain fixed status/error codes only. Failure details stay
in the workspace's `logs/runtime.log`.

| Output | Contents |
| --- | --- |
| `results/dashboard.html` | Offline filters, error comparisons and actual-versus-predicted charts |
| `results/target_summary.csv` | One row per material/target, status, coverage and holdout errors |
| `results/model_leaderboard.csv` | Consolidated development CV candidate comparisons |
| `results/predictions.csv` | Combined predictions with holdout/development partition labels |
| `results/materials/` | Detailed per-target models, predictions, folds and summaries |
| `results/run_report.json` | Inspection accounting and target outcomes |
| `results/setup_report.json` | Input inventory, mappings, features, grouping and source hashes |
| `results/manifest.json` | Reproducibility fingerprints and output checksums |
| `run.json`, `inputs/`, `originals/` | Generated configuration, normalized inputs and unchanged source snapshots |

Charts distinguish development CV from held-out evaluation. Lower weighted MAE
is better; negative improvement means worse than the condition baseline. A zero
baseline error has no defined percentage improvement. Inspect coverage and
per-target results instead of ranking unrelated target scales by raw error.
Scatters may be capped for rendering; exported prediction tables remain complete.
The dashboard is self-contained and makes no external network requests.

All generated files, including synthetic inputs/results, remain outside the public
repository. Never upload the dashboard, logs, fitted models or exports to public
issues, Actions artifacts or the source tree.

## Optional private overrides

For an unusual schema or additional reviewed predictors, place a JSON options
file outside the public checkout, then run:

```bash
pavement analyze "/path/to/private/data" --options "/path/to/private/options.json" --open
```

Options accept `columns` (logical role to original source header), `features`
(post-normalization pair-table names, including `PCI_start`), `deduct_families`,
`date_format`, `cutoff_year`, `min_delta_days`, `max_delta_days`, `folds`, `seed`,
`holdout_fraction`, `association_policy`, and `models`. Unknown options fail.
Column overrides apply to each file. Core headers normalize to section, date,
pci, surface, group and construction; other predictor headers remain unchanged
and receive `_start` during pairing. Overriding families replaces the defaults;
codes must not overlap between families. No example populated configuration is
stored in the repository.

To verify an unchanged completed run, activate the legacy execution opt-in and
use its generated configuration:

```bash
export PAVEMENT_PRIVATE_EXECUTION=1
pavement --workspace "/path/to/completed/workspace" run \
  --config "/path/to/completed/workspace/run.json" --resume
```

Resume verifies configuration, inputs, original snapshots, code, environment and
all output files. It does not refit. Changed options or code require a new run.
