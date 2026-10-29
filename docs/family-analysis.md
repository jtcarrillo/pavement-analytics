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
`Inspected Surface Type`. Known canonical fields take precedence over secondary
aliases; unresolved ties produce a private candidate report. Text identities
retain leading zeros. `IsConstruction` is recognized as the construction flag,
so construction observations reset pairing.

Date-only values and common timestamps are normalized to inspection calendar
dates before pairing. Native Excel dates and year-first dates are supported.
For numeric month/day or day/month dates, the entire source file must provide a
consistent, unambiguous order; the first row and workstation locale do not decide.
A value valid in only one order can resolve other ambiguous values in that file.
Files are assessed independently. An ambiguous file or conflicting orders stop
setup with `AMBIGUOUS_DATE_FORMAT_USE_DATE_FORMAT` or
`MIXED_DATE_FORMATS_USE_DATE_FORMAT`.

For a known format, add `--date-format "%m/%d/%Y"` for month-first dates,
`--date-format "%d/%m/%Y"` for day-first dates, or, for example,
`--date-format "%m/%d/%Y %I:%M:%S %p"` for month-first timestamps with AM/PM.
The equivalent private options key is `date_format`. Explicit formats bypass
automatic normalization and apply to every source. If both mechanisms are used,
their values must agree. Numeric serial dates, two-digit years and timezone
suffixes are not inferred. Convert serials or timezone-bearing values to inspection
calendar dates privately, or supply an appropriate explicit text-date format.

Automatic normalization preserves the original date text in a unique
`source_extra_date` column and leaves the source snapshot unchanged. Invalid and
blank dates remain row rejections. In automatic mode, timestamp clocks are
validated before time is discarded, and intervals use calendar dates. Explicit
timestamp formats retain the legacy elapsed-day calculation. The private
`date_parsing_report.json` records the per-source decision and counts, including
when ambiguity stops setup. Successful setup also embeds this report in the
source entries in `setup_report.json`.

The selection rules are independent of column order:

| Role | Automatic precedence |
| --- | --- |
| Section | Unique `SectionUniqueID` over `SectionID` or `section` |
| Surface | Unique `Inspected Surface Type` over `Surface`, `Surface Type` or `Material` |
| Validation group | Explicit `group_id`; otherwise `NetworkID` over `Network` only under the PAVER profile |
| Inspection date and PCI | `Date` and `PCI` over other aliases only under the PAVER profile |

The PAVER profile requires unique canonical `SectionUniqueID`, `Date`, `PCI`,
`Inspected Surface Type` headers and a recognized deduct column. Capitalization
and separator variants are recognized. Multiple equivalent spellings of the
preferred field remain ambiguous; an exact-header private override can resolve
them. A generic `Surface` plus `Material`, for example, still requires a choice.
No alias is selected because its values look more complete.

Selected fields normalize to the pipeline's role names. If an unselected source
field already uses that name, it is preserved under `source_extra_<name>` with a
unique suffix when needed. These preserved fields are excluded from predictors.
Original source snapshots remain unchanged. Duplicate normalized deduct columns
still fail to prevent double counting. Different validation-group field types
across source files require private reconciliation.

`column_mapping_report.json` in the generated workspace records candidates,
selected fields and selection reasons, even if setup fails. Successful runs also
include these choices and the complete original-to-normalized header mapping in
`results/setup_report.json`. For an unresolved role, fill the report's override
template with an exact candidate header in a private options file and rerun with
`--options`. Keep these diagnostics private.

Recognized two-digit distress codes include both L/M/H and ungraded deduct columns.
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
deduct points per year. L/M/H and ungraded channels are summed subject to the
same-code overlap check below. Sums cover columns actually exported by
each source; confirm equivalent family inventories before comparing sources.
The private setup report records the exact columns. By default, blank or invalid
exported values are not assumed zero. Export gaps exclude deduct targets but retain the
PCI-rate task. Columns structurally absent from another file do not invalidate
the current source. These sums are not corrected PCI contributions or Shapley
attributions, and no correction curves are required for this workflow.

If empty exported deduct cells mean absent distress in your source, declare that
policy explicitly:

```bash
pavement analyze "/path/to/private/data" --blank-deducts zero --open
```

The equivalent private options key is `blank_deducts` with value `zero`;
`missing` is the default. A CLI value and an options value must agree when both
are supplied. The zero policy applies only to blank/whitespace cells in recognized
exported deduct channels. It does not fill columns absent from a source, invent
absent families, or repair text such as `N/A`, infinities, or values outside 0–100.
Original snapshots and normalized inputs are unchanged by this policy. Converted
values and per-endpoint blank/fill counts appear in the derived pair dataset.

Any remaining invalid exported deduct excludes every family for that pair under
the conservative validation rule. A low-PCI inspection with all recognized
deducts zero still triggers the existing export-gap rule. Inspect
`results/deduct_audit.json` and the dashboard's availability section for the
policy, cell counts, exclusions, and per-target coverage. Counts refer to retained
pair endpoints; an inspection shared by two consecutive pairs is counted twice.
Ungraded headers such as `73 deduct` are recognized and listed in the audit.
When an inspection has positive ungraded and graded values for the same code,
their relationship is ambiguous: they could double count a deduction. That
inspection is flagged, and pairs touching it are excluded from deduct targets
until the source is reconciled privately. The pipeline does not guess that an
ungraded value is a separate severity or a redundant total. Zero-valued alternative
channels do not trigger the overlap flag. Source inventories may still differ;
review coverage before comparing family sums.

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

If every row fails date parsing, the command returns
`NO_VALID_DATES_CHECK_DATE_FORMAT`. Other empty pairing outcomes retain
`NO_VALID_PAIRS`. In either case, `failed/run-*/pairing_report.json` contains
accounting and reason totals beside `rejected_rows.csv` and `transitions.csv`.
Resolve the reported issue in a new invocation; spacing and reset rules remain
in force after dates are accepted.

| Output | Contents |
| --- | --- |
| `results/dashboard.html` | Offline filters, error comparisons and actual-versus-predicted charts |
| `results/target_summary.csv` | One row per material/target, status, coverage and holdout errors |
| `results/model_leaderboard.csv` | Consolidated development CV candidate comparisons |
| `results/predictions.csv` | Combined predictions with holdout/development partition labels |
| `results/materials/` | Detailed per-target models, predictions, folds and summaries |
| `results/run_report.json` | Inspection accounting and target outcomes |
| `results/pairing_report.json` | Inspection accounting and rejection/transition reason totals |
| `results/deduct_audit.json` | Blank policy, fill/exclusion/overlap counts, ungraded headers and family-target availability |
| `results/setup_report.json` | Input inventory, mappings, features, grouping and source hashes |
| `results/manifest.json` | Reproducibility fingerprints and output checksums |
| `run.json`, `inputs/`, `originals/` | Generated configuration, normalized inputs and unchanged source snapshots |

Charts distinguish development CV from held-out evaluation. Lower weighted MAE
is better; negative improvement means worse than the condition baseline. A zero
baseline error has no defined percentage improvement. Inspect coverage and
per-target results instead of ranking unrelated target scales by raw error.
Scatters may be capped for rendering; exported prediction tables remain complete.
The dashboard is self-contained and makes no external network requests.

`no_valid_target` means no usable observations remained and fitting was skipped.
`no_model` means candidates were evaluated but the condition baseline was retained.
`champion` means a candidate passed the development-CV selection rule; consult its
held-out errors for performance. A completed dashboard can therefore still contain
unavailable family tasks.

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
`date_format`, `blank_deducts`, `cutoff_year`, `min_delta_days`, `max_delta_days`, `folds`, `seed`,
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
