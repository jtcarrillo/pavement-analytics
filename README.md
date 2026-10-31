# Pavement Analytics

Generic Python tools for Pavement Condition Index (PCI), deduct values, and PAVER
inspection exports. Version 0.5.4 provides a runnable curation and analysis
foundation. All operational inputs, configuration, provenance, fitted models,
and results belong in a separate private workspace.

## Implemented

- CSV, TSV and first-worksheet XLSX/XLSM ingestion; text identifiers remain text.
- Guarded, auditable cell corrections without altering the source exports.
- Surface classification, rejection accounting and consecutive inspection pairs.
- A 365-day minimum interval by default; increases and construction reset pairing.
- Deduct-family growth targets and missing/invalid deduct checks.
- Curve-driven PCI calculations and an exact Shapley library function. Approved
  correction curves must be supplied privately; no standard tables are bundled.
- Condition-adjusted group rankings with shrinkage and rank stability.
- Explicit numeric start-state feature admission, column audit and Spearman review.
- Group-disjoint model selection and a reserved group holdout, with mean and
  condition-only baselines, ElasticNet and RandomForest candidates.
- Observed deduct-level searches add four ExtraTrees settings through a
  versioned model registry, with weighted fitting and candidate failure audits.
- Optional XGBoost adds four fixed-budget and four grouped-stopping settings,
  with CPU-only fitting and recorded iteration counts and stopping groups.
- Optional LightGBM adds eight CPU settings through the same evaluation plan,
  with encoded feature schemas and actual/refit iteration counts in its audits.
- Immutable run outputs, file hashes and verified resume.
- An SSH adapter that fetches an exact private data commit into an external workspace.
- Publication checks for allowed files, Git history and private denylist matches.

The `full` command restores the original inspection-pair pipeline, including
its complete ERA5 calculations, distress preparation, PCI reconstruction and
Shapley decomposition, then runs our own local models for each deduct-family
total. DataRobot is retired; no SDK, account, upload or external training service
is required. See [the full pipeline guide](docs/full-pipeline.md).

The separate `study` command adds location matching, cached ERA5-Land enrichment, observed
deduct-family totals, shared validation splits, climate ablation and feature
importance. Full model tuning and specialized sensitivity analysis remain on
[the roadmap](docs/roadmap.md).

## Install and test

Use Python 3.11, 3.12 or 3.13. Hosted CI is currently disabled; run generated
regression tests and source publication checks locally for each build. The
previous hosted matrix covered Ubuntu with all three versions and macOS with
Python 3.12 and 3.13. Validate the revision on the deployment workstation. Use an
approved package mirror on a restricted network, and keep the environment
outside the source checkout.

```bash
python3 -m venv "$HOME/.venvs/pavement-analytics"
source "$HOME/.venvs/pavement-analytics/bin/activate"
python -m pip install -r requirements-tested.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
```

Tests generate mathematical examples at runtime. There are no input files or
saved example datasets in the repository. For an installed command that is not
found, or a macOS temporary-path test failure, see
[installation and test troubleshooting](docs/deployment.md#installation-and-test-troubleshooting).

For a complete first run, follow the [synthetic example walkthrough](docs/synthetic-example.md).
It generates invented records in a new external workspace, runs the models and
rankings, and verifies resume. Only the generator source is stored here.

## Run the entire original pipeline with local models

```bash
python -m pip install -e '.[original]'
pavement full "/path/to/private/data" --open
```

This runs every original preparation stage and then compares end-inspection
deduct-family totals using local baseline, ElasticNet, random-forest and
ExtraTrees models.
Install the optional boosting backend to include its eight configurations
automatically in `full` and `study`:

```bash
python -m pip install -e '.[original,xgboost]'
```

The supported XGBoost range is `>=3.1.3,<3.2`; admission was tested with 3.1.3.
Absent or unsupported optional candidates remain visible in model comparisons
while the base candidates continue. Dependency changes require a new run under
the full-pipeline resume checks.

To include both optional boosting families:

```bash
python -m pip install -e '.[original,xgboost,lightgbm]'
```

LightGBM is bounded to `>=4.6.0,<4.7`, with 4.6.0 used for admission. See
[deployment instructions](docs/deployment.md) for its installation checks.
It retains the original inspection intervals and climate definitions, including
PG, cooling, rainflow and construction-aging calculations. The final offline
dashboard ranks target predictability and shows the best predictors per family.
`--open` opens a local progress page immediately and the final dashboard after
success. The terminal and progress page show selected files, stage counts,
weather cache/download activity, completed locations and climate-feature rows.
Subprocess heartbeats keep long waits visible. The readable terminal output is
also saved in `logs/progress.log`; add `--quiet` to suppress that terminal summary.
Setup also lists unavailable auxiliary paths and invalid mapping entries with
reasons; review `full_setup_report.json` before starting a new run.

To view the latest run that is already running, use a second terminal:

```bash
pavement monitor --open
```

You can supply a specific workspace as `pavement monitor "/path/to/private/run" --open`.
Monitoring an older run uses its existing logs and checkpoints; it does not
restart the pipeline. The [live progress guide](docs/full-pipeline.md#live-progress-and-existing-runs)
explains what can be observed while CDS is waiting.

The command reuses a private `pavement-full.json`, `pipeline.json`, or
`pipeline.example.json` beside the data or in its parent. Your existing location
mapping, location CSVs, CDS credentials and private CDV curve profile are still
needed. The original `dr-multitarget/cdv_curves.json` can be discovered privately;
it is not included in this public repository. Legacy DataRobot settings are
ignored and recorded. The [full pipeline guide](docs/full-pipeline.md) covers
discovery, optional curve fitting, local model validation and stage resume.

## Alternative inspection-level study

Install the climate extra once, configure your CDS account, and keep the private
location mapping beside the data or in a saved private study profile:

```bash
python -m pip install -e '.[climate]'
pavement study "/path/to/private/data" --blank-deducts zero --open
```

Use `--blank-deducts zero` only for an export whose blanks mean absent distress.
The command runs ingestion, inspection checks, location matching, ERA5-Land,
one model search per material/deduct-family total, and a consolidated offline
dashboard. It ranks predictability relative to a baseline and shows the most
useful predictors for each target. It uses historical climate before each
inspection (365 days by default), not annual deduct changes. Same-inspection
PCI and raw distress values cannot be predictors. See the
[study setup and interpretation guide](docs/deduct-study.md) for the one-time
location/credential setup, exact climate features, and saved profile discovery.

## Legacy analysis of annual deduct change

After installation, on your private workstation:

```bash
pavement analyze "/path/to/private/data" --open
```

This accepts a CSV/TSV/XLSX file or a folder of exports, detects common headers,
creates the family targets, runs each material/target, and opens a consolidated
offline dashboard. It writes a new workspace under `~/pavement-runs/`, including
CSV summaries, predictions, models and provenance. No input or output is written
into this repository. No environment variables or configuration are required
for this command. Default predictors use starting PCI only; unsupported or
ambiguous schemas require private overrides. Read the
[family-analysis guide](docs/family-analysis.md) for the default family map,
grouping assumptions, output definitions and optional predictor settings.

Common dates and timestamps are normalized automatically per source. When
month/day order cannot be resolved, supply `--date-format "%m/%d/%Y"` for
month-first dates or `--date-format "%d/%m/%Y"` for day-first dates; include
time directives if present. See the guide for private date and pairing diagnostics.

If the export uses empty deduct cells to mean absent distress, run with
`--blank-deducts zero`. This explicit policy fills only blanks in exported deduct
columns; malformed values remain invalid and absent columns remain unobserved.
The dashboard records the policy, fill counts, and target exclusions. Without
that option, blanks remain missing.
Both `NN deduct` and `NNL`/`NNM`/`NNH deduct` channels are recognized. Positive
values in both ungraded and graded forms of one code at an inspection are flagged
for private reconciliation and excluded from deduct targets to avoid double counting.

## Run privately

Run on an authorized workstation or private job system. Never attach a runner
with private network access to this public repository. GitHub documents the
[risks of public repositories using self-hosted runners](https://docs.github.com/en/actions/reference/security/secure-use#self-hosted-runners).

```bash
export PAVEMENT_PRIVATE_EXECUTION=1
export PAVEMENT_WORKSPACE="$HOME/pavement-work"
pavement --workspace "$PAVEMENT_WORKSPACE" init
```

Edit `run.json` in that external workspace. Set the input directory, exact input
file list, column mapping and any deduct families. The empty input list created
by `init` intentionally cannot run until configured. Configuration is never
stored with the source code.

For a private Git source, an administrator sets `PAVEMENT_DATA_REMOTE` through
the internal runtime environment and configures SSH authentication and trusted
host keys. `PAVEMENT_DATA_COMMIT` must contain the approved full commit SHA.

```bash
pavement --workspace "$PAVEMENT_WORKSPACE" sync --commit "$PAVEMENT_DATA_COMMIT"
pavement --workspace "$PAVEMENT_WORKSPACE" run --config "$PAVEMENT_WORKSPACE/run.json"
pavement --workspace "$PAVEMENT_WORKSPACE" run --config "$PAVEMENT_WORKSPACE/run.json" --resume
```

After `sync`, set `input_root` in the private configuration to the desired
subdirectory under `sources/<commit>`. The program reads explicit files only;
it does not discover or execute code from the data repository. Git LFS objects
need an approved internal materialization step; unresolved pointers are rejected.

All results stay under the configured private output directory. `full` displays
local file paths, progress and aggregate counts by default; `full --quiet`
prints only fixed terminal status/error codes. Detailed diagnostics stay in
private `logs/runtime.log`. No automatic publication, telemetry or external model
service is part of the analysis pipeline.

## Publish source changes

Use a dedicated code-only repository with a clean history. Install the local
hooks immediately after initializing or cloning it:

```bash
python tools/install_hooks.py
python tools/check_public_tree.py --working-tree
```

Before a push, set `PAVEMENT_DENYLIST_FILE` to an external, nonempty file containing
private names, identifiers, source basenames, aliases, hostnames and other terms
that must never be published. The pre-push hook requires that list and scans
reachable history. Do not commit the list or put it in GitHub secrets.

Read [SECURITY.md](SECURITY.md) and [deployment](docs/deployment.md) before the
first push. A clean automated scan is not proof that arbitrary source prose
contains no private information; review the exact outbound source and history
inside the private environment before publishing. Hosted CI is disabled. If
re-enabled later, it remains a secondary check after content reaches GitHub.

## Documentation

- [Architecture and trust boundary](docs/architecture.md)
- [Private configuration and data contracts](docs/contracts.md)
- [Deployment and repository setup](docs/deployment.md)
- [Scope, validation and next components](docs/roadmap.md)
- [Regression model expansion and synthetic test dataset](docs/regression-model-roadmap.md)

The project provides analysis tools, not an authoritative replacement for a
specified pavement survey standard. Validate privately supplied curve profiles
and procedures against the approved reference implementation before relying on
recomputed PCI.
