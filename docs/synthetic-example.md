# Run a synthetic example

This walkthrough runs the implemented pipeline on invented numeric histories.
No actual locations, inventories, exports, or project results are used. The
repository contains only the generator source. All generated inputs,
configuration and outputs are written outside the source checkout and must
never be committed, including in public issues or workflow artifacts.

## Install

Use Python 3.11 or 3.12. These commands are for macOS/Linux in a terminal:

```bash
git clone https://github.com/jtcarrillo/pavement-analytics.git
cd pavement-analytics
python3 -m venv "$HOME/.venvs/pavement-analytics"
source "$HOME/.venvs/pavement-analytics/bin/activate"
python -m pip install -r requirements-tested.txt
python -m pip install --no-deps -e .
```

If you already cloned and installed the project, update the checkout and activate
its environment. Keep the environment outside the checkout.

## Generate, run, and verify

Choose a new empty workspace outside every Git checkout. The generator refuses
to overwrite a populated directory. It does not contact a data repository.

```bash
export PAVEMENT_PRIVATE_EXECUTION=1
export PAVEMENT_WORKSPACE="$HOME/pavement-synthetic-example"

python tools/make_synthetic_example.py --workspace "$PAVEMENT_WORKSPACE"
pavement --workspace "$PAVEMENT_WORKSPACE" run --config "$PAVEMENT_WORKSPACE/run.json"
pavement --workspace "$PAVEMENT_WORKSPACE" run --config "$PAVEMENT_WORKSPACE/run.json" --resume
```

Successful commands print `SYNTHETIC_EXAMPLE_CREATED`, `PRIVATE_RUN_COMPLETE`,
then `VERIFIED_EXISTING_RUN`. The last command verifies the input, code,
environment and output fingerprints without fitting the models again.

This example follows the same execution restrictions as operational runs:
the intent flag is required, and GitHub Actions execution is refused. The flag
does not detect a firewall. Only invented records are appropriate for this demo
outside the approved private environment.

## Read the outputs

The input file is `inputs/synthetic_inspections.csv`; the complete configuration
is `run.json`. Outputs appear under `runs/v1` in the external workspace:

| File | What to inspect |
| --- | --- |
| `run_report.json` | Row accounting and material/target model summaries |
| `dataset.csv` | Consecutive inspection pairs, PCI rates and deduct rates |
| `rejected_rows.csv` | Excluded inspections and their reasons |
| `transitions.csv` | Paired, reset, short-interval and same-day decisions |
| `rankings.csv` | Condition-adjusted rankings by material and synthetic group |
| `features.csv`, `feature_audit.csv` | Admitted numeric features and every column decision |
| `materials/<material>/association.csv` | Feature correlation review |
| `materials/<material>/<target>/leaderboard.csv` | Cross-validation comparison with baselines |
| `materials/<material>/<target>/holdout_predictions.csv` | Predictions for reserved groups |
| `materials/<material>/<target>/summary.json` | Chosen candidate and weighted holdout errors |
| `materials/<material>/<target>/model.joblib` | Fitted estimator; load only files from a trusted run |
| `source_manifest.json`, `manifest.json` | Source provenance, configuration and output checksums |

Materials are `AC`, `APC` and `PCC`. Targets are `DetRatePerYear` (PCI points per
year; negative indicates deterioration) and `rate_example` (growth in the
invented deduct channel per year). Lower weighted mean absolute error (`wMAE`)
means closer predictions. Inspect the reserved-group holdout separately from
the cross-validation scores used to select a model.

The generator includes controlled invalid rows and transition cases. Their
expected behavior can be derived from the generator; empirical run results are
kept outside the repository. If a model does not sufficiently improve on the
condition baseline, `no_model` retains that baseline rather than forcing a
winner.

The numeric `driver_value` is an artificial predictor deliberately used in the
generating equations. PCI and the deduct channel are invented independently.
No approved correction curves are supplied and no PCI recalculation is run.
Good scores here demonstrate software behavior only; they do not establish
accuracy on operational data, valid PCI/CDV curves, or internal SSH connectivity.

## Use private data later

Create a separate operational workspace with `pavement --workspace ... init`,
then configure real file paths, column mappings, features and deduct families
there. If needed, use the pinned SSH `sync` command described in
[deployment](deployment.md). Keep all operational configuration, endpoints,
records and results inside the approved environment.
