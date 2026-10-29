# Pavement Analytics

Generic Python tools for Pavement Condition Index (PCI), deduct values, and PAVER
inspection exports. Version 0.2.0 provides a runnable curation and analysis
foundation. All operational inputs, configuration, provenance, fitted models,
and results belong in a separate private workspace.

## Implemented

- CSV, TSV and first-worksheet XLSX ingestion; text identifiers remain text.
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
- Immutable run outputs, file hashes and verified resume.
- An SSH adapter that fetches an exact private data commit into an external workspace.
- Publication checks for allowed files, Git history and private denylist matches.

This release does not implement the complete research workflow. Live climate
retrieval, full model tuning, specialized sensitivity analysis and the remaining
adapters are listed in [the roadmap](docs/roadmap.md).

## Install and test

Use Python 3.11, 3.12 or 3.13. The public CI matrix runs the generated tests on
Ubuntu with all three versions and on macOS with Python 3.12 and 3.13. Check the
CI results for the revision you install. Use an approved package mirror on a
restricted network, and keep the environment outside the source checkout.

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

## Run all deduct families from a data path

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

All results stay under the configured private output directory. Console output
contains fixed status/error codes only. Detailed diagnostics stay in private
`logs/runtime.log`. No automatic publication, telemetry or external model service
is part of the analysis pipeline.

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
inside the private environment before publishing. CI is a secondary check and
runs only after content has reached GitHub.

## Documentation

- [Architecture and trust boundary](docs/architecture.md)
- [Private configuration and data contracts](docs/contracts.md)
- [Deployment and repository setup](docs/deployment.md)
- [Scope, validation and next components](docs/roadmap.md)

The project provides analysis tools, not an authoritative replacement for a
specified pavement survey standard. Validate privately supplied curve profiles
and procedures against the approved reference implementation before relying on
recomputed PCI.
