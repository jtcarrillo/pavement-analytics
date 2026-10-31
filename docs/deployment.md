# Deployment

Observed deduct-level searches can include the optional XGBoost backend:

```bash
python -m pip install -e '.[original,xgboost]'
```

Use XGBoost `>=3.1.3,<3.2`; the local admission build uses 3.1.3. For that exact
backend, install `xgboost==3.1.3` alongside `requirements-tested.txt`. The base
installation remains usable without XGBoost. Unsupported versions and backend
load failures are retained in candidate audits and model comparisons; explicitly
requested unusable candidates fail preflight. Dependency changes invalidate
full-run resume fingerprints. Validate the installation on the deployment OS.

Install LightGBM independently, or alongside XGBoost:

```bash
python -m pip install -e '.[original,lightgbm]'
# Both boosting backends:
python -m pip install -e '.[original,xgboost,lightgbm]'
PYTHONPATH=src python -m unittest discover -s tests -p 'test_lightgbm.py' -v
```

Use LightGBM `>=4.6.0,<4.7`; admission uses 4.6.0. Its Linux CPU runtime was
tested, and pip wheel resolution was checked for macOS 12+ ARM64 with Python
3.12. A downloaded wheel does not establish successful native loading or
fitting on macOS: run the command above in the deployment environment. If
native loading reports missing `libomp.dylib`, supply the OpenMP runtime; for
Homebrew installations, `brew install libomp` provides it. Follow the
[upstream installation guide](https://lightgbm.readthedocs.io/en/v4.6.0/Installation-Guide.html)
for platform-specific requirements. Actual macOS runtime qualification remains
in the R8 release gate. The program records load failures privately and keeps
base candidates usable; it does not install system dependencies itself.

Additive spline regression uses the existing base scikit-learn dependency:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_spline_model.py' -v
```

No GAM service, pyGAM package or native extension is added for this family.
The four spline candidates join `full` and `study` automatically after updating
the source. Existing resume rules require a new run for the changed code.

Occurrence/severity regression also uses only the base scikit-learn dependency:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_two_stage.py' -v
```

Its four settings join observed-level `full` and `study` searches automatically.
The saved point prediction is occurrence probability times positive mean;
component diagnostics are separate from primary MAE selection. Update the
editable checkout, reinstall it if needed, and start a new run under the existing
resume fingerprint rules. The ERA5 cache can still be reused. No training
service, probability-calibration package or new native extension is added.

## Public repository

Use a new dedicated code-only repository, or an existing repository only after
its full history is reviewed. A private operational checkout must never become
the public repository by changing its remote or visibility. Start from a clean
source export containing only `.publication-allowlist` entries.

Initialize the source checkout and install the local guards before the first
commit. The supplied archive excludes Git history, inputs and generated outputs.

```bash
git init -b main
python3 tools/install_hooks.py
python3 tools/check_public_tree.py --working-tree
git add .
python3 tools/check_public_tree.py
git diff --cached --stat
git diff --cached
git commit -m "Add generic pavement analytics tools"
```

Set `PAVEMENT_DENYLIST_FILE` privately on the authorized publishing workstation.
It must refer to a nonempty external UTF-8 file with one restricted name, alias,
identifier, source basename or endpoint per line. No comment syntax is used;
every nonempty line is a literal case-insensitive match. Do not include empty or
unnecessarily broad terms. The file must not be placed under this repository.

Set `PUBLIC_CODE_REMOTE` to the selected public code repository only. After
reviewing the source and all outbound commit metadata in the internal environment:

```bash
python3 tools/check_public_tree.py --history --require-denylist
git remote add origin "$PUBLIC_CODE_REMOTE"
git push -u origin main
```

The public check is an aid to controlled publication, not a substitute for review.
Do not push a suspected data-bearing commit merely to see whether CI catches it.
Never bypass the pre-push hook, and do not put the private denylist in GitHub.

In GitHub settings, configure default-branch protection/rulesets, required PRs,
required checks for each `public-ci` OS/Python matrix entry, and named owners
for security/workflow changes.
Enable available secret protection and block force-pushes. These settings have
to be applied to the selected repository; no administrative settings are assumed
to exist just because these files are present.

## Internal execution

Import an approved source revision to the internal workstation. Keep code,
private data and private results in distinct directories. Install dependencies
from an approved internal mirror or reviewed offline wheelhouse. The tested
requirements pin the direct scientific packages; for production, freeze and
approve the complete transitive environment in the private deployment system.

Provide SSH credentials through the usual OS/SSH agent and configure host keys
before fetching. `PAVEMENT_DATA_REMOTE` is a restricted SCP-style SSH remote
supplied at runtime. URLs with embedded passwords, HTTPS URLs, command helpers
and mutable branch/tag names are not supported by the adapter.

`sync --commit` requires a full 40-character lowercase commit SHA. It fetches
only that revision into `sources/<commit>` and refuses to overwrite an existing
checkout. Symlinks and Git submodules in the fetched tree are rejected. It does
not run scripts from that repository. If data use LFS, materialize the objects
through an approved internal process and then use the local input adapter;
automatic LFS materialization is not provided by this release.

Configure `input_root`, the exact file list and column mapping in the private
`run.json`. `PAVEMENT_PRIVATE_EXECUTION=1` must be set on that machine. This is an
intent flag, not firewall detection. Enforce the network/host boundary through
your infrastructure and account policy. Run the analysis only there.

Alternatively, `pavement analyze /path/to/private/data --open` snapshots the local
exports and generates the configuration automatically in a new external workspace.
This command supplies the explicit local-execution intent without requiring an
environment variable. It retains the GitHub Actions prohibition and all path
boundaries. See [family analysis](family-analysis.md) for assumptions and outputs.

Internal job schedulers may call the same CLI from their own private definitions.
The public repository has no private-workflow trigger, inbound webhook, internal
runner configuration, data remote secret or results uploader. Do not run private
jobs through this repository's GitHub Actions, even using a self-hosted runner.

## First private acceptance run

Confirm the authoritative group mapping, date format, pairing scope and interval
cap. Reconcile raw-row counts, rejection reasons, transitions and final pairs.
Review every feature disposition and material-specific association report.
Validate all curve profiles against a trusted implementation before using
recomputed PCI. Check a second run with `--resume` and verify that edited input
or output causes refusal. Keep the acceptance record in private storage.

The provided tests use generated records. They verify code behavior, not the
connectivity, policy, scientific validity or reference accuracy of a private
deployment. A real SSH fetch and real-data acceptance must occur internally.

## Installation and test troubleshooting

Hosted CI is disabled. Run the regression suite locally for the revision being
deployed and record the Python, dependency and operating-system versions. The
previous CI matrix covered Ubuntu with Python 3.11, 3.12 and 3.13 and macOS with
Python 3.12 and 3.13; those historical results do not validate later revisions
on every local hardware configuration.

If installation succeeds but `pavement` is not found, activate the same virtual
environment in the current terminal, then verify the module entry point:

```bash
source "$HOME/.venvs/pavement-analytics/bin/activate"
python -m pavement_analytics --help
```

If the module is also missing, run `python -m pip install --no-deps -e .` from
the source checkout in that environment after installing the numerical
dependencies. You can use `python -m pavement_analytics analyze ...` instead of
the `pavement` executable; both invoke the same CLI.

On macOS, system temporary directories may be reached through a symlink alias
such as `/var/folders`. A `SYMLINK_PATH_FORBIDDEN` failure in generated tests can
therefore occur even when installation succeeded. Update the checkout and rerun
the tests: the test helpers resolve temporary roots before exercising the path
boundary. Production inputs and output paths still reject symlinks.

When testing an older checkout, a temporary workaround is to select a canonical
scratch directory outside every Git checkout for that test command:

```bash
mkdir -p "$HOME/.cache/pavement-test-tmp"
PAVEMENT_TEST_TMPDIR="$(python -c 'from pathlib import Path; print((Path.home() / ".cache" / "pavement-test-tmp").resolve())')"
TMPDIR="$PAVEMENT_TEST_TMPDIR" python -m unittest discover -s tests -v
```

This changes the test process's temporary directory only. Do not change `HOME`
or disable the production symlink checks. For an operational input/output error,
use the actual canonical external path instead of a symlink alias.

## Troubleshooting without disclosure

`full` shows local input/mapping paths, progress and aggregate counts. Add
`--quiet` to keep terminal output to fixed codes; other commands already use
fixed codes. Consult the private `logs/progress.log` (full runs),
`logs/runtime.log` and `failed` directories locally. Typical codes include:

| Code | Action |
| --- | --- |
| `PRIVATE_EXECUTION_OPT_IN_REQUIRED` | Set the runtime flag only in the approved environment |
| `GITHUB_ACTIONS_DATA_EXECUTION_FORBIDDEN` | Move the job to the independent private execution system |
| `PATH_OUTSIDE_PRIVATE_WORKSPACE` | Correct the private configuration paths |
| `OUTPUT_MUST_NOT_BE_VERSIONED` | Put outputs outside all Git checkouts |
| `FULL_COMMIT_SHA_REQUIRED` | Supply the approved immutable data revision |
| `PRIVATE_SOURCE_FETCH_FAILED` | Check internal reachability, SSH permissions and host trust locally |
| `INPUT_NOT_MATERIALIZED` | Materialize the private LFS object internally |
| `RESUME_INPUT_OR_CODE_CHANGED` | Choose a new output version |
| `RESUME_OUTPUT_TAMPERED` | Investigate locally; create a new output version |
| `FEATURE_ASSOCIATION_REVIEW_REQUIRED` | Review private feature correlations and revise the allowlist |
| `AMBIGUOUS_COLUMNS_USE_PRIVATE_OPTIONS` | Review the private workspace's `column_mapping_report.json`; set exact candidate headers in a private `columns` override if canonical PAVER precedence cannot resolve the role |
| `FULL_SETUP_INPUT_PATHS_INVALID` | Correct every reported missing, wrong-type or unreadable auxiliary path; `full_setup_report.json` retains the `input_checks` audit |
| `INVALID_LOCATION_MAPPING` | Review entry reasons in the progress log and `full_setup_report.json` under `mapping_entries`; see the [mapping format and path boundaries](full-pipeline.md#required-private-inputs) |
| `CLIMATE_CACHE_MUST_NOT_BE_VERSIONED` | Select a cache outside every Git checkout with `--cache-dir`; see the [existing-cache copy example](full-pipeline.md#required-private-inputs) |
| `PUBLICATION_REJECTED` | Stop before push; inspect and repair the candidate locally |

Never paste private logs or records into a public issue. Create a generated
reproduction with equivalent behavior.

## Model comparison runtime

Version 0.5.7 enables family comparisons and the isolated model worker in
`full` and `study`. Install the updated base requirements; `threadpoolctl` is
now explicit (`>=3.2,<4`, admission version 3.6.0), alongside the existing
optional boosting extras. It limits native numerical threads in the spawned
worker. Linux admission and the deployment macOS run remain distinct checks.

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_model_comparison.py' -v
```

The worker uses Python's spawn context rather than inheriting native fitting
state through fork. Command-line entry points handle this automatically. A
Python script calling the isolated library API must put its launch code under
`if __name__ == "__main__":`. Keep `isolated: true` for hard fit/prediction
deadlines; in-process debug mode does not enforce them. Ctrl+C terminates the
active worker and preserves verified completed-target checkpoints. Do not run
two writers against the same workspace/cache.

For full-run cancellation, resume the same workspace using the existing
`full --resume` workflow. Code/config/dependency changes still require a new
full run. An unchanged external ERA5 cache can be reused. See
[model runtime and task checkpoints](full-pipeline.md#model-runtime-and-task-checkpoints)
for fit budgets, model options, private artifact definitions and cache integrity.
