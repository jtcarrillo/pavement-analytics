# Deployment

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
required `public-ci` checks, and named owners for security/workflow changes.
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

## Troubleshooting without disclosure

Console output contains fixed codes. Consult the private `logs/runtime.log` and
`failed` directories locally. Typical codes include:

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
| `PUBLICATION_REJECTED` | Stop before push; inspect and repair the candidate locally |

Never paste private logs or records into a public issue. Create a generated
reproduction with equivalent behavior.
