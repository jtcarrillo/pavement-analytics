# Data boundary

The public repository is source-only. Operational records, locations, network
identities, source filenames, input hashes, private endpoint addresses, credentials,
reports, figures, fitted models and execution logs stay in the private environment.
Replacing a name with an identifier does not make its measurements public data.
The source handbook and project-specific planning documents do not belong here.

## Controls included

1. Runtime configuration and the data checkout live outside the public tree.
   Outputs may not be inside any Git checkout. Resolved paths and symlink
   components are checked before file operations.
2. Every operational CLI command requires an explicit private-execution
   environment flag and refuses GitHub Actions, including self-hosted Actions.
   The flag expresses operator intent; it does not detect a firewall or establish
   authorization. Restrict execution with network and OS controls as well.
3. Private Git fetching requires a full commit SHA, runtime SSH remote, existing
   authentication and strict host verification. Fetch logs are captured, hooks
   and submodule recursion are disabled, and linked tree entries are rejected.
4. The default-deny `.gitignore` is supplemented by an exact publication file
   allowlist. The scanner reads the Git index, so `git add -f` does not bypass it.
5. The pre-push hook checks reachable history, commit metadata and references,
   including files deleted in later commits. It requires a private denylist on
   the publishing workstation; it never prints matches or private paths.
6. Public CI uses hosted, disposable runners for generated tests only. It has no
   private credentials, endpoints, data checkout, artifact upload or runtime cache.
   Workflow changes and publication policy changes require maintainer review.
7. Generated files are written with a restrictive process umask. On POSIX the
   workspace is set to owner-only permissions. Windows requires an equivalent
   private directory ACL managed by the operator.

## Limits and operating rules

- Local hooks can be bypassed or removed. A public GitHub workflow executes
  after upload and cannot prevent the original disclosure. The authoritative
  export gate must run on the controlled internal publishing workstation before
  any GitHub write. Only reviewed source is transferred to a clean public clone.
- No text scanner can recognize every private name or information embedded in
  arbitrary code, commit messages or documentation. Maintain the external denylist
  and conduct human review. The denylist itself is private input and must remain
  outside this repository, CI, pull requests and issues.
- The source scanner rejects common formats and markers. It is not a classifier
  or a claim of accreditation. OS permissions, restricted egress, code review,
  source provenance and access controls supply the actual security boundary.
- Do not use Git submodules, DVC pointers or LFS pointers in this public tree to
  refer to private sources. Private versions and hashes belong in private manifests.
- Never submit real records, screenshots, paths or error logs in GitHub issues,
  discussions, pull requests, releases or Actions logs. Reproduce problems with
  records generated in memory.
- Load serialized model files only from trusted, verified private runs. Joblib
  model loading is executable deserialization, not a safe exchange format.
- Approve imported code revisions inside the private environment before running
  them on private data. Public changes must not auto-trigger private execution.
- If restricted content has already been pushed, stop publication and follow the
  organization's response process. Removing it from the current tree does not
  erase old commits, forks, cached views, logs or downloaded copies.

## Maintainer setup

Use a dedicated repository. Require pull requests and successful `public-ci`
checks on the default branch; restrict direct pushes and review changes to
`.github`, `.githooks`, `tools/check_public_tree.py` and `.publication-allowlist`.
Configure the named reviewers in repository settings. These server-side settings
are administrative setup steps; this source package does not enable them.

Keep private runner registration and private secrets out of this public project.
The internal scheduler pulls an approved code version; public CI never reaches
into the internal data repository.
