# Contributing

Implement reusable PCI, deduct-value and PAVER tooling. Use generic terminology
and generate all test records inside temporary directories or memory. Do not add
operational datasets, report excerpts, screenshots, charts, addresses, source
names, environment files, fitted coefficients or cached outputs.

Run `python -m unittest discover -s tests -v` after a meaningful logic change.
Hosted GitHub CI is paused: the workflow has no automatic triggers and its job
is disabled. Run the regression suite and source publication checks locally
for each build, and include the results in its pull request.
Add a regression test for changes to curation, feature admission, splits,
calculation rules or data boundaries. Generate its inputs in code.

Add approved source files to `.publication-allowlist` explicitly. Stage only the
reviewed paths, run `python tools/check_public_tree.py`, and review the exact
staged diff. Configure the external denylist and install the supplied Git hooks
before the first push. Never disable the hooks to work around a failed scan.

Record changes to numerical behavior in release notes and increment the package
version. Run artifacts are immutable; create a new output version when input,
configuration, code or dependency versions change.

Pull requests and commit messages describe generic behavior and validation.
Keep private reproduction instructions and diagnostics in the internal tracker.
