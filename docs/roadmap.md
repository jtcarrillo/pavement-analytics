# Scope and extension plan

Version 0.1.0 is a new generic implementation, not a verified migration of a
previous private pipeline. Its tests demonstrate behavior on generated inputs;
they do not claim agreement with prior project results.

| Area | Included now | Next acceptance work |
| --- | --- | --- |
| Source store | Pinned SSH fetch, explicit local file list, input checksums, guarded cell corrections | Private accession manifest, append-only history, exclusions, duplicate reconciliation and LFS adapter |
| Shared core | Table I/O, weighted errors, grouped folds, deduct rates, profile-driven maximum CDV and exact Shapley API | Validated reference profiles, multi-severity correction, curve fitting and weighted rank metrics |
| Curation | Input audits, canonical surfaces, conservative pairing, interval filters | Compatibility adapters and privately benchmarked reconciliation |
| Climate | Existing numeric start-state source features may be explicitly admitted | Private location adapter, cached climate acquisition, interval feature contract and forecast/as-of timing tests |
| Features | Explicit numeric admission, identity/outcome rejection, per-material association reports | Categorical inputs, rule-file versioning, family-aware association decisions and controlled variants |
| Ranking | Condition bins, interval weighting, shrinkage integral and rank-swing stability | Private reference comparisons, climate attachments and section-count variant |
| Models | Two baselines, small ElasticNet/RandomForest grid, grouped CV, reserved group holdout, saved model | Optuna, boosted trees, nested grouped stacking, target-specific losses, SHAP, permutation/PDP diagnostics |
| Registry | Immutable per-run task summaries and models | Append-only cross-run promotion registry, trusted loaders and group-bootstrap ensembles |
| Sensitivity/UQ | No specialized backend yet | Dakota driver/decks, analytic toy-problem checks, uncertainty and calibration adapters |
| Workflows | Local command and verified whole-run resume | Stage-level caching and internal scheduler profiles with private manifests |
| Reporting | Private audit/result tables and machine-readable run report | Generic local figure/table builders and independently reproducible report assembly |

## Gates

1. **Private deployment acceptance.** Confirm connectivity and access, agree on
   source identity and pairing rules, reconcile row accounting, validate curves,
   and demonstrate repeatability on approved private inputs.
2. **Feature/model extension.** Add the reviewed feature rules and model families;
   preserve group isolation, fit transforms only on training folds, and keep the
   final assessment separate from model selection.
3. **Scale and registry.** Add append-only promotion records, group-bootstrap
   ensembles and internal scheduling without adding a public-to-private trigger.
4. **Sensitivity and reporting.** Validate specialized methods on known toy
   functions, record their assumptions, then generate private report artifacts
   from immutable runs.

Climate measured over the outcome interval is retrospective explanatory data,
not automatically available at prediction time. Any such extension must label
the evaluation task and avoid claiming a prospective forecast. Sobol indices
also require explicit input-distribution and dependence assumptions; summing
individual indices is not generally a valid grouped sensitivity result.

All gates are generic software and scientific validation tasks. Operational
decisions, benchmark numbers, inventories, source identities and deployment
details remain in private planning records.
