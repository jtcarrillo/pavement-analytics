# Scope and extension plan

Version 0.3.0 is a generic implementation, not a verified numerical migration of a
previous private pipeline. Its tests demonstrate behavior on generated inputs;
they do not claim agreement with prior project results.

| Area | Included now | Next acceptance work |
| --- | --- | --- |
| Source store | Pinned SSH fetch, explicit local file list, input checksums, guarded cell corrections | Private accession manifest, append-only history, exclusions, duplicate reconciliation and LFS adapter |
| Shared core | Table I/O, weighted errors, grouped folds, deduct rates, profile-driven maximum CDV and exact Shapley API | Validated reference profiles, multi-severity correction, curve fitting and weighted rank metrics |
| Curation | Input audits, canonical surfaces, conservative pairing, interval filters | Compatibility adapters and privately benchmarked reconciliation |
| Climate | Private location adapter, cached ERA5-Land hourly time series, 25 bounded pre-inspection exposures and timing/coverage tests | Live CDS acceptance on the deployment workstation, additional variables, lifetime-exposure and pavement-temperature contracts |
| Features | Explicit admission, reviewed inspection metadata, numeric/categorical training-fold transforms, identity/outcome rejection | Rule-file versioning, family-aware association decisions and controlled variants |
| Ranking | Condition bins, interval weighting, shrinkage integral and rank-swing stability | Private reference comparisons, climate attachments and section-count variant |
| Models | Rate and observed-level tasks, baseline/ElasticNet/RandomForest grids, shared study splits, reserved holdout, saved models, climate ablation and permutation importance | Optuna, boosted trees, nested grouped stacking, target-specific losses, SHAP/PDP diagnostics and group-bootstrap uncertainty |
| Registry | Immutable per-run task summaries and models | Append-only cross-run promotion registry, trusted loaders and group-bootstrap ensembles |
| Sensitivity/UQ | No specialized backend yet | Dakota driver/decks, analytic toy-problem checks, uncertainty and calibration adapters |
| Workflows | Local rate command with verified whole-run resume; end-to-end study command with climate cache | Study fitting resume, stage-level caching and internal scheduler profiles with private manifests |
| Reporting | Offline rate/study dashboards, target ranking, predictor plots, consolidated CSVs and private audit reports | Cross-run comparison and additional uncertainty displays |

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
