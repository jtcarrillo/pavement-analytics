# Scope and extension plan

Version 0.4.0 adds the original preparation pipeline and a local deduct-family
model sweep through `pavement full`. Original numerical helpers have generated
input parity checks against the recovered implementation. Automated tests do
not establish agreement on operational inputs or live ERA5 access. The earlier
`run`, `analyze` and `study` commands retain their separate definitions.

| Area | Included now | Next acceptance work |
| --- | --- | --- |
| Source store | Pinned SSH fetch, explicit local file list, input checksums, guarded cell corrections | Private accession manifest, append-only history, exclusions, duplicate reconciliation and LFS adapter |
| Shared core | Table I/O, weighted errors, grouped folds, deduct rates, private-profile CDV, original severity correction, optional deduct-curve fitting and exact Shapley | Private profile validation and operational numerical reconciliation |
| Curation | Input audits, header/date normalization, original surface/pairing stages, location matching, original distress derivation and quarantine | Privately benchmarked reconciliation and additional export adapters |
| Climate | Original ERA5-Land worker with its 37 output fields, PG/cooling/rainflow/aging calculations and cache; separate 25-feature pre-inspection study mode | Live CDS acceptance on the deployment workstation and operational coverage review |
| Features | Original column audit plus identity/outcome guards, numeric/categorical training-fold transforms, fitted-curve leakage exclusion | Rule-file versioning, family-aware association decisions and controlled variants |
| Ranking | Condition bins, interval weighting, shrinkage integral and rank-swing stability | Private reference comparisons, climate attachments and section-count variant |
| Models | Rate and observed-level tasks, baseline/ElasticNet/RandomForest grids, shared study splits, reserved holdout, saved models, climate ablation and permutation importance | Optuna, boosted trees, nested grouped stacking, target-specific losses, SHAP/PDP diagnostics and group-bootstrap uncertainty |
| Registry | Immutable per-run task summaries and models | Append-only cross-run promotion registry, trusted loaders and group-bootstrap ensembles |
| Sensitivity/UQ | No specialized backend yet | Dakota driver/decks, analytic toy-problem checks, uncertainty and calibration adapters |
| Workflows | Full original preparation with stage resume, local model sweep and consolidated report; separate rate/study commands | Study fitting resume and internal scheduler profiles with private manifests |
| Reporting | Offline dashboards, baseline-relative target rankings, predictor plots, climate ablation, consolidated CSVs and private stage audits | Cross-run comparison and additional uncertainty displays |

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

The full pipeline replaces the retired DataRobot backend with local baseline,
ElasticNet and random-forest searches. It needs no training service or remote
model credentials. See [the full pipeline guide](full-pipeline.md) for its
required private inputs, original stage definitions and validation outputs.

Climate measured over the outcome interval is retrospective explanatory data,
not automatically available at prediction time. The full report labels its
original mixed climate windows accordingly. Sobol indices
also require explicit input-distribution and dependence assumptions; summing
individual indices is not generally a valid grouped sensitivity result.

All gates are generic software and scientific validation tasks. Operational
decisions, benchmark numbers, inventories, source identities and deployment
details remain in private planning records.
