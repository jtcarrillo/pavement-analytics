"""Invent reproducible model-stage fixtures; no source data or weather service.

Generated tables deliberately live outside versioned source. Truth columns,
future outcomes and adversarial cases are kept apart from allowed predictors.
This module prepares and validates a benchmark, not new production estimators.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from .io import sha256, write_json
from .security import Boundary, SafeError, disjoint, private_umask, reject_links, source_root
from .study_modeling import make_partitions, search_levels


GENERATOR_VERSION = 1
DEFAULT_SEED = 20261031
PROFILES = {"smoke": (15, 3, 4, 3), "standard": (60, 9, 8, 5), "scale": (120, 12, 12, 5)}
MATERIALS = ("PCC", "APC", "AC_AAC_ACC")
CLIMATE_FEATURES = ["freeze_thaw_cycles", "wet_days", "thermal_aging_index", "temperature_range_c",
                    "soil_moisture_index", "cooling_excursion_c", "climate_proxy_index", "climate_noise"]
FEATURES = ["PCI_start", "pavement_age_start_years", "traffic_index", "structural_index",
            "subgrade_category", "branch_use", "nonclimate_noise", *CLIMATE_FEATURES]
TARGETS = ["level_linear", "level_smooth", "level_threshold", "level_interaction",
           "level_zero_inflated", "level_nonclimate", "level_no_signal",
           "level_constant_zero", "level_constant_positive"]
IDENTITY_COLUMNS = ["row_id", "group_id", "section_id", "material", "start_date", "end_date", "delta_days"]
CONTRACTS = {
    "level_linear": {"pattern": "Linear climate and traffic signal", "drivers": ["thermal_aging_index", "wet_days", "traffic_index"],
                     "models": ["ElasticNet", "ExtraTrees", "XGBoost", "LightGBM", "GAM"],
                     "mean_formula": "3 + 1.8*thermal_aging_index + 0.025*wet_days + 4*traffic_index + material_offset",
                     "expectation": "A signal model should beat the median baseline. No fixed winner is required."},
    "level_smooth": {"pattern": "Smooth additive nonlinear signal", "drivers": ["thermal_aging_index", "temperature_range_c", "wet_days"],
                     "models": ["GAM", "ExtraTrees", "XGBoost", "LightGBM"],
                     "mean_formula": "5 + 9*(1-cos(2*pi*a)) + 12*(t-0.5)^2 + 5*w^2; a=aging/12, t=(temperature_range_c-15)/40, w=wet_days/220",
                     "expectation": "Check recovery of additive shapes and performance against a linear model."},
    "level_threshold": {"pattern": "Thresholds and interactions", "drivers": ["freeze_thaw_cycles", "wet_days", "traffic_index", "soil_moisture_index", "structural_index"],
                        "models": ["ExtraTrees", "XGBoost", "LightGBM", "RandomForest"],
                        "mean_formula": "2 + 22*I(freeze>90 and wet>110) + 11*I(traffic>1.2) + 7*I(moisture>0.6 and structure<0.9)",
                        "expectation": "Tree models should recover thresholds. An additive-only GAM has a deliberate limitation."},
    "level_interaction": {"pattern": "Continuous product interactions", "drivers": ["freeze_thaw_cycles", "wet_days", "structural_index", "traffic_index"],
                          "models": ["ExtraTrees", "XGBoost", "LightGBM", "RandomForest"],
                          "mean_formula": "3 + 28*(freeze/180)*(wet/220) + 7*(1.7-structure)*traffic",
                          "expectation": "Test interaction learning without requiring a specific tree implementation to win."},
    "level_zero_inflated": {"pattern": "Zero atom plus positive uniform severity", "drivers": ["freeze_thaw_cycles", "wet_days", "thermal_aging_index", "traffic_index"],
                            "models": ["Two-stage", "XGBoost", "LightGBM", "ExtraTrees"],
                            "mean_formula": "p=sigmoid(-1.2+4*(freeze/180-0.5)+2*(wet/220-0.5)); s=8+18*(aging/12)+8*traffic; Y=0 with probability 1-p, else Uniform(0.6*s,1.4*s); E[Y|X]=p*s",
                            "median_formula": "0 if p<=0.5, otherwise s*(1.4-0.4/p)",
                            "expectation": "Check occurrence probability and positive severity separately. The conditional mean need not win MAE."},
    "level_nonclimate": {"pattern": "Pavement predictors only", "drivers": ["pavement_age_start_years", "traffic_index", "structural_index", "subgrade_category"],
                         "models": ["All"], "mean_formula": "3 + 0.8*age + 8*traffic + 3*structure + 2*subgrade_code",
                         "expectation": "Adding climate has no planted benefit; finite-sample changes can still occur."},
    "level_no_signal": {"pattern": "Independent random outcome", "drivers": [], "models": ["All"],
                        "mean_formula": "Uniform(5,35), independent of all predictors; mean=median=20",
                        "expectation": "Do not demand positive skill or meaningful feature importance."},
    "level_constant_zero": {"pattern": "All-zero outcome", "drivers": [], "models": ["All"], "mean_formula": "0",
                           "expectation": "Report constant_target and avoid meaningless R-squared or importance."},
    "level_constant_positive": {"pattern": "Constant positive outcome", "drivers": [], "models": ["All"], "mean_formula": "7",
                               "expectation": "Report constant_target; the constant prediction must be reproduced."},
}


def _root(path: Path, *, empty=False) -> Boundary:
    reject_links(path)
    public, resolved = source_root().resolve(), path.expanduser().resolve()
    if not disjoint(public, resolved):
        raise SafeError("BENCHMARK_MUST_BE_OUTSIDE_PUBLIC_TREE")
    if any((p / ".git").exists() for p in (resolved, *resolved.parents)):
        raise SafeError("BENCHMARK_MUST_NOT_BE_VERSIONED")
    if empty:
        if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
            raise SafeError("BENCHMARK_REQUIRES_EMPTY_DIRECTORY")
        return Boundary.create(public, resolved)
    if not resolved.is_dir():
        raise SafeError("BENCHMARK_DIRECTORY_NOT_FOUND")
    return Boundary(public, resolved)


def _write_table(boundary, name, frame):
    path = boundary.output(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def records(seed=DEFAULT_SEED, profile="standard"):
    """Return independent observation, oracle and control tables plus one plan."""
    if profile not in PROFILES or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SafeError("INVALID_BENCHMARK_OPTIONS")
    groups, sections, intervals, folds = PROFILES[profile]
    rng, labels, stress_rng = (np.random.default_rng(seed + offset) for offset in (0, 1103, 2207))
    rows, truths, traps = [], [], []
    for group in range(groups):
        group_id = f"SYN{group:03d}"
        group_climate = rng.uniform(0.15, 0.85, size=2)
        trap_value = labels.uniform(5, 35)
        for section in range(sections):
            material = MATERIALS[section % len(MATERIALS)]
            start = date(2010, 1, 1)
            initial_age = rng.uniform(1, 14)
            for visit in range(intervals):
                row_id = len(rows)
                delta = int(rng.integers(365, 731))
                end = start + timedelta(days=delta)
                f, w = 0.65*rng.random(2) + 0.35*group_climate
                a, t, m = rng.random(3)
                traffic, structure = rng.uniform(0.2, 2.0), rng.uniform(0.6, 1.6)
                subgrade = int(rng.integers(0, 3))
                age = initial_age + (start-date(2010, 1, 1)).days/365.25
                feature = {"PCI_start": rng.uniform(45, 98), "pavement_age_start_years": age,
                           "traffic_index": traffic, "structural_index": structure,
                           "subgrade_category": ["A", "B", "C"][subgrade],
                           "branch_use": ["runway", "taxiway", "apron"][int(rng.integers(0, 3))],
                           "nonclimate_noise": rng.normal(), "freeze_thaw_cycles": 180*f, "wet_days": 220*w,
                           "thermal_aging_index": 12*a, "temperature_range_c": 15+40*t,
                           "soil_moisture_index": 0.1+0.8*m, "cooling_excursion_c": rng.uniform(3, 35),
                           "climate_proxy_index": 180*f+rng.normal(0, 4), "climate_noise": rng.normal()}
                means = {
                    "level_linear": 3+1.8*feature["thermal_aging_index"]+0.025*feature["wet_days"]+4*traffic+(section%3),
                    "level_smooth": 5+9*(1-np.cos(2*np.pi*a))+12*(t-0.5)**2+5*w*w,
                    "level_threshold": 2+22*int(f>0.5 and w>0.5)+11*int(traffic>1.2)+7*int(feature["soil_moisture_index"]>0.6 and structure<0.9),
                    "level_interaction": 3+28*f*w+7*(1.7-structure)*traffic,
                    "level_nonclimate": 3+0.8*age+8*traffic+3*structure+2*subgrade,
                    "level_no_signal": 20., "level_constant_zero": 0., "level_constant_positive": 7.,
                }
                p = float(1/(1+np.exp(-(-1.2+4*(f-0.5)+2*(w-0.5)))))
                severity = 8+18*a+8*traffic
                targets = {name: float(mean+labels.uniform(-0.7, 0.7)) for name, mean in means.items()
                           if name not in {"level_no_signal", "level_constant_zero", "level_constant_positive"}}
                targets.update(level_zero_inflated=float(severity*labels.uniform(0.6, 1.4)) if labels.random()<p else 0.,
                               level_no_signal=float(labels.uniform(5, 35)), level_constant_zero=0., level_constant_positive=7.)
                row = {"row_id": row_id, "group_id": group_id, "section_id": f"{group_id}_S{section:02d}",
                       "material": material, "start_date": start.isoformat(), "end_date": end.isoformat(),
                       "delta_days": delta, **feature, **targets}
                rows.append(row)
                means["level_zero_inflated"] = p*severity
                truths.append({"row_id": row_id, **{name+"__conditional_mean": float(value) for name, value in means.items()},
                               "zero__p_positive": p, "zero__positive_mean": severity,
                               "zero__conditional_median": severity*(1.4-0.4/p) if p>0.5 else 0.})
                traps.append({"row_id": row_id, "group_id": group_id, "group_token_numeric": group,
                              "random_noise": float(labels.normal()), "level_group_memorization": trap_value+float(labels.uniform(-0.1, 0.1)),
                              "target_copy_end": targets["level_linear"], "PCI_end": targets["level_linear"]})
                start = end
    table = pd.DataFrame(rows)[[*IDENTITY_COLUMNS, *FEATURES, *TARGETS]]
    plan = make_partitions(table, folds=folds, seed=seed)
    hold = table.group_id.isin(plan.loc[plan.partition.eq("holdout"), "group_id"])
    stressed = table.copy()
    for name in FEATURES:
        if name not in {"subgrade_category", "branch_use"}:
            stressed.loc[stress_rng.random(len(table))<0.1, name] = np.nan
    stressed.loc[hold, "subgrade_category"] = "HOLDOUT_ONLY"
    stressed["all_missing_predictor"] = np.nan
    for name in TARGETS:
        stressed.loc[stress_rng.random(len(table))<0.02, name] = np.nan
    edges = table[[*IDENTITY_COLUMNS, *FEATURES]].copy()
    edges["level_all_missing"] = np.nan
    edges["level_constant_zero"] = 0.
    edges["level_constant_positive"] = 7.
    edges["level_development_constant"] = np.where(hold, table.level_linear, 0.)
    edges["level_positive_only_holdout"] = np.where(hold, table.level_linear, 0.)
    edges["level_no_holdout_labels"] = np.where(hold, np.nan, table.level_linear)
    positive_group = plan.loc[plan.partition.eq("development"), "group_id"].iloc[0]
    edges["level_one_positive_group"] = np.where(table.group_id.eq(positive_group), table.level_linear, 0.)
    return table, stressed, pd.DataFrame(truths), plan, pd.DataFrame(traps), edges


def create_bundle(path: Path, *, seed=DEFAULT_SEED, profile="standard") -> Path:
    table, stressed, oracle, plan, traps, edges = records(seed, profile)
    with private_umask():
        boundary = _root(Path(path), empty=True)
        tables = {"model_table.csv": table, "stress_model_table.csv": stressed, "partitions.csv": plan,
                  "truth/oracle.csv": oracle, "controls/leakage_challenge.csv": traps, "controls/edge_cases.csv": edges}
        invalid = table.iloc[:6].copy()
        invalid["delta_days"] = invalid.delta_days.astype(float)
        invalid.loc[invalid.index[:3], "delta_days"] = [0., -1., np.nan]
        tables["controls/invalid_weights.csv"] = invalid
        duplicate = table.iloc[:6].copy()
        duplicate.loc[duplicate.index[1], "row_id"] = duplicate.iloc[0].row_id
        tables["controls/duplicate_row_ids.csv"] = duplicate
        for name, frame in tables.items():
            _write_table(boundary, name, frame)
        contracts = {"synthetic": True, "not_real_distress_codes": True, "targets": CONTRACTS,
                     "correlated_feature_blocks": [["freeze_thaw_cycles", "climate_proxy_index"]],
                     "importance_rule": "Test planted driver blocks; correlated proxies can share importance. Numerical oracle functions are not causal evidence about real pavements."}
        boundary.output("truth").mkdir(exist_ok=True)
        write_json(boundary.output("truth/target_contracts.json"), contracts)
        columns = []
        for name in table:
            role = "target" if name in TARGETS else "climate_feature" if name in CLIMATE_FEATURES else "feature" if name in FEATURES else "weight" if name == "delta_days" else "identity_or_context"
            units = {"freeze_thaw_cycles": "synthetic cycles per inspection interval", "wet_days": "synthetic days per inspection interval",
                     "thermal_aging_index": "synthetic index", "temperature_range_c": "deg C", "soil_moisture_index": "unitless",
                     "cooling_excursion_c": "deg C", "climate_proxy_index": "synthetic cycle proxy", "PCI_start": "synthetic PCI points",
                     "pavement_age_start_years": "years", "delta_days": "days", "traffic_index": "unitless", "structural_index": "unitless"}.get(name, "synthetic deduct points" if name in TARGETS else "text" if table[name].dtype==object else "unitless")
            columns.append({"column": name, "role": role, "units": units, "dtype": str(table[name].dtype)})
        write_json(boundary.output("data_dictionary.json"), {"columns": columns, "stress_only_feature": "all_missing_predictor",
                                                            "source": "Invented using seeded NumPy draws; climate values are not ERA5 observations."})
        cases = [{"id": name, "table": "model_table.csv", "target": name, "expected": CONTRACTS[name]["expectation"]} for name in TARGETS]
        cases += [{"id": "stress_mixed_inputs", "table": "stress_model_table.csv", "target": "level_linear", "expected": "Finite nonnegative predictions with missing numeric inputs, an all-missing column and an unseen holdout category."},
                  {"id": "all_missing_target", "table": "controls/edge_cases.csv", "target": "level_all_missing", "expected_status": "no_valid_target"},
                  {"id": "training_constant", "table": "controls/edge_cases.csv", "target": "level_development_constant", "expected_status": "constant_target"},
                  {"id": "no_holdout_labels", "table": "controls/edge_cases.csv", "target": "level_no_holdout_labels", "expected_status": "insufficient_groups"},
                  {"id": "zero_positive_training", "table": "controls/edge_cases.csv", "target": "level_positive_only_holdout", "expected": "Two-stage path handles no positive training labels without using holdout labels."},
                  {"id": "positive_only_training", "table": "model_table.csv", "target": "level_linear", "expected": "Two-stage path uses a constant probability of one when every training label is positive."},
                  {"id": "one_positive_group", "table": "controls/edge_cases.csv", "target": "level_one_positive_group", "expected": "Only one development group has positive labels; at least one outer training fold has no positive severity examples."},
                  {"id": "invalid_weights", "table": "controls/invalid_weights.csv", "target": "level_linear", "expected": "Reject zero, negative and missing interval weights before fitting."},
                  {"id": "duplicate_ids", "table": "controls/duplicate_row_ids.csv", "target": "level_linear", "expected": "Reject duplicate row identifiers before fitting."},
                  {"id": "leakage", "table": "controls/leakage_challenge.csv", "target": "level_group_memorization", "expected": "Outcome copies are forbidden predictors. Random row splitting can memorize group tokens; grouped splitting tests unseen groups."},
                  {"id": "insufficient_fold_coverage", "table": "model_table.csv", "target": "level_linear", "filter": "keep only one development group plus one holdout group", "expected_status": "insufficient_groups"}]
        write_json(boundary.output("cases.json"), {"cases": cases})
        readme = f"""# Synthetic pavement model test data

All values are invented. This is a model-stage test fixture, not a raw PAVER
export or an ERA5/CDV integration test. It does not require CDS credentials.
The synthetic target labels are diagnostic scenarios, not actual distress codes.

Seed: {seed}. Profile: {profile}. Rows: {len(table):,}. Groups: {table.group_id.nunique()}.
There are three pavement surfaces, repeated sections, nine targets, mixed
numeric/categorical predictors and positive inspection-interval weights.

Use only the predictor names in bundle.json. Select one target and surface at a
time. Join partitions.csv by group_id; do not split individual rows at random.
Truth and controls are separate files and must never become automatic features.
stress_model_table.csv shares row identities and partitions with the clean table.
Its HOLDOUT_ONLY category never appears in development. Missing labels are
excluded per target, after applying the fixed partition plan.

Run from an installed pavement-analytics checkout:

    python -m pavement_analytics.synthetic_benchmark validate --workspace PATH
    python -m pavement_analytics.synthetic_benchmark smoke --workspace PATH --output NEW_EXTERNAL_RESULTS

The smoke command uses the currently installed model candidates. New models are
planned in docs/regression-model-roadmap.md; their presence in a target contract
is not a claim that they have been implemented or tested. The default smoke run
uses PCC and linear, smooth and zero-inflated targets. Use --all-targets and
--all-materials for the complete existing-candidate sweep.

The two-stage oracle records both conditional mean and conditional median.
Probability times positive mean predicts an expected value. It need not beat a
zero prediction under MAE when most outcomes are zero. Reproducibility is within
the recorded software environment; dependency upgrades can change fitted models.
"""
        boundary.output("README.md").write_text(readme, encoding="utf-8")
        files = sorted([*tables, "truth/target_contracts.json", "data_dictionary.json", "cases.json", "README.md"])
        manifest = {"schema_version": 1, "generator_version": GENERATOR_VERSION, "synthetic": True,
                    "seed": seed, "profile": profile, "rows": len(table), "groups": table.group_id.nunique(),
                    "sections": table.section_id.nunique(), "materials": list(MATERIALS),
                    "features": FEATURES, "climate_features": CLIMATE_FEATURES, "stress_features": [*FEATURES, "all_missing_predictor"],
                    "targets": TARGETS, "weight_column": "delta_days", "partition_file": "partitions.csv",
                    "selection_metric": "mean_interval_weighted_cv_MAE", "holdout_fraction": 0.2,
                    "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
                    "files": {name: sha256(boundary.path(name)) for name in files}}
        write_json(boundary.output("bundle.json"), manifest)
    validate_bundle(boundary.workspace)
    return boundary.workspace


def validate_bundle(path: Path) -> dict:
    """Validate contracts and hashes, including intentionally invalid fixtures."""
    boundary = _root(Path(path))
    manifest = json.loads(boundary.path("bundle.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("synthetic") is not True:
        raise SafeError("INVALID_BENCHMARK_MANIFEST")
    if manifest.get("features") != FEATURES or manifest.get("targets") != TARGETS:
        raise SafeError("BENCHMARK_FEATURE_TARGET_CONTRACT_CHANGED")
    for name, digest in manifest["files"].items():
        if sha256(boundary.path(name)) != digest:
            raise SafeError("BENCHMARK_ARTIFACT_CHANGED")
    table = pd.read_csv(boundary.path("model_table.csv"), dtype={"group_id": str, "section_id": str})
    stress = pd.read_csv(boundary.path("stress_model_table.csv"), dtype={"group_id": str})
    plan = pd.read_csv(boundary.path("partitions.csv"), dtype={"group_id": str})
    if len(table) != manifest["rows"] or table.row_id.duplicated().any() or table.group_id.nunique()!=manifest["groups"]:
        raise SafeError("BENCHMARK_ROW_IDENTITY_INVALID")
    if not np.isfinite(table.delta_days).all() or not table.delta_days.gt(0).all():
        raise SafeError("BENCHMARK_VALID_WEIGHTS_REQUIRED")
    days = (pd.to_datetime(table.end_date)-pd.to_datetime(table.start_date)).dt.days
    if not days.eq(table.delta_days).all():
        raise SafeError("BENCHMARK_INTERVAL_MISMATCH")
    if not np.isfinite(table[TARGETS].to_numpy()).all() or not table[TARGETS].ge(0).all().all():
        raise SafeError("BENCHMARK_TARGETS_INVALID")
    if set(FEATURES)&set(TARGETS) or set(FEATURES)&set(IDENTITY_COLUMNS):
        raise SafeError("BENCHMARK_PREDICTOR_LEAKAGE")
    if plan.group_id.duplicated().any() or set(plan.group_id)!=set(table.group_id):
        raise SafeError("BENCHMARK_PARTITION_GROUP_MISMATCH")
    expected = make_partitions(table, folds=PROFILES[manifest["profile"]][3], seed=manifest["seed"])
    if not plan.equals(expected):
        raise SafeError("BENCHMARK_SHARED_PLAN_CHANGED")
    if not stress.row_id.equals(table.row_id) or not stress.group_id.equals(table.group_id):
        raise SafeError("BENCHMARK_STRESS_IDENTITY_CHANGED")
    held = table.group_id.isin(plan.loc[plan.partition.eq("holdout"), "group_id"])
    if "HOLDOUT_ONLY" in set(stress.loc[~held, "subgrade_category"]) or not stress.loc[held, "subgrade_category"].eq("HOLDOUT_ONLY").all():
        raise SafeError("BENCHMARK_UNSEEN_CATEGORY_NOT_ISOLATED")
    if not stress.all_missing_predictor.isna().all():
        raise SafeError("BENCHMARK_MISSING_COLUMN_CHANGED")
    zero_fraction = float(table.level_zero_inflated.eq(0).mean())
    if not 0.4 < zero_fraction < 0.95:
        raise SafeError("BENCHMARK_ZERO_INFLATION_OUT_OF_RANGE")
    oracle = pd.read_csv(boundary.path("truth/oracle.csv"))
    if not oracle.row_id.equals(table.row_id):
        raise SafeError("BENCHMARK_ORACLE_IDENTITY_CHANGED")
    p, severity = oracle.zero__p_positive, oracle.zero__positive_mean
    if not np.allclose(oracle.level_zero_inflated__conditional_mean, p*severity):
        raise SafeError("BENCHMARK_HURDLE_MEAN_INVALID")
    median = np.where(p>0.5, severity*(1.4-0.4/p), 0)
    if not np.allclose(oracle.zero__conditional_median, median):
        raise SafeError("BENCHMARK_HURDLE_MEDIAN_INVALID")
    return {"status": "validated", "rows": len(table), "groups": len(plan), "targets": len(TARGETS),
            "development_groups": int(plan.partition.eq("development").sum()),
            "holdout_groups": int(plan.partition.eq("holdout").sum()), "zero_fraction": zero_fraction,
            "materials": table.material.value_counts().sort_index().to_dict()}


def run_smoke(path: Path, output: Path, *, all_targets=False, all_materials=False, candidate_ids=None, model_options=None) -> list[dict]:
    """Exercise the installed candidate registry with fixed grouped partitions."""
    validate_bundle(path)
    source = _root(Path(path))
    if not disjoint(source.workspace, Path(output).expanduser().resolve()):
        raise SafeError("BENCHMARK_RESULTS_MUST_BE_SEPARATE")
    manifest = json.loads(source.path("bundle.json").read_text())
    if candidate_ids is not None:
        from .model_registry import build_candidates
        build_candidates(manifest["seed"], candidate_ids)
    table = pd.read_csv(source.path("model_table.csv"), dtype={"group_id": str})
    partitions = pd.read_csv(source.path("partitions.csv"), dtype={"group_id": str})
    targets = TARGETS if all_targets else ["level_linear", "level_smooth", "level_zero_inflated"]
    materials = MATERIALS if all_materials else ["PCC"]
    with private_umask():
        result = _root(Path(output), empty=True)
        summaries = []
        for material in materials:
            material_path = result.output(material)
            material_path.mkdir()
            data = table.loc[table.material.eq(material)].copy()
            for target in targets:
                summary = search_levels(data, FEATURES, target, material_path/target, partitions, CLIMATE_FEATURES,
                                        seed=manifest["seed"], weight_column="delta_days", candidate_ids=candidate_ids,
                                        model_options=model_options)
                summaries.append({"material": material, **summary})
        _write_table(result, "target_summary.csv", pd.DataFrame(summaries))
        import sklearn
        write_json(result.output("benchmark_result.json"), {"synthetic": True, "bundle_sha256": sha256(source.path("bundle.json")),
                   "candidate_source": "pavement_analytics.study_modeling.candidates", "targets": targets,
                   "materials": list(materials), "requested_candidates": candidate_ids, "sklearn_version": sklearn.__version__,
                   "model_options": model_options,
                   "note": "Results apply only to generated model-stage fixtures. They do not measure real pavement accuracy.",
                   "summaries": summaries})
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--profile", choices=PROFILES, default="standard")
    create.add_argument("--seed", type=int, default=DEFAULT_SEED)
    validate = sub.add_parser("validate")
    validate.add_argument("--workspace", type=Path, required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--workspace", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--all-targets", action="store_true")
    smoke.add_argument("--all-materials", action="store_true")
    smoke.add_argument("--compare", action="store_true", help="Enable R7 model-family comparisons and the bounded worker")
    smoke.add_argument("--candidate", action="append", dest="candidate_ids", help="Require an implemented candidate ID; repeat to compare a subset plus the median baseline")
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            create_bundle(args.output, seed=args.seed, profile=args.profile)
            print("SYNTHETIC_BENCHMARK_CREATED")
        elif args.command == "validate":
            print(json.dumps(validate_bundle(args.workspace), sort_keys=True))
        else:
            summaries = run_smoke(args.workspace, args.output, all_targets=args.all_targets, all_materials=args.all_materials,
                                  candidate_ids=args.candidate_ids, model_options={} if args.compare else None)
            print(json.dumps({"status": "completed", "tasks": len(summaries)}))
    except (SafeError, ValueError, OSError, KeyError) as error:
        print(str(error) if isinstance(error, SafeError) else "SYNTHETIC_BENCHMARK_FAILED")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
