"""Generated mathematical checks for the preserved distress and PCI stages."""
import contextlib
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pavement_analytics.original import distress, fit_curves, pci
from pavement_analytics.security import SafeError
from temp_support import temporary_directory


def toy_cdv_profile():
    """Invent linear curves at runtime; these are not pavement reference tables."""
    curves = {str(q): [[0, 1000, 0, max(.3, 1.1 - .15 * q), 0, 0]] for q in range(2, 11)}
    return {"source": "Generated mathematical test fixture; not a reference standard",
            "rules": {"D5340": {"q_threshold": 5, "m_divisor": 95},
                      "D6433": {"q_threshold": 2, "m_divisor": 98}},
            "q1_scale_error_s": {"D5340": {}, "D6433": {}},
            "curves": {standard: {surface: curves for surface in ("asphalt", "concrete")}
                       for standard in ("D5340", "D6433")}}


class OriginalDistressTests(unittest.TestCase):
    def table(self):
        return pd.DataFrame({
            "group": ["g1", "g2", "g2"],
            "64L deduct_start": [1., 0., 4.], "64L deduct_end": [2., 0., 5.],
            "64M deduct_start": [2., 0., 2.], "64M deduct_end": [3., 0., 0.],
            "74 deduct_start": [0., 0., 0.], "74 deduct_end": [1., 0., 2.],
            "64L (Slabs)_start": [1., 0., 4.], "64L (Slabs)_end": [2., 0., 5.],
            "64M (Slabs)_start": [1., 0., 1.], "64M (Slabs)_end": [2., 0., 0.],
            "64L (Count)_end": [9., 0., 8.],
            "InspectedSlabs_start": [10., 0., 10.], "InspectedSlabs_end": [10., 0., 10.],
        })

    def test_family_targets_preserve_original_end_level_names_and_values(self):
        frame = self.table()
        meta = distress.parse_columns(frame.columns)
        with contextlib.redirect_stdout(io.StringIO()):
            result, _ = distress.derive(frame, meta, distress.FAMILIES, "group")
        np.testing.assert_allclose(result["fam_pcc_freeze_thaw deduct_end"], [6, 0, 7])
        np.testing.assert_allclose(result["64 deduct_total_end"], [5, 0, 5])
        np.testing.assert_allclose(result["fam_pcc_freeze_thaw deduct_delta"], [3, 0, 1])
        self.assertNotIn("fam_ac_load deduct_end", result)
        self.assertEqual(result["74 emerged"].tolist(), [1, 0, 1])

    def test_densities_and_severity_index_ignore_count_channels(self):
        frame = self.table()
        with contextlib.redirect_stdout(io.StringIO()):
            result, _ = distress.derive(frame, distress.parse_columns(frame), distress.FAMILIES)
        np.testing.assert_allclose(result["64 density_total_end"], [.4, np.nan, .5], equal_nan=True)
        np.testing.assert_allclose(result["64 sevindex_end"], [.6, np.nan, .5], equal_nan=True)

    def test_prevalence_preserves_group_support_and_emergence(self):
        frame = self.table()
        report = distress.prevalence(frame, distress.parse_columns(frame), "group")
        row = report.loc[(report.code == "74") & (report.measure == "deduct")].iloc[0]
        self.assertEqual(row.n_nonzero_end, 2)
        self.assertEqual(row.n_groups_nonzero_end, 2)
        self.assertAlmostEqual(row.pct_emerged, 66.67)

    def test_min_group_filter_and_all_zero_dropping_match_legacy_contract(self):
        frame = self.table()
        with contextlib.redirect_stdout(io.StringIO()):
            result, dropped = distress.derive(frame, distress.parse_columns(frame), distress.FAMILIES,
                                               group_col="group", min_groups=3)
        self.assertGreater(dropped, 0)
        self.assertEqual(result.columns.tolist(), frame.columns.tolist())


class OriginalPCITests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.profile = self.root / "toy-curves.json"
        self.profile.write_text(json.dumps(toy_cdv_profile()))
        self.curves = pci.CDVCurves(self.profile, rules="D6433")
        environment = patch.dict(os.environ, {"PAVEMENT_PRIVATE_EXECUTION": "1", "GITHUB_ACTIONS": "false"})
        environment.start()
        self.addCleanup(environment.stop)

    def test_maximum_cdv_uses_iterative_reduction_and_threshold(self):
        result = pci.max_cdv([20., 10.], "concrete", self.curves)
        self.assertAlmostEqual(result[0], 24.)
        self.assertAlmostEqual(pci.pci_from([20., 10.], "concrete", self.curves), 76.)
        self.assertEqual(pci.pci_from([], "concrete", self.curves), 100.)
        self.assertEqual(pci.pci_from([10., 2., 1.], "concrete", self.curves), 87.)

    def test_family_decomposition_is_additive_and_symmetric(self):
        total, parts = pci.decompose({"a": [20.], "b": [10.]}, "concrete", self.curves)
        self.assertAlmostEqual(total, 76.)
        self.assertAlmostEqual(parts["a"][2], 17.)
        self.assertAlmostEqual(parts["b"][2], 7.)
        rng = np.random.default_rng(931)
        for _ in range(12):
            families = {f"family_{i}": rng.uniform(0, 25, 2).tolist() for i in range(5)}
            total, parts = pci.decompose(families, "asphalt", self.curves)
            self.assertAlmostEqual(sum(values[2] for values in parts.values()), 100 - total)
        _, equal = pci.decompose({"a": [20.], "b": [20.]}, "concrete", self.curves)
        self.assertAlmostEqual(equal["a"][2], equal["b"][2])

    def test_private_curve_profile_and_standard_are_required_explicitly(self):
        with patch("sys.argv", ["pci", "input.csv", "--check-only"]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                pci.main()
        self.assertEqual(error.exception.code, 2)

    def test_cli_produces_pci_export_gap_and_shapley_outputs(self):
        frame = pd.DataFrame({"SectionUniqueID": ["001", "002"],
                              "PCI_start": [80, 90], "PCI_end": [70, 85],
                              "64L deduct_start": [10., 0.], "64L deduct_end": [20., 0.],
                              "62L deduct_start": [5., 0.], "62L deduct_end": [10., 0.]})
        source = self.root / "paired.csv"
        destination = self.root / "curated"
        frame.to_csv(source, index=False)
        with patch("sys.argv", ["pci", str(source), "--curves", str(self.profile), "--out", str(destination)]), contextlib.redirect_stdout(io.StringIO()):
            pci.main()
        result = pd.read_csv(destination.with_suffix(".csv"), dtype={"SectionUniqueID": str})
        self.assertEqual(result.SectionUniqueID.tolist(), ["001", "002"])
        self.assertEqual(result.export_gap_end.tolist(), [0, 1])
        for when in ("start", "end"):
            contributions = result.filter(regex=f"^loss_shap_.*_{when}$").sum(axis=1)
            np.testing.assert_allclose(contributions, 100 - result[f"pci_calc_{when}"])
        self.assertIn("pci_calc_delta", result)

    def test_no_shapley_flag_does_not_emit_fabricated_family_pci(self):
        source, destination = self.root / "paired.csv", self.root / "curated"
        pd.DataFrame({"PCI_end": [80], "64L deduct_end": [20]}).to_csv(source, index=False)
        with patch("sys.argv", ["pci", str(source), "--curves", str(self.profile), "--out", str(destination), "--no-shapley"]), contextlib.redirect_stdout(io.StringIO()):
            pci.main()
        result = pd.read_csv(destination.with_suffix(".csv"))
        self.assertFalse(any(col.startswith(("pci_only_", "loss_shap_", "loss_loo_")) for col in result))
        self.assertEqual(result.pci_calc_end.iloc[0], 80.)

    def test_joint_and_sequential_severity_corrections_never_lower_pci(self):
        deduct_path = self.root / "fitted.json"
        deduct_path.write_text(json.dumps({"curves": {"concrete": {
            name: {"knots": [[1, 1], [10, value]], "density_range": [1, 10]}
            for name, value in [("64M", 8), ("64H", 12), ("62H", 10)]}}}))
        fitted = pci.DeductCurves(deduct_path)
        present = [("64", "L", "thermal", 12., 2.), ("64", "M", "thermal", 14., 3.),
                   ("62", "L", "load", 10., 1.), ("62", "H", "load", 15., 4.)]
        baseline = pci.pci_from([entry[3] for entry in present], "concrete", self.curves)
        for mode in ("sequential", "joint"):
            result, gain, count = pci.apply_severity_correction(present, "concrete", self.curves, fitted, mode)
            corrected = pci.pci_from([entry[3] for entry in result], "concrete", self.curves)
            self.assertGreaterEqual(corrected, baseline)
            self.assertAlmostEqual(corrected - baseline, gain)
            self.assertGreater(count, 0)

    def test_standalone_cli_enforces_private_execution_opt_in(self):
        with patch.dict(os.environ, {"PAVEMENT_PRIVATE_EXECUTION": "0"}), patch(
                "sys.argv", ["pci", str(self.root / "input.csv"), "--curves", str(self.profile), "--check-only"]):
            with self.assertRaisesRegex(SafeError, "OPT_IN"):
                pci.main()


class OriginalCurveFitTests(unittest.TestCase):
    def test_weighted_pava_pools_decreasing_neighbors(self):
        np.testing.assert_allclose(fit_curves.pava([1, 3, 2, 5], [1, 1, 3, 1]), [1, 2.25, 2.25, 5])

    def test_fitted_density_curve_is_monotone_and_reports_sample_support(self):
        density = np.geomspace(.01, 100, 400)
        deduct = np.clip(15 * np.log10(1 + density) + np.sin(np.arange(400)) * 2, 0, 100)
        result = fit_curves.fit_curve(density, deduct)
        self.assertEqual(result["n"], len(density))
        self.assertTrue(np.all(np.diff([point[1] for point in result["knots"]]) >= 0))
        self.assertGreater(result["r2"], .95)

    def test_collect_samples_keeps_original_density_units_and_inconsistency_counts(self):
        frame = pd.DataFrame({"64L (Slabs)_start": [2., 0., 5.], "64L deduct_start": [10., 4., 0.],
                              "InspectedSlabs_start": [10., 10., 10.],
                              "43M (SqFt)_end": [5., 7., 0.], "43M deduct_end": [3., 5., 0.],
                              "InspectedArea (SqFt)_end": [100., 100., 100.]})
        samples, bad = fit_curves.collect_samples(frame)
        self.assertEqual(bad, {"qty_no_deduct": 1, "deduct_no_qty": 1})
        np.testing.assert_allclose(samples[("concrete", "64", "L")][0], [20.])
        np.testing.assert_allclose(samples[("asphalt", "43", "M")][0], [5., 7.])


if __name__ == "__main__":
    unittest.main()
