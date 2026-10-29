import json
from temp_support import temporary_directory
from pathlib import Path
import unittest

import pandas as pd

from pavement_analytics.reporting import build_report
from pavement_analytics.security import SafeError


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = temporary_directory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = 'rate_example'
        self.task = self.root / 'materials/AC' / self.target
        self.task.mkdir(parents=True)
        pd.DataFrame({'material': ['AC'] * 4, self.target: [1., 2., None, 4.]}).to_csv(self.root / 'dataset.csv', index=False)
        pd.DataFrame({'pair_id': [1, 2], 'group_id': ['0001', '0002'], 'delta_days': [365, 730],
                      self.target: [1., 4.], 'prediction': [2., 2.],
                      'condition_prediction': [0., 0.]}).to_csv(self.task / 'holdout_predictions.csv', index=False)
        pd.DataFrame({'candidate': ['rf_2', 'condition'], 'wMAE': [.5, 1.], 'wRMSE': [.7, 1.3],
                      'wins_vs_condition': [3, 0]}).to_csv(self.task / 'leaderboard.csv', index=False)
        pd.DataFrame({'pair_id': [3], 'group_id': ['0003'], 'fold': [0], 'candidate': ['rf_2'],
                      'actual': [2.], 'prediction': [2.1], 'weight': [365]}).to_csv(self.task / 'oof_predictions.csv', index=False)
        self.slot = {'material': 'AC', 'target': self.target, 'status': 'champion', 'candidate': 'rf_2',
                     'development_rows': 1, 'holdout_rows': 2, 'excluded_missing_target': 99,
                     'holdout': {'wMAE': 999}}

    def test_consolidated_metrics_recomputed_and_partitions_separate(self):
        build_report(self.root, [self.slot], {'pair_rows': 4})
        summary = pd.read_csv(self.root / 'target_summary.csv').iloc[0]
        self.assertAlmostEqual(summary.holdout_wMAE, 5 / 3)
        self.assertAlmostEqual(summary.condition_wMAE, 3.)
        self.assertAlmostEqual(summary.improvement_percent, 100 * (3 - 5 / 3) / 3)
        self.assertEqual(summary.excluded_missing_target, 1)
        self.assertEqual(summary.valid_target_rows, 3)
        leader = pd.read_csv(self.root / 'model_leaderboard.csv')
        self.assertAlmostEqual(leader.iloc[0].cv_mean_wMAE, .5)
        self.assertEqual(leader.selected.tolist(), [True, False])
        predictions = pd.read_csv(self.root / 'predictions.csv', dtype={'group_id': str})
        self.assertEqual(len(predictions), 3)
        self.assertEqual(set(predictions.partition), {'holdout', 'development_oof'})
        self.assertEqual(predictions.iloc[0].group_id, '0001')
        html = (self.root / 'dashboard.html').read_text()
        self.assertIn('Independent holdout error', html)
        self.assertIn('Actual versus predicted', html)
        self.assertNotIn('https://', html)
        self.assertNotIn('src="http', html)

    def test_missing_targets_and_zero_baseline_still_reported(self):
        predictions = pd.read_csv(self.task / 'holdout_predictions.csv')
        predictions['prediction'] = predictions[self.target]
        predictions['condition_prediction'] = predictions[self.target]
        predictions.to_csv(self.task / 'holdout_predictions.csv', index=False)
        self.slot.update(status='no_model', candidate='condition')
        slots = [self.slot, {'material': 'PCC', 'target': 'rate_other', 'status': 'no_valid_target'},
                 {'material': 'APC', 'target': self.target, 'status': 'insufficient_groups'}]
        build_report(self.root, slots, {})
        summary = pd.read_csv(self.root / 'target_summary.csv')
        self.assertEqual(len(summary), 3)
        self.assertTrue(pd.isna(summary.iloc[0].improvement_percent))
        self.assertTrue(pd.isna(summary.iloc[1].holdout_wMAE))
        html = (self.root / 'dashboard.html').read_text()
        for status in ['no_model', 'no_valid_target', 'insufficient_groups']:
            self.assertIn(status, html)
        self.assertIn('condition baseline was retained', html)

    def test_escaping_and_path_validation(self):
        self.slot['candidate'] = '<img src=x onerror=alert(1)>'
        build_report(self.root, [self.slot], {'<script>': '</script><img>'})
        html = (self.root / 'dashboard.html').read_text()
        self.assertNotIn('<img', html)
        self.assertIn('&lt;img', html)
        for key, value in [('target', '../escape'), ('material', '<script>'), ('target', 'a/b')]:
            bad = {**self.slot, key: value}
            with self.assertRaisesRegex(SafeError, 'INVALID_REPORT_TASK'):
                build_report(self.root, [bad], {})
        with self.assertRaisesRegex(SafeError, 'DUPLICATE_REPORT_TASK'):
            build_report(self.root, [self.slot, self.slot], {})

    def test_empty_report_and_symlink_rejected(self):
        build_report(self.root, [], {})
        self.assertEqual(len(pd.read_csv(self.root / 'predictions.csv')), 0)
        html = (self.root / 'dashboard.html').read_text()
        self.assertIn('No target tasks were recorded', html)
        self.assertNotIn('Deduct target availability', html)
        self.assertNotIn('synthetic run', html)
        destination = self.root / 'target_summary.csv'
        destination.unlink()
        destination.symlink_to(self.root / 'dataset.csv')
        with self.assertRaisesRegex(SafeError, 'SYMLINK_PATH_FORBIDDEN'):
            build_report(self.root, [], {})

    def test_setup_features_grouping_and_source_inventory_are_visible_and_escaped(self):
        setup = {"features": ["PCI_start"], "sources": [{"original_path": "<img>.csv", "rows": 4,
                 "deduct_header_mapping": {"63l deduct": "63L deduct"}}],
                 "grouping": [{"input": "example.csv", "strategy": "section_identity",
                 "source_column": None, "limitation": "related sections may overlap"}]}
        (self.root / 'setup_report.json').write_text(json.dumps(setup))
        build_report(self.root, [self.slot], {})
        html = (self.root / 'dashboard.html').read_text()
        self.assertIn('starting PCI alone', html)
        self.assertIn('Section-level grouping', html)
        self.assertIn('63L deduct', html)
        self.assertIn('&lt;img&gt;.csv', html)
        self.assertNotIn('<img>', html)

    def test_zero_policy_and_endpoint_audit_counts_are_visible(self):
        audit = {"schema_version": 1, "blank_deducts": "zero", "pair_rows": 4,
                 "blank_deduct_cells": {"start": 12, "end": 10},
                 "zero_filled_deduct_cells": {"start": 12, "end": 10},
                 "invalid_deduct_pairs": {"start": 0, "end": 1},
                 "export_gap_pairs": {"start": 0, "end": 1, "either": 1},
                 "targets": [{"material": "AC", "target": self.target,
                              "valid_target_rows": 3, "excluded_missing_target": 1}]}
        (self.root / 'deduct_audit.json').write_text(json.dumps(audit))
        build_report(self.root, [self.slot], {})
        html = (self.root / 'dashboard.html').read_text()
        self.assertIn('Deduct target availability', html)
        self.assertIn('Blank deduct policy: zero', html)
        self.assertIn('Empty exported deduct cells were explicitly treated as zero', html)
        self.assertIn('<td>Empty deduct cells treated as zero</td><td>12</td><td>10</td>', html)
        self.assertIn('An inspection shared by consecutive pairs is counted again', html)
        self.assertIn('href="deduct_audit.json"', html)
        self.assertNotIn('All deduct-family tasks have zero valid target rows', html)
        self.assertIn('skipped before fitting', html)
        self.assertIn('condition baseline retained', html)

    def test_missing_policy_explains_all_skipped_deduct_targets_and_escapes_headers(self):
        audit = {"blank_deducts": "missing", "targets": [
            {"material": "AC", "target": "rate_first", "valid_target_rows": 0},
            {"material": "PCC", "target": "rate_second", "valid_target_rows": 0}],
            "ignored_ungraded_deduct_columns": ["<img src=x onerror=alert(1)> deduct"]}
        (self.root / 'deduct_audit.json').write_text(json.dumps(audit))
        slots = [{"material": target["material"], "target": target["target"], "status": "no_valid_target"}
                 for target in audit["targets"]]
        build_report(self.root, slots, {})
        html = (self.root / 'dashboard.html').read_text()
        self.assertIn('Blank deduct policy: missing', html)
        self.assertIn('kept as missing and can exclude deduct targets', html)
        self.assertIn('All deduct-family tasks have zero valid target rows', html)
        self.assertIn('Model fitting was skipped for these tasks', html)
        self.assertIn('&lt;img src=x onerror=alert(1)&gt; deduct', html)
        self.assertNotIn('<img', html)

    def test_empty_deduct_audit_does_not_claim_all_targets_were_skipped(self):
        (self.root / 'deduct_audit.json').write_text(json.dumps({"blank_deducts": "missing", "targets": []}))
        build_report(self.root, [], {})
        html = (self.root / 'dashboard.html').read_text()
        self.assertIn('Deduct target availability', html)
        self.assertNotIn('All deduct-family tasks have zero valid target rows', html)

    def test_ungraded_deduct_coverage_and_overlap_counts_are_visible(self):
        audit = {"blank_deducts": "zero", "ambiguous_deduct_pairs": {"start": 2, "end": 1},
                 "ungraded_deduct_columns": ["49 deduct", "<img> deduct"], "targets": []}
        (self.root / 'deduct_audit.json').write_text(json.dumps(audit))
        build_report(self.root, [], {})
        html = (self.root / 'dashboard.html').read_text()
        self.assertIn('Recognized deduct columns without an L/M/H severity: 49 deduct, &lt;img&gt; deduct', html)
        self.assertIn('<td>Pairs with overlapping deduct representations</td><td>2</td><td>1</td>', html)
        self.assertIn('Positive ungraded and severity values for the same code require reconciliation', html)
        self.assertNotIn('were excluded from family sums', html)
        self.assertNotIn('<img>', html)

    def test_chart_is_capped_but_predictions_export_is_complete(self):
        frame = pd.read_csv(self.task / 'holdout_predictions.csv')
        pd.concat([frame] * 400).to_csv(self.task / 'holdout_predictions.csv', index=False)
        build_report(self.root, [self.slot], {})
        html = (self.root / 'dashboard.html').read_text()
        self.assertEqual(html.count('<circle '), 600)
        self.assertIn('600 of 800 holdout points shown', html)
        exported = pd.read_csv(self.root / 'predictions.csv')
        self.assertEqual(len(exported.loc[exported.partition.eq('holdout')]), 800)


if __name__ == '__main__':
    unittest.main()
