from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from temp_support import temporary_directory
from pavement_analytics.security import SafeError
from pavement_analytics.study_reporting import build_study_report


class StudyReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = temporary_directory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.target = 'family_example'
        self.task = self.out / 'materials/AC' / self.target
        self.task.mkdir(parents=True)
        self.predictions = pd.DataFrame({
            'row_id': [1, 2, 3], 'group_id': ['private_group_1', 'private_group_2', 'private_group_2'],
            'actual': [0., 6., 12.], 'prediction': [0., 4., 10.], 'baseline_prediction': [2., 2., 2.],
            'nonclimate_prediction': [2., 4., 4.], 'weight': [1., .5, .5]})
        self.predictions.to_csv(self.task / 'holdout_predictions.csv', index=False)
        pd.DataFrame({
            'feature': ['climate_temp', 'age', 'all_climate'],
            'kind': ['feature', 'feature', 'climate_block'], 'is_climate': [True, False, True],
            'importance_mean': [2., .5, 3.], 'importance_std': [.2, .1, .3],
            'repeats': [5, 5, 5], 'reliable': [True, True, True]
        }).to_csv(self.task / 'importance.csv', index=False)
        pd.DataFrame({'candidate': ['rf_2', 'median'], 'feature_set': ['all', 'all'],
                      'cv_mae': [1., 4.], 'cv_mae_std': [.1, .2], 'cv_rmse': [2., 5.],
                      'cv_wins_vs_baseline': [5, 0]}).to_csv(self.task / 'leaderboard.csv', index=False)
        self.summary = {'material': 'AC', 'target': self.target, 'status': 'modeled', 'candidate': 'rf_2',
                        'holdout_mae': 99., 'baseline_mae': 100., 'mae_skill': .01,
                        'valid_target_rows': 20, 'valid_target_groups': 7, 'nonzero_fraction': .8,
                        'importance_reliable': True}

    def test_recomputes_metrics_and_consolidates_without_exposing_groups_in_html(self):
        build_study_report(self.out, [self.summary], {})
        summary = pd.read_csv(self.out / 'target_summary.csv').iloc[0]
        self.assertAlmostEqual(summary.holdout_mae, 1.)
        self.assertAlmostEqual(summary.baseline_mae, 4.5)
        self.assertAlmostEqual(summary.mae_skill, 1 - 1 / 4.5)
        self.assertAlmostEqual(summary.nonclimate_mae, 3.5)
        self.assertAlmostEqual(summary.climate_added_skill, 1 - 1 / 3.5)
        self.assertEqual(summary.holdout_rows, 3)
        self.assertEqual(summary.holdout_groups, 2)
        importance = pd.read_csv(self.out / 'feature_importance.csv')
        self.assertAlmostEqual(importance.iloc[0].baseline_relative_importance, 2 / 4.5)
        self.assertEqual(len(pd.read_csv(self.out / 'model_leaderboard.csv')), 2)
        predictions = pd.read_csv(self.out / 'predictions.csv')
        self.assertEqual(len(predictions), 3)
        self.assertIn('private_group_1', set(predictions.group_id))
        html = (self.out / 'dashboard.html').read_text()
        self.assertNotIn('private_group_', html)
        self.assertIn('Observed deduct-family total', html)
        self.assertNotIn('points/year', html)
        self.assertIn('Permuting the whole climate block', html)
        self.assertIn('climate_temp [climate]', html)
        self.assertIn('repeat standard deviation, not confidence intervals', html)
        self.assertIn('not causal impact', html)
        self.assertIn('@media print', html)
        self.assertIn('table-layout:fixed', html)
        self.assertNotIn('https://', html)
        self.assertNotIn('<script', html)

    def test_rank_uses_skill_not_raw_mae_and_preserves_skipped_slots(self):
        second = self.out / 'materials/PCC/family_other'
        second.mkdir(parents=True)
        pd.DataFrame({'actual': [100., 200.], 'prediction': [110., 190.],
                      'baseline_prediction': [0., 0.], 'weight': [1., 1.]}).to_csv(
                          second / 'holdout_predictions.csv', index=False)
        other = {'material': 'PCC', 'target': 'family_other', 'status': 'modeled'}
        skipped = {'material': 'APC', 'target': 'family_none', 'status': 'no_valid_target'}
        build_study_report(self.out, [self.summary, other, skipped], {})
        table = pd.read_csv(self.out / 'target_summary.csv')
        self.assertEqual(table.target.tolist(), ['family_other', 'family_example', 'family_none'])
        self.assertTrue(pd.isna(table.iloc[2].holdout_mae))
        html = (self.out / 'dashboard.html').read_text()
        self.assertIn('no_valid_target', html)
        self.assertIn('Feature rankings are exploratory', html)

    def test_zero_baseline_and_constant_holdout_are_undefined_not_perfect(self):
        self.predictions['actual'] = 0.
        self.predictions['prediction'] = 0.
        self.predictions['baseline_prediction'] = 0.
        self.predictions['nonclimate_prediction'] = 0.
        self.predictions.to_csv(self.task / 'holdout_predictions.csv', index=False)
        build_study_report(self.out, [self.summary], {})
        row = pd.read_csv(self.out / 'target_summary.csv').iloc[0]
        self.assertTrue(pd.isna(row.mae_skill))
        self.assertTrue(pd.isna(row.holdout_r2))
        self.assertTrue(pd.isna(row.climate_added_skill))
        self.assertTrue(pd.read_csv(self.out / 'feature_importance.csv').baseline_relative_importance.isna().all())

    def test_chart_caps_points_but_export_keeps_every_prediction(self):
        pd.concat([self.predictions] * 300, ignore_index=True).to_csv(self.task / 'holdout_predictions.csv', index=False)
        build_study_report(self.out, [self.summary], {})
        html = (self.out / 'dashboard.html').read_text()
        self.assertEqual(html.count('<circle '), 600)
        self.assertIn('600 of 900 holdout points shown', html)
        self.assertEqual(len(pd.read_csv(self.out / 'predictions.csv')), 900)

    def test_html_escaping_and_path_guards(self):
        self.summary['candidate'] = '<img src=x onerror=alert(1)>'
        importance = pd.read_csv(self.task / 'importance.csv')
        importance.loc[0, 'feature'] = '<svg onload=alert(1)>'
        importance.to_csv(self.task / 'importance.csv', index=False)
        build_study_report(self.out, [self.summary], {'features': ['<script>']})
        html = (self.out / 'dashboard.html').read_text()
        self.assertNotIn('<img', html)
        self.assertNotIn('<script>', html)
        self.assertNotIn('<svg onload', html)
        self.assertIn('&lt;img', html)
        self.assertIn('&lt;svg', html)
        with self.assertRaisesRegex(SafeError, 'INVALID_REPORT_TASK'):
            build_study_report(self.out, [{**self.summary, 'target': '../escape'}], {})
        with self.assertRaisesRegex(SafeError, 'DUPLICATE_REPORT_TASK'):
            build_study_report(self.out, [self.summary, self.summary], {})
        (self.out / 'target_summary.csv').unlink()
        (self.out / 'target_summary.csv').symlink_to(self.task / 'holdout_predictions.csv')
        with self.assertRaisesRegex(SafeError, 'SYMLINK_PATH_FORBIDDEN'):
            build_study_report(self.out, [self.summary], {})

    def test_empty_study_and_invalid_saved_predictions(self):
        build_study_report(self.out, [], {})
        for filename in ['target_summary.csv', 'feature_importance.csv', 'model_leaderboard.csv', 'predictions.csv']:
            self.assertTrue(pd.read_csv(self.out / filename).empty)
        self.assertIn('No target tasks were recorded', (self.out / 'dashboard.html').read_text())
        self.predictions.loc[0, 'weight'] = -1
        self.predictions.to_csv(self.task / 'holdout_predictions.csv', index=False)
        with self.assertRaisesRegex(SafeError, 'INVALID_STUDY_REPORT_PREDICTIONS'):
            build_study_report(self.out, [self.summary], {})

    def test_climate_coverage_and_section_grouping_are_visible_without_private_provenance(self):
        audit = {'observations': {'accounting': {'input_rows': 30, 'accepted_rows': 25, 'rejected_rows': 5}},
                 'climate': {'mode': 'existing', 'rows': 25, 'eligible_rows': 20,
                             'lookback_days': 365, 'historical_window_verified': False,
                             'counts': {'missing_location': 2, 'incomplete_hourly_coverage': 3},
                             'requests': ['private_location_request'],
                             'location_sources': ['private_location_source']},
                 'partition': {'grouping': 'section_identity'},
                 'model_rows': 20, 'excluded_climate_rows': 5}
        build_study_report(self.out, [self.summary], audit)
        html = (self.out / 'dashboard.html').read_text()
        self.assertIn('Incomplete hourly climate coverage', html)
        self.assertIn('Rows with complete eligible climate', html)
        self.assertIn('<td>Input inspections</td><td>30</td>', html)
        self.assertIn('<td>Accepted inspections</td><td>25</td>', html)
        self.assertIn('not independently verified', html)
        self.assertIn('does not establish performance', html)
        self.assertNotIn('private_location', html)

    def test_climate_grouping_and_constant_development_scope_remain_explicit(self):
        self.summary.update(status='constant_target', constant_target_scope='development',
                            candidate='median', importance_reliable=False)
        build_study_report(self.out, [self.summary], {'partition': {
            'grouping': 'climate_location', 'declared_groups': 15, 'validation_groups': 8}})
        html = (self.out / 'dashboard.html').read_text()
        self.assertIn('Groups joined by shared ERA5 grid locations', html)
        self.assertIn('Declared groups: 15', html)
        self.assertIn('Final validation groups: 8', html)
        self.assertIn('constant in the development data', html)
        self.assertIn('which may have different values', html)
        row = pd.read_csv(self.out / 'target_summary.csv').iloc[0]
        self.assertEqual(row.constant_target_scope, 'development')

    def test_sparsity_flag_is_descriptive_and_uses_strict_five_percent_threshold(self):
        for prevalence, expected in [(0., True), (.049, True), (.05, False), (.8, False)]:
            with self.subTest(prevalence=prevalence):
                self.summary['nonzero_fraction'] = prevalence
                build_study_report(self.out, [self.summary], {})
                row = pd.read_csv(self.out / 'target_summary.csv').iloc[0]
                self.assertEqual(bool(row.mostly_zero_target), expected)
                html = (self.out / 'dashboard.html').read_text()
                self.assertEqual('Mostly zero target:' in html, expected)
                self.assertIn('not an accuracy judgment', html)
                self.assertIn('Comparing many targets on this holdout is exploratory', html)
                self.assertIn('additional independent data', html)


if __name__ == '__main__':
    unittest.main()
