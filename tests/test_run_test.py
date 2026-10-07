"""Synthetic checks for final-test scoring and recovery (no production data)."""
from __future__ import annotations


import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


import numpy as np
import pandas as pd


from e_evaluate_and_report import run_test as runner, make_reports as reports




def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()




def population(frame, columns):
    return hashlib.sha256(frame[columns].to_csv(index=False).encode()).hexdigest()




def index_digest(indices):
    return hashlib.sha256(np.sort(np.asarray(indices)).astype('<i8').tobytes()).hexdigest()




class FinalTestRunnerTests(unittest.TestCase):
    def fixture(self):
        dates = [pd.Timestamp('2020-01-01')] * 2
        for i in range(5):
            dates.extend([pd.Timestamp('2022-05-01') + pd.DateOffset(months=3*i)] * 2)
        for date in pd.date_range('2023-08-01', periods=6, freq='MS'):
            dates.extend([date] * 2)
        frame = pd.DataFrame({'ID': [f'id{i}' for i in range(24)], 'COHORT': dates,
                              'TARGET': [0, 1] * 12, 'X': [.2, .8] * 12, 'TENURE': ['1'] * 24})
        frame['CHECK'] = frame.COHORT + pd.DateOffset(months=12)
        indices = np.arange(12, 24)
        excluded = dict(rows=12, cohort_months=6, cohort_start='2023-08-01', cohort_end='2024-01-01',
                        identity_columns=['ID', 'COHORT'], population_sha256=population(frame.loc[indices], ['ID', 'COHORT']),
                        row_index_sha256=index_digest(indices))
        manifest = dict(model_family='logistic', features=['X'], target_column='TARGET', derived_feature_contracts={},
                        source_metadata={'bundle_role': 'FINAL_REFIT', 'sealed_final_holdout': excluded},
                        training_population={'identity_columns': ['ID', 'COHORT'], 'rows': 2,
                                             'population_sha256': population(frame.iloc[:2], ['ID', 'COHORT'])})
        def iterator(path, predictors, batch_size, verify_hashes):
            self.assertEqual(list(predictors.columns), ['X'])
            for start in range(0, len(predictors), batch_size):
                stop = min(start+batch_size, len(predictors))
                yield start, stop, predictors.X.iloc[start:stop].to_numpy()
        api = types.SimpleNamespace(population_sha256=population, sha256_file=digest,
                                    iter_logistic_bundle_scores=mock.Mock(side_effect=iterator),
                                    iter_xgboost_bundle_scores=mock.Mock(side_effect=iterator))
        c4 = types.SimpleNamespace(_index_sha256=index_digest,
                                   stable_row_ids=lambda part, cols: part.ID)
        return frame, indices, excluded, manifest, api, c4


    def test_exact_test_identity_checked_before_transform(self):
        frame, indices, excluded, manifest, api, c4 = self.fixture()
        frame.loc[12, 'ID'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'population_sha256'):
            runner.score_test(frame, indices, excluded, Path('unused'), manifest, api, c4, cohort_col='COHORT', batch_rows=4)
        api.iter_logistic_bundle_scores.assert_not_called()


    def test_both_polarities_predictors_only_and_batch_progress(self):
        frame, indices, excluded, manifest, api, c4 = self.fixture()
        for family in ('logistic', 'xgboost'):
            manifest['model_family'] = family
            completed = []
            result = runner.score_test(frame, indices, excluded, Path('unused'), manifest, api, c4,
                                       cohort_col='COHORT', batch_rows=5, on_batch=completed.append)
            expected = frame.loc[indices, 'X'].to_numpy()
            np.testing.assert_allclose(result, expected if family == 'logistic' else 1-expected)
            self.assertEqual(completed, [5, 10, 12])


    def test_incomplete_or_nonfinite_batches_are_rejected(self):
        frame, indices, excluded, manifest, api, c4 = self.fixture()
        for broken in ([(0, 2, np.array([.2, np.nan]))], [(1, 3, np.array([.2, .8]))], [(0, 2, np.array([.2, .8]))]):
            api.iter_logistic_bundle_scores.side_effect = None
            api.iter_logistic_bundle_scores.return_value = iter(broken)
            with self.assertRaises(ValueError):
                runner.score_test(frame, indices, excluded, Path('unused'), manifest, api, c4, cohort_col='COHORT', batch_rows=4)


    def test_cached_reports_refuse_changed_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Previously evaluated frozen model'):
                runner.cached_predictions(Path(directory), {'source_identity': {'model': 'old'}, 'scoring_complete': True},
                                          {'model': 'new'}, types.SimpleNamespace())


    def test_replot_failure_preserves_test_exposure(self):
        part = pd.DataFrame(dict(y_retention=[0, 1], p_retention=[.2, .8], fold=[1, 1]))
        validation = pd.concat([part.assign(fold=i) for i in range(1, 6)])
        charts = types.SimpleNamespace(write_final_test_comparison=mock.Mock(side_effect=RuntimeError('plot failure')))
        status = dict(selected_model_family='logistic', final_training_rows=2,
                      scoring_complete=True, final_holdout_rows_scored=2, plots_complete=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'plot failure'):
                runner.write_outputs(dict(train=part, validation=validation, test=part), output, status, charts, None, None)
            saved = json.loads((output/'test_status.json').read_text())
            self.assertEqual(saved['status'], 'FAILED')
            self.assertFalse(saved['plots_complete'])
            self.assertTrue(saved['scoring_complete'])
            self.assertEqual(saved['final_holdout_rows_scored'], 2)


    def test_complete_run_then_cached_retry_without_refit_or_rescore(self):
        frame, indices, excluded, manifest, api, c4 = self.fixture()
        train_idx = np.arange(2)
        audit = {'label_cutoff_exclusive': '2023-08-01'}
        c4.final_evaluation_training_indices = mock.Mock(return_value=(train_idx, audit))
        folds = [dict(fold=i+1, train_idx=train_idx, validation_idx=np.arange(2+2*i, 4+2*i), label_cutoff='2022-05-01') for i in range(5)]
        config = types.SimpleNamespace(cohort_date_col='COHORT', target_check_col='CHECK', target_col='TARGET', tenure_col='TENURE')
        params = types.SimpleNamespace(ROLLING_VALIDATION_FOLDS=5, OUTER_VALIDATION_MONTHS=3,
                                       FINAL_HOLDOUT_MONTHS=6, LABEL_AVAILABILITY_BUFFER_DAYS=0)
        utils = types.SimpleNamespace(read_table=mock.Mock(return_value=frame),
                                      load_project_population_and_folds=mock.Mock(return_value=(frame, folds, indices, {}, config)),
                                      fold_definitions_frame=mock.Mock(return_value=pd.DataFrame()),
                                      verify_fold_definition_contract=mock.Mock())
        charts = types.SimpleNamespace(write_final_test_comparison=mock.Mock(return_value={'artifacts': []}))
        diagnostics = types.SimpleNamespace(prediction_metrics=mock.Mock(return_value={
            'retention_roc_auc': 1.0, 'churn_average_precision': 1.0, 'top10_lift': 2.0}))
        perf = types.SimpleNamespace(generate_performance_plots=mock.Mock(return_value={'artifacts': []}))
        package = types.ModuleType('c_modeling')
        for name, value in dict(deployment_bundle=api, modeling_utils=utils,
                                c4_evaluate_and_finalize=c4,
                                model_diagnostics=diagnostics, performance_plots=perf).items():
            setattr(package, name, value)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'e_evaluate_and_report').mkdir()
            run_root = root/'output'/'pipeline_runs'/'RUN'
            for name in ('preparation', 'outer_evaluation', 'deployment', 'input_snapshot'):
                (run_root/name).mkdir(parents=True)
            def write(relative, value):
                path = run_root/relative
                path.write_text(json.dumps(value))
                return path
            snapshot = run_root/'input_snapshot'/'retention_13_1_population.parquet'
            snapshot.write_bytes(b'frozen synthetic snapshot')
            selection = write('outer_evaluation/family_selection.json', {'selected_model_family': 'logistic'})
            final = dict(run_id='RUN', profile_hash='profile', final_holdout_rows_scored=0,
                         selected_model_family='logistic', selected_bundle={'sha256': 'a'*64},
                         family_selection_sha256=digest(selection), excluded_holdout=excluded, final_training_reconstruction=audit)
            final_path = write('deployment/final_model_manifest.json', final)
            oof = pd.concat([pd.DataFrame(dict(row_id=frame.loc[f['validation_idx'], 'ID'], model_family='logistic', fold=f['fold'],
                                               cohort_date=frame.loc[f['validation_idx'], 'COHORT'],
                                               y_retention=frame.loc[f['validation_idx'], 'TARGET'],
                                               p_retention=frame.loc[f['validation_idx'], 'X'])) for f in folds], ignore_index=True)
            oof_path = run_root/'outer_evaluation'/'outer_oof_predictions.parquet'
            oof.to_pickle(oof_path)
            write('outer_evaluation/outer_fold_bundle_manifest.json', dict(run_id='RUN', profile_hash='profile', selected_model_family='logistic',
                  final_holdout_rows_scored=0, final_model_manifest={'sha256': digest(final_path)},
                  source_data_identity={'sha256': digest(snapshot)}, identity_columns=['ID', 'COHORT'],
                  outer_oof_predictions={'sha256': digest(oof_path), 'rows': len(oof)}))
            write('outer_evaluation/run_status.json', {'status': 'COMPLETE'})
            write('preparation/preparation_info.json', {'profile_hash': 'profile', 'input_sha256': digest(snapshot),
                'split_settings': dict(rolling_validation_folds=5, outer_validation_months=3, final_holdout_months=6,
                                       label_availability_buffer_days=0, min_outer_train_months=12)})
            original_hashes = {p: digest(p) for p in run_root.rglob('*') if p.is_file()}
            with mock.patch.dict(sys.modules, {'c_modeling': package}), \
                 mock.patch.object(runner, '__file__', str(root/'e_evaluate_and_report'/'run_test.py')), \
                 mock.patch.object(runner, 'charts', charts), \
                 mock.patch.object(reports, 'activate_profile', return_value=params), \
                 mock.patch.object(reports, 'load_saved_bundle', return_value=(Path('unused'), manifest)), \
                 mock.patch.object(reports, 'selected_training_indices', return_value=train_idx), \
                 mock.patch.object(reports, 'score_batches', return_value=[(train_idx, np.array([.2, .8]))]) as train_score, \
                 mock.patch.object(pd, 'read_parquet', side_effect=pd.read_pickle), \
                 mock.patch.object(pd.DataFrame, 'to_parquet', lambda self, path, index=False: self.to_pickle(path)):
                output = runner.run(runner.parse_args(['--run-id', 'RUN', '--batch-rows', '5']))
                self.assertEqual(api.iter_logistic_bundle_scores.call_count, 1)
                self.assertEqual(train_score.call_count, 1)
                self.assertEqual(perf.generate_performance_plots.call_args.kwargs['evaluation_scope'], 'final_test')
                utils.read_table.reset_mock()
                runner.run(runner.parse_args(['--run-id', 'RUN']))
                utils.read_table.assert_not_called()
                self.assertEqual(api.iter_logistic_bundle_scores.call_count, 1)
                self.assertEqual(train_score.call_count, 1)
            saved = json.loads((output/'test_status.json').read_text())
            self.assertTrue(saved['scoring_complete'])
            self.assertEqual(saved['final_holdout_rows_scored'], 12)
            self.assertFalse(saved['training_performed'])
            self.assertFalse(saved['deployment_eligible'])
            self.assertEqual(saved['status'], 'COMPLETE')
            self.assertEqual({p: digest(p) for p in original_hashes}, original_hashes)




if __name__ == '__main__':
    unittest.main()
