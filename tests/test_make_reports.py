from __future__ import annotations


from pathlib import Path
import sys
import types
import unittest
from unittest import mock


import numpy as np
import pandas as pd


from e_evaluate_and_report import make_reports as reports




class FakeEstimator:
    classes_ = np.array([0, 1])


    def fit(self, *args, **kwargs):
        raise AssertionError("Reporting must never fit a model")


    def load_model(self, path):
        pass


    def predict_proba(self, matrix):
        probabilities = np.asarray(matrix)[:, 0]
        return np.column_stack([1 - probabilities, probabilities])




class SavedReportingTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "ID": ["a", "b", "c", "d"],
            "COHORT": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-02-01", "2023-08-01"]),
            "CHECK": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-02-01", "2024-08-01"]),
            "TARGET": [1, 0, 1, 0], "FEATURE": [0.2, 0.4, 0.8, 0.9],
        })


    def manifest(self, family="logistic"):
        return {
            "model_family": family, "features": ["FEATURE"],
            "derived_feature_contracts": {}, "target_column": "TARGET",
            "model_file": "model.joblib", "model_files": ["seed_a.ubj", "seed_b.ubj"],
        }


    def api(self, manifest):
        def transform(fitted, part):
            self.assertEqual(list(part.columns), ["FEATURE"])
            self.assertLessEqual(len(part), 2)
            return part.to_numpy()
        return types.SimpleNamespace(
            load_bundle=mock.Mock(return_value=(manifest, {})),
            transform_frame=mock.Mock(side_effect=transform),
        )


    def test_holdout_is_rejected_before_loading_or_transforming(self):
        api = self.api(self.manifest())
        with self.assertRaisesRegex(ValueError, "Final test"):
            list(reports.score_batches(
                self.frame(), np.array([0, 3]), Path("unused"), self.manifest(), api,
                holdout_start="2023-08-01", cohort_col="COHORT", batch_rows=2,
            ))
        api.load_bundle.assert_not_called()
        api.transform_frame.assert_not_called()


    def test_predictor_only_batching_and_logistic_probability_meaning(self):
        manifest = self.manifest()
        api = self.api(manifest)
        with mock.patch("joblib.load", return_value=FakeEstimator()) as load:
            result = list(reports.score_batches(
                self.frame(), np.array([0, 1, 2]), Path("unused"), manifest, api,
                holdout_start="2023-08-01", cohort_col="COHORT", batch_rows=2,
            ))
        np.testing.assert_allclose(np.concatenate([p for _, p in result]), [0.2, 0.4, 0.8])
        load.assert_called_once()
        api.load_bundle.assert_called_once()
        self.assertEqual(api.transform_frame.call_count, 2)


    def test_xgboost_churn_is_inverted_once_and_seeds_load_once(self):
        manifest = self.manifest("xgboost")
        api = self.api(manifest)
        factory = mock.Mock(side_effect=FakeEstimator)
        with mock.patch.dict(sys.modules, {"xgboost": types.SimpleNamespace(XGBClassifier=factory)}):
            result = list(reports.score_batches(
                self.frame(), np.array([0, 1, 2]), Path("unused"), manifest, api,
                holdout_start="2023-08-01", cohort_col="COHORT", batch_rows=2,
            ))
        np.testing.assert_allclose(np.concatenate([p for _, p in result]), [0.8, 0.6, 0.2])
        self.assertEqual(factory.call_count, 2)
        api.load_bundle.assert_called_once()


    def test_training_signature_and_cutoff_fail_closed(self):
        frame = self.frame()
        population = {"identity_columns": ["ID"], "rows": 2, "population_sha256": "expected"}
        api = types.SimpleNamespace(population_sha256=lambda part, cols: "wrong")
        with self.assertRaisesRegex(ValueError, "population identity"):
            reports.verify_training_population(
                frame, np.array([0, 1]), {"training_population": population}, api,
                cohort_col="COHORT", target_check_col="CHECK",
                label_cutoff="2023-08-01", holdout_start="2023-08-01",
            )
        with self.assertRaisesRegex(ValueError, "strict label cutoff"):
            reports.verify_training_population(
                frame, np.array([0, 1]), {"training_population": population}, api,
                cohort_col="COHORT", target_check_col="CHECK",
                label_cutoff="2023-01-02", holdout_start="2023-08-01",
            )
        population["identity_columns"] = ["MISSING_ID"]
        with self.assertRaisesRegex(ValueError, "identity column"):
            reports.verify_training_population(
                frame, np.array([0, 1]), {"training_population": population}, api,
                cohort_col="COHORT", target_check_col="CHECK",
                label_cutoff="2023-08-01", holdout_start="2023-08-01",
            )


    def test_partial_training_month_uses_exact_membership_rates(self):
        part = reports.monthly_partial(
            self.frame(), np.array([0, 1, 2]), np.array([0.2, 0.4, 0.8]), np.array([0, 2]),
            cohort_col="COHORT", target_col="TARGET",
        )
        result = reports.finalize_monthly([part])
        jan = result.iloc[0]
        self.assertEqual(jan["policies"], 2)
        self.assertEqual(jan["actual_retention"], 0.5)
        self.assertEqual(jan["selected_history_training_rows"], 1)
        self.assertEqual(jan["training_actual_retention"], 1.0)
        self.assertAlmostEqual(jan["training_predicted_retention"], 0.2)
        self.assertAlmostEqual(jan["predicted_retention"], 0.3)


    def test_derived_inputs_require_raw_sources_and_never_target(self):
        manifest = self.manifest()
        manifest["features"] = ["YEAR", "STATE_GROUP", "PREM_BIN"]
        manifest["derived_feature_contracts"] = {
            "year": {"derived_feature": "YEAR", "source_feature": "COHORT"},
            "state": {"derived_feature": "STATE_GROUP", "source_feature": "STATE"},
            "premium": {"derived_feature": "PREM_BIN", "source_feature": "PREMIUM"},
        }
        self.assertEqual(reports.raw_scoring_columns(manifest), ["COHORT", "STATE", "PREMIUM"])
        manifest["features"].append("TARGET")
        with self.assertRaisesRegex(ValueError, "target"):
            reports.raw_scoring_columns(manifest)


    def test_validation_predictions_are_aligned_and_outcomes_checked(self):
        validation = self.frame().iloc[:3]
        predictions = pd.DataFrame({
            "model_family": ["logistic"] * 3, "fold": [1] * 3,
            "row_id": ["c", "a", "b"], "y_retention": [1, 1, 0],
            "cohort_date": validation["COHORT"].iloc[[2, 0, 1]].to_numpy(),
            "p_retention": [0.8, 0.2, 0.4],
        })
        c4 = types.SimpleNamespace(stable_row_ids=lambda part, columns: part["ID"])
        kwargs = dict(family="logistic", fold=1, target_col="TARGET", cohort_col="COHORT",
                      identity_columns=["ID"], c4=c4)
        result = reports.validated_outer_rows(predictions, validation, **kwargs)
        self.assertEqual(result.index.tolist(), ["a", "b", "c"])
        predictions.loc[0, "y_retention"] = 0
        with self.assertRaisesRegex(ValueError, "outcomes"):
            reports.validated_outer_rows(predictions, validation, **kwargs)




if __name__ == "__main__":
    unittest.main()
