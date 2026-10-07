"""Focused tests for fold-honest performance visualization contracts."""


from __future__ import annotations


from pathlib import Path
import tempfile
import unittest


import pandas as pd
from sklearn.metrics import roc_auc_score


from c_modeling.performance_plots import (
    generate_performance_plots,
    plot_roc_pr,
    validate_prediction_frame,
)




class PerformancePlotContractTests(unittest.TestCase):
    def test_final_test_plots_keep_their_evaluation_scope_and_write_real_images(self) -> None:
        frame = self.shifted_fold_scores().assign(fold=1)
        original = frame.copy(deep=True)
        expected = {"roc_pr", "percentile_lift_gains", "decile_lift", "calibration", "temporal_stability"}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = generate_performance_plots(frame, output, evaluation_scope="final_test")
            self.assertEqual(manifest["evaluation_scope"], "final_test")
            self.assertEqual(manifest["model_families"], ["logistic"])
            self.assertEqual(
                {Path(item["path"]).name for item in manifest["artifacts"]},
                {f"final_test_{kind}.png" for kind in expected},
            )
            for item in manifest["artifacts"]:
                path = Path(item["path"])
                self.assertEqual(path.parent, output.resolve())
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                self.assertEqual(item["evaluation_scope"], "final_test")
        pd.testing.assert_frame_equal(frame, original)


    def test_final_test_rejects_multiple_fold_refits(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one evaluation partition"):
            validate_prediction_frame(self.shifted_fold_scores(), evaluation_scope="final_test")


    @staticmethod
    def shifted_fold_scores() -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        fold_scores = {
            1: ((0.42, 0.40), (0.35, 0.33)),
            2: ((0.22, 0.20), (0.15, 0.13)),
        }
        for fold, (churn_scores, retained_scores) in fold_scores.items():
            for position, p_churn in enumerate(churn_scores):
                rows.append(
                    {
                        "row_id": f"F{fold}-C{position}",
                        "model_family": "logistic",
                        "y_retention": 0,
                        "p_retention": 1.0 - p_churn,
                        "fold": fold,
                        "cohort_date": f"2022-0{fold}-01",
                    }
                )
            for position, p_churn in enumerate(retained_scores):
                rows.append(
                    {
                        "row_id": f"F{fold}-R{position}",
                        "model_family": "logistic",
                        "y_retention": 1,
                        "p_retention": 1.0 - p_churn,
                        "fold": fold,
                        "cohort_date": f"2022-0{fold}-01",
                    }
                )
        return pd.DataFrame(rows)


    def test_outer_roc_reports_macro_fold_metrics_not_pooled_scores(self) -> None:
        frame = validate_prediction_frame(
            self.shifted_fold_scores(),
            evaluation_scope="outer_oof",
        )
        pooled_auc = roc_auc_score(
            1 - frame["y_retention"],
            1 - frame["p_retention"],
        )
        self.assertLess(pooled_auc, 1.0)


        with tempfile.TemporaryDirectory() as temporary:
            artifact = plot_roc_pr(
                frame,
                Path(temporary) / "macro_roc_pr.png",
                evaluation_scope="outer_oof",
            )
            summary = artifact["fold_metric_summary"]["logistic"]
            self.assertEqual(artifact["curve_aggregation"], "macro_average_equal_fold_weight")
            self.assertEqual(summary["fold_count"], 2)
            self.assertAlmostEqual(summary["mean_fold_auc"], 1.0)
            self.assertAlmostEqual(summary["sd_fold_auc"], 0.0)
            self.assertTrue(Path(artifact["path"]).is_file())


    def test_row_identity_is_unique_and_aligned_across_models(self) -> None:
        logistic = self.shifted_fold_scores()
        xgboost = logistic.assign(model_family="xgboost")
        aligned = pd.concat([logistic, xgboost], ignore_index=True)
        validate_prediction_frame(aligned, evaluation_scope="outer_oof")


        duplicate = pd.concat([aligned, aligned.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_prediction_frame(duplicate, evaluation_scope="outer_oof")


        misaligned = aligned.copy()
        xgb_first = misaligned.index[misaligned["model_family"].eq("xgboost")][0]
        misaligned.loc[xgb_first, "row_id"] = "DIFFERENT-POLICY"
        with self.assertRaisesRegex(ValueError, "aligned"):
            validate_prediction_frame(misaligned, evaluation_scope="outer_oof")




if __name__ == "__main__":
    unittest.main()
