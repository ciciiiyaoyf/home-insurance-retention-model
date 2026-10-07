from __future__ import annotations


import unittest
from pathlib import Path
import tempfile
import types


import numpy as np
import pandas as pd


from c_modeling.c4_evaluate_and_finalize import (
    _archive_partial_c4_attempt,
    _declared_input_sha256,
    _separate_outer_performance_artifacts,
    aggregate_final_refit_descriptive_monthly,
    choose_model_family,
    final_evaluation_training_indices,
    final_refit_descriptive_indices,
    main,
    score_full_history_family,
    stable_row_ids,
    validated_history_months,
)




class C4EvaluationContractTests(unittest.TestCase):
    def test_full_history_scoring_excludes_holdout_for_both_families(self) -> None:
        frame = pd.DataFrame({
            "COHORT": pd.to_datetime(["2021-01-01"] * 2 + ["2021-02-01"] * 2 + ["2021-03-01"] * 2),
            "TARGET": [0, 1] * 3,
        })
        retention_probability = np.asarray([0.1, 0.5, 0.2, 0.4])
        for family in ("logistic", "xgboost"):
            with self.subTest(family=family):
                scored = []


                def batches(path, rows, *, batch_size, verify_hashes):
                    scored.extend(rows.index.tolist())
                    self.assertTrue(verify_hashes)
                    probability = retention_probability if family == "logistic" else 1 - retention_probability
                    yield 0, 3, probability[:3]
                    yield 3, 4, probability[3:]


                api = types.SimpleNamespace(iter_logistic_bundle_scores=batches, iter_xgboost_bundle_scores=batches)
                monthly, by_role, audit = score_full_history_family(
                    prepared=frame, final_holdout_idx=np.asarray([4, 5]),
                    selected_training_idx=np.asarray([1, 2]), family=family,
                    bundle_path=Path("unused"), cohort_col="COHORT", target_col="TARGET",
                    history_audit={"cutoff_month_inclusive": "2021-01", "anchor_month": "2021-02"},
                    deployment_bundle=api, batch_rows=3,
                )
                self.assertEqual(scored, [0, 1, 2, 3])
                self.assertEqual(monthly["policies"].tolist(), [2, 2])
                self.assertEqual(monthly["selected_history_training_rows"].tolist(), [1, 1])
                np.testing.assert_allclose(monthly["predicted_retention"], [0.3, 0.3])
                self.assertEqual(set(by_role["model_family"]), {family})
                self.assertEqual(audit["holdout_rows_scored"], 0)
                self.assertFalse(audit["performance_evidence"])


    def test_selector_shap_is_separate_from_outer_performance_artifacts(self) -> None:
        performance_path = "/tmp/outer_oof_roc_pr.png"
        shap_path = "/tmp/xgboost_inner_shap_beeswarm.png"
        manifest = _separate_outer_performance_artifacts(
            {
                "evaluation_scope": "outer_oof",
                "artifacts": [
                    {
                        "kind": "roc_pr",
                        "path": performance_path,
                        "evaluation_scope": "outer_oof",
                    }
                ],
            },
            interpretability_artifacts=[
                {
                    "kind": "shap_beeswarm_interpretability",
                    "path": shap_path,
                    "evaluation_scope": "inner_newest_chronological_validation",
                    "performance_evidence": False,
                    "interpretability_only": True,
                }
            ],
        )
        self.assertEqual(manifest["artifacts"], [performance_path])
        self.assertEqual(
            [item["path"] for item in manifest["artifact_metadata"]],
            [performance_path],
        )
        self.assertEqual(
            [item["path"] for item in manifest["interpretability_artifacts"]],
            [shap_path],
        )
        self.assertNotIn(
            shap_path,
            {item["path"] for item in manifest["artifact_metadata"]},
        )


    def test_failed_c4_retry_archives_partial_run_trees(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "RUN"
            (run_root / "outer_evaluation").mkdir(parents=True)
            (run_root / "outer_evaluation" / "partial.txt").write_text("x")
            (run_root / "deployment").mkdir()
            records = _archive_partial_c4_attempt(
                run_root, recovery_attempt_id="20260903-120000-000001"
            )
            self.assertEqual(len(records), 2)
            self.assertFalse((run_root / "outer_evaluation").exists())
            self.assertTrue(
                (
                    run_root
                    / "outer_evaluation_failed_attempt_20260903-120000-000001"
                    / "partial.txt"
                ).is_file()
            )


    def test_declared_input_sha256_supports_selector_receipts(self) -> None:
        self.assertEqual(
            _declared_input_sha256(
                {"data": {"sha256": "ABC"}}, label="Logistic run_info"
            ),
            "abc",
        )
        self.assertEqual(
            _declared_input_sha256(
                {"input": {"sha256": "DEF"}}, label="XGBoost run_info"
            ),
            "def",
        )


    def test_c4_refuses_to_open_the_final_holdout(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "never opens the final holdout"):
            main(["--score-final-holdout"])


    def test_validated_history_months_accepts_only_governed_recent_grid(self) -> None:
        self.assertEqual(
            validated_history_months(
                {"history_id": "RECENT_36M", "kind": "recent", "months": 36},
                family="Logistic",
            ),
            36,
        )
        with self.assertRaisesRegex(ValueError, "RECENT_36M"):
            validated_history_months(
                {
                    "history_id": "DECAY_HALF_LIFE_24M",
                    "kind": "decay",
                    "months": 24,
                },
                family="XGBoost",
            )


    def test_final_training_excludes_holdout_and_unmatured_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "cohort": pd.to_datetime(
                    ["2022-06-01", "2022-07-01", "2023-07-01", "2023-08-01"]
                ),
                "check": pd.to_datetime(
                    ["2023-06-30", "2023-07-31", "2024-07-31", "2024-08-31"]
                ),
            }
        )
        selected, audit = final_evaluation_training_indices(
            frame,
            np.asarray([3]),
            cohort_col="cohort",
            target_check_col="check",
            label_buffer_days=0,
        )
        self.assertEqual(selected.tolist(), [0, 1])
        self.assertEqual(audit["holdout_start"], "2023-08-01")
        self.assertEqual(audit["candidate_rows"], 2)


    def test_final_refit_descriptive_indices_exclude_sealed_holdout(self) -> None:
        frame = pd.DataFrame(
            {
                "cohort": pd.to_datetime(
                    [
                        "2023-05-01",
                        "2023-06-01",
                        "2023-07-01",
                        "2023-08-01",
                        "2023-09-01",
                    ]
                )
            },
            index=np.arange(10, 15),
        )
        selected, audit = final_refit_descriptive_indices(
            frame,
            np.asarray([13, 14]),
            np.asarray([10, 11]),
            cohort_col="cohort",
        )
        self.assertEqual(selected.tolist(), [10, 11, 12])
        self.assertEqual(audit["cohort_end"], "2023-07-01")
        self.assertEqual(audit["holdout_start"], "2023-08-01")
        self.assertEqual(audit["holdout_rows_scored"], 0)
        self.assertFalse(audit["performance_evidence"])
        self.assertEqual(np.intersect1d(selected, [13, 14]).size, 0)


    def test_final_refit_monthly_summary_uses_exact_training_membership(self) -> None:
        frame = pd.DataFrame(
            {
                "cohort": pd.to_datetime(
                    [
                        "2020-01-01",
                        "2020-01-01",
                        "2020-02-01",
                        "2020-02-01",
                        "2020-03-01",
                        "2020-03-01",
                    ]
                ),
                "target": [1, 0, 1, 1, 0, 1],
            },
            index=[10, 11, 12, 13, 14, 15],
        )
        probability = np.asarray([0.8, 0.4, 0.7, 0.9, 0.2, 0.6])
        # Deliberately choose non-contiguous rows. Membership must come from
        # these exact indices, not from a broad date-range approximation.
        monthly = aggregate_final_refit_descriptive_monthly(
            frame,
            probability,
            cohort_col="cohort",
            target_col="target",
            selected_training_idx=np.asarray([11, 12, 15]),
        )
        self.assertEqual(monthly["YEAR_MONTH"].tolist(), [
            "2020-01", "2020-02", "2020-03"
        ])
        self.assertEqual(
            monthly["selected_history_training_rows"].tolist(), [1, 1, 1]
        )
        self.assertEqual(
            monthly["population_role"].tolist(),
            ["MIXED_HISTORY_MEMBERSHIP"] * 3,
        )
        self.assertTrue(
            np.allclose(monthly["actual_retention"], [0.5, 1.0, 0.5])
        )
        self.assertTrue(
            np.allclose(monthly["predicted_retention"], [0.6, 0.8, 0.4])
        )
        self.assertTrue(
            np.allclose(
                monthly["predicted_retention_3m_weighted"],
                [0.6, 0.7, 0.6],
            )
        )


    def test_stable_row_ids_use_identity_and_fail_on_duplicates(self) -> None:
        frame = pd.DataFrame(
            {
                "policy": ["A", "B"],
                "anchor": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            }
        )
        first = stable_row_ids(frame, ["policy", "anchor"])
        second = stable_row_ids(
            frame.sample(frac=1.0, random_state=17), ["policy", "anchor"]
        )
        self.assertEqual(first.sort_index().tolist(), second.sort_index().tolist())
        duplicate = pd.concat([frame.iloc[[0]], frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "do not uniquely identify"):
            stable_row_ids(duplicate, ["policy", "anchor"])


    def test_family_selection_prefers_logistic_when_within_one_se(self) -> None:
        rows = []
        for fold, logistic_ap, xgb_ap, logistic_recall, xgb_recall in (
            (1, 0.120, 0.122, 0.31, 0.32),
            (2, 0.125, 0.129, 0.32, 0.34),
            (3, 0.118, 0.120, 0.30, 0.31),
            (4, 0.130, 0.127, 0.34, 0.32),
            (5, 0.122, 0.119, 0.31, 0.30),
        ):
            rows.extend(
                [
                    {
                        "fold": fold,
                        "model_family": "logistic",
                        "churn_ap_skill": logistic_ap,
                        "churn_recall_10pct": logistic_recall,
                    },
                    {
                        "fold": fold,
                        "model_family": "xgboost",
                        "churn_ap_skill": xgb_ap,
                        "churn_recall_10pct": xgb_recall,
                    },
                ]
            )
        selected, comparison, decision = choose_model_family(pd.DataFrame(rows))
        self.assertEqual(selected, "logistic")
        self.assertTrue(
            comparison.loc[
                comparison["model_family"].eq("logistic"), "selected"
            ].item()
        )
        self.assertFalse(decision["final_holdout_used"])


    def test_family_selection_chooses_materially_better_xgboost(self) -> None:
        rows = []
        for fold in range(1, 6):
            rows.extend(
                [
                    {
                        "fold": fold,
                        "model_family": "logistic",
                        "churn_ap_skill": 0.10,
                        "churn_recall_10pct": 0.25,
                    },
                    {
                        "fold": fold,
                        "model_family": "xgboost",
                        "churn_ap_skill": 0.16,
                        "churn_recall_10pct": 0.35,
                    },
                ]
            )
        selected, _, _ = choose_model_family(pd.DataFrame(rows))
        self.assertEqual(selected, "xgboost")


    def test_family_selection_rejects_nonfinite_or_nonfive_fold_evidence(self) -> None:
        rows = []
        for fold in range(1, 6):
            for family in ("logistic", "xgboost"):
                rows.append(
                    {
                        "fold": fold,
                        "model_family": family,
                        "churn_ap_skill": 0.1,
                        "churn_recall_10pct": 0.2,
                    }
                )
        frame = pd.DataFrame(rows)
        with self.assertRaisesRegex(ValueError, "finite"):
            choose_model_family(frame.assign(churn_recall_10pct=np.nan))
        with self.assertRaisesRegex(ValueError, "exactly five"):
            choose_model_family(frame.loc[frame["fold"].lt(5)])




if __name__ == "__main__":
    unittest.main()
