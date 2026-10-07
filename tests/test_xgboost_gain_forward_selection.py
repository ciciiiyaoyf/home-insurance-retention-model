from __future__ import annotations


import ast
from pathlib import Path
from typing import Mapping, Sequence
import unittest


import numpy as np
import pandas as pd




PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = PROJECT_ROOT / "c_modeling" / "xgboost_feature_selection.py"
RUNNER_PATH = PROJECT_ROOT / "c_modeling" / "c3_run_xgboost.py"




def _load_pure_helpers() -> dict[str, object]:
    """Load pure selection helpers without importing the XGBoost runtime."""


    source = ENGINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {
        "aggregate_gain_ranking",
        "cumulative_forward_feature_counts",
        "build_cumulative_forward_groups",
        "build_local_forward_refinement_groups",
        "choose_performance_group",
        "_metric_column",
        "pareto_relevant_group_ids",
        "choose_confirmation_group_ids",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    if {node.name for node in nodes} != wanted:
        raise AssertionError("Expected gain-forward helpers are missing.")
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace: dict[str, object] = {
        "np": np,
        "pd": pd,
        "Mapping": Mapping,
        "Sequence": Sequence,
    }
    exec(compile(module, str(ENGINE_PATH), "exec"), namespace)
    return namespace




class XGBoostGainForwardSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.helpers = _load_pure_helpers()


    def test_grouped_gain_ranking_is_deterministic(self) -> None:
        gain = pd.DataFrame(
            [
                ("B", 1, 2.0, 0.20, 3.0),
                ("A", 2, 4.0, 0.40, 5.0),
                ("C", 1, 0.0, 0.00, 0.0),
                ("B", 1, 1.0, 0.10, 2.0),
                ("A", 2, 3.0, 0.30, 4.0),
                ("C", 1, 0.0, 0.00, 0.0),
            ],
            columns=[
                "raw_feature",
                "transformed_column_count",
                "total_gain",
                "normalized_total_gain",
                "split_count",
            ],
        )


        ranking = self.helpers["aggregate_gain_ranking"](gain)


        self.assertEqual(ranking["raw_feature"].tolist(), ["A", "B", "C"])
        self.assertEqual(ranking["gain_rank"].tolist(), [1, 2, 3])


    def test_forward_schedule_is_nested_and_includes_full_pool(self) -> None:
        counts = self.helpers["cumulative_forward_feature_counts"](
            23,
            min_features=8,
            step_size=8,
        )
        self.assertEqual(counts, [8, 16, 23])


        ranking = pd.DataFrame(
            {"raw_feature": [f"F{position:02d}" for position in range(1, 24)]}
        )
        groups = self.helpers["build_cumulative_forward_groups"](
            ranking,
            min_features=8,
            step_size=8,
        )
        feature_sets = list(groups.values())
        self.assertEqual([len(values) for values in feature_sets], counts)
        self.assertTrue(set(feature_sets[0]).issubset(feature_sets[1]))
        self.assertTrue(set(feature_sets[1]).issubset(feature_sets[2]))
        self.assertEqual(feature_sets[-1], ranking["raw_feature"].tolist())


    def test_smallest_tolerance_qualified_group_wins(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "group_id": "N048",
                    "raw_feature_count": 48,
                    "mean_churn_ap_skill": 0.09663513334276037,
                    "mean_top10_lift": 1.8846309964007701,
                    "mean_retention_roc_auc": 0.6302660902417176,
                    "newest_fold_ap_skill": 0.09422809178416416,
                },
                {
                    "group_id": "N080",
                    "raw_feature_count": 80,
                    "mean_churn_ap_skill": 0.09730353499272465,
                    "mean_top10_lift": 1.8853856659843655,
                    "mean_retention_roc_auc": 0.6311312616233516,
                    "newest_fold_ap_skill": 0.09508840455180402,
                },
            ]
        )


        selected, annotated, reason = self.helpers["choose_performance_group"](
            summary,
            id_col="group_id",
            ap_skill_tolerance=0.001,
            lift_tolerance=0.005,
            auc_tolerance=0.001,
            recent_ap_skill_tolerance=0.003,
        )


        self.assertEqual(selected, "N048")
        self.assertEqual(int(annotated["selected_by_rule"].sum()), 1)
        self.assertIn("smallest", reason.lower())


    def test_full_pool_can_win_without_a_hard_feature_cap(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "group_id": "N048",
                    "raw_feature_count": 48,
                    "mean_churn_ap_skill": 0.080,
                    "mean_top10_lift": 1.70,
                    "mean_retention_roc_auc": 0.60,
                    "newest_fold_ap_skill": 0.078,
                },
                {
                    "group_id": "N091",
                    "raw_feature_count": 91,
                    "mean_churn_ap_skill": 0.100,
                    "mean_top10_lift": 1.90,
                    "mean_retention_roc_auc": 0.64,
                    "newest_fold_ap_skill": 0.099,
                },
            ]
        )


        selected, _, _ = self.helpers["choose_performance_group"](
            summary,
            id_col="group_id",
            ap_skill_tolerance=0.001,
            lift_tolerance=0.005,
            auc_tolerance=0.001,
            recent_ap_skill_tolerance=0.003,
        )


        self.assertEqual(selected, "N091")


    def test_step_four_refinement_fills_the_selected_count_gap(self) -> None:
        ranking = pd.DataFrame(
            {"raw_feature": [f"F{position:03d}" for position in range(1, 92)]}
        )
        groups = self.helpers["build_cumulative_forward_groups"](
            ranking,
            min_features=8,
            step_size=4,
        )
        self.assertEqual(
            [len(features) for features in groups.values()],
            list(range(8, 89, 4)) + [91],
        )


        refinement = self.helpers["build_local_forward_refinement_groups"](
            ranking,
            groups,
            selected_group_id="XGB_GAIN_FORWARD_N048",
            min_features=8,
        )
        self.assertEqual(
            [len(features) for features in refinement.values()],
            [45, 46, 47],
        )


    def test_refinement_supports_selection_rebracketing(self) -> None:
        ranking = pd.DataFrame(
            {"raw_feature": [f"F{position:03d}" for position in range(1, 24)]}
        )
        groups = self.helpers["build_cumulative_forward_groups"](
            ranking,
            min_features=8,
            step_size=4,
        )


        first = self.helpers["build_local_forward_refinement_groups"](
            ranking,
            groups,
            selected_group_id="XGB_GAIN_FORWARD_N016",
            min_features=8,
        )
        self.assertEqual([len(features) for features in first.values()], [13, 14, 15])
        groups.update(first)


        second = self.helpers["build_local_forward_refinement_groups"](
            ranking,
            groups,
            selected_group_id="XGB_GAIN_FORWARD_N020",
            min_features=8,
        )
        self.assertEqual([len(features) for features in second.values()], [17, 18, 19])
        groups.update(second)


        complete = self.helpers["build_local_forward_refinement_groups"](
            ranking,
            groups,
            selected_group_id="XGB_GAIN_FORWARD_N018",
            min_features=8,
        )
        self.assertEqual(complete, {})


    def test_confirmation_queue_keeps_required_groups_beyond_one_batch(self) -> None:
        ranking = pd.DataFrame(
            {"raw_feature": [f"F{position:03d}" for position in range(1, 92)]}
        )
        groups = self.helpers["build_cumulative_forward_groups"](
            ranking,
            min_features=8,
            step_size=4,
        )
        summary = pd.DataFrame(
            [
                {
                    "group_id": "XGB_GAIN_FORWARD_N048",
                    "raw_feature_count": 48,
                    "mean_churn_ap_skill": 0.0966,
                    "mean_churn_recall_10pct": 0.18846,
                    "mean_top10_lift": 1.8846,
                    "mean_retention_roc_auc": 0.6303,
                    "newest_fold_ap_skill": 0.0942,
                    "eligible_performance_group": True,
                },
                {
                    "group_id": "XGB_GAIN_FORWARD_N072",
                    "raw_feature_count": 72,
                    "mean_churn_ap_skill": 0.0974,
                    "mean_churn_recall_10pct": 0.18760,
                    "mean_top10_lift": 1.8760,
                    "mean_retention_roc_auc": 0.6314,
                    "newest_fold_ap_skill": 0.0948,
                    "eligible_performance_group": False,
                },
                {
                    "group_id": "XGB_GAIN_FORWARD_N080",
                    "raw_feature_count": 80,
                    "mean_churn_ap_skill": 0.0973,
                    "mean_churn_recall_10pct": 0.18854,
                    "mean_top10_lift": 1.8854,
                    "mean_retention_roc_auc": 0.6311,
                    "newest_fold_ap_skill": 0.0951,
                    "eligible_performance_group": True,
                },
            ]
        )


        selected = self.helpers["choose_confirmation_group_ids"](
            forward_summary=summary,
            forward_annotated=summary,
            groups_by_id=groups,
            initial_selected_id="XGB_GAIN_FORWARD_N048",
            confirm_candidate_count=4,
        )


        self.assertEqual(selected[0], "XGB_GAIN_FORWARD_N048")
        self.assertIn("XGB_GAIN_FORWARD_N072", selected)
        self.assertEqual(len(selected), len(set(selected)))
        # The execution budget limits one confirmation pass, not the complete
        # queue. Keep eligible prefixes, metric anchors, full pool and neighbor.
        self.assertEqual(
            set(selected),
            {
                "XGB_GAIN_FORWARD_N048",
                "XGB_GAIN_FORWARD_N072",
                "XGB_GAIN_FORWARD_N080",
                "XGB_GAIN_FORWARD_N088",
                "XGB_GAIN_FORWARD_N091",
            },
        )
        self.assertGreater(len(selected), 4)


    def test_main_executes_forward_path_not_rfe(self) -> None:
        source = ENGINE_PATH.read_text(encoding="utf-8")
        main_source = source[source.index("def main()") :]


        self.assertIn("evaluate_full_pool_gain_ranking(", main_source)
        self.assertIn("build_cumulative_forward_groups(", main_source)
        self.assertIn("build_local_forward_refinement_groups(", main_source)
        self.assertIn("choose_confirmation_group_ids(", main_source)
        self.assertIn('stage_label="FORWARD_REFINE"', main_source)
        self.assertIn("xgb_cumulative_forward_summary.csv", main_source)
        self.assertIn(
            'parser.add_argument("--forward-step-size", type=int, default=4)',
            source,
        )
        self.assertIn(
            'os.environ.get("RETENTION_XGB_FORWARD_STEP_SIZE", "4")',
            RUNNER_PATH.read_text(encoding="utf-8"),
        )
        self.assertNotIn("evaluate_rfe_round(", main_source)
        self.assertNotIn("args.rfe_", main_source)
        self.assertNotIn("def evaluate_rfe_round", source)
        self.assertNotIn("def next_feature_set", source)
        self.assertNotIn("def feature_block_permutation", source)




if __name__ == "__main__":
    unittest.main()
