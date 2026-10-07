from __future__ import annotations


import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


import numpy as np
import pandas as pd


from c_modeling.model_diagnostics import (
    ACTUAL_COLUMN,
    WITH_YEAR_COLUMN,
    YEAR_CONTROL_FEATURE,
    YEAR_MONTH_COLUMN,
    aggregate_raw_shap,
    business_retention_monthly_summary,
    calendar_control_performance,
    one_way_summaries,
    prediction_metrics,
    write_business_retention_trend_artifacts,
    write_calendar_control_artifacts,
    write_xgboost_shap_artifacts,
)
from c_modeling.final_refit_diagnostics import (
    write_final_refit_descriptive_trend_artifacts,
    write_full_history_model_comparison_artifacts,
)




def diagnostic_frame(rows: int = 40) -> pd.DataFrame:
    actual = np.tile([0, 1], rows // 2)
    month = pd.period_range("2020-01", periods=rows, freq="M")
    return pd.DataFrame(
        {
            "fold": np.repeat([1, 2], rows // 2),
            "row_id": [f"row-{position:03d}" for position in range(rows)],
            ACTUAL_COLUMN: actual,
            WITH_YEAR_COLUMN: np.where(actual == 1, 0.8, 0.25),
            YEAR_CONTROL_FEATURE: month.year,
            YEAR_MONTH_COLUMN: month.astype(str),
            "FEATURE_A": np.arange(rows),
        }
    )




def _check_recall_is_exact_capture_alias() -> None:
    metrics = prediction_metrics(diagnostic_frame(), WITH_YEAR_COLUMN)
    assert metrics["churn_recall_10pct"] == metrics["churn_capture_10pct"]




def _check_calendar_control_performance_contains_pooled_and_fold_rows() -> None:
    performance = calendar_control_performance(diagnostic_frame())
    assert set(performance["scope"]) == {"POOLED_OOF", "FOLD"}
    assert "variant" not in performance




def _check_business_trends_use_dynamic_history_and_policy_weighting(
    tmp_path: Path,
) -> None:
    records: list[dict[str, object]] = []
    months = pd.period_range("2017-01", periods=73, freq="M")
    for position, month in enumerate(months):
        policies = position % 4 + 1
        retained = max(0, policies - (position % 2))
        outcomes = [1] * retained + [0] * (policies - retained)
        records.extend(
            {
                "MONTH_1_EVALUATION_DATE": month.start_time,
                "RETENTION_IND": (
                    np.nan if position == len(months) - 1 else outcome
                ),
            }
            for outcome in outcomes
        )
    monthly = business_retention_monthly_summary(
        population=pd.DataFrame(records),
        target_col="RETENTION_IND",
        cohort_col="MONTH_1_EVALUATION_DATE",
        end_month_exclusive="2023-01-01",
    )
    assert len(monthly) == 72
    assert monthly[YEAR_MONTH_COLUMN].iloc[-1] == "2022-12"
    weighted_three_month = (
        monthly["retained_policies"].tail(3).sum()
        / monthly["policies"].tail(3).sum()
    )
    assert np.isclose(
        monthly["actual_retention_3m_weighted"].iloc[-1],
        weighted_three_month,
    )
    assert not np.isclose(
        weighted_three_month,
        monthly["actual_retention"].tail(3).mean(),
    )


    recent_manifest = write_business_retention_trend_artifacts(
        monthly=monthly,
        output_dir=tmp_path,
        model_family="logistic",
        cohort_col="MONTH_1_EVALUATION_DATE",
        selected_history={
            "history_id": "RECENT_24M",
            "kind": "recent",
            "months": 24,
        },
    )
    recent = pd.read_csv(
        tmp_path / "logistic_business_retention_trend_selected_history.csv"
    )
    assert len(recent) == 24
    assert np.isclose(
        recent["actual_retention_12m_weighted"].iloc[0],
        monthly["actual_retention_12m_weighted"].iloc[-24],
    )
    assert not any("predicted" in column for column in recent.columns)
    assert recent_manifest["sealed_holdout_scored"] is False
    assert recent_manifest["predictions_included"] is False
    assert recent_manifest["selected_history_window_months_requested"] == 24
    assert (tmp_path / "logistic_business_retention_trend_full_history.png").is_file()
    assert (
        tmp_path / "logistic_business_retention_trend_selected_history.png"
    ).is_file()


    decay_manifest = write_business_retention_trend_artifacts(
        monthly=monthly,
        output_dir=tmp_path,
        model_family="logistic_decay",
        cohort_col="MONTH_1_EVALUATION_DATE",
        selected_history={
            "history_id": "DECAY_HALF_LIFE_12M_CAPPED_60M",
            "kind": "decay",
            "months": 12,
            "max_months": 60,
        },
    )
    decay = pd.read_csv(
        tmp_path / "logistic_decay_business_retention_trend_selected_history.csv"
    )
    assert len(decay) == 60
    assert decay_manifest["selected_history_window_months_requested"] == 60




def _final_refit_monthly_frame() -> pd.DataFrame:
    months = pd.period_range("2018-01", periods=72, freq="M")
    position = np.arange(len(months))
    actual = 0.80 + 0.015 * np.sin(position / 6.0)
    predicted = 0.805 + 0.012 * np.sin((position + 1) / 6.0)
    return pd.DataFrame(
        {
            YEAR_MONTH_COLUMN: months.astype(str),
            "policies": 1000 + position,
            "retained_policies": np.rint(actual * (1000 + position)).astype(int),
            "actual_retention": actual,
            "predicted_retention": predicted,
            "calibration_gap": predicted - actual,
        }
    )




def _check_final_refit_descriptive_trends_are_separate_and_labeled(
    tmp_path: Path,
) -> None:
    scored = _final_refit_monthly_frame()
    actual_only = scored.copy()
    for window in (3, 12):
        rolling_policies = actual_only["policies"].rolling(
            window, min_periods=window
        ).sum()
        rolling_retained = actual_only["retained_policies"].rolling(
            window, min_periods=window
        ).sum()
        actual_only[f"actual_retention_{window}m_weighted"] = (
            rolling_retained / rolling_policies
        )
    write_business_retention_trend_artifacts(
        monthly=actual_only,
        output_dir=tmp_path,
        model_family="logistic",
        cohort_col="MONTH_1_EVALUATION_DATE",
        selected_history={
            "history_id": "RECENT_36M",
            "kind": "recent",
            "months": 36,
        },
    )
    actual_artifacts = {
        path.name: path.read_bytes()
        for path in tmp_path.glob("logistic_business_retention_trend_*")
    }


    manifest = write_final_refit_descriptive_trend_artifacts(
        monthly=scored,
        output_dir=tmp_path,
        model_family="logistic",
        cohort_col="MONTH_1_EVALUATION_DATE",
        selected_history={
            "history_id": "RECENT_36M",
            "kind": "recent",
            "months": 36,
        },
        training_start="2020-01",
        training_end="2022-12",
    )
    prefix = "logistic_final_refit_actual_vs_predicted"
    full = pd.read_csv(tmp_path / f"{prefix}_full_history.csv")
    selected = pd.read_csv(
        tmp_path / f"{prefix}_selected_training_history.csv"
    )
    assert len(full) == 72
    assert len(selected) == 36
    assert selected[YEAR_MONTH_COLUMN].iloc[0] == "2020-01"
    assert selected[YEAR_MONTH_COLUMN].iloc[-1] == "2022-12"
    assert set(selected["prediction_role"]) == {"FINAL_REFIT_TRAINING_WINDOW"}
    assert set(full["prediction_role"]) == {
        "BACKCAST_PRE_TRAINING",
        "FINAL_REFIT_TRAINING_WINDOW",
        "POST_TRAINING_PREHOLDOUT_DIAGNOSTIC",
    }
    assert {"actual_retention", "predicted_retention", "calibration_gap"}.issubset(
        full.columns
    )
    assert (tmp_path / f"{prefix}_full_history.png").is_file()
    assert (tmp_path / f"{prefix}_selected_training_history.png").is_file()
    manifest_path = tmp_path / "final_refit_actual_vs_predicted_manifest.json"
    assert manifest_path.is_file()
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest"] == manifest_path.name
    assert saved["evaluation_scope"] == "FINAL_REFIT_DESCRIPTIVE_NOT_OOF"
    assert saved["performance_evidence"] is False
    assert saved["sealed_holdout_scored"] is False
    assert saved["final_holdout_rows_scored"] == 0
    assert saved["plot_warning"] == "NOT OOF PERFORMANCE"
    assert saved["training_start"] == "2020-01"
    assert saved["training_end"] == "2022-12"
    assert saved["training_months"] == 36
    assert saved["full_history_end"] == "2023-12"
    assert set(saved["artifacts"]) == {
        f"{prefix}_full_history.csv",
        f"{prefix}_full_history.png",
        f"{prefix}_selected_training_history.csv",
        f"{prefix}_selected_training_history.png",
    }
    for name, contents in actual_artifacts.items():
        assert (tmp_path / name).read_bytes() == contents




def _check_xgboost_uses_official_shap_package() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "c_modeling"
        / "xgboost_feature_selection.py"
    ).read_text(encoding="utf-8")
    assert "shap.TreeExplainer" in source
    assert "pred_contribs=True" not in source




def _check_calendar_artifacts_and_xgboost_shap_outputs(
    tmp_path: Path,
    module_registry: dict[str, object],
) -> None:
    frame = diagnostic_frame()
    manifest = write_calendar_control_artifacts(
        predictions=frame,
        output_dir=tmp_path,
        model_family="logistic",
        one_way_features=["FEATURE_A"],
    )
    assert manifest["sealed_holdout_scored"] is False
    saved = json.loads((tmp_path / "logistic_diagnostics_manifest.json").read_text())
    assert saved["metric_hierarchy"]["co_primary_business_metric"] == (
        "churn_recall_10pct"
    )
    assert (tmp_path / "logistic_policy_start_year_month_trend.png").exists()
    assert (tmp_path / "logistic_calendar_year_control_one_way.png").exists()
    one_way_plots = sorted(tmp_path.glob("logistic_one_way_*.png"))
    assert len(one_way_plots) == 1
    assert "feature_a" in one_way_plots[0].name
    assert saved["one_way_plot_features"] == ["FEATURE_A"]
    trend = pd.read_csv(tmp_path / "logistic_policy_start_year_month_trend.csv")
    assert "predicted_retention" in trend
    assert "predicted_retention_3m" in trend
    assert not any("without" in column for column in trend.columns)
    year_one_way = pd.read_csv(
        tmp_path / "logistic_calendar_year_one_way_summary.csv"
    )
    feature_one_way = pd.read_csv(
        tmp_path / "logistic_selected_feature_one_way_summaries.csv"
    )
    assert set(year_one_way[YEAR_CONTROL_FEATURE]) == set(frame[YEAR_CONTROL_FEATURE])
    assert {YEAR_CONTROL_FEATURE, "FEATURE_A"}.issubset(
        set(feature_one_way["feature"])
    )
    feature_levels = feature_one_way.loc[
        feature_one_way["feature"].eq("FEATURE_A"), "level"
    ].astype(str).tolist()
    assert feature_levels[0].startswith("Q01:")
    assert feature_levels[-1].startswith("Q10:")
    assert not any("without" in column for column in feature_one_way.columns)


    transformed = np.arange(24, dtype=float).reshape(4, 6)
    raw, features = aggregate_raw_shap(
        transformed,
        [
            {
                "raw_feature": YEAR_CONTROL_FEATURE,
                "start_column": 0,
                "end_column_exclusive": 1,
            },
            {
                "raw_feature": "FEATURE_A",
                "start_column": 1,
                "end_column_exclusive": 6,
            },
        ],
    )
    assert features == [YEAR_CONTROL_FEATURE, "FEATURE_A"]
    assert np.allclose(raw[:, 1], transformed[:, 1:].sum(axis=1))


    class FakeExplanation:
        def __init__(self, *, values, data, display_data, feature_names):
            self.values = values
            self.data = data
            self.display_data = display_data
            self.feature_names = list(feature_names)


        def __getitem__(self, key):
            rows, feature = key
            position = self.feature_names.index(feature)
            return FakeExplanation(
                values=self.values[rows, position],
                data=self.data[rows, position],
                display_data=self.display_data[rows, position],
                feature_names=[feature],
            )


    plot_calls = []


    def fake_beeswarm(explanation, **kwargs):
        plot_calls.append("beeswarm")
        import matplotlib.pyplot as plt


        plt.figure(figsize=(5, 3))
        plt.scatter(explanation.values[:, 0], np.arange(len(explanation.values)))


    def fake_scatter(explanation, *, ax, **kwargs):
        plot_calls.append("scatter")
        ax.scatter(explanation.data, explanation.values)


    module_registry["shap"] = types.SimpleNamespace(
        Explanation=FakeExplanation,
        plots=types.SimpleNamespace(
            beeswarm=fake_beeswarm,
            scatter=fake_scatter,
        ),
    )
    top = write_xgboost_shap_artifacts(
        raw_shap=raw,
        feature_values=pd.DataFrame(
            {
                YEAR_CONTROL_FEATURE: [2020, 2021, 2022, 2023],
                "FEATURE_A": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        features=features,
        output_dir=tmp_path,
    )
    assert top
    assert (tmp_path / "xgboost_shap_beeswarm.png").exists()
    assert not (tmp_path / "xgboost_shap_scatter.png").exists()
    scatter_plots = sorted(tmp_path.glob("xgboost_shap_scatter_*.png"))
    assert len(scatter_plots) == 1
    assert "feature_a" in scatter_plots[0].name
    shap_manifest = json.loads(
        (tmp_path / "xgboost_shap_plot_manifest.json").read_text()
    )
    assert shap_manifest["scatter_features"] == ["FEATURE_A"]
    assert shap_manifest["scatter_control_feature_excluded"] == (
        YEAR_CONTROL_FEATURE
    )
    assert "beeswarm" in plot_calls
    assert "scatter" in plot_calls




class ModelDiagnosticsTests(unittest.TestCase):
    """Exercise diagnostic contracts without pytest-only fixture injection."""


    def test_full_history_comparison_weights_counts_and_keeps_descriptive_label(self) -> None:
        monthly = pd.DataFrame([
            ["2021-01", "logistic", "TRAINING", 1, 1, 0.9],
            ["2021-01", "logistic", "BACKCAST", 3, 1, 1.5],
            ["2021-01", "xgboost", "TRAINING", 3, 2, 2.1],
            ["2021-01", "xgboost", "BACKCAST", 1, 0, 0.2],
        ], columns=["YEAR_MONTH", "model_family", "population_role", "policies",
                    "retained_policies", "predicted_retained_policies"])
        original = monthly.copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = write_full_history_model_comparison_artifacts(
                monthly=monthly, output_dir=output, cohort_col="COHORT",
                final_holdout_start="2021-02-01",
            )
            saved = pd.read_csv(output / "model_retention_trend_full_history.csv")
            self.assertAlmostEqual(saved.loc[0, "actual_retention"], 0.5)
            self.assertAlmostEqual(saved.loc[0, "logistic_predicted_retention"], 0.6)
            self.assertAlmostEqual(saved.loc[0, "xgboost_predicted_retention"], 0.575)
            self.assertFalse(manifest["performance_evidence"])
            self.assertEqual(manifest["final_holdout_rows_scored"], 0)
            for name in [*manifest["artifacts"], manifest["manifest"]]:
                self.assertGreater((output / name).stat().st_size, 0)
            with self.assertRaisesRegex(ValueError, "exclude all final-holdout months"):
                write_full_history_model_comparison_artifacts(
                    monthly=monthly, output_dir=output / "invalid", cohort_col="COHORT",
                    final_holdout_start="2021-01-01",
                )
            mismatched = monthly.copy()
            mismatched.loc[0, "policies"] = 2
            with self.assertRaisesRegex(ValueError, "identical monthly row"):
                write_full_history_model_comparison_artifacts(
                    monthly=mismatched, output_dir=output / "invalid", cohort_col="COHORT",
                    final_holdout_start="2021-02-01",
                )
        pd.testing.assert_frame_equal(monthly, original)


    def test_recall_is_exact_capture_alias(self) -> None:
        _check_recall_is_exact_capture_alias()


    def test_calendar_control_performance_contains_pooled_and_fold_rows(
        self,
    ) -> None:
        _check_calendar_control_performance_contains_pooled_and_fold_rows()


    def test_business_trends_use_dynamic_history_and_policy_weighting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _check_business_trends_use_dynamic_history_and_policy_weighting(
                Path(temporary)
            )


    def test_final_refit_descriptive_trends_are_separate_and_labeled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _check_final_refit_descriptive_trends_are_separate_and_labeled(
                Path(temporary)
            )


    def test_final_refit_descriptive_trends_reject_invalid_inputs(self) -> None:
        history = {
            "history_id": "RECENT_36M",
            "kind": "recent",
            "months": 36,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            missing_prediction = _final_refit_monthly_frame().drop(
                columns="predicted_retention"
            )
            with self.assertRaisesRegex(KeyError, "predicted_retention"):
                write_final_refit_descriptive_trend_artifacts(
                    monthly=missing_prediction,
                    output_dir=output,
                    model_family="logistic",
                    cohort_col="MONTH_1_EVALUATION_DATE",
                    selected_history=history,
                    training_start="2020-01",
                    training_end="2022-12",
                )


            invalid_probability = _final_refit_monthly_frame()
            invalid_probability.loc[0, "predicted_retention"] = np.inf
            with self.assertRaisesRegex(ValueError, "finite probabilities"):
                write_final_refit_descriptive_trend_artifacts(
                    monthly=invalid_probability,
                    output_dir=output,
                    model_family="logistic",
                    cohort_col="MONTH_1_EVALUATION_DATE",
                    selected_history=history,
                    training_start="2020-01",
                    training_end="2022-12",
                )


            with self.assertRaisesRegex(ValueError, "do not match selected history"):
                write_final_refit_descriptive_trend_artifacts(
                    monthly=_final_refit_monthly_frame(),
                    output_dir=output,
                    model_family="logistic",
                    cohort_col="MONTH_1_EVALUATION_DATE",
                    selected_history=history,
                    training_start="2020-02",
                    training_end="2022-12",
                )


    def test_xgboost_uses_official_shap_package(self) -> None:
        _check_xgboost_uses_official_shap_package()


    def test_calendar_artifacts_and_xgboost_shap_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            sys.modules, {}, clear=False
        ):
            _check_calendar_artifacts_and_xgboost_shap_outputs(
                Path(temporary), sys.modules
            )


    def test_continuous_axes_keep_numeric_strings_and_preserve_categories(self) -> None:
        from c_modeling.model_diagnostics import _plot_one_way_summary
        import matplotlib.pyplot as plt


        frame = diagnostic_frame(40)
        frame["WRITTEN_PREM"] = ["-5562", "bad", None, "inf"] + [str(200 + i * 137) for i in range(36)]
        frame["CATEGORY_CODE"] = [str(i % 20) for i in range(40)]
        original = frame.copy(deep=True)
        table, audit = one_way_summaries(
            frame, ["WRITTEN_PREM", "CATEGORY_CODE"],
            feature_types={"WRITTEN_PREM": "continuous", "CATEGORY_CODE": "discrete"},
        )
        pd.testing.assert_frame_equal(frame, original)
        premium = table.loc[table.feature.eq("WRITTEN_PREM")]
        counts = audit.set_index("feature").loc["WRITTEN_PREM"]
        self.assertEqual(counts["negative_rows_hidden_from_plot"], 1)
        self.assertEqual(counts["nonfinite_rows_hidden_from_plot"], 3)
        self.assertEqual(counts["plot_rows"], 36)
        self.assertEqual(premium.rows.sum(), 40)
        self.assertTrue(table.loc[table.feature.eq("CATEGORY_CODE"), "axis_kind"].eq("categorical").all())
        self.assertFalse(table.loc[table.feature.eq("CATEGORY_CODE"), "level"].str.startswith("Q").any())
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(plt, "close"):
            _plot_one_way_summary(premium, Path(temporary) / "premium.png", "Premium")
            figure = plt.gcf()
            figure.canvas.draw()
            axis = figure.axes[1]
            self.assertEqual(axis.get_xlim()[0], 0.0)
            self.assertLessEqual(len(axis.get_xticks()), 10)
            xdata = figure.axes[0].lines[0].get_xdata()
            self.assertTrue(np.all(xdata >= 0))
            self.assertGreater(xdata.max(), 4000)
        plt.close(figure)


    def test_shap_numeric_display_filters_only_requested_negative_premium(self) -> None:
        from c_modeling import model_diagnostics as diagnostics
        import matplotlib.pyplot as plt


        class Explanation:
            def __init__(self, *, values, data, display_data, feature_names):
                self.values, self.data, self.display_data = values, data, display_data
                self.feature_names = list(feature_names)


            def __getitem__(self, key):
                rows, name = key
                column = self.feature_names.index(name)
                return Explanation(values=self.values[rows, column], data=self.data[rows, column],
                                   display_data=self.display_data[rows, column], feature_names=[name])


        captured = {}


        def beeswarm(explanation, **_kwargs):
            plt.figure()


        def scatter(explanation, *, ax, **_kwargs):
            captured[explanation.feature_names[0]] = (explanation, ax)
            ax.scatter(explanation.data, explanation.values)


        stub = types.SimpleNamespace(Explanation=Explanation, plots=types.SimpleNamespace(beeswarm=beeswarm, scatter=scatter))
        count = 1000
        frame = pd.DataFrame({
            "WRITTEN_PREM": ["-5562", "nan", "inf", "-inf"] + [str(i * 13.75) for i in range(count - 4)],
            "COV_A_DWELLING_CVG_AMT": np.linspace(-100, 5_000_000, count),
            "CATEGORY_CODE": [str(i % 15) for i in range(count)],
        })
        original = frame.copy(deep=True)
        values = np.column_stack([np.linspace(-.5, .8, count), np.linspace(.3, -.5, count), np.linspace(-.1, .1, count)])
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(diagnostics, "_shap_package", return_value=stub):
            write_xgboost_shap_artifacts(raw_shap=values, feature_values=frame,
                                        features=list(frame), output_dir=temporary,
                                        feature_types={"WRITTEN_PREM": "continuous", "CATEGORY_CODE": "discrete"})
            manifest = json.loads((Path(temporary) / "xgboost_shap_plot_manifest.json").read_text())
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(set(captured), {"WRITTEN_PREM", "COV_A_DWELLING_CVG_AMT"})
        explanation, axis = captured["WRITTEN_PREM"]
        self.assertEqual(explanation.display_data.dtype.kind, "f")
        self.assertTrue(np.isfinite(explanation.data).all())
        self.assertTrue((explanation.data >= 0).all())
        self.assertEqual(axis.get_xlim()[0], 0)
        self.assertLessEqual(len(axis.get_xticks()), 10)
        self.assertEqual(len(explanation.data), count - 4)
        self.assertLess(captured["COV_A_DWELLING_CVG_AMT"][0].data.min(), 0)
        audit = next(item for item in manifest["scatter_display_audit"] if item["feature"] == "WRITTEN_PREM")
        self.assertEqual(audit["negative_rows_hidden_from_plot"], 1)
        self.assertEqual(audit["total_rows_hidden_from_plot"], 4)
        self.assertFalse(audit["raw_values_modified"])
        self.assertFalse(audit["positive_values_capped"])


    def test_numeric_inference_respects_dtype_and_empty_features(self) -> None:
        from c_modeling.model_diagnostics import _is_continuous_plot_feature
        self.assertTrue(_is_continuous_plot_feature(pd.Series(["1.0", "2.5", None])))
        self.assertFalse(_is_continuous_plot_feature(pd.Series(["1", "2"]), "discrete"))
        self.assertFalse(_is_continuous_plot_feature(pd.Series(pd.Categorical([1, 2]))))
        self.assertFalse(_is_continuous_plot_feature(pd.Series([True, False])))
        self.assertFalse(_is_continuous_plot_feature(pd.Series(["A", "2"])))
        self.assertFalse(_is_continuous_plot_feature(pd.Series([None, None])))
        self.assertTrue(_is_continuous_plot_feature(pd.Series([None, None]), "continuous"))




if __name__ == "__main__":
    unittest.main()
