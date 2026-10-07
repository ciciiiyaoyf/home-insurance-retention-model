"""Fast contract tests for history, outer evaluation, and cross-tenure guards.


These tests intentionally use :mod:`unittest` only.  They can therefore run
in the project environment even when pytest is not installed::


    python -m unittest discover -s tests \
        -p "test_outer_evaluation_contracts.py" -v
"""


from __future__ import annotations


from contextlib import contextmanager, redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


import numpy as np
import pandas as pd




PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from c_modeling.c4_evaluate_and_finalize import (  # noqa: E402
    choose_model_family,
    final_evaluation_training_indices,
    validated_history_months,
)
from c_modeling.modeling_utils import (  # noqa: E402
    select_recent_mature_training_indices,
)
from c_modeling.performance_plots import (  # noqa: E402
    generate_performance_plots,
    validate_prediction_frame,
)
from d_cross_tenure import cross_tenure_evaluation  # noqa: E402
from d_cross_tenure import d0_cross_tenure_one_click  # noqa: E402




class MatureHistoryContractTests(unittest.TestCase):
    def setUp(self) -> None:
        months = pd.period_range("2017-01", periods=72, freq="M")
        self.frame = pd.DataFrame(
            {
                "MONTH_1_EVALUATION_DATE": np.repeat(
                    months.to_timestamp(how="start"), 2
                )
            },
            index=np.arange(1000, 1000 + 2 * len(months)),
        )


    def test_recent_history_is_inclusive_and_exact_for_every_governed_option(self) -> None:
        train_idx = self.frame.index.to_numpy()
        newest = pd.Period("2022-12", freq="M")


        for requested_months in (36, 48, 60):
            with self.subTest(months=requested_months):
                selected, audit = select_recent_mature_training_indices(
                    self.frame,
                    train_idx,
                    months=requested_months,
                )
                selected_periods = pd.to_datetime(
                    self.frame.loc[selected, "MONTH_1_EVALUATION_DATE"]
                ).dt.to_period("M")


                self.assertEqual(selected_periods.nunique(), requested_months)
                self.assertEqual(len(selected), requested_months * 2)
                self.assertEqual(selected_periods.max(), newest)
                self.assertEqual(
                    selected_periods.min(), newest - (requested_months - 1)
                )
                self.assertEqual(
                    audit["history_anchor"],
                    "newest_label_mature_training_cohort",
                )
                self.assertEqual(audit["requested_months"], requested_months)
                self.assertEqual(
                    audit["cutoff_month_inclusive"],
                    str(newest - (requested_months - 1)),
                )
                self.assertTrue(audit["requested_months_fully_available"])


    def test_short_history_reports_shortfall_instead_of_moving_the_anchor(self) -> None:
        available = self.frame.index[-24:]
        selected, audit = select_recent_mature_training_indices(
            self.frame,
            available,
            months=36,
        )
        self.assertTrue(np.array_equal(selected, available.to_numpy()))
        self.assertEqual(audit["available_months_in_window"], 12)
        self.assertFalse(audit["requested_months_fully_available"])


    def test_finalization_accepts_only_exact_equal_weight_history_contracts(self) -> None:
        for months in (36, 48, 60):
            with self.subTest(months=months):
                self.assertEqual(
                    validated_history_months(
                        {
                            "history_id": f"RECENT_{months}M",
                            "kind": "recent",
                            "months": months,
                        },
                        family="logistic",
                    ),
                    months,
                )


        invalid = (
            {"history_id": "RECENT_12M", "kind": "recent", "months": 12},
            {
                "history_id": "DECAY_HALF_LIFE_36M_CAPPED_60M",
                "kind": "decay",
                "months": 36,
            },
            {"history_id": "RECENT_48M", "kind": "recent", "months": 36},
        )
        for history in invalid:
            with self.subTest(history=history), self.assertRaises(ValueError):
                validated_history_months(history, family="xgboost")




class OuterEvaluationContractTests(unittest.TestCase):
    @staticmethod
    def prediction_frame(*, misalign_xgboost: bool = False) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for family_offset, family in enumerate(("logistic", "xgboost")):
            for fold in (1, 2):
                for position in range(12):
                    cohort = pd.Timestamp("2022-01-01") + pd.DateOffset(
                        months=(fold - 1) * 3 + position // 4
                    )
                    if misalign_xgboost and family == "xgboost" and position == 0:
                        cohort += pd.DateOffset(months=1)
                    retained = position % 2
                    rows.append(
                        {
                            "model_family": family,
                            "y_retention": retained,
                            "p_retention": (
                                0.25 + 0.5 * retained + 0.01 * family_offset
                            ),
                            "fold": fold,
                            "cohort_date": cohort,
                        }
                    )
        return pd.DataFrame(rows)


    def test_prediction_input_is_strict(self) -> None:
        valid = self.prediction_frame()
        normalized = validate_prediction_frame(
            valid,
            evaluation_scope="outer_oof",
        )
        self.assertEqual(len(normalized), len(valid))


        with self.assertRaises(KeyError):
            validate_prediction_frame(
                valid.drop(columns="fold"),
                evaluation_scope="outer_oof",
            )
        invalid_probability = valid.copy()
        invalid_probability.loc[0, "p_retention"] = 1.01
        with self.assertRaises(ValueError):
            validate_prediction_frame(
                invalid_probability,
                evaluation_scope="outer_oof",
            )
        with self.assertRaises(ValueError):
            validate_prediction_frame(
                self.prediction_frame(misalign_xgboost=True),
                evaluation_scope="outer_oof",
            )
        with self.assertRaises(ValueError):
            validate_prediction_frame(valid, evaluation_scope="inner_tuning")


    def test_synthetic_outer_oof_plot_smoke_writes_five_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = generate_performance_plots(
                self.prediction_frame(),
                temporary,
                evaluation_scope="outer_oof",
                filename_prefix="contract_smoke",
            )
            self.assertEqual(manifest["evaluation_scope"], "outer_oof")
            self.assertEqual(manifest["ranking_scope"], "within_fold")
            self.assertEqual(len(manifest["artifacts"]), 5)
            for artifact in manifest["artifacts"]:
                path = Path(artifact["path"])
                self.assertEqual(path.suffix.lower(), ".png")
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)


    def test_final_refit_population_excludes_holdout_and_unmature_labels(self) -> None:
        frame = pd.DataFrame(
            {
                "cohort": pd.to_datetime(
                    [
                        "2022-06-01",
                        "2022-07-01",
                        "2022-08-01",
                        "2023-08-01",
                        "2023-09-01",
                    ]
                ),
                "target_check": pd.to_datetime(
                    [
                        "2023-06-30",
                        "2023-07-31",
                        "2023-08-01",  # Boundary is excluded: cutoff is strict.
                        "2024-08-01",
                        "2024-09-01",
                    ]
                ),
            }
        )
        selected, audit = final_evaluation_training_indices(
            frame,
            np.array([3, 4]),
            cohort_col="cohort",
            target_check_col="target_check",
            label_buffer_days=0,
        )
        self.assertTrue(np.array_equal(selected, np.array([0, 1])))
        self.assertEqual(audit["holdout_start"], "2023-08-01")
        self.assertEqual(audit["label_cutoff_exclusive"], "2023-08-01")


    def test_outer_family_selection_uses_aligned_folds_and_parsimony_tie_break(self) -> None:
        fold_metrics = pd.DataFrame(
            [
                {
                    "fold": fold,
                    "model_family": family,
                    "churn_ap_skill": 0.2 + 0.001 * fold,
                    "churn_recall_10pct": 0.3 + 0.001 * fold,
                }
                for fold in range(1, 6)
                for family in ("logistic", "xgboost")
            ]
        )
        selected, comparison, decision = choose_model_family(fold_metrics)
        self.assertEqual(selected, "logistic")
        self.assertEqual(comparison["outer_folds"].tolist(), [5, 5])
        self.assertFalse(decision["final_holdout_used"])
        self.assertEqual(
            decision["selection_population"],
            "five aligned outer validation folds",
        )


        missing_fold = fold_metrics.loc[
            ~(
                fold_metrics["model_family"].eq("xgboost")
                & fold_metrics["fold"].eq(5)
            )
        ]
        with self.assertRaises(ValueError):
            choose_model_family(missing_fold)




class CrossTenureCommandGuardTests(unittest.TestCase):
    def _quiet_system_exit(self, callable_object, argv: list[str]) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            callable_object(argv)


    def test_evaluator_defaults_to_outer_oof_grouping_evidence(self) -> None:
        args = cross_tenure_evaluation.parse_args(
            ["--source-tenure", "1", "--target-tenure", "2"]
        )
        self.assertEqual(
            args.purpose,
            cross_tenure_evaluation.TENURE_GROUPING_PURPOSE,
        )
        self.assertEqual(args.evaluation_window, "outer-oof")


    def test_evaluator_blocks_non_outer_windows_from_grouping_selection(self) -> None:
        for window in ("source-holdout", "all-target", "final-test"):
            with self.subTest(window=window):
                self._quiet_system_exit(
                    cross_tenure_evaluation.parse_args,
                    [
                        "--source-tenure",
                        "1",
                        "--target-tenure",
                        "2",
                        "--evaluation-window",
                        window,
                    ],
                )


    def test_legacy_diagnostic_requires_explicit_purpose_and_acknowledgement(self) -> None:
        base = [
            "--source-tenure",
            "1",
            "--target-tenure",
            "2",
            "--purpose",
            cross_tenure_evaluation.NON_SELECTION_DIAGNOSTIC_PURPOSE,
            "--evaluation-window",
            "source-holdout",
        ]
        self._quiet_system_exit(cross_tenure_evaluation.parse_args, base)
        args = cross_tenure_evaluation.parse_args(
            [*base, "--confirm-final-holdout-diagnostic"]
        )
        self.assertEqual(args.evaluation_window, "source-holdout")


    def test_one_click_launcher_cannot_bypass_grouping_guard(self) -> None:
        args, forwarded = d0_cross_tenure_one_click.parse_launcher_args(
            [
                "--source-tenure",
                "1",
                "--target-tenure",
                "2",
                "--source-outer-manifest",
                "source.json",
            ]
        )
        self.assertEqual(args.evaluation_window, "outer-oof")
        self.assertEqual(
            forwarded,
            ["--source-outer-manifest", "source.json"],
        )
        self._quiet_system_exit(
            lambda argv: d0_cross_tenure_one_click.parse_launcher_args(argv),
            [
                "--source-tenure",
                "1",
                "--target-tenure",
                "2",
                "--evaluation-window",
                "all-target",
            ],
        )




def _load_a0_with_stubbed_project_modules():
    """Load the root parser without requiring a copied full project tree."""


    calls: list[dict[str, object]] = []


    class StubProfile:
        business_scope = "ALL"
        tenure_groups = ("0", "1", "2", "3", "4+")
        filter_mode = "ALL"
        slug = "all"


    def profile_from_values(**kwargs):
        calls.append(dict(kwargs))
        return StubProfile()


    @contextmanager
    def exclusive_operation_lock(_path):
        yield


    func = types.ModuleType("func")
    func.__path__ = []
    modeling = types.ModuleType("func.modeling")
    modeling.run_scripts = lambda **_kwargs: None
    operation_lock = types.ModuleType("func.operation_lock")
    operation_lock.exclusive_operation_lock = exclusive_operation_lock
    tenure_profile = types.ModuleType("func.tenure_profile")
    tenure_profile.BUSINESS_SCOPE_ENV = "RETENTION_BUSINESS_SCOPE"
    tenure_profile.LEGACY_TENURE_ENV_VARS = ()
    tenure_profile.PROFILE_HASH_ENV = "RETENTION_PROFILE_HASH"
    tenure_profile.PROFILE_PAYLOAD_ENV = "RETENTION_PROFILE_PAYLOAD"
    tenure_profile.PROFILE_SLUG_ENV = "RETENTION_PROFILE_SLUG"
    tenure_profile.TENURE_FILTER_MODE_ENV = "RETENTION_TENURE_FILTER_MODE"
    tenure_profile.TENURE_GROUPS_ENV = "RETENTION_TENURE_GROUPS"
    tenure_profile.TenureProfile = StubProfile
    tenure_profile.profile_from_values = profile_from_values
    stubs = {
        "func": func,
        "func.modeling": modeling,
        "func.operation_lock": operation_lock,
        "func.tenure_profile": tenure_profile,
    }
    module_path = PROJECT_ROOT / "a_business_understanding" / "a0_one_click.py"
    module_name = "_contract_test_a0_one_click"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module, calls




class RootOneClickParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.a0, cls.profile_calls = _load_a0_with_stubbed_project_modules()


    def _quiet_system_exit(self, argv: list[str]) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.a0.parse_args(argv)


    def test_no_selector_defaults_to_all_business_all_tenures(self) -> None:
        self.profile_calls.clear()
        args = self.a0.parse_args([])
        self.assertEqual(self.a0.DEFAULT_TENURE_FILTER, "ALL")
        self.assertEqual(self.profile_calls[-1]["tenure_values"], "ALL")
        self.assertEqual(args.profile.tenure_groups, ("0", "1", "2", "3", "4+"))


    def test_explicit_tenure_selector_overrides_default_all(self) -> None:
        self.profile_calls.clear()
        self.a0.parse_args(["--tenures", "2"])
        self.assertEqual(self.profile_calls[-1]["tenure_values"], ["2"])


    def test_relocated_launcher_resolves_stages_from_project_root(self) -> None:
        self.assertEqual(self.a0.PROJECT_ROOT, PROJECT_ROOT)
        self.assertEqual(
            [path.relative_to(PROJECT_ROOT).as_posix() for path in self.a0.STAGES],
            ["b_data_acquisition/b0_one_click.py", "c_modeling/c0_one_click.py"],
        )


    def test_one_click_does_not_expose_a_final_holdout_unseal_flag(self) -> None:
        ordinary = self.a0.parse_args([])
        self.assertFalse(hasattr(ordinary, "score_final_holdout"))
        self._quiet_system_exit(["--score-final-holdout"])


    def test_sample_scope_cannot_enter_outer_evaluation_or_deployment(self) -> None:
        self._quiet_system_exit(["--sample-percent", "2"])
        allowed = self.a0.parse_args(
            ["--sample-percent", "2", "--acquisition-only"]
        )
        self.assertTrue(allowed.acquisition_only)




class ModelingStageDispatchTests(unittest.TestCase):
    def test_normal_modeling_runs_through_c4_and_validates_outputs(self) -> None:
        from c_modeling import c0_one_click as controller


        setup = mock.Mock()
        events = []


        def dispatch(*, scripts, path, env):
            events.append("stages")
            self.assertEqual(list(scripts), [
                "c1_prepare_modeling_data.py", "c2_run_logistic_regression.py",
                "c3_run_xgboost.py", "c4_evaluate_and_finalize.py",
            ])
            self.assertEqual(path, PROJECT_ROOT / "c_modeling")
            self.assertEqual(env["RETENTION_PIPELINE_RUN_ID"], "SYNTHETIC_RUN")


        def validate(run_id):
            events.append("validate")
            self.assertEqual(run_id, "SYNTHETIC_RUN")
            return {"final_holdout_rows_scored": 0}


        with mock.patch.multiple(
            controller, ROOT_ORCHESTRATED=True, FAILED_RUN_RESUME=False,
            XGB_RECONFIRMATION=False, MODEL_RESTART_STAGE="",
        ), mock.patch.object(controller, "_run_id", return_value="SYNTHETIC_RUN"), \
             mock.patch.object(controller, "_read_state", return_value={}), \
             mock.patch.object(controller, "_create_setup", return_value=setup), \
             mock.patch.object(controller, "profile_env", return_value={}), \
             mock.patch.object(controller, "run_scripts", side_effect=dispatch), \
             mock.patch.object(controller, "_validate_finalization_outputs", side_effect=validate), \
             mock.patch.object(controller, "_write_state") as write_state, \
             redirect_stdout(io.StringIO()):
            controller.main()
        self.assertEqual(events, ["stages", "validate"])
        setup.shutdown.assert_called_once()
        write_state.assert_not_called()




if __name__ == "__main__":
    unittest.main()
