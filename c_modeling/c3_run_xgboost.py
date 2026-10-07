# -*- coding: utf-8 -*-
"""Run XGBoost feature selection on the prepared modeling population."""




from __future__ import annotations




import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys








PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))




from c_modeling.feature_routing import (  # noqa: E402
    FEATURE_VARIANT_MAP_ENV,
    FEATURE_VARIANT_MAP_PATH,
)
from c_modeling.xgboost_selection_policy import (  # noqa: E402
    FINAL_FEATURE_SELECTION_RULE,
    policy_sha256,
    validated_policy,
)
from params import (  # noqa: E402
    DOWNLOAD_PATH,
    FEATURE_CONTRACT_PATH,
    FINAL_HOLDOUT_MONTHS,
    INNER_SELECTION_OUTER_FOLD,
    INNER_VALIDATION_FOLDS,
    INNER_VALIDATION_MONTHS,
    MIN_INNER_TRAIN_MONTHS,
    MIN_OUTER_TRAIN_MONTHS,
    N_JOBS,
    OUTER_VALIDATION_MONTHS,
    PIPELINE_OUTPUT_ROOT,
    PROFILE_HASH,
    PROFILE_PAYLOAD,
    ROLLING_VALIDATION_FOLDS,
    SEED,
    STRICT_DATASET_CONTRACT,
    XGB_DEVICE,
    profile_env,
)




RUN_ID = os.environ.get("RETENTION_PIPELINE_RUN_ID", "manual").strip() or "manual"
PREPARATION_DIR = Path(PIPELINE_OUTPUT_ROOT) / RUN_ID / "preparation"
OUTPUT_DIR = Path(PIPELINE_OUTPUT_ROOT) / RUN_ID / "xgboost"
DATA = Path(DOWNLOAD_PATH)
MANIFEST = Path(FEATURE_CONTRACT_PATH)
STARTING_POOL = PREPARATION_DIR / "quality_screened_xgboost_candidate_pool.txt"
FEATURE_TYPES = PREPARATION_DIR / "xgboost_feature_types.json"
ROUTING_AUDIT = PREPARATION_DIR / "feature_routing_audit.csv"
PREPARATION_INFO = PREPARATION_DIR / "preparation_info.json"
OUTER_FOLD_DEFINITIONS = PREPARATION_DIR / "outer_fold_definitions.csv"
INNER_FOLD_DEFINITIONS = PREPARATION_DIR / "inner_fold_definitions.csv"
MODELING_UTILS = PROJECT_ROOT / "c_modeling" / "modeling_utils.py"
MODEL_DIAGNOSTICS = PROJECT_ROOT / "c_modeling" / "model_diagnostics.py"
FEATURE_ROUTING = PROJECT_ROOT / "c_modeling" / "feature_routing.py"
FEATURE_VARIANT_MAP = Path(FEATURE_VARIANT_MAP_PATH).resolve()
SCRIPT = PROJECT_ROOT / "c_modeling" / "xgboost_feature_selection.py"
SELECTION_POLICY_MODULE = PROJECT_ROOT / "c_modeling" / "xgboost_selection_policy.py"
FINAL_SELECTION_POLICY = validated_policy({
    "mean_ap_skill_loss": float(os.environ.get("RETENTION_XGB_FINAL_AP_SKILL_LOSS", "0.001")),
    "mean_recall_loss": float(os.environ.get("RETENTION_XGB_FINAL_RECALL_LOSS", "0.001")),
    "max_fold_ap_skill_loss": float(os.environ.get("RETENTION_XGB_FINAL_MAX_FOLD_AP_SKILL_LOSS", "0.003")),
    "max_fold_recall_loss": float(os.environ.get("RETENTION_XGB_FINAL_MAX_FOLD_RECALL_LOSS", "0.005")),
})
FORWARD_TRAIN_MAX_ROWS = int(
    os.environ.get("RETENTION_XGB_FORWARD_TRAIN_MAX_ROWS", "500000")
)
FORWARD_VALIDATION_MAX_ROWS = int(
    os.environ.get("RETENTION_XGB_FORWARD_VALIDATION_MAX_ROWS", "60000")
)
FORWARD_MIN_FEATURES = int(
    os.environ.get("RETENTION_XGB_FORWARD_MIN_FEATURES", "8")
)
FORWARD_STEP_SIZE = int(
    os.environ.get("RETENTION_XGB_FORWARD_STEP_SIZE", "4")
)
OPTUNA_TRIALS = int(
    os.environ.get("RETENTION_XGB_OPTUNA_TRIALS", "8")
)
OPTUNA_BATCH_SIZE = int(
    os.environ.get("RETENTION_XGB_OPTUNA_BATCH_SIZE", "8")
)
OPTUNA_TIMEOUT_SECONDS = int(
    os.environ.get("RETENTION_XGB_OPTUNA_TIMEOUT_SECONDS", "0")
)
CONFIRM_TRAIN_MAX_ROWS = int(
    os.environ.get("RETENTION_XGB_CONFIRM_TRAIN_MAX_ROWS", "1000000")
)
CONFIRM_CANDIDATE_COUNT = int(
    os.environ.get("RETENTION_XGB_CONFIRM_CANDIDATE_COUNT", "4")
)
CONFIRM_MAX_PASSES = int(
    os.environ.get("RETENTION_XGB_CONFIRM_MAX_PASSES", "5")
)
# These legacy tolerances still affect the Stage-3 eligible-prefix pool.  The
# discovery is chosen with paired AP-skill/recall one-standard-error rules;
# final feature-count selection uses separate practical limits. Changing a
# Stage-3 tolerance can change which groups reach the final decision, so
# all four remain explicit and are included in the reuse contract.
AP_SKILL_TOLERANCE = float(
    os.environ.get("RETENTION_XGB_AP_SKILL_TOLERANCE", "0.001")
)
LIFT_TOLERANCE = float(
    os.environ.get("RETENTION_XGB_LIFT_TOLERANCE", "0.005")
)
AUC_TOLERANCE = float(
    os.environ.get("RETENTION_XGB_AUC_TOLERANCE", "0.001")
)
RECENT_AP_SKILL_TOLERANCE = float(
    os.environ.get("RETENTION_XGB_RECENT_AP_SKILL_TOLERANCE", "0.003")
)
POST_SELECTION_RECOVERY_ENV = "RETENTION_XGB_RESUME_POST_SELECTION"
STAGE4_SELECTION_CHECKPOINT_NAME = "xgb_stage4_selection_checkpoint.json"
STAGE4_SELECTION_CONTRACT_VERSION = 5
STAGE4_SELECTION_RULE = "PAIRED_FOLD_ONE_STANDARD_ERROR_AP_SKILL_RECALL_10PCT"
FINALIST_TUNING_CONTRACT_VERSION = 3
FINALIST_OPTUNA_TIMEOUT_SECONDS = 0
FINALIST_ARTIFACTS = (
    "xgb_finalist_inventory.csv",
    "xgb_finalist_optuna_trials.csv",
    "xgb_finalist_optuna_studies.json",
    "xgb_finalist_configuration_screen_summary.csv",
    "xgb_finalist_configuration_confirmation_long.csv",
    "xgb_finalist_configuration_confirmation_summary.csv",
    "xgb_finalist_selected_configurations.json",
    "xgb_finalist_tuned_group_confirmation_long.csv",
    "xgb_finalist_tuned_group_confirmation_summary.csv",
)








def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()








def _read_pool() -> list[str]:
    values = [
        line.strip().upper()
        for line in STARTING_POOL.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError(
            "XGBoost candidate pool is empty or contains duplicates."
        )
    return values








def _read_feature_types(pool: list[str]) -> dict[str, str]:
    """Validate the XGBoost type contract emitted with the family pool."""




    try:
        raw = json.loads(FEATURE_TYPES.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid XGBoost feature-types JSON: {FEATURE_TYPES}"
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError("XGBoost feature-types artifact must be a JSON object.")
    feature_types = {
        str(feature).strip().upper(): str(var_type).strip().lower()
        for feature, var_type in raw.items()
    }
    if set(feature_types) != set(pool):
        missing = sorted(set(pool).difference(feature_types))
        extra = sorted(set(feature_types).difference(pool))
        raise ValueError(
            "XGBoost pool and feature-types artifact disagree: "
            f"missing_types={missing}; extra_types={extra}."
        )
    invalid = {
        feature: var_type
        for feature, var_type in feature_types.items()
        if var_type not in {"continuous", "discrete"}
    }
    if invalid:
        raise ValueError(f"Invalid XGBoost feature types: {invalid}")
    return feature_types








def _read_resolved_stage4_checkpoint() -> dict[str, object] | None:
    """Read the resolved v4 finalist-retuning boundary used by reuse."""




    checkpoint_path = OUTPUT_DIR / STAGE4_SELECTION_CHECKPOINT_NAME
    try:
        if (
            checkpoint_path.is_symlink()
            or not checkpoint_path.is_file()
            or checkpoint_path.stat().st_size <= 0
        ):
            return None
        payload = json.loads(
            checkpoint_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    policy_module_receipt = payload.get("selection_policy_module")
    if (
        not isinstance(payload, dict)
        or payload.get("contract_version") != STAGE4_SELECTION_CONTRACT_VERSION
        or str(payload.get("status", "")).strip().upper() != "RESOLVED"
        or payload.get("selection_rule") != STAGE4_SELECTION_RULE
        or payload.get("profile_hash") != PROFILE_HASH
        or payload.get("confirm_candidate_count_per_pass")
        != CONFIRM_CANDIDATE_COUNT
        or payload.get("confirm_max_passes") != CONFIRM_MAX_PASSES
        or payload.get("finalist_tuning_contract_version")
        != FINALIST_TUNING_CONTRACT_VERSION
        or payload.get("finalist_selection_rule") != FINAL_FEATURE_SELECTION_RULE
        or payload.get("final_feature_selection_policy") != FINAL_SELECTION_POLICY
        or payload.get("final_feature_selection_policy_sha256") != policy_sha256(FINAL_SELECTION_POLICY)
        or not isinstance(policy_module_receipt, dict)
        or policy_module_receipt.get("sha256") != _sha256(SELECTION_POLICY_MODULE)
        or payload.get("finalist_equal_adaptive_trials") != OPTUNA_TRIALS
        or payload.get("finalist_optuna_timeout_seconds")
        != FINALIST_OPTUNA_TIMEOUT_SECONDS
        or payload.get("outer_validation_rows_scored") != 0
        or payload.get("final_holdout_rows_scored") != 0
    ):
        return None
    finalist_ids = payload.get("finalist_group_ids")
    selected_group_id = str(payload.get("selected_group_id", ""))
    selected_configuration = payload.get("selected_configuration")
    bound_artifacts = payload.get("bound_artifacts")
    if (
        not isinstance(finalist_ids, list)
        or not finalist_ids
        or len(finalist_ids) != len({str(value) for value in finalist_ids})
        or selected_group_id not in {str(value) for value in finalist_ids}
        or not isinstance(selected_configuration, dict)
        or str(selected_configuration.get("group_id", "")) != selected_group_id
        or not isinstance(bound_artifacts, dict)
    ):
        return None
    for name in FINALIST_ARTIFACTS:
        artifact_path = OUTPUT_DIR / name
        artifact_info = bound_artifacts.get(name)
        try:
            if (
                not isinstance(artifact_info, dict)
                or artifact_path.is_symlink()
                or not artifact_path.is_file()
                or artifact_path.stat().st_size <= 0
                or str(artifact_info.get("sha256", "")) != _sha256(artifact_path)
            ):
                return None
        except OSError:
            return None
    return payload








def _selected_finalist_matches_checkpoint(
    checkpoint: dict[str, object],
) -> bool:
    """Require the tuned-finalist summary to carry the sole winner marker."""




    summary_path = OUTPUT_DIR / "xgb_finalist_tuned_group_confirmation_summary.csv"
    try:
        if (
            summary_path.is_symlink()
            or not summary_path.is_file()
            or summary_path.stat().st_size <= 0
        ):
            return False
        with summary_path.open("r", encoding="utf-8-sig", newline="") as stream:
            selected = [
                row
                for row in csv.DictReader(stream)
                if str(row.get("selected_by_rule", "")).strip().lower()
                in {"true", "1"}
            ]
    except (OSError, UnicodeError, csv.Error):
        return False
    return (
        len(selected) == 1
        and str(selected[0].get("group_id", ""))
        == str(checkpoint.get("selected_group_id", ""))
    )








def _expected_arguments() -> dict[str, object]:
    """Settings controlled by this wrapper and stored by the engine."""




    return {
        "final_ap_skill_loss": FINAL_SELECTION_POLICY["mean_ap_skill_loss"],
        "final_recall_loss": FINAL_SELECTION_POLICY["mean_recall_loss"],
        "final_max_fold_ap_skill_loss": FINAL_SELECTION_POLICY["max_fold_ap_skill_loss"],
        "final_max_fold_recall_loss": FINAL_SELECTION_POLICY["max_fold_recall_loss"],
        "device": os.environ.get("RETENTION_XGB_DEVICE", XGB_DEVICE),
        "n_jobs": int(os.environ.get("RETENTION_N_JOBS", str(N_JOBS))),
        "outer_folds": ROLLING_VALIDATION_FOLDS,
        "outer_validation_months": OUTER_VALIDATION_MONTHS,
        "test_months": FINAL_HOLDOUT_MONTHS,
        "min_outer_train_months": MIN_OUTER_TRAIN_MONTHS,
        "inner_folds": INNER_VALIDATION_FOLDS,
        "inner_validation_months": INNER_VALIDATION_MONTHS,
        "min_inner_train_months": MIN_INNER_TRAIN_MONTHS,
        "forward_train_max_rows": FORWARD_TRAIN_MAX_ROWS,
        "forward_validation_max_rows": FORWARD_VALIDATION_MAX_ROWS,
        "min_features": FORWARD_MIN_FEATURES,
        "forward_step_size": FORWARD_STEP_SIZE,
        "optuna_trials": OPTUNA_TRIALS,
        "optuna_batch_size": OPTUNA_BATCH_SIZE,
        "optuna_timeout_seconds": OPTUNA_TIMEOUT_SECONDS,
        "confirm_train_max_rows": CONFIRM_TRAIN_MAX_ROWS,
        "confirm_candidate_count": CONFIRM_CANDIDATE_COUNT,
        "confirm_max_passes": CONFIRM_MAX_PASSES,
        "ap_skill_tolerance": AP_SKILL_TOLERANCE,
        "lift_tolerance": LIFT_TOLERANCE,
        "auc_tolerance": AUC_TOLERANCE,
        "recent_ap_skill_tolerance": RECENT_AP_SKILL_TOLERANCE,
        "allow_contract_violations": not STRICT_DATASET_CONTRACT,
    }








def _verify_preparation_profile() -> dict[str, object]:
    try:
        payload = json.loads(PREPARATION_INFO.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid modeling-preparation JSON: {PREPARATION_INFO}"
        ) from exc
    if payload.get("profile_hash") != PROFILE_HASH or payload.get(
        "profile"
    ) != PROFILE_PAYLOAD:
        raise ValueError(
            "XGBoost preparation profile does not match the active tenure "
            "profile; refusing cross-profile reuse or fitting."
        )
    split_settings = payload.get("split_settings")
    anchor = (
        split_settings.get("inner_selection_outer_fold")
        if isinstance(split_settings, dict)
        else None
    )
    if type(anchor) is not int or anchor != INNER_SELECTION_OUTER_FOLD:
        raise ValueError(
            "XGBoost preparation does not use the governed oldest outer-fold "
            f"discovery boundary ({INNER_SELECTION_OUTER_FOLD}); observed "
            f"{anchor!r}. Rerun preparation before fitting."
        )
    artifacts = payload.get("model_family_artifacts", {})
    xgboost = artifacts.get("xgboost", {}) if isinstance(artifacts, dict) else {}
    expected = {
        "quality_screened_pool": str(STARTING_POOL.resolve()),
        "feature_types": str(FEATURE_TYPES.resolve()),
    }
    if not isinstance(xgboost, dict) or any(
        xgboost.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(
            "Modeling preparation does not declare the expected XGBoost "
            "family artifacts."
        )
    if payload.get("feature_routing_audit") != str(ROUTING_AUDIT.resolve()):
        raise ValueError(
            "Modeling preparation does not declare the expected feature-routing "
            "audit."
        )
    if (
        payload.get("feature_variant_map") != str(FEATURE_VARIANT_MAP)
        or payload.get("feature_variant_map_sha256")
        != _sha256(FEATURE_VARIANT_MAP)
    ):
        raise ValueError(
            "Modeling preparation feature-variant mapping does not match the "
            "current mapping path and content."
        )
    return payload








def _can_reuse() -> bool:
    info_path = OUTPUT_DIR / "run_info.json"
    inputs = [
        info_path,
        OUTPUT_DIR / "selected_features.json",
        OUTPUT_DIR / "selected_features.txt",
        OUTPUT_DIR / "xgb_resolved_model_feature_types.json",
        OUTPUT_DIR / "feature_selection_summary.md",
        OUTPUT_DIR / "xgb_optuna_trials.csv",
        OUTPUT_DIR / "xgb_optuna_study.json",
        OUTPUT_DIR / STAGE4_SELECTION_CHECKPOINT_NAME,
        *[OUTPUT_DIR / name for name in FINALIST_ARTIFACTS],
        OUTPUT_DIR / "diagnostics" / "xgboost_diagnostics_manifest.json",
        OUTPUT_DIR / "diagnostics" / "xgboost_calendar_year_performance.csv",
        OUTPUT_DIR / "diagnostics" / "xgboost_calendar_year_one_way_summary.csv",
        OUTPUT_DIR / "diagnostics" / "xgboost_selected_feature_one_way_summaries.csv",
        OUTPUT_DIR / "diagnostics" / "xgboost_policy_start_year_month_trend.csv",
        OUTPUT_DIR / "diagnostics" / "xgboost_policy_start_year_month_trend.png",
        OUTPUT_DIR / "diagnostics" / "xgboost_shap_beeswarm.png",
        OUTPUT_DIR / "diagnostics" / "xgboost_shap_plot_manifest.json",
        DATA,
        MANIFEST,
        MODELING_UTILS,
        MODEL_DIAGNOSTICS,
        FEATURE_ROUTING,
        FEATURE_VARIANT_MAP,
        SCRIPT,
        SELECTION_POLICY_MODULE,
        STARTING_POOL,
        FEATURE_TYPES,
        ROUTING_AUDIT,
        PREPARATION_INFO,
        OUTER_FOLD_DEFINITIONS,
        INNER_FOLD_DEFINITIONS,
    ]
    if not all(path.is_file() for path in inputs):
        return False
    diagnostics_dir = OUTPUT_DIR / "diagnostics"
    try:
        diagnostics_manifest = json.loads(
            (diagnostics_dir / "xgboost_diagnostics_manifest.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError):
        return False
    diagnostic_artifacts = diagnostics_manifest.get("artifacts")
    if not isinstance(diagnostic_artifacts, list) or not all(
        isinstance(name, str)
        and name
        and not Path(name).is_absolute()
        and Path(name).parent == Path(".")
        and (diagnostics_dir / name).is_file()
        for name in diagnostic_artifacts
    ):
        return False
    checkpoint = _read_resolved_stage4_checkpoint()
    if checkpoint is None or not _selected_finalist_matches_checkpoint(checkpoint):
        return False
    summary_path = OUTPUT_DIR / "feature_selection_summary.md"
    try:
        if summary_path.is_symlink() or summary_path.stat().st_size <= 0:
            return False
    except OSError:
        return False
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if info.get("post_selection_recovery") is not None:
        return False
    observed_arguments = info.get("arguments", {})
    expected_arguments = _expected_arguments()
    info_checkpoint = info.get("stage4_selection_checkpoint", {})
    info_finalist_artifacts = info.get("finalist_artifacts", {})
    checkpoint_artifacts = checkpoint.get("bound_artifacts", {})
    finalist_manifests_match = (
        isinstance(info_finalist_artifacts, dict)
        and isinstance(checkpoint_artifacts, dict)
        and all(
            isinstance(info_finalist_artifacts.get(name), dict)
            and isinstance(checkpoint_artifacts.get(name), dict)
            and str(info_finalist_artifacts[name].get("sha256", ""))
            == str(checkpoint_artifacts[name].get("sha256", ""))
            for name in FINALIST_ARTIFACTS
        )
    )
    _read_feature_types(_read_pool())
    return (
        info.get("profile_hash") == PROFILE_HASH
        and info.get("profile") == PROFILE_PAYLOAD
        and info.get("stage4_selection_contract_version")
        == STAGE4_SELECTION_CONTRACT_VERSION
        and info.get("stage4_selection_rule") == STAGE4_SELECTION_RULE
        and info.get("final_feature_selection_rule") == FINAL_FEATURE_SELECTION_RULE
        and info.get("final_feature_selection_policy") == FINAL_SELECTION_POLICY
        and info.get("final_feature_selection_policy_sha256") == policy_sha256(FINAL_SELECTION_POLICY)
        and isinstance(info.get("selection_policy_module"), dict)
        and info["selection_policy_module"].get("sha256") == _sha256(SELECTION_POLICY_MODULE)
        and isinstance(info_checkpoint, dict)
        and info_checkpoint.get("sha256")
        == _sha256(OUTPUT_DIR / STAGE4_SELECTION_CHECKPOINT_NAME)
        and info.get("selected_group_id") == checkpoint.get("selected_group_id")
        and info.get("finalist_tuning_contract_version")
        == FINALIST_TUNING_CONTRACT_VERSION
        and info.get("finalist_group_ids") == checkpoint.get("finalist_group_ids")
        and info.get("finalist_equal_completed_adaptive_trials") == OPTUNA_TRIALS
        and info.get("finalist_optuna_timeout_seconds")
        == FINALIST_OPTUNA_TIMEOUT_SECONDS
        and finalist_manifests_match
        and info.get("input", {}).get("sha256") == _sha256(DATA)
        and info.get("manifest", {}).get("sha256") == _sha256(MANIFEST)
        and info.get("modeling_utils", {}).get("sha256")
        == _sha256(MODELING_UTILS)
        and info.get("model_diagnostics", {}).get("sha256")
        == _sha256(MODEL_DIAGNOSTICS)
        and info.get("feature_routing", {}).get("sha256")
        == _sha256(FEATURE_ROUTING)
        and info.get("feature_types_file", {}).get("sha256")
        == _sha256(FEATURE_TYPES)
        and info.get("routing_audit", {}).get("sha256")
        == _sha256(ROUTING_AUDIT)
        and info.get("script", {}).get("sha256") == _sha256(SCRIPT)
        and info.get("starting_feature_pool", {}).get("sha256")
        == _sha256(STARTING_POOL)
        and info.get("preparation_info", {}).get("sha256")
        == _sha256(PREPARATION_INFO)
        and info.get("outer_fold_definitions", {}).get("sha256")
        == _sha256(OUTER_FOLD_DEFINITIONS)
        and info.get("inner_fold_definitions", {}).get("sha256")
        == _sha256(INNER_FOLD_DEFINITIONS)
        and info.get("random_seed") == SEED
        and isinstance(observed_arguments, dict)
        and all(
            observed_arguments.get(key) == value
            for key, value in expected_arguments.items()
        )
        and info.get("final_holdout_rows_scored") == 0
    )




def _can_resume_post_selection() -> bool:
    """V4 recovery is disabled because finalist retuning must complete fresh."""




    return False








def validate_preparation_inputs() -> dict[str, object]:
    """Validate the frozen inputs without creating or changing any output."""
    required = [
        DATA,
        MANIFEST,
        STARTING_POOL,
        FEATURE_TYPES,
        ROUTING_AUDIT,
        PREPARATION_INFO,
        OUTER_FOLD_DEFINITIONS,
        INNER_FOLD_DEFINITIONS,
        MODELING_UTILS,
        MODEL_DIAGNOSTICS,
        FEATURE_ROUTING,
        FEATURE_VARIANT_MAP,
        SCRIPT,
        SELECTION_POLICY_MODULE,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required XGBoost file(s) missing:\n"
            + "\n".join(f"- {path}" for path in missing)
        )
    preparation = _verify_preparation_profile()
    pool = _read_pool()
    _read_feature_types(pool)
    return preparation




def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-preparation-only", action="store_true",
        help="Check frozen run inputs without fitting or changing outputs.",
    )
    args = parser.parse_args(argv)
    validate_preparation_inputs()
    if args.validate_preparation_only:
        print("XGBOOST PREPARATION INPUTS VERIFIED; NO OUTPUTS CHANGED")
        return
    pool = _read_pool()
    resume_post_selection = (
        os.environ.get(POST_SELECTION_RECOVERY_ENV, "").strip() == "1"
    )
    if resume_post_selection:
        raise RuntimeError(
            "XGBoost post-selection recovery is disabled under the Stage-4 v4 "
            "finalist-retuning contract. Preserve/archive the incomplete attempt "
            "and rerun XGBoost selection from a fresh output directory."
        )
    if _can_reuse():
        print(f"XGBOOST FEATURE SELECTION ALREADY COMPLETE; REUSING {OUTPUT_DIR}")
        return
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Incomplete or mismatched XGBoost output exists: {OUTPUT_DIR}. "
            "Restart the normal pipeline with a0_one_click.py, the same tenure "
            "profile, --resume-run-id <run-id> --restart-modeling-from xgboost. "
            "That operation archives previous XGBoost and downstream results "
            "before fitting the current selection procedure."
        )




    command = [
        sys.executable,
        "-u",
        str(SCRIPT),
        "--project-root",
        str(PROJECT_ROOT),
        "--output-dir",
        str(OUTPUT_DIR),
        "--modeling-utils",
        str(MODELING_UTILS),
        "--input",
        str(DATA),
        "--manifest",
        str(MANIFEST),
        "--starting-feature-pool",
        str(STARTING_POOL),
        "--preparation-info",
        str(PREPARATION_INFO),
        "--outer-fold-definitions",
        str(OUTER_FOLD_DEFINITIONS),
        "--inner-fold-definitions",
        str(INNER_FOLD_DEFINITIONS),
        "--device",
        str(_expected_arguments()["device"]),
        "--n-jobs",
        str(_expected_arguments()["n_jobs"]),
        "--outer-folds",
        str(ROLLING_VALIDATION_FOLDS),
        "--outer-validation-months",
        str(OUTER_VALIDATION_MONTHS),
        "--test-months",
        str(FINAL_HOLDOUT_MONTHS),
        "--min-outer-train-months",
        str(MIN_OUTER_TRAIN_MONTHS),
        "--inner-folds",
        str(INNER_VALIDATION_FOLDS),
        "--inner-validation-months",
        str(INNER_VALIDATION_MONTHS),
        "--min-inner-train-months",
        str(MIN_INNER_TRAIN_MONTHS),
        "--forward-train-max-rows",
        str(FORWARD_TRAIN_MAX_ROWS),
        "--forward-validation-max-rows",
        str(FORWARD_VALIDATION_MAX_ROWS),
        "--min-features",
        str(FORWARD_MIN_FEATURES),
        "--forward-step-size",
        str(FORWARD_STEP_SIZE),
        "--optuna-trials",
        str(OPTUNA_TRIALS),
        "--optuna-batch-size",
        str(OPTUNA_BATCH_SIZE),
        "--optuna-timeout-seconds",
        str(OPTUNA_TIMEOUT_SECONDS),
        "--confirm-train-max-rows",
        str(CONFIRM_TRAIN_MAX_ROWS),
        "--confirm-candidate-count",
        str(CONFIRM_CANDIDATE_COUNT),
        "--confirm-max-passes",
        str(CONFIRM_MAX_PASSES),
        "--ap-skill-tolerance",
        str(AP_SKILL_TOLERANCE),
        "--lift-tolerance",
        str(LIFT_TOLERANCE),
        "--auc-tolerance",
        str(AUC_TOLERANCE),
        "--recent-ap-skill-tolerance",
        str(RECENT_AP_SKILL_TOLERANCE),
        "--final-ap-skill-loss",
        str(FINAL_SELECTION_POLICY["mean_ap_skill_loss"]),
        "--final-recall-loss",
        str(FINAL_SELECTION_POLICY["mean_recall_loss"]),
        "--final-max-fold-ap-skill-loss",
        str(FINAL_SELECTION_POLICY["max_fold_ap_skill_loss"]),
        "--final-max-fold-recall-loss",
        str(FINAL_SELECTION_POLICY["max_fold_recall_loss"]),
    ]
    if not STRICT_DATASET_CONTRACT:
        command.append("--allow-contract-violations")
    env = os.environ.copy()
    env.update(profile_env())
    env["RETENTION_RANDOM_SEED"] = str(SEED)
    env["RETENTION_XGB_FEATURE_TYPES"] = str(FEATURE_TYPES)
    env["RETENTION_FEATURE_ROUTING_AUDIT"] = str(ROUTING_AUDIT)
    env[FEATURE_VARIANT_MAP_ENV] = str(FEATURE_VARIANT_MAP)
    print("XGBOOST FEATURE SELECTION")
    print(f"Starting pool: {len(pool)} features")
    print(
        f"Optuna adaptive trials: {OPTUNA_TRIALS}; "
        f"batch size: {OPTUNA_BATCH_SIZE}; deterministic coverage anchors: YES"
    )
    print(
        "Adaptive feature-group confirmation: "
        f"{CONFIRM_CANDIDATE_COUNT} candidates per pass; "
        f"maximum {CONFIRM_MAX_PASSES} passes; "
        f"train-row cap {CONFIRM_TRAIN_MAX_ROWS:,}"
    )
    print("Final sealed holdout will be scored: NO")
    print(f"Final feature-count practical loss limits: {FINAL_SELECTION_POLICY}")
    subprocess.run(command, cwd=str(PROJECT_ROOT), env=env, check=True)








if __name__ == "__main__":
    main()
