# -*- coding: utf-8 -*-
"""Run or resume Logistic Regression floating-forward feature selection."""


from __future__ import annotations


from datetime import datetime
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


import sklearn




PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from params import (  # noqa: E402
    DOWNLOAD_PATH,
    FEATURE_CONTRACT_PATH,
    FINAL_HOLDOUT_MONTHS,
    INNER_SELECTION_OUTER_FOLD,
    INNER_VALIDATION_FOLDS,
    INNER_VALIDATION_MONTHS,
    LABEL_AVAILABILITY_BUFFER_DAYS,
    MIN_OUTER_TRAIN_MONTHS,
    MIN_INNER_TRAIN_MONTHS,
    OUTER_VALIDATION_MONTHS,
    PIPELINE_OUTPUT_ROOT,
    PROFILE_HASH,
    PROFILE_PAYLOAD,
    ROLLING_VALIDATION_FOLDS,
    SEED,
    STRICT_DATASET_CONTRACT,
    profile_env,
)


RUN_ID = os.environ.get("RETENTION_PIPELINE_RUN_ID", "manual").strip() or "manual"
PREPARATION_DIR = Path(PIPELINE_OUTPUT_ROOT) / RUN_ID / "preparation"
OUTPUT_DIR = Path(PIPELINE_OUTPUT_ROOT) / RUN_ID / "logistic"
DATA = Path(DOWNLOAD_PATH)
MANIFEST = Path(FEATURE_CONTRACT_PATH)
POOL = PREPARATION_DIR / "quality_screened_logistic_candidate_pool.txt"
FEATURE_TYPES = PREPARATION_DIR / "logistic_feature_types.json"
ROUTING_AUDIT = PREPARATION_DIR / "feature_routing_audit.csv"
PREPARATION_INFO = PREPARATION_DIR / "preparation_info.json"
OUTER_FOLD_DEFINITIONS = PREPARATION_DIR / "outer_fold_definitions.csv"
INNER_FOLD_DEFINITIONS = PREPARATION_DIR / "inner_fold_definitions.csv"
CODE_DIR = PROJECT_ROOT / "c_modeling"
SCRIPT = CODE_DIR / "logistic_feature_selection.py"
SUPPORT = CODE_DIR / "logistic_utils.py"
MODELING_UTILS = CODE_DIR / "modeling_utils.py"
MODEL_DIAGNOSTICS = CODE_DIR / "model_diagnostics.py"
FEATURE_ROUTING = CODE_DIR / "feature_routing.py"
STATE_GROUPING = CODE_DIR / "state_grouping.py"


LR_TRAIN_MAX_ROWS = 100_000
LR_VALIDATION_MAX_ROWS = 50_000
LR_MAX_ITER_RIDGE = 2_000
LR_FIXED_C = 0.01
LR_AP_STEP_TOL = 0.00025
LR_AP_TOTAL_TOL = 0.001
LR_LIFT_STEP_TOL = 0.01
LR_LIFT_TOTAL_TOL = 0.02
LR_FLOATING_REMOVAL_AP_GAIN = 0.00025


LR_SCHEMA_VERSION = 19
LR_MODE = "history_tuned_calendar_year_control_grouped_sequential_floating_forward"
LR_MANDATORY_CONTROL_FEATURES = ["COHORT_CALENDAR_YEAR"]
LR_HISTORY_GRID = [
    {"history_id": "RECENT_36M", "kind": "recent", "months": 36},
    {"history_id": "RECENT_48M", "kind": "recent", "months": 48},
    {"history_id": "RECENT_60M", "kind": "recent", "months": 60},
]
LR_HISTORY_SELECTION_POLICY = {
    "screening_c": LR_FIXED_C,
    "primary_metric": "mean_churn_ap_skill",
    "secondary_metric": "mean_top10_lift",
    "tertiary_metric": "newest_fold_churn_ap_skill",
    "sealed_holdout_used": False,
}


# The 2026-09-04 patch adds missing lift-spread fields to every newly computed
# fit row.  It does not alter preprocessing, fitting, predictions, or any
# selection metric.  c2 may therefore migrate only this exact predecessor
# utility identity, and only inside the root controller's guarded failed-
# Logistic recovery path.  Binding both endpoints prevents a future code edit
# from silently inheriting this one-time checkpoint exception.
LR_CHECKPOINT_COMPATIBLE_SELECTOR_SHA256 = frozenset(
    {
        # LF checkout
        "903c94fc5aeaf78473798d73f1212bb8deb0207d5e973f9853637ed3f64d770b",
        # CRLF checkout
        "0fdc5c0daa39efecf89f0f7d6422dbc20c6cd272758df15f479b7079baa00d4f",
    }
)
LR_CHECKPOINT_BROKEN_UTILS_SHA256 = frozenset(
    {
        # LF checkout
        "45b97b192723553f49626f1dd31fdbf365a5a945bb6d19fd027cfc6a31ee7272",
        # CRLF checkout
        "d2fae6a36950d102cc3cf2130805680a7c5ed1c50d8bb2a07252008699f66af7",
    }
)
LR_CHECKPOINT_FIXED_UTILS_SHA256 = frozenset(
    {
        # LF checkout
        "be3b501556f141a94e6d1484330c0821402b495f110ef9c5b5ccaf0c95215a9b",
        # CRLF checkout
        "ccd0a2e577e80c7b6d49c1a65676b338b43feb15212dea95e3494164679eeefb",
    }
)
FAILED_RUN_RESUME_ENV = "RETENTION_RESUME_FAILED_RUN"
FAILED_RUN_RESUME_STAGE_ENV = "RETENTION_RESUME_FAILED_STAGE"
FAILED_RUN_RESUME_ATTEMPT_ID_ENV = "RETENTION_RESUME_ATTEMPT_ID"




def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()




def _read_pool() -> list[str]:
    values = [
        line.strip().upper()
        for line in POOL.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError("Candidate pool is empty or contains duplicates.")
    return values




def _read_feature_types(pool: list[str]) -> dict[str, str]:
    """Validate the Logistic type contract emitted with the family pool."""


    try:
        raw = json.loads(FEATURE_TYPES.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid Logistic feature-types JSON: {FEATURE_TYPES}"
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError("Logistic feature-types artifact must be a JSON object.")
    feature_types = {
        str(feature).strip().upper(): str(var_type).strip().lower()
        for feature, var_type in raw.items()
    }
    if set(feature_types) != set(pool):
        missing = sorted(set(pool).difference(feature_types))
        extra = sorted(set(feature_types).difference(pool))
        raise ValueError(
            "Logistic pool and feature-types artifact disagree: "
            f"missing_types={missing}; extra_types={extra}."
        )
    invalid = {
        feature: var_type
        for feature, var_type in feature_types.items()
        if var_type not in {"continuous", "discrete"}
    }
    if invalid:
        raise ValueError(f"Invalid Logistic feature types: {invalid}")
    return feature_types




def _budget_environment() -> dict[str, str]:
    """Default to an uninterrupted run while honoring explicit operator caps."""


    return {
        "RETENTION_LR_RUNTIME_BUDGET_HOURS": os.environ.get(
            "RETENTION_LR_RUNTIME_BUDGET_HOURS", "0"
        ),
        "RETENTION_LR_MAX_NEW_FITS": os.environ.get(
            "RETENTION_LR_MAX_NEW_FITS", "0"
        ),
    }




def _file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256(path),
    }




def _config_fingerprint(payload: dict[str, object]) -> str:
    invocation_keys = {
        "invocation_runtime_budget_hours",
        "invocation_max_new_fits",
        "invocation_max_actions_safety_cap",
    }
    immutable = {
        key: value
        for key, value in payload.items()
        if key != "config_fingerprint" and key not in invocation_keys
    }
    return hashlib.sha256(
        json.dumps(immutable, sort_keys=True).encode("utf-8")
    ).hexdigest()




def _migrate_lift_spread_checkpoint_config() -> bool:
    """Authorize one exact reporting-only utility fix for saved fit rows."""


    guarded = (
        os.environ.get("RETENTION_ROOT_ORCHESTRATED", "").strip() == "1"
        and os.environ.get(FAILED_RUN_RESUME_ENV, "").strip() == "1"
        and os.environ.get(FAILED_RUN_RESUME_STAGE_ENV, "").strip().lower()
        == "logistic"
        and bool(os.environ.get(FAILED_RUN_RESUME_ATTEMPT_ID_ENV, "").strip())
    )
    if not guarded:
        return False


    config_path = OUTPUT_DIR / "run_config.json"
    checkpoint_path = OUTPUT_DIR / ".fit_checkpoint.sqlite3"
    if (
        config_path.is_symlink()
        or not config_path.is_file()
        or checkpoint_path.is_symlink()
        or not checkpoint_path.is_file()
        or checkpoint_path.stat().st_size <= 0
    ):
        raise RuntimeError(
            "Guarded Logistic recovery has no safe run config and fit checkpoint."
        )
    try:
        original_config_text = config_path.read_text(encoding="utf-8")
        config = json.loads(original_config_text)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Saved Logistic run config is invalid.") from exc
    if not isinstance(config, dict):
        raise RuntimeError("Saved Logistic run config must be a JSON object.")
    if config.get("config_fingerprint") != _config_fingerprint(config):
        raise RuntimeError("Saved Logistic run-config fingerprint is invalid.")
    if (
        config.get("profile_hash") != PROFILE_HASH
        or config.get("profile") != PROFILE_PAYLOAD
    ):
        raise RuntimeError(
            "Saved Logistic run config does not match the active profile."
        )
    status_path = OUTPUT_DIR / "run_status.json"
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Saved Logistic run status is invalid.") from exc
        if isinstance(status, dict) and str(status.get("status", "")).upper() == "COMPLETE":
            raise RuntimeError(
                "A COMPLETE Logistic run cannot use incomplete-checkpoint migration."
            )


    script_identity = config.get("script")
    utils_identity = config.get("logistic_utils")
    prior_script_sha = (
        str(script_identity.get("sha256", "")).strip().lower()
        if isinstance(script_identity, dict)
        else ""
    )
    prior_utils_sha = (
        str(utils_identity.get("sha256", "")).strip().lower()
        if isinstance(utils_identity, dict)
        else ""
    )
    current_script_sha = _sha256(SCRIPT)
    current_utils_sha = _sha256(SUPPORT)
    if prior_utils_sha == current_utils_sha:
        return False
    expected = (
        prior_script_sha in LR_CHECKPOINT_COMPATIBLE_SELECTOR_SHA256
        and current_script_sha == prior_script_sha
        and prior_utils_sha in LR_CHECKPOINT_BROKEN_UTILS_SHA256
        and current_utils_sha in LR_CHECKPOINT_FIXED_UTILS_SHA256
    )
    if not expected:
        raise RuntimeError(
            "The interrupted Logistic checkpoint does not match the one exact "
            "reporting-only compatibility migration; migration was refused."
        )


    unchanged_identities = {
        "script": SCRIPT,
        "modeling_utils": MODELING_UTILS,
        "model_diagnostics": MODEL_DIAGNOSTICS,
        "feature_routing": FEATURE_ROUTING,
        "state_grouping": STATE_GROUPING,
        "feature_types_file": FEATURE_TYPES,
        "routing_audit": ROUTING_AUDIT,
        "preparation_info": PREPARATION_INFO,
        "data": DATA,
        "manifest": MANIFEST,
        "pool_file": POOL,
        "outer_fold_definitions": OUTER_FOLD_DEFINITIONS,
        "inner_fold_definitions": INNER_FOLD_DEFINITIONS,
    }
    changed = []
    for key, path in unchanged_identities.items():
        saved_identity = config.get(key)
        if (
            not isinstance(saved_identity, dict)
            or saved_identity != _file_identity(path)
        ):
            changed.append(key)
    if changed:
        raise RuntimeError(
            "Checkpoint migration permits only the exact Logistic utility fix; "
            "these additional identities changed: " + ", ".join(changed)
        )


    migrated = dict(config)
    migrated["logistic_utils"] = _file_identity(SUPPORT)
    migrated["config_fingerprint"] = _config_fingerprint(migrated)
    backup_path = OUTPUT_DIR / "run_config.before_lift_spread_fix.json"
    if backup_path.exists() or backup_path.is_symlink():
        try:
            matching_backup = (
                not backup_path.is_symlink()
                and backup_path.is_file()
                and backup_path.read_text(encoding="utf-8")
                == original_config_text
            )
        except OSError:
            matching_backup = False
        if not matching_backup:
            raise FileExistsError(
                "Checkpoint migration backup exists but does not exactly match "
                f"the pre-migration config: {backup_path}"
            )
    else:
        backup_temporary = backup_path.with_name(backup_path.name + ".tmp")
        backup_temporary.write_text(original_config_text, encoding="utf-8")
        os.replace(backup_temporary, backup_path)
    temporary = config_path.with_name(config_path.name + ".tmp")
    temporary.write_text(json.dumps(migrated, indent=2), encoding="utf-8")
    os.replace(temporary, config_path)
    receipt = {
        "status": "COMPLETE",
        "migration": "LIFT_SPREAD_REPORTING_FIELDS_ONLY",
        "prior_logistic_utils_sha256": prior_utils_sha,
        "current_logistic_utils_sha256": current_utils_sha,
        "selector_sha256": current_script_sha,
        "checkpoint_database_reused": True,
        "legacy_fit_rows_recomputed": True,
        "statistical_fit_contract_changed": False,
        "recovery_attempt_id": os.environ[
            FAILED_RUN_RESUME_ATTEMPT_ID_ENV
        ].strip(),
        "migrated_at": datetime.now().isoformat(timespec="seconds"),
    }
    receipt_path = OUTPUT_DIR / "checkpoint_compatibility_receipt.json"
    receipt_temporary = receipt_path.with_name(receipt_path.name + ".tmp")
    receipt_temporary.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    os.replace(receipt_temporary, receipt_path)
    print(
        "Validated reporting-only Logistic checkpoint migration. The checkpoint "
        "database and run identity are preserved; legacy fit rows missing the "
        "new diagnostics will be recomputed.",
        flush=True,
    )
    return True




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
            "Logistic preparation profile does not match the active tenure "
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
            "Logistic preparation does not use the governed oldest outer-fold "
            f"discovery boundary ({INNER_SELECTION_OUTER_FOLD}); observed "
            f"{anchor!r}. Rerun preparation before fitting."
        )
    artifacts = payload.get("model_family_artifacts", {})
    logistic = artifacts.get("logistic", {}) if isinstance(artifacts, dict) else {}
    expected = {
        "quality_screened_pool": str(POOL.resolve()),
        "feature_types": str(FEATURE_TYPES.resolve()),
    }
    if not isinstance(logistic, dict) or any(
        logistic.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(
            "Modeling preparation does not declare the expected Logistic "
            "family artifacts."
        )
    if payload.get("feature_routing_audit") != str(ROUTING_AUDIT.resolve()):
        raise ValueError(
            "Modeling preparation does not declare the expected feature-routing "
            "audit."
        )
    return payload




def _selector_c_grid() -> tuple[float, ...]:
    """Read the selector-owned grid for reuse checks without importing it.


    The selector performs runtime setup at import time. Inspect only its
    C_GRID declaration so c2 never needs a second copy of the default values.
    """
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8-sig"), filename=str(SCRIPT))
    grid_values = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "C_GRID"
            for target in node.targets
        )
    ]
    if len(grid_values) != 1:
        raise RuntimeError("Expected one selector-owned C_GRID declaration.")
    defaults = [
        node.args[1].value
        for node in ast.walk(grid_values[0])
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "os.environ.get"
        and len(node.args) == 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "RETENTION_LR_C_GRID"
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ]
    if len(defaults) != 1:
        raise RuntimeError("Cannot resolve the selector-owned C-grid default.")
    raw = os.environ.get("RETENTION_LR_C_GRID", defaults[0])
    return tuple(
        sorted({float(value.strip()) for value in raw.split(",") if value.strip()})
    )


def _can_reuse() -> bool:
    status_path = OUTPUT_DIR / "run_status.json"
    info_path = OUTPUT_DIR / "run_info.json"
    recommendation_path = OUTPUT_DIR / "recommended_model.json"
    recommended_features_path = OUTPUT_DIR / "recommended_features.txt"
    resolved_types_path = OUTPUT_DIR / "resolved_logistic_model_feature_types.json"
    history_decision_path = OUTPUT_DIR / "history_selection_decision.json"
    inputs = [
        status_path,
        info_path,
        recommendation_path,
        recommended_features_path,
        resolved_types_path,
        history_decision_path,
        OUTPUT_DIR / "run_config.json",
        OUTPUT_DIR / "metric_contract.json",
        OUTPUT_DIR / "history_population_identity.json",
        OUTPUT_DIR / "selected_history_fold_definitions.csv",
        DATA,
        MANIFEST,
        POOL,
        FEATURE_TYPES,
        ROUTING_AUDIT,
        PREPARATION_INFO,
        OUTER_FOLD_DEFINITIONS,
        INNER_FOLD_DEFINITIONS,
        OUTPUT_DIR / "diagnostics" / "logistic_diagnostics_manifest.json",
        SCRIPT,
        SUPPORT,
        MODELING_UTILS,
        MODEL_DIAGNOSTICS,
        FEATURE_ROUTING,
        STATE_GROUPING,
    ]
    if not all(path.is_file() for path in inputs):
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        info = json.loads(info_path.read_text(encoding="utf-8"))
        recommendation = json.loads(
            recommendation_path.read_text(encoding="utf-8")
        )
        resolved_types = json.loads(resolved_types_path.read_text(encoding="utf-8"))
        history_decision = json.loads(
            history_decision_path.read_text(encoding="utf-8")
        )
        recommended_features = [
            value.strip().upper()
            for value in recommended_features_path.read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if value.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not all(
        isinstance(payload, dict)
        for payload in (status, info, recommendation, resolved_types, history_decision)
    ):
        return False
    selected_features = recommendation.get("features")
    if (
        not isinstance(selected_features, list)
        or recommended_features != [str(value).upper() for value in selected_features]
        or any(feature not in resolved_types for feature in recommended_features)
        or any(
            str(resolved_types[feature]).lower() not in {"continuous", "discrete"}
            for feature in recommended_features
        )
    ):
        return False
    diagnostics_dir = OUTPUT_DIR / "diagnostics"
    try:
        diagnostics_manifest = json.loads(
            (diagnostics_dir / "logistic_diagnostics_manifest.json").read_text(
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
    split = info.get("outer_split_settings", {})
    pool = _read_pool()
    _read_feature_types(pool)
    return (
        status.get("status") == "COMPLETE"
        and status.get("final_holdout_rows_scored") == 0
        and recommendation.get("status") == "COMPLETE"
        and recommendation.get("features") == info.get("features")
        and recommendation.get("C") == info.get("C")
        and recommendation.get("selected_history") == info.get("selected_history")
        and history_decision.get("selected_history") == info.get("selected_history")
        and info.get("selected_history") in LR_HISTORY_GRID
        and status.get("selected_history") == info.get("selected_history")
        and info.get("schema_version") == LR_SCHEMA_VERSION
        and info.get("profile_hash") == PROFILE_HASH
        and info.get("profile") == PROFILE_PAYLOAD
        and info.get("mode") == LR_MODE
        and info.get("sklearn_version") == sklearn.__version__
        and info.get("outer_fold_definitions", {}).get("sha256")
        == _sha256(OUTER_FOLD_DEFINITIONS)
        and info.get("inner_fold_definitions", {}).get("sha256")
        == _sha256(INNER_FOLD_DEFINITIONS)
        and info.get("data", {}).get("sha256") == _sha256(DATA)
        and info.get("manifest", {}).get("sha256") == _sha256(MANIFEST)
        and info.get("pool_file", {}).get("sha256") == _sha256(POOL)
        and info.get("script", {}).get("sha256") == _sha256(SCRIPT)
        and info.get("logistic_utils", {}).get("sha256") == _sha256(SUPPORT)
        and info.get("modeling_utils", {}).get("sha256")
        == _sha256(MODELING_UTILS)
        and info.get("model_diagnostics", {}).get("sha256")
        == _sha256(MODEL_DIAGNOSTICS)
        and info.get("feature_routing", {}).get("sha256")
        == _sha256(FEATURE_ROUTING)
        and info.get("state_grouping", {}).get("sha256")
        == _sha256(STATE_GROUPING)
        and info.get("feature_types_file", {}).get("sha256")
        == _sha256(FEATURE_TYPES)
        and info.get("routing_audit", {}).get("sha256")
        == _sha256(ROUTING_AUDIT)
        and info.get("preparation_info", {}).get("sha256")
        == _sha256(PREPARATION_INFO)
        and info.get("expected_feature_count") == len(pool)
        and info.get("model_feature_count")
        == len(pool) + len(LR_MANDATORY_CONTROL_FEATURES)
        and info.get("mandatory_control_features")
        == LR_MANDATORY_CONTROL_FEATURES
        and info.get("protected_features") == LR_MANDATORY_CONTROL_FEATURES
        and info.get("max_iter_ridge") == LR_MAX_ITER_RIDGE
        and info.get("fixed_selection_c") == LR_FIXED_C
        and info.get("ridge_c_grid") == list(_selector_c_grid())
        and info.get("history_grid") == LR_HISTORY_GRID
        and info.get("history_selection_policy")
        == LR_HISTORY_SELECTION_POLICY
        and info.get("sampled_development_folds")
        == INNER_VALIDATION_FOLDS
        and info.get("sample_train_max_rows") == LR_TRAIN_MAX_ROWS
        and info.get("sample_validation_max_rows")
        == LR_VALIDATION_MAX_ROWS
        and info.get("minimum_stable_folds")
        == min(3, INNER_VALIDATION_FOLDS)
        and info.get("tolerances")
        == {
            "minimum_addition_ap_skill_gain": LR_AP_STEP_TOL,
            "full_row_ap_skill_noninferiority": LR_AP_TOTAL_TOL,
            "ridge_c_ap_skill_noninferiority": LR_AP_STEP_TOL,
            "lift10_step_loss": LR_LIFT_STEP_TOL,
            "lift10_total_loss_vs_intercept": LR_LIFT_TOTAL_TOL,
            "full_row_lift10_noninferiority": LR_LIFT_TOTAL_TOL,
            "minimum_floating_removal_ap_skill_gain": (
                LR_FLOATING_REMOVAL_AP_GAIN
            ),
        }
        and split
        == {
            "folds": ROLLING_VALIDATION_FOLDS,
            "validation_months": OUTER_VALIDATION_MONTHS,
            "holdout_months": FINAL_HOLDOUT_MONTHS,
            "minimum_train_months": MIN_OUTER_TRAIN_MONTHS,
            "label_availability_buffer_days": LABEL_AVAILABILITY_BUFFER_DAYS,
            "strict_dataset_contract": STRICT_DATASET_CONTRACT,
            "seed": SEED,
        }
        and info.get("development_split_settings")
        == {
            "folds": INNER_VALIDATION_FOLDS,
            "validation_months": INNER_VALIDATION_MONTHS,
            "minimum_train_months": MIN_INNER_TRAIN_MONTHS,
            "label_availability_buffer_days": LABEL_AVAILABILITY_BUFFER_DAYS,
            "seed": SEED,
        }
    )




def main() -> None:
    required = [
        DATA,
        MANIFEST,
        POOL,
        FEATURE_TYPES,
        ROUTING_AUDIT,
        PREPARATION_INFO,
        OUTER_FOLD_DEFINITIONS,
        INNER_FOLD_DEFINITIONS,
        SCRIPT,
        SUPPORT,
        MODELING_UTILS,
        MODEL_DIAGNOSTICS,
        FEATURE_ROUTING,
        STATE_GROUPING,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required Logistic file(s) missing:\n"
            + "\n".join(f"- {path}" for path in missing)
        )


    _verify_preparation_profile()
    _migrate_lift_spread_checkpoint_config()


    if _can_reuse():
        print(f"LOGISTIC REGRESSION ALREADY COMPLETE; REUSING {OUTPUT_DIR}")
        return


    pool = _read_pool()
    _read_feature_types(pool)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(profile_env())
    env.update(
        {
            "RETENTION_LR_CODE_DIR": str(CODE_DIR),
            "RETENTION_LR_DATA": str(DATA),
            "RETENTION_LR_MANIFEST": str(MANIFEST),
            "RETENTION_LR_POOL": str(POOL),
            "RETENTION_LR_FEATURE_TYPES": str(FEATURE_TYPES),
            "RETENTION_FEATURE_ROUTING_AUDIT": str(ROUTING_AUDIT),
            "RETENTION_LR_PREPARATION_INFO": str(PREPARATION_INFO),
            "RETENTION_LR_OUTPUT_DIR": str(OUTPUT_DIR),
            "RETENTION_LR_SEED": str(SEED),
            "RETENTION_LR_PRIMARY_METRIC": "ap_skill",
            "RETENTION_LR_TRAIN_MAX_ROWS": str(LR_TRAIN_MAX_ROWS),
            "RETENTION_LR_VALIDATION_MAX_ROWS": str(LR_VALIDATION_MAX_ROWS),
            "RETENTION_LR_MAX_ITER_RIDGE": str(LR_MAX_ITER_RIDGE),
            "RETENTION_LR_EXPECTED_FEATURE_COUNT": str(len(pool)),
            "RETENTION_LR_SCHEMA_VERSION": str(LR_SCHEMA_VERSION),
            "RETENTION_LR_MODE": LR_MODE,
            "RETENTION_LR_FIXED_C": str(LR_FIXED_C),
            "RETENTION_LR_HISTORY_GRID_JSON": json.dumps(
                LR_HISTORY_GRID, separators=(",", ":")
            ),
            "RETENTION_LR_HISTORY_SELECTION_POLICY_JSON": json.dumps(
                LR_HISTORY_SELECTION_POLICY, separators=(",", ":")
            ),
            "RETENTION_LR_MAX_ACTIONS": "1000",
            "RETENTION_LR_MIN_STABLE_FOLDS": str(
                min(3, INNER_VALIDATION_FOLDS)
            ),
            "RETENTION_LR_AP_STEP_TOL": str(LR_AP_STEP_TOL),
            "RETENTION_LR_AP_TOTAL_TOL": str(LR_AP_TOTAL_TOL),
            "RETENTION_LR_LIFT_STEP_TOL": str(LR_LIFT_STEP_TOL),
            "RETENTION_LR_LIFT_TOTAL_TOL": str(LR_LIFT_TOTAL_TOL),
            "RETENTION_LR_FLOATING_REMOVAL_AP_GAIN": str(
                LR_FLOATING_REMOVAL_AP_GAIN
            ),
            **_budget_environment(),
            
            "RETENTION_LR_PROTECTED_FEATURES_FILE": "",
            "RETENTION_LR_OUTER_FOLDS": str(ROLLING_VALIDATION_FOLDS),
            "RETENTION_LR_OUTER_VALIDATION_MONTHS": str(
                OUTER_VALIDATION_MONTHS
            ),
            "RETENTION_LR_HOLDOUT_MONTHS": str(FINAL_HOLDOUT_MONTHS),
            "RETENTION_LR_MIN_OUTER_TRAIN_MONTHS": str(MIN_OUTER_TRAIN_MONTHS),
            "RETENTION_LR_LABEL_AVAILABILITY_BUFFER_DAYS": str(
                LABEL_AVAILABILITY_BUFFER_DAYS
            ),
            "RETENTION_LR_STRICT_DATASET_CONTRACT": (
                "1" if STRICT_DATASET_CONTRACT else "0"
            ),
            "RETENTION_LR_OUTER_FOLD_DEFINITIONS": str(
                OUTER_FOLD_DEFINITIONS
            ),
            "RETENTION_LR_INNER_FOLDS": str(INNER_VALIDATION_FOLDS),
            "RETENTION_LR_INNER_VALIDATION_MONTHS": str(
                INNER_VALIDATION_MONTHS
            ),
            "RETENTION_LR_MIN_INNER_TRAIN_MONTHS": str(
                MIN_INNER_TRAIN_MONTHS
            ),
            "RETENTION_LR_INNER_FOLD_DEFINITIONS": str(
                INNER_FOLD_DEFINITIONS
            ),
        }
    )
    print("LOGISTIC REGRESSION - FLOATING-FORWARD FEATURE SELECTION")
    print(f"Selectable pool after routing/quality screen: {len(pool)} features")
    print(
        "Mandatory model control: "
        + ", ".join(LR_MANDATORY_CONTROL_FEATURES)
        + f"; total model groups: {len(pool) + len(LR_MANDATORY_CONTROL_FEATURES)}"
    )
    print("Primary metric: churn average precision; guardrail: Lift@10%")
    print("Final sealed holdout will be scored: NO")
    result = subprocess.run(
        [sys.executable, "-u", str(SCRIPT)],
        cwd=str(PROJECT_ROOT),
        env=env,
        check=False,
    )
    if result.returncode == 75:
        # Preserve the engine's safe-pause signal across the wrapper process so
        # c0_one_click can retain and resume the same pipeline run ID.
        raise SystemExit(75)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, result.args)




if __name__ == "__main__":
    main()
