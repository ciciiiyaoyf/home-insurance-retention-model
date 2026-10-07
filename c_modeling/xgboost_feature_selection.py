#!/usr/bin/env python
"""Select an XGBoost raw-feature group from the current eligible universe.




This Home 13/1 retention workflow starts from the candidate pool and temporal
folds created by the modeling-preparation stage. It does not read a previously
selected feature group or manually constructed feature family.




The workflow is:




1. Read the current modeling parquet and current production feature manifest.
2. Build the complete eligible candidate universe from the manifest
   and the columns actually present in the parquet.
3. Apply the established training-only quality gates from ``modeling_utils.py``.
4. Use only the oldest outer-fold training population for feature discovery.
5. Tune XGBoost with a deterministic Optuna TPE study inside the chronological
   inner-fold boundary, with churn as the positive class and AUCPR as the
   early-stopping metric. A small set of established configurations is enqueued
   as coverage anchors before adaptive trials begin.
6. Fit the tuned XGBoost model on chronological inner folds and rank raw
   features by grouped total gain averaged across those folds. Every
   categorical feature and all of its one-hot columns remain one raw group.
7. Run a coarse cumulative forward scan over nested top-gain prefixes, then
   iteratively test every intermediate feature count immediately below the
   smallest discovery-qualified prefix until no local count remains untested.
8. Confirm every required eligible or AP/recall Pareto-relevant prefix across
   multiple XGBoost seeds. If the full universe leads, adaptively confirm
   intermediate coarse and exact neighboring prefixes without binary search.
9. Retune the shortlisted feature sets; choose the smallest final set within
   explicit practical AP-skill/capture losses and per-fold deterioration limits.
   Discovery and configuration selection retain their paired one-SE rules.
10. Write one explicit XGBoost-selected feature-group text file and JSON file.




No outer validation fold is scored and the designated final six-month holdout
is never scored. This file creates the XGBoost feature group only; model-family
comparison should be a later, separate step.
"""




from __future__ import annotations




import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence




import numpy as np
import pandas as pd
from scipy import sparse




try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover - user environment dependency
    raise ImportError(
        "xgboost is not installed. Run: python -m pip install 'xgboost>=2.1,<4'"
    ) from exc


try:
    import optuna
except ImportError as exc:  # pragma: no cover - user environment dependency
    raise ImportError(
        "optuna is not installed. Run: python -m pip install 'optuna>=4,<5'"
    ) from exc


try:
    import shap
except ImportError as exc:  # pragma: no cover - user environment dependency
    raise ImportError(
        "shap is not installed. Run: python -m pip install 'shap>=0.46,<1'"
    ) from exc








PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))




from func.tenure_profile import (  # noqa: E402
    PROFILE_HASH_ENV,
    profile_from_environment,
)
from c_modeling.xgboost_selection_policy import (  # noqa: E402
    DEFAULT_POLICY,
    FINAL_FEATURE_SELECTION_RULE,
    policy_sha256,
    select_practical_feature_group,
    validated_policy,
)
from c_modeling.model_diagnostics import (  # noqa: E402
    ACTUAL_COLUMN,
    WITH_YEAR_COLUMN,
    YEAR_CONTROL_FEATURE,
    YEAR_MONTH_COLUMN,
    aggregate_raw_shap,
    business_retention_monthly_summary,
    write_business_retention_trend_artifacts,
    write_calendar_control_artifacts,
    write_xgboost_shap_artifacts,
)








DEFAULT_SEEDS = (42, 2026, 31415)
STAGE4_SELECTION_CONTRACT_VERSION = 5
STAGE4_SELECTION_RULE = "PAIRED_FOLD_ONE_STANDARD_ERROR_AP_SKILL_RECALL_10PCT"
STAGE4_SELECTION_CHECKPOINT = "xgb_stage4_selection_checkpoint.json"
FINALIST_TUNING_CONTRACT_VERSION = 3
FINALIST_TUNING_TIMEOUT_SECONDS = 0
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
BASE_SEED = int(os.environ.get("RETENTION_RANDOM_SEED", "42"))
PROFILE_PAYLOAD = profile_from_environment().payload
PROFILE_HASH = os.environ.get(PROFILE_HASH_ENV, "").strip()
if not PROFILE_HASH:
    raise RuntimeError(f"{PROFILE_HASH_ENV} is required for run isolation.")
METRICS = (
    "retention_roc_auc",
    "churn_average_precision",
    "top10_lift",
    "churn_precision_10pct",
    "churn_capture_10pct",
    "churn_recall_10pct",
    "log_loss",
    "brier_score",
    "calibration_gap",
)
LIFT_SPREAD_BIN_COUNTS = (3, 4, 5, 10, 20)
LIFT_SPREAD_RATIO_METRICS = tuple(
    f"highest_divided_by_lowest_lift_{bins}bins"
    for bins in LIFT_SPREAD_BIN_COUNTS
)
METRICS = (*METRICS, *LIFT_SPREAD_RATIO_METRICS)








def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--modeling-utils",
        type=Path,
        default=Path("c_modeling/modeling_utils.py"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Explicit population snapshot for the active run; no population fallback.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Explicit feature contract belonging to the supplied population snapshot.",
    )
    parser.add_argument(
        "--starting-feature-pool",
        type=Path,
        default=None,
        help=(
            "Training-screened raw-feature pool from modeling preparation. "
            "When supplied, "
            "XGBoost validates and uses it instead of independently screening a "
            "different starting universe."
        ),
    )
    parser.add_argument("--preparation-info", type=Path, required=True)
    parser.add_argument("--outer-fold-definitions", type=Path, required=True)
    parser.add_argument("--inner-fold-definitions", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--n-jobs", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )
    parser.add_argument(
        "--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS)
    )




    # Existing validated temporal design. The outer folds define the discovery
    # boundary only; they are not scored by this script.
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-validation-months", type=int, default=3)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--min-outer-train-months", type=int, default=12)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--inner-validation-months", type=int, default=3)
    parser.add_argument("--min-inner-train-months", type=int, default=12)




    # Training-only quality screening.
    parser.add_argument("--quality-screen-max-rows", type=int, default=100_000)
    parser.add_argument("--missing-drop-threshold", type=float, default=0.95)
    parser.add_argument(
        "--dominant-level-drop-threshold", type=float, default=0.995
    )
    parser.add_argument("--high-cardinality-ratio", type=float, default=0.50)
    parser.add_argument("--max-category-levels", type=int, default=500)




    # Configuration tuning on the full audited feature universe. Optuna trials
    # are sequential; each XGBoost fit already uses --n-jobs internally.
    parser.add_argument("--config-screen-train-max-rows", type=int, default=300_000)
    parser.add_argument(
        "--config-confirm-train-max-rows", type=int, default=600_000
    )
    parser.add_argument("--validation-max-rows", type=int, default=80_000)
    parser.add_argument("--confirm-top-configs", type=int, default=4)
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=8,
        help=(
            "Adaptive Optuna trials added after the fixed coverage anchors. "
            "The anchors guarantee that every allowed history treatment is "
            "screened before TPE exploration begins."
        ),
    )
    parser.add_argument(
        "--optuna-batch-size",
        type=int,
        default=8,
        help=(
            "Sequential ask/tell batch size. Trials in a batch are grouped by "
            "history so fold-local preprocessing can be reused."
        ),
    )
    parser.add_argument(
        "--optuna-timeout-seconds",
        type=int,
        default=0,
        help=(
            "Optional wall-clock limit for adaptive trials only; 0 disables "
            "the limit. Coverage anchors always finish."
        ),
    )




    # Gain-ranked cumulative forward selection. The configured step controls
    # the coarse scan; the engine then refines every count immediately below
    # the smallest eligible coarse prefix so the final cutoff is not restricted
    # to multiples of this value.
    parser.add_argument("--forward-train-max-rows", type=int, default=500_000)
    parser.add_argument(
        "--forward-validation-max-rows", type=int, default=60_000
    )
    parser.add_argument("--min-features", type=int, default=8)
    parser.add_argument("--forward-step-size", type=int, default=4)




    # Multi-seed confirmation of the algorithm-generated finalist neighborhood.
    # Required candidates may span multiple passes and are never truncated.
    parser.add_argument("--confirm-train-max-rows", type=int, default=1_000_000)
    parser.add_argument("--confirm-candidate-count", type=int, default=4)
    parser.add_argument(
        "--confirm-max-passes",
        type=int,
        default=5,
        help=(
            "Maximum adaptive Stage-4 confirmation passes. Each pass evaluates "
            "at most --confirm-candidate-count new groups. If required or "
            "full-winner intermediate prefixes remain, selection fails closed."
        ),
    )




    parser.add_argument("--max-rounds", type=int, default=1_800)
    parser.add_argument("--early-stopping-rounds", type=int, default=75)
    parser.add_argument("--rare-min-frequency", type=float, default=0.001)




    # Discovery tolerances. These control the coarse/local Stage-3 search and
    # its required confirmation inventory. The final feature-count decision
    # has separate practical-loss limits, recorded in the selection receipt.
    parser.add_argument("--ap-skill-tolerance", type=float, default=0.001)
    parser.add_argument("--lift-tolerance", type=float, default=0.005)
    parser.add_argument("--auc-tolerance", type=float, default=0.001)
    parser.add_argument("--recent-ap-skill-tolerance", type=float, default=0.003)
    parser.add_argument("--final-ap-skill-loss", type=float, default=DEFAULT_POLICY["mean_ap_skill_loss"])
    parser.add_argument("--final-recall-loss", type=float, default=DEFAULT_POLICY["mean_recall_loss"])
    parser.add_argument("--final-max-fold-ap-skill-loss", type=float, default=DEFAULT_POLICY["max_fold_ap_skill_loss"])
    parser.add_argument("--final-max-fold-recall-loss", type=float, default=DEFAULT_POLICY["max_fold_recall_loss"])
    parser.add_argument("--allow-contract-violations", action="store_true")
    parser.add_argument(
        "--resume-post-selection",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()




def final_selection_policy(args: argparse.Namespace) -> dict[str, Any]:
    return validated_policy({
        "mean_ap_skill_loss": args.final_ap_skill_loss,
        "mean_recall_loss": args.final_recall_loss,
        "max_fold_ap_skill_loss": args.final_max_fold_ap_skill_loss,
        "max_fold_recall_loss": args.final_max_fold_recall_loss,
    })








def resolve(root: Path, value: Path) -> Path:
    return value.expanduser().resolve() if value.is_absolute() else (root / value).resolve()








def version_tuple(text: str) -> tuple[int, int, int]:
    values = [int(piece) for piece in re.findall(r"\d+", text)[:3]]
    while len(values) < 3:
        values.append(0)
    return tuple(values)  # type: ignore[return-value]








def validate_xgboost_version() -> None:
    if version_tuple(xgb.__version__) < (2, 1, 0):
        raise RuntimeError(
            f"XGBoost {xgb.__version__} is installed; this workflow requires >=2.1. "
            "Run: python -m pip install --upgrade 'xgboost>=2.1,<4'"
        )




def validate_optuna_version() -> None:
    if version_tuple(optuna.__version__) < (4, 0, 0):
        raise RuntimeError(
            f"Optuna {optuna.__version__} is installed; this workflow requires >=4.0. "
            "Run: python -m pip install --upgrade 'optuna>=4,<5'"
        )




def validate_shap_version() -> None:
    if version_tuple(shap.__version__) < (0, 46, 0):
        raise RuntimeError(
            f"SHAP {shap.__version__} is installed; this workflow requires >=0.46. "
            "Run: python -m pip install --upgrade 'shap>=0.46,<1'"
        )




def tree_shap_values(
    model: Any,
    matrix: sparse.spmatrix | np.ndarray,
    *,
    best_iteration: int,
) -> np.ndarray:
    """Return positive-class raw-margin values from ``shap.TreeExplainer``."""


    booster = model.get_booster()
    limited_booster = booster[: int(best_iteration) + 1]
    explainer = shap.TreeExplainer(limited_booster, model_output="raw")
    values = explainer.shap_values(
        matrix,
        check_additivity=False,
    )
    if isinstance(values, list):
        if not values:
            raise RuntimeError("SHAP TreeExplainer returned no class values.")
        values = values[-1]
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        array = array[:, :, -1]
    if array.ndim != 2 or array.shape[1] != matrix.shape[1]:
        raise RuntimeError(
            "Unexpected SHAP TreeExplainer output shape: "
            f"{array.shape}; expected ({matrix.shape[0]}, {matrix.shape[1]})."
        )
    return array








def parse_integer_list(text: str, name: str) -> list[int]:
    values = [int(piece.strip()) for piece in text.split(",") if piece.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one integer.")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate values.")
    return values








def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()




def feature_sequence_sha256(features: Sequence[str]) -> str:
    """Hash an ordered raw-feature contract with an unambiguous terminator."""


    normalized = [str(value) for value in features]
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError("Feature sequence must be non-empty and unique.")
    return hashlib.sha256(
        ("\n".join(normalized) + "\n").encode("utf-8")
    ).hexdigest()








def file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "modified_epoch_seconds": float(stat.st_mtime),
        "sha256": sha256(path),
    }








def strict_json_value(value: Any) -> Any:
    """Convert nested run metadata to standards-compliant JSON values."""


    if value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Mapping):
        return {str(key): strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [strict_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [strict_json_value(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value








def read_xgboost_feature_contract(
    pool: Sequence[str],
    feature_types_path: Path,
    routing_audit_path: Path,
) -> dict[str, str]:
    """Verify C1's XGBoost type and kept-routing artifacts."""




    try:
        raw_types = json.loads(feature_types_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid XGBoost family feature-types artifact.") from exc
    if not isinstance(raw_types, dict):
        raise ValueError("XGBoost family feature-types artifact must be an object.")
    feature_types = {
        str(feature).strip().upper(): str(var_type).strip().lower()
        for feature, var_type in raw_types.items()
    }
    expected = {str(feature).strip().upper() for feature in pool}
    if set(feature_types) != expected:
        raise ValueError("XGBoost pool and family feature-types artifact disagree.")
    if any(value not in {"continuous", "discrete"} for value in feature_types.values()):
        raise ValueError("XGBoost family feature-types artifact has invalid types.")
    audit = pd.read_csv(routing_audit_path, dtype=str, keep_default_na=False)
    required = {
        "model_family",
        "selected_feature",
        "quality_screen_decision",
    }
    if not required.issubset(audit.columns):
        raise ValueError("Feature-routing audit is missing required columns.")
    kept = audit.loc[
        audit["model_family"].str.strip().str.lower().eq("xgboost")
        & audit["quality_screen_decision"].str.strip().str.upper().eq("KEEP"),
        "selected_feature",
    ].str.strip().str.upper().tolist()
    if len(kept) != len(set(kept)) or set(kept) != expected:
        raise ValueError("XGBoost pool and kept feature-routing audit disagree.")
    return feature_types








def verify_preparation_info(
    path: Path,
    *,
    input_path: Path,
    manifest_path: Path,
    modeling_utils_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Fail before fitting when C1 and XGBoost inputs/splits disagree."""




    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid modeling-preparation JSON: {path}") from exc
    expected_hashes = {
        "input_sha256": sha256(input_path),
        "feature_contract_sha256": sha256(manifest_path),
        "modeling_utils_sha256": sha256(modeling_utils_path),
        "feature_routing_sha256": sha256(
            modeling_utils_path.parent / "feature_routing.py"
        ),
    }
    hash_mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected_hashes.items()
        if payload.get(key) != value
    }
    settings = payload.get("split_settings", {})
    expected_settings = {
        "seed": BASE_SEED,
        "rolling_validation_folds": args.outer_folds,
        "outer_validation_months": args.outer_validation_months,
        "final_holdout_months": args.test_months,
        "min_outer_train_months": args.min_outer_train_months,
        "inner_validation_folds": args.inner_folds,
        "inner_validation_months": args.inner_validation_months,
        "min_inner_train_months": args.min_inner_train_months,
        "strict_dataset_contract": not args.allow_contract_violations,
    }
    setting_mismatches = {
        key: {"expected": value, "observed": settings.get(key)}
        for key, value in expected_settings.items()
        if settings.get(key) != value
    }
    profile_mismatches = {}
    if payload.get("profile_hash") != PROFILE_HASH:
        profile_mismatches["profile_hash"] = {
            "expected": PROFILE_HASH,
            "observed": payload.get("profile_hash"),
        }
    if payload.get("profile") != PROFILE_PAYLOAD:
        profile_mismatches["profile"] = {
            "expected": PROFILE_PAYLOAD,
            "observed": payload.get("profile"),
        }
    if hash_mismatches or setting_mismatches or profile_mismatches:
        raise ValueError(
            "XGBoost inputs/splits differ from the modeling-preparation contract: "
            f"hashes={hash_mismatches}; settings={setting_mismatches}; "
            f"profile={profile_mismatches}"
        )
    return payload








def anchor_model_grid() -> list[dict[str, Any]]:
    """Established configurations used only to anchor the Optuna search."""




    return [
        {
            "model_id": "DEPTH3_REGULARIZED",
            "max_depth": 3,
            "learning_rate": 0.05,
            "min_child_weight": 30.0,
            "gamma": 0.00,
            "subsample": 0.85,
            "colsample_bytree": 0.80,
            "reg_lambda": 20.0,
            "reg_alpha": 0.10,
            "grow_policy": "depthwise",
            "max_leaves": 0,
            "max_bin": 256,
        },
        {
            "model_id": "DEPTH5_REGULARIZED",
            "max_depth": 5,
            "learning_rate": 0.03,
            "min_child_weight": 40.0,
            "gamma": 0.05,
            "subsample": 0.80,
            "colsample_bytree": 0.75,
            "reg_lambda": 25.0,
            "reg_alpha": 0.50,
            "grow_policy": "depthwise",
            "max_leaves": 0,
            "max_bin": 256,
        },
        {
            "model_id": "DEPTH6_STRONG_REG",
            "max_depth": 6,
            "learning_rate": 0.03,
            "min_child_weight": 80.0,
            "gamma": 0.10,
            "subsample": 0.80,
            "colsample_bytree": 0.70,
            "reg_lambda": 50.0,
            "reg_alpha": 1.00,
            "grow_policy": "depthwise",
            "max_leaves": 0,
            "max_bin": 256,
        },
        {
            "model_id": "LOSSGUIDE_64_REG",
            "max_depth": 8,
            "learning_rate": 0.03,
            "min_child_weight": 50.0,
            "gamma": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.80,
            "reg_lambda": 30.0,
            "reg_alpha": 0.50,
            "grow_policy": "lossguide",
            "max_leaves": 64,
            "max_bin": 256,
        },
    ]




def optuna_search_space() -> dict[str, Any]:
    """Auditable search bounds for the adaptive XGBoost trials."""


    return {
        "max_depth": {"type": "int", "low": 2, "high": 8},
        "learning_rate": {
            "type": "float",
            "low": 0.01,
            "high": 0.10,
            "log": True,
        },
        "min_child_weight": {
            "type": "float",
            "low": 10.0,
            "high": 150.0,
            "log": True,
        },
        "gamma": {"type": "float", "low": 0.0, "high": 1.0},
        "subsample": {"type": "float", "low": 0.65, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.60, "high": 1.0},
        "reg_lambda": {
            "type": "float",
            "low": 1.0,
            "high": 100.0,
            "log": True,
        },
        "reg_alpha": {
            "type": "float",
            "low": 1.0e-4,
            "high": 5.0,
            "log": True,
        },
        "grow_policy": {
            "type": "categorical",
            "choices": ["depthwise", "lossguide"],
        },
        "max_leaves": {
            "type": "conditional_int",
            "condition": {"grow_policy": "lossguide"},
            "low": 16,
            "high": 128,
            "step": 16,
        },
        "max_bin": {"type": "fixed", "value": 256},
    }




def enqueue_optuna_anchor_trials(
    study: Any,
    histories: Sequence[Mapping[str, Any]],
) -> int:
    """Queue the previous compact grid as deterministic coverage anchors."""


    count = 0
    for history in histories:
        history_id = str(history["history_id"])
        for model in anchor_model_grid():
            grow_policy = str(model.get("grow_policy", "depthwise"))
            params: dict[str, Any] = {
                "history_id": history_id,
                "max_depth": int(model["max_depth"]),
                "learning_rate": float(model["learning_rate"]),
                "min_child_weight": float(model["min_child_weight"]),
                "gamma": float(model["gamma"]),
                "subsample": float(model["subsample"]),
                "colsample_bytree": float(model["colsample_bytree"]),
                "reg_lambda": float(model["reg_lambda"]),
                "reg_alpha": float(model["reg_alpha"]),
                "grow_policy": grow_policy,
            }
            if grow_policy == "lossguide":
                params["max_leaves"] = int(model.get("max_leaves", 64))
            study.enqueue_trial(
                params,
                user_attrs={
                    "trial_origin": "COVERAGE_ANCHOR",
                    "anchor_model_id": str(model["model_id"]),
                },
            )
            count += 1
    return count




def suggest_optuna_configuration(
    trial: Any,
    history_lookup: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Materialize one history/model configuration from an Optuna trial."""


    if not history_lookup:
        raise ValueError("Optuna requires at least one history candidate.")
    history_id = str(
        trial.suggest_categorical("history_id", list(history_lookup))
    )
    grow_policy = str(
        trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    )
    max_leaves = (
        int(trial.suggest_int("max_leaves", 16, 128, step=16))
        if grow_policy == "lossguide"
        else 0
    )
    model = {
        "model_id": f"OPTUNA_T{int(trial.number):04d}",
        "max_depth": int(trial.suggest_int("max_depth", 2, 8)),
        "learning_rate": float(
            trial.suggest_float("learning_rate", 0.01, 0.10, log=True)
        ),
        "min_child_weight": float(
            trial.suggest_float("min_child_weight", 10.0, 150.0, log=True)
        ),
        "gamma": float(trial.suggest_float("gamma", 0.0, 1.0)),
        "subsample": float(trial.suggest_float("subsample", 0.65, 1.0)),
        "colsample_bytree": float(
            trial.suggest_float("colsample_bytree", 0.60, 1.0)
        ),
        "reg_lambda": float(
            trial.suggest_float("reg_lambda", 1.0, 100.0, log=True)
        ),
        "reg_alpha": float(
            trial.suggest_float("reg_alpha", 1.0e-4, 5.0, log=True)
        ),
        "grow_policy": grow_policy,
        "max_leaves": max_leaves,
        "max_bin": 256,
    }
    return {
        "history": dict(history_lookup[history_id]),
        "model": model,
    }








def history_grid() -> list[dict[str, Any]]:
    return [
        {"history_id": "RECENT_36M", "kind": "recent", "months": 36},
        {"history_id": "RECENT_48M", "kind": "recent", "months": 48},
        {"history_id": "RECENT_60M", "kind": "recent", "months": 60},
    ]








def history_window_months(strategy: Mapping[str, Any]) -> int:
    """Return the hard training-history cap for a supported strategy."""


    kind = str(strategy.get("kind", "")).strip().lower()
    if kind != "recent":
        raise ValueError(
            "Unsupported XGBoost history strategy; only equal-weight "
            f"RECENT_36M/48M/60M is allowed: {dict(strategy)}"
        )
    field = "months"
    if field not in strategy:
        raise ValueError(
            f"XGBoost {kind} history strategy is missing required {field!r}: "
            f"{dict(strategy)}"
        )
    months = int(strategy[field])
    if months <= 0:
        raise ValueError(
            f"XGBoost history window must be positive, observed {months}."
        )
    return months








def make_model(
    spec: Mapping[str, Any],
    *,
    n_estimators: int,
    early_stopping_rounds: int | None,
    device: str,
    n_jobs: int,
    seed: int,
) -> xgb.XGBClassifier:
    """Train churn directly so AUCPR early stopping targets the business class."""




    kwargs: dict[str, Any] = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "tree_method": "hist",
        "device": device,
        "n_estimators": int(n_estimators),
        "max_depth": int(spec["max_depth"]),
        "learning_rate": float(spec["learning_rate"]),
        "min_child_weight": float(spec["min_child_weight"]),
        "gamma": float(spec["gamma"]),
        "subsample": float(spec["subsample"]),
        "colsample_bytree": float(spec["colsample_bytree"]),
        "reg_lambda": float(spec["reg_lambda"]),
        "reg_alpha": float(spec["reg_alpha"]),
        "grow_policy": str(spec.get("grow_policy", "depthwise")),
        "max_leaves": int(spec.get("max_leaves", 0)),
        "max_bin": int(spec.get("max_bin", 256)),
        "eval_metric": "aucpr",
        "random_state": int(seed),
        "n_jobs": int(n_jobs),
        "verbosity": 0,
    }
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
    return xgb.XGBClassifier(**kwargs)








def preprocessing_spec(modeling_utils: Any, rare_min_frequency: float) -> Any:
    # ModelSpec controls fold-local preprocessing here; its logistic fields are
    # not used to fit the XGBoost model.
    return modeling_utils.ModelSpec(
        c_value=1.0,
        max_iter=1_000,
        solver="lbfgs",
        random_seed=BASE_SEED,
        rare_min_frequency=float(rare_min_frequency),
        fail_on_nonconvergence=False,
    )








def oldest_complete_train(
    frame: pd.DataFrame,
    outer_fold: Mapping[str, Any],
    cohort_col: str,
) -> pd.DataFrame:
    train_idx = np.asarray(outer_fold["train_idx"])
    month = pd.to_datetime(frame[cohort_col], errors="raise").dt.to_period("M")
    in_train = pd.Series(frame.index.isin(train_idx), index=frame.index)
    complete = in_train.groupby(month).all()
    output = frame.loc[month.isin(complete[complete].index)].copy()
    if output.empty or np.setdiff1d(output.index.to_numpy(), train_idx).size:
        raise AssertionError("Invalid oldest-complete-month selection population.")
    return output








def filtered_train_indices(
    modeling_utils: Any,
    frame: pd.DataFrame,
    fold_def: Mapping[str, Any],
    cohort_col: str,
    strategy: Mapping[str, Any],
) -> np.ndarray:
    train_idx = np.asarray(fold_def["train_idx"])
    validation_idx = np.asarray(fold_def["validation_idx"])
    validation_start = pd.to_datetime(
        frame.loc[validation_idx, cohort_col], errors="raise"
    ).min()
    train_dates = pd.to_datetime(frame.loc[train_idx, cohort_col], errors="raise")
    if train_dates.ge(validation_start).any():
        raise ValueError(
            f"Fold {fold_def.get('fold', '?')} contains a training row on or "
            "after validation_start; refusing non-chronological history filtering."
        )
    output, _ = modeling_utils.select_recent_mature_training_indices(
        frame,
        train_idx,
        cohort_col=cohort_col,
        months=history_window_months(strategy),
    )
    if not len(output):
        raise ValueError(
            f"{strategy['history_id']} removed every training row for fold "
            f"{fold_def.get('fold', '?')}."
        )
    return output








def custom_fold_definition(
    modeling_utils: Any,
    frame: pd.DataFrame,
    fold_def: Mapping[str, Any],
    cohort_col: str,
    strategy: Mapping[str, Any],
) -> dict[str, Any]:
    output = dict(fold_def)
    output["train_idx"] = filtered_train_indices(
        modeling_utils, frame, fold_def, cohort_col, strategy
    )
    output["validation_idx"] = np.asarray(fold_def["validation_idx"])
    return output








def prepare_fold(
    *,
    modeling_utils: Any,
    frame: pd.DataFrame,
    fold_def: Mapping[str, Any],
    feature_types: Mapping[str, str],
    config: Any,
    strategy: Mapping[str, Any],
    train_max_rows: int | None,
    validation_max_rows: int | None,
    rare_min_frequency: float,
) -> Any:
    custom = custom_fold_definition(
        modeling_utils, frame, fold_def, config.cohort_date_col, strategy
    )
    spec = preprocessing_spec(modeling_utils, rare_min_frequency)
    if train_max_rows is None and validation_max_rows is None:
        return modeling_utils.PreparedFold(
            int(custom.get("fold", 1)),
            frame.loc[np.asarray(custom["train_idx"])],
            frame.loc[np.asarray(custom["validation_idx"])],
            feature_types,
            spec,
            target_col=config.target_col,
            cohort_col=config.cohort_date_col,
        )
    return modeling_utils.prepare_fold_objects(
        frame,
        [custom],
        feature_types,
        spec,
        train_max_rows=train_max_rows,
        validation_max_rows=validation_max_rows,
        cohort_col=config.cohort_date_col,
        target_col=config.target_col,
    )[0]








def training_weights(
    train_frame: pd.DataFrame,
    cohort_col: str,
    strategy: Mapping[str, Any],
) -> np.ndarray | None:
    kind = str(strategy.get("kind", "")).strip().lower()
    if kind != "recent":
        raise ValueError(
            "XGBoost training weights permit only equal-weight recent history: "
            f"{dict(strategy)}"
        )
    history_window_months(strategy)
    del train_frame, cohort_col
    return None








def build_sparse_matrices(
    fold: Any,
    features: Sequence[str],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, list[dict[str, Any]]]:
    train_parts: list[sparse.csr_matrix] = []
    validation_parts: list[sparse.csr_matrix] = []
    ranges: list[dict[str, Any]] = []
    start = 0
    for feature in features:
        block = fold.get_feature_block(feature)
        width = int(block.transformed_columns)
        if width <= 0:
            raise RuntimeError(
                f"{feature} produced {width} transformed columns in fold "
                f"{fold.fold_number}."
            )
        train_parts.append(block.train_matrix)
        validation_parts.append(block.validation_matrix)
        ranges.append(
            {
                "raw_feature": feature,
                "start_column": int(start),
                "end_column_exclusive": int(start + width),
                "transformed_column_count": int(width),
            }
        )
        start += width
    if not train_parts:
        raise ValueError("XGBoost requires at least one raw feature.")
    train_matrix = sparse.hstack(train_parts, format="csr", dtype=np.float32)
    validation_matrix = sparse.hstack(
        validation_parts, format="csr", dtype=np.float32
    )
    if train_matrix.shape[1] != validation_matrix.shape[1]:
        raise RuntimeError("Train and validation transformed widths do not match.")
    return train_matrix, validation_matrix, ranges








def predict_churn_probability(
    model: xgb.XGBClassifier,
    matrix: sparse.csr_matrix,
    best_iteration: int,
) -> np.ndarray:
    try:
        probability = model.predict_proba(
            matrix, iteration_range=(0, int(best_iteration) + 1)
        )[:, 1]
    except TypeError:  # pragma: no cover - compatibility fallback
        probability = model.predict_proba(matrix)[:, 1]
    return np.clip(np.asarray(probability, dtype=float), 1e-7, 1.0 - 1e-7)








def ap_skill(metrics: Mapping[str, Any]) -> float:
    prevalence = float(metrics["churn_prevalence_baseline"])
    average_precision = float(metrics["churn_average_precision"])
    denominator = 1.0 - prevalence
    if denominator <= 0:
        return 0.0
    return (average_precision - prevalence) / denominator








def ensure_recall_alias(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the fixed-capacity recall metric across utility versions."""


    output = dict(metrics)
    if "churn_capture_10pct" not in output:
        raise KeyError(
            "compute_metrics() must return 'churn_capture_10pct' so "
            "'churn_recall_10pct' can be defined consistently."
        )
    output["churn_recall_10pct"] = float(output["churn_capture_10pct"])
    return output








def lift_spread_metrics(
    y_retention: np.ndarray,
    p_churn: np.ndarray,
    *,
    bin_counts: tuple[int, ...] = LIFT_SPREAD_BIN_COUNTS,
    row_id: np.ndarray | None = None,
) -> dict[str, Any]:
    """Report equal-count churn-lift spread without selecting on the ratio."""


    y_churn = 1 - np.asarray(y_retention, dtype=int).reshape(-1)
    scores = np.asarray(p_churn, dtype=float).reshape(-1)
    if y_churn.shape != scores.shape or not len(y_churn):
        raise ValueError("Lift-spread inputs must be non-empty and aligned.")
    if not np.isfinite(scores).all() or not np.isin(y_churn, [0, 1]).all():
        raise ValueError("Lift-spread inputs must be finite with a binary target.")
    tie_breaker = (
        np.arange(len(y_churn), dtype=np.int64)
        if row_id is None
        else np.asarray(row_id).reshape(-1)
    )
    if tie_breaker.shape != y_churn.shape:
        raise ValueError("row_id must align to the lift-spread inputs.")


    ranked = pd.DataFrame(
        {"y": y_churn, "score": scores, "row_id": tie_breaker}
    ).sort_values(
        ["score", "row_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    prevalence = float(ranked["y"].mean())
    output: dict[str, Any] = {}
    for n_bins in bin_counts:
        bins = int(n_bins)
        if bins < 2 or bins > len(ranked) or prevalence <= 0.0:
            highest = lowest = ratio = np.nan
            highest_group = lowest_group = np.nan
            status = "NO_EVENTS_OR_TOO_FEW_ROWS"
            defined = False
        else:
            ranked["bin_rank"] = pd.qcut(
                ranked.index + 1,
                q=bins,
                labels=np.arange(1, bins + 1),
            ).astype(int)
            lifts = ranked.groupby("bin_rank", sort=True)["y"].mean() / prevalence
            highest_group = int(lifts.idxmax())
            lowest_group = int(lifts.idxmin())
            highest = float(lifts.loc[highest_group])
            lowest = float(lifts.loc[lowest_group])
            if lowest == 0.0:
                ratio = np.nan
                status = "ZERO_LOWEST_LIFT"
                defined = False
            else:
                ratio = float(highest / lowest)
                status = "FINITE"
                defined = True
        suffix = f"_{bins}bins"
        output[f"highest_lift{suffix}"] = highest
        output[f"highest_lift_group{suffix}"] = highest_group
        output[f"lowest_lift{suffix}"] = lowest
        output[f"lowest_lift_group{suffix}"] = lowest_group
        output[f"highest_divided_by_lowest_lift{suffix}"] = ratio
        output[f"lift_spread_ratio_defined{suffix}"] = defined
        output[f"lift_spread_ratio_status{suffix}"] = status
    return output








def metric_dict(modeling_utils: Any, fold: Any, p_churn: np.ndarray) -> dict[str, Any]:
    metrics = ensure_recall_alias(
        modeling_utils.compute_metrics(
            fold.validation,
            1.0 - np.asarray(p_churn, dtype=float),
            target_col=fold.target_col,
            policy_id_col=fold.policy_id_col,
            policy_inception_col=fold.policy_inception_col,
        )
    )
    metrics["churn_ap_skill"] = ap_skill(metrics)
    metrics.update(
        lift_spread_metrics(
            fold.validation[fold.target_col].astype(int).to_numpy(),
            np.asarray(p_churn, dtype=float),
            row_id=pd.util.hash_pandas_object(
                fold.validation.index,
                index=False,
            ).to_numpy(dtype=np.uint64),
        )
    )
    return metrics








def booster_score_array(
    booster: xgb.Booster,
    importance_type: str,
    width: int,
) -> np.ndarray:
    raw = booster.get_score(importance_type=importance_type)
    result = np.zeros(width, dtype=float)
    for key, value in raw.items():
        match = re.fullmatch(r"f(\d+)", str(key))
        if not match:
            continue
        position = int(match.group(1))
        if 0 <= position < width:
            result[position] = float(value)
    return result








def aggregate_raw_importance(
    booster: xgb.Booster,
    ranges: Sequence[Mapping[str, Any]],
    *,
    fold_number: int,
) -> pd.DataFrame:
    width = max(int(item["end_column_exclusive"]) for item in ranges)
    gain = booster_score_array(booster, "total_gain", width)
    weight = booster_score_array(booster, "weight", width)
    rows: list[dict[str, Any]] = []
    total_gain = float(gain.sum())
    for item in ranges:
        start = int(item["start_column"])
        end = int(item["end_column_exclusive"])
        feature_gain = float(gain[start:end].sum())
        rows.append(
            {
                "fold": int(fold_number),
                "raw_feature": str(item["raw_feature"]),
                "transformed_column_count": int(
                    item["transformed_column_count"]
                ),
                "total_gain": feature_gain,
                "normalized_total_gain": (
                    feature_gain / total_gain if total_gain > 0 else 0.0
                ),
                "split_count": float(weight[start:end].sum()),
            }
        )
    return pd.DataFrame(rows)








def aggregate_gain_ranking(gain_long: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level grouped gain into one deterministic raw-feature order."""




    required = {
        "raw_feature",
        "transformed_column_count",
        "total_gain",
        "normalized_total_gain",
        "split_count",
    }
    missing = sorted(required.difference(gain_long.columns))
    if missing:
        raise ValueError(f"Gain input is missing required columns: {missing}")
    if gain_long.empty:
        raise ValueError("Cannot rank features from an empty gain table.")




    ranking = (
        gain_long.groupby("raw_feature", as_index=False)
        .agg(
            transformed_column_count=("transformed_column_count", "max"),
            mean_total_gain=("total_gain", "mean"),
            mean_normalized_total_gain=("normalized_total_gain", "mean"),
            median_normalized_total_gain=("normalized_total_gain", "median"),
            min_normalized_total_gain=("normalized_total_gain", "min"),
            positive_gain_fraction=(
                "normalized_total_gain",
                lambda values: float(
                    np.mean(np.asarray(values, dtype=float) > 0.0)
                ),
            ),
            mean_split_count=("split_count", "mean"),
        )
        .sort_values(
            [
                "mean_normalized_total_gain",
                "positive_gain_fraction",
                "median_normalized_total_gain",
                "mean_split_count",
                "raw_feature",
            ],
            ascending=[False, False, False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    signal = ranking[
        ["mean_normalized_total_gain", "positive_gain_fraction", "mean_split_count"]
    ].to_numpy(dtype=float)
    if float(np.abs(signal).max()) <= 1e-12:
        raise RuntimeError(
            "XGBoost produced no non-zero grouped gain signal on the full "
            "feature universe. Refusing to create an arbitrary forward order."
        )
    ranking.insert(0, "gain_rank", np.arange(1, len(ranking) + 1))
    return ranking








def cumulative_forward_feature_counts(
    total_features: int,
    *,
    min_features: int,
    step_size: int,
) -> list[int]:
    """Return a coarse cumulative prefix schedule that always includes full N."""




    if total_features < 1:
        raise ValueError("total_features must be at least 1.")
    if min_features < 1:
        raise ValueError("min_features must be at least 1.")
    if min_features > total_features:
        raise ValueError("min_features cannot exceed total_features.")
    if step_size < 1:
        raise ValueError("step_size must be at least 1.")
    counts = list(range(min_features, total_features, step_size))
    if not counts or counts[-1] != total_features:
        counts.append(total_features)
    return counts








def build_cumulative_forward_groups(
    gain_ranking: pd.DataFrame,
    *,
    min_features: int,
    step_size: int,
    required_features: Sequence[str] = (),
) -> dict[str, list[str]]:
    """Build nested prefixes while retaining every required control."""




    if "raw_feature" not in gain_ranking:
        raise ValueError("Gain ranking must contain raw_feature.")
    gain_order = gain_ranking["raw_feature"].astype(str).tolist()
    if not gain_order or len(gain_order) != len(set(gain_order)):
        raise ValueError("Gain ranking is empty or contains duplicate features.")
    required = [str(value) for value in required_features]
    if len(required) != len(set(required)):
        raise ValueError("Required forward features contain duplicates.")
    missing = [feature for feature in required if feature not in gain_order]
    if missing:
        raise ValueError(f"Required features are absent from gain ranking: {missing}")
    required_set = set(required)
    ordered = [*required, *[value for value in gain_order if value not in required_set]]
    counts = cumulative_forward_feature_counts(
        len(ordered), min_features=min_features, step_size=step_size
    )
    width = max(3, len(str(len(ordered))))
    return {
        f"XGB_GAIN_FORWARD_N{count:0{width}d}": ordered[:count]
        for count in counts
    }








def build_local_forward_refinement_groups(
    gain_ranking: pd.DataFrame,
    evaluated_groups: Mapping[str, Sequence[str]],
    *,
    selected_group_id: str,
    min_features: int,
    required_features: Sequence[str] = (),
) -> dict[str, list[str]]:
    """Fill the count gap below the smallest eligible coarse prefix.




    The coarse forward step controls runtime. Once it identifies the smallest
    tolerance-qualified prefix, testing every intermediate count back to the
    preceding coarse prefix removes step-size overshoot without evaluating the
    entire feature-count path one feature at a time.
    """




    if "raw_feature" not in gain_ranking:
        raise ValueError("Gain ranking must contain raw_feature.")
    if selected_group_id not in evaluated_groups:
        raise ValueError("Selected forward group is missing from evaluated_groups.")
    if min_features < 1:
        raise ValueError("min_features must be at least 1.")




    gain_order = gain_ranking["raw_feature"].astype(str).tolist()
    required = [str(value) for value in required_features]
    if len(required) != len(set(required)):
        raise ValueError("Required refinement features contain duplicates.")
    missing = [feature for feature in required if feature not in gain_order]
    if missing:
        raise ValueError(f"Required features are absent from gain ranking: {missing}")
    required_set = set(required)
    ordered = [*required, *[value for value in gain_order if value not in required_set]]
    selected_count = len(evaluated_groups[selected_group_id])
    existing_counts = sorted(
        {len(features) for features in evaluated_groups.values()}
    )
    lower_counts = [count for count in existing_counts if count < selected_count]
    lower_bound = max(lower_counts) if lower_counts else min_features - 1
    refinement_counts = [
        count
        for count in range(max(min_features, lower_bound + 1), selected_count)
        if count not in existing_counts
    ]
    width = max(3, len(str(len(ordered))))
    return {
        f"XGB_GAIN_FORWARD_N{count:0{width}d}": ordered[:count]
        for count in refinement_counts
    }




def annotate_forward_ranking(
    gain_ranking: pd.DataFrame,
    required_features: Sequence[str],
) -> pd.DataFrame:
    """Expose genuine gain rank and the required-control forward order."""


    if "raw_feature" not in gain_ranking:
        raise ValueError("Gain ranking must contain raw_feature.")
    gain_order = gain_ranking["raw_feature"].astype(str).tolist()
    required = [str(value) for value in required_features]
    if len(gain_order) != len(set(gain_order)) or len(required) != len(set(required)):
        raise ValueError("Gain or required feature order contains duplicates.")
    missing = [feature for feature in required if feature not in gain_order]
    if missing:
        raise ValueError(f"Required features are absent from gain ranking: {missing}")
    required_set = set(required)
    forward_order = [
        *required,
        *[feature for feature in gain_order if feature not in required_set],
    ]
    forward_rank = {
        feature: position for position, feature in enumerate(forward_order, start=1)
    }
    output = gain_ranking.copy()
    if "gain_rank" not in output:
        output.insert(0, "gain_rank", np.arange(1, len(output) + 1))
    output["forward_rank"] = output["raw_feature"].astype(str).map(forward_rank)
    output["selection_role"] = np.where(
        output["raw_feature"].astype(str).isin(required_set),
        "MANDATORY_CONTROL",
        "GAIN_RANKED_CANDIDATE",
    )
    return output.sort_values("forward_rank", kind="stable").reset_index(drop=True)








def fit_one_model(
    *,
    modeling_utils: Any,
    fold: Any,
    train_matrix: sparse.csr_matrix,
    validation_matrix: sparse.csr_matrix,
    spec: Mapping[str, Any],
    sample_weight: np.ndarray | None,
    max_rounds: int,
    early_stopping_rounds: int,
    device: str,
    n_jobs: int,
    seed: int,
) -> tuple[xgb.XGBClassifier, np.ndarray, dict[str, Any], int, float]:
    model = make_model(
        spec,
        n_estimators=max_rounds,
        early_stopping_rounds=early_stopping_rounds,
        device=device,
        n_jobs=n_jobs,
        seed=seed,
    )
    y_train_churn = 1 - np.asarray(fold.y_train, dtype=int)
    y_validation_churn = 1 - np.asarray(fold.y_validation, dtype=int)
    fit_kwargs: dict[str, Any] = {
        "eval_set": [(validation_matrix, y_validation_churn)],
        "verbose": False,
    }
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    started = time.perf_counter()
    model.fit(train_matrix, y_train_churn, **fit_kwargs)
    elapsed = float(time.perf_counter() - started)
    best_iteration = int(getattr(model, "best_iteration", max_rounds - 1))
    p_churn = predict_churn_probability(model, validation_matrix, best_iteration)
    metrics = metric_dict(modeling_utils, fold, p_churn)
    return model, p_churn, metrics, best_iteration, elapsed








def evaluate_configuration_rows(
    *,
    modeling_utils: Any,
    selection_frame: pd.DataFrame,
    fold_defs: Sequence[Mapping[str, Any]],
    feature_types: Mapping[str, str],
    features: Sequence[str],
    config: Any,
    configurations: Sequence[Mapping[str, Any]],
    train_max_rows: int,
    validation_max_rows: int,
    max_rounds: int,
    early_stopping_rounds: int,
    device: str,
    n_jobs: int,
    rare_min_frequency: float,
    stage: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    # Reuse prepared matrices for configurations sharing the same history rule.
    by_history: dict[str, list[Mapping[str, Any]]] = {}
    for item in configurations:
        by_history.setdefault(str(item["history"]["history_id"]), []).append(item)




    history_lookup = {
        str(item["history"]["history_id"]): dict(item["history"])
        for item in configurations
    }
    for fold_position, fold_def in enumerate(fold_defs, start=1):
        for history_id, group in by_history.items():
            history = history_lookup[history_id]
            print(
                f"{stage}: fold {fold_position}/{len(fold_defs)}, "
                f"history {history_id}, {len(group)} model specification(s)",
                flush=True,
            )
            fold = prepare_fold(
                modeling_utils=modeling_utils,
                frame=selection_frame,
                fold_def=fold_def,
                feature_types=feature_types,
                config=config,
                strategy=history,
                train_max_rows=train_max_rows,
                validation_max_rows=validation_max_rows,
                rare_min_frequency=rare_min_frequency,
            )
            train_matrix, validation_matrix, _ = build_sparse_matrices(fold, features)
            weights = training_weights(
                fold.train, config.cohort_date_col, history
            )
            for item in group:
                spec = dict(item["model"])
                model_id = str(spec["model_id"])
                candidate_id = f"{history_id}__{model_id}"
                model, _, metrics, best_iteration, elapsed = fit_one_model(
                    modeling_utils=modeling_utils,
                    fold=fold,
                    train_matrix=train_matrix,
                    validation_matrix=validation_matrix,
                    spec=spec,
                    sample_weight=weights,
                    max_rounds=max_rounds,
                    early_stopping_rounds=early_stopping_rounds,
                    device=device,
                    n_jobs=n_jobs,
                    seed=BASE_SEED,
                )
                rows.append(
                    {
                        "stage": stage,
                        "fold": int(fold.fold_number),
                        "candidate_id": candidate_id,
                        "history_id": history_id,
                        "history_kind": str(history["kind"]),
                        "history_months": int(history["months"]),
                        "history_max_months": int(
                            history.get("max_months", history["months"])
                        ),
                        "model_id": model_id,
                        "raw_feature_count": int(len(features)),
                        "transformed_column_count": int(train_matrix.shape[1]),
                        "train_rows": int(len(fold.train)),
                        "validation_rows": int(len(fold.validation)),
                        "best_iteration_zero_based": int(best_iteration),
                        "best_rounds": int(best_iteration + 1),
                        "elapsed_seconds": elapsed,
                        **metrics,
                    }
                )
                del model
                gc.collect()
            del fold, train_matrix, validation_matrix
            gc.collect()
    return pd.DataFrame(rows)




def run_optuna_configuration_screen(
    *,
    modeling_utils: Any,
    selection_frame: pd.DataFrame,
    fold_def: Mapping[str, Any],
    feature_types: Mapping[str, str],
    features: Sequence[str],
    config: Any,
    histories: Sequence[Mapping[str, Any]],
    adaptive_trials: int,
    batch_size: int,
    timeout_seconds: int,
    train_max_rows: int,
    validation_max_rows: int,
    max_rounds: int,
    early_stopping_rounds: int,
    device: str,
    n_jobs: int,
    rare_min_frequency: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]:
    """Screen anchored and adaptive Optuna trials on one inner validation fold.


    Trials are asked and told sequentially in small batches. Within each batch,
    ``evaluate_configuration_rows`` groups trials by history treatment, so the
    expensive fold-local preprocessing and sparse matrix are reused rather than
    rebuilt once per trial. Only the supplied inner fold is ever scored here.
    """


    if adaptive_trials < 0:
        raise ValueError("--optuna-trials cannot be negative.")
    if batch_size < 1:
        raise ValueError("--optuna-batch-size must be at least 1.")
    if timeout_seconds < 0:
        raise ValueError("--optuna-timeout-seconds cannot be negative.")
    history_lookup = {
        str(history["history_id"]): dict(history) for history in histories
    }
    if len(history_lookup) != len(histories) or not history_lookup:
        raise ValueError("Optuna history candidates must be non-empty and unique.")


    feature_hash = feature_sequence_sha256(features)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=BASE_SEED)
    study_name = f"retention_xgb_{PROFILE_HASH[:12]}_{feature_hash[:12]}"
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
    )
    anchor_count = enqueue_optuna_anchor_trials(study, histories)
    requested_total = anchor_count + int(adaptive_trials)
    completed = 0
    adaptive_started: float | None = None
    timeout_reached = False
    all_rows: list[pd.DataFrame] = []
    configuration_lookup: dict[str, dict[str, Any]] = {}


    while completed < requested_total:
        if completed >= anchor_count:
            if adaptive_started is None:
                adaptive_started = time.perf_counter()
            if (
                timeout_seconds > 0
                and time.perf_counter() - adaptive_started >= timeout_seconds
            ):
                timeout_reached = True
                break
            remaining_in_phase = requested_total - completed
        else:
            remaining_in_phase = anchor_count - completed
        current_batch_size = min(int(batch_size), int(remaining_in_phase))
        trials = [study.ask() for _ in range(current_batch_size)]
        configurations = [
            suggest_optuna_configuration(trial, history_lookup) for trial in trials
        ]
        batch_rows = evaluate_configuration_rows(
            modeling_utils=modeling_utils,
            selection_frame=selection_frame,
            fold_defs=[fold_def],
            feature_types=feature_types,
            features=features,
            config=config,
            configurations=configurations,
            train_max_rows=train_max_rows,
            validation_max_rows=validation_max_rows,
            max_rounds=max_rounds,
            early_stopping_rounds=early_stopping_rounds,
            device=device,
            n_jobs=n_jobs,
            rare_min_frequency=rare_min_frequency,
            stage="OPTUNA_SCREEN",
        )


        for trial, configuration in zip(trials, configurations):
            history_id = str(configuration["history"]["history_id"])
            model_spec = dict(configuration["model"])
            candidate_id = f"{history_id}__{model_spec['model_id']}"
            matches = batch_rows.loc[batch_rows["candidate_id"].eq(candidate_id)]
            if len(matches) != 1:
                raise RuntimeError(
                    "Optuna screen produced an unexpected candidate-row count for "
                    f"{candidate_id}: {len(matches)}"
                )
            row_index = matches.index[0]
            objective_value = float(matches.at[row_index, "churn_ap_skill"])
            if not np.isfinite(objective_value):
                raise RuntimeError(
                    f"Optuna objective is non-finite for {candidate_id}: "
                    f"{objective_value!r}"
                )
            origin = str(
                trial.user_attrs.get("trial_origin", "TPE_ADAPTIVE")
            )
            trial.set_user_attr("candidate_id", candidate_id)
            trial.set_user_attr("history_id", history_id)
            trial.set_user_attr("model_spec", strict_json_value(model_spec))
            for metric in ("churn_ap_skill", *METRICS):
                if metric in matches:
                    trial.set_user_attr(
                        metric,
                        strict_json_value(matches.at[row_index, metric]),
                    )
            study.tell(trial, objective_value)


            batch_rows.at[row_index, "optuna_trial_number"] = int(trial.number)
            batch_rows.at[row_index, "optuna_trial_origin"] = origin
            batch_rows.at[row_index, "optuna_objective_value"] = objective_value
            for name, value in trial.params.items():
                batch_rows.at[row_index, f"param_{name}"] = value
            batch_rows.at[row_index, "param_max_bin"] = int(model_spec["max_bin"])
            batch_rows.at[row_index, "param_max_leaves_resolved"] = int(
                model_spec["max_leaves"]
            )
            configuration_lookup[candidate_id] = {
                "history": dict(configuration["history"]),
                "model": model_spec,
                "optuna_trial_number": int(trial.number),
                "optuna_trial_origin": origin,
            }


        all_rows.append(batch_rows)
        completed += current_batch_size


    if not all_rows:
        raise RuntimeError("Optuna configuration search completed no trials.")
    screen = pd.concat(all_rows, ignore_index=True)
    best_trial = study.best_trial
    metadata = {
        "optuna_version": optuna.__version__,
        "study_name": study_name,
        "sampler": "TPESampler",
        "direction": "maximize",
        "objective": "churn_ap_skill",
        "seed": int(BASE_SEED),
        "parallel_optuna_jobs": 1,
        "anchor_trial_count": int(anchor_count),
        "requested_adaptive_trials": int(adaptive_trials),
        "completed_adaptive_trials": int(max(0, completed - anchor_count)),
        "completed_total_trials": int(completed),
        "batch_size": int(batch_size),
        "adaptive_timeout_seconds": int(timeout_seconds),
        "adaptive_timeout_reached": bool(timeout_reached),
        "best_screen_trial_number": int(best_trial.number),
        "best_screen_objective_value": float(best_trial.value),
        "best_screen_params": strict_json_value(best_trial.params),
        "search_space": optuna_search_space(),
        "history_candidates": [dict(value) for value in histories],
        "screened_inner_fold": int(fold_def.get("fold", 1)),
        "raw_feature_count": int(len(features)),
        "ordered_feature_sha256": feature_hash,
        "screen_train_max_rows": int(train_max_rows),
        "validation_max_rows": int(validation_max_rows),
        "max_rounds": int(max_rounds),
        "early_stopping_rounds": int(early_stopping_rounds),
        "rare_min_frequency": float(rare_min_frequency),
        "outer_validation_rows_scored": 0,
        "final_holdout_rows_scored": 0,
    }
    return screen, configuration_lookup, metadata








def summarize_candidates(frame: pd.DataFrame, id_col: str) -> pd.DataFrame:
    frame = frame.copy()
    if (
        "churn_recall_10pct" not in frame.columns
        and "churn_capture_10pct" in frame.columns
    ):
        frame["churn_recall_10pct"] = frame["churn_capture_10pct"]
    rows: list[dict[str, Any]] = []
    for candidate_id, part in frame.groupby(id_col, sort=False):
        part = part.sort_values("fold")
        newest_fold = int(part["fold"].max())
        row: dict[str, Any] = {
            id_col: str(candidate_id),
            "fold_count": int(part["fold"].nunique()),
            "raw_feature_count": int(part["raw_feature_count"].iloc[0]),
            "median_best_rounds": int(
                round(float(pd.to_numeric(part["best_rounds"]).median()))
            ),
            "newest_fold": newest_fold,
            "newest_fold_ap_skill": float(
                part.loc[part["fold"].eq(newest_fold), "churn_ap_skill"].iloc[0]
            ),
        }
        for optional in (
            "history_id",
            "history_kind",
            "history_months",
            "history_max_months",
            "model_id",
            "group_id",
        ):
            if optional in part.columns:
                row[optional] = part[optional].iloc[0]
        for metric in ("churn_ap_skill", *METRICS):
            values = pd.to_numeric(part[metric], errors="raise")
            row[f"mean_{metric}"] = float(values.mean())
            row[f"std_{metric}"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"min_{metric}"] = float(values.min())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [
            "mean_churn_ap_skill",
            "mean_top10_lift",
            "mean_retention_roc_auc",
            "raw_feature_count",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)








def choose_best_configuration(summary: pd.DataFrame) -> str:
    if summary.empty:
        raise ValueError("Configuration summary is empty.")
    ranked = summary.sort_values(
        [
            "mean_churn_ap_skill",
            "mean_top10_lift",
            "mean_retention_roc_auc",
            "mean_log_loss",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    return str(ranked.iloc[0]["candidate_id"])




def choose_best_finalist_configuration(
    confirmation_long: pd.DataFrame,
) -> tuple[str, pd.DataFrame, str]:
    """Choose a configuration through hierarchical paired AP/recall gates.


    Configuration confirmation rows, not aggregate point estimates, are the
    input. All configurations for one finalist must use the same raw features,
    so after both one-SE gates the strongest deterministic AP/recall result is
    selected rather than applying feature-count parsimony across configurations.
    """


    if confirmation_long.empty:
        raise ValueError("Finalist configuration confirmation is empty.")
    if "raw_feature_count" not in confirmation_long:
        raise ValueError("Finalist configuration rows lack raw_feature_count.")
    raw_counts = pd.to_numeric(
        confirmation_long["raw_feature_count"], errors="raise"
    ).astype(int)
    if raw_counts.nunique() != 1:
        raise ValueError(
            "Finalist configurations must all use the same raw-feature count."
        )
    selected_id, annotated, _ = choose_paired_one_se_group(
        confirmation_long,
        id_col="candidate_id",
    )
    annotated["eligible_configuration"] = annotated[
        "eligible_performance_group"
    ].astype(bool)
    annotated["selected_configuration_by_rule"] = annotated[
        "candidate_id"
    ].astype(str).eq(selected_id)
    reason = (
        "Selected the strongest deterministic configuration among configurations "
        "whose paired chronological-fold AP-skill loss versus the AP-skill "
        "leader was no larger than one standard error, then whose recall@10% "
        "loss versus the best-recall AP-non-inferior configuration was no "
        "larger than one standard error."
    )
    annotated["configuration_selection_reason"] = reason
    return selected_id, annotated, reason




def choose_finalist_configuration_ids(
    screen_summary: pd.DataFrame,
    *,
    confirmation_budget: int,
) -> list[str]:
    """Keep AP and recall leaders inside one fixed confirmation budget."""


    if screen_summary.empty:
        raise ValueError("Finalist Optuna screen summary is empty.")
    if confirmation_budget < 1:
        raise ValueError("Finalist configuration confirmation budget must be positive.")
    recall_col = _metric_column(
        screen_summary,
        "mean_churn_recall_10pct",
        "mean_churn_capture_10pct",
    )
    ap_ranked = screen_summary.sort_values(
        [
            "mean_churn_ap_skill",
            recall_col,
            "mean_retention_roc_auc",
            "candidate_id",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    recall_ranked = screen_summary.sort_values(
        [
            recall_col,
            "mean_churn_ap_skill",
            "mean_retention_roc_auc",
            "candidate_id",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    ordered = [
        str(ap_ranked.iloc[0]["candidate_id"]),
        str(recall_ranked.iloc[0]["candidate_id"]),
        *ap_ranked["candidate_id"].astype(str).tolist(),
    ]
    selected: list[str] = []
    for candidate_id in ordered:
        if candidate_id not in selected:
            selected.append(candidate_id)
        if len(selected) >= confirmation_budget:
            break
    return selected




def choose_finalist_group_ids(
    confirmation_summary: pd.DataFrame,
    *,
    stage4_selected_group_id: str,
    full_group_id: str,
    groups_by_id: Mapping[str, Sequence[str]],
) -> tuple[list[str], pd.DataFrame]:
    """Build the complete Stage-4 eligible/Pareto finalist union.


    Every confirmed one-SE-eligible prefix and every confirmed count/AP/recall
    Pareto-relevant prefix is retuned. The discovery winner, AP leader, recall
    leader, and full universe remain explicit audit anchors even when they are
    already present through an eligibility or Pareto role.
    """


    if confirmation_summary.empty:
        raise ValueError("Stage-4 confirmation summary is empty.")
    required = {
        "group_id",
        "raw_feature_count",
        "mean_churn_ap_skill",
        "eligible_performance_group",
    }
    missing = sorted(required.difference(confirmation_summary.columns))
    if missing:
        raise ValueError(f"Finalist summary is missing columns: {missing}")
    recall_col = _metric_column(
        confirmation_summary,
        "mean_churn_recall_10pct",
        "mean_churn_capture_10pct",
    )
    summary = confirmation_summary.copy()
    summary["group_id"] = summary["group_id"].astype(str)
    eligible_mask = (
        summary["eligible_performance_group"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1"})
    )
    eligible_ids = (
        summary.loc[eligible_mask]
        .sort_values(
            ["raw_feature_count", "mean_churn_ap_skill", recall_col, "group_id"],
            ascending=[True, False, False, True],
            kind="mergesort",
        )["group_id"]
        .astype(str)
        .tolist()
    )
    pareto_ids = pareto_relevant_group_ids(summary, id_col="group_id")
    available = set(summary["group_id"])
    for group_id in (stage4_selected_group_id, full_group_id):
        if group_id not in available or group_id not in groups_by_id:
            raise ValueError(f"Required finalist group is missing: {group_id}")
    ap_leader = str(
        summary.sort_values(
            ["mean_churn_ap_skill", recall_col, "raw_feature_count", "group_id"],
            ascending=[False, False, True, True],
            kind="mergesort",
        ).iloc[0]["group_id"]
    )
    recall_leader = str(
        summary.sort_values(
            [recall_col, "mean_churn_ap_skill", "raw_feature_count", "group_id"],
            ascending=[False, False, True, True],
            kind="mergesort",
        ).iloc[0]["group_id"]
    )
    role_pairs = [
        *(("STAGE4_ONE_SE_ELIGIBLE", group_id) for group_id in eligible_ids),
        *(("COUNT_AP_RECALL_PARETO", group_id) for group_id in pareto_ids),
        ("STAGE4_SMALLEST_ONE_SE", str(stage4_selected_group_id)),
        ("BEST_MEAN_AP_SKILL", ap_leader),
        ("BEST_MEAN_RECALL_10PCT", recall_leader),
        ("FULL_FEATURE_UNIVERSE", str(full_group_id)),
    ]
    finalist_ids: list[str] = []
    roles: dict[str, list[str]] = {}
    for role, group_id in role_pairs:
        if group_id not in available or group_id not in groups_by_id:
            raise ValueError(f"Finalist group is missing: {group_id}")
        roles.setdefault(group_id, []).append(role)
        if group_id not in finalist_ids:
            finalist_ids.append(group_id)
    rows: list[dict[str, Any]] = []
    for finalist_order, group_id in enumerate(finalist_ids, start=1):
        features = list(groups_by_id[group_id])
        summary_row = summary.loc[summary["group_id"].eq(group_id)].iloc[0]
        expected_count = int(summary_row["raw_feature_count"])
        if len(features) != expected_count:
            raise ValueError(
                f"Finalist feature count changed for {group_id}: "
                f"inventory={len(features)}, summary={expected_count}."
            )
        rows.append(
            {
                "finalist_order": int(finalist_order),
                "group_id": group_id,
                "raw_feature_count": int(len(features)),
                "ordered_feature_sha256": feature_sequence_sha256(features),
                "finalist_roles": ";".join(roles[group_id]),
                "stage4_one_se_eligible": bool(group_id in eligible_ids),
                "stage4_count_ap_recall_pareto_relevant": bool(
                    group_id in pareto_ids
                ),
                "stage4_mean_churn_ap_skill": float(
                    summary_row["mean_churn_ap_skill"]
                ),
                "stage4_mean_churn_recall_10pct": float(summary_row[recall_col]),
                "features": ";".join(features),
            }
        )
    return finalist_ids, pd.DataFrame(rows)




def stage2_optuna_screen_is_exactly_reusable(
    *,
    screen: pd.DataFrame,
    metadata: Mapping[str, Any],
    features: Sequence[str],
    histories: Sequence[Mapping[str, Any]],
    fold_def: Mapping[str, Any],
    adaptive_trials: int,
    batch_size: int,
    train_max_rows: int,
    validation_max_rows: int,
    max_rounds: int,
    early_stopping_rounds: int,
    rare_min_frequency: float,
) -> bool:
    """Require exact equality before reusing the full-universe Stage-2 study."""


    if screen.empty or int(metadata.get("adaptive_timeout_seconds", -1)) != 0:
        return False
    expected = {
        "objective": "churn_ap_skill",
        "seed": int(BASE_SEED),
        "parallel_optuna_jobs": 1,
        "requested_adaptive_trials": int(adaptive_trials),
        "completed_adaptive_trials": int(adaptive_trials),
        "batch_size": int(batch_size),
        "adaptive_timeout_reached": False,
        "search_space": optuna_search_space(),
        "history_candidates": [dict(value) for value in histories],
        "screened_inner_fold": int(fold_def.get("fold", 1)),
        "raw_feature_count": int(len(features)),
        "ordered_feature_sha256": feature_sequence_sha256(features),
        "screen_train_max_rows": int(train_max_rows),
        "validation_max_rows": int(validation_max_rows),
        "max_rounds": int(max_rounds),
        "early_stopping_rounds": int(early_stopping_rounds),
        "rare_min_frequency": float(rare_min_frequency),
    }
    if any(strict_json_value(metadata.get(name)) != strict_json_value(value) for name, value in expected.items()):
        return False
    anchor_count = len(histories) * len(anchor_model_grid())
    expected_total = anchor_count + int(adaptive_trials)
    if (
        int(metadata.get("anchor_trial_count", -1)) != anchor_count
        or int(metadata.get("completed_total_trials", -1)) != expected_total
        or len(screen) != expected_total
    ):
        return False
    required = {
        "fold",
        "raw_feature_count",
        "optuna_trial_number",
        "optuna_objective_value",
    }
    if not required.issubset(screen.columns):
        return False
    trial_numbers = pd.to_numeric(
        screen["optuna_trial_number"], errors="coerce"
    )
    objective = pd.to_numeric(
        screen["optuna_objective_value"], errors="coerce"
    )
    return bool(
        trial_numbers.notna().all()
        and objective.notna().all()
        and np.isfinite(objective.to_numpy(dtype=float)).all()
        and sorted(trial_numbers.astype(int).tolist()) == list(range(expected_total))
        and pd.to_numeric(screen["fold"], errors="coerce")
        .eq(int(fold_def.get("fold", 1)))
        .all()
        and pd.to_numeric(screen["raw_feature_count"], errors="coerce")
        .eq(len(features))
        .all()
    )








def choose_performance_group(
    summary: pd.DataFrame,
    *,
    id_col: str,
    ap_skill_tolerance: float,
    lift_tolerance: float,
    auc_tolerance: float,
    recent_ap_skill_tolerance: float,
) -> tuple[str, pd.DataFrame, str]:
    """Choose the smallest group with performance inside every tolerance.




    The selection sequence is deliberately aligned to the project objective:
    1. stay within a very small tolerance of the best churn AP skill;
    2. among those groups, stay within tolerance of the best lift@10%;
    3. among those, stay within tolerance of the best ROC AUC;
    4. enforce recent-fold AP-skill stability;
    5. treat the surviving groups as effectively tied and choose the smallest;
    6. use exact performance only to break ties between equal-size groups.
    """




    if summary.empty:
        raise ValueError("Cannot choose a feature group from an empty summary.")
    annotated = summary.copy()




    best_ap = float(annotated["mean_churn_ap_skill"].max())
    ap_floor = best_ap - float(ap_skill_tolerance)
    annotated["best_mean_ap_skill"] = best_ap
    annotated["ap_skill_floor"] = ap_floor
    annotated["passes_ap_skill_tolerance"] = annotated[
        "mean_churn_ap_skill"
    ].ge(ap_floor)




    ap_pool = annotated.loc[annotated["passes_ap_skill_tolerance"]].copy()
    best_lift = float(ap_pool["mean_top10_lift"].max())
    lift_floor = best_lift - float(lift_tolerance)
    annotated["best_lift_within_ap_pool"] = best_lift
    annotated["lift_floor"] = lift_floor
    annotated["passes_lift_tolerance"] = annotated["mean_top10_lift"].ge(
        lift_floor
    )




    lift_pool = annotated.loc[
        annotated["passes_ap_skill_tolerance"]
        & annotated["passes_lift_tolerance"]
    ].copy()
    best_auc = float(lift_pool["mean_retention_roc_auc"].max())
    auc_floor = best_auc - float(auc_tolerance)
    annotated["best_auc_within_ap_lift_pool"] = best_auc
    annotated["auc_floor"] = auc_floor
    annotated["passes_auc_tolerance"] = annotated[
        "mean_retention_roc_auc"
    ].ge(auc_floor)




    auc_pool = annotated.loc[
        annotated["passes_ap_skill_tolerance"]
        & annotated["passes_lift_tolerance"]
        & annotated["passes_auc_tolerance"]
    ].copy()
    best_recent = float(auc_pool["newest_fold_ap_skill"].max())
    recent_floor = best_recent - float(recent_ap_skill_tolerance)
    annotated["best_recent_ap_skill_within_pool"] = best_recent
    annotated["recent_ap_skill_floor"] = recent_floor
    annotated["passes_recent_ap_skill_tolerance"] = annotated[
        "newest_fold_ap_skill"
    ].ge(recent_floor)




    annotated["eligible_performance_group"] = annotated[
        [
            "passes_ap_skill_tolerance",
            "passes_lift_tolerance",
            "passes_auc_tolerance",
            "passes_recent_ap_skill_tolerance",
        ]
    ].all(axis=1)
    eligible = annotated.loc[annotated["eligible_performance_group"]].copy()
    if eligible.empty:
        eligible = annotated.sort_values(
            [
                "mean_churn_ap_skill",
                "mean_top10_lift",
                "mean_retention_roc_auc",
                "newest_fold_ap_skill",
                "raw_feature_count",
            ],
            ascending=[False, False, False, False, True],
            kind="mergesort",
        ).head(1)
        reason = (
            "No feature group passed every tolerance simultaneously; selected "
            "the strongest AP/lift/AUC group."
        )
    else:
        eligible = eligible.sort_values(
            [
                "raw_feature_count",
                "mean_churn_ap_skill",
                "mean_top10_lift",
                "mean_retention_roc_auc",
                "newest_fold_ap_skill",
            ],
            ascending=[True, False, False, False, False],
            kind="mergesort",
        )
        reason = (
            "Selected the smallest raw-feature group among groups within all "
            "churn AP-skill, lift@10%, ROC-AUC, and newest-fold AP-skill "
            "tolerances; exact performance was used only to break equal-size "
            "ties."
        )
    selected_id = str(eligible.iloc[0][id_col])
    annotated["selected_by_rule"] = annotated[id_col].astype(str).eq(selected_id)
    annotated = annotated.sort_values(
        [
            "selected_by_rule",
            "mean_churn_ap_skill",
            "mean_top10_lift",
            "mean_retention_roc_auc",
            "raw_feature_count",
        ],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    )
    return selected_id, annotated, reason








def _metric_column(frame: pd.DataFrame, preferred: str, fallback: str) -> str:
    """Return a required metric column, accepting one audited legacy alias."""


    if preferred in frame.columns:
        return preferred
    if fallback in frame.columns:
        return fallback
    raise ValueError(
        f"Required metric is missing: expected {preferred!r} or {fallback!r}."
    )




def choose_final_feature_group(
    long_frame: pd.DataFrame, *, policy: Mapping[str, Any],
) -> tuple[str, pd.DataFrame, str]:
    """Apply practical parsimony only after each finalist has been retuned."""
    winner, practical, reason = select_practical_feature_group(long_frame, policy=policy)
    summary = summarize_candidates(long_frame, "group_id")
    added = [column for column in practical if column not in summary]
    summary = summary.merge(practical[["group_id", *added]], on="group_id", validate="one_to_one")
    summary = summary.set_index("group_id").loc[practical["group_id"]].reset_index()
    return winner, summary, reason




def choose_paired_one_se_group(
    long_frame: pd.DataFrame,
    *,
    id_col: str,
) -> tuple[str, pd.DataFrame, str]:
    """Choose the smallest AP-skill/recall non-inferior feature group.


    Seed predictions are already averaged before each row is scored, so the
    chronological folds, rather than seeds, are the independent paired units.
    AP-skill losses are first paired against the global AP-skill leader. The
    recall@10% reference is then the best-recall candidate among candidates
    that passed the AP gate, and recall losses are paired against that
    conditional reference. A candidate must pass both one-standard-error
    gates. Among eligible feature groups the smallest group wins, with
    performance used only for deterministic ties.
    This is an uncertainty-aware parsimony rule, not a claim of formal
    statistical equivalence.
    """


    if long_frame.empty:
        raise ValueError("Cannot choose a feature group from empty fold results.")
    required = {id_col, "fold", "raw_feature_count", "churn_ap_skill"}
    missing = sorted(required.difference(long_frame.columns))
    if missing:
        raise ValueError(f"Paired one-SE input is missing columns: {missing}")
    recall_col = _metric_column(
        long_frame, "churn_recall_10pct", "churn_capture_10pct"
    )
    frame = long_frame.copy()
    frame[id_col] = frame[id_col].astype(str)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    if frame.duplicated([id_col, "fold"]).any():
        raise ValueError("Paired one-SE input contains duplicate group/fold rows.")
    for metric in ("churn_ap_skill", recall_col):
        values = pd.to_numeric(frame[metric], errors="raise")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Paired one-SE metric is non-finite: {metric}")
        frame[metric] = values.astype(float)


    fold_sets = {
        candidate_id: tuple(sorted(part["fold"].astype(int).tolist()))
        for candidate_id, part in frame.groupby(id_col, sort=False)
    }
    expected_folds = next(iter(fold_sets.values()))
    if not expected_folds or any(
        candidate_folds != expected_folds
        for candidate_folds in fold_sets.values()
    ):
        raise ValueError(
            "Paired one-SE selection requires the same chronological folds "
            "for every feature group."
        )


    summary = summarize_candidates(frame, id_col)
    recall_summary_col = "mean_churn_recall_10pct"
    if recall_summary_col not in summary.columns:
        # ``summarize_candidates`` creates this field after installing the
        # audited capture alias, but keep the invariant explicit here.
        raise RuntimeError("Recall@10% was not summarized for one-SE selection.")
    ap_reference = summary.sort_values(
        [
            "mean_churn_ap_skill",
            recall_summary_col,
            "raw_feature_count",
            id_col,
        ],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).iloc[0]
    ap_reference_id = str(ap_reference[id_col])
    ap_reference_rows = (
        frame.loc[
            frame[id_col].eq(ap_reference_id), ["fold", "churn_ap_skill"]
        ]
        .sort_values("fold", kind="stable")
        .set_index("fold")
    )


    def paired_stats(losses: np.ndarray) -> tuple[float, float, bool]:
        mean_loss = float(np.mean(losses))
        standard_error = (
            float(np.std(losses, ddof=1) / np.sqrt(len(losses)))
            if len(losses) > 1
            else 0.0
        )
        # Only a scale-aware machine epsilon is used; this is not another
        # user-selected performance tolerance.
        numeric_epsilon = float(
            np.finfo(float).eps
            * max(1.0, abs(mean_loss), abs(standard_error))
            * 16.0
        )
        return (
            mean_loss,
            standard_error,
            bool(mean_loss <= standard_error + numeric_epsilon),
        )


    candidate_rows_by_id: dict[str, pd.DataFrame] = {}
    ap_statistics: dict[str, tuple[float, float, bool]] = {}
    for candidate_id_value, part in frame.groupby(id_col, sort=False):
        candidate_id = str(candidate_id_value)
        candidate_rows = (
            part[["fold", "churn_ap_skill", recall_col]]
            .sort_values("fold", kind="stable")
            .set_index("fold")
        )
        candidate_rows_by_id[candidate_id] = candidate_rows
        ap_losses = (
            ap_reference_rows["churn_ap_skill"]
            - candidate_rows["churn_ap_skill"]
        ).to_numpy(dtype=float)
        ap_statistics[candidate_id] = paired_stats(ap_losses)


    ap_pass_ids = {
        candidate_id
        for candidate_id, (_, _, passed) in ap_statistics.items()
        if passed
    }
    if not ap_pass_ids:
        raise RuntimeError("The AP-skill leader did not pass its own one-SE gate.")
    recall_reference = summary.loc[
        summary[id_col].astype(str).isin(ap_pass_ids)
    ].sort_values(
        [
            recall_summary_col,
            "mean_churn_ap_skill",
            "raw_feature_count",
            id_col,
        ],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).iloc[0]
    recall_reference_id = str(recall_reference[id_col])
    recall_reference_rows = candidate_rows_by_id[recall_reference_id]


    annotations: list[dict[str, Any]] = []
    for candidate_id, candidate_rows in candidate_rows_by_id.items():
        recall_losses = (
            recall_reference_rows[recall_col] - candidate_rows[recall_col]
        ).to_numpy(dtype=float)
        fold_count = int(len(recall_losses))
        ap_mean_loss, ap_se, ap_pass = ap_statistics[candidate_id]
        recall_mean_loss, recall_se, recall_pass = paired_stats(recall_losses)
        annotations.append(
            {
                id_col: str(candidate_id),
                "selection_rule": STAGE4_SELECTION_RULE,
                "ap_skill_one_se_reference_id": ap_reference_id,
                "recall_10pct_one_se_reference_id": recall_reference_id,
                "recall_reference_condition": (
                    "BEST_MEAN_RECALL_AMONG_AP_SKILL_ONE_SE_PASS_CANDIDATES"
                ),
                "paired_fold_count": fold_count,
                "mean_ap_skill_loss_vs_reference": ap_mean_loss,
                "se_ap_skill_loss_vs_reference": ap_se,
                "passes_ap_skill_one_se": ap_pass,
                "mean_recall_10pct_loss_vs_reference": recall_mean_loss,
                "se_recall_10pct_loss_vs_reference": recall_se,
                "passes_recall_10pct_one_se": recall_pass,
                "eligible_performance_group": bool(ap_pass and recall_pass),
            }
        )


    annotated = summary.merge(pd.DataFrame(annotations), on=id_col, how="left")
    eligible = annotated.loc[annotated["eligible_performance_group"]].copy()
    if eligible.empty:
        raise RuntimeError(
            "Paired one-SE selection produced no candidate that passed both "
            "the AP-skill gate and the conditional recall@10% gate."
        )
    eligible = eligible.sort_values(
        [
            "raw_feature_count",
            "mean_churn_ap_skill",
            recall_summary_col,
            "newest_fold_ap_skill",
            "mean_retention_roc_auc",
            "mean_top10_lift",
            id_col,
        ],
        ascending=[True, False, False, False, False, False, True],
        kind="mergesort",
    )
    selected_id = str(eligible.iloc[0][id_col])
    annotated["selected_by_rule"] = annotated[id_col].astype(str).eq(selected_id)
    annotated = annotated.sort_values(
        [
            "selected_by_rule",
            "eligible_performance_group",
            "raw_feature_count",
            "mean_churn_ap_skill",
            recall_summary_col,
            id_col,
        ],
        ascending=[False, False, True, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    reason = (
        "Selected the smallest raw-feature prefix whose paired chronological-"
        "fold AP-skill loss versus the AP-skill leader and recall@10% loss "
        "versus the best-recall AP-non-inferior candidate were each no larger "
        "than one standard error; lift@10%, ROC-AUC, calibration, and newest-"
        "fold metrics remain diagnostics."
    )
    return selected_id, annotated, reason




def pareto_relevant_group_ids(
    summary: pd.DataFrame,
    *,
    id_col: str = "group_id",
    candidate_ids: Sequence[str] | None = None,
) -> list[str]:
    """Return deterministic count/AP/recall non-dominated prefixes."""


    if summary.empty:
        return []
    required = {id_col, "raw_feature_count", "mean_churn_ap_skill"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Pareto summary is missing columns: {missing}")
    recall_col = _metric_column(
        summary, "mean_churn_recall_10pct", "mean_churn_capture_10pct"
    )
    pool = summary.copy()
    pool[id_col] = pool[id_col].astype(str)
    if candidate_ids is not None:
        allowed = {str(value) for value in candidate_ids}
        pool = pool.loc[pool[id_col].isin(allowed)].copy()
    if pool.empty:
        return []
    rows = list(pool.to_dict(orient="records"))
    relevant: list[dict[str, Any]] = []
    epsilon = 1.0e-15
    for candidate in rows:
        candidate_count = int(candidate["raw_feature_count"])
        candidate_ap = float(candidate["mean_churn_ap_skill"])
        candidate_recall = float(candidate[recall_col])
        dominated = False
        for challenger in rows:
            if challenger[id_col] == candidate[id_col]:
                continue
            challenger_count = int(challenger["raw_feature_count"])
            challenger_ap = float(challenger["mean_churn_ap_skill"])
            challenger_recall = float(challenger[recall_col])
            weakly_better = (
                challenger_count <= candidate_count
                and challenger_ap + epsilon >= candidate_ap
                and challenger_recall + epsilon >= candidate_recall
            )
            strictly_better = (
                challenger_count < candidate_count
                or challenger_ap > candidate_ap + epsilon
                or challenger_recall > candidate_recall + epsilon
            )
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            relevant.append(candidate)
    return [
        str(row[id_col])
        for row in sorted(
            relevant,
            key=lambda row: (
                int(row["raw_feature_count"]),
                -float(row["mean_churn_ap_skill"]),
                -float(row[recall_col]),
                str(row[id_col]),
            ),
        )
    ]




def choose_confirmation_group_ids(
    *,
    forward_summary: pd.DataFrame,
    forward_annotated: pd.DataFrame,
    groups_by_id: Mapping[str, Sequence[str]],
    initial_selected_id: str,
    confirm_candidate_count: int,
    full_group_id: str | None = None,
    coarse_group_ids: Sequence[str] | None = None,
) -> list[str]:
    """Build the complete required confirmation queue without truncation.


    ``confirm_candidate_count`` is the per-pass execution cap, not a shortlist
    cap. Required anchors, discovery-eligible prefixes, and count/AP/recall
    Pareto-relevant coarse prefixes therefore remain in the returned queue even
    when more than one Stage-4 pass is needed.
    """




    if confirm_candidate_count < 1:
        raise ValueError("confirm_candidate_count must be at least 1.")
    if initial_selected_id not in groups_by_id:
        raise ValueError(
            "The initially selected feature group is missing from the forward groups."
        )




    if full_group_id is None:
        full_group_id = max(groups_by_id, key=lambda value: len(groups_by_id[value]))
    if full_group_id not in groups_by_id:
        raise ValueError("The full feature group is missing from groups_by_id.")
    coarse_ids = (
        list(groups_by_id)
        if coarse_group_ids is None
        else [str(value) for value in coarse_group_ids]
    )
    invalid_coarse = sorted(set(coarse_ids).difference(groups_by_id))
    if invalid_coarse:
        raise ValueError(f"Unknown coarse confirmation groups: {invalid_coarse}")


    recall_col = _metric_column(
        forward_summary,
        "mean_churn_recall_10pct",
        "mean_churn_capture_10pct",
    )
    metric_anchors: list[str] = []
    for metric in (
        "mean_churn_ap_skill",
        recall_col,
        "mean_top10_lift",
        "mean_retention_roc_auc",
        "newest_fold_ap_skill",
    ):
        if metric in forward_summary.columns:
            metric_anchors.append(
                str(
                    forward_summary.sort_values(
                        [metric, "raw_feature_count", "group_id"],
                        ascending=[False, True, True],
                        kind="mergesort",
                    ).iloc[0]["group_id"]
                )
            )
    eligible = forward_annotated.loc[
        forward_annotated["eligible_performance_group"].eq(True)
    ].sort_values(
        ["raw_feature_count", "mean_churn_ap_skill", "group_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    eligible_ids = eligible["group_id"].astype(str).tolist()
    ordered_coarse = sorted(
        coarse_ids, key=lambda value: (len(groups_by_id[value]), value)
    )
    full_position = ordered_coarse.index(full_group_id)
    full_neighbor = (
        [] if full_position == 0 else [ordered_coarse[full_position - 1]]
    )
    pareto_candidate_ids = list(
        dict.fromkeys(
            [
                initial_selected_id,
                full_group_id,
                *metric_anchors,
                *eligible_ids,
                *full_neighbor,
            ]
        )
    )
    pareto_ids = pareto_relevant_group_ids(
        forward_summary,
        id_col="group_id",
        candidate_ids=[
            group_id
            for group_id in pareto_candidate_ids
            if group_id in set(coarse_ids)
        ],
    )
    confirmation_ids: list[str] = [
        initial_selected_id,
        full_group_id,
        *metric_anchors,
        *eligible_ids,
        *pareto_ids,
        *full_neighbor,
    ]




    deduplicated_ids: list[str] = []
    for group_id in confirmation_ids:
        if group_id not in groups_by_id:
            raise ValueError(
                f"Confirmation feature group is missing from the forward groups: {group_id}"
            )
        if group_id not in deduplicated_ids:
            deduplicated_ids.append(group_id)
    return deduplicated_ids




def build_adaptive_confirmation_groups(
    gain_ranking: pd.DataFrame,
    groups_by_id: Mapping[str, Sequence[str]],
    confirmed_group_ids: Sequence[str],
    *,
    initial_selected_id: str,
    full_group_id: str,
    coarse_group_ids: Sequence[str],
    required_features: Sequence[str] = (),
    max_candidates: int,
) -> dict[str, list[str]]:
    """Return the next deterministic full-winner confirmation batch.


    All unconfirmed coarse prefixes between the Stage-3 winner and full pool
    are visited in increasing count order. Only after those are exhausted are
    exact counts immediately below the full pool visited in decreasing count
    order. This is an explicit linear schedule, never a binary search.
    """


    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1.")
    if initial_selected_id not in groups_by_id or full_group_id not in groups_by_id:
        raise ValueError("Adaptive confirmation anchors are missing.")
    confirmed = {str(value) for value in confirmed_group_ids}
    initial_count = len(groups_by_id[initial_selected_id])
    full_count = len(groups_by_id[full_group_id])
    if initial_count > full_count:
        raise ValueError("Initial feature group cannot be larger than full group.")


    coarse_ids = [str(value) for value in coarse_group_ids]
    invalid = sorted(set(coarse_ids).difference(groups_by_id))
    if invalid:
        raise ValueError(f"Unknown adaptive coarse groups: {invalid}")
    confirmed_lower_counts = [
        len(groups_by_id[group_id])
        for group_id in confirmed
        if group_id in groups_by_id and len(groups_by_id[group_id]) < full_count
    ]
    adaptive_lower_count = (
        max(confirmed_lower_counts)
        if initial_count == full_count and confirmed_lower_counts
        else initial_count
    )
    pending_coarse = [
        group_id
        for group_id in sorted(
            coarse_ids, key=lambda value: (len(groups_by_id[value]), value)
        )
        if adaptive_lower_count < len(groups_by_id[group_id]) < full_count
        and group_id not in confirmed
    ]
    if pending_coarse:
        return {
            group_id: list(groups_by_id[group_id])
            for group_id in pending_coarse[:max_candidates]
        }


    if "raw_feature" not in gain_ranking:
        raise ValueError("Gain ranking must contain raw_feature.")
    gain_order = gain_ranking["raw_feature"].astype(str).tolist()
    required = [str(value) for value in required_features]
    if len(gain_order) != len(set(gain_order)) or len(required) != len(set(required)):
        raise ValueError("Adaptive gain or required feature order has duplicates.")
    missing = [feature for feature in required if feature not in gain_order]
    if missing:
        raise ValueError(f"Adaptive required features are absent: {missing}")
    required_set = set(required)
    ordered_features = [
        *required,
        *[feature for feature in gain_order if feature not in required_set],
    ]
    lower_coarse_counts = sorted(
        {
            len(groups_by_id[group_id])
            for group_id in coarse_ids
            if len(groups_by_id[group_id]) < full_count
        }
    )
    local_lower_bound = (
        max(lower_coarse_counts) if lower_coarse_counts else initial_count
    )
    existing_by_count = {
        len(features): str(group_id) for group_id, features in groups_by_id.items()
    }
    width = max(3, len(str(full_count)))
    pending_exact: list[tuple[str, list[str]]] = []
    for count in range(full_count - 1, local_lower_bound, -1):
        group_id = existing_by_count.get(
            count, f"XGB_GAIN_FORWARD_N{count:0{width}d}"
        )
        if group_id not in confirmed:
            pending_exact.append((group_id, ordered_features[:count]))
    return dict(pending_exact[:max_candidates])




def confirmation_unresolved_message(
    *,
    completed_passes: int,
    max_passes: int,
    pending_required_ids: Sequence[str],
    pending_adaptive_ids: Sequence[str],
    selected_group_id: str | None,
    full_group_id: str,
) -> str | None:
    """Return the deterministic fail-closed message for an exhausted budget."""


    pending = [*map(str, pending_required_ids), *map(str, pending_adaptive_ids)]
    if completed_passes < max_passes or not pending:
        return None
    return (
        "Stage-4 adaptive confirmation is unresolved after "
        f"{completed_passes}/{max_passes} passes; selected={selected_group_id!r}, "
        f"full={full_group_id!r}, pending={pending}. Increase "
        "--confirm-max-passes or --confirm-candidate-count; no feature group "
        "was accepted."
    )








def evaluate_full_pool_gain_ranking(
    *,
    modeling_utils: Any,
    selection_frame: pd.DataFrame,
    inner_folds: Sequence[Mapping[str, Any]],
    feature_types: Mapping[str, str],
    features: Sequence[str],
    config: Any,
    history: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    train_max_rows: int,
    validation_max_rows: int,
    max_rounds: int,
    early_stopping_rounds: int,
    device: str,
    n_jobs: int,
    rare_min_frequency: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit the full pool on inner folds and derive grouped total-gain order."""




    metric_rows: list[dict[str, Any]] = []
    gain_frames: list[pd.DataFrame] = []
    width = max(3, len(str(len(features))))
    group_id = f"XGB_GAIN_FORWARD_N{len(features):0{width}d}"




    for fold_position, fold_def in enumerate(inner_folds, start=1):
        print(
            f"GAIN RANKING: fold {fold_position}/{len(inner_folds)}, "
            f"{len(features)} raw features",
            flush=True,
        )
        fold = prepare_fold(
            modeling_utils=modeling_utils,
            frame=selection_frame,
            fold_def=fold_def,
            feature_types=feature_types,
            config=config,
            strategy=history,
            train_max_rows=train_max_rows,
            validation_max_rows=validation_max_rows,
            rare_min_frequency=rare_min_frequency,
        )
        train_matrix, validation_matrix, ranges = build_sparse_matrices(
            fold, features
        )
        weights = training_weights(fold.train, config.cohort_date_col, history)
        model, _, metrics, best_iteration, elapsed = fit_one_model(
            modeling_utils=modeling_utils,
            fold=fold,
            train_matrix=train_matrix,
            validation_matrix=validation_matrix,
            spec=model_spec,
            sample_weight=weights,
            max_rounds=max_rounds,
            early_stopping_rounds=early_stopping_rounds,
            device=device,
            n_jobs=n_jobs,
            seed=BASE_SEED,
        )
        metric_rows.append(
            {
                "group_id": group_id,
                "fold": int(fold.fold_number),
                "raw_feature_count": int(len(features)),
                "transformed_column_count": int(train_matrix.shape[1]),
                "seed_count": 1,
                "seeds": str(BASE_SEED),
                "train_rows": int(len(fold.train)),
                "validation_rows": int(len(fold.validation)),
                "best_iteration_zero_based": int(best_iteration),
                "best_rounds": int(best_iteration + 1),
                "elapsed_seconds": elapsed,
                **metrics,
            }
        )
        gain_frames.append(
            aggregate_raw_importance(
                model.get_booster(),
                ranges,
                fold_number=int(fold.fold_number),
            )
        )
        del fold, train_matrix, validation_matrix, model
        gc.collect()




    full_pool_metrics = pd.DataFrame(metric_rows)
    gain_long = pd.concat(gain_frames, ignore_index=True)
    gain_ranking = aggregate_gain_ranking(gain_long)
    return full_pool_metrics, gain_long, gain_ranking








def evaluate_feature_groups_multiseed(
    *,
    modeling_utils: Any,
    selection_frame: pd.DataFrame,
    inner_folds: Sequence[Mapping[str, Any]],
    feature_types: Mapping[str, str],
    groups: Mapping[str, Sequence[str]],
    config: Any,
    history: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    seeds: Sequence[int],
    train_max_rows: int,
    validation_max_rows: int,
    max_rounds: int,
    early_stopping_rounds: int,
    device: str,
    n_jobs: int,
    rare_min_frequency: float,
    stage_label: str = "CONFIRM",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold_position, fold_def in enumerate(inner_folds, start=1):
        fold = prepare_fold(
            modeling_utils=modeling_utils,
            frame=selection_frame,
            fold_def=fold_def,
            feature_types=feature_types,
            config=config,
            strategy=history,
            train_max_rows=train_max_rows,
            validation_max_rows=validation_max_rows,
            rare_min_frequency=rare_min_frequency,
        )
        weights = training_weights(fold.train, config.cohort_date_col, history)
        for group_position, (group_id, features) in enumerate(groups.items(), start=1):
            print(
                f"{stage_label}: fold {fold_position}/{len(inner_folds)}, group "
                f"{group_position}/{len(groups)} {group_id}, "
                f"{len(features)} features, seeds={list(seeds)}",
                flush=True,
            )
            train_matrix, validation_matrix, _ = build_sparse_matrices(
                fold, features
            )
            seed_probabilities: list[np.ndarray] = []
            seed_rounds: list[int] = []
            elapsed_total = 0.0
            for seed in seeds:
                model, p_churn, _, best_iteration, elapsed = fit_one_model(
                    modeling_utils=modeling_utils,
                    fold=fold,
                    train_matrix=train_matrix,
                    validation_matrix=validation_matrix,
                    spec=model_spec,
                    sample_weight=weights,
                    max_rounds=max_rounds,
                    early_stopping_rounds=early_stopping_rounds,
                    device=device,
                    n_jobs=n_jobs,
                    seed=int(seed),
                )
                seed_probabilities.append(p_churn)
                seed_rounds.append(int(best_iteration + 1))
                elapsed_total += elapsed
                del model
                gc.collect()
            ensemble = np.mean(np.vstack(seed_probabilities), axis=0)
            metrics = metric_dict(modeling_utils, fold, ensemble)
            rows.append(
                {
                    "group_id": group_id,
                    "fold": int(fold.fold_number),
                    "raw_feature_count": int(len(features)),
                    "transformed_column_count": int(train_matrix.shape[1]),
                    "seed_count": int(len(seeds)),
                    "seeds": ";".join(str(seed) for seed in seeds),
                    "train_rows": int(len(fold.train)),
                    "validation_rows": int(len(fold.validation)),
                    "best_rounds": int(round(float(np.median(seed_rounds)))),
                    "elapsed_seconds": elapsed_total,
                    **metrics,
                }
            )
            del train_matrix, validation_matrix, ensemble, seed_probabilities
            gc.collect()
        del fold
        gc.collect()
    return pd.DataFrame(rows)








def governance_warnings(
    candidate_audit: pd.DataFrame,
    selected_features: Sequence[str],
) -> list[dict[str, str]]:
    if candidate_audit.empty:
        return []
    indexed = candidate_audit.set_index("feature_name", drop=False)
    warnings: list[dict[str, str]] = []
    for feature in selected_features:
        if feature not in indexed.index:
            continue
        row = indexed.loc[feature]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        governance = str(row.get("governance_status", "")).strip()
        enabled = str(row.get("original_enabled_flag", "")).strip()
        note = str(row.get("note", "")).strip()
        if governance.upper() not in {"", "STANDARD_ELIGIBLE"} or enabled.upper() != "Y":
            warnings.append(
                {
                    "feature": feature,
                    "original_enabled_flag": enabled,
                    "governance_status": governance,
                    "note": note,
                }
            )
    return warnings




def write_selected_model_diagnostics(
    *,
    modeling_utils: Any,
    selection_frame: pd.DataFrame,
    inner_folds: Sequence[Mapping[str, Any]],
    feature_types: Mapping[str, str],
    selected_features: Sequence[str],
    config: Any,
    history: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    seeds: Sequence[int],
    train_max_rows: int,
    validation_max_rows: int,
    max_rounds: int,
    early_stopping_rounds: int,
    device: str,
    n_jobs: int,
    rare_min_frequency: float,
    output_dir: Path,
) -> dict[str, Any]:
    """Write calendar-controlled OOF diagnostics and XGBoost SHAP outputs."""


    with_year = list(selected_features)
    non_control_features = [
        feature for feature in with_year if feature != YEAR_CONTROL_FEATURE
    ]
    if YEAR_CONTROL_FEATURE not in with_year or not non_control_features:
        raise RuntimeError(
            "XGBoost diagnostics require calendar year plus at least one "
            "non-control feature."
        )
    policy_start_col = str(getattr(config, "anchor_term_start_col", ""))
    if not policy_start_col:
        raise RuntimeError("The modeling contract has no policy-start column.")


    parts: list[pd.DataFrame] = []


    def raw_values(fold: Any, frame: pd.DataFrame, feature: str) -> pd.Series | None:
        if feature in frame:
            return frame[feature]
        if feature == getattr(modeling_utils, "WRITTEN_PREM_BIN_FEATURE", ""):
            block = fold.get_feature_block(feature)
            source = str(block.fitted_artifact["source_feature"])
            return modeling_utils.apply_quantile_bin_artifact(
                frame[source], block.fitted_artifact
            )
        if feature == getattr(modeling_utils, "STATE_GROUP_FEATURE", ""):
            # FeatureBlock.fitted_artifact is an export-safe dictionary.  The
            # state grouping API requires the live object cached by the fold.
            fold.get_feature_block(feature)
            fitted_state_grouper = getattr(
                fold, "_derived_feature_artifacts", {}
            ).get(feature)
            if fitted_state_grouper is None or not hasattr(
                fitted_state_grouper, "spec"
            ):
                raise RuntimeError(
                    "Missing live fitted state grouper for XGBoost "
                    f"fold {fold.fold_number}."
                )
            _, apply_state_grouper, _ = modeling_utils._state_grouping_api()
            grouped, _ = apply_state_grouper(
                fitted_state_grouper,
                frame,
                scope=f"fold_{fold.fold_number}_diagnostic",
            )
            return grouped
        return None


    newest_models: list[tuple[xgb.XGBClassifier, int]] = []
    newest_matrix: sparse.csr_matrix | None = None
    newest_ranges: list[dict[str, Any]] | None = None
    newest_validation: pd.DataFrame | None = None
    newest_feature_values: pd.DataFrame | None = None
    for fold_position, fold_def in enumerate(inner_folds, start=1):
        fold = prepare_fold(
            modeling_utils=modeling_utils,
            frame=selection_frame,
            fold_def=fold_def,
            feature_types=feature_types,
            config=config,
            strategy=history,
            train_max_rows=train_max_rows,
            validation_max_rows=validation_max_rows,
            rare_min_frequency=rare_min_frequency,
        )
        if policy_start_col not in fold.validation:
            raise KeyError(
                f"Policy-start column {policy_start_col!r} is unavailable for "
                "calendar diagnostics."
            )
        if YEAR_CONTROL_FEATURE not in fold.validation:
            fold.validation[YEAR_CONTROL_FEATURE] = pd.to_datetime(
                fold.validation[config.cohort_date_col], errors="raise"
            ).dt.year.astype(int)
        weights = training_weights(fold.train, config.cohort_date_col, history)
        train_matrix, validation_matrix, ranges = build_sparse_matrices(
            fold, with_year
        )
        probabilities: list[np.ndarray] = []
        fitted_models: list[tuple[xgb.XGBClassifier, int]] = []
        for seed in seeds:
            model, p_churn, _, best_iteration, _ = fit_one_model(
                modeling_utils=modeling_utils,
                fold=fold,
                train_matrix=train_matrix,
                validation_matrix=validation_matrix,
                spec=model_spec,
                sample_weight=weights,
                max_rounds=max_rounds,
                early_stopping_rounds=early_stopping_rounds,
                device=device,
                n_jobs=n_jobs,
                seed=int(seed),
            )
            probabilities.append(1.0 - p_churn)
            if fold_position == len(inner_folds):
                fitted_models.append((model, best_iteration))
            else:
                del model
        predicted_retention = np.mean(np.vstack(probabilities), axis=0)
        if fold_position == len(inner_folds):
            newest_models = fitted_models
            newest_matrix = validation_matrix
            newest_ranges = ranges
            newest_validation = fold.validation.copy()
            newest_feature_values = pd.DataFrame(index=fold.validation.index)
            for feature in with_year:
                values = raw_values(fold, fold.validation, feature)
                if values is not None:
                    newest_feature_values[feature] = values.to_numpy()
        else:
            del train_matrix, validation_matrix


        validation = fold.validation
        diagnostic = pd.DataFrame(
            {
                "fold": int(fold.fold_number),
                "row_id": [
                    f"{int(fold.fold_number)}:{value}" for value in validation.index
                ],
                ACTUAL_COLUMN: pd.to_numeric(
                    validation[config.target_col], errors="raise"
                ).astype(int).to_numpy(),
                WITH_YEAR_COLUMN: predicted_retention,
                YEAR_CONTROL_FEATURE: pd.to_numeric(
                    validation[YEAR_CONTROL_FEATURE], errors="raise"
                ).astype(int).to_numpy(),
                YEAR_MONTH_COLUMN: pd.to_datetime(
                    validation[policy_start_col], errors="raise"
                ).dt.to_period("M").astype(str).to_numpy(),
            }
        )
        for feature in with_year:
            if feature in diagnostic:
                continue
            values = raw_values(fold, validation, feature)
            if values is not None:
                diagnostic[feature] = values.to_numpy()
        parts.append(diagnostic)
        del fold
        gc.collect()


    if (
        newest_matrix is None
        or newest_ranges is None
        or newest_validation is None
        or newest_feature_values is None
    ):
        raise RuntimeError("XGBoost diagnostics did not retain the newest fold.")
    sample_size = min(5_000, newest_matrix.shape[0])
    sample_positions = np.sort(
        np.random.default_rng(BASE_SEED).choice(
            newest_matrix.shape[0], size=sample_size, replace=False
        )
    )
    transformed_parts: list[np.ndarray] = []
    sample_matrix = newest_matrix[sample_positions]
    for model, best_iteration in newest_models:
        transformed_parts.append(
            tree_shap_values(
                model,
                sample_matrix,
                best_iteration=int(best_iteration),
            )
        )
        del model
    transformed_shap = np.mean(np.stack(transformed_parts), axis=0)
    raw_shap, shap_features = aggregate_raw_shap(
        transformed_shap, newest_ranges
    )
    sampled_validation = newest_validation.iloc[sample_positions]
    sampled_feature_values = newest_feature_values.iloc[sample_positions]
    feature_values = pd.DataFrame(index=np.arange(sample_size))
    for feature in shap_features:
        if feature in sampled_feature_values:
            feature_values[feature] = sampled_feature_values[feature].to_numpy()
        elif feature == YEAR_CONTROL_FEATURE:
            feature_values[feature] = pd.to_datetime(
                sampled_validation[config.cohort_date_col], errors="raise"
            ).dt.year.to_numpy()
        else:
            feature_values[feature] = "__DERIVED_OR_ENCODED__"
    diagnostics_dir = output_dir / "diagnostics"
    top_shap_features = write_xgboost_shap_artifacts(
        raw_shap=raw_shap,
        feature_values=feature_values,
        features=shap_features,
        output_dir=diagnostics_dir,
        top_n=15,
        feature_types=feature_types,
    )
    shap_plot_manifest_path = diagnostics_dir / "xgboost_shap_plot_manifest.json"
    shap_plot_manifest = json.loads(
        shap_plot_manifest_path.read_text(encoding="utf-8")
    )
    shap_artifacts = shap_plot_manifest.get("artifacts")
    if not isinstance(shap_artifacts, list) or not all(
        isinstance(value, str) and value for value in shap_artifacts
    ):
        raise RuntimeError("Invalid XGBoost SHAP plot manifest artifacts.")
    manifest = write_calendar_control_artifacts(
        predictions=pd.concat(parts, ignore_index=True),
        output_dir=diagnostics_dir,
        model_family="xgboost",
        one_way_features=top_shap_features,
        feature_types=feature_types,
    )
    manifest["shap_scope"] = "xgboost_newest_chronological_fold_sample"
    manifest["shap_package"] = "shap"
    manifest["shap_version"] = shap.__version__
    manifest["shap_explainer"] = "shap.TreeExplainer"
    manifest["shap_plot_api"] = ["shap.plots.beeswarm", "shap.plots.scatter"]
    manifest["shap_sample_rows"] = int(sample_size)
    manifest["shap_top_features"] = top_shap_features
    manifest["shap_scatter_features"] = shap_plot_manifest.get(
        "scatter_features", []
    )
    manifest["shap_scatter_control_feature_excluded"] = (
        shap_plot_manifest.get("scatter_control_feature_excluded")
    )
    manifest["artifacts"] = [
        *manifest["artifacts"],
        *shap_artifacts,
    ]
    (diagnostics_dir / "xgboost_diagnostics_manifest.json").write_text(
        json.dumps(strict_json_value(manifest), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return manifest




def add_business_retention_trend_diagnostics(
    *,
    diagnostics_manifest: Mapping[str, Any],
    population: pd.DataFrame,
    test_idx: np.ndarray,
    config: Any,
    selected_history: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Add full-history and selected-history actual-only business charts."""


    holdout_dates = pd.to_datetime(
        population.loc[test_idx, config.cohort_date_col], errors="raise"
    )
    monthly = business_retention_monthly_summary(
        population=population,
        target_col=config.target_col,
        cohort_col=config.cohort_date_col,
        end_month_exclusive=holdout_dates.min(),
    )
    business_manifest = write_business_retention_trend_artifacts(
        monthly=monthly,
        output_dir=output_dir / "diagnostics",
        model_family="xgboost",
        cohort_col=config.cohort_date_col,
        selected_history=selected_history,
    )
    merged = {
        **diagnostics_manifest,
        "artifacts": list(
            dict.fromkeys(
                [
                    *diagnostics_manifest["artifacts"],
                    *business_manifest["artifacts"],
                    business_manifest["manifest"],
                ]
            )
        ),
        "business_retention_trend": business_manifest,
    }
    (output_dir / "diagnostics" / "xgboost_diagnostics_manifest.json").write_text(
        json.dumps(strict_json_value(merged), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return merged




POST_SELECTION_ARTIFACTS = (
    "xgb_candidate_universe_audit.csv",
    "xgb_manifest_driven_candidate_universe.txt",
    "xgb_training_only_quality_screen.csv",
    "xgb_quality_screened_feature_universe.txt",
    "xgb_resolved_model_feature_types.json",
    "xgb_feature_discovery_inner_folds.csv",
    "xgb_configuration_screen_long.csv",
    "xgb_optuna_trials.csv",
    "xgb_optuna_study.json",
    "xgb_configuration_screen_summary.csv",
    "xgb_configuration_confirmation_long.csv",
    "xgb_configuration_confirmation_summary.csv",
    "xgb_gain_ranking_model_metrics.csv",
    "xgb_gain_ranking_long.csv",
    "xgb_gain_ranking_summary.csv",
    "xgb_cumulative_forward_coarse_summary.csv",
    "xgb_cumulative_forward_metrics_long.csv",
    "xgb_cumulative_forward_summary.csv",
    "xgb_cumulative_forward_path.csv",
    "xgb_confirmation_group_inventory.csv",
    "xgb_feature_group_confirmation_long.csv",
    "xgb_feature_group_confirmation_summary.csv",
    *FINALIST_ARTIFACTS,
    STAGE4_SELECTION_CHECKPOINT,
)




def _read_stage4_selection_checkpoint(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Read and validate the versioned Stage-4 recovery contract."""


    path = output_dir / STAGE4_SELECTION_CHECKPOINT
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(
            "Versioned Stage-4 selection checkpoint is missing; legacy "
            "post-selection replay is rejected under the paired one-SE policy."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Invalid Stage-4 selection checkpoint.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Stage-4 selection checkpoint must be a JSON object.")
    expected = {
        "contract_version": STAGE4_SELECTION_CONTRACT_VERSION,
        "status": "RESOLVED",
        "selection_rule": STAGE4_SELECTION_RULE,
        "profile_hash": PROFILE_HASH,
        "seeds": [int(value) for value in seeds],
        "confirm_candidate_count_per_pass": int(args.confirm_candidate_count),
        "confirm_max_passes": int(args.confirm_max_passes),
        "finalist_tuning_contract_version": FINALIST_TUNING_CONTRACT_VERSION,
        "finalist_equal_adaptive_trials": int(args.optuna_trials),
        "finalist_optuna_timeout_seconds": FINALIST_TUNING_TIMEOUT_SECONDS,
        "finalist_selection_rule": FINAL_FEATURE_SELECTION_RULE,
        "final_feature_selection_policy": final_selection_policy(args),
        "final_feature_selection_policy_sha256": policy_sha256(final_selection_policy(args)),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(
                f"Stage-4 selection checkpoint changed {name}: "
                f"{payload.get(name)!r} != {value!r}."
            )
    selected_group_id = str(payload.get("selected_group_id", ""))
    discovery_selected_group_id = str(
        payload.get("discovery_selected_group_id", "")
    )
    finalist_group_ids = payload.get("finalist_group_ids")
    selected_configuration = payload.get("selected_configuration")
    confirmed_group_ids = payload.get("confirmed_group_ids")
    pass_records = payload.get("passes")
    artifacts = payload.get("bound_artifacts")
    if (
        not selected_group_id
        or not discovery_selected_group_id
        or not isinstance(finalist_group_ids, list)
        or selected_group_id not in [str(value) for value in finalist_group_ids]
        or discovery_selected_group_id not in [
            str(value) for value in confirmed_group_ids or []
        ]
        or not isinstance(selected_configuration, dict)
        or not isinstance(confirmed_group_ids, list)
        or selected_group_id not in [str(value) for value in confirmed_group_ids]
        or not isinstance(pass_records, list)
        or not pass_records
        or not isinstance(artifacts, dict)
    ):
        raise RuntimeError("Stage-4 selection checkpoint is incomplete.")
    if payload.get("completed_passes") != len(pass_records):
        raise RuntimeError("Stage-4 checkpoint pass count is inconsistent.")
    flattened_pass_groups: list[str] = []
    expected_pass_number = 1
    for record in pass_records:
        if not isinstance(record, dict) or record.get("pass") != expected_pass_number:
            raise RuntimeError("Stage-4 checkpoint pass sequence is invalid.")
        evaluated = record.get("evaluated_group_ids")
        if not isinstance(evaluated, list) or not evaluated:
            raise RuntimeError("Stage-4 checkpoint pass has no evaluated groups.")
        flattened_pass_groups.extend(str(value) for value in evaluated)
        expected_pass_number += 1
    normalized_confirmed = [str(value) for value in confirmed_group_ids]
    if (
        len(flattened_pass_groups) != len(set(flattened_pass_groups))
        or flattened_pass_groups != normalized_confirmed
    ):
        raise RuntimeError("Stage-4 checkpoint pass inventory is inconsistent.")
    for name in (
        "xgb_confirmation_group_inventory.csv",
        "xgb_feature_group_confirmation_long.csv",
        "xgb_feature_group_confirmation_summary.csv",
        *FINALIST_ARTIFACTS,
    ):
        expected_hash = str(artifacts.get(name, {}).get("sha256", ""))
        artifact_path = output_dir / name
        if not expected_hash or not artifact_path.is_file():
            raise RuntimeError(f"Stage-4 checkpoint did not bind {name}.")
        if sha256(artifact_path) != expected_hash:
            raise RuntimeError(f"Stage-4 checkpoint artifact changed: {name}.")
    return payload




def write_stage4_selection_checkpoint(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    seeds: Sequence[int],
    status: str,
    full_group_id: str,
    discovery_selected_group_id: str,
    selected_group_id: str | None,
    selected_features: Sequence[str],
    finalist_group_ids: Sequence[str],
    selected_configuration: Mapping[str, Any],
    required_group_ids: Sequence[str],
    confirmed_group_ids: Sequence[str],
    pending_required_ids: Sequence[str],
    pending_adaptive_ids: Sequence[str],
    pass_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Atomically publish the signed Stage-4 selection/recovery boundary."""


    normalized_finalists = [str(value) for value in finalist_group_ids]
    normalized_confirmed = [str(value) for value in confirmed_group_ids]
    if status != "RESOLVED":
        raise ValueError("The v3 checkpoint is published only for resolved runs.")
    if (
        selected_group_id is None
        or str(selected_group_id) not in normalized_finalists
        or str(selected_group_id) not in normalized_confirmed
        or str(discovery_selected_group_id) not in normalized_confirmed
        or len(normalized_finalists) != len(set(normalized_finalists))
        or not normalized_finalists
    ):
        raise ValueError("Invalid v3 finalist/confirmation checkpoint inventory.")
    if str(selected_configuration.get("group_id", "")) != str(selected_group_id):
        raise ValueError("Selected configuration does not belong to the winner.")
    bound_names = (
        "xgb_confirmation_group_inventory.csv",
        "xgb_feature_group_confirmation_long.csv",
        "xgb_feature_group_confirmation_summary.csv",
        *FINALIST_ARTIFACTS,
    )
    missing_bound_artifacts = [
        name for name in bound_names if not (output_dir / name).is_file()
    ]
    if missing_bound_artifacts:
        raise RuntimeError(
            "Cannot publish v3 checkpoint; finalist artifacts are missing: "
            f"{missing_bound_artifacts}"
        )
    bound_artifacts = {
        name: file_info(output_dir / name)
        for name in bound_names
        if (output_dir / name).is_file()
    }
    payload = {
        "contract_version": STAGE4_SELECTION_CONTRACT_VERSION,
        "status": str(status),
        "selection_rule": STAGE4_SELECTION_RULE,
        "selection_unit": "chronological_fold_after_multi_seed_prediction_ensemble",
        "profile_hash": PROFILE_HASH,
        "profile": PROFILE_PAYLOAD,
        "seeds": [int(value) for value in seeds],
        "confirm_candidate_count_per_pass": int(args.confirm_candidate_count),
        "confirm_max_passes": int(args.confirm_max_passes),
        "completed_passes": int(len(pass_records)),
        "full_group_id": str(full_group_id),
        "discovery_selected_group_id": str(discovery_selected_group_id),
        "selected_group_id": (
            None if selected_group_id is None else str(selected_group_id)
        ),
        "selected_raw_feature_count": int(len(selected_features)),
        "selected_feature_sha256": feature_sequence_sha256(selected_features),
        "finalist_tuning_contract_version": FINALIST_TUNING_CONTRACT_VERSION,
        "finalist_selection_rule": FINAL_FEATURE_SELECTION_RULE,
        "final_feature_selection_policy": final_selection_policy(args),
        "final_feature_selection_policy_sha256": policy_sha256(final_selection_policy(args)),
        "selection_policy_module": file_info(Path(__file__).with_name("xgboost_selection_policy.py")),
        "finalist_group_ids": normalized_finalists,
        "finalist_count": int(len(normalized_finalists)),
        "finalist_union_policy": (
            "ALL_STAGE4_ONE_SE_ELIGIBLE_AND_COUNT_AP_RECALL_PARETO_PLUS_ANCHORS"
        ),
        "discovery_performance_hierarchy": (
            "AP_SKILL_ONE_SE_THEN_CONDITIONAL_RECALL_10PCT_ONE_SE"
        ),
        "discovery_recall_reference_condition": (
            "BEST_MEAN_RECALL_AMONG_AP_SKILL_ONE_SE_PASS_CANDIDATES"
        ),
        "finalist_equal_adaptive_trials": int(args.optuna_trials),
        "finalist_optuna_timeout_seconds": FINALIST_TUNING_TIMEOUT_SECONDS,
        "performance_hierarchy": "PRACTICAL_MEAN_AND_WORST_FOLD_AP_THEN_CONDITIONAL_RECALL_LIMITS",
        "recall_reference_condition": "BEST_MEAN_RECALL_AMONG_AP_PRACTICAL_LIMIT_PASS_CANDIDATES",
        "selected_configuration": strict_json_value(dict(selected_configuration)),
        "required_group_ids": [str(value) for value in required_group_ids],
        "confirmed_group_ids": normalized_confirmed,
        "pending_required_ids": [str(value) for value in pending_required_ids],
        "pending_adaptive_ids": [str(value) for value in pending_adaptive_ids],
        "passes": [strict_json_value(dict(value)) for value in pass_records],
        "bound_artifacts": bound_artifacts,
        "outer_validation_rows_scored": 0,
        "final_holdout_rows_scored": 0,
    }
    path = output_dir / STAGE4_SELECTION_CHECKPOINT
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(strict_json_value(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return payload




def _read_recovery_csv(output_dir: Path, name: str) -> pd.DataFrame:
    path = output_dir / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Post-selection recovery artifact is missing: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Invalid post-selection recovery CSV: {path}") from exc
    if frame.empty:
        raise RuntimeError(f"Post-selection recovery CSV is empty: {path}")
    return frame




def _truth_values(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})




def _feature_inventory(frame: pd.DataFrame) -> dict[str, list[str]]:
    required = {"group_id", "raw_feature_count", "features"}
    if not required.issubset(frame.columns):
        raise RuntimeError("Confirmation inventory is missing required columns.")
    if frame["group_id"].astype(str).duplicated().any():
        raise RuntimeError("Confirmation inventory contains duplicate group IDs.")
    output: dict[str, list[str]] = {}
    for row in frame.itertuples(index=False):
        group_id = str(row.group_id)
        features = [
            value.strip().upper()
            for value in str(row.features).split(";")
            if value.strip()
        ]
        if not features or len(features) != len(set(features)):
            raise RuntimeError(f"Invalid feature inventory for {group_id}.")
        if int(row.raw_feature_count) != len(features):
            raise RuntimeError(f"Feature count mismatch for {group_id}.")
        match = re.fullmatch(r"XGB_GAIN_FORWARD_N(\d+)", group_id)
        if match is None or int(match.group(1)) != len(features):
            raise RuntimeError(
                f"Feature-group identifier does not encode its count: {group_id}."
            )
        output[group_id] = features
    return output




def _assert_recomputed_summary(
    saved: pd.DataFrame,
    recomputed: pd.DataFrame,
    *,
    id_col: str,
    label: str,
) -> None:
    if id_col not in saved or id_col not in recomputed:
        raise RuntimeError(f"{label} summary is missing {id_col}.")
    if saved[id_col].astype(str).duplicated().any():
        raise RuntimeError(f"{label} saved summary has duplicate identifiers.")
    saved_by_id = saved.set_index(saved[id_col].astype(str), drop=False)
    recomputed_by_id = recomputed.set_index(
        recomputed[id_col].astype(str), drop=False
    )
    if set(saved_by_id.index) != set(recomputed_by_id.index):
        raise RuntimeError(f"{label} saved and recomputed identifiers disagree.")
    missing_columns = sorted(set(recomputed.columns).difference(saved.columns))
    if missing_columns:
        raise RuntimeError(
            f"{label} saved summary is missing recomputed columns: "
            f"{missing_columns}"
        )
    for column in recomputed.columns:
        if column == id_col:
            continue
        left = saved_by_id.loc[recomputed_by_id.index, column]
        right = recomputed_by_id[column]
        left_numeric = pd.to_numeric(left, errors="coerce")
        right_numeric = pd.to_numeric(right, errors="coerce")
        numeric = left_numeric.notna() & right_numeric.notna()
        if numeric.any() and not np.allclose(
            left_numeric.loc[numeric].to_numpy(dtype=float),
            right_numeric.loc[numeric].to_numpy(dtype=float),
            rtol=1.0e-10,
            atol=1.0e-12,
            equal_nan=True,
        ):
            raise RuntimeError(f"{label} summary changed numeric column {column}.")
        non_numeric = ~numeric
        if non_numeric.any() and not left.loc[non_numeric].astype(str).equals(
            right.loc[non_numeric].astype(str)
        ):
            raise RuntimeError(f"{label} summary changed column {column}.")




def _recovery_configuration(
    screen: pd.DataFrame,
    selected_config_id: str,
    histories: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], int]:
    matches = screen.loc[screen["candidate_id"].astype(str).eq(selected_config_id)]
    if len(matches) != 1:
        raise RuntimeError(
            "Selected configuration must have exactly one Optuna screen row; "
            f"found {len(matches)} for {selected_config_id}."
        )
    row = matches.iloc[0]
    history_id = str(row["history_id"])
    history_lookup = {str(item["history_id"]): dict(item) for item in histories}
    if history_id not in history_lookup:
        raise RuntimeError(f"Unknown recovered history treatment: {history_id}")
    selected_history = history_lookup[history_id]
    observed_history = {
        "kind": str(row["history_kind"]),
        "months": int(row["history_months"]),
        "max_months": int(row["history_max_months"]),
    }
    expected_history = {
        "kind": str(selected_history["kind"]),
        "months": int(selected_history["months"]),
        "max_months": int(
            selected_history.get("max_months", selected_history["months"])
        ),
    }
    if observed_history != expected_history:
        raise RuntimeError("Recovered Optuna history metadata is inconsistent.")
    model_id = str(row["model_id"])
    if selected_config_id != f"{history_id}__{model_id}":
        raise RuntimeError("Recovered configuration ID is internally inconsistent.")


    def number(name: str, cast: Any) -> Any:
        value = pd.to_numeric(pd.Series([row[name]]), errors="raise").iloc[0]
        if not np.isfinite(float(value)):
            raise RuntimeError(f"Recovered Optuna parameter is non-finite: {name}")
        if cast is int and float(value) != float(int(value)):
            raise RuntimeError(f"Recovered Optuna parameter is not integral: {name}")
        return cast(value)


    model = {
        "model_id": model_id,
        "max_depth": number("param_max_depth", int),
        "learning_rate": number("param_learning_rate", float),
        "min_child_weight": number("param_min_child_weight", float),
        "gamma": number("param_gamma", float),
        "subsample": number("param_subsample", float),
        "colsample_bytree": number("param_colsample_bytree", float),
        "reg_lambda": number("param_reg_lambda", float),
        "reg_alpha": number("param_reg_alpha", float),
        "grow_policy": str(row["param_grow_policy"]),
        "max_leaves": number("param_max_leaves_resolved", int),
        "max_bin": number("param_max_bin", int),
    }
    if model["grow_policy"] not in {"depthwise", "lossguide"}:
        raise RuntimeError("Recovered Optuna grow_policy is invalid.")
    if model["grow_policy"] == "depthwise" and model["max_leaves"] != 0:
        raise RuntimeError("Depthwise recovery unexpectedly has nonzero max_leaves.")
    trial_number = number("optuna_trial_number", int)
    if model_id != f"OPTUNA_T{trial_number:04d}":
        raise RuntimeError("Recovered Optuna model ID does not match its trial number.")
    bounds = {
        "max_depth": (2, 8),
        "learning_rate": (0.01, 0.10),
        "min_child_weight": (10.0, 150.0),
        "gamma": (0.0, 1.0),
        "subsample": (0.65, 1.0),
        "colsample_bytree": (0.60, 1.0),
        "reg_lambda": (1.0, 100.0),
        "reg_alpha": (1.0e-4, 5.0),
    }
    for name, (lower, upper) in bounds.items():
        if not lower <= model[name] <= upper:
            raise RuntimeError(f"Recovered Optuna parameter is out of bounds: {name}")
    if model["max_bin"] != 256:
        raise RuntimeError("Recovered Optuna max_bin is not the fixed value 256.")
    if model["grow_policy"] == "lossguide" and (
        not 16 <= model["max_leaves"] <= 128
        or model["max_leaves"] % 16 != 0
    ):
        raise RuntimeError("Recovered Optuna max_leaves is outside its search space.")
    return selected_history, model, trial_number




def _validate_optuna_recovery_metadata(
    *,
    screen: pd.DataFrame,
    metadata: Mapping[str, Any],
    histories: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Bind the saved trial table to the study contract used by Stage 2."""


    required_columns = {
        "candidate_id",
        "history_id",
        "model_id",
        "optuna_trial_number",
        "optuna_objective_value",
        "churn_ap_skill",
        "param_max_depth",
        "param_learning_rate",
        "param_min_child_weight",
        "param_gamma",
        "param_subsample",
        "param_colsample_bytree",
        "param_reg_lambda",
        "param_reg_alpha",
        "param_grow_policy",
        "param_max_leaves_resolved",
        "param_max_bin",
    }
    missing = sorted(required_columns.difference(screen.columns))
    if missing:
        raise RuntimeError(f"Recovered Optuna trial table is missing columns: {missing}")
    raw_trial_numbers = pd.to_numeric(
        screen["optuna_trial_number"], errors="raise"
    )
    if not np.isfinite(raw_trial_numbers.to_numpy(dtype=float)).all() or not np.equal(
        raw_trial_numbers.to_numpy(dtype=float),
        raw_trial_numbers.astype(int).to_numpy(dtype=float),
    ).all():
        raise RuntimeError("Recovered Optuna trial numbers must be finite integers.")
    trial_numbers = raw_trial_numbers.astype(int)
    if trial_numbers.duplicated().any() or sorted(trial_numbers.tolist()) != list(
        range(len(screen))
    ):
        raise RuntimeError("Recovered Optuna trial numbers are not unique and contiguous.")
    expected_model_ids = trial_numbers.map(lambda value: f"OPTUNA_T{value:04d}")
    if not screen["model_id"].astype(str).reset_index(drop=True).equals(
        expected_model_ids.astype(str).reset_index(drop=True)
    ):
        raise RuntimeError("Recovered Optuna model IDs do not match trial numbers.")
    expected_candidate_ids = (
        screen["history_id"].astype(str)
        + "__"
        + screen["model_id"].astype(str)
    )
    if not screen["candidate_id"].astype(str).equals(expected_candidate_ids):
        raise RuntimeError("Recovered Optuna candidate IDs are inconsistent.")
    objective = pd.to_numeric(screen["optuna_objective_value"], errors="raise")
    ap_skill_values = pd.to_numeric(screen["churn_ap_skill"], errors="raise")
    if not np.isfinite(objective.to_numpy(dtype=float)).all() or not np.allclose(
        objective.to_numpy(dtype=float),
        ap_skill_values.to_numpy(dtype=float),
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise RuntimeError("Recovered Optuna objectives do not equal churn AP skill.")


    expected_metadata = {
        "optuna_version": optuna.__version__,
        "sampler": "TPESampler",
        "direction": "maximize",
        "objective": "churn_ap_skill",
        "seed": int(BASE_SEED),
        "parallel_optuna_jobs": 1,
        "batch_size": int(args.optuna_batch_size),
        "adaptive_timeout_seconds": int(args.optuna_timeout_seconds),
        "search_space": optuna_search_space(),
        "history_candidates": [dict(value) for value in histories],
        "screened_inner_fold": int(args.inner_folds),
        "outer_validation_rows_scored": 0,
        "final_holdout_rows_scored": 0,
    }
    for name, expected in expected_metadata.items():
        if metadata.get(name) != expected:
            raise RuntimeError(
                f"Recovered Optuna study metadata changed {name}: "
                f"{metadata.get(name)!r} != {expected!r}."
            )
    integer_fields = (
        "anchor_trial_count",
        "requested_adaptive_trials",
        "completed_adaptive_trials",
        "completed_total_trials",
        "best_screen_trial_number",
    )
    parsed: dict[str, int] = {}
    for name in integer_fields:
        value = metadata.get(name)
        if isinstance(value, bool):
            raise RuntimeError(f"Recovered Optuna metadata has invalid {name}.")
        try:
            parsed[name] = int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Recovered Optuna metadata has invalid {name}."
            ) from exc
        if float(value) != float(parsed[name]):
            raise RuntimeError(f"Recovered Optuna metadata has non-integral {name}.")
    expected_anchor_count = len(histories) * len(anchor_model_grid())
    if parsed["anchor_trial_count"] != expected_anchor_count:
        raise RuntimeError("Recovered Optuna anchor trial count is inconsistent.")
    if parsed["requested_adaptive_trials"] != int(args.optuna_trials):
        raise RuntimeError("Recovered Optuna requested trial count changed.")
    if parsed["completed_total_trials"] != len(screen):
        raise RuntimeError("Recovered Optuna total trial count is inconsistent.")
    if (
        parsed["anchor_trial_count"] + parsed["completed_adaptive_trials"]
        != parsed["completed_total_trials"]
    ):
        raise RuntimeError("Recovered Optuna completed trial counts are inconsistent.")
    best_position = int(objective.idxmax())
    best_trial = int(trial_numbers.loc[best_position])
    best_value = float(objective.loc[best_position])
    if parsed["best_screen_trial_number"] != best_trial or not np.isclose(
        float(metadata.get("best_screen_objective_value")),
        best_value,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise RuntimeError("Recovered Optuna best-trial metadata is inconsistent.")




def load_post_selection_recovery(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    seeds: Sequence[int],
    model_features: Sequence[str],
    mandatory_control_features: Sequence[str],
    inner_fold_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Prove and reconstruct the completed Stage-4 selection boundary."""


    if not output_dir.is_dir() or output_dir.is_symlink():
        raise RuntimeError("Post-selection recovery requires a safe output directory.")
    if (output_dir / "run_info.json").exists():
        raise RuntimeError("A final XGBoost run receipt already exists; recovery refused.")
    for name in POST_SELECTION_ARTIFACTS:
        path = output_dir / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Incomplete Stage-4 recovery boundary: {path}")
    stage4_checkpoint = _read_stage4_selection_checkpoint(
        output_dir,
        args=args,
        seeds=seeds,
    )
    raise RuntimeError(
        "Post-selection recovery is intentionally disabled for the v3 "
        "group-specific finalist-Optuna contract. Start a fresh XGBoost "
        "selection stage; v2 checkpoints are not accepted as v3."
    )


    fold_path = output_dir / "xgb_feature_discovery_inner_folds.csv"
    if not fold_path.is_file():
        raise RuntimeError("Saved XGBoost inner-fold audit is missing.")
    expected_fold_text = inner_fold_frame.to_csv(index=False, lineterminator="\n")
    observed_fold_text = fold_path.read_text(encoding="utf-8-sig").replace(
        "\r\n", "\n"
    )
    if observed_fold_text != expected_fold_text:
        raise RuntimeError("Saved XGBoost inner folds differ from the frozen contract.")


    gain = _read_recovery_csv(output_dir, "xgb_gain_ranking_summary.csv")
    gain_long = _read_recovery_csv(output_dir, "xgb_gain_ranking_long.csv")
    expected_gain_pairs = {
        (feature, fold)
        for feature in model_features
        for fold in range(1, int(args.inner_folds) + 1)
    }
    observed_gain_pairs = list(
        zip(
            gain_long["raw_feature"].astype(str).str.upper(),
            pd.to_numeric(gain_long["fold"], errors="raise").astype(int),
        )
    )
    if len(observed_gain_pairs) != len(set(observed_gain_pairs)) or set(
        observed_gain_pairs
    ) != expected_gain_pairs:
        raise RuntimeError("Recovered gain feature-by-fold grid is incomplete.")
    for column in (
        "transformed_column_count",
        "total_gain",
        "normalized_total_gain",
        "split_count",
    ):
        values = pd.to_numeric(gain_long[column], errors="raise")
        if not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0).any():
            raise RuntimeError(f"Recovered gain values are invalid: {column}")
    if not pd.to_numeric(
        gain_long["transformed_column_count"], errors="raise"
    ).gt(0).all():
        raise RuntimeError("Recovered gain transformed-column counts are not positive.")
    recomputed_gain = annotate_forward_ranking(
        aggregate_gain_ranking(gain_long), mandatory_control_features
    )
    _assert_recomputed_summary(
        gain,
        recomputed_gain,
        id_col="raw_feature",
        label="Gain ranking",
    )
    required_gain = {"raw_feature", "forward_rank"}
    if not required_gain.issubset(gain.columns):
        raise RuntimeError("Gain ranking is missing recovery columns.")
    gain = gain.sort_values("forward_rank", kind="mergesort")
    ranks = pd.to_numeric(gain["forward_rank"], errors="raise").astype(int).tolist()
    if ranks != list(range(1, len(gain) + 1)):
        raise RuntimeError("Recovered forward ranks are not contiguous.")
    ranked_features = gain["raw_feature"].astype(str).str.upper().tolist()
    if len(ranked_features) != len(set(ranked_features)) or set(
        ranked_features
    ) != set(model_features):
        raise RuntimeError("Recovered gain universe differs from the frozen model pool.")
    if ranked_features[: len(mandatory_control_features)] != list(
        mandatory_control_features
    ):
        raise RuntimeError("Recovered gain ranking lost the mandatory control prefix.")


    path_frame = _read_recovery_csv(output_dir, "xgb_cumulative_forward_path.csv")
    required_path = {
        "forward_step",
        "group_id",
        "raw_feature_count",
        "features",
        "added_at_step",
        "refinement_group",
    }
    missing_path = sorted(required_path.difference(path_frame.columns))
    if missing_path:
        raise RuntimeError(f"Recovered forward path is missing columns: {missing_path}")
    forward_steps = pd.to_numeric(
        path_frame["forward_step"], errors="raise"
    ).astype(int).tolist()
    if forward_steps != list(range(1, len(path_frame) + 1)):
        raise RuntimeError("Recovered forward path steps are not contiguous.")
    path_groups = _feature_inventory(
        path_frame[["group_id", "raw_feature_count", "features"]]
    )
    path_counts = [len(features) for features in path_groups.values()]
    if path_counts != sorted(path_counts) or len(path_counts) != len(set(path_counts)):
        raise RuntimeError("Recovered forward feature counts are not strictly increasing.")
    previous_count = 0
    for row, (group_id, features) in zip(
        path_frame.itertuples(index=False), path_groups.items()
    ):
        added = [
            value.strip().upper()
            for value in str(row.added_at_step).split(";")
            if value.strip()
        ]
        if added != features[previous_count:]:
            raise RuntimeError(f"Recovered forward added-at-step changed {group_id}.")
        previous_count = len(features)
    for group_id, features in path_groups.items():
        if features != ranked_features[: len(features)]:
            raise RuntimeError(f"Recovered forward group is not an exact prefix: {group_id}")


    config_long = _read_recovery_csv(
        output_dir, "xgb_configuration_confirmation_long.csv"
    )
    optuna_screen = _read_recovery_csv(output_dir, "xgb_optuna_trials.csv")
    histories = history_grid()
    try:
        optuna_metadata = json.loads(
            (output_dir / "xgb_optuna_study.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Invalid recovered Optuna study metadata.") from exc
    if not isinstance(optuna_metadata, dict):
        raise RuntimeError("Recovered Optuna study metadata must be an object.")
    _validate_optuna_recovery_metadata(
        screen=optuna_screen,
        metadata=optuna_metadata,
        histories=histories,
        args=args,
    )
    # Validate every row's history, trial identity, and model search bounds, not
    # only the eventual configuration winner.
    for candidate_id in optuna_screen["candidate_id"].astype(str):
        _recovery_configuration(optuna_screen, candidate_id, histories)
    screen_summary = summarize_candidates(optuna_screen, "candidate_id")
    expected_config_ids = (
        screen_summary.sort_values(
            ["mean_churn_ap_skill", "mean_top10_lift", "mean_retention_roc_auc"],
            ascending=[False, False, False],
            kind="mergesort",
        )["candidate_id"]
        .head(max(1, int(args.confirm_top_configs)))
        .astype(str)
        .tolist()
    )
    observed_config_ids = config_long["candidate_id"].astype(str).drop_duplicates().tolist()
    if set(observed_config_ids) != set(expected_config_ids):
        raise RuntimeError("Recovered configuration-confirmation shortlist changed.")
    expected_config_pairs = {
        (candidate_id, fold)
        for candidate_id in expected_config_ids
        for fold in range(1, int(args.inner_folds) + 1)
    }
    observed_config_pairs = list(
        zip(
            config_long["candidate_id"].astype(str),
            pd.to_numeric(config_long["fold"], errors="raise").astype(int),
        )
    )
    if len(observed_config_pairs) != len(set(observed_config_pairs)) or set(
        observed_config_pairs
    ) != expected_config_pairs:
        raise RuntimeError(
            "Recovered configuration-confirmation fold grid is incomplete."
        )
    if not pd.to_numeric(
        config_long["raw_feature_count"], errors="raise"
    ).astype(int).eq(len(model_features)).all():
        raise RuntimeError("Recovered configuration raw-feature counts changed.")
    for metric in (
        "churn_ap_skill",
        "top10_lift",
        "retention_roc_auc",
        "best_rounds",
    ):
        values = pd.to_numeric(config_long[metric], errors="raise")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise RuntimeError(
                f"Recovered configuration metric is non-finite: {metric}"
            )
    recomputed_config_summary = summarize_candidates(config_long, "candidate_id")
    selected_config_id = choose_best_configuration(recomputed_config_summary)
    saved_config_summary = _read_recovery_csv(
        output_dir, "xgb_configuration_confirmation_summary.csv"
    )
    _assert_recomputed_summary(
        saved_config_summary,
        recomputed_config_summary,
        id_col="candidate_id",
        label="Configuration confirmation",
    )
    if choose_best_configuration(saved_config_summary) != selected_config_id:
        raise RuntimeError("Saved and recomputed configuration winners disagree.")
    selected_history, selected_model_spec, trial_number = _recovery_configuration(
        optuna_screen,
        selected_config_id,
        histories,
    )


    inventory = _feature_inventory(
        _read_recovery_csv(output_dir, "xgb_confirmation_group_inventory.csv")
    )
    ranked_features = (
        gain.sort_values("forward_rank", kind="stable")["raw_feature"]
        .astype(str)
        .tolist()
    )
    for group_id, features in inventory.items():
        if features != ranked_features[: len(features)]:
            raise RuntimeError(
                f"Confirmation inventory is not the saved gain prefix: {group_id}."
            )
        if group_id in path_groups and features != path_groups[group_id]:
            raise RuntimeError(f"Confirmation inventory changed {group_id}.")
    confirmation_long = _read_recovery_csv(
        output_dir, "xgb_feature_group_confirmation_long.csv"
    )
    expected_pairs = {
        (group_id, fold)
        for group_id in inventory
        for fold in range(1, int(args.inner_folds) + 1)
    }
    observed_pairs = list(
        zip(
            confirmation_long["group_id"].astype(str),
            pd.to_numeric(confirmation_long["fold"], errors="raise").astype(int),
        )
    )
    if len(observed_pairs) != len(set(observed_pairs)) or set(observed_pairs) != expected_pairs:
        raise RuntimeError("Confirmation group-by-fold grid is incomplete or duplicated.")
    if "seed_count" not in confirmation_long or not pd.to_numeric(
        confirmation_long["seed_count"], errors="raise"
    ).eq(len(seeds)).all():
        raise RuntimeError("Recovered confirmation seed count is inconsistent.")
    expected_seed_text = ";".join(str(value) for value in seeds)
    if "seeds" not in confirmation_long or not confirmation_long[
        "seeds"
    ].astype(str).eq(expected_seed_text).all():
        raise RuntimeError("Recovered confirmation seed identities are inconsistent.")
    expected_counts = {
        group_id: len(features) for group_id, features in inventory.items()
    }
    observed_counts = pd.to_numeric(
        confirmation_long["raw_feature_count"], errors="raise"
    ).astype(int)
    if not all(
        observed == expected_counts[group_id]
        for group_id, observed in zip(
            confirmation_long["group_id"].astype(str), observed_counts
        )
    ):
        raise RuntimeError("Recovered confirmation raw-feature counts disagree.")
    for metric in (
        "churn_ap_skill",
        "churn_recall_10pct",
        "top10_lift",
        "retention_roc_auc",
        "best_rounds",
    ):
        values = pd.to_numeric(confirmation_long[metric], errors="raise")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise RuntimeError(f"Recovered confirmation metric is non-finite: {metric}")
    selected_group_id, confirmation_annotated, selection_reason = (
        choose_paired_one_se_group(
            confirmation_long,
            id_col="group_id",
        )
    )
    saved_confirmation = _read_recovery_csv(
        output_dir, "xgb_feature_group_confirmation_summary.csv"
    )
    _assert_recomputed_summary(
        saved_confirmation,
        confirmation_annotated,
        id_col="group_id",
        label="Feature-group confirmation",
    )
    if "selected_by_rule" not in saved_confirmation:
        raise RuntimeError("Saved confirmation summary has no selected marker.")
    saved_selected = saved_confirmation.loc[
        _truth_values(saved_confirmation["selected_by_rule"]), "group_id"
    ].astype(str).tolist()
    if saved_selected != [selected_group_id] or set(
        saved_confirmation["group_id"].astype(str)
    ) != set(confirmation_annotated["group_id"].astype(str)):
        raise RuntimeError("Saved and recomputed feature-group winners disagree.")
    if str(stage4_checkpoint["selected_group_id"]) != selected_group_id:
        raise RuntimeError("Stage-4 checkpoint and recomputed winner disagree.")
    checkpoint_confirmed = [
        str(value) for value in stage4_checkpoint["confirmed_group_ids"]
    ]
    if list(inventory) != checkpoint_confirmed:
        raise RuntimeError("Stage-4 checkpoint confirmation inventory changed.")
    if selected_group_id not in inventory:
        raise RuntimeError("Recovered selected group is absent from its inventory.")
    selected_features = inventory[selected_group_id]
    if not set(mandatory_control_features).issubset(selected_features):
        raise RuntimeError("Recovered selected group lost a mandatory control.")
    recovered_feature_hash = hashlib.sha256(
        ("\n".join(selected_features) + "\n").encode("utf-8")
    ).hexdigest()
    if stage4_checkpoint.get("selected_feature_sha256") != recovered_feature_hash:
        raise RuntimeError("Stage-4 checkpoint selected-feature hash changed.")
    selected_rows = confirmation_long.loc[
        confirmation_long["group_id"].astype(str).eq(selected_group_id)
    ]
    selected_rounds = int(
        round(float(pd.to_numeric(selected_rows["best_rounds"]).median()))
    )
    selected_summary = confirmation_annotated.loc[
        confirmation_annotated["group_id"].astype(str).eq(selected_group_id)
    ].iloc[0].to_dict()


    forward_summary = _read_recovery_csv(
        output_dir, "xgb_cumulative_forward_summary.csv"
    )
    forward_long = _read_recovery_csv(
        output_dir, "xgb_cumulative_forward_metrics_long.csv"
    )
    expected_forward_pairs = {
        (group_id, fold)
        for group_id in path_groups
        for fold in range(1, int(args.inner_folds) + 1)
    }
    observed_forward_pairs = list(
        zip(
            forward_long["group_id"].astype(str),
            pd.to_numeric(forward_long["fold"], errors="raise").astype(int),
        )
    )
    if len(observed_forward_pairs) != len(set(observed_forward_pairs)) or set(
        observed_forward_pairs
    ) != expected_forward_pairs:
        raise RuntimeError("Recovered forward group-by-fold grid is incomplete.")
    if "seed_count" not in forward_long or not pd.to_numeric(
        forward_long["seed_count"], errors="raise"
    ).eq(1).all():
        raise RuntimeError("Recovered forward seed count is not one.")
    if "seeds" not in forward_long or not forward_long["seeds"].astype(str).eq(
        str(BASE_SEED)
    ).all():
        raise RuntimeError("Recovered forward seed identity changed.")
    forward_counts = pd.to_numeric(
        forward_long["raw_feature_count"], errors="raise"
    ).astype(int)
    if not all(
        observed == len(path_groups[group_id])
        for group_id, observed in zip(
            forward_long["group_id"].astype(str), forward_counts
        )
    ):
        raise RuntimeError("Recovered forward raw-feature counts disagree.")
    for metric in (
        "churn_ap_skill",
        "top10_lift",
        "retention_roc_auc",
        "best_rounds",
    ):
        values = pd.to_numeric(forward_long[metric], errors="raise")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise RuntimeError(f"Recovered forward metric is non-finite: {metric}")


    refinement_mask = _truth_values(path_frame["refinement_group"])
    coarse_groups = build_cumulative_forward_groups(
        gain,
        min_features=int(args.min_features),
        step_size=int(args.forward_step_size),
        required_features=mandatory_control_features,
    )
    observed_coarse_ids = path_frame.loc[
        ~refinement_mask, "group_id"
    ].astype(str).tolist()
    if observed_coarse_ids != list(coarse_groups):
        raise RuntimeError("Recovered coarse forward schedule changed.")
    if any(path_groups[group_id] != features for group_id, features in coarse_groups.items()):
        raise RuntimeError("Recovered coarse forward feature prefixes changed.")


    evaluated_groups: dict[str, list[str]] = {
        group_id: list(features) for group_id, features in coarse_groups.items()
    }
    evaluated_ids = set(evaluated_groups)


    def replay_forward_choice() -> tuple[str, pd.DataFrame, str]:
        subset = forward_long.loc[
            forward_long["group_id"].astype(str).isin(evaluated_ids)
        ].copy()
        summarized = summarize_candidates(subset, "group_id")
        return choose_performance_group(
            summarized,
            id_col="group_id",
            ap_skill_tolerance=args.ap_skill_tolerance,
            lift_tolerance=args.lift_tolerance,
            auc_tolerance=args.auc_tolerance,
            recent_ap_skill_tolerance=args.recent_ap_skill_tolerance,
        )


    initial_group_id, recomputed_forward_annotated, initial_reason = (
        replay_forward_choice()
    )
    refinement_pass_count = 0
    while True:
        next_refinement = build_local_forward_refinement_groups(
            gain,
            evaluated_groups,
            selected_group_id=initial_group_id,
            min_features=int(args.min_features),
            required_features=mandatory_control_features,
        )
        if not next_refinement:
            break
        refinement_pass_count += 1
        unexpected = [
            group_id
            for group_id, features in next_refinement.items()
            if group_id not in path_groups
            or path_groups[group_id] != features
            or not bool(
                refinement_mask.loc[
                    path_frame["group_id"].astype(str).eq(group_id)
                ].all()
            )
        ]
        if unexpected:
            raise RuntimeError(
                "Recovered refinement path cannot be replayed: "
                f"{sorted(unexpected)}"
            )
        evaluated_groups.update(
            {group_id: list(features) for group_id, features in next_refinement.items()}
        )
        evaluated_ids.update(next_refinement)
        initial_group_id, recomputed_forward_annotated, initial_reason = (
            replay_forward_choice()
        )
    if evaluated_ids != set(path_groups):
        raise RuntimeError("Recovered forward path has unexplained refinement groups.")
    _assert_recomputed_summary(
        forward_summary,
        recomputed_forward_annotated,
        id_col="group_id",
        label="Cumulative forward",
    )
    saved_initial = forward_summary.loc[
        _truth_values(forward_summary["selected_by_rule"]), "group_id"
    ].astype(str).tolist()
    if saved_initial != [initial_group_id]:
        raise RuntimeError("Saved and recomputed forward winners disagree.")
    full_group_id = max(
        coarse_groups, key=lambda group_id: len(coarse_groups[group_id])
    )
    if str(stage4_checkpoint.get("full_group_id")) != full_group_id:
        raise RuntimeError("Stage-4 checkpoint full-group anchor changed.")
    expected_required_ids = choose_confirmation_group_ids(
        forward_summary=summarize_candidates(forward_long, "group_id"),
        forward_annotated=recomputed_forward_annotated,
        groups_by_id=evaluated_groups,
        initial_selected_id=initial_group_id,
        confirm_candidate_count=int(args.confirm_candidate_count),
        full_group_id=full_group_id,
        coarse_group_ids=list(coarse_groups),
    )
    if [str(value) for value in stage4_checkpoint["required_group_ids"]] != (
        expected_required_ids
    ):
        raise RuntimeError("Stage-4 checkpoint required-group policy changed.")
    refinement_counts = sorted(
        pd.to_numeric(
            path_frame.loc[refinement_mask, "raw_feature_count"], errors="raise"
        ).astype(int).tolist()
    )
    recovery_artifacts = {
        name: file_info(output_dir / name) for name in POST_SELECTION_ARTIFACTS
    }
    return {
        "history_candidates": histories,
        "selected_history": selected_history,
        "selected_model_spec": selected_model_spec,
        "selected_optuna_trial_number": trial_number,
        "optuna_metadata": optuna_metadata,
        "selected_group_id": selected_group_id,
        "selected_features": selected_features,
        "selected_rounds": selected_rounds,
        "selected_summary": selected_summary,
        "selection_reason": selection_reason,
        "initial_reason": initial_reason,
        "refinement_pass_count": refinement_pass_count,
        "refinement_counts": refinement_counts,
        "artifact_manifest": recovery_artifacts,
        "stage4_selection_checkpoint": stage4_checkpoint,
    }




def finalize_recovered_selection(
    *,
    modeling_utils: Any,
    selection_frame: pd.DataFrame,
    inner_folds: Sequence[Mapping[str, Any]],
    feature_types: Mapping[str, str],
    config: Any,
    args: argparse.Namespace,
    seeds: Sequence[int],
    output_dir: Path,
    recovered: Mapping[str, Any],
    candidate_audit: pd.DataFrame,
    mandatory_control_features: Sequence[str],
    cohort_control_contract: Mapping[str, Any],
    starting_pool_path: Path | None,
    audit_candidates: Sequence[str],
    screened_features: Sequence[str],
    model_features: Sequence[str],
    prepared: pd.DataFrame,
    test_idx: np.ndarray,
    outer_folds: Sequence[Mapping[str, Any]],
    root: Path,
    input_path: Path,
    manifest_path: Path,
    modeling_utils_path: Path,
    feature_routing_path: Path,
    feature_types_contract_path: Path,
    routing_audit_path: Path,
    preparation_info_path: Path,
    outer_fold_definitions_path: Path,
    inner_fold_definitions_path: Path,
) -> None:
    """Run diagnostics and Stage 5 from a semantically proven Stage-4 boundary."""


    selected_features = list(recovered["selected_features"])
    selected_history = dict(recovered["selected_history"])
    selected_model_spec = dict(recovered["selected_model_spec"])
    selected_group_id = str(recovered["selected_group_id"])
    selected_rounds = int(recovered["selected_rounds"])
    selection_reason = str(recovered["selection_reason"])
    warnings = governance_warnings(candidate_audit, selected_features)
    print(
        "POST-SELECTION RECOVERY: Stage-4 artifacts validated; skipping Optuna, "
        "gain ranking, forward selection, and confirmation.",
        flush=True,
    )
    print(
        "POST-SELECTION DIAGNOSTICS: calendar control and shap.TreeExplainer",
        flush=True,
    )
    diagnostics_manifest = write_selected_model_diagnostics(
        modeling_utils=modeling_utils,
        selection_frame=selection_frame,
        inner_folds=inner_folds,
        feature_types=feature_types,
        selected_features=selected_features,
        config=config,
        history=selected_history,
        model_spec=selected_model_spec,
        seeds=seeds,
        train_max_rows=args.confirm_train_max_rows,
        validation_max_rows=args.validation_max_rows,
        max_rounds=args.max_rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        device=args.device,
        n_jobs=args.n_jobs,
        rare_min_frequency=args.rare_min_frequency,
        output_dir=output_dir,
    )
    diagnostics_manifest = add_business_retention_trend_diagnostics(
        diagnostics_manifest=diagnostics_manifest,
        population=prepared,
        test_idx=test_idx,
        config=config,
        selected_history=selected_history,
        output_dir=output_dir,
    )
    recovery_receipt = {
        "recovery_kind": "VERSIONED_STAGE4_SELECTION_CHECKPOINT_FINALIZATION",
        "stage4_selection_contract_version": STAGE4_SELECTION_CONTRACT_VERSION,
        "legacy_stage4_boundary_without_preexisting_checkpoint": False,
        "original_selection_provenance_status": (
            "Stage-4 fold results and the paired AP-skill/recall one-SE choice "
            "were structurally replayed and verified against the versioned "
            "checkpoint and its bound artifact hashes."
        ),
        "profile_hash": PROFILE_HASH,
        "profile": PROFILE_PAYLOAD,
        "selection_recomputed_from_long_artifacts": True,
        "selection_stages_refit": False,
        "current_finalizer": {
            "script_role": "post_selection_recovery_finalizer",
            "script": file_info(Path(__file__).resolve()),
            "xgboost_version": xgb.__version__,
            "optuna_version": optuna.__version__,
            "shap_version": shap.__version__,
            "arguments_scope": (
                "Current finalization arguments; selection-critical history, "
                "Optuna, fold, seed, feature-path, and tolerance fields were "
                "reconstructed or cross-validated against saved artifacts."
            ),
        },
        "outer_validation_rows_scored": 0,
        "final_holdout_rows_scored": 0,
        "validated_selection_artifacts": recovered["artifact_manifest"],
        "stage4_selection_checkpoint": recovered[
            "stage4_selection_checkpoint"
        ],
    }
    selected_payload = {
        "xgboost_version": xgb.__version__,
        "optuna_version": optuna.__version__,
        "shap_version": shap.__version__,
        "selection_method": (
            "manifest-driven full candidate universe -> training-only quality "
            "screen -> anchored deterministic Optuna TPE configuration tuning -> "
            "chronological inner-fold confirmation -> grouped total-gain ranking "
            "across chronological folds -> coarse cumulative forward-prefix "
            "validation -> iterative local one-feature count refinement -> "
            "adaptive multi-pass multi-seed confirmation -> paired-fold "
            "AP-skill/recall@10% one-standard-error parsimony selection"
        ),
        "stage4_selection_rule": STAGE4_SELECTION_RULE,
        "final_feature_selection_rule": FINAL_FEATURE_SELECTION_RULE,
        "final_feature_selection_policy": final_selection_policy(args),
        "final_feature_selection_policy_sha256": policy_sha256(final_selection_policy(args)),
        "selection_policy_module": file_info(Path(__file__).with_name("xgboost_selection_policy.py")),
        "stage4_selection_contract_version": STAGE4_SELECTION_CONTRACT_VERSION,
        "stage4_selection_checkpoint": dict(
            recovered["stage4_selection_checkpoint"]
        ),
        "gain_ranking_method": "mean grouped normalized total_gain across inner folds",
        "forward_step_size": int(args.forward_step_size),
        "forward_step_size_role": "coarse scan only",
        "local_refinement_pass_count": int(recovered["refinement_pass_count"]),
        "locally_refined_feature_counts": list(recovered["refinement_counts"]),
        "previous_feature_group_used": False,
        "previous_feature_plan_inventory_used": False,
        "previous_selected_parameter_file_used": False,
        "prepared_starting_pool_used": starting_pool_path is not None,
        "prepared_starting_pool": (
            None if starting_pool_path is None else file_info(starting_pool_path)
        ),
        "manifest_audit_candidate_count": int(len(audit_candidates)),
        "quality_screened_starting_feature_count": int(len(screened_features)),
        "model_feature_universe_count": int(len(model_features)),
        "mandatory_control_features": list(mandatory_control_features),
        "cohort_calendar_year_contract": dict(cohort_control_contract),
        "selected_group_id": selected_group_id,
        "selected_raw_feature_count": int(len(selected_features)),
        "selected_candidate_feature_count": int(
            len(selected_features) - len(mandatory_control_features)
        ),
        "selected_features": selected_features,
        "selection_reason": selection_reason,
        "initial_forward_selection_reason": str(recovered["initial_reason"]),
        "history_candidate_grid": list(recovered["history_candidates"]),
        "selected_history": selected_history,
        "selected_model_spec": selected_model_spec,
        "selected_optuna_trial_number": int(
            recovered["selected_optuna_trial_number"]
        ),
        "optuna_study": dict(recovered["optuna_metadata"]),
        "recommended_boosting_rounds_from_inner_confirmation": selected_rounds,
        "seeds": list(seeds),
        "primary_statistical_metric": "CHURN_AVERAGE_PRECISION_SKILL",
        "co_primary_business_metric": "CHURN_RECALL_10PCT",
        "reported_business_translation": [
            "TOP10_CHURN_LIFT",
            "CHURN_PRECISION_10PCT",
        ],
        "ranking_guardrail": "RETENTION_ROC_AUC",
        "calibration_guardrails": [
            "LOG_LOSS",
            "BRIER_SCORE",
            "CALIBRATION_GAP",
        ],
        "diagnostics": diagnostics_manifest,
        "stage3_discovery_tolerances": {
            "ap_skill": args.ap_skill_tolerance,
            "lift10": args.lift_tolerance,
            "roc_auc": args.auc_tolerance,
            "recent_ap_skill": args.recent_ap_skill_tolerance,
        },
        "selection_tolerances": {
            "ap_skill": args.ap_skill_tolerance,
            "lift10": args.lift_tolerance,
            "roc_auc": args.auc_tolerance,
            "recent_ap_skill": args.recent_ap_skill_tolerance,
            "role": "STAGE3_DISCOVERY_ONLY",
        },
        "stage3_discovery_tolerances_affect_final_choice_directly": False,
        "selected_inner_summary": dict(recovered["selected_summary"]),
        "governance_review_features_in_selected_group": warnings,
        "outer_validation_rows_scored": 0,
        "final_holdout_rows_identified": int(len(test_idx)),
        "final_holdout_rows_scored": 0,
        "post_selection_recovery": recovery_receipt,
    }
    (output_dir / "selected_features.json").write_text(
        json.dumps(
            strict_json_value(selected_payload),
            indent=2,
            default=str,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "selected_features.txt").write_text(
        "\n".join(selected_features) + "\n", encoding="utf-8"
    )


    print("STAGE 5/5: WRITE AUDIT AND SUMMARY", flush=True)
    test_dates = pd.to_datetime(prepared.loc[test_idx, config.cohort_date_col])
    run_info = {
        "profile": PROFILE_PAYLOAD,
        "profile_hash": PROFILE_HASH,
        "xgboost_version": xgb.__version__,
        "optuna_version": optuna.__version__,
        "shap_version": shap.__version__,
        "device": args.device,
        "n_jobs": args.n_jobs,
        "project_root": str(root),
        "input": file_info(input_path),
        "manifest": file_info(manifest_path),
        "script": file_info(Path(__file__).resolve()),
        "modeling_utils": file_info(modeling_utils_path),
        "model_diagnostics": file_info(
            PROJECT_ROOT / "c_modeling" / "model_diagnostics.py"
        ),
        "feature_routing": file_info(feature_routing_path),
        "feature_types_file": file_info(feature_types_contract_path),
        "routing_audit": file_info(routing_audit_path),
        "starting_feature_pool": (
            None if starting_pool_path is None else file_info(starting_pool_path)
        ),
        "preparation_info": file_info(preparation_info_path),
        "outer_fold_definitions": file_info(outer_fold_definitions_path),
        "inner_fold_definitions": file_info(inner_fold_definitions_path),
        "random_seed": BASE_SEED,
        "manifest_audit_candidate_count": int(len(audit_candidates)),
        "quality_screened_starting_feature_count": int(len(screened_features)),
        "model_feature_universe_count": int(len(model_features)),
        "mandatory_control_features": list(mandatory_control_features),
        "cohort_calendar_year_contract": dict(cohort_control_contract),
        "selected_raw_feature_count": int(len(selected_features)),
        "selected_candidate_feature_count": int(
            len(selected_features) - len(mandatory_control_features)
        ),
        "selected_group_id": selected_group_id,
        "stage4_selection_rule": STAGE4_SELECTION_RULE,
        "final_feature_selection_rule": FINAL_FEATURE_SELECTION_RULE,
        "final_feature_selection_policy": final_selection_policy(args),
        "final_feature_selection_policy_sha256": policy_sha256(final_selection_policy(args)),
        "selection_policy_module": file_info(Path(__file__).with_name("xgboost_selection_policy.py")),
        "stage4_selection_contract_version": STAGE4_SELECTION_CONTRACT_VERSION,
        "stage4_selection_checkpoint": file_info(
            output_dir / STAGE4_SELECTION_CHECKPOINT
        ),
        "selection_population_rows": int(len(selection_frame)),
        "inner_fold_count": int(len(inner_folds)),
        "history_candidate_grid": list(recovered["history_candidates"]),
        "selected_history": selected_history,
        "selected_model_spec": selected_model_spec,
        "selected_optuna_trial_number": int(
            recovered["selected_optuna_trial_number"]
        ),
        "optuna_study": dict(recovered["optuna_metadata"]),
        "outer_fold_count_defined_but_not_scored": int(len(outer_folds)),
        "final_holdout_rows_identified": int(len(test_idx)),
        "final_holdout_start": str(test_dates.min().date()),
        "final_holdout_end": str(test_dates.max().date()),
        "outer_validation_rows_scored": 0,
        "final_holdout_rows_scored": 0,
        "arguments": vars(args),
        "diagnostics": diagnostics_manifest,
        "post_selection_recovery": recovery_receipt,
    }
    summary_lines = [
        "# XGBoost feature selection",
        "",
        "## Recovery",
        "",
        "- Validated Stage-4 selection artifacts were reused.",
        "- Optuna, gain ranking, forward selection, and confirmation were not refit.",
        "- Post-selection diagnostics and final receipts were regenerated.",
        "",
        "## Selected group",
        "",
        f"- Group ID: `{selected_group_id}`",
        f"- Raw feature count: **{len(selected_features)}**",
        f"- Selection reason: {selection_reason}",
        f"- Selected history: `{selected_history['history_id']}`",
        f"- Selected model specification: `{selected_model_spec['model_id']}`",
        f"- Selected Optuna trial: **{recovered['selected_optuna_trial_number']}**",
        f"- Median recommended boosting rounds: **{selected_rounds}**",
        "",
        "```text",
        *selected_features,
        "```",
        "",
        "## Validation boundary",
        "",
        "- Outer validation folds scored: **0**",
        f"- Final holdout rows identified: **{len(test_idx):,}**",
        "- Final holdout rows scored: **0**",
    ]
    (output_dir / "feature_selection_summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    (output_dir / "post_selection_recovery.json").write_text(
        json.dumps(
            strict_json_value(recovery_receipt), indent=2, allow_nan=False
        ),
        encoding="utf-8",
    )
    # ``run_info.json`` is the runner's completion marker.  Write it last so a
    # crash cannot make a partially finalized recovery appear reusable.
    run_info_path = output_dir / "run_info.json"
    run_info_temporary = run_info_path.with_name(run_info_path.name + ".tmp")
    run_info_temporary.write_text(
        json.dumps(
            strict_json_value(run_info),
            indent=2,
            default=str,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    os.replace(run_info_temporary, run_info_path)
    print("", flush=True)
    print("XGBOOST FEATURE SELECTION COMPLETE", flush=True)
    print(f"Selected group: {selected_group_id}", flush=True)
    print(f"Selected raw features: {len(selected_features)}", flush=True)
    print(f"Feature list: {output_dir / 'selected_features.txt'}", flush=True)
    print("Outer validation rows scored: 0", flush=True)
    print(f"Final holdout rows scored: 0 of {len(test_idx)}", flush=True)








def main() -> None:
    args = parse_args()
    final_selection_policy(args)  # Validate before reading data or fitting models.
    validate_xgboost_version()
    validate_optuna_version()
    validate_shap_version()
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be at least 1.")
    if args.min_features < 2:
        raise ValueError("--min-features must be at least 2.")
    if args.forward_step_size < 1:
        raise ValueError("--forward-step-size must be at least 1.")
    if args.optuna_trials < 0:
        raise ValueError("--optuna-trials cannot be negative.")
    if args.optuna_batch_size < 1:
        raise ValueError("--optuna-batch-size must be at least 1.")
    if args.optuna_timeout_seconds < 0:
        raise ValueError("--optuna-timeout-seconds cannot be negative.")
    if args.confirm_candidate_count < 1:
        raise ValueError("--confirm-candidate-count must be at least 1.")
    if args.confirm_max_passes < 1:
        raise ValueError("--confirm-max-passes must be at least 1.")
    seeds = parse_integer_list(args.seeds, "--seeds")




    root = args.project_root.expanduser().resolve()
    output_dir = resolve(root, args.output_dir)
    modeling_utils_path = resolve(root, args.modeling_utils)
    input_path = resolve(root, args.input)
    manifest_path = resolve(root, args.manifest)
    starting_pool_path = (
        None
        if args.starting_feature_pool is None
        else resolve(root, args.starting_feature_pool)
    )
    preparation_info_path = resolve(root, args.preparation_info)
    outer_fold_definitions_path = resolve(root, args.outer_fold_definitions)
    inner_fold_definitions_path = resolve(root, args.inner_fold_definitions)
    feature_routing_path = modeling_utils_path.parent / "feature_routing.py"
    feature_types_value = os.environ.get("RETENTION_XGB_FEATURE_TYPES", "").strip()
    routing_audit_value = os.environ.get(
        "RETENTION_FEATURE_ROUTING_AUDIT", ""
    ).strip()
    if not feature_types_value or not routing_audit_value:
        raise EnvironmentError(
            "RETENTION_XGB_FEATURE_TYPES and RETENTION_FEATURE_ROUTING_AUDIT "
            "are required."
        )
    feature_types_contract_path = Path(feature_types_value).expanduser().resolve()
    routing_audit_path = Path(routing_audit_value).expanduser().resolve()
    required = [
        modeling_utils_path,
        feature_routing_path,
        feature_types_contract_path,
        routing_audit_path,
        input_path,
        manifest_path,
    ]
    if starting_pool_path is not None:
        required.append(starting_pool_path)
    required.extend(
        [
            preparation_info_path,
            outer_fold_definitions_path,
            inner_fold_definitions_path,
        ]
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required file(s) missing:\n" + "\n".join(f"- {path}" for path in missing)
        )




    preparation_info = verify_preparation_info(
        preparation_info_path,
        input_path=input_path,
        manifest_path=manifest_path,
        modeling_utils_path=modeling_utils_path,
        args=args,
    )
    if args.resume_post_selection:
        if not output_dir.is_dir() or output_dir.is_symlink():
            raise RuntimeError(
                "Post-selection recovery requires the existing safe XGBoost "
                "output directory."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)




    sys.path[:0] = [str(modeling_utils_path.parent), str(root)]
    modeling_utils = importlib.import_module("modeling_utils")
    feature_routing = importlib.import_module("feature_routing")
    cohort_control_contract = modeling_utils.cohort_calendar_year_contract()
    cohort_control_feature = str(
        cohort_control_contract["derived_feature"]
    ).strip().upper()
    mandatory_control_features = [cohort_control_feature]




    print("STAGE 1/5: BUILD FULL MANIFEST-DRIVEN CANDIDATE UNIVERSE", flush=True)
    raw = modeling_utils.read_table(input_path)
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    prepared, outer_folds, test_idx, split_summary, config = (
        modeling_utils.load_project_population_and_folds(
            raw,
            rolling_validation_folds=args.outer_folds,
            validation_months=args.outer_validation_months,
            test_months=args.test_months,
            min_train_months=args.min_outer_train_months,
            strict_dataset_contract=not args.allow_contract_violations,
        )
    )
    preparation_label_buffer = preparation_info.get("split_settings", {}).get(
        "label_availability_buffer_days"
    )
    if preparation_label_buffer != config.label_availability_buffer_days:
        raise ValueError(
            "XGBoost label-availability buffer differs from modeling preparation: "
            f"expected={preparation_label_buffer!r}; "
            f"observed={config.label_availability_buffer_days!r}."
        )
    candidate_audit, union_candidates, union_feature_types = modeling_utils.build_candidate_audit(
        manifest, prepared.columns
    )
    if not args.resume_post_selection:
        candidate_audit.to_csv(
            output_dir / "xgb_candidate_universe_audit.csv", index=False
        )
    audit_candidates, feature_types = feature_routing.route_candidate_list_and_types(
        union_candidates,
        union_feature_types,
        "xgboost",
    )
    feature_routing.validate_one_variant_per_concept(
        audit_candidates, "xgboost"
    )
    audit_candidates = [str(value).upper() for value in audit_candidates]
    if not audit_candidates:
        raise ValueError("The current manifest/data audit produced no eligible candidates.")
    if not args.resume_post_selection:
        (output_dir / "xgb_manifest_driven_candidate_universe.txt").write_text(
            "\n".join(audit_candidates) + "\n", encoding="utf-8"
        )




    for fold_def in outer_folds:
        if np.intersect1d(test_idx, fold_def["train_idx"]).size:
            raise AssertionError("Final holdout entered an outer training fold.")
        if np.intersect1d(test_idx, fold_def["validation_idx"]).size:
            raise AssertionError("Final holdout entered an outer validation fold.")




    outer_fold_frame = modeling_utils.fold_definitions_frame(
        prepared,
        outer_folds,
        cohort_col=config.cohort_date_col,
        target_check_col=config.target_check_col,
        target_col=config.target_col,
    )
    modeling_utils.verify_fold_definition_contract(
        outer_fold_frame,
        outer_fold_definitions_path,
        label="XGBoost outer discovery boundary",
    )




    selection_frame = oldest_complete_train(
        prepared, outer_folds[0], config.cohort_date_col
    )
    if starting_pool_path is not None:
        screened_features = [
            line.strip().upper()
            for line in starting_pool_path.read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if line.strip()
        ]
        if not screened_features or len(screened_features) != len(
            set(screened_features)
        ):
            raise ValueError(
                "The prepared starting feature pool is empty or contains duplicates."
            )
        invalid_starting = sorted(set(screened_features).difference(audit_candidates))
        if invalid_starting:
            raise ValueError(
                "The prepared starting pool contains non-eligible features: "
                + ", ".join(invalid_starting)
            )
        family_feature_types = read_xgboost_feature_contract(
            screened_features,
            feature_types_contract_path,
            routing_audit_path,
        )
        type_mismatches = {
            feature: (feature_types.get(feature), family_feature_types[feature])
            for feature in screened_features
            if feature_types.get(feature) != family_feature_types[feature]
        }
        if type_mismatches:
            raise ValueError(
                "XGBoost routed types differ from the preparation contract: "
                f"{type_mismatches}"
            )
        feature_types.update(family_feature_types)
        starting_set = set(screened_features)
        quality_screen = pd.DataFrame(
            [
                {
                    "feature": feature,
                    "decision": (
                        "KEEP_PREPARATION_SCREEN"
                        if feature in starting_set
                        else "DROP_PREPARATION_SCREEN"
                    ),
                    "reason": "training-only quality screen from modeling preparation",
                }
                for feature in audit_candidates
            ]
        )
    else:
        quality_frame = modeling_utils.proportional_month_target_sample(
            selection_frame,
            max_rows=(
                None
                if args.quality_screen_max_rows <= 0
                else args.quality_screen_max_rows
            ),
            cohort_col=config.cohort_date_col,
            target_col=config.target_col,
            random_seed=BASE_SEED,
        )
        screened_features, quality_screen = modeling_utils.quality_screen_candidates(
            quality_frame,
            audit_candidates,
            feature_types,
            missing_drop_threshold=args.missing_drop_threshold,
            dominant_level_drop_threshold=args.dominant_level_drop_threshold,
            high_cardinality_ratio=args.high_cardinality_ratio,
            max_category_levels=args.max_category_levels,
        )
        screened_features = [str(value).upper() for value in screened_features]
    if not args.resume_post_selection:
        quality_screen.to_csv(
            output_dir / "xgb_training_only_quality_screen.csv", index=False
        )
        (output_dir / "xgb_quality_screened_feature_universe.txt").write_text(
            "\n".join(screened_features) + "\n", encoding="utf-8"
        )
    if len(screened_features) < args.min_features:
        raise ValueError(
            f"Quality screening retained only {len(screened_features)} features, "
            f"below --min-features={args.min_features}."
        )
    for feature in screened_features:
        feature_types.setdefault(feature, "discrete")
    collision = set(mandatory_control_features).intersection(screened_features)
    if collision:
        raise ValueError(
            "The engine-owned mandatory control collided with the preparation "
            f"candidate pool: {sorted(collision)}."
        )
    if cohort_control_feature not in prepared:
        raise KeyError(
            f"Prepared population is missing mandatory {cohort_control_feature}."
        )
    if str(cohort_control_contract.get("configured_type", "")).lower() != "continuous":
        raise ValueError(
            "COHORT_CALENDAR_YEAR must have continuous configured type."
        )
    feature_types[cohort_control_feature] = "continuous"
    model_features = [*mandatory_control_features, *screened_features]
    if not args.resume_post_selection:
        (output_dir / "xgb_resolved_model_feature_types.json").write_text(
            json.dumps(
                {feature: feature_types[feature] for feature in model_features},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )




    inner_folds, inner_holdout = modeling_utils.build_expanding_folds(
        selection_frame,
        cohort_col=config.cohort_date_col,
        target_check_col=config.target_check_col,
        target_col=config.target_col,
        validation_folds=args.inner_folds,
        validation_months=args.inner_validation_months,
        min_train_months=args.min_inner_train_months,
        holdout_months=0,
        label_buffer_days=config.label_availability_buffer_days,
    )
    if len(inner_holdout):
        raise AssertionError("Inner feature discovery unexpectedly created a holdout.")
    inner_fold_frame = modeling_utils.fold_definitions_frame(
        selection_frame,
        inner_folds,
        cohort_col=config.cohort_date_col,
        target_check_col=config.target_check_col,
        target_col=config.target_col,
    )
    modeling_utils.verify_fold_definition_contract(
        inner_fold_frame,
        inner_fold_definitions_path,
        label="XGBoost inner feature-discovery folds",
    )
    if not args.resume_post_selection:
        inner_fold_frame.to_csv(
            output_dir / "xgb_feature_discovery_inner_folds.csv", index=False
        )




    print(
        f"Manifest-driven audit candidates: {len(audit_candidates)}; "
        f"quality-screened candidates: {len(screened_features)}; "
        f"model universe with mandatory control: {len(model_features)}",
        flush=True,
    )
    print("Exact starting universe:", flush=True)
    for feature in model_features:
        print(f"  {feature}", flush=True)




    if args.resume_post_selection:
        recovered = load_post_selection_recovery(
            output_dir=output_dir,
            args=args,
            seeds=seeds,
            model_features=model_features,
            mandatory_control_features=mandatory_control_features,
            inner_fold_frame=inner_fold_frame,
        )
        finalize_recovered_selection(
            modeling_utils=modeling_utils,
            selection_frame=selection_frame,
            inner_folds=inner_folds,
            feature_types=feature_types,
            config=config,
            args=args,
            seeds=seeds,
            output_dir=output_dir,
            recovered=recovered,
            candidate_audit=candidate_audit,
            mandatory_control_features=mandatory_control_features,
            cohort_control_contract=cohort_control_contract,
            starting_pool_path=starting_pool_path,
            audit_candidates=audit_candidates,
            screened_features=screened_features,
            model_features=model_features,
            prepared=prepared,
            test_idx=test_idx,
            outer_folds=outer_folds,
            root=root,
            input_path=input_path,
            manifest_path=manifest_path,
            modeling_utils_path=modeling_utils_path,
            feature_routing_path=feature_routing_path,
            feature_types_contract_path=feature_types_contract_path,
            routing_audit_path=routing_audit_path,
            preparation_info_path=preparation_info_path,
            outer_fold_definitions_path=outer_fold_definitions_path,
            inner_fold_definitions_path=inner_fold_definitions_path,
        )
        return




    print("STAGE 2/5: OPTUNA XGBOOST CONFIGURATION TUNING", flush=True)
    history_candidates = history_grid()
    config_screen, lookup, optuna_metadata = run_optuna_configuration_screen(
        modeling_utils=modeling_utils,
        selection_frame=selection_frame,
        fold_def=inner_folds[-1],
        feature_types=feature_types,
        features=model_features,
        config=config,
        histories=history_candidates,
        adaptive_trials=args.optuna_trials,
        batch_size=args.optuna_batch_size,
        timeout_seconds=args.optuna_timeout_seconds,
        train_max_rows=args.config_screen_train_max_rows,
        validation_max_rows=args.validation_max_rows,
        max_rounds=args.max_rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        device=args.device,
        n_jobs=args.n_jobs,
        rare_min_frequency=args.rare_min_frequency,
    )
    config_screen.to_csv(output_dir / "xgb_configuration_screen_long.csv", index=False)
    config_screen.to_csv(output_dir / "xgb_optuna_trials.csv", index=False)
    (output_dir / "xgb_optuna_study.json").write_text(
        json.dumps(
            strict_json_value(optuna_metadata),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    screen_summary = summarize_candidates(config_screen, "candidate_id")
    screen_summary.to_csv(output_dir / "xgb_configuration_screen_summary.csv", index=False)
    top_ids = (
        screen_summary.sort_values(
            ["mean_churn_ap_skill", "mean_top10_lift", "mean_retention_roc_auc"],
            ascending=[False, False, False],
            kind="mergesort",
        )["candidate_id"]
        .head(max(1, args.confirm_top_configs))
        .astype(str)
        .tolist()
    )
    confirm_configurations = [lookup[candidate_id] for candidate_id in top_ids]
    config_confirm = evaluate_configuration_rows(
        modeling_utils=modeling_utils,
        selection_frame=selection_frame,
        fold_defs=inner_folds,
        feature_types=feature_types,
        features=model_features,
        config=config,
        configurations=confirm_configurations,
        train_max_rows=args.config_confirm_train_max_rows,
        validation_max_rows=args.validation_max_rows,
        max_rounds=args.max_rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        device=args.device,
        n_jobs=args.n_jobs,
        rare_min_frequency=args.rare_min_frequency,
        stage="CONFIG_CONFIRM",
    )
    config_confirm.to_csv(output_dir / "xgb_configuration_confirmation_long.csv", index=False)
    config_confirm_summary = summarize_candidates(config_confirm, "candidate_id")
    config_confirm_summary.to_csv(
        output_dir / "xgb_configuration_confirmation_summary.csv", index=False
    )
    selected_config_id = choose_best_configuration(config_confirm_summary)
    selected_configuration = lookup[selected_config_id]
    selected_history = dict(selected_configuration["history"])
    selected_model_spec = dict(selected_configuration["model"])
    selected_optuna_trial_number = int(
        selected_configuration["optuna_trial_number"]
    )
    discovery_selected_history = dict(selected_history)
    discovery_selected_model_spec = dict(selected_model_spec)
    discovery_selected_optuna_trial_number = int(selected_optuna_trial_number)
    discovery_selected_config_id = str(selected_config_id)
    print(
        "Selected XGBoost configuration: "
        f"{selected_config_id} (Optuna trial {selected_optuna_trial_number})",
        flush=True,
    )




    print("STAGE 3/5: GAIN-RANKED CUMULATIVE FORWARD SELECTION", flush=True)
    full_pool_metrics, gain_long, gain_ranking = evaluate_full_pool_gain_ranking(
        modeling_utils=modeling_utils,
        selection_frame=selection_frame,
        inner_folds=inner_folds,
        feature_types=feature_types,
        features=model_features,
        config=config,
        history=selected_history,
        model_spec=selected_model_spec,
        train_max_rows=args.forward_train_max_rows,
        validation_max_rows=args.forward_validation_max_rows,
        max_rounds=args.max_rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        device=args.device,
        n_jobs=args.n_jobs,
        rare_min_frequency=args.rare_min_frequency,
    )
    full_pool_metrics.to_csv(
        output_dir / "xgb_gain_ranking_model_metrics.csv", index=False
    )
    gain_long.to_csv(output_dir / "xgb_gain_ranking_long.csv", index=False)
    gain_ranking = annotate_forward_ranking(
        gain_ranking,
        mandatory_control_features,
    )
    gain_ranking.to_csv(output_dir / "xgb_gain_ranking_summary.csv", index=False)




    groups_by_id = build_cumulative_forward_groups(
        gain_ranking,
        min_features=args.min_features,
        step_size=args.forward_step_size,
        required_features=mandatory_control_features,
    )
    coarse_group_ids = list(groups_by_id)
    full_group_id = next(
        group_id
        for group_id, features in groups_by_id.items()
        if len(features) == len(model_features)
    )
    if set(groups_by_id[full_group_id]) != set(model_features):
        raise AssertionError("The full forward prefix lost a screened feature.")
    if any(
        not set(mandatory_control_features).issubset(features)
        for features in groups_by_id.values()
    ):
        raise AssertionError("A forward group lost the mandatory time control.")
    forward_metrics = evaluate_feature_groups_multiseed(
        modeling_utils=modeling_utils,
        selection_frame=selection_frame,
        inner_folds=inner_folds,
        feature_types=feature_types,
        groups=groups_by_id,
        config=config,
        history=selected_history,
        model_spec=selected_model_spec,
        seeds=[BASE_SEED],
        train_max_rows=args.forward_train_max_rows,
        validation_max_rows=args.forward_validation_max_rows,
        max_rounds=args.max_rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        device=args.device,
        n_jobs=args.n_jobs,
        rare_min_frequency=args.rare_min_frequency,
        stage_label="FORWARD",
    )
    coarse_forward_summary = summarize_candidates(forward_metrics, "group_id")
    (
        coarse_selected_id,
        coarse_forward_annotated,
        coarse_reason,
    ) = choose_performance_group(
        coarse_forward_summary,
        id_col="group_id",
        ap_skill_tolerance=args.ap_skill_tolerance,
        lift_tolerance=args.lift_tolerance,
        auc_tolerance=args.auc_tolerance,
        recent_ap_skill_tolerance=args.recent_ap_skill_tolerance,
    )
    coarse_forward_annotated.to_csv(
        output_dir / "xgb_cumulative_forward_coarse_summary.csv", index=False
    )




    forward_summary = coarse_forward_summary
    initial_selected_id = coarse_selected_id
    forward_annotated = coarse_forward_annotated
    initial_reason = coarse_reason
    refinement_groups: dict[str, list[str]] = {}
    refinement_pass_count = 0
    while True:
        next_refinement_groups = build_local_forward_refinement_groups(
            gain_ranking,
            groups_by_id,
            selected_group_id=initial_selected_id,
            min_features=args.min_features,
            required_features=mandatory_control_features,
        )
        if not next_refinement_groups:
            break
        refinement_pass_count += 1
        print(
            "STAGE 3B/5: LOCALLY REFINE THE SMALLEST ELIGIBLE FORWARD PREFIX "
            f"(pass {refinement_pass_count})",
            flush=True,
        )
        refinement_metrics = evaluate_feature_groups_multiseed(
            modeling_utils=modeling_utils,
            selection_frame=selection_frame,
            inner_folds=inner_folds,
            feature_types=feature_types,
            groups=next_refinement_groups,
            config=config,
            history=selected_history,
            model_spec=selected_model_spec,
            seeds=[BASE_SEED],
            train_max_rows=args.forward_train_max_rows,
            validation_max_rows=args.forward_validation_max_rows,
            max_rounds=args.max_rounds,
            early_stopping_rounds=args.early_stopping_rounds,
            device=args.device,
            n_jobs=args.n_jobs,
            rare_min_frequency=args.rare_min_frequency,
            stage_label="FORWARD_REFINE",
        )
        forward_metrics = pd.concat(
            [forward_metrics, refinement_metrics], ignore_index=True
        )
        groups_by_id.update(next_refinement_groups)
        refinement_groups.update(next_refinement_groups)
        forward_summary = summarize_candidates(forward_metrics, "group_id")
        (
            initial_selected_id,
            forward_annotated,
            initial_reason,
        ) = choose_performance_group(
            forward_summary,
            id_col="group_id",
            ap_skill_tolerance=args.ap_skill_tolerance,
            lift_tolerance=args.lift_tolerance,
            auc_tolerance=args.auc_tolerance,
            recent_ap_skill_tolerance=args.recent_ap_skill_tolerance,
        )




    forward_metrics.to_csv(
        output_dir / "xgb_cumulative_forward_metrics_long.csv", index=False
    )
    forward_annotated.to_csv(
        output_dir / "xgb_cumulative_forward_summary.csv", index=False
    )




    path_rows = []
    previous_count = 0
    for path_position, (group_id, features) in enumerate(
        sorted(groups_by_id.items(), key=lambda item: len(item[1])), start=1
    ):
        path_rows.append(
            {
                "forward_step": int(path_position),
                "group_id": group_id,
                "raw_feature_count": int(len(features)),
                "features": ";".join(features),
                "added_at_step": ";".join(features[previous_count:]),
                "refinement_group": bool(group_id in refinement_groups),
            }
        )
        previous_count = len(features)
    pd.DataFrame(path_rows).to_csv(
        output_dir / "xgb_cumulative_forward_path.csv", index=False
    )




    print("STAGE 4/5: MULTI-SEED CONFIRMATION OF FORWARD PREFIXES", flush=True)
    required_confirmation_ids = choose_confirmation_group_ids(
        forward_summary=forward_summary,
        forward_annotated=forward_annotated,
        groups_by_id=groups_by_id,
        initial_selected_id=initial_selected_id,
        confirm_candidate_count=args.confirm_candidate_count,
        full_group_id=full_group_id,
        coarse_group_ids=coarse_group_ids,
    )
    pending_required_ids = list(required_confirmation_ids)
    confirmation_groups: dict[str, list[str]] = {}
    confirmation_parts: list[pd.DataFrame] = []
    pass_records: list[dict[str, Any]] = []
    selected_group_id: str | None = None
    confirmation_annotated = pd.DataFrame()
    selection_reason = ""
    pending_adaptive: dict[str, list[str]] = {}
    resolved = False


    def publish_confirmation_artifacts() -> pd.DataFrame:
        inventory_frame = pd.DataFrame(
            [
                {
                    "group_id": group_id,
                    "raw_feature_count": len(features),
                    "features": ";".join(features),
                }
                for group_id, features in confirmation_groups.items()
            ]
        )
        inventory_frame.to_csv(
            output_dir / "xgb_confirmation_group_inventory.csv", index=False
        )
        combined = pd.concat(confirmation_parts, ignore_index=True)
        combined.to_csv(
            output_dir / "xgb_feature_group_confirmation_long.csv", index=False
        )
        confirmation_annotated.to_csv(
            output_dir / "xgb_feature_group_confirmation_summary.csv", index=False
        )
        return combined


    for confirmation_pass in range(1, args.confirm_max_passes + 1):
        if pending_required_ids:
            batch_ids = pending_required_ids[: args.confirm_candidate_count]
            pending_required_ids = pending_required_ids[args.confirm_candidate_count :]
            batch_groups = {
                group_id: list(groups_by_id[group_id]) for group_id in batch_ids
            }
            phase = "REQUIRED_ELIGIBLE_PARETO"
        elif selected_group_id == full_group_id:
            pending_adaptive = build_adaptive_confirmation_groups(
                gain_ranking,
                groups_by_id,
                list(confirmation_groups),
                initial_selected_id=initial_selected_id,
                full_group_id=full_group_id,
                coarse_group_ids=coarse_group_ids,
                required_features=mandatory_control_features,
                max_candidates=args.confirm_candidate_count,
            )
            if not pending_adaptive:
                resolved = True
                break
            batch_groups = pending_adaptive
            batch_ids = list(batch_groups)
            pending_adaptive = {}
            phase = "FULL_WINNER_INTERMEDIATE_LINEAR_SCAN"
        else:
            resolved = True
            break


        if any(group_id in confirmation_groups for group_id in batch_ids):
            raise RuntimeError("Stage-4 confirmation attempted a duplicate group.")
        if any(
            not set(mandatory_control_features).issubset(features)
            for features in batch_groups.values()
        ):
            raise AssertionError(
                "An adaptive confirmation prefix lost the mandatory time control."
            )
        print(
            f"STAGE 4/5: confirmation pass {confirmation_pass}/"
            f"{args.confirm_max_passes} ({phase}); groups={batch_ids}",
            flush=True,
        )
        batch_long = evaluate_feature_groups_multiseed(
            modeling_utils=modeling_utils,
            selection_frame=selection_frame,
            inner_folds=inner_folds,
            feature_types=feature_types,
            groups=batch_groups,
            config=config,
            history=selected_history,
            model_spec=selected_model_spec,
            seeds=seeds,
            train_max_rows=args.confirm_train_max_rows,
            validation_max_rows=args.validation_max_rows,
            max_rounds=args.max_rounds,
            early_stopping_rounds=args.early_stopping_rounds,
            device=args.device,
            n_jobs=args.n_jobs,
            rare_min_frequency=args.rare_min_frequency,
            stage_label=f"CONFIRM_PASS_{confirmation_pass}",
        )
        confirmation_groups.update(batch_groups)
        groups_by_id.update(batch_groups)
        confirmation_parts.append(batch_long)
        confirmation_long = pd.concat(confirmation_parts, ignore_index=True)
        (
            selected_group_id,
            confirmation_annotated,
            selection_reason,
        ) = choose_paired_one_se_group(confirmation_long, id_col="group_id")
        selected_row = confirmation_annotated.loc[
            confirmation_annotated["group_id"].astype(str).eq(selected_group_id)
        ].iloc[0]
        pass_records.append(
            {
                "pass": int(confirmation_pass),
                "phase": phase,
                "evaluated_group_ids": batch_ids,
                "confirmed_group_count": int(len(confirmation_groups)),
                "selected_group_id_after_pass": selected_group_id,
                "selected_raw_feature_count_after_pass": int(
                    selected_row["raw_feature_count"]
                ),
                "ap_skill_one_se_reference_id_after_pass": str(
                    selected_row["ap_skill_one_se_reference_id"]
                ),
                "recall_10pct_one_se_reference_id_after_pass": str(
                    selected_row["recall_10pct_one_se_reference_id"]
                ),
                "recall_reference_condition_after_pass": str(
                    selected_row["recall_reference_condition"]
                ),
                "full_group_wins_after_pass": bool(
                    selected_group_id == full_group_id
                ),
                "pending_required_count": int(len(pending_required_ids)),
            }
        )
        publish_confirmation_artifacts()


        if pending_required_ids:
            continue
        if selected_group_id != full_group_id:
            resolved = True
            break
        pending_adaptive = build_adaptive_confirmation_groups(
            gain_ranking,
            groups_by_id,
            list(confirmation_groups),
            initial_selected_id=initial_selected_id,
            full_group_id=full_group_id,
            coarse_group_ids=coarse_group_ids,
            required_features=mandatory_control_features,
            max_candidates=args.confirm_candidate_count,
        )
        if not pending_adaptive:
            resolved = True
            break


    if selected_group_id is None or not confirmation_parts:
        raise RuntimeError("Stage-4 confirmation produced no scored feature groups.")
    confirmation_long = publish_confirmation_artifacts()
    unresolved = confirmation_unresolved_message(
        completed_passes=len(pass_records),
        max_passes=args.confirm_max_passes,
        pending_required_ids=pending_required_ids,
        pending_adaptive_ids=list(pending_adaptive),
        selected_group_id=selected_group_id,
        full_group_id=full_group_id,
    )
    if not resolved and unresolved is None:
        unresolved = (
            "Stage-4 adaptive confirmation ended without a resolved selection; "
            "no feature group was accepted."
        )
    if unresolved is not None:
        raise RuntimeError(unresolved)


    discovery_selected_group_id = str(selected_group_id)
    discovery_selection_reason = str(selection_reason)
    discovery_selected_summary = confirmation_annotated.loc[
        confirmation_annotated["group_id"].astype(str).eq(
            discovery_selected_group_id
        )
    ].iloc[0].to_dict()


    print(
        "STAGE 4B/5: EQUAL-BUDGET OPTUNA RETUNING OF FEATURE-COUNT FINALISTS",
        flush=True,
    )
    finalist_group_ids, finalist_inventory = choose_finalist_group_ids(
        confirmation_annotated,
        stage4_selected_group_id=discovery_selected_group_id,
        full_group_id=full_group_id,
        groups_by_id=confirmation_groups,
    )
    finalist_inventory.to_csv(
        output_dir / "xgb_finalist_inventory.csv", index=False
    )
    configuration_confirmation_budget = max(2, int(args.confirm_top_configs))
    finalist_trial_parts: list[pd.DataFrame] = []
    finalist_screen_summary_parts: list[pd.DataFrame] = []
    finalist_config_confirm_parts: list[pd.DataFrame] = []
    finalist_config_summary_parts: list[pd.DataFrame] = []
    finalist_studies: dict[str, dict[str, Any]] = {}
    finalist_selected_configurations: dict[str, dict[str, Any]] = {}
    full_stage2_screen_reusable = stage2_optuna_screen_is_exactly_reusable(
        screen=config_screen,
        metadata=optuna_metadata,
        features=confirmation_groups[full_group_id],
        histories=history_candidates,
        fold_def=inner_folds[-1],
        adaptive_trials=args.optuna_trials,
        batch_size=args.optuna_batch_size,
        train_max_rows=args.config_screen_train_max_rows,
        validation_max_rows=args.validation_max_rows,
        max_rounds=args.max_rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        rare_min_frequency=args.rare_min_frequency,
    )


    for finalist_order, group_id in enumerate(finalist_group_ids, start=1):
        features = list(confirmation_groups[group_id])
        feature_hash = feature_sequence_sha256(features)
        reuse_screen = bool(
            group_id == full_group_id and full_stage2_screen_reusable
        )
        if reuse_screen:
            group_screen = config_screen.copy()
            group_lookup = {
                candidate_id: {
                    "history": dict(value["history"]),
                    "model": dict(value["model"]),
                    "optuna_trial_number": int(value["optuna_trial_number"]),
                    "optuna_trial_origin": str(value["optuna_trial_origin"]),
                }
                for candidate_id, value in lookup.items()
            }
            group_study = dict(optuna_metadata)
            screen_origin = "STAGE2_EXACT_SCREEN_REUSE"
        else:
            group_screen, group_lookup, group_study = (
                run_optuna_configuration_screen(
                    modeling_utils=modeling_utils,
                    selection_frame=selection_frame,
                    fold_def=inner_folds[-1],
                    feature_types=feature_types,
                    features=features,
                    config=config,
                    histories=history_candidates,
                    adaptive_trials=args.optuna_trials,
                    batch_size=args.optuna_batch_size,
                    timeout_seconds=FINALIST_TUNING_TIMEOUT_SECONDS,
                    train_max_rows=args.config_screen_train_max_rows,
                    validation_max_rows=args.validation_max_rows,
                    max_rounds=args.max_rounds,
                    early_stopping_rounds=args.early_stopping_rounds,
                    device=args.device,
                    n_jobs=args.n_jobs,
                    rare_min_frequency=args.rare_min_frequency,
                )
            )
            screen_origin = "FINALIST_EQUAL_BUDGET_REFIT"


        completed_adaptive = int(
            group_study.get("completed_adaptive_trials", -1)
        )
        expected_anchor_trials = len(history_candidates) * len(
            anchor_model_grid()
        )
        expected_total_trials = expected_anchor_trials + int(args.optuna_trials)
        if (
            completed_adaptive != int(args.optuna_trials)
            or bool(group_study.get("adaptive_timeout_reached", True))
            or int(group_study.get("adaptive_timeout_seconds", -1))
            != FINALIST_TUNING_TIMEOUT_SECONDS
            or int(group_study.get("anchor_trial_count", -1))
            != expected_anchor_trials
            or int(group_study.get("completed_total_trials", -1))
            != expected_total_trials
            or len(group_screen) != expected_total_trials
        ):
            raise RuntimeError(
                "Finalist Optuna studies must complete the same anchor and "
                "adaptive-trial budgets without timeout: "
                f"group={group_id}, adaptive_completed={completed_adaptive}, "
                f"adaptive_required={args.optuna_trials}, total_completed="
                f"{group_study.get('completed_total_trials')}, "
                f"total_required={expected_total_trials}."
            )
        if group_study.get("ordered_feature_sha256") != feature_hash:
            raise RuntimeError(
                "Finalist study feature order does not match the fitted feature list; "
                "refusing to relabel cached results."
            )
        group_study = {
            **group_study,
            "finalist_tuning_contract_version": FINALIST_TUNING_CONTRACT_VERSION,
            "group_id": group_id,
            "finalist_order": int(finalist_order),
            "raw_feature_count": int(len(features)),
            "ordered_feature_sha256": feature_hash,
            "screen_origin": screen_origin,
            "exact_stage2_screen_reused": reuse_screen,
            "finalist_timeout_seconds": FINALIST_TUNING_TIMEOUT_SECONDS,
            "configuration_confirmation_budget": int(
                configuration_confirmation_budget
            ),
        }
        group_screen = group_screen.copy()
        group_screen.insert(0, "group_id", group_id)
        group_screen.insert(1, "finalist_order", int(finalist_order))
        group_screen.insert(2, "ordered_feature_sha256", feature_hash)
        group_screen.insert(3, "screen_origin", screen_origin)
        finalist_trial_parts.append(group_screen)


        group_screen_summary = summarize_candidates(
            group_screen, "candidate_id"
        )
        group_screen_summary.insert(1, "finalist_order", int(finalist_order))
        group_screen_summary.insert(2, "ordered_feature_sha256", feature_hash)
        finalist_screen_summary_parts.append(group_screen_summary)
        confirmation_ids = choose_finalist_configuration_ids(
            group_screen_summary,
            confirmation_budget=configuration_confirmation_budget,
        )
        missing_configurations = sorted(
            set(confirmation_ids).difference(group_lookup)
        )
        if missing_configurations:
            raise RuntimeError(
                f"Finalist Optuna lookup is incomplete for {group_id}: "
                f"{missing_configurations}"
            )
        group_configurations = [
            group_lookup[candidate_id] for candidate_id in confirmation_ids
        ]
        stage2_confirm_ids = (
            config_confirm["candidate_id"].astype(str).drop_duplicates().tolist()
        )
        expected_confirmation_pairs = {
            (candidate_id, int(fold_def.get("fold", fold_position)))
            for candidate_id in confirmation_ids
            for fold_position, fold_def in enumerate(inner_folds, start=1)
        }
        observed_confirmation_pairs = list(
            zip(
                config_confirm["candidate_id"].astype(str),
                pd.to_numeric(config_confirm["fold"], errors="coerce")
                .fillna(-1)
                .astype(int),
            )
        )
        reuse_configuration_confirmation = bool(
            reuse_screen
            and set(stage2_confirm_ids) == set(confirmation_ids)
            and len(observed_confirmation_pairs)
            == len(set(observed_confirmation_pairs))
            and set(observed_confirmation_pairs) == expected_confirmation_pairs
            and pd.to_numeric(
                config_confirm["raw_feature_count"], errors="coerce"
            )
            .eq(len(features))
            .all()
        )
        if reuse_configuration_confirmation:
            group_config_confirm = config_confirm.copy()
            confirmation_origin = "STAGE2_EXACT_CONFIRMATION_REUSE"
        else:
            group_config_confirm = evaluate_configuration_rows(
                modeling_utils=modeling_utils,
                selection_frame=selection_frame,
                fold_defs=inner_folds,
                feature_types=feature_types,
                features=features,
                config=config,
                configurations=group_configurations,
                train_max_rows=args.config_confirm_train_max_rows,
                validation_max_rows=args.validation_max_rows,
                max_rounds=args.max_rounds,
                early_stopping_rounds=args.early_stopping_rounds,
                device=args.device,
                n_jobs=args.n_jobs,
                rare_min_frequency=args.rare_min_frequency,
                stage="FINALIST_CONFIG_CONFIRM",
            )
            confirmation_origin = "FINALIST_CONFIG_CONFIRM_REFIT"
        group_config_confirm = group_config_confirm.copy()
        group_config_confirm.insert(0, "group_id", group_id)
        group_config_confirm.insert(1, "finalist_order", int(finalist_order))
        group_config_confirm.insert(2, "ordered_feature_sha256", feature_hash)
        group_config_confirm.insert(
            3, "confirmation_origin", confirmation_origin
        )
        finalist_config_confirm_parts.append(group_config_confirm)
        (
            selected_candidate_id,
            group_config_summary,
            configuration_selection_reason,
        ) = choose_best_finalist_configuration(
            group_config_confirm
        )
        group_config_summary.insert(1, "finalist_order", int(finalist_order))
        group_config_summary.insert(2, "ordered_feature_sha256", feature_hash)
        finalist_config_summary_parts.append(group_config_summary)
        selected_configuration_for_group = group_lookup[selected_candidate_id]
        selected_config_rows = group_config_confirm.loc[
            group_config_confirm["candidate_id"].astype(str).eq(
                selected_candidate_id
            )
        ]
        finalist_selected_configurations[group_id] = {
            "group_id": group_id,
            "raw_feature_count": int(len(features)),
            "ordered_feature_sha256": feature_hash,
            "selected_candidate_id": selected_candidate_id,
            "selected_history": dict(
                selected_configuration_for_group["history"]
            ),
            "selected_model_spec": dict(
                selected_configuration_for_group["model"]
            ),
            "selected_optuna_trial_number": int(
                selected_configuration_for_group["optuna_trial_number"]
            ),
            "selected_optuna_trial_origin": str(
                selected_configuration_for_group["optuna_trial_origin"]
            ),
            "median_best_rounds_from_configuration_confirmation": int(
                round(
                    float(
                        pd.to_numeric(
                            selected_config_rows["best_rounds"], errors="raise"
                        ).median()
                    )
                )
            ),
            "screen_origin": screen_origin,
            "confirmation_origin": confirmation_origin,
            "confirmation_candidate_ids": list(confirmation_ids),
            "configuration_selection_reason": configuration_selection_reason,
        }
        group_study["configuration_confirmation_origin"] = confirmation_origin
        group_study["configuration_confirmation_candidate_ids"] = list(
            confirmation_ids
        )
        group_study["selected_candidate_id"] = selected_candidate_id
        group_study[
            "configuration_selection_reason"
        ] = configuration_selection_reason
        finalist_studies[group_id] = group_study


    completed_trial_counts = {
        int(value["completed_adaptive_trials"])
        for value in finalist_studies.values()
    }
    if completed_trial_counts != {int(args.optuna_trials)}:
        raise RuntimeError(
            "Finalist Optuna adaptive-trial budgets are unequal: "
            f"{sorted(completed_trial_counts)}"
        )
    finalist_trials = pd.concat(finalist_trial_parts, ignore_index=True)
    finalist_screen_summary = pd.concat(
        finalist_screen_summary_parts, ignore_index=True
    )
    finalist_config_confirm = pd.concat(
        finalist_config_confirm_parts, ignore_index=True
    )
    finalist_config_summary = pd.concat(
        finalist_config_summary_parts, ignore_index=True
    )
    finalist_trials.to_csv(
        output_dir / "xgb_finalist_optuna_trials.csv", index=False
    )
    finalist_screen_summary.to_csv(
        output_dir / "xgb_finalist_configuration_screen_summary.csv",
        index=False,
    )
    finalist_config_confirm.to_csv(
        output_dir / "xgb_finalist_configuration_confirmation_long.csv",
        index=False,
    )
    finalist_config_summary.to_csv(
        output_dir / "xgb_finalist_configuration_confirmation_summary.csv",
        index=False,
    )
    finalist_study_payload = {
        "contract_version": FINALIST_TUNING_CONTRACT_VERSION,
        "profile_hash": PROFILE_HASH,
        "finalist_group_ids": list(finalist_group_ids),
        "finalist_count": int(len(finalist_group_ids)),
        "finalist_union_policy": (
            "ALL_STAGE4_ONE_SE_ELIGIBLE_AND_COUNT_AP_RECALL_PARETO_PLUS_ANCHORS"
        ),
        "equal_completed_adaptive_trials": int(args.optuna_trials),
        "timeout_seconds": FINALIST_TUNING_TIMEOUT_SECONDS,
        "configuration_confirmation_budget": int(
            configuration_confirmation_budget
        ),
        "full_stage2_screen_reusable": bool(full_stage2_screen_reusable),
        "studies": finalist_studies,
        "outer_validation_rows_scored": 0,
        "final_holdout_rows_scored": 0,
    }
    (output_dir / "xgb_finalist_optuna_studies.json").write_text(
        json.dumps(
            strict_json_value(finalist_study_payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    selected_configurations_payload = {
        "contract_version": FINALIST_TUNING_CONTRACT_VERSION,
        "profile_hash": PROFILE_HASH,
        "finalist_group_ids": list(finalist_group_ids),
        "finalist_count": int(len(finalist_group_ids)),
        "configuration_selection_rule": (
            "PAIRED_AP_ONE_SE_THEN_CONDITIONAL_RECALL_10PCT_ONE_SE"
        ),
        "recall_reference_condition": (
            "BEST_MEAN_RECALL_AMONG_AP_SKILL_ONE_SE_PASS_CANDIDATES"
        ),
        "configurations": finalist_selected_configurations,
        "outer_validation_rows_scored": 0,
        "final_holdout_rows_scored": 0,
    }
    (output_dir / "xgb_finalist_selected_configurations.json").write_text(
        json.dumps(
            strict_json_value(selected_configurations_payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


    tuned_confirmation_parts: list[pd.DataFrame] = []
    for group_id in finalist_group_ids:
        features = list(confirmation_groups[group_id])
        configuration_for_group = finalist_selected_configurations[group_id]
        tuned_part = evaluate_feature_groups_multiseed(
            modeling_utils=modeling_utils,
            selection_frame=selection_frame,
            inner_folds=inner_folds,
            feature_types=feature_types,
            groups={group_id: features},
            config=config,
            history=configuration_for_group["selected_history"],
            model_spec=configuration_for_group["selected_model_spec"],
            seeds=seeds,
            train_max_rows=args.confirm_train_max_rows,
            validation_max_rows=args.validation_max_rows,
            max_rounds=args.max_rounds,
            early_stopping_rounds=args.early_stopping_rounds,
            device=args.device,
            n_jobs=args.n_jobs,
            rare_min_frequency=args.rare_min_frequency,
            stage_label="FINALIST_TUNED_CONFIRM",
        )
        tuned_part.insert(
            2,
            "selected_candidate_id",
            str(configuration_for_group["selected_candidate_id"]),
        )
        tuned_part.insert(
            3,
            "selected_history_id",
            str(configuration_for_group["selected_history"]["history_id"]),
        )
        tuned_part.insert(
            4,
            "selected_model_id",
            str(configuration_for_group["selected_model_spec"]["model_id"]),
        )
        tuned_part.insert(
            5,
            "ordered_feature_sha256",
            str(configuration_for_group["ordered_feature_sha256"]),
        )
        tuned_confirmation_parts.append(tuned_part)
    tuned_confirmation_long = pd.concat(
        tuned_confirmation_parts, ignore_index=True
    )
    (
        selected_group_id,
        tuned_confirmation_annotated,
        selection_reason,
    ) = choose_final_feature_group(
        tuned_confirmation_long,
        policy=final_selection_policy(args),
    )
    tuned_confirmation_long.to_csv(
        output_dir / "xgb_finalist_tuned_group_confirmation_long.csv",
        index=False,
    )
    tuned_confirmation_annotated.to_csv(
        output_dir / "xgb_finalist_tuned_group_confirmation_summary.csv",
        index=False,
    )


    selected_configuration = finalist_selected_configurations[selected_group_id]
    selected_features = list(confirmation_groups[selected_group_id])
    if not set(mandatory_control_features).issubset(selected_features):
        raise AssertionError("Selected XGBoost group lost the mandatory time control.")
    selected_history = dict(selected_configuration["selected_history"])
    selected_model_spec = dict(selected_configuration["selected_model_spec"])
    selected_optuna_trial_number = int(
        selected_configuration["selected_optuna_trial_number"]
    )
    selected_config_id = str(selected_configuration["selected_candidate_id"])
    selected_optuna_metadata = dict(finalist_studies[selected_group_id])
    selected_rows = tuned_confirmation_long.loc[
        tuned_confirmation_long["group_id"].astype(str).eq(selected_group_id)
    ]
    selected_rounds = int(
        round(float(pd.to_numeric(selected_rows["best_rounds"]).median()))
    )
    selected_summary = tuned_confirmation_annotated.loc[
        tuned_confirmation_annotated["group_id"].astype(str).eq(selected_group_id)
    ].iloc[0].to_dict()
    stage4_checkpoint = write_stage4_selection_checkpoint(
        output_dir=output_dir,
        args=args,
        seeds=seeds,
        status="RESOLVED",
        full_group_id=full_group_id,
        discovery_selected_group_id=discovery_selected_group_id,
        selected_group_id=selected_group_id,
        selected_features=selected_features,
        finalist_group_ids=finalist_group_ids,
        selected_configuration=selected_configuration,
        required_group_ids=required_confirmation_ids,
        confirmed_group_ids=list(confirmation_groups),
        pending_required_ids=[],
        pending_adaptive_ids=[],
        pass_records=pass_records,
    )




    warnings = governance_warnings(candidate_audit, selected_features)
    print(
        "POST-SELECTION DIAGNOSTICS: calendar control and shap.TreeExplainer",
        flush=True,
    )
    diagnostics_manifest = write_selected_model_diagnostics(
        modeling_utils=modeling_utils,
        selection_frame=selection_frame,
        inner_folds=inner_folds,
        feature_types=feature_types,
        selected_features=selected_features,
        config=config,
        history=selected_history,
        model_spec=selected_model_spec,
        seeds=seeds,
        train_max_rows=args.confirm_train_max_rows,
        validation_max_rows=args.validation_max_rows,
        max_rounds=args.max_rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        device=args.device,
        n_jobs=args.n_jobs,
        rare_min_frequency=args.rare_min_frequency,
        output_dir=output_dir,
    )
    diagnostics_manifest = add_business_retention_trend_diagnostics(
        diagnostics_manifest=diagnostics_manifest,
        population=prepared,
        test_idx=test_idx,
        config=config,
        selected_history=selected_history,
        output_dir=output_dir,
    )
    selected_payload = {
        "xgboost_version": xgb.__version__,
        "optuna_version": optuna.__version__,
        "shap_version": shap.__version__,
        "selection_method": (
            "manifest-driven full candidate universe -> training-only quality "
            "screen -> anchored deterministic Optuna TPE configuration tuning -> "
            "chronological inner-fold confirmation -> grouped total-gain "
            "ranking across chronological folds -> coarse cumulative forward-prefix "
            "validation -> iterative local one-feature count refinement -> "
            "adaptive multi-pass multi-seed confirmation -> paired-fold "
            "AP-skill/recall@10% one-standard-error discovery -> complete eligible/"
            "Pareto finalist "
            "union -> equal-budget group-specific Optuna retuning without timeout -> "
            "practical AP-skill/recall final selection with worst-fold guards"
        ),
        "stage4_selection_rule": STAGE4_SELECTION_RULE,
        "final_feature_selection_rule": FINAL_FEATURE_SELECTION_RULE,
        "final_feature_selection_policy": final_selection_policy(args),
        "final_feature_selection_policy_sha256": policy_sha256(final_selection_policy(args)),
        "selection_policy_module": file_info(Path(__file__).with_name("xgboost_selection_policy.py")),
        "stage4_selection_contract_version": STAGE4_SELECTION_CONTRACT_VERSION,
        "stage4_confirmation_policy": {
            "candidate_count_per_pass": int(args.confirm_candidate_count),
            "max_passes": int(args.confirm_max_passes),
            "completed_passes": int(len(pass_records)),
            "required_group_ids": list(required_confirmation_ids),
            "confirmed_group_ids": list(confirmation_groups),
            "discovery_selected_group_id": discovery_selected_group_id,
            "discovery_selection_reason": discovery_selection_reason,
            "adaptive_full_winner_scan": True,
            "binary_search_used": False,
            "performance_hierarchy": (
                "AP_SKILL_ONE_SE_THEN_CONDITIONAL_RECALL_10PCT_ONE_SE"
            ),
            "recall_reference_condition": (
                "BEST_MEAN_RECALL_AMONG_AP_SKILL_ONE_SE_PASS_CANDIDATES"
            ),
        },
        "stage4_selection_checkpoint": stage4_checkpoint,
        "finalist_tuning_policy": {
            "contract_version": FINALIST_TUNING_CONTRACT_VERSION,
            "finalist_group_ids": list(finalist_group_ids),
            "finalist_count": int(len(finalist_group_ids)),
            "union_policy": (
                "ALL_STAGE4_ONE_SE_ELIGIBLE_AND_COUNT_AP_RECALL_PARETO_PLUS_ANCHORS"
            ),
            "union_roles": [
                "STAGE4_ONE_SE_ELIGIBLE",
                "COUNT_AP_RECALL_PARETO",
                "STAGE4_SMALLEST_ONE_SE",
                "BEST_MEAN_AP_SKILL",
                "BEST_MEAN_RECALL_10PCT",
                "FULL_FEATURE_UNIVERSE",
            ],
            "equal_completed_adaptive_trials_per_group": int(
                args.optuna_trials
            ),
            "timeout_seconds": FINALIST_TUNING_TIMEOUT_SECONDS,
            "configuration_confirmation_budget_per_group": int(
                configuration_confirmation_budget
            ),
            "configuration_selection_rule": (
                "PAIRED_AP_ONE_SE_THEN_CONDITIONAL_RECALL_10PCT_ONE_SE"
            ),
            "recall_reference_condition": (
                "BEST_MEAN_RECALL_AMONG_AP_SKILL_ONE_SE_PASS_CANDIDATES"
            ),
            "full_stage2_screen_reused_if_exact": bool(
                full_stage2_screen_reusable
            ),
            "artifacts": {
                name: file_info(output_dir / name) for name in FINALIST_ARTIFACTS
            },
        },
        "gain_ranking_method": "mean grouped normalized total_gain across inner folds",
        "forward_step_size": int(args.forward_step_size),
        "forward_step_size_role": "coarse scan only",
        "local_refinement_pass_count": int(refinement_pass_count),
        "locally_refined_feature_counts": sorted(
            len(features) for features in refinement_groups.values()
        ),
        "previous_feature_group_used": False,
        "previous_feature_plan_inventory_used": False,
        "previous_selected_parameter_file_used": False,
        "prepared_starting_pool_used": starting_pool_path is not None,
        "prepared_starting_pool": (
            None if starting_pool_path is None else file_info(starting_pool_path)
        ),
        "manifest_audit_candidate_count": int(len(audit_candidates)),
        "quality_screened_starting_feature_count": int(len(screened_features)),
        "model_feature_universe_count": int(len(model_features)),
        "mandatory_control_features": mandatory_control_features,
        "cohort_calendar_year_contract": cohort_control_contract,
        "selected_group_id": selected_group_id,
        "selected_raw_feature_count": int(len(selected_features)),
        "selected_candidate_feature_count": int(
            len(selected_features) - len(mandatory_control_features)
        ),
        "selected_features": selected_features,
        "selection_reason": selection_reason,
        "discovery_selected_group_id": discovery_selected_group_id,
        "discovery_selection_reason": discovery_selection_reason,
        "discovery_selected_inner_summary": discovery_selected_summary,
        "initial_forward_selection_reason": initial_reason,
        "history_candidate_grid": history_candidates,
        "selected_configuration_id": selected_config_id,
        "selected_history": selected_history,
        "selected_model_spec": selected_model_spec,
        "selected_optuna_trial_number": selected_optuna_trial_number,
        "optuna_study": selected_optuna_metadata,
        "discovery_selected_configuration_id": discovery_selected_config_id,
        "discovery_selected_history": discovery_selected_history,
        "discovery_selected_model_spec": discovery_selected_model_spec,
        "discovery_selected_optuna_trial_number": (
            discovery_selected_optuna_trial_number
        ),
        "discovery_optuna_study": optuna_metadata,
        "recommended_boosting_rounds_from_inner_confirmation": selected_rounds,
        "seeds": seeds,
        "primary_statistical_metric": "CHURN_AVERAGE_PRECISION_SKILL",
        "co_primary_business_metric": "CHURN_RECALL_10PCT",
        "reported_business_translation": [
            "TOP10_CHURN_LIFT",
            "CHURN_PRECISION_10PCT",
        ],
        "ranking_guardrail": "RETENTION_ROC_AUC",
        "calibration_guardrails": [
            "LOG_LOSS",
            "BRIER_SCORE",
            "CALIBRATION_GAP",
        ],
        "diagnostics": diagnostics_manifest,
        "stage3_discovery_tolerances": {
            "ap_skill": args.ap_skill_tolerance,
            "lift10": args.lift_tolerance,
            "roc_auc": args.auc_tolerance,
            "recent_ap_skill": args.recent_ap_skill_tolerance,
        },
        "selection_tolerances": {
            "ap_skill": args.ap_skill_tolerance,
            "lift10": args.lift_tolerance,
            "roc_auc": args.auc_tolerance,
            "recent_ap_skill": args.recent_ap_skill_tolerance,
            "role": "STAGE3_DISCOVERY_ONLY",
        },
        "stage3_discovery_tolerances_affect_final_choice_directly": False,
        "selected_inner_summary": selected_summary,
        "governance_review_features_in_selected_group": warnings,
        "outer_validation_rows_scored": 0,
        "final_holdout_rows_identified": int(len(test_idx)),
        "final_holdout_rows_scored": 0,
    }
    (output_dir / "selected_features.json").write_text(
        json.dumps(
            strict_json_value(selected_payload),
            indent=2,
            default=str,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "selected_features.txt").write_text(
        "\n".join(selected_features) + "\n", encoding="utf-8"
    )




    print("STAGE 5/5: WRITE AUDIT AND SUMMARY", flush=True)
    test_dates = pd.to_datetime(prepared.loc[test_idx, config.cohort_date_col])
    run_info = {
        "profile": PROFILE_PAYLOAD,
        "profile_hash": PROFILE_HASH,
        "xgboost_version": xgb.__version__,
        "optuna_version": optuna.__version__,
        "shap_version": shap.__version__,
        "device": args.device,
        "n_jobs": args.n_jobs,
        "project_root": str(root),
        "input": file_info(input_path),
        "manifest": file_info(manifest_path),
        "script": file_info(Path(__file__).resolve()),
        "modeling_utils": file_info(modeling_utils_path),
        "model_diagnostics": file_info(
            PROJECT_ROOT / "c_modeling" / "model_diagnostics.py"
        ),
        "feature_routing": file_info(feature_routing_path),
        "feature_types_file": file_info(feature_types_contract_path),
        "routing_audit": file_info(routing_audit_path),
        "starting_feature_pool": (
            None if starting_pool_path is None else file_info(starting_pool_path)
        ),
        "preparation_info": file_info(preparation_info_path),
        "outer_fold_definitions": file_info(outer_fold_definitions_path),
        "inner_fold_definitions": file_info(inner_fold_definitions_path),
        "random_seed": BASE_SEED,
        "manifest_audit_candidate_count": int(len(audit_candidates)),
        "quality_screened_starting_feature_count": int(len(screened_features)),
        "model_feature_universe_count": int(len(model_features)),
        "mandatory_control_features": mandatory_control_features,
        "cohort_calendar_year_contract": cohort_control_contract,
        "selected_raw_feature_count": int(len(selected_features)),
        "selected_candidate_feature_count": int(
            len(selected_features) - len(mandatory_control_features)
        ),
        "selected_group_id": selected_group_id,
        "stage4_selection_rule": STAGE4_SELECTION_RULE,
        "final_feature_selection_rule": FINAL_FEATURE_SELECTION_RULE,
        "final_feature_selection_policy": final_selection_policy(args),
        "final_feature_selection_policy_sha256": policy_sha256(final_selection_policy(args)),
        "selection_policy_module": file_info(Path(__file__).with_name("xgboost_selection_policy.py")),
        "stage4_selection_contract_version": STAGE4_SELECTION_CONTRACT_VERSION,
        "stage4_selection_checkpoint": file_info(
            output_dir / STAGE4_SELECTION_CHECKPOINT
        ),
        "stage4_completed_confirmation_passes": int(len(pass_records)),
        "stage4_discovery_selected_group_id": discovery_selected_group_id,
        "finalist_tuning_contract_version": FINALIST_TUNING_CONTRACT_VERSION,
        "finalist_group_ids": list(finalist_group_ids),
        "finalist_equal_completed_adaptive_trials": int(args.optuna_trials),
        "finalist_optuna_timeout_seconds": FINALIST_TUNING_TIMEOUT_SECONDS,
        "finalist_artifacts": {
            name: file_info(output_dir / name) for name in FINALIST_ARTIFACTS
        },
        "selection_population_rows": int(len(selection_frame)),
        "inner_fold_count": int(len(inner_folds)),
        "history_candidate_grid": history_candidates,
        "selected_configuration_id": selected_config_id,
        "selected_history": selected_history,
        "selected_model_spec": selected_model_spec,
        "selected_optuna_trial_number": selected_optuna_trial_number,
        "optuna_study": selected_optuna_metadata,
        "discovery_selected_configuration_id": discovery_selected_config_id,
        "discovery_selected_history": discovery_selected_history,
        "discovery_selected_model_spec": discovery_selected_model_spec,
        "discovery_selected_optuna_trial_number": (
            discovery_selected_optuna_trial_number
        ),
        "discovery_optuna_study": optuna_metadata,
        "outer_fold_count_defined_but_not_scored": int(len(outer_folds)),
        "final_holdout_rows_identified": int(len(test_idx)),
        "final_holdout_start": str(test_dates.min().date()),
        "final_holdout_end": str(test_dates.max().date()),
        "outer_validation_rows_scored": 0,
        "final_holdout_rows_scored": 0,
        "arguments": vars(args),
        "diagnostics": diagnostics_manifest,
    }
    lines = [
        "# XGBoost feature selection",
        "",
        "## What this run used",
        "",
        "- Current modeling parquet",
        "- Current feature manifest",
        "- Manifest/data eligibility audit",
        "- Training-only quality screening",
        "- Chronological inner folds inside the oldest outer-training population",
        "- Deterministic Optuna TPE search with established coverage anchors",
        "- XGBoost AUCPR early stopping with churn as the positive class",
        "- Grouped total-gain ranking averaged across chronological inner folds",
        "- Coarse-to-fine cumulative forward-prefix validation",
        "- Required eligible/Pareto-relevant prefixes retained across deterministic confirmation passes",
        "- Full-pool winner triggers a deterministic linear intermediate-prefix scan",
        "- Finalists include every confirmed one-SE-eligible and count/AP/recall Pareto-relevant prefix, plus the discovery, AP, recall, and full-pool anchors",
        "- Every distinct finalist receives the same completed Optuna trial budget with timeout disabled",
        "- Final choice uses practical mean-loss limits and worst-fold guards after group-specific retuning",
        f"- Final feature-count policy: `{final_selection_policy(args)}`",
        "- No hard feature-count cap",
        f"- Multi-seed chronological confirmation passes: **{len(pass_records)}**",
        "",
        "It did not read any previously selected feature group or feature-plan inventory.",
        "",
        "## Selected group",
        "",
        f"- Group ID: `{selected_group_id}`",
        f"- Stage-4 discovery group ID: `{discovery_selected_group_id}`",
        f"- Raw feature count: **{len(selected_features)}**",
        f"- Mandatory time control: `{cohort_control_feature}`",
        f"- Selection reason: {selection_reason}",
        f"- Selected history: `{selected_history['history_id']}`",
        f"- Selected model specification: `{selected_model_spec['model_id']}`",
        f"- Selected group-specific configuration: `{selected_config_id}`",
        f"- Selected Optuna trial: **{selected_optuna_trial_number}**",
        f"- Median recommended boosting rounds: **{selected_rounds}**",
        "",
        "```text",
        *selected_features,
        "```",
        "",
        "## Validation boundary",
        "",
        "- Outer validation folds scored: **0**",
        f"- Final holdout rows identified: **{len(test_idx):,}**",
        "- Final holdout rows scored: **0**",
    ]
    if warnings:
        lines.extend(
            [
                "",
                "## Governance note",
                "",
                "The following selected fields are not marked as ordinary standard-eligible "
                "in the current audit and therefore remain experiment-only until approved:",
                "",
            ]
        )
        for item in warnings:
            lines.append(
                f"- `{item['feature']}` — enabled={item['original_enabled_flag']}; "
                f"governance={item['governance_status']}"
            )
    (output_dir / "feature_selection_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


    # Completion marker: atomically publish only after every required output.
    run_info_path = output_dir / "run_info.json"
    run_info_temporary = run_info_path.with_name(run_info_path.name + ".tmp")
    run_info_temporary.write_text(
        json.dumps(
            strict_json_value(run_info),
            indent=2,
            default=str,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    os.replace(run_info_temporary, run_info_path)




    print("", flush=True)
    print("XGBOOST FEATURE SELECTION COMPLETE", flush=True)
    print(f"Selected group: {selected_group_id}", flush=True)
    print(f"Selected raw features: {len(selected_features)}", flush=True)
    print(f"Feature list: {output_dir / 'selected_features.txt'}", flush=True)
    print("Outer validation rows scored: 0", flush=True)
    print(f"Final holdout rows scored: 0 of {len(test_idx)}", flush=True)








if __name__ == "__main__":
    main()
