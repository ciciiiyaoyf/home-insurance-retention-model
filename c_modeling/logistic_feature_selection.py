#!/usr/bin/env python
"""Grouped sequential floating-forward selection for Logistic Regression.




Statistical contract
--------------------
* Preserve an intercept-only audit fit, force the continuous calendar-year
  control into the working structure, and then add complete raw-feature groups.
* Select one prespecified recent training-history policy on
  the chronological development folds before changing the feature structure.
* Keep Ridge logistic regression, the selected history policy, and C fixed
  while features change.
* Evaluate every proposed addition on the configured chronological development
  folds, using the same capped rows for paired comparisons.
* Optimize prevalence-normalized churn average-precision skill, with
  recall@10% as the business co-primary. Lift@10% is the backward-compatible
  fixed-capacity gate and fold stability remains required.
* After each addition, repeatedly permit an included feature to float out only
  when its removal materially improves the current structure under those same
  paired guardrails. A visited-set prevents cycles.
* Score every accepted unique structure on all rows of those same
  development folds, select the smallest non-inferior structure, and only then
  tune Ridge C.
* Never prepare, fit, predict, or score the final six-month holdout.




This is a greedy sequential floating-forward search, not exhaustive
best-subset selection.  The full-row checks are development confirmation,
not an independent holdout, because capped versions of the same periods drive
the search.




The module uses ``logistic_utils.py`` for
fold-local preprocessing, sparse raw-feature blocks, metrics, and atomic fit
checkpoints. Each encoded block belonging to a raw feature enters or leaves as
one indivisible group.
"""




from __future__ import annotations




import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any




import numpy as np
import pandas as pd
import sklearn




PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))




from func.tenure_profile import (  # noqa: E402
    PROFILE_HASH_ENV,
    profile_from_environment,
)




import logistic_utils as base
from feature_routing import (
    route_candidate_list_and_types,
    validate_one_variant_per_concept,
)
from modeling_utils import (
    COHORT_CALENDAR_YEAR_FEATURE,
    COHORT_CALENDAR_YEAR_TYPE,
    STATE_GROUP_FEATURE,
    WRITTEN_PREM_BIN_FEATURE,
    apply_quantile_bin_artifact,
    cohort_calendar_year_contract,
    select_recent_mature_training_indices,
    verify_fold_definition_contract,
)
from state_grouping import apply_state_grouper
from c_modeling.model_diagnostics import (
    ACTUAL_COLUMN,
    WITH_YEAR_COLUMN,
    YEAR_CONTROL_FEATURE,
    YEAR_MONTH_COLUMN,
    business_retention_monthly_summary,
    write_business_retention_trend_artifacts,
    write_calendar_control_artifacts,
)








OUT = base.OUT
CHECKPOINT_FILE = OUT / ".fit_checkpoint.sqlite3"
CONFIG_FILE = OUT / "run_config.json"
STATUS_FILE = OUT / "run_status.json"








PROFILE_PAYLOAD = profile_from_environment().payload
PROFILE_HASH = os.environ.get(PROFILE_HASH_ENV, "").strip()
if not PROFILE_HASH:
    raise RuntimeError(f"{PROFILE_HASH_ENV} is required for checkpoint isolation.")




EXPECTED_FEATURE_COUNT = int(
    os.environ.get("RETENTION_LR_EXPECTED_FEATURE_COUNT", "44")
)
FIXED_C = float(os.environ.get("RETENTION_LR_FIXED_C", "0.01"))
MAX_ACTIONS = int(os.environ.get("RETENTION_LR_MAX_ACTIONS", "500"))
MIN_STABLE_FOLDS = int(
    os.environ.get("RETENTION_LR_MIN_STABLE_FOLDS", "3")
)
OUTER_FOLDS = int(os.environ.get("RETENTION_LR_OUTER_FOLDS", "5"))
OUTER_VALIDATION_MONTHS = int(
    os.environ.get("RETENTION_LR_OUTER_VALIDATION_MONTHS", "3")
)
HOLDOUT_MONTHS = int(os.environ.get("RETENTION_LR_HOLDOUT_MONTHS", "6"))
MIN_OUTER_TRAIN_MONTHS = int(
    os.environ.get("RETENTION_LR_MIN_OUTER_TRAIN_MONTHS", "12")
)
LABEL_AVAILABILITY_BUFFER_DAYS = int(
    os.environ.get("RETENTION_LR_LABEL_AVAILABILITY_BUFFER_DAYS", "0")
)
STRICT_DATASET_CONTRACT = os.environ.get(
    "RETENTION_LR_STRICT_DATASET_CONTRACT", "1"
).strip() in {"1", "TRUE", "T", "Y", "YES"}
OUTER_FOLD_DEFINITIONS = base.required_path(
    "RETENTION_LR_OUTER_FOLD_DEFINITIONS"
)
INNER_FOLDS = int(os.environ.get("RETENTION_LR_INNER_FOLDS", "3"))
INNER_VALIDATION_MONTHS = int(
    os.environ.get("RETENTION_LR_INNER_VALIDATION_MONTHS", "3")
)
MIN_INNER_TRAIN_MONTHS = int(
    os.environ.get("RETENTION_LR_MIN_INNER_TRAIN_MONTHS", "12")
)
INNER_FOLD_DEFINITIONS = base.required_path(
    "RETENTION_LR_INNER_FOLD_DEFINITIONS"
)




AP_STEP_TOL = float(os.environ.get("RETENTION_LR_AP_STEP_TOL", "0.00025"))
AP_TOTAL_TOL = float(os.environ.get("RETENTION_LR_AP_TOTAL_TOL", "0.001"))
LIFT_STEP_TOL = float(os.environ.get("RETENTION_LR_LIFT_STEP_TOL", "0.01"))
LIFT_TOTAL_TOL = float(os.environ.get("RETENTION_LR_LIFT_TOTAL_TOL", "0.02"))
FLOATING_REMOVAL_AP_GAIN = float(
    os.environ.get("RETENTION_LR_FLOATING_REMOVAL_AP_GAIN", "0.00025")
)




RUNTIME_BUDGET_HOURS = float(
    os.environ.get("RETENTION_LR_RUNTIME_BUDGET_HOURS", "0")
)
MAX_NEW_FITS = int(os.environ.get("RETENTION_LR_MAX_NEW_FITS", "0"))
PROTECTED_FEATURES_FILE = os.environ.get(
    "RETENTION_LR_PROTECTED_FEATURES_FILE", ""
).strip()
C_GRID = tuple(
    sorted(
        {
            float(value.strip())
            for value in os.environ.get(
                "RETENTION_LR_C_GRID", "0.001,0.003,0.01,0.03,0.1,0.3,1.0"
            ).split(",")
            if value.strip()
        }
    )
)




DEFAULT_HISTORY_GRID: tuple[dict[str, Any], ...] = (
    {"history_id": "RECENT_36M", "kind": "recent", "months": 36},
    {"history_id": "RECENT_48M", "kind": "recent", "months": 48},
    {"history_id": "RECENT_60M", "kind": "recent", "months": 60},
)
DEFAULT_HISTORY_SELECTION_POLICY: dict[str, Any] = {
    "screening_c": FIXED_C,
    "primary_metric": "mean_churn_ap_skill",
    "secondary_metric": "mean_top10_lift",
    "tertiary_metric": "newest_fold_churn_ap_skill",
    "sealed_holdout_used": False,
}
SCHEMA_VERSION = int(os.environ.get("RETENTION_LR_SCHEMA_VERSION", "19"))
MODE = os.environ.get(
    "RETENTION_LR_MODE",
    "history_tuned_calendar_year_control_grouped_sequential_floating_forward",
).strip()
MANDATORY_CONTROL_FEATURES = (COHORT_CALENDAR_YEAR_FEATURE,)




def normalized_history_grid(payload: Any) -> tuple[dict[str, Any], ...]:
    """Validate the prespecified no-all-history Logistic experiment."""


    if not isinstance(payload, list):
        raise ValueError("RETENTION_LR_HISTORY_GRID_JSON must encode a list.")
    normalized: list[dict[str, Any]] = []
    for position, raw in enumerate(payload, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"History candidate {position} must be an object.")
        history_id = str(raw.get("history_id", "")).strip()
        kind = str(raw.get("kind", "")).strip().lower()
        if not history_id:
            raise ValueError(f"History candidate {position} has no history_id.")
        if kind != "recent":
            raise ValueError(
                f"History candidate {history_id} must use recent equal-weight "
                "history; decay and ALL_HISTORY are not permitted."
            )
        expected_keys = {"history_id", "kind", "months"}
        if set(raw) != expected_keys:
            raise ValueError(
                f"History candidate {history_id} must contain exactly "
                f"{sorted(expected_keys)}."
            )
        try:
            months = int(raw["months"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"History candidate {history_id} has invalid months."
            ) from exc
        if isinstance(raw["months"], bool) or months <= 0 or float(
            raw["months"]
        ) != months:
            raise ValueError(
                f"History candidate {history_id} months must be a positive integer."
            )
        candidate: dict[str, Any] = {
            "history_id": history_id,
            "kind": kind,
            "months": months,
        }
        normalized.append(candidate)
    if len({item["history_id"] for item in normalized}) != len(normalized):
        raise ValueError("History candidate IDs must be unique.")
    expected = tuple(dict(item) for item in DEFAULT_HISTORY_GRID)
    if tuple(normalized) != expected:
        raise ValueError(
            "The Logistic history grid must exactly match the prespecified "
            "RECENT_36M/48M/60M equal-weight experiment."
        )
    return tuple(normalized)




def history_grid_from_environment() -> tuple[dict[str, Any], ...]:
    raw = os.environ.get("RETENTION_LR_HISTORY_GRID_JSON", "").strip()
    if not raw:
        return tuple(dict(item) for item in DEFAULT_HISTORY_GRID)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("RETENTION_LR_HISTORY_GRID_JSON is invalid JSON.") from exc
    return normalized_history_grid(payload)




def history_selection_policy_from_environment() -> dict[str, Any]:
    raw = os.environ.get(
        "RETENTION_LR_HISTORY_SELECTION_POLICY_JSON", ""
    ).strip()
    if not raw:
        return dict(DEFAULT_HISTORY_SELECTION_POLICY)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "RETENTION_LR_HISTORY_SELECTION_POLICY_JSON is invalid JSON."
        ) from exc
    if payload != DEFAULT_HISTORY_SELECTION_POLICY:
        raise ValueError(
            "The Logistic history-selection policy does not match the "
            "prespecified AP/Lift@10/newest-fold-AP contract."
        )
    return dict(payload)




HISTORY_GRID = history_grid_from_environment()
HISTORY_SELECTION_POLICY = history_selection_policy_from_environment()




if base.PRIMARY_METRIC != "ap_skill":
    raise ValueError(
        "Set RETENTION_LR_PRIMARY_METRIC=ap_skill. "
        "Logistic selection uses prevalence-normalized churn_ap_skill."
    )
if SCHEMA_VERSION != 19:
    raise ValueError("RETENTION_LR_SCHEMA_VERSION must be 19.")
if MODE != "history_tuned_calendar_year_control_grouped_sequential_floating_forward":
    raise ValueError(
        "RETENTION_LR_MODE must be "
        "history_tuned_calendar_year_control_grouped_sequential_floating_forward."
    )
if EXPECTED_FEATURE_COUNT < 1:
    raise ValueError("RETENTION_LR_EXPECTED_FEATURE_COUNT must be positive.")
if min(
    OUTER_FOLDS,
    OUTER_VALIDATION_MONTHS,
    HOLDOUT_MONTHS,
    MIN_OUTER_TRAIN_MONTHS,
    INNER_FOLDS,
    INNER_VALIDATION_MONTHS,
    MIN_INNER_TRAIN_MONTHS,
) < 1:
    raise ValueError("Logistic split settings must all be positive.")
if not 1 <= MIN_STABLE_FOLDS <= INNER_FOLDS:
    raise ValueError(
        "RETENTION_LR_MIN_STABLE_FOLDS must be between 1 and RETENTION_LR_INNER_FOLDS."
    )
if MAX_ACTIONS < 1:
    raise ValueError("RETENTION_LR_MAX_ACTIONS must be positive.")
if FIXED_C <= 0 or not C_GRID or any(value <= 0 for value in C_GRID):
    raise ValueError("Ridge C values must be positive.")
if not any(math.isclose(value, FIXED_C, rel_tol=0, abs_tol=1e-12) for value in C_GRID):
    raise ValueError(
        "RETENTION_LR_C_GRID must include RETENTION_LR_FIXED_C for the tuning guardrail."
    )
if RUNTIME_BUDGET_HOURS < 0 or (
    0 < RUNTIME_BUDGET_HOURS < 0.25
):
    raise ValueError(
        "Runtime budget must be 0 (unlimited) or at least 0.25 hours."
    )
if MAX_NEW_FITS < 0 or 0 < MAX_NEW_FITS < 10:
    raise ValueError("Fit budget must be 0 (unlimited) or at least 10.")
for value, name in (
    (AP_STEP_TOL, "AP step tolerance"),
    (AP_TOTAL_TOL, "AP total tolerance"),
    (LIFT_STEP_TOL, "Lift step tolerance"),
    (LIFT_TOTAL_TOL, "Lift total tolerance"),
    (FLOATING_REMOVAL_AP_GAIN, "floating-removal AP gain"),
):
    if value < 0:
        raise ValueError(f"{name} cannot be negative.")








class BudgetPause(RuntimeError):
    """A safe pause raised only before beginning a new fit."""




    def __init__(self, reason: str, stage: str, fits: int, elapsed: float):
        super().__init__(reason)
        self.stage = stage
        self.fits = fits
        self.elapsed = elapsed








class RunBudget:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.seconds = (
            None
            if RUNTIME_BUDGET_HOURS == 0
            else RUNTIME_BUDGET_HOURS * 3600.0
        )
        self.new_fits = 0




    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started




    def before_fit(self, stage: str) -> None:
        if self.seconds is not None and self.elapsed >= self.seconds:
            raise BudgetPause(
                f"runtime budget of {RUNTIME_BUDGET_HOURS:g} hours reached",
                stage,
                self.new_fits,
                self.elapsed,
            )
        if MAX_NEW_FITS > 0 and self.new_fits >= MAX_NEW_FITS:
            raise BudgetPause(
                f"new-fit budget of {MAX_NEW_FITS} reached",
                stage,
                self.new_fits,
                self.elapsed,
            )
        self.new_fits += 1








def ordered(pool: list[str], values: set[str] | list[str]) -> list[str]:
    wanted = set(values)
    return [feature for feature in pool if feature in wanted]








def feature_set_hash(features: list[str]) -> str:
    return base.feature_hash(features)






def history_fit_identity(strategy: dict[str, Any]) -> str:
    canonical = json.dumps(
        base.jsonable(strategy),
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        f"{strategy['history_id']}:"
        f"{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:20]}"
    )






def filtered_history_definition(
    frame: pd.DataFrame,
    definition: dict[str, Any],
    cohort_col: str,
    strategy: dict[str, Any],
) -> dict[str, Any]:
    """Apply a recent hard cap without changing validation membership."""


    train_idx = np.asarray(definition["train_idx"])
    validation_idx = np.asarray(definition["validation_idx"])
    validation_start = pd.to_datetime(
        frame.loc[validation_idx, cohort_col], errors="raise"
    ).min()
    if strategy.get("kind") != "recent":
        raise ValueError(
            "Logistic history filtering permits only RECENT_36M/48M/60M "
            "equal-weight candidates."
        )
    train_dates = pd.to_datetime(
        frame.loc[train_idx, cohort_col], errors="raise"
    )
    if train_dates.ge(validation_start).any():
        raise ValueError(
            f"{strategy['history_id']} received non-chronological training rows "
            f"for fold {definition.get('fold', '?')}."
        )
    selected_train_idx, history_audit = select_recent_mature_training_indices(
        frame,
        train_idx,
        cohort_col=cohort_col,
        months=int(strategy["months"]),
    )
    if not len(selected_train_idx):
        raise ValueError(
            f"{strategy['history_id']} removed every training row for fold "
            f"{definition.get('fold', '?')}."
        )
    if np.intersect1d(selected_train_idx, validation_idx).size:
        raise AssertionError("History-filtered training and validation rows overlap.")
    output = dict(definition)
    output["train_idx"] = selected_train_idx
    output["validation_idx"] = validation_idx.copy()
    output["history_id"] = strategy["history_id"]
    output["history_anchor"] = history_audit["history_anchor"]
    output["history_anchor_month"] = history_audit["anchor_month"]
    output["history_cutoff"] = history_audit["cutoff_month_inclusive"]
    output["history_available_months"] = history_audit[
        "available_months_in_window"
    ]
    output["history_validation_start"] = validation_start
    return output






def history_definitions(
    frame: pd.DataFrame,
    definitions: list[dict[str, Any]],
    cohort_col: str,
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        filtered_history_definition(frame, definition, cohort_col, strategy)
        for definition in definitions
    ]






def history_training_weights(
    train_frame: pd.DataFrame,
    cohort_col: str,
    strategy: dict[str, Any],
) -> np.ndarray | None:
    """Return equal weights implicitly for a governed recent-history window."""


    del train_frame, cohort_col
    if strategy.get("kind") != "recent":
        raise ValueError(
            "Logistic history weights permit only equal-weight recent history."
        )
    return None






def history_population_identity(
    frame: pd.DataFrame,
    definitions: list[dict[str, Any]],
    cohort_col: str,
) -> dict[str, Any]:
    candidates = []
    for strategy in HISTORY_GRID:
        custom = history_definitions(frame, definitions, cohort_col, strategy)
        candidates.append(
            {
                "history": dict(strategy),
                "history_identity": history_fit_identity(strategy),
                "folds": [
                    {
                        "fold": int(item["fold"]),
                        "train_rows": int(len(item["train_idx"])),
                        "validation_rows": int(len(item["validation_idx"])),
                        "train_index_hash": index_hash(
                            pd.Index(item["train_idx"])
                        ),
                        "validation_index_hash": index_hash(
                            pd.Index(item["validation_idx"])
                        ),
                    }
                    for item in custom
                ],
            }
        )
    payload = {
        "history_grid": [dict(item) for item in HISTORY_GRID],
        "sample_train_max_rows": base.TRAIN_MAX_ROWS,
        "sample_validation_max_rows": base.VALIDATION_MAX_ROWS,
        "sampling_seed": base.SEED,
        "candidates": candidates,
    }
    payload["identity_hash"] = hashlib.sha256(
        json.dumps(base.jsonable(payload), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload






def validate_or_write_history_population_identity(payload: dict[str, Any]) -> None:
    path = OUT / "history_population_identity.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("identity_hash") != payload["identity_hash"]:
            raise RuntimeError(
                "History-screen training or validation populations changed for "
                "this RunTag. Start a new RunTag instead of reusing checkpoints."
            )
        return
    if CHECKPOINT_FILE.is_file():
        raise RuntimeError(
            "A fit checkpoint exists without history_population_identity.json; "
            "refusing unsafe checkpoint reuse. Start a new RunTag."
        )
    base.atomic_write_json(path, payload)






def prepare_history_caches(
    frame: pd.DataFrame,
    definitions: list[dict[str, Any]],
    feature_types: dict[str, str],
    preprocessing_spec: Any,
    project_config: Any,
    pool: list[str],
    strategy: dict[str, Any],
    *,
    sampled: bool,
) -> tuple[list[dict[str, Any]], list[base.FoldMatrixCache]]:
    custom = history_definitions(
        frame,
        definitions,
        project_config.cohort_date_col,
        strategy,
    )
    if sampled:
        prepared = base.prepare_fold_objects(
            frame,
            custom,
            feature_types,
            preprocessing_spec,
            train_max_rows=base.TRAIN_MAX_ROWS,
            validation_max_rows=base.VALIDATION_MAX_ROWS,
            cohort_col=project_config.cohort_date_col,
            target_col=project_config.target_col,
        )
    else:
        prepared = [
            base.PreparedFold(
                int(definition.get("fold", position)),
                frame.loc[np.asarray(definition["train_idx"])],
                frame.loc[np.asarray(definition["validation_idx"])],
                feature_types,
                preprocessing_spec,
                target_col=project_config.target_col,
                cohort_col=project_config.cohort_date_col,
            )
            for position, definition in enumerate(custom, start=1)
        ]
    identity = history_fit_identity(strategy)
    caches = [
        base.FoldMatrixCache.build(
            fold,
            pool,
            sample_weight=history_training_weights(
                fold.train,
                project_config.cohort_date_col,
                strategy,
            ),
            history_identity=identity,
        )
        for fold in prepared
    ]
    return custom, caches








def read_protected(pool: list[str]) -> list[str]:
    values = list(MANDATORY_CONTROL_FEATURES)
    if PROTECTED_FEATURES_FILE:
        path = Path(PROTECTED_FEATURES_FILE).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Protected-feature file not found: {path}")
        supplied = [
            value.strip().upper()
            for value in path.read_text(encoding="utf-8-sig").splitlines()
            if value.strip()
        ]
        if len(supplied) != len(set(supplied)):
            raise ValueError("Protected-feature file contains duplicates.")
        values.extend(supplied)
    values = list(dict.fromkeys(values))
    unknown = [value for value in values if value not in pool]
    if unknown:
        raise ValueError("Protected features are outside the pool: " + ", ".join(unknown))
    return ordered(pool, values)








def read_family_feature_contract(pool: list[str]) -> dict[str, str]:
    """Verify the preparation-time Logistic types and routing decisions."""




    try:
        raw_types = json.loads(
            base.FEATURE_TYPES_CONTRACT.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid Logistic family feature-types artifact.") from exc
    if not isinstance(raw_types, dict):
        raise ValueError("Logistic family feature-types artifact must be an object.")
    feature_types = {
        str(feature).strip().upper(): str(var_type).strip().lower()
        for feature, var_type in raw_types.items()
    }
    if set(feature_types) != set(pool):
        raise ValueError("Logistic pool and family feature-types artifact disagree.")
    if any(value not in {"continuous", "discrete"} for value in feature_types.values()):
        raise ValueError("Logistic family feature-types artifact has invalid types.")




    audit = pd.read_csv(base.ROUTING_AUDIT, dtype=str, keep_default_na=False)
    required = {
        "model_family",
        "selected_feature",
        "quality_screen_decision",
    }
    if not required.issubset(audit.columns):
        raise ValueError("Feature-routing audit is missing required columns.")
    logistic = audit.loc[
        audit["model_family"].str.strip().str.lower().eq("logistic")
        & audit["quality_screen_decision"].str.strip().str.upper().eq("KEEP")
    ]
    audited = logistic["selected_feature"].str.strip().str.upper().tolist()
    if len(audited) != len(set(audited)) or set(audited) != set(pool):
        raise ValueError("Logistic pool and kept feature-routing audit disagree.")
    return feature_types








def config_payload(
    candidate_pool: list[str],
    model_pool: list[str],
    protected: list[str],
) -> dict[str, Any]:
    immutable = {
        "schema_version": SCHEMA_VERSION,
        "profile": PROFILE_PAYLOAD,
        "profile_hash": PROFILE_HASH,
        "mode": MODE,
        "script": base.file_identity(Path(__file__).resolve(), True),
        "logistic_utils": base.file_identity(Path(base.__file__).resolve(), True),
        "modeling_utils": base.file_identity(
            base.CODE / "modeling_utils.py", True
        ),
        "model_diagnostics": base.file_identity(
            PROJECT_ROOT / "c_modeling" / "model_diagnostics.py", True
        ),
        "feature_routing": base.file_identity(
            base.CODE / "feature_routing.py", True
        ),
        "state_grouping": base.file_identity(
            base.CODE / "state_grouping.py", True
        ),
        "feature_types_file": base.file_identity(
            base.FEATURE_TYPES_CONTRACT, True
        ),
        "routing_audit": base.file_identity(base.ROUTING_AUDIT, True),
        "preparation_info": base.file_identity(base.PREPARATION_INFO, True),
        "sklearn_version": sklearn.__version__,
        "max_iter_ridge": base.MAX_ITER_RIDGE,
        "data": base.file_identity(base.DATA, True),
        "manifest": base.file_identity(base.MANIFEST, True),
        "pool_file": base.file_identity(base.POOL, True),
        "outer_fold_definitions": base.file_identity(
            OUTER_FOLD_DEFINITIONS, True
        ),
        "inner_fold_definitions": base.file_identity(
            INNER_FOLD_DEFINITIONS, True
        ),
        "pool_hash": feature_set_hash(model_pool),
        "pool": model_pool,
        "quality_screened_candidate_pool": candidate_pool,
        "expected_feature_count": EXPECTED_FEATURE_COUNT,
        "model_feature_count": len(model_pool),
        "mandatory_control_features": list(MANDATORY_CONTROL_FEATURES),
        "cohort_calendar_year_contract": cohort_calendar_year_contract(),
        "protected_features": protected,
        "selection_metric": "churn_ap_skill",
        "metric_definition": (
            "(churn_average_precision - churn_prevalence) / "
            "(1 - churn_prevalence)"
        ),
        "operational_guardrail": "top10_lift",
        "operational_guardrail_business_alias": "churn_recall_10pct",
        "fixed_selection_c": FIXED_C,
        "ridge_c_grid": list(C_GRID),
        "history_grid": [dict(item) for item in HISTORY_GRID],
        "history_selection_policy": dict(HISTORY_SELECTION_POLICY),
        "sampled_development_folds": INNER_FOLDS,
        "outer_split_settings": {
            "folds": OUTER_FOLDS,
            "validation_months": OUTER_VALIDATION_MONTHS,
            "holdout_months": HOLDOUT_MONTHS,
            "minimum_train_months": MIN_OUTER_TRAIN_MONTHS,
            "label_availability_buffer_days": (
                LABEL_AVAILABILITY_BUFFER_DAYS
            ),
            "strict_dataset_contract": STRICT_DATASET_CONTRACT,
            "seed": base.SEED,
        },
        "development_split_settings": {
            "folds": INNER_FOLDS,
            "validation_months": INNER_VALIDATION_MONTHS,
            "minimum_train_months": MIN_INNER_TRAIN_MONTHS,
            "label_availability_buffer_days": (
                LABEL_AVAILABILITY_BUFFER_DAYS
            ),
            "seed": base.SEED,
        },
        "sample_train_max_rows": base.TRAIN_MAX_ROWS,
        "sample_validation_max_rows": base.VALIDATION_MAX_ROWS,
        "starting_structure": "intercept_only_audit_then_mandatory_control",
        "maximum_features": len(model_pool),
        "minimum_stable_folds": MIN_STABLE_FOLDS,
        "tolerances": {
            "minimum_addition_ap_skill_gain": AP_STEP_TOL,
            "full_row_ap_skill_noninferiority": AP_TOTAL_TOL,
            "ridge_c_ap_skill_noninferiority": AP_STEP_TOL,
            "lift10_step_loss": LIFT_STEP_TOL,
            "lift10_total_loss_vs_intercept": LIFT_TOTAL_TOL,
            "full_row_lift10_noninferiority": LIFT_TOTAL_TOL,
            "minimum_floating_removal_ap_skill_gain": (
                FLOATING_REMOVAL_AP_GAIN
            ),
        },
        "full_row_scope": "every accepted unique feature structure",
        "final_holdout_scored": False,
    }
    fingerprint = hashlib.sha256(
        json.dumps(base.jsonable(immutable), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        **immutable,
        "config_fingerprint": fingerprint,
        "invocation_runtime_budget_hours": RUNTIME_BUDGET_HOURS,
        "invocation_max_new_fits": MAX_NEW_FITS,
        "invocation_max_actions_safety_cap": MAX_ACTIONS,
    }








def validate_or_write_config(payload: dict[str, Any]) -> None:
    if CONFIG_FILE.is_file():
        prior = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        if prior.get("config_fingerprint") != payload["config_fingerprint"]:
            raise RuntimeError(
                "This RunTag contains checkpoints from different statistical "
                "settings or inputs. Restore the original settings or use a new RunTag."
            )
        # Refresh invocation-only controls while retaining the same immutable
        # statistical fingerprint.
        base.atomic_write_json(CONFIG_FILE, payload)
        return
    unrelated = [path for path in OUT.iterdir() if path.name != CONFIG_FILE.name]
    if unrelated:
        raise RuntimeError(
            f"Output directory is non-empty but has no compatible config: {OUT}"
        )
    base.atomic_write_json(CONFIG_FILE, payload)








def budgeted_plan(
    budget: RunBudget,
    store: base.FitCheckpoint,
    context: str,
    phase: str,
    label: str,
    caches: list[base.FoldMatrixCache],
    features: list[str],
    spec: dict[str, Any],
    require_convergence: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cache in caches:
        key = base.fit_key(
            context,
            cache.fold.fold_number,
            features,
            spec,
            False,
            cache.history_identity,
        )
        if store.completed(key) is None:
            budget.before_fit(phase)
        row = base.checkpointed_fit(
            store,
            context,
            phase,
            label,
            cache,
            features,
            spec,
            include_train_diagnostics=False,
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    if require_convergence and not frame.converged.astype(str).str.lower().isin(
        {"true", "1"}
    ).all():
        raise RuntimeError(f"At least one Ridge fit failed to converge for {label}.")
    return frame








def values_by_fold(frame: pd.DataFrame, column: str) -> pd.Series:
    out = frame.copy()
    out["fold"] = pd.to_numeric(out["fold"], errors="raise").astype(int)
    out[column] = pd.to_numeric(out[column], errors="raise")
    return out.set_index("fold")[column].sort_index()








def mean_metric(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="raise").mean())








def paired_delta(candidate: pd.DataFrame, reference: pd.DataFrame, column: str) -> pd.Series:
    left = values_by_fold(candidate, column)
    right = values_by_fold(reference, column)
    if not left.index.equals(right.index):
        raise RuntimeError(f"Fold mismatch for paired {column} comparison.")
    return left - right








def addition_gates(
    candidate: pd.DataFrame,
    current: pd.DataFrame,
    seed: pd.DataFrame,
) -> dict[str, Any]:
    ap_current = paired_delta(candidate, current, "churn_ap_skill")
    lift_current = paired_delta(candidate, current, "top10_lift")
    lift_seed = paired_delta(candidate, seed, "top10_lift")
    improved_ap = int((ap_current > 0).sum())
    stable_lift = int((lift_current >= -LIFT_STEP_TOL).sum())
    mean_ap_gain = float(ap_current.mean())
    eligible = (
        mean_ap_gain > 0
        and mean_ap_gain >= AP_STEP_TOL
        and float(lift_current.mean()) >= -LIFT_STEP_TOL
        and float(lift_seed.mean()) >= -LIFT_TOTAL_TOL
        and improved_ap >= MIN_STABLE_FOLDS
        and stable_lift >= MIN_STABLE_FOLDS
    )
    return {
        "mean_delta_ap_vs_current": mean_ap_gain,
        "worst_delta_ap_vs_current": float(ap_current.min()),
        "mean_delta_lift10_vs_current": float(lift_current.mean()),
        "mean_delta_lift10_vs_seed": float(lift_seed.mean()),
        "improved_ap_folds": improved_ap,
        "stable_lift_folds": stable_lift,
        "eligible": bool(eligible),
    }








def floating_removal_gates(
    candidate: pd.DataFrame,
    current: pd.DataFrame,
    seed: pd.DataFrame,
) -> dict[str, Any]:
    ap = paired_delta(candidate, current, "churn_ap_skill")
    lift = paired_delta(candidate, current, "top10_lift")
    lift_seed = paired_delta(candidate, seed, "top10_lift")
    improved_folds = int((ap > 0).sum())
    stable_lift = int((lift >= -LIFT_STEP_TOL).sum())
    mean_ap_gain = float(ap.mean())
    eligible = (
        mean_ap_gain > 0
        and mean_ap_gain >= FLOATING_REMOVAL_AP_GAIN
        and float(lift.mean()) >= -LIFT_STEP_TOL
        and float(lift_seed.mean()) >= -LIFT_TOTAL_TOL
        and improved_folds >= MIN_STABLE_FOLDS
        and stable_lift >= MIN_STABLE_FOLDS
    )
    return {
        "mean_delta_ap_vs_current": mean_ap_gain,
        "worst_delta_ap_vs_current": float(ap.min()),
        "mean_delta_lift10_vs_current": float(lift.mean()),
        "mean_delta_lift10_vs_seed": float(lift_seed.mean()),
        "improved_ap_folds": improved_folds,
        "stable_lift_folds": stable_lift,
        "eligible": bool(eligible),
    }








def summary_fields(results: pd.DataFrame) -> dict[str, Any]:
    summary = base.summarize_folds(results)
    return {
        **summary,
        "mean_churn_average_precision": mean_metric(
            results, "churn_average_precision"
        ),
        "min_churn_average_precision": float(
            pd.to_numeric(results.churn_average_precision, errors="raise").min()
        ),
        "mean_churn_ap_skill": mean_metric(results, "churn_ap_skill"),
        "min_churn_ap_skill": float(
            pd.to_numeric(results.churn_ap_skill, errors="raise").min()
        ),
        "mean_top10_lift": mean_metric(results, "top10_lift"),
        "mean_retention_roc_auc": mean_metric(results, "retention_roc_auc"),
    }




def write_logistic_diagnostics(
    *,
    df: pd.DataFrame,
    definitions: list[dict[str, Any]],
    feature_types: dict[str, str],
    preprocessing_spec: Any,
    project_config: Any,
    pool: list[str],
    selected_features: list[str],
    selected_history: dict[str, Any],
    final_c: float,
) -> dict[str, Any]:
    """Fit chronological OOF predictions with the mandatory year control."""


    non_control_features = [
        feature for feature in selected_features if feature != YEAR_CONTROL_FEATURE
    ]
    if YEAR_CONTROL_FEATURE not in selected_features or not non_control_features:
        raise RuntimeError(
            "Logistic diagnostics require calendar year plus at least one "
            "non-control feature."
        )
    policy_start_col = str(project_config.anchor_term_start_col)
    parts: list[pd.DataFrame] = []
    spec = base.ridge_spec(final_c)
    for position, definition in enumerate(definitions, 1):
        prepared = base.PreparedFold(
            int(definition.get("fold", position)),
            df.loc[np.asarray(definition["train_idx"])],
            df.loc[np.asarray(definition["validation_idx"])],
            feature_types,
            preprocessing_spec,
            target_col=project_config.target_col,
            cohort_col=project_config.cohort_date_col,
        )
        if policy_start_col not in prepared.validation:
            raise KeyError(
                f"Policy-start column {policy_start_col!r} is unavailable for "
                "calendar diagnostics."
            )
        if YEAR_CONTROL_FEATURE not in prepared.validation:
            prepared.validation[YEAR_CONTROL_FEATURE] = pd.to_datetime(
                prepared.validation[project_config.cohort_date_col], errors="raise"
            ).dt.year.astype(int)
        cache = base.FoldMatrixCache.build(
            prepared,
            pool,
            sample_weight=history_training_weights(
                prepared.train,
                project_config.cohort_date_col,
                selected_history,
            ),
            history_identity=history_fit_identity(selected_history),
        )
        x_train, x_validation = cache.matrices(selected_features)
        model = base.make_ridge_model(spec)
        fit_kwargs: dict[str, Any] = {}
        if cache.sample_weight is not None:
            fit_kwargs["sample_weight"] = cache.sample_weight
        model.fit(x_train, prepared.y_train, **fit_kwargs)
        class_index = {
            int(value): index for index, value in enumerate(model.classes_)
        }
        predicted_retention = model.predict_proba(x_validation)[:, class_index[1]]


        validation = prepared.validation
        diagnostic = pd.DataFrame(
            {
                "fold": int(prepared.fold_number),
                "row_id": [
                    f"{int(prepared.fold_number)}:{value}"
                    for value in validation.index
                ],
                ACTUAL_COLUMN: pd.to_numeric(
                    validation[project_config.target_col], errors="raise"
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
        for feature in selected_features:
            if feature in validation and feature not in diagnostic:
                diagnostic[feature] = validation[feature].to_numpy()
            elif feature == WRITTEN_PREM_BIN_FEATURE:
                block = prepared.get_feature_block(feature)
                diagnostic[feature] = apply_quantile_bin_artifact(
                    validation[str(block.fitted_artifact["source_feature"])],
                    block.fitted_artifact,
                ).to_numpy()
            elif feature == STATE_GROUP_FEATURE:
                # FeatureBlock.fitted_artifact is the export-safe audit
                # dictionary.  Applying the grouping to diagnostic rows needs
                # the live object that PreparedFold fitted and cached.
                prepared.get_feature_block(feature)
                fitted_state_grouper = getattr(
                    prepared, "_derived_feature_artifacts", {}
                ).get(feature)
                if fitted_state_grouper is None or not hasattr(
                    fitted_state_grouper, "spec"
                ):
                    raise RuntimeError(
                        "Missing live fitted state grouper for Logistic "
                        f"fold {prepared.fold_number}."
                    )
                grouped, _ = apply_state_grouper(
                    fitted_state_grouper,
                    validation,
                    scope=f"fold_{prepared.fold_number}_diagnostic",
                )
                diagnostic[feature] = grouped.to_numpy()
        parts.append(diagnostic)
        del cache, prepared
        gc.collect()
    return write_calendar_control_artifacts(
        predictions=pd.concat(parts, ignore_index=True),
        output_dir=OUT / "diagnostics",
        model_family="logistic",
        one_way_features=selected_features,
    )






def history_screen_summary_row(
    strategy: dict[str, Any],
    candidate_order: int,
    results: pd.DataFrame,
) -> dict[str, Any]:
    folds = pd.to_numeric(results["fold"], errors="raise").astype(int)
    newest_fold = int(folds.max())
    newest_rows = results.loc[folds.eq(newest_fold)]
    if len(newest_rows) != 1:
        raise RuntimeError(
            f"History {strategy['history_id']} does not have exactly one newest-fold row."
        )
    return {
        "candidate_order": int(candidate_order),
        "history_id": strategy["history_id"],
        "kind": strategy["kind"],
        "months": int(strategy["months"]),
        "max_months": (
            int(strategy["max_months"])
            if "max_months" in strategy
            else pd.NA
        ),
        "history_identity": history_fit_identity(strategy),
        "newest_fold": newest_fold,
        "newest_fold_churn_ap_skill": float(
            newest_rows.iloc[0].churn_ap_skill
        ),
        **summary_fields(results),
        **{
            f"mean_highest_divided_by_lowest_lift_{bins}bins": float(
                pd.to_numeric(
                    results[
                        f"highest_divided_by_lowest_lift_{bins}bins"
                    ],
                    errors="coerce",
                ).mean()
            )
            for bins in (3, 4, 5, 10, 20)
        },
    }






def choose_history(
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    required = {
        "candidate_order",
        "history_id",
        "valid_folds",
        "converged_folds",
        "mean_churn_ap_skill",
        "mean_top10_lift",
        "newest_fold_churn_ap_skill",
    }
    missing = required - set(summary.columns)
    if missing:
        raise KeyError(f"History summary is missing columns: {sorted(missing)}")
    decision = summary.copy()
    decision["eligible"] = (
        pd.to_numeric(decision["valid_folds"], errors="raise").eq(INNER_FOLDS)
        & pd.to_numeric(decision["converged_folds"], errors="raise").eq(
            INNER_FOLDS
        )
    )
    eligible = decision.loc[decision["eligible"]].copy()
    if eligible.empty:
        raise RuntimeError(
            "No Logistic history candidate converged on every development fold."
        )
    ranked = eligible.sort_values(
        [
            "mean_churn_ap_skill",
            "mean_top10_lift",
            "newest_fold_churn_ap_skill",
            "candidate_order",
            "history_id",
        ],
        ascending=[False, False, False, True, True],
        kind="stable",
    )
    winner_id = str(ranked.iloc[0].history_id)
    decision["selected"] = decision.history_id.astype(str).eq(winner_id)
    decision["selection_rank"] = pd.NA
    decision.loc[ranked.index, "selection_rank"] = np.arange(1, len(ranked) + 1)
    winner = decision.loc[decision.selected].iloc[0]
    return decision.sort_values("candidate_order", kind="stable"), winner








def path_row(
    action_index: int,
    action: str,
    feature: str,
    features: list[str],
    results: pd.DataFrame,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "action_index": action_index,
        "action": action,
        "changed_feature": feature,
        "raw_feature_count": len(features),
        "feature_hash": feature_set_hash(features),
        "features": ";".join(features),
        "reason": reason,
        **summary_fields(results),
    }








def candidate_sort(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        [
            "mean_churn_ap_skill",
            "worst_delta_ap_vs_current",
            "mean_top10_lift",
            "mean_retention_roc_auc",
            "candidate_feature",
        ],
        ascending=[False, False, False, False, True],
        kind="stable",
    )








def floating_removal_sort(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        [
            "mean_churn_ap_skill",
            "mean_delta_ap_vs_current",
            "worst_delta_ap_vs_current",
            "mean_top10_lift",
            "candidate_feature",
        ],
        ascending=[False, False, False, False, True],
        kind="stable",
    )








def fold_audit(prepared: list[Any], cohort_col: str, target_col: str) -> pd.DataFrame:
    rows = []
    for fold in prepared:
        train_month = pd.to_datetime(fold.train[cohort_col])
        valid_month = pd.to_datetime(fold.validation[cohort_col])
        rows.append(
            {
                "fold": int(fold.fold_number),
                "sample_train_rows": len(fold.train),
                "sample_validation_rows": len(fold.validation),
                "train_start": str(train_month.min().date()),
                "train_end": str(train_month.max().date()),
                "validation_start": str(valid_month.min().date()),
                "validation_end": str(valid_month.max().date()),
                "sample_train_churn_rate": float(
                    1.0 - fold.train[target_col].astype(int).mean()
                ),
                "sample_validation_churn_rate": float(
                    1.0 - fold.validation[target_col].astype(int).mean()
                ),
            }
        )
    return pd.DataFrame(rows)








def index_hash(index: pd.Index) -> str:
    values = pd.util.hash_pandas_object(pd.Index(index), index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()








def validate_or_write_sample_identity(
    prepared: list[Any],
    *,
    history: dict[str, Any] | None = None,
    allow_existing_checkpoint: bool = False,
) -> None:
    path = OUT / "sample_identity.json"
    payload = {
        "history": None if history is None else dict(history),
        "history_identity": (
            "UNWEIGHTED_DEFAULT"
            if history is None
            else history_fit_identity(history)
        ),
        "folds": [
            {
                "fold": int(fold.fold_number),
                "train_rows": len(fold.train),
                "validation_rows": len(fold.validation),
                "train_index_hash": index_hash(fold.train.index),
                "validation_index_hash": index_hash(fold.validation.index),
            }
            for fold in prepared
        ]
    }
    payload["identity_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("identity_hash") != payload["identity_hash"]:
            raise RuntimeError(
                "The sampled train/validation rows changed for this RunTag. "
                "Do not reuse its fit checkpoint; restore deterministic sampling "
                "or start a new RunTag."
            )
        return
    if CHECKPOINT_FILE.is_file() and not allow_existing_checkpoint:
        raise RuntimeError(
            "A fit checkpoint exists without sample_identity.json; refusing "
            "unsafe checkpoint reuse. Start a new RunTag."
        )
    base.atomic_write_json(path, payload)








def choose_full_row_structure(
    long_results: pd.DataFrame,
    summary: pd.DataFrame,
    protected: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    required = set(protected)
    summary = summary.copy()
    summary["satisfies_protected_features"] = summary.features.map(
        lambda value: required.issubset(
            {feature for feature in str(value).split(";") if feature}
        )
    )
    valid_summary = summary[summary.satisfies_protected_features].copy()
    if valid_summary.empty:
        raise RuntimeError("No accepted structure contains every protected feature.")
    best = valid_summary.sort_values(
        ["mean_churn_ap_skill", "mean_top10_lift"],
        ascending=[False, False],
        kind="stable",
    ).iloc[0]
    best_rows = long_results[long_results.plan_id == best.plan_id].copy()
    seed_candidates = summary[pd.to_numeric(summary.action_index) == 0]
    if len(seed_candidates) != 1:
        raise RuntimeError(
            "Full-row path must contain exactly one intercept-only seed plan."
        )
    seed_plan = seed_candidates.iloc[0]
    seed_rows = long_results[long_results.plan_id == seed_plan.plan_id].copy()
    decisions = []
    for _, candidate in summary.iterrows():
        rows = long_results[long_results.plan_id == candidate.plan_id].copy()
        ap_delta = paired_delta(rows, best_rows, "churn_ap_skill")
        lift_delta = paired_delta(rows, best_rows, "top10_lift")
        lift_seed_delta = paired_delta(rows, seed_rows, "top10_lift")
        paired_se = float(ap_delta.std(ddof=1) / math.sqrt(len(ap_delta)))
        se_loss = paired_se if np.isfinite(paired_se) else AP_TOTAL_TOL
        allowed_ap_loss = min(AP_TOTAL_TOL, max(0.0, se_loss))
        stable_ap = int((ap_delta >= -allowed_ap_loss).sum())
        stable_lift = int((lift_delta >= -LIFT_TOTAL_TOL).sum())
        eligible = (
            bool(candidate.satisfies_protected_features)
            and float(ap_delta.mean()) >= -allowed_ap_loss
            and float(lift_delta.mean()) >= -LIFT_TOTAL_TOL
            and float(lift_seed_delta.mean()) >= -LIFT_TOTAL_TOL
            and stable_ap >= MIN_STABLE_FOLDS
            and stable_lift >= MIN_STABLE_FOLDS
            and int((lift_seed_delta >= -LIFT_TOTAL_TOL).sum()) >= MIN_STABLE_FOLDS
        )
        decisions.append(
            {
                **candidate.to_dict(),
                "ap_best_plan": str(best.plan_id),
                "lift_seed_plan": str(seed_plan.plan_id),
                "mean_delta_ap_vs_best": float(ap_delta.mean()),
                "paired_ap_standard_error": paired_se,
                "allowed_ap_loss": allowed_ap_loss,
                "mean_delta_lift10_vs_best": float(lift_delta.mean()),
                "mean_delta_lift10_vs_seed": float(lift_seed_delta.mean()),
                "stable_ap_folds_vs_best": stable_ap,
                "stable_lift_folds_vs_best": stable_lift,
                "eligible_noninferior": bool(eligible),
            }
        )
    decision = pd.DataFrame(decisions)
    eligible = decision[decision.eligible_noninferior].copy()
    if eligible.empty:
        recommended = decision[decision.plan_id == best.plan_id].iloc[0]
    else:
        recommended = eligible.sort_values(
            [
                "raw_feature_count",
                "mean_churn_ap_skill",
                "mean_top10_lift",
            ],
            ascending=[True, False, False],
            kind="stable",
        ).iloc[0]
    decision["ap_best"] = decision.plan_id.astype(str) == str(best.plan_id)
    decision["recommended_structure"] = (
        decision.plan_id.astype(str) == str(recommended.plan_id)
    )
    decision = decision.sort_values(
        ["recommended_structure", "eligible_noninferior", "raw_feature_count"],
        ascending=[False, False, True],
        kind="stable",
    )
    return decision, recommended








def choose_c(
    tuning_long: pd.DataFrame, fixed_candidate_id: str
) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    for candidate_id, part in tuning_long.groupby("candidate_id", sort=False):
        rows.append(
            {
                "candidate_id": candidate_id,
                "C": float(pd.to_numeric(part.C, errors="raise").iloc[0]),
                **summary_fields(part),
            }
        )
    summary = pd.DataFrame(rows)
    summary = summary[
        (
            pd.to_numeric(summary.valid_folds, errors="coerce")
            == INNER_FOLDS
        )
        & (
            pd.to_numeric(summary.converged_folds, errors="coerce")
            == INNER_FOLDS
        )
    ].copy()
    if summary.empty:
        raise RuntimeError(
            "No Ridge C-grid setting converged on all development folds."
        )
    if fixed_candidate_id not in set(summary.candidate_id.astype(str)):
        raise RuntimeError("The fixed-C tuning reference did not converge on all folds.")
    tuning_long = tuning_long[
        tuning_long.candidate_id.astype(str).isin(summary.candidate_id.astype(str))
    ].copy()
    best = summary.sort_values(
        ["mean_churn_ap_skill", "mean_top10_lift"],
        ascending=[False, False],
        kind="stable",
    ).iloc[0]
    best_rows = tuning_long[tuning_long.candidate_id == best.candidate_id]
    fixed_rows = tuning_long[
        tuning_long.candidate_id.astype(str) == str(fixed_candidate_id)
    ]
    decisions = []
    for _, candidate in summary.iterrows():
        part = tuning_long[tuning_long.candidate_id == candidate.candidate_id]
        ap_delta = paired_delta(part, best_rows, "churn_ap_skill")
        lift_delta = paired_delta(part, best_rows, "top10_lift")
        lift_fixed_delta = paired_delta(part, fixed_rows, "top10_lift")
        paired_se = float(ap_delta.std(ddof=1) / math.sqrt(len(ap_delta)))
        se_loss = paired_se if np.isfinite(paired_se) else AP_STEP_TOL
        allowed = min(AP_STEP_TOL, max(0.0, se_loss))
        mean_ap_pass = float(ap_delta.mean()) >= -allowed
        mean_lift_best_pass = float(lift_delta.mean()) >= -LIFT_STEP_TOL
        mean_lift_fixed_pass = (
            float(lift_fixed_delta.mean()) >= -LIFT_STEP_TOL
        )
        stable_ap_folds = int((ap_delta >= -allowed).sum())
        stable_lift_best_folds = int(
            (lift_delta >= -LIFT_STEP_TOL).sum()
        )
        stable_lift_fixed_folds = int(
            (lift_fixed_delta >= -LIFT_STEP_TOL).sum()
        )
        stable_ap_pass = stable_ap_folds >= MIN_STABLE_FOLDS
        stable_lift_fixed_pass = (
            stable_lift_fixed_folds >= MIN_STABLE_FOLDS
        )
        eligible = (
            mean_ap_pass
            and mean_lift_best_pass
            and mean_lift_fixed_pass
            and stable_ap_pass
            and stable_lift_fixed_pass
        )
        failure_reasons = []
        if not mean_ap_pass:
            failure_reasons.append("MEAN_AP_VS_BEST")
        if not mean_lift_best_pass:
            failure_reasons.append("MEAN_LIFT10_VS_BEST")
        if not mean_lift_fixed_pass:
            failure_reasons.append("MEAN_LIFT10_VS_FIXED_C")
        if not stable_ap_pass:
            failure_reasons.append("STABLE_AP_FOLDS_VS_BEST")
        if not stable_lift_fixed_pass:
            failure_reasons.append("STABLE_LIFT10_FOLDS_VS_FIXED_C")
        decisions.append(
            {
                **candidate.to_dict(),
                "mean_delta_ap_vs_best": float(ap_delta.mean()),
                "paired_ap_standard_error": paired_se,
                "mean_delta_lift10_vs_best": float(lift_delta.mean()),
                "mean_delta_lift10_vs_fixed_c": float(lift_fixed_delta.mean()),
                "allowed_ap_loss": allowed,
                "mean_ap_noninferior_vs_best": bool(mean_ap_pass),
                "mean_lift10_noninferior_vs_best": bool(mean_lift_best_pass),
                "mean_lift10_noninferior_vs_fixed_c": bool(
                    mean_lift_fixed_pass
                ),
                "stable_ap_folds_vs_best": stable_ap_folds,
                "stable_lift10_folds_vs_best": stable_lift_best_folds,
                "stable_lift10_folds_vs_fixed_c": stable_lift_fixed_folds,
                "stable_ap_requirement_passed": bool(stable_ap_pass),
                "stable_lift10_fixed_requirement_passed": bool(
                    stable_lift_fixed_pass
                ),
                "eligibility_failure_reasons": ";".join(failure_reasons),
                "is_ap_best_candidate": (
                    str(candidate.candidate_id) == str(best.candidate_id)
                ),
                "is_fixed_c_reference": (
                    str(candidate.candidate_id) == str(fixed_candidate_id)
                ),
                "eligible_one_se": bool(eligible),
            }
        )
    decision = pd.DataFrame(decisions)
    eligible = decision[decision.eligible_one_se].copy()
    eligible_count = int(len(eligible))
    fallback_used = eligible.empty
    if fallback_used:
        fixed = decision[
            decision.candidate_id.astype(str) == str(fixed_candidate_id)
        ].copy()
        if len(fixed) != 1:
            raise RuntimeError(
                "Ridge tuning must contain exactly one fixed-C reference row."
            )
        winner_id = str(fixed.iloc[0].candidate_id)
        fallback_reason = "NO_CANDIDATE_PASSED_JOINT_AP_LIFT_GUARDRAILS"
        selection_basis = "PRESPECIFIED_FIXED_C_FALLBACK"
    else:
        winner_id = str(
            eligible.sort_values(
                ["C", "mean_churn_ap_skill"],
                ascending=[True, False],
                kind="stable",
            ).iloc[0].candidate_id
        )
        fallback_reason = ""
        selection_basis = "SMALLEST_C_AMONG_JOINTLY_ELIGIBLE"
    decision["joint_eligible_candidate_count"] = eligible_count
    decision["fallback_used"] = bool(fallback_used)
    decision["fallback_reason"] = fallback_reason
    decision["selection_basis"] = selection_basis
    decision["selected_c"] = (
        decision.candidate_id.astype(str) == winner_id
    )
    winner = decision.loc[decision.selected_c].iloc[0]
    return decision.sort_values("C"), winner








def run() -> None:
    candidate_pool = base.read_pool(base.POOL)
    family_feature_types = read_family_feature_contract(candidate_pool)
    if len(candidate_pool) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"Expected the audited {EXPECTED_FEATURE_COUNT}-feature pool, found "
            f"{len(candidate_pool)}. Do not run on B42/B43 or another inherited subset."
        )
    collision = set(MANDATORY_CONTROL_FEATURES).intersection(candidate_pool)
    if collision:
        raise RuntimeError(
            "The engine-owned mandatory control collided with the preparation "
            f"candidate pool: {sorted(collision)}."
        )
    pool = [*MANDATORY_CONTROL_FEATURES, *candidate_pool]
    # The complete starting pool is defined by the full ROC config followed by
    # the training-only quality screen. Do not hard-require historical
    # B42/B43 fields after that screen legitimately removes one.
    protected = read_protected(pool)
    config = config_payload(candidate_pool, pool, protected)
    validate_or_write_config(config)
    base.atomic_write_json(
        OUT / "metric_contract.json",
        {
            "primary_statistical_metric": "churn_ap_skill",
            "co_primary_business_metric": "churn_recall_10pct",
            "definition": (
                "prevalence-normalized average precision for churn; not "
                "trapezoidal PR-AUC"
            ),
            "operational_gate_implementation": "top10_lift",
            "fixed_capacity_equivalence": (
                "At fixed 10% capacity, recall/capture and lift have the same "
                "candidate ordering."
            ),
            "reported_business_translation": [
                "top10_lift",
                "churn_precision_10pct",
            ],
            "ranking_guardrail": "retention_roc_auc",
            "calibration_guardrails": [
                "log_loss",
                "brier_score",
                "calibration_gap",
            ],
            "fold_comparisons": "paired candidate-minus-reference",
            "structure_search": "grouped_sequential_floating_forward",
            "starting_structure": "intercept_only_audit_then_mandatory_control",
            "addition_rule": (
                "minimum positive mean AP gain with stable positive AP folds "
                "and Lift@10 step/total guardrails"
            ),
            "floating_removal_rule": (
                "minimum positive mean AP gain from removal with stable positive "
                "AP folds and Lift@10 step/total guardrails"
            ),
            "history_screen": {
                "candidates": [dict(item) for item in HISTORY_GRID],
                "features": "full_quality_screened_pool_plus_mandatory_control",
                "selection_order": [
                    "mean_churn_ap_skill",
                    "mean_top10_lift",
                    "newest_fold_churn_ap_skill",
                ],
                "lift_spread": (
                    "reported for 3/4/5/10/20 bins but excluded from selection"
                ),
            },
            "model_during_structure_search": (
                "Ridge logistic regression under the selected history policy"
            ),
            "fixed_c": FIXED_C,
        },
    )




    budget = RunBudget()
    fixed_spec = base.ridge_spec(FIXED_C)




    print(
        f"LOGISTIC REGRESSION: intercept-only start; {len(pool)} candidate "
        f"feature groups; raw churn AP primary; Lift@10 guardrail; "
        f"fixed Ridge C={FIXED_C:g}",
        flush=True,
    )
    raw = base.read_table(base.DATA)
    manifest = pd.read_csv(base.MANIFEST, dtype=str, keep_default_na=False)
    full_df, outer_definitions, final_test_idx, _, project_config = (
        base.load_project_population_and_folds(
            raw,
            rolling_validation_folds=OUTER_FOLDS,
            validation_months=OUTER_VALIDATION_MONTHS,
            test_months=HOLDOUT_MONTHS,
            min_train_months=MIN_OUTER_TRAIN_MONTHS,
            strict_dataset_contract=STRICT_DATASET_CONTRACT,
        )
    )
    del raw
    gc.collect()
    if len(outer_definitions) != OUTER_FOLDS:
        raise RuntimeError(
            f"Logistic Regression requires exactly {OUTER_FOLDS} outer folds."
        )
    if (
        project_config.label_availability_buffer_days
        != LABEL_AVAILABILITY_BUFFER_DAYS
    ):
        raise RuntimeError(
            "params.py and the Logistic split environment disagree on the "
            "label-availability buffer."
        )
    for definition in outer_definitions:
        if np.intersect1d(final_test_idx, definition["train_idx"]).size:
            raise RuntimeError("Final holdout entered a development training fold.")
        if np.intersect1d(final_test_idx, definition["validation_idx"]).size:
            raise RuntimeError("Final holdout entered a development validation fold.")




    outer_fold_frame = base.fold_definitions_frame(
        full_df,
        outer_definitions,
        cohort_col=project_config.cohort_date_col,
        target_check_col=project_config.target_check_col,
        target_col=project_config.target_col,
    )
    verify_fold_definition_contract(
        outer_fold_frame,
        OUTER_FOLD_DEFINITIONS,
        label="Logistic outer discovery boundary",
    )




    df = base.oldest_complete_train(
        full_df, outer_definitions[0], project_config.cohort_date_col
    )
    definitions, inner_holdout = base.build_expanding_folds(
        df,
        cohort_col=project_config.cohort_date_col,
        target_check_col=project_config.target_check_col,
        target_col=project_config.target_col,
        validation_folds=INNER_FOLDS,
        validation_months=INNER_VALIDATION_MONTHS,
        min_train_months=MIN_INNER_TRAIN_MONTHS,
        holdout_months=0,
        label_buffer_days=LABEL_AVAILABILITY_BUFFER_DAYS,
    )
    if len(inner_holdout):
        raise RuntimeError("Logistic inner development unexpectedly created a holdout.")
    if len(definitions) != INNER_FOLDS:
        raise RuntimeError(
            f"Logistic Regression requires exactly {INNER_FOLDS} inner folds."
        )




    _, eligible_union, union_feature_types = base.build_candidate_audit(
        manifest, full_df.columns
    )
    eligible, feature_types = route_candidate_list_and_types(
        eligible_union,
        union_feature_types,
        "logistic",
    )
    validate_one_variant_per_concept(candidate_pool, "logistic")
    invalid = [feature for feature in candidate_pool if feature not in eligible]
    if invalid:
        raise RuntimeError("Non-eligible features entered the pool: " + ", ".join(invalid))
    type_mismatches = {
        feature: (feature_types.get(feature), family_feature_types[feature])
        for feature in candidate_pool
        if feature_types.get(feature) != family_feature_types[feature]
    }
    if type_mismatches:
        raise RuntimeError(
            "Logistic routed types differ from the preparation contract: "
            f"{type_mismatches}"
        )
    feature_types = {
        **family_feature_types,
        COHORT_CALENDAR_YEAR_FEATURE: COHORT_CALENDAR_YEAR_TYPE,
    }
    base.atomic_write_json(
        OUT / "resolved_logistic_model_feature_types.json",
        {feature: feature_types[feature] for feature in pool},
    )




    development_fold_frame = base.fold_definitions_frame(
        df,
        definitions,
        cohort_col=project_config.cohort_date_col,
        target_check_col=project_config.target_check_col,
        target_col=project_config.target_col,
    )
    verify_fold_definition_contract(
        development_fold_frame,
        INNER_FOLD_DEFINITIONS,
        label="Logistic inner development folds",
    )
    development_fold_frame.to_csv(
        OUT / "inner_fold_definitions.csv", index=False
    )
    holdout_dates = pd.to_datetime(
        full_df.loc[np.asarray(final_test_idx), project_config.cohort_date_col]
    )
    business_trend_monthly = business_retention_monthly_summary(
        population=full_df,
        target_col=project_config.target_col,
        cohort_col=project_config.cohort_date_col,
        end_month_exclusive=holdout_dates.min(),
    )
    base.atomic_write_json(
        OUT / "holdout_seal.json",
        {
            "row_count": int(len(final_test_idx)),
            "cohort_start": str(holdout_dates.min().date()),
            "cohort_end": str(holdout_dates.max().date()),
            "predictions_created": False,
            "target_metrics_created": False,
        },
    )
    del full_df
    gc.collect()




    preprocessing_spec = base.ModelSpec(
        c_value=1.0,
        max_iter=base.MAX_ITER_RIDGE,
        solver="lbfgs",
        random_seed=base.SEED,
        rare_min_frequency=0.001,
        fail_on_nonconvergence=False,
    )
    validate_or_write_history_population_identity(
        history_population_identity(
            df,
            definitions,
            project_config.cohort_date_col,
        )
    )
    # Open SQLite only after establishing the complete, prespecified history-
    # population identity. Every fit key additionally carries its exact
    # history identity, so the three mature-month windows cannot collide.
    checkpoint = base.FitCheckpoint(CHECKPOINT_FILE)


    print(
        f"HISTORY SCREEN: {len(HISTORY_GRID)} candidates x {INNER_FOLDS} folds; "
        f"full {len(pool)}-feature pool; fixed Ridge C={FIXED_C:g}",
        flush=True,
    )
    history_parts: list[pd.DataFrame] = []
    history_summary_rows: list[dict[str, Any]] = []
    history_spread_rows: list[dict[str, Any]] = []
    for candidate_order, strategy in enumerate(HISTORY_GRID, start=1):
        _, screen_caches = prepare_history_caches(
            df,
            definitions,
            feature_types,
            preprocessing_spec,
            project_config,
            pool,
            strategy,
            sampled=True,
        )
        part = budgeted_plan(
            budget,
            checkpoint,
            "history_screen",
            "history_screen",
            str(strategy["history_id"]),
            screen_caches,
            pool,
            fixed_spec,
            require_convergence=False,
        ).copy()
        part["candidate_order"] = candidate_order
        part["history_id"] = strategy["history_id"]
        part["history_kind"] = strategy["kind"]
        part["history_months"] = int(strategy["months"])
        part["history_max_months"] = strategy.get("max_months", pd.NA)
        history_parts.append(part)
        history_summary_rows.append(
            history_screen_summary_row(strategy, candidate_order, part)
        )
        for _, fit_row in part.iterrows():
            for bins in (3, 4, 5, 10, 20):
                history_spread_rows.append(
                    {
                        "candidate_order": candidate_order,
                        "history_id": strategy["history_id"],
                        "kind": strategy["kind"],
                        "months": int(strategy["months"]),
                        "max_months": strategy.get("max_months", pd.NA),
                        "fold": int(fit_row["fold"]),
                        "n_bins": bins,
                        "highest_lift": fit_row[f"highest_lift_{bins}bins"],
                        "highest_lift_group": fit_row[
                            f"highest_lift_group_{bins}bins"
                        ],
                        "lowest_lift": fit_row[f"lowest_lift_{bins}bins"],
                        "lowest_lift_group": fit_row[
                            f"lowest_lift_group_{bins}bins"
                        ],
                        "highest_divided_by_lowest_lift": fit_row[
                            f"highest_divided_by_lowest_lift_{bins}bins"
                        ],
                        "ratio_defined": fit_row[
                            f"lift_spread_ratio_defined_{bins}bins"
                        ],
                        "ratio_status": fit_row[
                            f"lift_spread_ratio_status_{bins}bins"
                        ],
                    }
                )
        del screen_caches
        gc.collect()


    history_long = pd.concat(history_parts, ignore_index=True)
    history_summary = pd.DataFrame(history_summary_rows)
    history_decision, selected_history_row = choose_history(history_summary)
    selected_history_id = str(selected_history_row.history_id)
    selected_history = next(
        dict(strategy)
        for strategy in HISTORY_GRID
        if strategy["history_id"] == selected_history_id
    )
    base.atomic_write_csv(OUT / "history_screen_results_long.csv", history_long)
    base.atomic_write_csv(OUT / "history_screen_summary.csv", history_decision)
    base.atomic_write_csv(
        OUT / "history_screen_lift_spread.csv",
        pd.DataFrame(history_spread_rows),
    )
    base.atomic_write_json(
        OUT / "history_selection_decision.json",
        {
            "selected_history": selected_history,
            "history_selection_policy": dict(HISTORY_SELECTION_POLICY),
            "screening_c": FIXED_C,
            "screening_feature_count": len(pool),
            "selection_order": [
                "mean_churn_ap_skill",
                "mean_top10_lift",
                "newest_fold_churn_ap_skill",
                "prespecified_candidate_order",
            ],
            "lift_spread_reported_but_not_selected_on": True,
            "sealed_holdout_rows_scored": 0,
        },
    )
    print(f"SELECTED HISTORY: {selected_history_id}", flush=True)


    selected_definitions, caches = prepare_history_caches(
        df,
        definitions,
        feature_types,
        preprocessing_spec,
        project_config,
        pool,
        selected_history,
        sampled=True,
    )
    definitions = selected_definitions
    validate_or_write_sample_identity(
        [cache.fold for cache in caches],
        history=selected_history,
        allow_existing_checkpoint=True,
    )
    fold_audit(
        [cache.fold for cache in caches],
        project_config.cohort_date_col,
        project_config.target_col,
    ).to_csv(OUT / "sample_audit.csv", index=False)
    base.atomic_write_csv(
        OUT / "selected_history_fold_definitions.csv",
        base.fold_definitions_frame(
            df,
            definitions,
            cohort_col=project_config.cohort_date_col,
            target_check_col=project_config.target_check_col,
            target_col=project_config.target_col,
        ),
    )
    base.export_fold_derived_artifacts(
        [cache.fold for cache in caches],
        OUT / "derived",
        # Keep the nested artifact path below classic Windows MAX_PATH. The
        # full stage meaning is already captured by this call site and the
        # exported fold/feature metadata; only the directory label is compact.
        stage="inner",
    )
    del history_parts
    gc.collect()




    seed_results = budgeted_plan(
        budget,
        checkpoint,
        "sampled_selected_history",
        "seed",
        "INTERCEPT_ONLY",
        caches,
        [],
        fixed_spec,
    )
    current_features: list[str] = []
    current_results = seed_results
    best_seen_features: list[str] = []
    best_seen_results = seed_results
    visited = {feature_set_hash([])}
    path_rows = [path_row(0, "SEED", "", [], seed_results)]
    candidate_rows: list[dict[str, Any]] = []
    action = 0




    # Protected groups are non-statistical constraints. Preserve the genuine
    # intercept-only seed for comparison, then force these groups into the
    # working structure in pool order before the greedy search begins.
    for feature in protected:
        if action >= MAX_ACTIONS:
            raise RuntimeError(
                "RETENTION_LR_MAX_ACTIONS reached before a natural statistical stop. "
                "Increase only this safety cap and rerun the same RunTag."
            )
        trial = ordered(pool, set(current_features) | {feature})
        results = budgeted_plan(
            budget,
            checkpoint,
            "sampled_selected_history",
            f"forced_add_action_{action + 1}",
            f"FORCED_ADD_{feature}",
            caches,
            trial,
            fixed_spec,
        )
        current_features = trial
        current_results = results
        action += 1
        visited.add(feature_set_hash(current_features))
        path_rows.append(
            path_row(
                action,
                "FORCED_ADDED",
                feature,
                current_features,
                current_results,
                "Protected feature constraint.",
            )
        )
        print(
            f"FORCED ADD {feature}; now {len(current_features)} features",
            flush=True,
        )
    if protected:
        best_seen_features = current_features.copy()
        best_seen_results = current_results




    while len(current_features) < len(pool):
        if action >= MAX_ACTIONS:
            raise RuntimeError(
                "RETENTION_LR_MAX_ACTIONS reached before a natural statistical stop. "
                "Increase only this safety cap and rerun the same RunTag."
            )
        addition_records = []
        addition_results: dict[str, pd.DataFrame] = {}
        excluded = [feature for feature in pool if feature not in current_features]
        for feature in excluded:
            trial = ordered(pool, set(current_features) | {feature})
            trial_hash = feature_set_hash(trial)
            if trial_hash in visited:
                continue
            results = budgeted_plan(
                budget,
                checkpoint,
                "sampled_selected_history",
                f"add_action_{action + 1}",
                f"ADD_{feature}",
                caches,
                trial,
                fixed_spec,
            )
            gates = addition_gates(results, current_results, seed_results)
            record = {
                "action_index": action + 1,
                "candidate_action": "ADD",
                "candidate_feature": feature,
                "candidate_raw_feature_count": len(trial),
                "feature_hash": trial_hash,
                "features": ";".join(trial),
                "accepted": False,
                **summary_fields(results),
                **gates,
            }
            addition_records.append(record)
            candidate_rows.append(record)
            addition_results[feature] = results
            print(
                f"add {feature}: AP skill={record['mean_churn_ap_skill']:.6f}; "
                f"Lift10={record['mean_top10_lift']:.4f}; eligible={record['eligible']}",
                flush=True,
            )
        table = pd.DataFrame(addition_records)
        eligible_additions = table[table.eligible].copy() if not table.empty else table
        if eligible_additions.empty:
            path_rows.append(
                path_row(
                    action + 1,
                    "STOP",
                    "",
                    current_features,
                    current_results,
                    "No remaining grouped addition achieved the required paired "
                    "raw-AP gain and Lift@10 guardrails.",
                )
            )
            break




        chosen = candidate_sort(eligible_additions).iloc[0]
        feature = str(chosen.candidate_feature)
        next(
            record
            for record in addition_records
            if record["candidate_feature"] == feature
        )["accepted"] = True
        current_features = ordered(pool, set(current_features) | {feature})
        current_results = addition_results[feature]
        action += 1
        visited.add(feature_set_hash(current_features))
        path_rows.append(
            path_row(action, "ADDED", feature, current_features, current_results)
        )
        if mean_metric(current_results, "churn_ap_skill") > mean_metric(
            best_seen_results, "churn_ap_skill"
        ):
            best_seen_features = current_features.copy()
            best_seen_results = current_results
        print(
            f"ACCEPTED ADD {feature}; now {len(current_features)} features",
            flush=True,
        )




        # Sequential floating step: after every accepted forward addition,
        # repeatedly consider removing any non-protected included group. A
        # removal is accepted only when the reduced structure materially
        # improves the current structure; this is not a permissive pruning pass.
        while action < MAX_ACTIONS:
            removal_records = []
            removal_results: dict[str, pd.DataFrame] = {}
            for feature in current_features:
                if feature in protected:
                    continue
                trial = [value for value in current_features if value != feature]
                trial_hash = feature_set_hash(trial)
                if trial_hash in visited:
                    continue
                results = budgeted_plan(
                    budget,
                    checkpoint,
                    "sampled_selected_history",
                    f"floating_remove_action_{action + 1}",
                    f"FLOATING_REMOVE_{feature}",
                    caches,
                    trial,
                    fixed_spec,
                )
                gates = floating_removal_gates(
                    results, current_results, seed_results
                )
                record = {
                    "action_index": action + 1,
                    "candidate_action": "FLOATING_REMOVE",
                    "candidate_feature": feature,
                    "candidate_raw_feature_count": len(trial),
                    "feature_hash": trial_hash,
                    "features": ";".join(trial),
                    "accepted": False,
                    **summary_fields(results),
                    **gates,
                }
                removal_records.append(record)
                candidate_rows.append(record)
                removal_results[feature] = results
                print(
                    f"floating-remove {feature}: "
                    f"AP skill={record['mean_churn_ap_skill']:.6f}; "
                    f"Lift10={record['mean_top10_lift']:.4f}; "
                    f"eligible={record['eligible']}",
                    flush=True,
                )
            removal_table = pd.DataFrame(removal_records)
            eligible_removals = (
                removal_table[removal_table.eligible].copy()
                if not removal_table.empty
                else removal_table
            )
            if eligible_removals.empty:
                break
            chosen_removal = floating_removal_sort(eligible_removals).iloc[0]
            feature = str(chosen_removal.candidate_feature)
            next(
                record
                for record in removal_records
                if record["candidate_feature"] == feature
            )["accepted"] = True
            current_features = [
                value for value in current_features if value != feature
            ]
            current_results = removal_results[feature]
            action += 1
            visited.add(feature_set_hash(current_features))
            if mean_metric(current_results, "churn_ap_skill") > mean_metric(
                best_seen_results, "churn_ap_skill"
            ):
                best_seen_features = current_features.copy()
                best_seen_results = current_results
            path_rows.append(
                path_row(
                    action,
                    "FLOATING_REMOVED",
                    feature,
                    current_features,
                    current_results,
                )
            )
            print(
                f"ACCEPTED FLOATING REMOVE {feature}; now "
                f"{len(current_features)} features",
                flush=True,
            )




    else:
        path_rows.append(
            path_row(
                action + 1,
                "STOP",
                "",
                current_features,
                current_results,
                "Every candidate raw-feature group is in the current structure.",
            )
        )




    # Candidate evaluations are an audit artifact, not resume state.  Keep them
    # in memory while searching and publish the complete CSV once; interrupted
    # runs reconstruct these rows cheaply from the durable per-fit checkpoint.
    base.atomic_write_csv(
        OUT / "candidate_evaluations.csv", pd.DataFrame(candidate_rows)
    )
    path = pd.DataFrame(path_rows)
    base.atomic_write_csv(OUT / "accepted_floating_path.csv", path)
    accepted = path[
        path.action.isin(["SEED", "FORCED_ADDED", "ADDED", "FLOATING_REMOVED"])
    ].copy()
    accepted = accepted.drop_duplicates("feature_hash", keep="last")
    plan_registry = []
    for _, row in accepted.iterrows():
        features = [value for value in str(row.features).split(";") if value]
        plan_registry.append(
            {
                "plan_id": (
                    f"PATH_A{int(row.action_index):03d}_N{len(features):03d}_"
                    f"{str(row.feature_hash)[:8]}"
                ),
                "action_index": int(row.action_index),
                "raw_feature_count": len(features),
                "feature_hash": str(row.feature_hash),
                "features": features,
            }
        )
    base.atomic_write_json(
        OUT / "accepted_path_structures.json", {"plans": plan_registry}
    )
    base.atomic_write_text(
        OUT / "sampled_ap_best_features.txt",
        "\n".join(best_seen_features) + "\n",
    )




    del caches
    gc.collect()
    full_rows: list[dict[str, Any]] = []
    full_output = OUT / "full_row_path_results_long.csv"
    print(
        f"FULL-ROW DEVELOPMENT: {len(plan_registry)} accepted structures x {INNER_FOLDS} folds",
        flush=True,
    )
    for position, definition in enumerate(definitions, 1):
        prepared_full = base.PreparedFold(
            int(definition.get("fold", position)),
            df.loc[np.asarray(definition["train_idx"])],
            df.loc[np.asarray(definition["validation_idx"])],
            feature_types,
            preprocessing_spec,
            target_col=project_config.target_col,
            cohort_col=project_config.cohort_date_col,
        )
        cache = base.FoldMatrixCache.build(
            prepared_full,
            pool,
            sample_weight=history_training_weights(
                prepared_full.train,
                project_config.cohort_date_col,
                selected_history,
            ),
            history_identity=history_fit_identity(selected_history),
        )
        for plan in plan_registry:
            results = budgeted_plan(
                budget,
                checkpoint,
                "full_row_path",
                "full_row_path",
                plan["plan_id"],
                [cache],
                plan["features"],
                fixed_spec,
            )
            row = results.iloc[0].to_dict()
            row.update(
                plan_id=plan["plan_id"],
                action_index=plan["action_index"],
                raw_feature_count=plan["raw_feature_count"],
                features=";".join(plan["features"]),
            )
            full_rows.append(row)
        del cache, prepared_full
        gc.collect()




    full_long = pd.DataFrame(full_rows)
    # As above, this long-form audit is derivable from the fit checkpoint.  A
    # single atomic publication avoids repeatedly serializing all prior rows.
    base.atomic_write_csv(full_output, full_long)
    summary_rows = []
    for plan_id, part in full_long.groupby("plan_id", sort=False):
        first = part.iloc[0]
        row = {
            "plan_id": plan_id,
            "action_index": int(first.action_index),
            "raw_feature_count": int(first.raw_feature_count),
            "features": str(first.features),
            "C": FIXED_C,
            **summary_fields(part),
            "mean_predicted_minus_actual_retention": mean_metric(
                part, "predicted_minus_actual_retention"
            ),
        }
        summary_rows.append(row)
    full_summary = pd.DataFrame(summary_rows)
    base.atomic_write_csv(OUT / "full_row_path_summary.csv", full_summary)
    structure_decision, recommended = choose_full_row_structure(
        full_long, full_summary, protected
    )
    base.atomic_write_csv(
        OUT / "structure_selection_decision.csv", structure_decision
    )
    recommended_features = [
        value for value in str(recommended.features).split(";") if value
    ]
    base.atomic_write_text(
        OUT / "recommended_features.txt", "\n".join(recommended_features) + "\n"
    )




    prepared_tune = base.prepare_fold_objects(
        df,
        definitions,
        feature_types,
        preprocessing_spec,
        train_max_rows=base.TRAIN_MAX_ROWS,
        validation_max_rows=base.VALIDATION_MAX_ROWS,
        cohort_col=project_config.cohort_date_col,
        target_col=project_config.target_col,
    )
    validate_or_write_sample_identity(
        prepared_tune,
        history=selected_history,
        allow_existing_checkpoint=True,
    )
    tune_caches = [
        base.FoldMatrixCache.build(
            fold,
            pool,
            sample_weight=history_training_weights(
                fold.train,
                project_config.cohort_date_col,
                selected_history,
            ),
            history_identity=history_fit_identity(selected_history),
        )
        for fold in prepared_tune
    ]
    del prepared_tune
    tuning_parts = []
    for c_value in C_GRID:
        spec = base.ridge_spec(c_value)
        part = budgeted_plan(
            budget,
            checkpoint,
            "sampled_selected_history_tuning",
            "ridge_tuning",
            f"RECOMMENDED_C{c_value:g}",
            tune_caches,
            recommended_features,
            spec,
            require_convergence=False,
        )
        tuning_parts.append(part)
    tuning_long = pd.concat(tuning_parts, ignore_index=True)
    base.atomic_write_csv(OUT / "ridge_tuning_results_long.csv", tuning_long)
    tuning_decision, selected_c = choose_c(tuning_long, fixed_spec["id"])
    base.atomic_write_csv(OUT / "ridge_tuning_decision.csv", tuning_decision)
    selected_spec = base.ridge_spec(float(selected_c.C))
    del tune_caches
    gc.collect()




    selected_fixed_rows = full_long[
        full_long.plan_id.astype(str) == str(recommended.plan_id)
    ].copy()
    if math.isclose(float(selected_c.C), FIXED_C, rel_tol=0, abs_tol=1e-12):
        tuned_full = selected_fixed_rows.copy()
    else:
        tuned_rows = []
        for position, definition in enumerate(definitions, 1):
            prepared_full = base.PreparedFold(
                int(definition.get("fold", position)),
                df.loc[np.asarray(definition["train_idx"])],
                df.loc[np.asarray(definition["validation_idx"])],
                feature_types,
                preprocessing_spec,
                target_col=project_config.target_col,
                cohort_col=project_config.cohort_date_col,
            )
            cache = base.FoldMatrixCache.build(
                prepared_full,
                pool,
                sample_weight=history_training_weights(
                    prepared_full.train,
                    project_config.cohort_date_col,
                    selected_history,
                ),
                history_identity=history_fit_identity(selected_history),
            )
            result = budgeted_plan(
                budget,
                checkpoint,
                "full_row_tuned",
                "full_row_tuned",
                "RECOMMENDED_TUNED_C",
                [cache],
                recommended_features,
                selected_spec,
            )
            tuned_rows.append(result.iloc[0].to_dict())
            del cache, prepared_full
            gc.collect()
        tuned_full = pd.DataFrame(tuned_rows)
    base.atomic_write_csv(OUT / "recommended_tuned_full_row.csv", tuned_full)
    tune_ap = paired_delta(tuned_full, selected_fixed_rows, "churn_ap_skill")
    tune_lift = paired_delta(tuned_full, selected_fixed_rows, "top10_lift")
    tuned_confirmed = (
        float(tune_ap.mean()) >= -AP_STEP_TOL
        and float(tune_lift.mean()) >= -LIFT_STEP_TOL
        and int((tune_ap >= -AP_STEP_TOL).sum()) >= MIN_STABLE_FOLDS
    )
    final_c = float(selected_c.C) if tuned_confirmed else FIXED_C
    final_rows = tuned_full if tuned_confirmed else selected_fixed_rows
    final_summary = summary_fields(final_rows)
    print(
        "POST-SELECTION DIAGNOSTICS: calendar-controlled one-way and trend outputs",
        flush=True,
    )
    diagnostics_manifest = write_logistic_diagnostics(
        df=df,
        definitions=definitions,
        feature_types=feature_types,
        preprocessing_spec=preprocessing_spec,
        project_config=project_config,
        pool=pool,
        selected_features=recommended_features,
        selected_history=selected_history,
        final_c=final_c,
    )
    business_trend_manifest = write_business_retention_trend_artifacts(
        monthly=business_trend_monthly,
        output_dir=OUT / "diagnostics",
        model_family="logistic",
        cohort_col=project_config.cohort_date_col,
        selected_history=selected_history,
    )
    diagnostics_manifest = {
        **diagnostics_manifest,
        "artifacts": list(
            dict.fromkeys(
                [
                    *diagnostics_manifest["artifacts"],
                    *business_trend_manifest["artifacts"],
                    business_trend_manifest["manifest"],
                ]
            )
        ),
        "business_retention_trend": business_trend_manifest,
    }
    base.atomic_write_json(
        OUT / "diagnostics" / "logistic_diagnostics_manifest.json",
        diagnostics_manifest,
    )
    recommendation = {
        "status": "COMPLETE",
        "recommended_plan_id": str(recommended.plan_id),
        "raw_feature_count": len(recommended_features),
        "features": recommended_features,
        "model_family": "ridge_logistic",
        "solver": "lbfgs",
        "class_weight": None,
        "C": final_c,
        "selected_history": selected_history,
        "history_selection_policy": dict(HISTORY_SELECTION_POLICY),
        "history_screen_artifacts": {
            "results_long": "history_screen_results_long.csv",
            "results_long_row_count": int(len(history_long)),
            "summary": "history_screen_summary.csv",
            "summary_row_count": int(len(history_decision)),
            "lift_spread": "history_screen_lift_spread.csv",
            "lift_spread_row_count": int(len(history_spread_rows)),
            "decision": "history_selection_decision.json",
        },
        "primary_statistical_metric": "churn_ap_skill",
        "co_primary_business_metric": "churn_recall_10pct",
        "reported_business_translation": [
            "top10_lift",
            "churn_precision_10pct",
        ],
        "ranking_guardrail": "retention_roc_auc",
        "calibration_guardrails": [
            "log_loss",
            "brier_score",
            "calibration_gap",
        ],
        # Preserve the historical fields because the frozen-bundle builder
        # reads old and new recipes. AP skill is the statistical search
        # objective, while recall@10% is the co-primary operational view.
        "selection_primary_metric": "churn_ap_skill",
        "selection_guardrail": "top10_lift",
        "selection_guardrail_business_alias": "churn_recall_10pct",
        "fixed_capacity_equivalence": (
            "At a fixed 10% contact capacity, recall/capture and lift induce "
            "the same ordering; lift remains the backward-compatible numerical gate."
        ),
        "selection_method": "grouped_sequential_floating_forward",
        "selection_started_from_intercept_only": True,
        "mandatory_control_forced_before_statistical_search": True,
        "mandatory_control_features": list(MANDATORY_CONTROL_FEATURES),
        "cohort_calendar_year_contract": cohort_calendar_year_contract(),
        "quality_screened_candidate_feature_count": len(candidate_pool),
        "fixed_c_structure_selected_on_full_row_development": True,
        "sample_tuned_c": float(selected_c.C),
        "sample_tuned_c_confirmed_on_full_rows": bool(tuned_confirmed),
        "full_row_development_summary": final_summary,
        "diagnostics": diagnostics_manifest,
        "final_holdout_rows_scored": 0,
        "final_holdout_metrics_written": False,
    }
    base.atomic_write_json(OUT / "recommended_model.json", recommendation)
    run_info = {
        **config,
        **recommendation,
        "accepted_actions": int(
            path.action.isin(
                ["FORCED_ADDED", "ADDED", "FLOATING_REMOVED"]
            ).sum()
        ),
        "accepted_search_actions": int(
            path.action.isin(["ADDED", "FLOATING_REMOVED"]).sum()
        ),
        "accepted_unique_structures_full_row_scored": len(plan_registry),
        "sampled_ap_best_feature_count": len(best_seen_features),
        "new_fits_this_invocation": budget.new_fits,
        "elapsed_seconds_this_invocation": budget.elapsed,
        "interpretation": (
            "Full-row development recommendation; final six-month holdout remains model-unscored."
        ),
    }
    base.atomic_write_json(OUT / "run_info.json", run_info)
    base.atomic_write_json(
        STATUS_FILE,
        {
            "status": "COMPLETE",
            "results": str(OUT),
            "selected_history": selected_history,
            "new_fits_this_invocation": budget.new_fits,
            "elapsed_seconds_this_invocation": budget.elapsed,
            "final_holdout_rows_scored": 0,
        },
    )
    # The checkpoint is recovery-only state.  Retain it for every incomplete
    # run, but remove it after the durable COMPLETE marker is published.
    checkpoint_leftovers = checkpoint.remove()
    if checkpoint_leftovers:
        print(
            "WARNING: completed successfully, but internal checkpoint cleanup "
            "was blocked: "
            + ", ".join(str(path) for path in checkpoint_leftovers),
            flush=True,
        )
    print(
        "\nLOGISTIC REGRESSION FLOATING-FORWARD SELECTION COMPLETE\n"
        f"Selected history: {selected_history_id}\n"
        f"Recommended: {len(recommended_features)} features, Ridge C={final_c:g}\n"
        f"Mean churn AP: {final_summary['mean_churn_average_precision']:.6f}\n"
        f"Mean churn AP skill: {final_summary['mean_churn_ap_skill']:.6f}\n"
        f"Mean Lift@10%: {final_summary['mean_top10_lift']:.4f}\n"
        "Final holdout rows scored: 0\n"
        f"Results: {OUT}",
        flush=True,
    )








if __name__ == "__main__":
    try:
        run()
    except BudgetPause as exc:
        base.atomic_write_json(
            STATUS_FILE,
            {
                "status": "PAUSED_SAFELY",
                "reason": str(exc),
                "stage": exc.stage,
                "new_fits_this_invocation": exc.fits,
                "elapsed_seconds_this_invocation": exc.elapsed,
                "checkpoint": str(CHECKPOINT_FILE),
                "final_holdout_rows_scored": 0,
            },
        )
        print(
            f"\nPAUSED SAFELY: {exc}. Every completed fit is checkpointed. "
            "Rerun the identical command.",
            flush=True,
        )
        sys.exit(75)
