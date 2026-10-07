"""Modeling data, chronological-fold, preprocessing, and metric utilities.


The preparation stage and both model families use these functions so they
operate on the same validated population and fold definitions. Model-specific
feature selection remains inside the Logistic Regression and XGBoost engines.
"""


from __future__ import annotations


from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler




TARGET_COL = "RETENTION_13_1_IND"
COHORT_COL = "MONTH_1_EVALUATION_DATE"
TARGET_CHECK_COL = "MONTH_13_EVALUATION_DATE"
POLICY_ID_COL = "POLICY_CONTRACT_ID"
POLICY_INCEPTION_COL = "POLICY_INCEPTION_DATE"
ANCHOR_TERM_START_COL = "ANCHOR_TERM_START_DATE"


# Calendar-year control shared by Logistic, XGBoost, outer evaluation, and
# deployable scoring.  It is one continuous integer derived from the Month-1
# evaluation date; no month-of-year or year-month category enters the model.
COHORT_CALENDAR_YEAR_FEATURE = "COHORT_CALENDAR_YEAR"
COHORT_CALENDAR_YEAR_SOURCE_FEATURE = COHORT_COL
COHORT_CALENDAR_YEAR_TYPE = "continuous"
COHORT_CALENDAR_YEAR_FORMULA = "calendar_year(MONTH_1_EVALUATION_DATE)"
COHORT_CALENDAR_YEAR_CONTRACT_VERSION = 1




def cohort_calendar_year_contract() -> dict[str, Any]:
    """Return the immutable, population-independent year-control contract."""


    return {
        "contract_version": COHORT_CALENDAR_YEAR_CONTRACT_VERSION,
        "derived_feature": COHORT_CALENDAR_YEAR_FEATURE,
        "source_feature": COHORT_CALENDAR_YEAR_SOURCE_FEATURE,
        "configured_type": COHORT_CALENDAR_YEAR_TYPE,
        "unit": "calendar_year",
        "encoding": "continuous_integer",
        "formula": COHORT_CALENDAR_YEAR_FORMULA,
        "known_at_scoring_time": True,
        "target_used": False,
        "month_component_used": False,
    }




def derive_cohort_calendar_year(
    values: pd.Series,
    *,
    derived_feature: str = COHORT_CALENDAR_YEAR_FEATURE,
) -> pd.Series:
    """Derive one continuous integer calendar year without target information."""


    parsed = pd.to_datetime(values, errors="coerce")
    invalid = parsed.isna()
    if invalid.any():
        raise ValueError(
            f"{derived_feature} requires nonmissing valid calendar dates; "
            f"invalid_rows={int(invalid.sum())}."
        )
    result = parsed.dt.year.astype(np.int64)
    result.name = derived_feature
    return result




def ensure_cohort_calendar_year(
    frame: pd.DataFrame,
    *,
    source_feature: str = COHORT_CALENDAR_YEAR_SOURCE_FEATURE,
    derived_feature: str = COHORT_CALENDAR_YEAR_FEATURE,
) -> pd.DataFrame:
    """Attach or verify the deterministic continuous calendar-year control."""


    if source_feature not in frame:
        raise KeyError(f"{derived_feature} requires {source_feature}.")
    expected = derive_cohort_calendar_year(
        frame[source_feature], derived_feature=derived_feature
    )
    if derived_feature in frame:
        observed = pd.to_numeric(frame[derived_feature], errors="coerce")
        mismatch = observed.isna() | observed.ne(expected)
        if mismatch.any():
            raise ValueError(
                f"Existing {derived_feature} conflicts with its calendar-year "
                f"contract; mismatched_rows={int(mismatch.sum())}."
            )
        frame[derived_feature] = expected
    else:
        frame[derived_feature] = expected
    return frame


# These are virtual modeling fields.  They intentionally do not appear in the
# Snowflake projection or the ROC configuration because they are learned from
# each fold's training rows and then frozen before validation is transformed.
WRITTEN_PREM_BIN_FEATURE = "WRITTEN_PREM_BIN"
WRITTEN_PREM_SOURCE_FEATURE = "WRITTEN_PREM"
WRITTEN_PREM_REQUESTED_BINS = 20
STATE_GROUP_FEATURE = "PROPERTY_STATE_RETENTION_GROUP"
STATE_SOURCE_FEATURE = "PROPERTY_STATE"




@dataclass(frozen=True)
class ModelSpec:
    """Fold-local sampling and preprocessing settings."""


    c_value: float = 1.0
    max_iter: int = 1000
    tol: float = 1e-4
    solver: str = "saga"
    random_seed: int = 42
    rare_min_frequency: float | int | None = 0.001
    top_fraction: float = 0.10
    fail_on_nonconvergence: bool = True




@dataclass
class FeatureBlock:
    """One raw feature transformed separately inside one training fold."""


    train_matrix: sparse.csr_matrix
    validation_matrix: sparse.csr_matrix
    transformed_columns: int
    raw_dtype: str
    configured_type: str
    missing_rate_train: float
    n_unique_train: int
    sample_values: str
    fitted_artifact: dict[str, Any] = field(default_factory=dict)




@dataclass
class PreparedFold:
    """Fold data and fold-specific preprocessing cache."""


    fold_number: int
    train: pd.DataFrame
    validation: pd.DataFrame
    feature_types: Mapping[str, str]
    model_spec: ModelSpec
    target_col: str = TARGET_COL
    cohort_col: str = COHORT_COL
    policy_id_col: str = POLICY_ID_COL
    policy_inception_col: str = POLICY_INCEPTION_COL
    derived_feature_artifacts: Mapping[str, Any] | None = None


    def __post_init__(self) -> None:
        self.train = self.train.copy()
        self.validation = self.validation.copy()
        self._blocks: dict[str, FeatureBlock] = {}
        self._derived_feature_artifacts: dict[str, Any] = dict(
            self.derived_feature_artifacts or {}
        )


        self.y_train = pd.to_numeric(
            self.train[self.target_col], errors="raise"
        ).astype(int).to_numpy()
        self.y_validation = pd.to_numeric(
            self.validation[self.target_col], errors="raise"
        ).astype(int).to_numpy()
        if set(np.unique(self.y_train)) != {0, 1}:
            raise ValueError(
                f"Fold {self.fold_number} training target must contain 0 and 1."
            )
        if set(np.unique(self.y_validation)) != {0, 1}:
            raise ValueError(
                f"Fold {self.fold_number} validation target must contain 0 and 1."
            )


    def get_feature_block(self, feature: str) -> FeatureBlock:
        if feature in self._blocks:
            return self._blocks[feature]
        if feature == WRITTEN_PREM_BIN_FEATURE:
            block = self._get_written_premium_bin_block()
            self._blocks[feature] = block
            return block
        if feature == STATE_GROUP_FEATURE:
            block = self._get_state_group_block()
            self._blocks[feature] = block
            return block
        if feature not in self.train or feature not in self.validation:
            raise KeyError(
                f"Feature {feature!r} is missing from fold {self.fold_number}."
            )
        configured_type = str(self.feature_types.get(feature, "discrete")).lower()
        if configured_type not in {"discrete", "continuous"}:
            raise ValueError(
                f"Unsupported type {configured_type!r} for {feature}."
            )


        train_series = self.train[feature]
        validation_series = self.validation[feature]
        block = _fit_single_feature_transformer(
            train_series=train_series,
            validation_series=validation_series,
            configured_type=configured_type,
            rare_min_frequency=self.model_spec.rare_min_frequency,
        )
        self._blocks[feature] = block
        return block


    def _get_written_premium_bin_block(self) -> FeatureBlock:
        """Apply training-only WRITTEN_PREM quantile cut points."""


        source = WRITTEN_PREM_SOURCE_FEATURE
        if source not in self.train or source not in self.validation:
            raise KeyError(
                f"{WRITTEN_PREM_BIN_FEATURE} requires raw {source} in fold "
                f"{self.fold_number}."
            )
        artifact = self._derived_feature_artifacts.get(WRITTEN_PREM_BIN_FEATURE)
        if artifact is None:
            artifact = fit_quantile_bin_artifact(
                self.train[source],
                source_feature=source,
                derived_feature=WRITTEN_PREM_BIN_FEATURE,
                requested_bins=WRITTEN_PREM_REQUESTED_BINS,
            )
            self._derived_feature_artifacts[WRITTEN_PREM_BIN_FEATURE] = artifact
        train_binned = apply_quantile_bin_artifact(self.train[source], artifact)
        validation_binned = apply_quantile_bin_artifact(
            self.validation[source], artifact
        )
        categories = [*artifact["labels"], artifact["missing_label"]]
        block = _fit_single_feature_transformer(
            train_series=train_binned,
            validation_series=validation_binned,
            configured_type="discrete",
            rare_min_frequency=None,
            discrete_categories=categories,
        )
        block.fitted_artifact = dict(artifact)
        return block


    def _get_state_group_block(self) -> FeatureBlock:
        """Fit/apply the retention-behavior state grouper without leakage."""


        source = STATE_SOURCE_FEATURE
        if source not in self.train or source not in self.validation:
            raise KeyError(
                f"{STATE_GROUP_FEATURE} requires raw {source} in fold "
                f"{self.fold_number}."
            )
        artifact = self._derived_feature_artifacts.get(STATE_GROUP_FEATURE)
        if artifact is None:
            fit_state_grouper, _, state_spec = _state_grouping_api()
            artifact = fit_state_grouper(
                self.train,
                state_spec(
                    state_col=source,
                    target_col=self.target_col,
                    tenure_col="TENURE_YEARS_GRP",
                ),
                context={"fold": self.fold_number, "fit_rows": "unsampled"},
            )
            self._derived_feature_artifacts[STATE_GROUP_FEATURE] = artifact
        _, apply_state_grouper, _ = _state_grouping_api()
        state_categories = _fitted_state_group_categories(artifact)
        train_group, train_audit = apply_state_grouper(
            artifact, self.train, scope=f"fold_{self.fold_number}_train"
        )
        validation_group, validation_audit = apply_state_grouper(
            artifact,
            self.validation,
            scope=f"fold_{self.fold_number}_validation",
        )
        block = _fit_single_feature_transformer(
            train_series=train_group,
            validation_series=validation_group,
            configured_type="discrete",
            rare_min_frequency=None,
            discrete_categories=list(state_categories),
        )
        block.fitted_artifact = {
            "artifact": state_grouping_artifact_payload(artifact),
            "train_application": train_audit,
            "validation_application": validation_audit,
        }
        return block


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input data not found: {path.resolve()}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported input type: {path.suffix}")




def normalize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"column", "role", "var_type"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    result = manifest.copy()
    result["column"] = result["column"].astype(str).str.strip().str.upper()
    result["role"] = result["role"].astype(str).str.strip().str.upper()
    result["var_type"] = result["var_type"].astype(str).str.strip().str.lower()
    for col in [
        "enabled_for_model",
        "availability_in_source",
        "point_in_time_status",
        "governance_status",
        "decision_reason",
    ]:
        if col not in result:
            result[col] = ""
        result[col] = result[col].astype(str).str.strip()
    if result["column"].duplicated().any():
        duplicate = result.loc[result["column"].duplicated(), "column"].tolist()
        raise ValueError(f"Duplicate manifest columns: {duplicate}")
    return result




def build_candidate_audit(
    manifest: pd.DataFrame,
    prepared_columns: Iterable[str],
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Create the candidate pool from the run-specific config contract.


    Python does not maintain a second feature-family allowlist.  A field must
    be a config-enabled candidate, present in the queried data, and have a
    supported modeling type.
    """


    contract = normalize_manifest(manifest)
    prepared_set = {str(col).upper() for col in prepared_columns}
    rows: list[dict[str, Any]] = []
    feature_types: dict[str, str] = {}


    for _, row in contract.iterrows():
        feature = row["column"]
        role = row["role"]
        availability = str(row.get("availability_in_source", "")).upper()
        present_source = availability == "PRESENT"
        present_prepared = feature in prepared_set
        configured_type = row["var_type"]
        enabled_flag = str(row.get("enabled_for_model", "")).upper()
        enabled = enabled_flag in {"1", "TRUE", "T", "Y", "YES"}


        included = False
        reason = ""


        if role != "CANDIDATE":
            reason = f"role={role}; structural/target fields never compete"
        elif not enabled:
            reason = "run-specific model contract disables this field"
        elif not present_source:
            reason = "manifest says feature is absent from the source CTE"
        elif not present_prepared:
            reason = (
                "present in source contract but absent from prepared input; rebuild "
                "the experiment extract before this feature can compete"
            )
        elif configured_type not in {"discrete", "continuous"}:
            reason = f"unsupported manifest var_type={configured_type!r}"
        else:
            included = True
            feature_types[feature] = configured_type


        rows.append(
            {
                "feature_name": feature,
                "role": role,
                "present_in_source": "Y" if present_source else "N",
                "present_in_prepared_data": "Y" if present_prepared else "N",
                "original_enabled_flag": enabled_flag,
                "variable_type": configured_type,
                "included_in_experiment_pool": "Y" if included else "N",
                "exclusion_reason": reason,
                "note": "Eligibility comes from the full ROC config and SQL extract.",
            }
        )


    audit = pd.DataFrame(rows)
    candidates = audit.loc[
        audit["included_in_experiment_pool"].eq("Y"), "feature_name"
    ].tolist()
    return audit, candidates, feature_types




def fit_quantile_bin_artifact(
    train_series: pd.Series,
    *,
    source_feature: str = WRITTEN_PREM_SOURCE_FEATURE,
    derived_feature: str = WRITTEN_PREM_BIN_FEATURE,
    requested_bins: int = WRITTEN_PREM_REQUESTED_BINS,
) -> dict[str, Any]:
    """Learn duplicate-safe quantile cut points from training rows only.


    Infinite scoring bounds make later values outside the observed training
    range deterministic instead of converting them to missing.  Repeated
    quantiles (common when premium has a large zero mass) are collapsed.
    """


    if requested_bins < 2:
        raise ValueError("requested_bins must be at least 2.")
    numeric = pd.to_numeric(train_series, errors="coerce").astype(float)
    numeric = numeric.where(np.isfinite(numeric), np.nan)
    observed = numeric.dropna().to_numpy(dtype=float)
    if np.unique(observed).size < 2:
        raise ValueError(
            f"{derived_feature} requires at least two finite training values."
        )
    quantiles = np.linspace(0.0, 1.0, int(requested_bins) + 1)
    values = np.quantile(observed, quantiles)
    lower = float(np.min(observed))
    upper = float(np.max(observed))
    interior = sorted(
        {
            float(value)
            for value in values[1:-1]
            if np.isfinite(value) and lower < float(value) < upper
        }
    )
    edges = [-np.inf, *interior, np.inf]
    actual_bins = len(edges) - 1
    width = max(2, len(str(actual_bins)))
    labels = [
        f"WRITTEN_PREM_Q{index:0{width}d}"
        for index in range(1, actual_bins + 1)
    ]
    return {
        "artifact_version": 1,
        "transform": "training_quantile_bins",
        "fit_scope": "fold_training_rows_only",
        "source_feature": str(source_feature).upper(),
        "derived_feature": str(derived_feature).upper(),
        "requested_bins": int(requested_bins),
        "actual_bins": int(actual_bins),
        "finite_cut_points": interior,
        "edges": edges,
        "labels": labels,
        "missing_label": "WRITTEN_PREM_MISSING",
        "train_row_count": int(len(train_series)),
        "train_nonmissing_count": int(len(observed)),
        "train_missing_count": int(numeric.isna().sum()),
        "train_min": lower,
        "train_max": upper,
    }




def apply_quantile_bin_artifact(
    series: pd.Series,
    artifact: Mapping[str, Any],
) -> pd.Series:
    """Apply frozen training quantile edges to a raw premium series."""


    edges = np.asarray(artifact["edges"], dtype=float)
    labels = [str(value) for value in artifact["labels"]]
    if len(edges) != len(labels) + 1 or not np.all(np.diff(edges) > 0):
        raise ValueError("Invalid quantile-bin artifact: edges are not increasing.")
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    numeric = numeric.where(np.isfinite(numeric), np.nan)
    result = pd.cut(
        numeric,
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
        ordered=True,
    ).astype("string")
    return result.fillna(str(artifact["missing_label"])).rename(
        str(artifact["derived_feature"])
    )




def _state_grouping_api() -> tuple[Any, Any, Any]:
    """Import the standalone state module in package and script contexts."""


    try:
        from c_modeling.state_grouping import (
            StateGroupingSpec,
            apply_state_grouper,
            fit_state_grouper,
        )
    except ImportError:
        from state_grouping import (  # type: ignore
            StateGroupingSpec,
            apply_state_grouper,
            fit_state_grouper,
        )
    return fit_state_grouper, apply_state_grouper, StateGroupingSpec




def _fitted_state_group_categories(artifact: Any) -> tuple[str, ...]:
    """Read the exact dynamic category contract from one fitted grouper.


    The selected number of non-TX groups is learned separately inside each
    training fold.  Therefore a module-level category constant cannot safely
    define the encoder width.  The fitted artifact owns that contract.
    """


    raw_categories = getattr(artifact, "categories", None)
    if callable(raw_categories):
        raw_categories = raw_categories()
    if raw_categories is None:
        raw_categories = getattr(artifact, "selected_categories", None)
    if raw_categories is None:
        raise TypeError(
            "Fitted state grouper must expose its selected categories."
        )
    categories = tuple(str(value) for value in raw_categories)
    if not categories or len(categories) != len(set(categories)):
        raise ValueError(
            "Fitted state grouper categories must be nonempty and unique."
        )
    mapped_labels = {str(value) for value in artifact.mapping.values()}
    if not mapped_labels.issubset(categories):
        raise ValueError(
            "Fitted state mapping contains labels outside its category contract."
        )
    return categories




def state_grouping_artifact_payload(artifact: Any) -> dict[str, Any]:
    """Return an inspectable payload without constraining artifact internals."""


    if hasattr(artifact, "payload"):
        payload = artifact.payload()
    elif hasattr(artifact, "to_dict"):
        payload = artifact.to_dict()
    elif is_dataclass(artifact):
        payload = asdict(artifact)
    elif isinstance(artifact, Mapping):
        payload = dict(artifact)
    else:
        raise TypeError(
            "State grouping artifact must be a dataclass, mapping, or expose to_dict()."
        )
    return payload




def prefit_derived_feature_artifacts(
    full_training_frame: pd.DataFrame,
    feature_types: Mapping[str, str],
    *,
    target_col: str = TARGET_COL,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Fit requested virtual features on an unsampled fold-training frame."""


    del random_seed  # Retained only for compatibility with existing callers.
    artifacts: dict[str, Any] = {}
    if WRITTEN_PREM_BIN_FEATURE in feature_types:
        if WRITTEN_PREM_SOURCE_FEATURE not in full_training_frame:
            raise KeyError(
                f"{WRITTEN_PREM_BIN_FEATURE} requires "
                f"{WRITTEN_PREM_SOURCE_FEATURE}."
            )
        artifacts[WRITTEN_PREM_BIN_FEATURE] = fit_quantile_bin_artifact(
            full_training_frame[WRITTEN_PREM_SOURCE_FEATURE],
            requested_bins=WRITTEN_PREM_REQUESTED_BINS,
        )
    if STATE_GROUP_FEATURE in feature_types:
        fit_state_grouper, _, state_spec = _state_grouping_api()
        artifacts[STATE_GROUP_FEATURE] = fit_state_grouper(
            full_training_frame,
            state_spec(
                state_col=STATE_SOURCE_FEATURE,
                target_col=target_col,
                tenure_col="TENURE_YEARS_GRP",
            ),
            context={"fit_rows": "full_unsampled_fold_training"},
        )
    return artifacts




def export_fold_derived_artifacts(
    folds: Sequence[PreparedFold],
    output_dir: str | Path,
    *,
    stage: str,
) -> pd.DataFrame:
    """Persist fold-fitted virtual-feature artifacts used by a model family."""


    safe_stage = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in str(stage)
    ).strip("_")
    if not safe_stage:
        raise ValueError("stage must contain at least one safe character.")
    destination = Path(output_dir).expanduser().resolve() / safe_stage
    destination.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []


    for fold in folds:
        fold_dir = destination / f"fold_{int(fold.fold_number):02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for feature in (WRITTEN_PREM_BIN_FEATURE, STATE_GROUP_FEATURE):
            if feature not in fold.feature_types:
                continue
            block = fold.get_feature_block(feature)
            feature_dir = fold_dir / feature.lower()
            feature_dir.mkdir(parents=True, exist_ok=True)
            if feature == WRITTEN_PREM_BIN_FEATURE:
                artifact = block.fitted_artifact
                edges = [float(value) for value in artifact["edges"]]
                labels = [str(value) for value in artifact["labels"]]
                bin_rows = []
                for index, label in enumerate(labels):
                    lower = edges[index]
                    upper = edges[index + 1]
                    bin_rows.append(
                        {
                            "bin_order": index + 1,
                            "bin_label": label,
                            "lower_bound_exclusive": (
                                "-INF" if np.isneginf(lower) else lower
                            ),
                            "upper_bound_inclusive": (
                                "INF" if np.isposinf(upper) else upper
                            ),
                        }
                    )
                _atomic_frame_csv(
                    feature_dir / "written_prem_bin_edges.csv",
                    pd.DataFrame(bin_rows),
                )
                _atomic_payload_json(
                    feature_dir / "written_prem_bin_artifact.json",
                    artifact,
                )
            else:
                fitted = fold._derived_feature_artifacts.get(STATE_GROUP_FEATURE)
                if fitted is None or not hasattr(fitted, "save"):
                    raise RuntimeError("Fitted state artifact is not serializable.")
                state_categories = _fitted_state_group_categories(fitted)
                category_group_count = len(state_categories) - 1
                selected_non_tx_groups = int(
                    fitted.assignment_audit.get(
                        "selected_non_tx_group_count", category_group_count
                    )
                )
                known_groups_including_tx = int(
                    fitted.assignment_audit.get(
                        "selected_known_group_count_including_tx",
                        selected_non_tx_groups + 1,
                    )
                )
                encoded_categories = int(
                    fitted.assignment_audit.get(
                        "encoded_category_count",
                        len(state_categories),
                    )
                )
                if (
                    selected_non_tx_groups < 1
                    or selected_non_tx_groups != category_group_count
                    or known_groups_including_tx != category_group_count + 1
                    or encoded_categories != len(state_categories)
                ):
                    raise RuntimeError(
                        "Fitted state artifact selected-K/category contract is "
                        "inconsistent."
                    )
                fitted.save(feature_dir)
            summary_row = {
                "stage": safe_stage,
                "fold": int(fold.fold_number),
                "feature": feature,
                "transformed_columns": int(block.transformed_columns),
                "training_rows_encoded": int(len(fold.train)),
                "validation_rows_encoded": int(len(fold.validation)),
                "selected_non_tx_groups": None,
                "total_known_state_groups_including_tx": None,
                "encoded_state_categories": None,
                "state_group_selection_method": "",
                "state_group_selection_status": "",
                "state_group_selection_evidence_tier": "",
                "state_group_selection_warnings": "",
                "state_group_fallback_used": None,
                "state_group_fallback_reason": "",
                "state_group_hard_constraints_relaxed": None,
                "state_group_stability_role": "",
                "state_group_behavior_rank_stability_role": "",
                "state_group_feasible_k": "",
                "state_group_bic_tolerance_eligible_k": "",
                "state_group_paired_one_se_eligible_k": "",
                "state_group_stability_preferred_k": "",
                "state_group_behavior_definition": "",
                "state_group_required_minimum_adjacent_retention_gap": None,
                "state_group_selected_minimum_adjacent_retention_gap": None,
                "state_group_adjacent_retention_gap_pass": None,
                "state_group_adjacent_retention_gap_definition": "",
                "state_group_tx_exempt_from_adjacent_retention_gap": None,
                "state_group_categories": "",
                "state_cleaned_contract_count": None,
                "state_mapped_state_count": None,
                "state_all_cleaned_states_mapped": None,
                "state_volume_role": "",
            }
            if feature == STATE_GROUP_FEATURE:
                assignment_audit = fitted.assignment_audit
                summary_row["selected_non_tx_groups"] = int(
                    selected_non_tx_groups
                )
                summary_row["total_known_state_groups_including_tx"] = int(
                    known_groups_including_tx
                )
                summary_row["encoded_state_categories"] = int(encoded_categories)
                summary_row["state_group_selection_method"] = str(
                    assignment_audit.get("selection_method", "")
                )
                summary_row["state_group_selection_status"] = str(
                    assignment_audit.get("selection_status", "")
                )
                summary_row["state_group_selection_evidence_tier"] = str(
                    assignment_audit.get("selection_evidence_tier", "")
                )
                summary_row["state_group_selection_warnings"] = " | ".join(
                    str(value)
                    for value in assignment_audit.get(
                        "selection_warnings", []
                    )
                )
                summary_row["state_group_fallback_used"] = bool(
                    assignment_audit.get("fallback_used", False)
                )
                summary_row["state_group_fallback_reason"] = str(
                    assignment_audit.get("fallback_reason") or ""
                )
                summary_row["state_group_hard_constraints_relaxed"] = bool(
                    assignment_audit.get("hard_constraints_relaxed", False)
                )
                summary_row["state_group_stability_role"] = str(
                    assignment_audit.get("stability_role", "")
                )
                summary_row["state_group_behavior_rank_stability_role"] = str(
                    assignment_audit.get("behavior_rank_stability_role", "")
                )
                for audit_key, summary_key in (
                    ("feasible_k", "state_group_feasible_k"),
                    (
                        "bic_tolerance_eligible_k",
                        "state_group_bic_tolerance_eligible_k",
                    ),
                    (
                        "paired_one_se_eligible_k",
                        "state_group_paired_one_se_eligible_k",
                    ),
                    (
                        "stability_preferred_k",
                        "state_group_stability_preferred_k",
                    ),
                ):
                    summary_row[summary_key] = ";".join(
                        str(value)
                        for value in assignment_audit.get(audit_key, [])
                    )
                summary_row["state_group_behavior_definition"] = str(
                    assignment_audit.get("behavior_definition", "")
                )
                summary_row[
                    "state_group_required_minimum_adjacent_retention_gap"
                ] = float(
                    assignment_audit.get(
                        "minimum_required_adjacent_non_tx_retention_rate_gap",
                        np.nan,
                    )
                )
                summary_row[
                    "state_group_selected_minimum_adjacent_retention_gap"
                ] = assignment_audit.get(
                    "minimum_selected_adjacent_non_tx_retention_rate_gap"
                )
                summary_row[
                    "state_group_adjacent_retention_gap_pass"
                ] = bool(
                    assignment_audit.get(
                        "adjacent_non_tx_retention_rate_gap_constraint_satisfied",
                        False,
                    )
                )
                summary_row[
                    "state_group_adjacent_retention_gap_definition"
                ] = str(
                    assignment_audit.get(
                        "adjacent_retention_rate_gap_definition", ""
                    )
                )
                summary_row[
                    "state_group_tx_exempt_from_adjacent_retention_gap"
                ] = bool(
                    assignment_audit.get(
                        "tx_exempt_from_adjacent_retention_rate_gap_constraint",
                        False,
                    )
                )
                summary_row["state_group_categories"] = ";".join(
                    state_categories
                )
                summary_row["state_cleaned_contract_count"] = int(
                    assignment_audit.get("cleaned_state_contract_count", -1)
                )
                summary_row["state_mapped_state_count"] = int(
                    assignment_audit.get("mapped_state_count", -1)
                )
                summary_row["state_all_cleaned_states_mapped"] = bool(
                    assignment_audit.get("all_cleaned_states_mapped", False)
                )
                summary_row["state_volume_role"] = str(
                    assignment_audit.get("volume_role", "")
                )
            summary_rows.append(summary_row)


    summary = pd.DataFrame(summary_rows)
    _atomic_frame_csv(destination / "derived_feature_summary.csv", summary)
    return summary




def _artifact_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _artifact_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_artifact_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_artifact_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _artifact_jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            return None
        return "INF" if value > 0 else "-INF"
    if isinstance(value, (pd.Timestamp, pd.Period, Path)):
        return str(value)
    return value




def _atomic_payload_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            _artifact_jsonable(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)




def _atomic_frame_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)




def quality_screen_candidates(
    training_frame: pd.DataFrame,
    candidates: Sequence[str],
    feature_types: Mapping[str, str],
    *,
    missing_drop_threshold: float = 0.95,
    dominant_level_drop_threshold: float = 0.995,
    high_cardinality_ratio: float = 0.50,
    max_category_levels: int = 500,
) -> tuple[list[str], pd.DataFrame]:
    """Apply quality gates using training data only."""


    rows: list[dict[str, Any]] = []
    kept: list[str] = []
    n_rows = len(training_frame)
    for feature in candidates:
        series = training_frame[feature]
        missing_rate = float(series.isna().mean())
        nonmissing = series.dropna()
        n_unique = int(nonmissing.nunique())
        unique_ratio = float(n_unique / max(len(nonmissing), 1))
        dominant_level_rate = (
            float(
                series.astype("string")
                .fillna("__MISSING__")
                .value_counts(normalize=True, dropna=False)
                .iloc[0]
            )
            if n_rows
            else 1.0
        )
        configured_type = feature_types[feature]
        decision = "KEEP"
        reason = "passed training-only quality gates"
        if n_unique <= 1:
            decision, reason = "DROP", "constant or all-missing in selection training"
        elif missing_rate >= missing_drop_threshold:
            decision, reason = "DROP", "missing rate exceeds threshold"
        elif dominant_level_rate >= dominant_level_drop_threshold:
            decision, reason = "DROP", "dominant level exceeds threshold"
        elif configured_type == "discrete" and n_unique > max_category_levels:
            decision, reason = "DROP", "category count exceeds threshold"
        elif configured_type == "discrete" and unique_ratio >= high_cardinality_ratio:
            decision, reason = "DROP", "category unique ratio exceeds threshold"
        else:
            kept.append(feature)
        rows.append(
            {
                "feature": feature,
                "configured_type": configured_type,
                "row_count": n_rows,
                "missing_rate": missing_rate,
                "n_unique": n_unique,
                "unique_ratio": unique_ratio,
                "dominant_level_rate": dominant_level_rate,
                "decision": decision,
                "reason": reason,
            }
        )
    if not kept:
        raise ValueError("Training-only quality screening removed every candidate.")
    return kept, pd.DataFrame(rows)




def build_expanding_folds(
    frame: pd.DataFrame,
    *,
    cohort_col: str = COHORT_COL,
    target_check_col: str = TARGET_CHECK_COL,
    target_col: str = TARGET_COL,
    validation_folds: int = 3,
    validation_months: int = 3,
    min_train_months: int = 12,
    holdout_months: int = 0,
    label_buffer_days: int = 0,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Build complete-month expanding folds with a Month-13 label embargo."""


    if validation_folds < 1 or validation_months < 1 or min_train_months < 1:
        raise ValueError("Fold counts and month counts must be positive.")
    if holdout_months < 0 or label_buffer_days < 0:
        raise ValueError("holdout_months and label_buffer_days cannot be negative.")
    work = frame
    cohort = pd.to_datetime(work[cohort_col], errors="raise").dt.to_period("M")
    target_check = pd.to_datetime(work[target_check_col], errors="raise")
    months = pd.Index(sorted(cohort.unique()))
    required = min_train_months + validation_folds * validation_months + holdout_months
    if len(months) < required:
        raise ValueError(
            f"Need at least {required} cohort months for the requested split; "
            f"found {len(months)}."
        )


    if holdout_months:
        holdout_values = months[-holdout_months:]
        development_months = months[:-holdout_months]
        holdout_idx = work.index[cohort.isin(holdout_values)].to_numpy()
    else:
        development_months = months
        holdout_idx = np.asarray([], dtype=int)


    first_validation_index = len(development_months) - (
        validation_folds * validation_months
    )
    folds: list[dict[str, Any]] = []
    for fold_offset in range(validation_folds):
        start = first_validation_index + fold_offset * validation_months
        stop = start + validation_months
        validation_values = development_months[start:stop]
        validation_start = validation_values[0].to_timestamp(how="start")
        label_cutoff = validation_start - pd.Timedelta(days=label_buffer_days)
        train_mask = cohort.isin(development_months[:start]) & target_check.lt(
            label_cutoff
        )
        validation_mask = cohort.isin(validation_values)
        train_idx = work.index[train_mask].to_numpy()
        validation_idx = work.index[validation_mask].to_numpy()
        if not len(train_idx) or not len(validation_idx):
            raise ValueError(f"Fold {fold_offset + 1} has an empty partition.")
        if np.intersect1d(train_idx, validation_idx).size:
            raise AssertionError("Train and validation rows overlap.")
        train_target = pd.to_numeric(work.loc[train_idx, target_col], errors="raise")
        valid_target = pd.to_numeric(
            work.loc[validation_idx, target_col], errors="raise"
        )
        if train_target.nunique() != 2 or valid_target.nunique() != 2:
            raise ValueError(
                f"Fold {fold_offset + 1} does not contain both target classes."
            )
        train_month_count = pd.to_datetime(
            work.loc[train_idx, cohort_col]
        ).dt.to_period("M").nunique()
        if train_month_count < min_train_months:
            raise ValueError(
                f"Fold {fold_offset + 1} has only {train_month_count} label-available "
                f"training months; need {min_train_months}."
            )
        folds.append(
            {
                "fold": fold_offset + 1,
                "train_idx": train_idx,
                "validation_idx": validation_idx,
                "validation_months": [str(value) for value in validation_values],
                "label_cutoff": label_cutoff,
            }
        )
    return folds, holdout_idx




def select_recent_mature_training_indices(
    frame: pd.DataFrame,
    train_idx: Iterable[Any],
    *,
    cohort_col: str = COHORT_COL,
    months: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select the latest *mature training* cohort months, inclusively.


    A 13/1 model has an approximately twelve-month gap between the newest
    label-mature training cohort and the prediction block.  Anchoring a
    ``RECENT_36M`` window to validation start would therefore retain only
    about 24 usable training months.  This helper anchors the window to the
    newest row that is already admitted to the fold's training partition.


    The selected calendar window is inclusive: ``months=36`` keeps the newest
    mature cohort month plus the preceding 35 calendar months.  If the source
    contains fewer than the requested number, all available mature months are
    retained and the audit reports the shortfall.
    """


    if isinstance(months, bool) or int(months) != months or int(months) < 1:
        raise ValueError("months must be a positive integer.")
    raw_idx = np.asarray(list(train_idx) if not isinstance(train_idx, np.ndarray) else train_idx)
    if raw_idx.ndim != 1 or not len(raw_idx):
        raise ValueError("train_idx must be a nonempty one-dimensional index.")
    cohort = pd.to_datetime(
        frame.loc[raw_idx, cohort_col], errors="raise"
    ).dt.to_period("M")
    newest = cohort.max()
    cutoff = newest - (int(months) - 1)
    keep = cohort.ge(cutoff).to_numpy()
    selected = raw_idx[keep]
    if not len(selected):
        raise AssertionError("Recent-history filtering removed every training row.")
    selected_months = cohort.loc[keep]
    distinct = int(selected_months.nunique())
    audit = {
        "history_anchor": "newest_label_mature_training_cohort",
        "requested_months": int(months),
        "anchor_month": str(newest),
        "cutoff_month_inclusive": str(cutoff),
        "available_months_in_window": distinct,
        "requested_months_fully_available": bool(distinct >= int(months)),
        "selected_rows": int(len(selected)),
    }
    return selected, audit




def fold_definitions_frame(
    frame: pd.DataFrame,
    folds: Sequence[Mapping[str, Any]],
    *,
    cohort_col: str = COHORT_COL,
    target_check_col: str = TARGET_CHECK_COL,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for position, fold in enumerate(folds, start=1):
        train = frame.loc[np.asarray(fold["train_idx"])]
        valid = frame.loc[np.asarray(fold["validation_idx"])]
        train_dates = pd.to_datetime(train[cohort_col])
        valid_dates = pd.to_datetime(valid[cohort_col])
        label_cutoff = pd.Timestamp(fold["label_cutoff"])
        rows.append(
            {
                "fold": int(fold.get("fold", position)),
                "train_start": str(train_dates.min().date()),
                "train_end": str(train_dates.max().date()),
                "validation_start": str(valid_dates.min().date()),
                "validation_end": str(valid_dates.max().date()),
                "label_cutoff": str(label_cutoff.date()),
                "train_rows": int(len(train)),
                "validation_rows": int(len(valid)),
                "train_index_sha256": _sorted_int64_index_sha256(
                    fold["train_idx"]
                ),
                "validation_index_sha256": _sorted_int64_index_sha256(
                    fold["validation_idx"]
                ),
                "train_cohort_months": int(train_dates.dt.to_period("M").nunique()),
                "validation_cohort_months": int(
                    valid_dates.dt.to_period("M").nunique()
                ),
                "train_retention_rate": float(train[target_col].mean()),
                "validation_retention_rate": float(valid[target_col].mean()),
                "validation_churn_rate": float(1.0 - valid[target_col].mean()),
                "max_train_target_check": str(
                    pd.to_datetime(train[target_check_col]).max().date()
                ),
                "label_embargo_pass": bool(
                    pd.to_datetime(train[target_check_col]).max() < label_cutoff
                ),
                "row_overlap_count": int(
                    np.intersect1d(train.index.to_numpy(), valid.index.to_numpy()).size
                ),
                "cohort_overlap_count": int(
                    len(
                        set(train_dates.dt.to_period("M"))
                        & set(valid_dates.dt.to_period("M"))
                    )
                ),
            }
        )
    result = pd.DataFrame(rows)
    if not result["label_embargo_pass"].all():
        raise AssertionError("At least one fold violates the label embargo.")
    if result["row_overlap_count"].sum() or result["cohort_overlap_count"].sum():
        raise AssertionError("At least one fold has train/validation overlap.")
    return result




FOLD_DEFINITION_CONTRACT_COLUMNS = (
    "fold",
    "label_cutoff",
    "train_rows",
    "validation_rows",
    "train_index_sha256",
    "validation_index_sha256",
)




def _sorted_int64_index_sha256(index_values: Iterable[Any]) -> str:
    """Hash index membership independently of the input index order."""


    raw = np.asarray(index_values)
    if raw.ndim != 1:
        raise ValueError(
            "Fold indices must be one-dimensional before contract hashing."
        )
    try:
        numeric = pd.to_numeric(pd.Series(raw), errors="raise").to_numpy()
    except (TypeError, ValueError) as exc:
        raise ValueError("Fold indices must be convertible to int64.") from exc
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("Fold indices must contain only finite integer values.")
    try:
        canonical = np.sort(numeric.astype(np.int64), kind="stable").astype(
            "<i8", copy=False
        )
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("Fold indices must fit in signed int64.") from exc
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()




def verify_fold_definition_contract(
    actual: pd.DataFrame,
    expected_csv: str | Path,
    *,
    label: str,
) -> None:
    """Verify the immutable fold-membership fields against a saved CSV.


    Date ranges and row counts alone cannot identify a fold: different rows can
    share the same cohort dates.  The sorted-index hashes make membership part
    of the reusable modeling contract while remaining insensitive to array
    ordering.
    """


    expected_path = Path(expected_csv)
    if not expected_path.is_file():
        raise FileNotFoundError(
            f"{label} fold-definition contract is missing: {expected_path}"
        )
    expected = pd.read_csv(expected_path, dtype=str, keep_default_na=False)


    def normalized(source: pd.DataFrame, source_label: str) -> pd.DataFrame:
        missing = [
            column
            for column in FOLD_DEFINITION_CONTRACT_COLUMNS
            if column not in source.columns
        ]
        if missing:
            raise ValueError(
                f"{label} {source_label} fold-definition contract is missing "
                f"columns: {missing}"
            )
        output = source.loc[:, FOLD_DEFINITION_CONTRACT_COLUMNS].copy()
        try:
            for column in ("fold", "train_rows", "validation_rows"):
                numeric = pd.to_numeric(output[column], errors="raise")
                if not np.equal(numeric, np.floor(numeric)).all():
                    raise ValueError(f"{column} contains a non-integer value")
                output[column] = numeric.astype("int64")
            output["label_cutoff"] = pd.to_datetime(
                output["label_cutoff"], errors="raise"
            ).dt.strftime("%Y-%m-%d")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} {source_label} fold-definition contract has invalid "
                f"typed values: {exc}"
            ) from exc
        if output["fold"].duplicated().any():
            duplicates = sorted(
                output.loc[output["fold"].duplicated(False), "fold"].unique()
            )
            raise ValueError(
                f"{label} {source_label} fold-definition contract has duplicate "
                f"fold numbers: {duplicates}"
            )
        for column in ("train_index_sha256", "validation_index_sha256"):
            output[column] = output[column].astype(str).str.strip().str.lower()
            valid_hash = output[column].str.fullmatch(r"[0-9a-f]{64}")
            if not valid_hash.all():
                bad_folds = output.loc[~valid_hash, "fold"].tolist()
                raise ValueError(
                    f"{label} {source_label} fold-definition contract has invalid "
                    f"{column} values for folds: {bad_folds}"
                )
        return output.sort_values("fold", kind="stable").reset_index(drop=True)


    observed = normalized(actual, "actual")
    baseline = normalized(expected, "expected")
    observed_folds = set(observed["fold"])
    baseline_folds = set(baseline["fold"])
    if observed_folds != baseline_folds:
        raise ValueError(
            f"{label} fold-definition contract fold numbers differ: "
            f"expected={sorted(baseline_folds)}, actual={sorted(observed_folds)}"
        )


    mismatches: list[str] = []
    observed_by_fold = observed.set_index("fold")
    baseline_by_fold = baseline.set_index("fold")
    for fold_number in sorted(baseline_folds):
        for column in FOLD_DEFINITION_CONTRACT_COLUMNS[1:]:
            expected_value = baseline_by_fold.at[fold_number, column]
            actual_value = observed_by_fold.at[fold_number, column]
            if actual_value != expected_value:
                mismatches.append(
                    f"fold {fold_number} {column}: expected={expected_value!r}, "
                    f"actual={actual_value!r}"
                )
    if mismatches:
        raise ValueError(
            f"{label} fold-definition contract mismatch:\n- "
            + "\n- ".join(mismatches)
        )




def proportional_month_target_sample(
    frame: pd.DataFrame,
    *,
    max_rows: int | None,
    cohort_col: str = COHORT_COL,
    target_col: str = TARGET_COL,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Sample proportionally within cohort-month Ã target groups."""


    if max_rows is None or max_rows <= 0 or len(frame) <= max_rows:
        return frame.copy()
    if max_rows < 2:
        raise ValueError("max_rows must be at least 2.")
    helper = frame[[cohort_col, target_col]].copy()
    helper["_month"] = pd.to_datetime(helper[cohort_col]).dt.to_period("M")
    helper["_target"] = pd.to_numeric(helper[target_col], errors="raise").astype(int)
    grouped = helper.groupby(["_month", "_target"], sort=True).size()
    group_count = len(grouped)
    if max_rows < group_count:
        raise ValueError(
            f"max_rows={max_rows} is smaller than the {group_count} month/target groups."
        )
    ideal = grouped.astype(float) * int(max_rows) / len(frame)
    allocation = np.floor(ideal).astype(int).clip(lower=1)
    allocation = np.minimum(allocation, grouped).astype(int)
    while int(allocation.sum()) < max_rows:
        capacity = grouped - allocation
        candidates = capacity[capacity > 0]
        if candidates.empty:
            break
        deficits = (ideal - allocation).loc[candidates.index]
        chosen = deficits.sort_values(ascending=False, kind="stable").index[0]
        allocation.loc[chosen] += 1
    while int(allocation.sum()) > max_rows:
        candidates = allocation[allocation > 1]
        if candidates.empty:
            raise RuntimeError("Unable to satisfy proportional sample allocation.")
        surplus = (allocation - ideal).loc[candidates.index]
        chosen = surplus.sort_values(ascending=False, kind="stable").index[0]
        allocation.loc[chosen] -= 1


    parts: list[pd.DataFrame] = []
    for offset, group_key in enumerate(grouped.index):
        month, label = group_key
        mask = helper["_month"].eq(month) & helper["_target"].eq(label)
        group_frame = frame.loc[mask]
        parts.append(
            group_frame.sample(
                n=int(allocation.loc[group_key]),
                replace=False,
                random_state=random_seed + offset,
            )
        )
    sampled = pd.concat(parts, axis=0).sort_index(kind="stable")
    if len(sampled) != max_rows:
        raise RuntimeError(
            f"Sampling returned {len(sampled)} rows; expected {max_rows}."
        )
    if sampled[target_col].nunique() != 2:
        raise RuntimeError("Sampling lost one target class.")
    return sampled




def prepare_fold_objects(
    frame: pd.DataFrame,
    folds: Sequence[Mapping[str, Any]],
    feature_types: Mapping[str, str],
    model_spec: ModelSpec,
    *,
    train_max_rows: int | None = None,
    validation_max_rows: int | None = None,
    cohort_col: str = COHORT_COL,
    target_col: str = TARGET_COL,
) -> list[PreparedFold]:
    prepared: list[PreparedFold] = []
    for position, fold in enumerate(folds, start=1):
        full_train = frame.loc[np.asarray(fold["train_idx"])]
        validation = frame.loc[np.asarray(fold["validation_idx"])]
        derived_artifacts = prefit_derived_feature_artifacts(
            full_train,
            feature_types,
            target_col=target_col,
            random_seed=model_spec.random_seed + 1000 * position,
        )
        train = full_train
        train = proportional_month_target_sample(
            train,
            max_rows=train_max_rows,
            cohort_col=cohort_col,
            target_col=target_col,
            random_seed=model_spec.random_seed + 1000 * position,
        )
        validation = proportional_month_target_sample(
            validation,
            max_rows=validation_max_rows,
            cohort_col=cohort_col,
            target_col=target_col,
            random_seed=model_spec.random_seed + 2000 * position,
        )
        prepared.append(
            PreparedFold(
                fold_number=int(fold.get("fold", position)),
                train=train,
                validation=validation,
                feature_types=feature_types,
                model_spec=model_spec,
                target_col=target_col,
                cohort_col=cohort_col,
                derived_feature_artifacts=derived_artifacts,
            )
        )
    return prepared




def compute_metrics(
    validation_frame: pd.DataFrame,
    p_retention: np.ndarray,
    *,
    target_col: str = TARGET_COL,
    policy_id_col: str = POLICY_ID_COL,
    policy_inception_col: str = POLICY_INCEPTION_COL,
    anchor_term_start_col: str = ANCHOR_TERM_START_COL,
    top_fraction: float = 0.10,
) -> dict[str, float]:
    y_retention = pd.to_numeric(
        validation_frame[target_col], errors="raise"
    ).astype(int).to_numpy()
    if set(np.unique(y_retention)) != {0, 1}:
        raise ValueError("Validation target must contain both 0 and 1.")
    probabilities = np.asarray(p_retention, dtype=float)
    if len(probabilities) != len(y_retention):
        raise ValueError("Probability length does not match validation rows.")
    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
    y_churn = 1 - y_retention
    p_churn = 1.0 - probabilities
    churn_rate = float(y_churn.mean())


    ranking = pd.DataFrame(
        {
            "_p_churn": p_churn,
            "_y_churn": y_churn,
            "_stable_index": np.arange(len(validation_frame)),
        },
        index=validation_frame.index,
    )
    sort_columns = ["_p_churn"]
    ascending = [False]
    if policy_id_col in validation_frame:
        ranking["_policy_id"] = validation_frame[policy_id_col].astype(str).to_numpy()
        sort_columns.append("_policy_id")
        ascending.append(True)
    if policy_inception_col in validation_frame:
        ranking["_inception"] = (
            pd.to_datetime(validation_frame[policy_inception_col], errors="coerce")
            .astype(str)
            .to_numpy()
        )
        sort_columns.append("_inception")
        ascending.append(True)
    if anchor_term_start_col in validation_frame:
        ranking["_anchor_term_start"] = (
            pd.to_datetime(
                validation_frame[anchor_term_start_col], errors="coerce"
            )
            .astype(str)
            .to_numpy()
        )
        sort_columns.append("_anchor_term_start")
        ascending.append(True)
    sort_columns.append("_stable_index")
    ascending.append(True)
    ranked = ranking.sort_values(sort_columns, ascending=ascending, kind="mergesort")
    top_n = max(1, int(math.ceil(len(ranked) * top_fraction)))
    top = ranked.iloc[:top_n]
    churn_precision = float(top["_y_churn"].mean())
    total_churn = float(ranked["_y_churn"].sum())
    churn_capture = float(top["_y_churn"].sum() / max(total_churn, 1.0))
    lift = float(churn_precision / churn_rate) if churn_rate > 0 else np.nan


    return {
        "retention_roc_auc": float(roc_auc_score(y_retention, probabilities)),
        "churn_average_precision": float(
            average_precision_score(y_churn, p_churn)
        ),
        "churn_prevalence_baseline": churn_rate,
        "log_loss": float(log_loss(y_retention, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_retention, probabilities)),
        "actual_retention_rate": float(y_retention.mean()),
        "mean_predicted_retention": float(probabilities.mean()),
        "calibration_gap": float(probabilities.mean() - y_retention.mean()),
        "top10_lift": lift,
        "churn_precision_10pct": churn_precision,
        "churn_capture_10pct": churn_capture,
        "top_group_rows": int(top_n),
    }




def _fit_single_feature_transformer(
    *,
    train_series: pd.Series,
    validation_series: pd.Series,
    configured_type: str,
    rare_min_frequency: float | int | None,
    discrete_categories: Sequence[str] | None = None,
) -> FeatureBlock:
    raw_dtype = str(train_series.dtype)
    missing_rate = float(train_series.isna().mean())
    n_unique = int(train_series.dropna().nunique())
    sample_values = ";".join(
        train_series.dropna().astype(str).drop_duplicates().head(10).tolist()
    )


    if configured_type == "discrete":
        train_values = _categorical_array(train_series)
        validation_values = _categorical_array(validation_series)
        encoder = _make_one_hot_encoder(
            rare_min_frequency,
            categories=discrete_categories,
        )
        transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="__MISSING__"),
                ),
                ("onehot", encoder),
            ]
        )
    else:
        train_values = (
            pd.to_numeric(train_series, errors="coerce")
            .to_numpy(dtype=np.float64, na_value=np.nan)
            .reshape(-1, 1)
        )
        validation_values = (
            pd.to_numeric(validation_series, errors="coerce")
            .to_numpy(dtype=np.float64, na_value=np.nan)
            .reshape(-1, 1)
        )
        transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False)),
            ]
        )


    train_matrix = transformer.fit_transform(train_values)
    validation_matrix = transformer.transform(validation_values)
    train_matrix = sparse.csr_matrix(train_matrix, dtype=np.float32)
    validation_matrix = sparse.csr_matrix(validation_matrix, dtype=np.float32)
    if n_unique > 1 and train_matrix.shape[1] == 0:
        raise RuntimeError(
            "A nonconstant feature produced zero transformed columns."
        )
    if train_matrix.shape[1] != validation_matrix.shape[1]:
        raise RuntimeError("Train/validation transformed widths do not match.")
    return FeatureBlock(
        train_matrix=train_matrix,
        validation_matrix=validation_matrix,
        transformed_columns=int(train_matrix.shape[1]),
        raw_dtype=raw_dtype,
        configured_type=configured_type,
        missing_rate_train=missing_rate,
        n_unique_train=n_unique,
        sample_values=sample_values,
    )




def _categorical_array(series: pd.Series) -> np.ndarray:
    values = series.astype("string").astype(object).to_numpy(copy=True)
    values[pd.isna(values)] = np.nan
    return values.reshape(-1, 1)




def _make_one_hot_encoder(
    min_frequency: float | int | None,
    *,
    categories: Sequence[str] | None = None,
) -> OneHotEncoder:
    base_kwargs: dict[str, Any] = {
        "handle_unknown": "ignore",
        "dtype": np.float32,
    }
    if min_frequency is not None:
        base_kwargs["min_frequency"] = min_frequency
    if categories is not None:
        normalized = [str(value) for value in categories]
        if not normalized or len(normalized) != len(set(normalized)):
            raise ValueError("One-hot categories must be nonempty and unique.")
        base_kwargs["categories"] = [np.asarray(normalized, dtype=object)]
    try:
        return OneHotEncoder(sparse_output=True, **base_kwargs)
    except TypeError:
        # scikit-learn <1.2 uses `sparse`. Preserve min_frequency so a truly
        # unsupported old runtime fails instead of changing model semantics.
        return OneHotEncoder(sparse=True, **base_kwargs)




def load_project_population_and_folds(
    raw: pd.DataFrame,
    *,
    rolling_validation_folds: int,
    validation_months: int,
    test_months: int,
    min_train_months: int,
    strict_dataset_contract: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]], np.ndarray, dict[str, Any], Any]:
    """Use the included retention population and outer-split contract."""


    import sys


    project_root = Path.cwd().resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


    try:
        from dataclasses import replace
        from func.retention_population import (
            ModelingDataConfig,
            prepare_population,
            rolling_time_splits,
        )
        from params import (
            ACTIVE_AT_MONTH_1_COL,
            ACTIVE_AT_MONTH_13_COL,
            ANCHOR_TENURE_COL,
            ANCHOR_TERM_END_COL,
            ANCHOR_TERM_START_COL,
            BUSINESS_SCOPE,
            COHORT_DATE_COL,
            EXPOSURE_END_COL,
            EXPOSURE_START_COL,
            EXPOSURE_YEAR_COL,
            LABEL_AVAILABILITY_BUFFER_DAYS,
            LATEST_ANCHOR_TERM_START_DATE_EXCLUSIVE,
            POLICY_ID_COL,
            POLICY_INCEPTION_COL,
            PROFILE_HASH,
            PROFILE_SLUG,
            RETAINED_LABEL,
            TARGET,
            TARGET_CHECK_COL,
            TENURE_COL,
            TENURE_GROUP_DOMAIN,
            TENURE_GROUPS,
            TERM_END_COL,
            TERM_START_COL,
            TRUSTED_DATA_AS_OF_DATE,
        )
    except ImportError as exc:
        raise ImportError(
            "Run these scripts from the project root so "
            "func.retention_population is importable."
        ) from exc


    runtime_contract = ModelingDataConfig(
        target_col=TARGET,
        policy_id_col=POLICY_ID_COL,
        policy_inception_col=POLICY_INCEPTION_COL,
        anchor_term_start_col=ANCHOR_TERM_START_COL,
        anchor_term_end_col=ANCHOR_TERM_END_COL,
        anchor_tenure_col=ANCHOR_TENURE_COL,
        active_at_month1_col=ACTIVE_AT_MONTH_1_COL,
        active_at_month13_col=ACTIVE_AT_MONTH_13_COL,
        cohort_date_col=COHORT_DATE_COL,
        target_check_col=TARGET_CHECK_COL,
        term_start_col=TERM_START_COL,
        term_end_col=TERM_END_COL,
        exposure_start_col=EXPOSURE_START_COL,
        exposure_end_col=EXPOSURE_END_COL,
        exposure_year_col=EXPOSURE_YEAR_COL,
        tenure_col=TENURE_COL,
        business_scope=BUSINESS_SCOPE,
        tenure_group_domain=tuple(TENURE_GROUP_DOMAIN),
        tenure_groups=tuple(TENURE_GROUPS),
        profile_slug=PROFILE_SLUG,
        profile_hash=PROFILE_HASH,
        trusted_data_as_of_date=TRUSTED_DATA_AS_OF_DATE,
        latest_anchor_term_start_date_exclusive=(
            LATEST_ANCHOR_TERM_START_DATE_EXCLUSIVE
        ),
        retained_label=RETAINED_LABEL,
        label_availability_buffer_days=LABEL_AVAILABILITY_BUFFER_DAYS,
    )
    config = replace(
        runtime_contract,
        rolling_validation_folds=rolling_validation_folds,
        validation_months=validation_months,
        test_months=test_months,
        min_train_months=min_train_months,
        strict_dataset_contract=strict_dataset_contract,
    )
    prepared, population_audit = prepare_population(raw, config)
    ensure_cohort_calendar_year(
        prepared,
        source_feature=config.cohort_date_col,
    )
    prepared = prepared.sort_values(
        [config.cohort_date_col, *config.record_key_cols],
        kind="stable",
        na_position="first",
    ).reset_index(drop=True)
    folds, _, test_idx, split_summary = rolling_time_splits(prepared, config)
    split_summary = dict(split_summary)
    split_summary["population_audit"] = population_audit
    split_summary["derived_feature_contracts"] = {
        "cohort_calendar_year": cohort_calendar_year_contract()
    }
    return prepared, folds, test_idx, split_summary, config
