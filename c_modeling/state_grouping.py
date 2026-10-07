"""Fold-local optimal categorical binning for property-state retention.


The grouping is supervised feature engineering: it uses retention outcomes to
learn a state-to-group mapping.  Consequently, callers must fit it on one
model-training partition and only *apply* the resulting mapping to validation,
test, or scoring rows.  The public API deliberately keeps fitting and applying
separate so a validation frame can never be passed to the fitter by accident.


Texas is a business-defined singleton and is excluded from every non-Texas
grouping and group-count comparison. Non-Texas states are ordered by their
marginal empirical-Bayes retention behavior and divided by exact contiguous
dynamic programming. Candidate K is business-governed, then selected using
paired later-cohort comparisons and the one-standard-error rule. Adjacent
non-Texas bins must satisfy a configured minimum difference in their weighted
raw retention rates; the production default is one percentage point. The
cleaned 39-state population is the accepted and mapped state contract. A
chronological training fold may legitimately contain no history for a rare
contract state. Such a state is excluded from ordering, partitioning, and K
selection, then assigned after selection to the existing retention group
nearest the fold-training non-TX prior. Geography is not an input, a
constraint, or a tie-breaker. Adjacent-window temporal stability and in-sample
behavior capture are diagnostics, not eligibility gates.
Every internal subtraining window enforces the same Month-13 label-availability
embargo used by the outer modeling folds. Missing, malformed, or out-of-contract
state values fail closed rather than being silently encoded.
"""


from __future__ import annotations


from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence


import numpy as np
import pandas as pd




ALGORITHM_ID = "marginal_binary_retention_optimal_binning"


# Exported names are the routing contract used by the modeling layer.
RAW_STATE_COLUMN = "PROPERTY_STATE"
STATE_GROUP_COLUMN = "PROPERTY_STATE_RETENTION_GROUP"
TARGET_COLUMN = "RETENTION_13_1_IND"
TENURE_GROUP_COLUMN = "TENURE_YEARS_GRP"
TARGET_CHECK_COLUMN = "MONTH_13_EVALUATION_DATE"


TX_GROUP_LABEL = "TX"
NON_TX_GROUP_LABELS = tuple(f"G{index:02d}" for index in range(1, 11))
CLEANED_STATE_CODES = (
    "AL", "AR", "AZ", "CO", "CT", "GA", "IA", "ID", "IL", "IN",
    "KS", "KY", "LA", "MA", "MD", "MI", "MN", "MO", "MS", "MT",
    "NC", "ND", "NE", "NJ", "NM", "NV", "NY", "OH", "OK", "OR",
    "PA", "SD", "TN", "TX", "UT", "VA", "WA", "WI", "WY",
)
CLEANED_STATE_SET = frozenset(CLEANED_STATE_CODES)
_INTERNAL_NON_TX_PRIOR = "__INTERNAL_NON_TX_PRIOR__"
_STATE_PATTERN = re.compile(r"^[A-Z]{2}$")
_MISSING_TOKENS = {"", "<NA>", "NAN", "NONE", "NULL"}




class StateGroupingError(RuntimeError):
    """Base exception for an invalid or infeasible state grouping."""




@dataclass(frozen=True)
class StateGroupingSpec:
    """Immutable statistical and business rules for one grouping fit."""


    state_col: str = RAW_STATE_COLUMN
    target_col: str = TARGET_COLUMN
    tenure_col: str | None = TENURE_GROUP_COLUMN
    cohort_col: str | None = "MONTH_1_EVALUATION_DATE"
    target_check_col: str | None = TARGET_CHECK_COLUMN
    label_availability_buffer_days: int = 0
    output_col: str = STATE_GROUP_COLUMN
    forced_state: str = "TX"
    forced_label: str = TX_GROUP_LABEL
    non_tx_group_labels: tuple[str, ...] = NON_TX_GROUP_LABELS
    # K is the number of non-TX groups. TX is added as a separate group.
    min_non_tx_groups: int = 3
    max_non_tx_groups: int = 10
    # Absolute retention-probability difference. 0.01 means one percentage
    # point, not a 1% relative difference and not a 1% row sample.
    min_adjacent_non_tx_group_retention_rate_gap: float = 0.01
    # Volume thresholds are audit flags only. They never affect state order,
    # partition boundaries, K eligibility, or K selection.
    small_volume_review_fraction: float = 0.01
    large_volume_review_fraction: float = 0.50
    # Audited adjacent-temporal-fold targets. They never make K eligible or
    # ineligible and cannot force a larger group count.
    min_mapping_stability: float = 0.60
    min_behavior_rank_stability: float = 0.90
    temporal_validation_folds: int = 9
    temporal_validation_months: int = 3
    min_temporal_training_months: int = 24
    min_temporal_validation_rows: int = 100
    behavior_score_tie_tolerance: float = 1e-6
    fallback_bic_tolerance: float = 2.0
    prior_strength: float = 2_000.0


    def __post_init__(self) -> None:
        object.__setattr__(
            self, "non_tx_group_labels", tuple(self.non_tx_group_labels)
        )
        if self.min_non_tx_groups < 3:
            raise ValueError(
                "Production candidate min_non_tx_groups must be at least 3; "
                "K=1 is reserved for the internal null reference."
            )
        if self.max_non_tx_groups < self.min_non_tx_groups:
            raise ValueError(
                "max_non_tx_groups must be at least min_non_tx_groups."
            )
        if self.max_non_tx_groups > 10:
            raise ValueError("max_non_tx_groups cannot exceed 10.")
        if len(self.non_tx_group_labels) < self.max_non_tx_groups:
            raise ValueError(
                "non_tx_group_labels must contain at least max_non_tx_groups "
                "unique labels."
            )
        labels = (self.forced_label, *self.non_tx_group_labels)
        if len(labels) != len(set(labels)):
            raise ValueError("State-group labels must be unique.")
        if self.prior_strength <= 0:
            raise ValueError("prior_strength must be positive.")
        if not 0 <= self.min_adjacent_non_tx_group_retention_rate_gap < 1:
            raise ValueError(
                "min_adjacent_non_tx_group_retention_rate_gap must be in [0, 1)."
            )
        if not 0 <= self.small_volume_review_fraction < 1:
            raise ValueError("small_volume_review_fraction must be in [0, 1).")
        if not 0 < self.large_volume_review_fraction <= 1:
            raise ValueError("large_volume_review_fraction must be in (0, 1].")
        if self.small_volume_review_fraction >= self.large_volume_review_fraction:
            raise ValueError(
                "small_volume_review_fraction must be below "
                "large_volume_review_fraction."
            )
        if not 0 <= self.min_mapping_stability <= 1:
            raise ValueError("min_mapping_stability must be between 0 and 1.")
        if not 0 <= self.min_behavior_rank_stability <= 1:
            raise ValueError(
                "min_behavior_rank_stability must be between 0 and 1."
            )
        if self.temporal_validation_folds < 2:
            raise ValueError("temporal_validation_folds must be at least 2.")
        if self.temporal_validation_months < 1:
            raise ValueError("temporal_validation_months must be positive.")
        if self.min_temporal_training_months < 1:
            raise ValueError("min_temporal_training_months must be positive.")
        if self.min_temporal_validation_rows < 1:
            raise ValueError("min_temporal_validation_rows must be positive.")
        if self.label_availability_buffer_days < 0:
            raise ValueError(
                "label_availability_buffer_days cannot be negative."
            )
        if not 0 <= self.behavior_score_tie_tolerance <= 1:
            raise ValueError(
                "behavior_score_tie_tolerance must be between 0 and 1."
            )
        if self.fallback_bic_tolerance < 0:
            raise ValueError("fallback_bic_tolerance cannot be negative.")
        forced = self.forced_state.strip().upper()
        if not _STATE_PATTERN.fullmatch(forced):
            raise ValueError("forced_state must be a two-letter state code.")
        if forced != "TX" or self.forced_label != TX_GROUP_LABEL:
            raise ValueError("The cleaned state contract requires TX -> TX.")


    @property
    def candidate_non_tx_groups(self) -> tuple[int, ...]:
        return tuple(
            range(self.min_non_tx_groups, self.max_non_tx_groups + 1)
        )


@dataclass
class FittedStateGrouper:
    """Serializable state mapping and its training-only audit evidence."""


    spec: StateGroupingSpec
    mapping: dict[str, str]
    behavior_columns: tuple[str, ...]
    behavior_stats: pd.DataFrame
    bin_summary: pd.DataFrame
    assignment_audit: dict[str, Any]
    fit_context: dict[str, Any] = field(default_factory=dict)


    @property
    def categories(self) -> tuple[str, ...]:
        """Dynamic encoding contract: TX followed by the selected G labels."""


        selected_k = int(
            self.assignment_audit.get(
                "selected_non_tx_group_count",
                len(set(self.mapping.values()).difference({self.spec.forced_label})),
            )
        )
        return (
            self.spec.forced_label,
            *self.spec.non_tx_group_labels[:selected_k],
        )


    def apply(
        self,
        scoring_frame: pd.DataFrame,
        *,
        scope: str = "scoring",
        artifact_dir: str | Path | None = None,
    ) -> tuple[pd.Series, dict[str, Any]]:
        return apply_state_grouper(
            self,
            scoring_frame,
            scope=scope,
            artifact_dir=artifact_dir,
        )


    def payload(self) -> dict[str, Any]:
        mapping = {key: self.mapping[key] for key in sorted(self.mapping)}
        core = {
            "algorithm_id": ALGORITHM_ID,
            "spec": _jsonable(asdict(self.spec)),
            "mapping": mapping,
            "behavior_columns": list(self.behavior_columns),
            "behavior_stats": _jsonable(
                self.behavior_stats.to_dict(orient="records")
            ),
            "bin_summary": _jsonable(
                self.bin_summary.to_dict(orient="records")
            ),
            "assignment_audit": _jsonable(self.assignment_audit),
            "fit_context": _jsonable(self.fit_context),
        }
        core["mapping_sha256"] = _sha256_json(mapping)
        core["artifact_sha256"] = _sha256_json(core)
        return core


    def save(self, artifact_dir: str | Path) -> dict[str, str]:
        destination = Path(artifact_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        mapping_path = destination / "state_mapping.csv"
        bin_summary_path = destination / "state_bin_summary.csv"
        fitted_path = destination / "fitted_state_grouper.json"
        mapping_columns = [
            "state",
            "retention_group",
            "observed_in_training",
            "no_training_history",
            "assignment_method",
            "rows",
            "training_volume_share",
            "non_tx_training_volume_share",
            "small_state_volume_review",
            "large_state_volume_review",
            "raw_retention_rate",
            "raw_churn_rate",
            "eb_shrunk_retention_rate",
            "eb_shrunk_churn_rate",
            "eb_shrunk_log_odds",
            "forced_singleton",
            "included_in_grouping",
            "included_in_optimization",
            "behavior_order",
            "distance_to_group_centroid",
        ]
        available = [
            column for column in mapping_columns if column in self.behavior_stats
        ]
        _atomic_csv(mapping_path, self.behavior_stats.loc[:, available])
        _atomic_csv(bin_summary_path, self.bin_summary)
        _atomic_json(fitted_path, self.payload())
        return {
            "mapping": str(mapping_path),
            "bin_summary": str(bin_summary_path),
            "fitted_grouper": str(fitted_path),
        }


    @classmethod
    def load(cls, artifact_dir: str | Path) -> "FittedStateGrouper":
        source = Path(artifact_dir).expanduser().resolve()
        payload_path = source / "fitted_state_grouper.json"
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid fitted state-grouper artifact: {payload_path}"
            ) from exc
        if payload.get("algorithm_id") != ALGORITHM_ID:
            raise ValueError(
                "State-grouper algorithm identity mismatch: "
                f"{payload.get('algorithm_id')!r}."
            )
        expected = payload.get("artifact_sha256")
        unhashed = dict(payload)
        unhashed.pop("artifact_sha256", None)
        if expected != _sha256_json(unhashed):
            raise ValueError("Fitted state-grouper artifact hash mismatch.")
        mapping = {
            str(key): str(value) for key, value in payload.get("mapping", {}).items()
        }
        if payload.get("mapping_sha256") != _sha256_json(mapping):
            raise ValueError("Fitted state mapping hash mismatch.")
        if set(mapping) != CLEANED_STATE_SET:
            raise ValueError("Fitted state mapping does not cover the 39-state contract.")
        if mapping.get("TX") != TX_GROUP_LABEL or any(
            state != "TX" and label == TX_GROUP_LABEL
            for state, label in mapping.items()
        ):
            raise ValueError("Fitted state mapping violates the TX singleton contract.")
        behavior = pd.DataFrame(payload.get("behavior_stats", []))
        summary = pd.DataFrame(payload.get("bin_summary", []))
        if len(behavior) != len(CLEANED_STATE_CODES) or behavior.empty or summary.empty:
            raise ValueError("Fitted state-grouper payload is incomplete.")
        if "state" not in behavior or set(behavior["state"].astype(str)) != CLEANED_STATE_SET:
            raise ValueError("Fitted behavior evidence does not cover the state contract.")
        required_behavior_columns = {
            "rows",
            "retention_group",
            "observed_in_training",
            "no_training_history",
            "assignment_method",
            "included_in_optimization",
        }
        missing_behavior_columns = sorted(
            required_behavior_columns.difference(behavior.columns)
        )
        if missing_behavior_columns:
            raise ValueError(
                "Fitted behavior evidence is missing no-history audit columns: "
                f"{missing_behavior_columns}."
            )
        zero_row_states = set(
            behavior.loc[
                pd.to_numeric(behavior["rows"], errors="coerce").eq(0),
                "state",
            ].astype(str)
        )
        audited_prior_states = set(
            str(value)
            for value in payload.get("assignment_audit", {}).get(
                "prior_assigned_states", []
            )
        )
        if zero_row_states != audited_prior_states or "TX" in zero_row_states:
            raise ValueError(
                "Fitted zero-row state evidence does not match its prior-assignment audit."
            )
        prior_rows = behavior["state"].astype(str).isin(zero_row_states)
        if zero_row_states and (
            not behavior.loc[prior_rows, "no_training_history"].astype(bool).all()
            or behavior.loc[prior_rows, "observed_in_training"].astype(bool).any()
            or behavior.loc[prior_rows, "included_in_optimization"].astype(bool).any()
            or behavior.loc[prior_rows, "retention_group"].eq(TX_GROUP_LABEL).any()
        ):
            raise ValueError("Fitted prior-assigned state evidence is inconsistent.")
        fitted = cls(
            spec=StateGroupingSpec(**payload["spec"]),
            mapping=mapping,
            behavior_columns=tuple(payload.get("behavior_columns", ())),
            behavior_stats=behavior,
            bin_summary=summary,
            assignment_audit=dict(payload.get("assignment_audit", {})),
            fit_context=dict(payload.get("fit_context", {})),
        )
        if set(mapping.values()) != set(fitted.categories):
            raise ValueError("Fitted mapping labels do not match its category contract.")
        return fitted




def apply_state_grouper(
    fitted: FittedStateGrouper,
    scoring_frame: pd.DataFrame,
    *,
    scope: str = "scoring",
    artifact_dir: str | Path | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Apply a fitted mapping without reading a target column."""


    del artifact_dir  # Application is summarized by the caller; no duplicate file.
    spec = fitted.spec
    if spec.state_col not in scoring_frame:
        raise KeyError(f"Scoring frame is missing {spec.state_col}.")
    state = _normalize_state_series(scoring_frame[spec.state_col])
    missing = state.isna()
    malformed = state.notna() & ~state.str.fullmatch(_STATE_PATTERN.pattern, na=False)
    outside = state.notna() & ~malformed & ~state.isin(CLEANED_STATE_SET)
    if missing.any() or malformed.any() or outside.any():
        raise StateGroupingError(
            f"{scope} violates the cleaned state contract: "
            f"missing_rows={int(missing.sum())}; "
            f"malformed={sorted(state.loc[malformed].astype(str).unique())}; "
            f"outside_contract={sorted(state.loc[outside].astype(str).unique())}."
        )
    if set(fitted.mapping) != CLEANED_STATE_SET:
        missing_mapping = sorted(CLEANED_STATE_SET.difference(fitted.mapping))
        extra_mapping = sorted(set(fitted.mapping).difference(CLEANED_STATE_SET))
        raise StateGroupingError(
            "Fitted state mapping violates the cleaned contract: "
            f"missing={missing_mapping}; extra={extra_mapping}."
        )
    result = state.map(fitted.mapping).astype("string")
    result.name = spec.output_col
    if result.isna().any() or not result.isin(fitted.categories).all():
        raise AssertionError("State grouping produced an invalid output category.")


    prior_assigned_states = frozenset(
        str(value)
        for value in fitted.assignment_audit.get("prior_assigned_states", [])
    )
    audit = {
        "algorithm_id": ALGORITHM_ID,
        "scope": str(scope),
        "rows": int(len(scoring_frame)),
        "non_tx_group_count": int(
            fitted.assignment_audit.get("selected_non_tx_group_count", 0)
        ),
        "known_group_count_including_tx": int(len(fitted.categories)),
        "encoded_category_count": int(len(fitted.categories)),
        "minimum_required_adjacent_non_tx_retention_rate_gap": float(
            spec.min_adjacent_non_tx_group_retention_rate_gap
        ),
        "fitted_minimum_adjacent_non_tx_retention_rate_gap": (
            fitted.assignment_audit.get(
                "minimum_selected_adjacent_non_tx_retention_rate_gap"
            )
        ),
        "known_state_rows": int(len(scoring_frame)),
        "tx_rows": int(state.eq(spec.forced_state.strip().upper()).sum()),
        "rows_from_states_without_fold_training_history": int(
            state.isin(prior_assigned_states).sum()
        ),
        "states_without_fold_training_history_present": sorted(
            set(state.loc[state.isin(prior_assigned_states)].astype(str))
        ),
        "mapped_state_contract_count": int(len(fitted.mapping)),
        "all_rows_mapped": True,
        "cleaned_state_contract_sha256": _sha256_json(CLEANED_STATE_CODES),
        "mapping_sha256": _sha256_json(fitted.mapping),
        "target_column_accessed": False,
    }
    return result, audit




def _bin_summary(
    behavior: pd.DataFrame,
    *,
    behavior_columns: Sequence[str],
    settings: StateGroupingSpec,
    selected_labels: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = float(behavior["rows"].sum())
    non_tx_total = float(
        behavior.loc[~behavior["forced_singleton"], "rows"].sum()
    )
    for label in (settings.forced_label, *selected_labels):
        part = behavior.loc[behavior["retention_group"].eq(label)].copy()
        if part.empty:
            raise AssertionError(f"Training mapping has an empty group: {label}")
        group_rows = float(part["rows"].sum())
        retained = float(part["retained"].sum())
        largest_state_row = part.sort_values(
            ["rows", "state"], ascending=[False, True], kind="stable"
        ).iloc[0]
        row: dict[str, Any] = {
            "retention_group": label,
            "forced_singleton": label == settings.forced_label,
            "state_count": int(len(part)),
            "states": ";".join(sorted(part["state"].astype(str).tolist())),
            "observed_state_count": int(part["observed_in_training"].sum()),
            "prior_assigned_state_count": int(part["no_training_history"].sum()),
            "prior_assigned_states": ";".join(
                sorted(
                    part.loc[
                        part["no_training_history"], "state"
                    ].astype(str).tolist()
                )
            ),
            "rows": int(group_rows),
            "largest_state": str(largest_state_row["state"]),
            "largest_state_rows": int(largest_state_row["rows"]),
            "largest_state_share_within_group": float(
                largest_state_row["rows"] / group_rows
            ),
            "training_volume_share": group_rows / total,
            "non_tx_volume_share": (
                np.nan
                if label == settings.forced_label
                else group_rows / non_tx_total
            ),
            "raw_retention_rate": retained / group_rows,
            "small_group_review": (
                False
                if label == settings.forced_label
                else group_rows / non_tx_total
                < settings.small_volume_review_fraction - 1e-12
            ),
            "large_group_review": (
                False
                if label == settings.forced_label
                else group_rows / non_tx_total
                > settings.large_volume_review_fraction + 1e-12
            ),
        }
        observed_part = part.loc[part["observed_in_training"]]
        if observed_part.empty:
            raise AssertionError(
                f"Training mapping has no observed state in group {label}."
            )
        for column in behavior_columns:
            row[f"centroid_{column}"] = float(
                np.average(
                    observed_part[column], weights=observed_part["rows"]
                )
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["previous_non_tx_retention_group"] = ""
    summary["raw_retention_rate_gap_from_previous_non_tx_group"] = np.nan
    summary["required_minimum_adjacent_non_tx_retention_rate_gap"] = np.nan
    summary["adjacent_non_tx_retention_rate_gap_constraint_applies"] = False
    summary["adjacent_non_tx_retention_rate_gap_constraint_pass"] = pd.Series(
        pd.NA, index=summary.index, dtype="boolean"
    )
    threshold = settings.min_adjacent_non_tx_group_retention_rate_gap
    for position, label in enumerate(selected_labels):
        index = summary.index[summary["retention_group"].eq(label)]
        if len(index) != 1:
            raise AssertionError(f"State-bin summary is missing group {label}.")
        row_index = int(index[0])
        summary.loc[
            row_index,
            "required_minimum_adjacent_non_tx_retention_rate_gap",
        ] = threshold
        if position == 0:
            continue
        previous_label = selected_labels[position - 1]
        previous_rate = float(
            summary.loc[
                summary["retention_group"].eq(previous_label),
                "raw_retention_rate",
            ].iloc[0]
        )
        current_rate = float(summary.loc[row_index, "raw_retention_rate"])
        gap = current_rate - previous_rate
        summary.loc[
            row_index, "previous_non_tx_retention_group"
        ] = previous_label
        summary.loc[
            row_index,
            "raw_retention_rate_gap_from_previous_non_tx_group",
        ] = gap
        summary.loc[
            row_index,
            "adjacent_non_tx_retention_rate_gap_constraint_applies",
        ] = True
        summary.loc[
            row_index,
            "adjacent_non_tx_retention_rate_gap_constraint_pass",
        ] = bool(threshold <= 0 or gap + 1e-12 >= threshold)
    return summary




def _normalize_state_series(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip().str.upper()
    return result.mask(result.isin(_MISSING_TOKENS), pd.NA)




def _normalize_tenure(
    frame: pd.DataFrame, settings: StateGroupingSpec
) -> pd.Series:
    if settings.tenure_col is None or settings.tenure_col not in frame:
        return pd.Series("ALL", index=frame.index, dtype="string")
    result = frame[settings.tenure_col].astype("string").str.strip().str.upper()
    return result.mask(result.isin(_MISSING_TOKENS), "MISSING").fillna("MISSING")




def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-8, 1 - 1e-8)
    return np.log(clipped / (1.0 - clipped))




def _fit_population_hash(frame: pd.DataFrame) -> str:
    canonical = frame.loc[
        :, ["state", "target", "tenure", "cohort", "target_check"]
    ].copy()
    canonical["state"] = canonical["state"].fillna("__MISSING__")
    canonical["tenure"] = canonical["tenure"].fillna("__MISSING__")
    canonical["cohort"] = (
        pd.to_datetime(canonical["cohort"], errors="coerce")
        .dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        .fillna("__MISSING__")
    )
    canonical["target_check"] = (
        pd.to_datetime(canonical["target_check"], errors="coerce")
        .dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        .fillna("__MISSING__")
    )
    row_hashes = pd.util.hash_pandas_object(canonical, index=False).to_numpy(
        dtype=np.uint64
    )
    row_hashes.sort()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()




def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, np.bool_):
        return bool(value)
    return value




def _sha256_json(payload: Any) -> str:
    rendered = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()




def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(_jsonable(payload), indent=2, sort_keys=True, default=str) + "\n"
    _atomic_bytes(path, text.encode("utf-8"))




def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    _atomic_bytes(path, content)




def _atomic_bytes(path: Path, content: bytes) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)




# ---------------------------------------------------------------------------
# Retention-ordered optimal categorical binning
# ---------------------------------------------------------------------------




@dataclass(frozen=True)
class _OptimalPartition:
    k: int
    assignment: tuple[int, ...]
    intervals: tuple[tuple[int, int], ...]
    nll: float
    saturated_nll: float
    deviance: float
    volume_cv: float
    group_rows: tuple[int, ...]
    group_state_counts: tuple[int, ...]
    group_states: tuple[tuple[str, ...], ...]
    group_retention_rates: tuple[float, ...]
    adjacent_retention_rate_gaps: tuple[float, ...]
    minimum_adjacent_retention_rate_gap: float | None
    required_minimum_adjacent_retention_rate_gap: float
    adjacent_retention_rate_gap_constraint_satisfied: bool




def fit_state_grouper(
    training_frame: pd.DataFrame,
    spec: StateGroupingSpec | None = None,
    *,
    context: Mapping[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
) -> FittedStateGrouper:
    """Fit a TX-isolated, fold-local optimal state grouping.


    K is selected using expanding chronological subvalidation whenever the
    training partition contains enough cohort months. Only when that evidence
    is unavailable is a deterministic, explicitly audited BIC fallback used.
    """


    settings = spec or StateGroupingSpec()
    required = {settings.state_col, settings.target_col}
    missing = sorted(required.difference(training_frame.columns))
    if missing:
        raise KeyError(f"State-group training frame is missing columns: {missing}")
    if training_frame.empty:
        raise ValueError("State-group training frame is empty.")


    fit = _prepare_grouping_frame(training_frame, settings)
    valid = fit.loc[fit["valid_state"]].copy()
    observed_valid_states = set(valid["state"].astype(str))
    missing_contract_states = sorted(CLEANED_STATE_SET.difference(observed_valid_states))
    unexpected_states = sorted(observed_valid_states.difference(CLEANED_STATE_SET))
    if unexpected_states:
        raise StateGroupingError(
            "Grouping training contains states outside the cleaned contract: "
            f"extra={unexpected_states}."
        )
    forced_state = settings.forced_state.strip().upper()
    if not valid["state"].eq(forced_state).any():
        raise ValueError(
            f"Forced singleton {forced_state} has no rows in grouping training."
        )
    if valid["target"].nunique() != 2:
        raise ValueError("State grouping training target must contain both 0 and 1.")


    behavior, behavior_columns = _state_behavior_stats(valid, settings)
    non_tx = _ordered_non_tx_behavior(behavior, settings)
    if non_tx.empty:
        raise ValueError("No non-Texas states are available for grouping.")


    full_partitions: dict[int, _OptimalPartition] = {}
    full_failures: dict[int, str] = {}
    for k in settings.candidate_non_tx_groups:
        try:
            full_partitions[k] = _optimal_binary_partition(non_tx, k, settings)
        except StateGroupingError as exc:
            full_failures[k] = str(exc)
    if not full_partitions:
        raise StateGroupingError(
            "No feasible retention-ordered non-TX partition. Requested-K "
            "failures: "
            + json.dumps(
                {str(key): value for key, value in sorted(full_failures.items())},
                sort_keys=True,
            )
        )
    # K=1 is diagnostic only. It defines the amount of between-state binary
    # deviance available to capture, but can never be selected as the model's
    # state representation.
    deviance_reference = _optimal_binary_partition(non_tx, 1, settings)


    selected_k, k_audit = _select_group_count(
        valid,
        settings,
        full_partitions=full_partitions,
        full_failures=full_failures,
        deviance_reference=deviance_reference,
    )
    partition = full_partitions[selected_k]
    selected_labels = settings.non_tx_group_labels[:selected_k]
    mapping = {forced_state: settings.forced_label}
    for position, state_code in enumerate(non_tx["state"].astype(str)):
        mapping[state_code] = selected_labels[partition.assignment[position]]
    mapping, behavior, prior_assignment = _complete_mapping_from_training_prior(
        mapping=mapping,
        behavior=behavior,
        missing_contract_states=missing_contract_states,
        selected_labels=selected_labels,
        partition=partition,
        settings=settings,
    )
    if set(mapping) != CLEANED_STATE_SET:
        raise AssertionError("Final state mapping does not cover all 39 cleaned states.")


    behavior = behavior.sort_values("state", kind="stable").reset_index(drop=True)
    behavior["retention_group"] = behavior["state"].map(mapping)
    behavior["forced_singleton"] = behavior["state"].eq(forced_state)
    behavior["observed_in_training"] = behavior["rows"].gt(0)
    behavior["no_training_history"] = ~behavior["observed_in_training"]
    behavior["assignment_method"] = np.select(
        [
            behavior["forced_singleton"],
            behavior["no_training_history"],
        ],
        [
            "FORCED_TX_SINGLETON",
            "TRAINING_NON_TX_PRIOR_NEAREST_GROUP_LOG_ODDS",
        ],
        default="OBSERVED_RETENTION_OPTIMAL_BIN",
    )
    behavior["included_in_grouping"] = ~behavior["forced_singleton"]
    behavior["included_in_optimization"] = (
        behavior["included_in_grouping"]
        & behavior["observed_in_training"]
    )
    behavior["training_volume_share"] = behavior["rows"] / float(behavior["rows"].sum())
    non_tx_rows = float(
        behavior.loc[behavior["included_in_optimization"], "rows"].sum()
    )
    behavior["non_tx_training_volume_share"] = np.where(
        behavior["included_in_optimization"],
        behavior["rows"] / non_tx_rows,
        np.nan,
    )
    behavior["small_state_volume_review"] = (
        behavior["included_in_optimization"]
        & behavior["non_tx_training_volume_share"].lt(
            settings.small_volume_review_fraction
        )
    )
    behavior["large_state_volume_review"] = (
        behavior["included_in_optimization"]
        & behavior["non_tx_training_volume_share"].gt(
            settings.large_volume_review_fraction
        )
    )
    order_lookup = {
        state: position + 1
        for position, state in enumerate(non_tx["state"].astype(str))
    }
    behavior["behavior_order"] = behavior["state"].map(order_lookup)
    centroids: dict[str, float] = {}
    grouped_behavior = behavior.loc[behavior["included_in_optimization"]]
    for label, part in grouped_behavior.groupby("retention_group", observed=True):
        centroids[str(label)] = float(
            np.average(part["behavior_order_score"], weights=part["rows"])
        )
    distances: list[float] = []
    for row in behavior.itertuples(index=False):
        if bool(row.forced_singleton):
            distances.append(0.0)
        else:
            distances.append(
                abs(
                    float(row.behavior_order_score)
                    - float(centroids[str(row.retention_group)])
                )
            )
    behavior["distance_to_group_centroid"] = distances


    bin_summary = _bin_summary(
        behavior,
        behavior_columns=behavior_columns,
        settings=settings,
        selected_labels=selected_labels,
    )
    non_tx_summary = bin_summary.loc[
        ~bin_summary["forced_singleton"].astype(bool)
    ].copy()
    exported_rates = tuple(
        non_tx_summary["raw_retention_rate"].to_numpy(dtype=float)
    )
    if not np.allclose(
        exported_rates,
        partition.group_retention_rates,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("Partition and exported group rates disagree.")
    applied_gap_rows = non_tx_summary.loc[
        non_tx_summary[
            "adjacent_non_tx_retention_rate_gap_constraint_applies"
        ].astype(bool)
    ]
    if not applied_gap_rows[
        "adjacent_non_tx_retention_rate_gap_constraint_pass"
    ].astype(bool).all():
        raise AssertionError("Exported state bins violate the adjacent-rate gap.")
    shares = np.asarray(partition.group_rows, dtype=float) / sum(partition.group_rows)
    final_metrics = {
        "binary_negative_log_likelihood": float(partition.nll),
        "binomial_deviance_from_state_saturated_model": float(partition.deviance),
        "volume_cv": float(partition.volume_cv),
        "volume_share_rmse_from_equal": float(
            np.sqrt(np.mean((shares - 1.0 / selected_k) ** 2))
        ),
        "max_abs_volume_share_deviation_from_equal": float(
            np.max(np.abs(shares - 1.0 / selected_k))
        ),
        "non_tx_group_rows": list(partition.group_rows),
        "non_tx_group_volume_shares": [float(value) for value in shares],
        "non_tx_group_raw_retention_rates": list(
            partition.group_retention_rates
        ),
        "adjacent_non_tx_group_raw_retention_rate_gaps": list(
            partition.adjacent_retention_rate_gaps
        ),
        "minimum_selected_adjacent_non_tx_retention_rate_gap": (
            partition.minimum_adjacent_retention_rate_gap
        ),
        "minimum_required_adjacent_non_tx_retention_rate_gap": float(
            partition.required_minimum_adjacent_retention_rate_gap
        ),
        "adjacent_non_tx_retention_rate_gap_constraint_satisfied": bool(
            partition.adjacent_retention_rate_gap_constraint_satisfied
        ),
        "adjacent_retention_rate_gap_unit": (
            "ABSOLUTE_PROBABILITY; 0.01_EQUALS_ONE_PERCENTAGE_POINT"
        ),
        "maximum_non_tx_group_volume_share": float(np.max(shares)),
        "small_volume_review_fraction": float(
            settings.small_volume_review_fraction
        ),
        "small_group_review_count": int(
            np.sum(shares < settings.small_volume_review_fraction - 1e-12)
        ),
        "large_volume_review_fraction": float(
            settings.large_volume_review_fraction
        ),
        "large_group_review_count": int(
            np.sum(shares > settings.large_volume_review_fraction + 1e-12)
        ),
        "non_tx_group_state_counts": list(partition.group_state_counts),
        "non_tx_group_mapped_state_counts": [
            int(value)
            for value in non_tx_summary["state_count"].to_numpy(dtype=int)
        ],
        "prior_assigned_state_count": int(
            prior_assignment["prior_assigned_state_count"]
        ),
        "volume_balance_used_in_partition_or_k_selection": False,
        "state_order": non_tx["state"].astype(str).tolist(),
    }
    fit_payload = {
        "algorithm_id": ALGORITHM_ID,
        "fit_rows": int(len(training_frame)),
        "valid_state_rows": int(fit["valid_state"].sum()),
        "cleaned_state_contract": list(CLEANED_STATE_CODES),
        "cleaned_state_contract_count": len(CLEANED_STATE_CODES),
        "cleaned_state_contract_sha256": _sha256_json(CLEANED_STATE_CODES),
        "mapped_state_count": len(mapping),
        "all_cleaned_states_mapped": set(mapping) == CLEANED_STATE_SET,
        "state_volume_source": "FULL_UNSAMPLED_FOLD_TRAINING_ROWS",
        "static_state_row_counts_used_for_fitting": False,
        "cohort_date_available_rows": int(fit["cohort"].notna().sum()),
        "target_check_col": settings.target_check_col,
        "target_check_date_available_rows": int(
            fit["target_check"].notna().sum()
        ),
        "label_availability_buffer_days": int(
            settings.label_availability_buffer_days
        ),
        "internal_temporal_label_embargo_rule": (
            "subtraining target_check is strictly earlier than validation "
            "start minus the configured buffer"
        ),
        "observed_valid_state_count": int(valid["state"].nunique()),
        "observed_training_states": sorted(observed_valid_states),
        "non_tx_state_count": int(len(non_tx)),
        "observed_non_tx_state_count": int(len(non_tx)),
        "missing_training_states": list(missing_contract_states),
        "missing_training_state_count": int(len(missing_contract_states)),
        "forced_state": forced_state,
        "forced_state_rows": int(valid["state"].eq(forced_state).sum()),
        "behavior_definition": (
            "training-only marginal observed state retention; empirical-Bayes "
            "rates establish a credible behavior order and exact dynamic "
            "programming minimizes binomial NLL over contiguous behavior bins "
            "whose adjacent pooled raw retention rates differ by at least "
            f"{settings.min_adjacent_non_tx_group_retention_rate_gap:.6f} "
            "absolute probability; no risk adjustment for premium, credit, "
            "tenure, or other predictors; TX excluded from the non-TX reference, "
            "partition, gap constraint, and K selection; contract states with "
            "no fold-training rows are mapped only after K selection from the "
            "training non-TX prior and never enter optimization"
        ),
        "partition_objective": (
            "exact contiguous dynamic programming after marginal empirical-Bayes "
            "retention ordering; binomial NLL and the configured "
            f"{settings.min_adjacent_non_tx_group_retention_rate_gap:.6f} "
            "adjacent raw-retention gap are the only partition criteria"
        ),
        "behavior_columns": list(behavior_columns),
        "fit_population_sha256": _fit_population_hash(fit),
        "spec_sha256": _sha256_json(asdict(settings)),
        "mapping_sha256": _sha256_json(mapping),
        "volume_balance_used_in_state_order": False,
        "volume_balance_used_in_partition_boundaries": False,
        "volume_balance_used_in_k_eligibility_or_selection": False,
        "row_counts_used_for_behavior_estimation_and_credibility": True,
        "k1_null_reference_has_no_adjacent_retention_gap_constraint": True,
        "group_labels_ordered_by_retention": "G01_LOWEST_TO_GK_HIGHEST",
        "tx_excluded_from_non_tx_ordering_partition_and_k_selection": True,
        "tx_excluded_from_optimization": True,
        "hard_minimum_adjacent_non_tx_retention_rate_gap_applied": bool(
            settings.min_adjacent_non_tx_group_retention_rate_gap > 0
        ),
        "minimum_required_adjacent_non_tx_retention_rate_gap": float(
            settings.min_adjacent_non_tx_group_retention_rate_gap
        ),
        "minimum_selected_adjacent_non_tx_retention_rate_gap": (
            partition.minimum_adjacent_retention_rate_gap
        ),
        "selected_adjacent_non_tx_retention_rate_gaps": list(
            partition.adjacent_retention_rate_gaps
        ),
        "selected_non_tx_group_raw_retention_rates": list(
            partition.group_retention_rates
        ),
        "adjacent_non_tx_retention_rate_gap_constraint_satisfied": bool(
            partition.adjacent_retention_rate_gap_constraint_satisfied
        ),
        "adjacent_retention_rate_gap_definition": (
            "SIGNED_DIFFERENCE_IN_POOLED_WEIGHTED_RAW_TRAINING_RETENTION_RATE_"
            "FROM_PREVIOUS_NON_TX_GROUP; 0.01_EQUALS_ONE_PERCENTAGE_POINT"
        ),
        "tx_exempt_from_adjacent_retention_rate_gap_constraint": True,
        "small_group_review_non_tx_volume_share": float(
            settings.small_volume_review_fraction
        ),
        "large_group_review_non_tx_volume_share": float(
            settings.large_volume_review_fraction
        ),
        "volume_role": (
            "fold-local row counts weight retention estimates, empirical-Bayes "
            "credibility, likelihood, and validation loss; no target group size, "
            "balance score, volume constraint, or volume tie-break is used"
        ),
        "candidate_k_requested": list(settings.candidate_non_tx_groups),
        "candidate_k_evaluated": sorted(full_partitions),
        "candidate_k_full_fit_failures": {
            str(key): value for key, value in full_failures.items()
        },
        "selected_non_tx_group_count": int(selected_k),
        "selected_total_modeled_group_count_including_tx": int(selected_k + 1),
        "selected_known_group_count_including_tx": int(selected_k + 1),
        "non_tx_group_count": int(selected_k),
        "total_modeled_group_count": int(selected_k + 1),
        "group_count_semantics": (
            "K counts non-TX groups; total modeled groups adds the forced TX "
            "singleton"
        ),
        "selected_categories": [
            settings.forced_label,
            *selected_labels,
        ],
        "encoded_category_count": int(selected_k + 1),
        "selected_encoded_category_count": int(selected_k + 1),
        "selected_volume_cv": float(partition.volume_cv),
        "selected_binomial_deviance": float(partition.deviance),
        "selected_negative_log_likelihood": float(partition.nll),
        "selected_non_tx_group_rows": list(partition.group_rows),
        "selected_non_tx_group_volume_shares": [float(value) for value in shares],
        "selected_non_tx_group_state_counts": list(partition.group_state_counts),
        "selected_non_tx_group_mapped_state_counts": [
            int(value)
            for value in non_tx_summary["state_count"].to_numpy(dtype=int)
        ],
        "maximum_selected_non_tx_group_volume_share": float(np.max(shares)),
        "selected_small_group_review_count": int(
            np.sum(shares < settings.small_volume_review_fraction - 1e-12)
        ),
        "selected_large_group_review_count": int(
            np.sum(shares > settings.large_volume_review_fraction + 1e-12)
        ),
        "partition_selection_reason": (
            "Selected the exact retention-behavior-optimal partition; volume "
            "was not a constraint, score, or tie-breaker."
        ),
        "final_behavior_and_volume_metrics": final_metrics,
        **prior_assignment,
        **k_audit,
    }
    fitted = FittedStateGrouper(
        spec=settings,
        mapping={key: mapping[key] for key in sorted(mapping)},
        behavior_columns=tuple(behavior_columns),
        behavior_stats=behavior,
        bin_summary=bin_summary,
        assignment_audit=fit_payload,
        fit_context=dict(context or {}),
    )
    if artifact_dir is not None:
        fitted.save(artifact_dir)
    return fitted




def _prepare_grouping_frame(
    frame: pd.DataFrame, settings: StateGroupingSpec
) -> pd.DataFrame:
    state = _normalize_state_series(frame[settings.state_col])
    missing_state = state.isna()
    malformed_state = state.notna() & ~state.str.fullmatch(
        _STATE_PATTERN.pattern, na=False
    )
    outside_contract = (
        state.notna() & ~malformed_state & ~state.isin(CLEANED_STATE_SET)
    )
    if missing_state.any() or malformed_state.any() or outside_contract.any():
        raise StateGroupingError(
            "Grouping training violates the cleaned state contract: "
            f"missing_rows={int(missing_state.sum())}; "
            f"malformed={sorted(state.loc[malformed_state].astype(str).unique())}; "
            f"outside_contract={sorted(state.loc[outside_contract].astype(str).unique())}."
        )
    target = pd.to_numeric(frame[settings.target_col], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError(
            f"{settings.target_col} must be complete and binary in grouping training."
        )
    cohort = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    if settings.cohort_col is not None and settings.cohort_col in frame:
        cohort = pd.to_datetime(frame[settings.cohort_col], errors="coerce")
    target_check = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    if (
        settings.target_check_col is not None
        and settings.target_check_col in frame
    ):
        target_check = pd.to_datetime(
            frame[settings.target_check_col], errors="coerce"
        )
    valid_state = pd.Series(True, index=frame.index, dtype=bool)
    return pd.DataFrame(
        {
            "state": state,
            "target": target.astype("int8"),
            "tenure": _normalize_tenure(frame, settings),
            "cohort": cohort,
            "target_check": target_check,
            "valid_state": valid_state,
        },
        index=frame.index,
    )




def _state_behavior_stats(
    valid: pd.DataFrame, settings: StateGroupingSpec
) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate marginal binary behavior and construct a credible EB order.


    The business definition is deliberately marginal: a state's observed
    retention experience is not residualized for premium, credit, tenure, or
    any other predictor. Empirical-Bayes shrinkage only stabilizes small-state
    estimates; it does not change the meaning to a risk-adjusted state effect.
    """


    forced_state = settings.forced_state.strip().upper()
    states = sorted(valid["state"].astype(str).unique().tolist())
    reference = valid.loc[~valid["state"].eq(forced_state)].copy()
    if reference.empty:
        raise ValueError("Non-Texas reference population is empty.")
    global_rate = _clip_probability(float(reference["target"].mean()))
    overall = (
        valid.groupby("state", observed=True)["target"]
        .agg(rows="size", retained="sum")
        .reindex(states, fill_value=0)
    )
    overall["churned"] = overall["rows"] - overall["retained"]
    overall["raw_retention_rate"] = overall["retained"] / overall["rows"]
    overall["raw_churn_rate"] = 1.0 - overall["raw_retention_rate"]
    overall["eb_shrunk_retention_rate"] = (
        overall["retained"] + settings.prior_strength * global_rate
    ) / (overall["rows"] + settings.prior_strength)
    overall["eb_shrunk_churn_rate"] = 1.0 - overall["eb_shrunk_retention_rate"]
    overall["eb_shrunk_log_odds"] = _logit(
        overall["eb_shrunk_retention_rate"].to_numpy(dtype=float)
    ) - _logit(np.asarray([global_rate]))[0]
    overall["behavior_order_score"] = overall["eb_shrunk_log_odds"]
    behavior_columns = ["eb_shrunk_log_odds"]
    output = overall.reset_index().rename(columns={"index": "state"})
    for column in ("rows", "retained", "churned"):
        output[column] = output[column].astype("int64")
    return output, behavior_columns




def _ordered_non_tx_behavior(
    behavior: pd.DataFrame, settings: StateGroupingSpec
) -> pd.DataFrame:
    forced_state = settings.forced_state.strip().upper()
    return (
        behavior.loc[~behavior["state"].eq(forced_state)]
        .sort_values(["behavior_order_score", "state"], kind="stable")
        .reset_index(drop=True)
    )




def _complete_mapping_from_training_prior(
    *,
    mapping: Mapping[str, str],
    behavior: pd.DataFrame,
    missing_contract_states: Sequence[str],
    selected_labels: Sequence[str],
    partition: _OptimalPartition,
    settings: StateGroupingSpec,
) -> tuple[dict[str, str], pd.DataFrame, dict[str, Any]]:
    """Map zero-history contract states without consulting future outcomes.


    A state with no fold-training rows has an empirical-Bayes posterior equal
    to the fold-training non-TX prior. After K and the observed-state partition
    are fixed, project that neutral prior onto the nearest existing group in
    retention-log-odds space. This completion cannot affect behavior order,
    partition boundaries, adjacent-gap checks, or K selection.
    """


    completed = {str(key): str(value) for key, value in mapping.items()}
    forced_state = settings.forced_state.strip().upper()
    missing_non_tx = tuple(
        sorted(
            state
            for state in missing_contract_states
            if str(state) != forced_state
        )
    )
    if forced_state in missing_contract_states:
        raise ValueError(
            f"Forced singleton {forced_state} has no rows in grouping training."
        )


    observed_non_tx = behavior.loc[~behavior["state"].eq(forced_state)].copy()
    non_tx_rows = float(observed_non_tx["rows"].sum())
    if non_tx_rows <= 0:
        raise ValueError("Non-Texas reference population is empty.")
    prior_rate = _clip_probability(
        float(observed_non_tx["retained"].sum() / non_tx_rows)
    )
    prior_log_odds = float(_logit(np.asarray([prior_rate]))[0])


    if len(selected_labels) != len(partition.group_retention_rates):
        raise AssertionError("Selected labels and partition groups disagree.")
    group_log_odds = tuple(
        float(_logit(np.asarray([_clip_probability(rate)]))[0])
        for rate in partition.group_retention_rates
    )
    distances = tuple(
        abs(value - prior_log_odds) for value in group_log_odds
    )
    nearest_position = min(
        range(len(selected_labels)),
        key=lambda position: (distances[position], position),
    )
    nearest_label = str(selected_labels[nearest_position])
    nearest_rate = float(partition.group_retention_rates[nearest_position])


    for state in missing_non_tx:
        completed[state] = nearest_label


    if missing_non_tx:
        placeholders = pd.DataFrame(
            {
                "state": list(missing_non_tx),
                "rows": np.zeros(len(missing_non_tx), dtype="int64"),
                "retained": np.zeros(len(missing_non_tx), dtype="int64"),
                "churned": np.zeros(len(missing_non_tx), dtype="int64"),
                "raw_retention_rate": np.full(len(missing_non_tx), np.nan),
                "raw_churn_rate": np.full(len(missing_non_tx), np.nan),
                "eb_shrunk_retention_rate": np.full(
                    len(missing_non_tx), prior_rate
                ),
                "eb_shrunk_churn_rate": np.full(
                    len(missing_non_tx), 1.0 - prior_rate
                ),
                "eb_shrunk_log_odds": np.zeros(len(missing_non_tx)),
                "behavior_order_score": np.zeros(len(missing_non_tx)),
            }
        )
        behavior = pd.concat([behavior, placeholders], ignore_index=True)


    audit = {
        "prior_assigned_states": list(missing_non_tx),
        "prior_assigned_state_count": int(len(missing_non_tx)),
        "prior_assignment_reference_non_tx_retention_rate": prior_rate,
        "prior_assignment_group_label": nearest_label,
        "prior_assignment_group_retention_rate": nearest_rate,
        "prior_assignment_log_odds_distance": float(
            distances[nearest_position]
        ),
        "prior_assignment_rule": (
            "AFTER_K_SELECTION_MAP_ZERO_HISTORY_NON_TX_STATES_TO_EXISTING_"
            "GROUP_NEAREST_FOLD_TRAINING_NON_TX_PRIOR_IN_LOG_ODDS"
        ),
        "prior_assignment_uses_validation_or_future_outcomes": False,
        "prior_assignment_uses_static_state_counts": False,
        "prior_assigned_states_used_in_order_partition_or_k_selection": False,
    }
    return completed, behavior, audit




def _optimal_binary_partition(
    ordered: pd.DataFrame,
    k: int,
    settings: StateGroupingSpec,
) -> _OptimalPartition:
    """Find the exact fixed-K partition after ordering states by retention.


    Candidate groups are contiguous on the empirical-Bayes retention order.
    Binomial likelihood and the adjacent retention-rate gap are the only
    boundary criteria. Geography and volume balance are not inputs,
    constraints, scores, or tie-breakers. K=1 is a diagnostic null reference.
    """


    state_count = len(ordered)
    if k < 1 or k > state_count:
        raise StateGroupingError(f"K={k} is invalid for {state_count} non-TX states.")
    states = ordered["state"].astype(str).tolist()
    rows = ordered["rows"].to_numpy(dtype=float)
    retained = ordered["retained"].to_numpy(dtype=float)
    total_rows = float(rows.sum())


    prefix_rows = np.concatenate(([0.0], np.cumsum(rows)))
    prefix_retained = np.concatenate(([0.0], np.cumsum(retained)))
    interval_nll = np.full((state_count, state_count + 1), np.inf)
    interval_rate = np.full((state_count, state_count + 1), np.nan)
    for start in range(state_count):
        for end in range(start + 1, state_count + 1):
            n = prefix_rows[end] - prefix_rows[start]
            retained_count = prefix_retained[end] - prefix_retained[start]
            raw_rate = retained_count / n
            interval_rate[start, end] = raw_rate
            probability = _clip_probability(raw_rate)
            interval_nll[start, end] = -(
                retained_count * math.log(probability)
                + (n - retained_count) * math.log(1.0 - probability)
            )


    saturated_nll = _state_saturated_nll(rows, retained)
    required_gap = float(
        settings.min_adjacent_non_tx_group_retention_rate_gap
    )
    gap_tolerance = 1e-12


    shape = (k + 1, state_count + 1, state_count + 1)
    cost = np.full(shape, np.inf)
    predecessor = np.full(shape, -1, dtype=int)
    for end in range(1, state_count + 1):
        cost[1, 0, end] = interval_nll[0, end]


    def better(candidate: float, current: float, candidate_order: int, current_order: int) -> bool:
        if not math.isfinite(current):
            return True
        tolerance = 32.0 * np.finfo(float).eps * max(
            1.0, abs(candidate), abs(current)
        )
        if candidate < current - tolerance:
            return True
        return abs(candidate - current) <= tolerance and (
            current_order < 0 or candidate_order < current_order
        )


    for groups in range(2, k + 1):
        for start in range(groups - 1, state_count):
            for end in range(start + 1, state_count + 1):
                current_rate = interval_rate[start, end]
                for previous_start in range(groups - 2, start):
                    prior = cost[groups - 1, previous_start, start]
                    if not np.isfinite(prior):
                        continue
                    previous_rate = interval_rate[previous_start, start]
                    if (
                        required_gap > 0
                        and current_rate - previous_rate + gap_tolerance < required_gap
                    ):
                        continue
                    candidate = prior + interval_nll[start, end]
                    if better(
                        candidate,
                        cost[groups, start, end],
                        previous_start,
                        predecessor[groups, start, end],
                    ):
                        cost[groups, start, end] = candidate
                        predecessor[groups, start, end] = previous_start


    best_start = -1
    nll = np.inf
    for start in range(k - 1, state_count):
        candidate = cost[k, start, state_count]
        if better(candidate, nll, start, best_start):
            best_start = start
            nll = candidate
    if best_start < 0 or not np.isfinite(nll):
        raise StateGroupingError(
            f"No retention-ordered K={k} partition satisfies the required "
            f"{required_gap:.6f} signed adjacent raw-retention rate gap."
        )


    intervals: list[tuple[int, int]] = []
    start = best_start
    end = state_count
    for groups in range(k, 0, -1):
        intervals.append((start, end))
        if groups == 1:
            if start != 0:
                raise AssertionError(
                    "Optimal partition predecessor chain does not start at 0."
                )
            break
        previous_start = int(predecessor[groups, start, end])
        if previous_start < 0:
            raise AssertionError(
                "Optimal partition predecessor chain is incomplete."
            )
        end = start
        start = previous_start
    selected_intervals = tuple(reversed(intervals))


    assignment = np.empty(state_count, dtype=int)
    group_rows: list[int] = []
    group_retention_rates: list[float] = []
    group_states: list[tuple[str, ...]] = []
    for group, (start, end) in enumerate(selected_intervals):
        assignment[start:end] = group
        interval_rows = prefix_rows[end] - prefix_rows[start]
        interval_retained = prefix_retained[end] - prefix_retained[start]
        group_rows.append(int(round(interval_rows)))
        group_retention_rates.append(float(interval_retained / interval_rows))
        group_states.append(tuple(states[start:end]))
    adjacent_retention_rate_gaps = [
        group_retention_rates[position]
        - group_retention_rates[position - 1]
        for position in range(1, len(group_retention_rates))
    ]
    minimum_observed_gap = (
        None
        if not adjacent_retention_rate_gaps
        else float(min(adjacent_retention_rate_gaps))
    )
    gap_constraint_satisfied = bool(
        required_gap <= 0
        or not adjacent_retention_rate_gaps
        or minimum_observed_gap + gap_tolerance >= required_gap
    )
    if not gap_constraint_satisfied:
        raise AssertionError(
            "Optimal partition violated the minimum adjacent-retention gap."
        )
    volume_shares = np.asarray(group_rows, dtype=float) / total_rows
    return _OptimalPartition(
        k=k,
        assignment=tuple(int(value) for value in assignment),
        intervals=selected_intervals,
        nll=nll,
        saturated_nll=saturated_nll,
        deviance=max(0.0, 2.0 * (nll - saturated_nll)),
        volume_cv=(
            float(np.std(volume_shares) / np.mean(volume_shares)) if k > 1 else 0.0
        ),
        group_rows=tuple(group_rows),
        group_state_counts=tuple(len(group) for group in group_states),
        group_states=tuple(group_states),
        group_retention_rates=tuple(group_retention_rates),
        adjacent_retention_rate_gaps=tuple(adjacent_retention_rate_gaps),
        minimum_adjacent_retention_rate_gap=minimum_observed_gap,
        required_minimum_adjacent_retention_rate_gap=required_gap,
        adjacent_retention_rate_gap_constraint_satisfied=(
            gap_constraint_satisfied
        ),
    )




def _select_group_count(
    valid: pd.DataFrame,
    settings: StateGroupingSpec,
    *,
    full_partitions: Mapping[int, _OptimalPartition],
    full_failures: Mapping[int, str],
    deviance_reference: _OptimalPartition,
) -> tuple[int, dict[str, Any]]:
    available_between_state_deviance = max(
        0.0,
        2.0 * (deviance_reference.nll - deviance_reference.saturated_nll),
    )


    def absolute_capture(partition: _OptimalPartition) -> float:
        if available_between_state_deviance <= 1e-12:
            return 1.0
        return float(
            np.clip(
                1.0 - partition.deviance / available_between_state_deviance,
                0.0,
                1.0,
            )
        )


    best_evaluated_deviance_capture = max(
        absolute_capture(partition) for partition in full_partitions.values()
    )


    def full_diagnostics(k: int) -> dict[str, Any]:
        partition = full_partitions[k]
        capture = absolute_capture(partition)
        relative_capture = (
            1.0
            if best_evaluated_deviance_capture <= 1e-12
            else float(capture / best_evaluated_deviance_capture)
        )
        return {
            "full_training_negative_log_likelihood": float(partition.nll),
            "full_training_binomial_deviance_unexplained": float(
                partition.deviance
            ),
            "full_training_between_state_deviance_available": float(
                available_between_state_deviance
            ),
            "full_training_between_state_deviance_captured": float(
                available_between_state_deviance - partition.deviance
            ),
            "full_training_deviance_capture": capture,
            "best_evaluated_deviance_capture": float(
                best_evaluated_deviance_capture
            ),
            "deviance_capture_ratio_to_best_evaluated": float(
                relative_capture
            ),
            "deviance_capture_is_diagnostic_only": True,
            "full_training_non_tx_group_raw_retention_rates": list(
                partition.group_retention_rates
            ),
            "full_training_adjacent_non_tx_retention_rate_gaps": list(
                partition.adjacent_retention_rate_gaps
            ),
            "full_training_minimum_adjacent_non_tx_retention_rate_gap": (
                partition.minimum_adjacent_retention_rate_gap
            ),
            "required_minimum_adjacent_non_tx_retention_rate_gap": float(
                partition.required_minimum_adjacent_retention_rate_gap
            ),
            "meets_minimum_adjacent_non_tx_retention_rate_gap": bool(
                partition.adjacent_retention_rate_gap_constraint_satisfied
            ),
            "full_training_volume_cv": float(partition.volume_cv),
            "maximum_group_volume_share": float(
                max(partition.group_rows) / sum(partition.group_rows)
            ),
            "small_group_review_count": int(
                sum(
                    row / sum(partition.group_rows)
                    < settings.small_volume_review_fraction - 1e-12
                    for row in partition.group_rows
                )
            ),
            "large_group_review_count": int(
                sum(
                    row / sum(partition.group_rows)
                    > settings.large_volume_review_fraction + 1e-12
                    for row in partition.group_rows
                )
            ),
            "volume_role": "AUDIT_ONLY",
        }


    temporal_failure_warning: str | None = None
    splits, split_audit = _temporal_grouping_splits(valid, settings)
    if len(splits) >= 2:
        # State behavior is the only full-row aggregation required per temporal
        # subtraining window. Cache it once and reuse it across every K and all
        # stability calculations.
        temporal_behavior_cache = [
            _state_behavior_stats(train, settings)[0]
            for train, _, _ in splits
        ]
        scores = _temporal_group_count_scores(
            splits,
            settings,
            sorted(full_partitions),
            behavior_cache=temporal_behavior_cache,
        )
        for row in scores:
            k = int(row["k"])
            if k not in full_partitions:
                continue
            row.update(full_diagnostics(k))
            stability = _mapping_stability(
                splits,
                settings,
                k,
                behavior_cache=temporal_behavior_cache,
            )
            row["temporal_mapping_stability"] = stability
            coassignment = stability.get(
                "mean_adjacent_coassignment_jaccard"
            )
            rank_stability = stability.get(
                "mean_adjacent_behavior_rank_correlation"
            )
            row["adjacent_mean_mapping_stability"] = coassignment
            row["adjacent_mean_behavior_rank_stability"] = rank_stability
            row["meets_adjacent_mean_mapping_stability_target"] = bool(
                coassignment is not None
                and float(coassignment) + 1e-12
                >= settings.min_mapping_stability
            )
            row["meets_adjacent_mean_behavior_rank_stability_target"] = bool(
                rank_stability is not None
                and float(rank_stability) + 1e-12
                >= settings.min_behavior_rank_stability
            )
            row["meets_both_adjacent_mean_stability_targets"] = bool(
                row["meets_adjacent_mean_mapping_stability_target"]
                and row[
                    "meets_adjacent_mean_behavior_rank_stability_target"
                ]
            )
            target_ratios = []
            if coassignment is not None:
                target_ratios.append(
                    1.0
                    if settings.min_mapping_stability <= 1e-12
                    else float(coassignment)
                    / settings.min_mapping_stability
                )
            if rank_stability is not None:
                target_ratios.append(
                    1.0
                    if settings.min_behavior_rank_stability <= 1e-12
                    else float(rank_stability)
                    / settings.min_behavior_rank_stability
                )
            row["robust_adjacent_stability_target_attainment"] = (
                None if len(target_ratios) != 2 else float(min(target_ratios))
            )
        eligible = [row for row in scores if row["eligible_all_temporal_folds"]]
        if eligible:
            global_best_k = _add_paired_temporal_comparisons(
                eligible,
                tolerance=settings.behavior_score_tie_tolerance,
                prefix="global_",
            )
            behavior_pool = eligible
            paired_pool = [
                row
                for row in behavior_pool
                if row["global_paired_one_se_eligible"]
            ]
            if not paired_pool:
                raise AssertionError(
                    "The paired temporal comparison must retain its best candidate."
                )


            target_pool = [
                row
                for row in paired_pool
                if row["meets_both_adjacent_mean_stability_targets"]
            ]
            stability_target_met = bool(target_pool)
            stability_status = (
                "STABILITY_TARGET_MET_BY_AT_LEAST_ONE_ONE_SE_CANDIDATE"
                if stability_target_met
                else "STABILITY_TARGET_NOT_MET_BY_ONE_SE_CANDIDATES"
            )
            fallback_reason = None


            selected = min(
                paired_pool,
                key=lambda row: (
                    row["k"],
                    row["mean_validation_log_loss"],
                ),
            )
            selected_k = int(selected["k"])
            selection_status = "SELECTED_PAIRED_OOT_ONE_SE_PARSIMONIOUS"
            selection_warning = []
            if not selected["meets_both_adjacent_mean_stability_targets"]:
                selection_warning.append(
                    "The selected K missed at least one temporal stability "
                    "target; stability is diagnostic and did not alter K."
                )
            if int(selected["large_group_review_count"]) > 0:
                selection_warning.append(
                    "At least one selected non-TX group exceeds 50% volume and "
                    "requires business review."
                )
            return selected_k, {
                "selection_method": (
                    "paired_expanding_chronological_one_se_then_parsimony"
                ),
                "selection_status": selection_status,
                "selection_evidence_tier": "PAIRED_OOT_ONE_STANDARD_ERROR",
                "selection_warnings": selection_warning,
                "fallback_used": False,
                "fallback_reason": fallback_reason,
                "hard_constraints_relaxed": False,
                "paired_oot_comparison_used": True,
                "stability_role": (
                    "AUDITED_WARNING_ONLY_NOT_SELECTION_OR_ELIGIBILITY_GATE"
                ),
                "behavior_rank_stability_role": (
                    "GLOBAL_TEMPORAL_DRIFT_DIAGNOSTIC_ONLY"
                ),
                "selection_priority": [
                    "HARD FEASIBILITY: require at least a configured "
                    f"{settings.min_adjacent_non_tx_group_retention_rate_gap:.6f} "
                    "increase between every adjacent non-TX training bin",
                    "TEMPORAL VALIDATION: compare K with paired same-window "
                    "log-loss differences and retain the one-SE set",
                    "PARSIMONY: choose the smallest remaining non-TX K",
                    "AUDIT: report behavior capture, stability, and volume "
                    "warnings without changing K",
                ],
                "k_selection_reason": (
                    "Every candidate first had to satisfy the hard configured "
                    f"{settings.min_adjacent_non_tx_group_retention_rate_gap:.6f} "
                    "non-TX retention-gap rule in the full fit and every temporal "
                    "subfit. All remaining K were compared on identical later "
                    "cohorts. The smallest K inside the paired one-standard-error "
                    "set was selected by parsimony. "
                    "Full-training deviance capture and temporal stability are "
                    "diagnostics only. State/group volumes are audit-only. TX "
                    "is fixed and is not scored in the K comparison."
                ),
                "k_selection_scoring_population": "NON_TX_VALIDATION_ROWS_ONLY",
                "temporal_subvalidation_used": True,
                "temporal_split_audit": split_audit,
                "temporal_k_selection_metrics": scores,
                "global_best_temporal_k": int(global_best_k),
                "feasible_k": sorted(
                    int(row["k"]) for row in behavior_pool
                ),
                "paired_one_se_eligible_k": sorted(
                    int(row["k"]) for row in paired_pool
                ),
                "stability_target_status": stability_status,
                "stability_target_is_hard_gate": False,
                "stability_target_fallback_used": False,
                "stability_target_fallback_reason": fallback_reason,
                "stability_target_basis": (
                    "MEAN_OF_ADJACENT_TEMPORAL_FOLD_PAIRS"
                ),
                "stability_target_met_k": sorted(
                    int(row["k"]) for row in target_pool
                ),
                "stability_preferred_k": sorted(
                    int(row["k"]) for row in target_pool
                ),
                "robust_stability_definition": (
                    "minimum of adjacent-mean coassignment-Jaccard divided by "
                    "its target and adjacent-mean behavior-rank correlation "
                    "divided by its target"
                ),
                "volume_balance_used_in_k_selection": False,
                "deviance_capture_reference_k": 1,
                "available_between_state_deviance": float(
                    available_between_state_deviance
                ),
                "best_evaluated_deviance_capture": float(
                    best_evaluated_deviance_capture
                ),
                "deviance_capture_role": "AUDIT_ONLY",
                "target_mean_adjacent_mapping_stability": float(
                    settings.min_mapping_stability
                ),
                "target_mean_adjacent_behavior_rank_stability": float(
                    settings.min_behavior_rank_stability
                ),
                "selected_k_meets_both_stability_targets": bool(
                    selected["meets_both_adjacent_mean_stability_targets"]
                ),
                "selected_k_robust_stability_target_attainment": selected[
                    "robust_adjacent_stability_target_attainment"
                ],
                "selected_k_mapping_stability": selected[
                    "temporal_mapping_stability"
                ],
                "k_selection_metrics": scores,
            }
        temporal_failures = {
            str(row["k"]): str(
                row.get("failure_reason") or "INELIGIBLE_UNSPECIFIED"
            )
            for row in scores
        }
        temporal_failure_warning = (
            f"Created {len(splits)} label-embargoed temporal folds, but no K "
            "was eligible across every fold; using the audited BIC fallback. "
            f"Per-K failures: {json.dumps(temporal_failures, sort_keys=True)}"
        )


    fallback_rows: list[dict[str, Any]] = []
    for k, partition in sorted(full_partitions.items()):
        diagnostics = full_diagnostics(int(k))
        diagnostics["full_training_bic"] = float(
            2.0 * partition.nll
            + int(k) * math.log(max(1, sum(partition.group_rows)))
        )
        fallback_rows.append({"k": int(k), "eligible": True, **diagnostics})
    best_bic = min(float(row["full_training_bic"]) for row in fallback_rows)
    behavior_pool = [
        row
        for row in fallback_rows
        if float(row["full_training_bic"])
        <= best_bic + settings.fallback_bic_tolerance + 1e-12
    ]
    selected = min(
        behavior_pool,
        key=lambda row: row["k"],
    )
    selected_k = int(selected["k"])
    fallback_warnings = [
        temporal_failure_warning
        or "Fewer than two valid chronological subvalidation folds were "
        "available; paired OOT and temporal stability evidence were not available."
    ]
    if int(selected["large_group_review_count"]) > 0:
        fallback_warnings.append(
            "At least one selected non-TX group exceeds 50% volume and "
            "requires business review."
        )
    return selected_k, {
        "selection_method": "full_training_bic_tolerance_fallback",
        "selection_status": "SELECTED_BIC_FALLBACK_TEMPORAL_UNAVAILABLE",
        "selection_evidence_tier": "FULL_TRAINING_BIC_NO_USABLE_PAIRED_OOT",
        "selection_warnings": fallback_warnings,
        "fallback_used": True,
        "fallback_reason": (
            temporal_failure_warning
            or "Fewer than two valid chronological subvalidation folds were available."
        ),
        "hard_constraints_relaxed": False,
        "paired_oot_comparison_used": False,
        "stability_role": (
            "AUDITED_WARNING_ONLY_NOT_SELECTION_OR_ELIGIBILITY_GATE"
        ),
        "behavior_rank_stability_role": (
            "GLOBAL_TEMPORAL_DRIFT_DIAGNOSTIC_ONLY"
        ),
        "selection_priority": [
            "HARD FEASIBILITY: require at least a configured "
            f"{settings.min_adjacent_non_tx_group_retention_rate_gap:.6f} "
            "increase between every adjacent non-TX training bin",
            "FALLBACK ONLY: BIC within configured tolerance of best",
            "PARSIMONY: smallest remaining non-TX K",
            "AUDIT: report state/group volumes without changing K",
        ],
        "k_selection_reason": (
            "Every candidate first had to satisfy the hard configured "
            f"{settings.min_adjacent_non_tx_group_retention_rate_gap:.6f} "
            "non-TX retention-gap rule in the full fit. Usable paired "
            "chronological evidence was unavailable. The smallest candidate "
            "within 2 BIC points of the best was selected by parsimony; "
            "volume was not used. This is not OOT evidence."
        ),
        "k_selection_scoring_population": (
            "FULL_TRAINING_NON_TX_ROWS_BIC_FALLBACK"
        ),
        "temporal_subvalidation_used": False,
        "temporal_split_audit": split_audit,
        "temporal_k_selection_metrics": [],
        "feasible_k": sorted(int(k) for k in full_partitions),
        "bic_tolerance_eligible_k": sorted(
            int(row["k"]) for row in behavior_pool
        ),
        "paired_one_se_eligible_k": [],
        "stability_preferred_k": [],
        "stability_target_status": (
            "STABILITY_NOT_EVALUATED_TEMPORAL_SUBVALIDATION_UNAVAILABLE"
        ),
        "stability_target_is_hard_gate": False,
        "stability_target_fallback_used": False,
        "stability_target_fallback_reason": None,
        "stability_target_basis": "MEAN_OF_ADJACENT_TEMPORAL_FOLD_PAIRS",
        "target_mean_adjacent_mapping_stability": float(
            settings.min_mapping_stability
        ),
        "target_mean_adjacent_behavior_rank_stability": float(
            settings.min_behavior_rank_stability
        ),
        "deviance_capture_reference_k": 1,
        "available_between_state_deviance": float(
            available_between_state_deviance
        ),
        "best_evaluated_deviance_capture": float(
            best_evaluated_deviance_capture
        ),
        "deviance_capture_role": "AUDIT_ONLY",
        "fallback_best_bic": float(best_bic),
        "fallback_bic_tolerance": float(settings.fallback_bic_tolerance),
        "volume_balance_used_in_k_selection": False,
        "k_selection_metrics": fallback_rows,
    }




def _add_paired_temporal_comparisons(
    candidates: Sequence[dict[str, Any]],
    *,
    tolerance: float,
    prefix: str,
) -> int:
    """Compare candidate K on identical validation windows.


    Absolute fold log loss moves with calendar-period base-rate drift. Pairing
    each candidate with the best candidate on the same windows removes that
    common drift before calculating uncertainty.
    """


    if not candidates:
        raise ValueError("Paired temporal comparison requires candidates.")
    best = min(
        candidates,
        key=lambda row: (row["mean_validation_log_loss"], row["k"]),
    )
    best_by_fold = {
        int(metric["fold"]): float(metric["validation_log_loss"])
        for metric in best["fold_metrics"]
    }
    for row in candidates:
        deltas: list[float] = []
        delta_rows: list[dict[str, Any]] = []
        for metric in row["fold_metrics"]:
            fold = int(metric["fold"])
            if fold not in best_by_fold:
                raise AssertionError(
                    "Temporal K candidates were not scored on identical folds."
                )
            delta = float(metric["validation_log_loss"]) - best_by_fold[fold]
            deltas.append(delta)
            delta_rows.append(
                {
                    "fold": fold,
                    "candidate_log_loss": float(metric["validation_log_loss"]),
                    "reference_best_k_log_loss": float(best_by_fold[fold]),
                    "paired_log_loss_delta": float(delta),
                }
            )
        if len(deltas) != len(best_by_fold):
            raise AssertionError(
                "Temporal K candidates were not scored on identical folds."
            )
        delta_array = np.asarray(deltas, dtype=float)
        mean_delta = float(np.mean(delta_array))
        standard_error = (
            0.0
            if len(delta_array) < 2
            else float(
                np.std(delta_array, ddof=1) / math.sqrt(len(delta_array))
            )
        )
        one_se_limit = max(float(tolerance), standard_error)
        row[f"{prefix}paired_reference_best_k"] = int(best["k"])
        row[f"{prefix}paired_log_loss_differences"] = delta_rows
        row[f"{prefix}mean_paired_log_loss_delta"] = mean_delta
        row[f"{prefix}standard_error_paired_log_loss_delta"] = standard_error
        row[f"{prefix}paired_one_se_limit"] = float(one_se_limit)
        row[f"{prefix}paired_one_se_eligible"] = bool(
            mean_delta <= one_se_limit + 1e-12
        )
    return int(best["k"])




def _temporal_grouping_splits(
    valid: pd.DataFrame, settings: StateGroupingSpec
) -> tuple[list[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]], dict[str, Any]]:
    cohort_available = valid["cohort"].notna()
    target_check_available = valid["target_check"].notna()
    dated = valid.loc[cohort_available & target_check_available].copy()
    if dated.empty:
        return [], {
            "status": "UNAVAILABLE_NO_VALID_COHORT_AND_TARGET_CHECK_DATES",
            "cohort_date_available_rows": int(cohort_available.sum()),
            "target_check_date_available_rows": int(
                target_check_available.sum()
            ),
            "target_check_col": settings.target_check_col,
            "requested_folds": settings.temporal_validation_folds,
            "created_folds": 0,
        }
    dated["cohort_month"] = dated["cohort"].dt.to_period("M")
    months = sorted(dated["cohort_month"].unique())
    required_months = (
        settings.min_temporal_training_months
        + settings.temporal_validation_folds
        * settings.temporal_validation_months
    )
    if len(months) < required_months:
        return [], {
            "status": "UNAVAILABLE_TOO_FEW_COHORT_MONTHS",
            "observed_months": len(months),
            "required_months": required_months,
            "cohort_date_available_rows": int(cohort_available.sum()),
            "target_check_date_available_rows": int(
                target_check_available.sum()
            ),
            "rows_with_both_temporal_dates": int(len(dated)),
            "target_check_col": settings.target_check_col,
            "label_availability_buffer_days": int(
                settings.label_availability_buffer_days
            ),
            "requested_folds": settings.temporal_validation_folds,
            "created_folds": 0,
        }
    first_validation = len(months) - (
        settings.temporal_validation_folds * settings.temporal_validation_months
    )
    splits: list[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]] = []
    skipped: list[dict[str, Any]] = []
    for fold in range(settings.temporal_validation_folds):
        start = first_validation + fold * settings.temporal_validation_months
        validation_months = months[start : start + settings.temporal_validation_months]
        training_months = months[:start]
        candidate_train = dated.loc[
            dated["cohort_month"].isin(training_months)
        ].copy()
        validation_start = validation_months[0].to_timestamp(how="start")
        label_cutoff = validation_start - pd.Timedelta(
            days=settings.label_availability_buffer_days
        )
        train = candidate_train.loc[
            candidate_train["target_check"].lt(label_cutoff)
        ].copy()
        validation = dated.loc[dated["cohort_month"].isin(validation_months)].copy()
        forced_state = settings.forced_state.strip().upper()
        train_non_tx = train.loc[~train["state"].eq(forced_state)]
        validation_non_tx = validation.loc[~validation["state"].eq(forced_state)]
        available_training_months = int(train["cohort_month"].nunique())
        max_train_target_check = (
            pd.NaT if train.empty else pd.Timestamp(train["target_check"].max())
        )
        label_embargo_pass = bool(
            not train.empty
            and pd.notna(max_train_target_check)
            and max_train_target_check < label_cutoff
        )
        reason = ""
        if available_training_months < settings.min_temporal_training_months:
            reason = "LABEL_AVAILABLE_TRAINING_MONTHS_BELOW_MINIMUM"
        elif not label_embargo_pass:
            reason = "LABEL_AVAILABILITY_EMBARGO_FAILED"
        elif len(validation_non_tx) < settings.min_temporal_validation_rows:
            reason = "NON_TX_VALIDATION_ROWS_BELOW_MINIMUM"
        elif (
            train_non_tx["target"].nunique() != 2
            or validation_non_tx["target"].nunique() != 2
        ):
            reason = "NON_TX_TARGET_CLASS_MISSING"
        elif not train["state"].eq(forced_state).any():
            reason = "TX_MISSING_FROM_SUBTRAINING"
        if reason:
            skipped.append(
                {
                    "fold": fold + 1,
                    "reason": reason,
                    "validation_start": str(validation_start.date()),
                    "label_cutoff_exclusive": str(label_cutoff.date()),
                    "candidate_training_rows_before_label_embargo": int(
                        len(candidate_train)
                    ),
                    "training_rows_after_label_embargo": int(len(train)),
                    "training_rows_removed_by_label_embargo": int(
                        len(candidate_train) - len(train)
                    ),
                    "label_available_training_cohort_months": int(
                        available_training_months
                    ),
                    "max_train_target_check": (
                        None
                        if pd.isna(max_train_target_check)
                        else str(max_train_target_check.date())
                    ),
                    "label_embargo_pass": label_embargo_pass,
                }
            )
            continue
        if not label_embargo_pass:
            raise AssertionError("Internal state-grouping label embargo failed.")
        metadata = {
            "fold": fold + 1,
            "training_month_start": str(train["cohort_month"].min()),
            "training_month_end": str(train["cohort_month"].max()),
            "validation_month_start": str(validation_months[0]),
            "validation_month_end": str(validation_months[-1]),
            "validation_start": str(validation_start.date()),
            "target_check_col": settings.target_check_col,
            "label_availability_buffer_days": int(
                settings.label_availability_buffer_days
            ),
            "label_cutoff_exclusive": str(label_cutoff.date()),
            "max_train_target_check": str(max_train_target_check.date()),
            "label_embargo_pass": label_embargo_pass,
            "candidate_training_rows_before_label_embargo": int(
                len(candidate_train)
            ),
            "training_rows_removed_by_label_embargo": int(
                len(candidate_train) - len(train)
            ),
            "label_available_training_cohort_months": int(
                available_training_months
            ),
            "training_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "training_non_tx_rows": int(len(train_non_tx)),
            "validation_non_tx_rows": int(len(validation_non_tx)),
            "training_retention_rate": float(train["target"].mean()),
            "validation_retention_rate": float(validation["target"].mean()),
            "training_non_tx_retention_rate": float(train_non_tx["target"].mean()),
            "validation_non_tx_retention_rate": float(
                validation_non_tx["target"].mean()
            ),
        }
        splits.append((train, validation, metadata))
    if any(not metadata["label_embargo_pass"] for _, _, metadata in splits):
        raise AssertionError(
            "At least one internal state split violates label embargo."
        )
    return splits, {
        "status": (
            "AVAILABLE"
            if len(splits) >= 2
            else "UNAVAILABLE_FEWER_THAN_TWO_VALID_FOLDS"
        ),
        "observed_months": len(months),
        "cohort_date_available_rows": int(cohort_available.sum()),
        "target_check_date_available_rows": int(
            target_check_available.sum()
        ),
        "rows_with_both_temporal_dates": int(len(dated)),
        "target_check_col": settings.target_check_col,
        "label_availability_buffer_days": int(
            settings.label_availability_buffer_days
        ),
        "label_embargo_rule": (
            "subtraining target_check must be strictly earlier than the "
            "validation start minus the configured buffer"
        ),
        "all_created_folds_label_embargo_pass": bool(
            splits
            and all(metadata["label_embargo_pass"] for _, _, metadata in splits)
        ),
        "requested_folds": settings.temporal_validation_folds,
        "created_folds": len(splits),
        "validation_months_per_fold": settings.temporal_validation_months,
        "folds": [metadata for _, _, metadata in splits],
        "skipped_folds": skipped,
    }




def _temporal_group_count_scores(
    splits: Sequence[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]],
    settings: StateGroupingSpec,
    candidates: Sequence[int],
    *,
    behavior_cache: Sequence[pd.DataFrame] | None = None,
) -> list[dict[str, Any]]:
    if behavior_cache is not None and len(behavior_cache) != len(splits):
        raise ValueError("Temporal behavior cache does not align with splits.")
    output: list[dict[str, Any]] = []
    for k in candidates:
        fold_metrics: list[dict[str, Any]] = []
        failure = ""
        for split_position, (train, validation, metadata) in enumerate(splits):
            try:
                if behavior_cache is None:
                    mapping, probabilities, partition = _fixed_group_count_model(
                        train, settings, k
                    )
                else:
                    mapping, probabilities, partition = (
                        _fixed_group_count_model_from_behavior(
                            behavior_cache[split_position], settings, k
                        )
                    )
            except (StateGroupingError, ValueError) as exc:
                failure = f"fold_{metadata['fold']}: {exc}"
                break
            # TX has an identical singleton prediction for every candidate K.
            # Excluding it prevents its large volume and temporal drift from
            # diluting differences among the non-TX partitions being compared.
            forced_state = settings.forced_state.strip().upper()
            scored = validation.loc[~validation["state"].eq(forced_state)].copy()
            if (
                len(scored) < settings.min_temporal_validation_rows
                or scored["target"].nunique() != 2
            ):
                failure = (
                    f"fold_{metadata['fold']}: insufficient non-TX validation "
                    "rows or target classes"
                )
                break
            groups = scored["state"].map(mapping)
            subtraining_unseen = groups.isna()
            prediction = groups.map(probabilities).fillna(
                probabilities[_INTERNAL_NON_TX_PRIOR]
            ).to_numpy(dtype=float)
            prediction = np.clip(prediction, 1e-8, 1 - 1e-8)
            actual = scored["target"].to_numpy(dtype=float)
            logloss = float(
                -np.mean(
                    actual * np.log(prediction)
                    + (1.0 - actual) * np.log(1.0 - prediction)
                )
            )
            fold_metrics.append(
                {
                    **metadata,
                    "validation_scored_rows_non_tx": int(len(scored)),
                    "validation_rows_with_state_absent_from_subtraining": int(
                        subtraining_unseen.sum()
                    ),
                    "tx_rows_excluded_from_k_scoring": int(
                        validation["state"].eq(forced_state).sum()
                    ),
                    "validation_log_loss": logloss,
                    "validation_brier": float(np.mean((actual - prediction) ** 2)),
                    "partition_volume_cv": float(partition.volume_cv),
                    "partition_binomial_deviance": float(partition.deviance),
                    "partition_non_tx_group_raw_retention_rates": list(
                        partition.group_retention_rates
                    ),
                    "partition_adjacent_non_tx_retention_rate_gaps": list(
                        partition.adjacent_retention_rate_gaps
                    ),
                    "partition_minimum_adjacent_non_tx_retention_rate_gap": (
                        partition.minimum_adjacent_retention_rate_gap
                    ),
                    "partition_required_minimum_adjacent_non_tx_retention_rate_gap": (
                        partition.required_minimum_adjacent_retention_rate_gap
                    ),
                    "partition_adjacent_non_tx_retention_rate_gap_pass": bool(
                        partition.adjacent_retention_rate_gap_constraint_satisfied
                    ),
                    "partition_volume_role": "AUDIT_ONLY",
                }
            )
        if failure or len(fold_metrics) != len(splits):
            output.append(
                {
                    "k": int(k),
                    "eligible_all_temporal_folds": False,
                    "failure_reason": failure or "INCOMPLETE_TEMPORAL_FOLDS",
                    "fold_metrics": fold_metrics,
                }
            )
            continue
        losses = np.asarray(
            [row["validation_log_loss"] for row in fold_metrics], dtype=float
        )
        briers = np.asarray(
            [row["validation_brier"] for row in fold_metrics], dtype=float
        )
        output.append(
            {
                "k": int(k),
                "eligible_all_temporal_folds": True,
                "failure_reason": "",
                "mean_validation_log_loss": float(np.mean(losses)),
                "standard_error_validation_log_loss": float(
                    np.std(losses, ddof=1) / math.sqrt(len(losses))
                ),
                "newest_fold_validation_log_loss": float(losses[-1]),
                "mean_validation_brier": float(np.mean(briers)),
                "mean_partition_volume_cv": float(
                    np.mean([row["partition_volume_cv"] for row in fold_metrics])
                ),
                "minimum_partition_adjacent_non_tx_retention_rate_gap": float(
                    min(
                        row[
                            "partition_minimum_adjacent_non_tx_retention_rate_gap"
                        ]
                        for row in fold_metrics
                    )
                ),
                "all_partitions_meet_minimum_adjacent_non_tx_retention_rate_gap": bool(
                    all(
                        row[
                            "partition_adjacent_non_tx_retention_rate_gap_pass"
                        ]
                        for row in fold_metrics
                    )
                ),
                "validation_rows_with_state_absent_from_subtraining": int(
                    sum(
                        row["validation_rows_with_state_absent_from_subtraining"]
                        for row in fold_metrics
                    )
                ),
                "fold_metrics": fold_metrics,
            }
        )
    return output




def _fixed_group_count_model(
    train: pd.DataFrame, settings: StateGroupingSpec, k: int
) -> tuple[dict[str, str], dict[str, float], _OptimalPartition]:
    behavior, _ = _state_behavior_stats(train, settings)
    return _fixed_group_count_model_from_behavior(behavior, settings, k)




def _fixed_group_count_model_from_behavior(
    behavior: pd.DataFrame,
    settings: StateGroupingSpec,
    k: int,
) -> tuple[dict[str, str], dict[str, float], _OptimalPartition]:
    """Fit fixed K from cached state-level sufficient statistics."""


    non_tx = _ordered_non_tx_behavior(behavior, settings)
    partition = _optimal_binary_partition(non_tx, k, settings)
    labels = settings.non_tx_group_labels[:k]
    forced_state = settings.forced_state.strip().upper()
    mapping = {forced_state: settings.forced_label}
    for position, state_code in enumerate(non_tx["state"].astype(str)):
        mapping[state_code] = labels[partition.assignment[position]]
    grouped = behavior.assign(retention_group=behavior["state"].map(mapping))
    overall_rate = _clip_probability(
        float(grouped["retained"].sum() / grouped["rows"].sum())
    )
    non_tx_rate = _clip_probability(
        float(non_tx["retained"].sum() / non_tx["rows"].sum())
    )
    probabilities: dict[str, float] = {}
    for label in (settings.forced_label, *labels):
        part = grouped.loc[grouped["retention_group"].eq(label)]
        if part.empty:
            raise StateGroupingError(f"K={k} produced empty subtraining group {label}.")
        prior_rate = overall_rate if label == settings.forced_label else non_tx_rate
        probabilities[label] = float(
            (part["retained"].sum() + settings.prior_strength * prior_rate)
            / (part["rows"].sum() + settings.prior_strength)
        )
    probabilities[_INTERNAL_NON_TX_PRIOR] = non_tx_rate
    return mapping, probabilities, partition




def _mapping_stability(
    splits: Sequence[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]],
    settings: StateGroupingSpec,
    k: int,
    *,
    behavior_cache: Sequence[pd.DataFrame] | None = None,
) -> dict[str, Any]:
    if behavior_cache is not None and len(behavior_cache) != len(splits):
        raise ValueError("Temporal behavior cache does not align with splits.")
    fitted_windows: list[dict[str, Any]] = []
    for split_position, (train, _, metadata) in enumerate(splits):
        try:
            if behavior_cache is None:
                behavior, _ = _state_behavior_stats(train, settings)
            else:
                behavior = behavior_cache[split_position]
            mapping, _, _ = _fixed_group_count_model_from_behavior(
                behavior, settings, k
            )
        except (StateGroupingError, ValueError):
            continue
        forced_state = settings.forced_state.strip().upper()
        fitted_windows.append(
            {
                "split_position": int(split_position + 1),
                "metadata": metadata,
                "mapping": mapping,
                "behavior_scores": {
                    str(row.state): float(row.behavior_order_score)
                    for row in behavior.itertuples(index=False)
                    if str(row.state) != forced_state
                },
            }
        )


    agreements: list[float] = []
    coassignment_jaccards: list[float] = []
    rank_correlations: list[float] = []
    adjacent_agreements: list[float] = []
    adjacent_coassignment_jaccards: list[float] = []
    adjacent_rank_correlations: list[float] = []
    pair_details: list[dict[str, Any]] = []
    for left_position in range(len(fitted_windows)):
        for right_position in range(left_position + 1, len(fitted_windows)):
            left_window = fitted_windows[left_position]
            right_window = fitted_windows[right_position]
            left = left_window["mapping"]
            right = right_window["mapping"]
            states = sorted(
                set(left)
                .intersection(right)
                .difference({settings.forced_state.strip().upper()})
            )
            comparisons: list[bool] = []
            left_coassigned: set[tuple[str, str]] = set()
            right_coassigned: set[tuple[str, str]] = set()
            for first in range(len(states)):
                for second in range(first + 1, len(states)):
                    a, b = states[first], states[second]
                    pair = (a, b)
                    if left[a] == left[b]:
                        left_coassigned.add(pair)
                    if right[a] == right[b]:
                        right_coassigned.add(pair)
                    comparisons.append(
                        (left[a] == left[b]) == (right[a] == right[b])
                    )
            agreement: float | None = None
            coassignment_jaccard: float | None = None
            if comparisons:
                agreement = float(np.mean(comparisons))
                union = left_coassigned.union(right_coassigned)
                coassignment_jaccard = (
                    1.0
                    if not union
                    else float(
                        len(left_coassigned.intersection(right_coassigned))
                        / len(union)
                    )
                )
                agreements.append(agreement)
                coassignment_jaccards.append(coassignment_jaccard)
            common_states = sorted(
                set(left_window["behavior_scores"]).intersection(
                    right_window["behavior_scores"]
                )
            )
            rank_correlation: float | None = None
            if len(common_states) >= 2:
                left_scores = pd.Series(
                    [
                        left_window["behavior_scores"][state]
                        for state in common_states
                    ],
                    dtype=float,
                )
                right_scores = pd.Series(
                    [
                        right_window["behavior_scores"][state]
                        for state in common_states
                    ],
                    dtype=float,
                )
                correlation = left_scores.rank(method="average").corr(
                    right_scores.rank(method="average"), method="pearson"
                )
                if pd.notna(correlation):
                    rank_correlation = float(correlation)
                    rank_correlations.append(rank_correlation)


            left_metadata = left_window["metadata"]
            right_metadata = right_window["metadata"]
            adjacent = bool(
                int(right_metadata["fold"])
                == int(left_metadata["fold"]) + 1
            )
            if adjacent:
                if agreement is not None:
                    adjacent_agreements.append(agreement)
                if coassignment_jaccard is not None:
                    adjacent_coassignment_jaccards.append(
                        coassignment_jaccard
                    )
                if rank_correlation is not None:
                    adjacent_rank_correlations.append(rank_correlation)


            identity_fields = (
                "fold",
                "training_month_start",
                "training_month_end",
                "validation_month_start",
                "validation_month_end",
                "validation_start",
                "label_cutoff_exclusive",
                "max_train_target_check",
            )
            pair_details.append(
                {
                    "left": {
                        "split_position": left_window["split_position"],
                        **{
                            key: left_metadata.get(key)
                            for key in identity_fields
                        },
                    },
                    "right": {
                        "split_position": right_window["split_position"],
                        **{
                            key: right_metadata.get(key)
                            for key in identity_fields
                        },
                    },
                    "adjacent_temporal_folds": adjacent,
                    "common_state_count": int(len(states)),
                    "state_pair_comparison_count": int(len(comparisons)),
                    "coassignment_agreement": agreement,
                    "coassignment_jaccard": coassignment_jaccard,
                    "behavior_rank_correlation": rank_correlation,
                }
            )


    def mean(values: Sequence[float]) -> float | None:
        return None if not values else float(np.mean(values))


    def minimum(values: Sequence[float]) -> float | None:
        return None if not values else float(np.min(values))


    return {
        "mapping_count": len(fitted_windows),
        "all_pair_comparisons": len(pair_details),
        "adjacent_pair_comparisons": int(
            sum(row["adjacent_temporal_folds"] for row in pair_details)
        ),
        "pairwise_comparisons": len(agreements),
        "mean_pairwise_coassignment_agreement": mean(agreements),
        "minimum_pairwise_coassignment_agreement": minimum(agreements),
        "mean_adjacent_coassignment_agreement": mean(adjacent_agreements),
        "minimum_adjacent_coassignment_agreement": minimum(
            adjacent_agreements
        ),
        "unadjusted_pairwise_coassignment_agreement_note": (
            "Includes jointly separated state pairs and may be optimistic at larger K; "
            "the adjacent-fold Jaccard mean below is used for the mapping-"
            "stability target."
        ),
        "coassignment_jaccard_pairwise_comparisons": len(coassignment_jaccards),
        "mean_pairwise_coassignment_jaccard": mean(coassignment_jaccards),
        "minimum_pairwise_coassignment_jaccard": minimum(
            coassignment_jaccards
        ),
        "adjacent_coassignment_jaccard_comparisons": len(
            adjacent_coassignment_jaccards
        ),
        "mean_adjacent_coassignment_jaccard": mean(
            adjacent_coassignment_jaccards
        ),
        "minimum_adjacent_coassignment_jaccard": minimum(
            adjacent_coassignment_jaccards
        ),
        "behavior_rank_pairwise_comparisons": len(rank_correlations),
        "mean_pairwise_behavior_rank_correlation": mean(rank_correlations),
        "minimum_pairwise_behavior_rank_correlation": minimum(
            rank_correlations
        ),
        "adjacent_behavior_rank_comparisons": len(
            adjacent_rank_correlations
        ),
        "mean_adjacent_behavior_rank_correlation": mean(
            adjacent_rank_correlations
        ),
        "minimum_adjacent_behavior_rank_correlation": minimum(
            adjacent_rank_correlations
        ),
        "stability_target_basis": "MEAN_OF_ADJACENT_TEMPORAL_FOLD_PAIRS",
        "all_pair_metrics_are_diagnostic_only": True,
        "pair_details": pair_details,
    }




def _state_saturated_nll(rows: np.ndarray, retained: np.ndarray) -> float:
    result = 0.0
    for n, r in zip(rows, retained):
        p = _clip_probability(float(r / n))
        result -= float(r * math.log(p) + (n - r) * math.log(1.0 - p))
    return result




def _clip_probability(value: float) -> float:
    return float(np.clip(value, 1e-8, 1 - 1e-8))




__all__ = [
    "ALGORITHM_ID",
    "RAW_STATE_COLUMN",
    "STATE_GROUP_COLUMN",
    "TARGET_COLUMN",
    "TENURE_GROUP_COLUMN",
    "TARGET_CHECK_COLUMN",
    "TX_GROUP_LABEL",
    "NON_TX_GROUP_LABELS",
    "CLEANED_STATE_CODES",
    "StateGroupingError",
    "StateGroupingSpec",
    "FittedStateGrouper",
    "fit_state_grouper",
    "apply_state_grouper",
]
