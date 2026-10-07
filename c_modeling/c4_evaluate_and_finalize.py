#!/usr/bin/env python
"""Score frozen recipes on the outer folds and build pre-holdout refits.


This stage is deliberately downstream of all feature, history, regularization,
and XGBoost/Optuna selection.  It performs no tuning.  It refits each frozen
family recipe on every outer training partition, writes aligned out-of-fold
predictions, selects the model family from those predictions, and finally
refits both recipes on their label-mature pre-holdout training histories so
the normal report can compare them over the same complete observed history.
Only the selected family's bundle is designated for subsequent final testing.
That scoreable bundle remains a provisional development artifact
until the tenure-grouping decision is locked and the separately governed final
holdout evaluation is approved and passed; this stage never deploys it.


The final six-month holdout is identified and fingerprinted but is never
transformed or scored here.  Opening that holdout must be a separate,
explicitly governed evaluation action after the tenure design and model family
have both been frozen.
"""


from __future__ import annotations


import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression




OUTER_CONTRACT = "retention_outer_evaluation"
OUTER_CONTRACT_REVISION = 1
FINAL_MODEL_CONTRACT = "retention_final_deployable_model"
FINAL_MODEL_CONTRACT_REVISION = 1
SUPPORTED_HISTORY_MONTHS = (36, 48, 60)
MODEL_FAMILIES = ("logistic", "xgboost")
OUTER_FOLD_IDS = (1, 2, 3, 4, 5)




@dataclass(frozen=True)
class FamilyRecipe:
    family: str
    features: tuple[str, ...]
    feature_types: dict[str, str]
    history: dict[str, Any]
    history_months: int
    rare_min_frequency: float
    model_recipe: dict[str, Any]
    seeds: tuple[int, ...] = ()
    boosting_rounds: int | None = None




def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    return value




def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp_", suffix=".json", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True,
                      allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)




def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp_", suffix=".csv", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)




def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp_", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)




def _atomic_copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required source artifact is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp_", suffix=destination.suffix, dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)




def _separate_outer_performance_artifacts(
    plot_manifest: Mapping[str, Any],
    *,
    interpretability_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Keep outer performance evidence separate from selector interpretation.


    The generic controller expects ``artifacts`` to be a list of paths. Rich
    metadata for those same outer-OOF plots lives in ``artifact_metadata``.
    Selector-era SHAP output is deliberately carried in a separate sibling
    collection so consumers cannot mistake it for outer-fold performance.
    """


    raw_performance = plot_manifest.get("artifacts")
    if not isinstance(raw_performance, list) or not raw_performance:
        raise ValueError("Outer performance plot manifest has no artifacts.")
    performance_metadata: list[dict[str, Any]] = []
    for value in raw_performance:
        if not isinstance(value, Mapping):
            raise TypeError("Outer performance artifact metadata must be mappings.")
        item = dict(value)
        if str(item.get("evaluation_scope", "")).strip().lower() != "outer_oof":
            raise ValueError(
                "Performance artifact_metadata may contain only outer_oof plots."
            )
        if bool(item.get("interpretability_only", False)):
            raise ValueError(
                "Interpretability-only output cannot enter performance artifacts."
            )
        path = str(item.get("path", "")).strip()
        if not path:
            raise ValueError("Every outer performance artifact requires a path.")
        performance_metadata.append(item)


    interpretation: list[dict[str, Any]] = []
    for value in interpretability_artifacts:
        if not isinstance(value, Mapping):
            raise TypeError("Interpretability artifact metadata must be mappings.")
        item = dict(value)
        if item.get("interpretability_only") is not True:
            raise ValueError(
                "Interpretability artifacts must be labeled interpretability_only."
            )
        if item.get("performance_evidence") is not False:
            raise ValueError(
                "Interpretability artifacts must not be labeled performance evidence."
            )
        scope = str(item.get("evaluation_scope", "")).strip().lower()
        if scope != "inner_newest_chronological_validation":
            raise ValueError(
                "Selector SHAP must retain its inner newest-validation scope."
            )
        if not str(item.get("path", "")).strip():
            raise ValueError("Every interpretability artifact requires a path.")
        interpretation.append(item)


    result = dict(plot_manifest)
    result["artifact_metadata"] = performance_metadata
    result["artifacts"] = [str(item["path"]) for item in performance_metadata]
    result["interpretability_artifacts"] = interpretation
    return result




def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload




def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()




def _json_sha256(payload: Any) -> str:
    rendered = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()




def _index_sha256(values: Sequence[Any] | np.ndarray) -> str:
    numeric = pd.to_numeric(pd.Series(np.asarray(values)), errors="raise").to_numpy()
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("Population indices must be finite integers.")
    canonical = np.sort(numeric.astype("<i8"), kind="stable")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()




def validated_history_months(history: Mapping[str, Any], *, family: str) -> int:
    """Validate the governed equal-weight mature-month history contract."""


    kind = str(history.get("kind", "")).strip().lower()
    try:
        months = int(history.get("months"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{family} selected history has no valid months value.") from exc
    expected_id = f"RECENT_{months}M"
    history_id = str(history.get("history_id", "")).strip().upper()
    if kind != "recent" or months not in SUPPORTED_HISTORY_MONTHS:
        raise ValueError(
            f"{family} selected history must be one of RECENT_36M, RECENT_48M, "
            f"or RECENT_60M; observed {dict(history)}."
        )
    if history_id != expected_id:
        raise ValueError(
            f"{family} history identity is inconsistent: expected {expected_id}, "
            f"observed {history_id!r}."
        )
    return months




def final_evaluation_training_indices(
    frame: pd.DataFrame,
    final_holdout_idx: Sequence[Any] | np.ndarray,
    *,
    cohort_col: str,
    target_check_col: str,
    label_buffer_days: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return every label-mature pre-holdout row without reading its target."""


    holdout_idx = np.asarray(final_holdout_idx)
    if holdout_idx.ndim != 1 or not len(holdout_idx):
        raise ValueError("The final holdout index must be a nonempty vector.")
    cohort = pd.to_datetime(frame[cohort_col], errors="raise")
    target_check = pd.to_datetime(frame[target_check_col], errors="raise")
    holdout_start = cohort.loc[holdout_idx].min()
    label_cutoff = holdout_start - pd.Timedelta(days=int(label_buffer_days))
    is_holdout = frame.index.isin(holdout_idx)
    eligible = (~is_holdout) & cohort.lt(holdout_start) & target_check.lt(label_cutoff)
    selected = frame.index[eligible].to_numpy()
    if not len(selected):
        raise ValueError("No label-mature pre-holdout rows are available for refit.")
    if np.intersect1d(selected, holdout_idx).size:
        raise AssertionError("Final holdout rows entered final refit training.")
    audit = {
        "definition": (
            "pre-holdout cohorts whose outcome check date is strictly before "
            "the final-holdout scoring boundary"
        ),
        "holdout_start": str(holdout_start.date()),
        "label_cutoff_exclusive": str(label_cutoff.date()),
        "label_buffer_days": int(label_buffer_days),
        "candidate_rows": int(len(selected)),
        "candidate_index_sha256": _index_sha256(selected),
    }
    return selected, audit




def final_refit_descriptive_indices(
    frame: pd.DataFrame,
    final_holdout_idx: Sequence[Any] | np.ndarray,
    selected_training_idx: Sequence[Any] | np.ndarray,
    *,
    cohort_col: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return observed pre-holdout rows for a descriptive final-refit trend.


    This population is deliberately broader than the label-mature final-refit
    population so the trend retains the complete historical context through
    the month immediately before the sealed holdout. The resulting predictions
    are descriptive fitted/backcast values, never OOF performance evidence.
    Exact selected-history membership is carried separately so the chart
    writer can distinguish the model-fitting window.
    """


    if cohort_col not in frame:
        raise KeyError(f"Descriptive trend population is missing {cohort_col!r}.")
    holdout_idx = np.asarray(final_holdout_idx)
    training_idx = np.asarray(selected_training_idx)
    for label, values in (
        ("final holdout", holdout_idx),
        ("selected final-refit training", training_idx),
    ):
        if values.ndim != 1 or not len(values):
            raise ValueError(f"The {label} index must be a nonempty vector.")
        if pd.Index(values).duplicated().any():
            raise ValueError(f"The {label} index contains duplicate rows.")
        missing = pd.Index(values).difference(frame.index)
        if len(missing):
            raise ValueError(
                f"The {label} index contains rows outside the prepared frame."
            )


    if np.intersect1d(training_idx, holdout_idx).size:
        raise AssertionError("Final holdout rows entered final-refit training.")
    cohort = pd.to_datetime(frame[cohort_col], errors="raise")
    holdout_start = cohort.loc[holdout_idx].min()
    full_mask = cohort.lt(holdout_start) & ~frame.index.isin(holdout_idx)
    full_idx = frame.index[full_mask].to_numpy()
    if not len(full_idx):
        raise ValueError("No observed pre-holdout rows are available for the trend.")
    if np.intersect1d(full_idx, holdout_idx).size:
        raise AssertionError("Final holdout rows entered descriptive trend scoring.")
    training_dates = cohort.loc[training_idx]
    if training_dates.ge(holdout_start).any():
        raise AssertionError(
            "Selected final-refit training is not contained in pre-holdout history."
        )
    # Every supplied training row was already proven to belong to ``frame``
    # and to precede ``holdout_start``. Because ``full_idx`` is defined as all
    # frame rows preceding that boundary, subset membership follows without a
    # second multi-million-row hash table.
    full_dates = cohort.loc[full_idx]
    return full_idx, {
        "definition": (
            "all observed Month-1 cohorts strictly before the sealed final "
            "holdout; predictions are descriptive and not OOF performance"
        ),
        "rows": int(len(full_idx)),
        "cohort_start": str(full_dates.min().date()),
        "cohort_end": str(full_dates.max().date()),
        "cohort_months": int(full_dates.dt.to_period("M").nunique()),
        "index_sha256": _index_sha256(full_idx),
        "selected_training_rows": int(len(training_idx)),
        "selected_training_index_sha256": _index_sha256(training_idx),
        "selected_training_start": str(training_dates.min().date()),
        "selected_training_end": str(training_dates.max().date()),
        "holdout_start": str(holdout_start.date()),
        "holdout_rows_scored": 0,
        "performance_evidence": False,
    }




def _sorted_membership_mask(
    values: Sequence[Any] | np.ndarray,
    selected_sorted: Sequence[Any] | np.ndarray,
) -> np.ndarray:
    """Return exact membership without building a large Python hash set."""


    raw = np.asarray(values)
    selected = np.asarray(selected_sorted)
    if raw.ndim != 1 or selected.ndim != 1:
        raise ValueError("Membership indices must be one-dimensional.")
    if not len(selected):
        return np.zeros(len(raw), dtype=bool)
    if len(selected) > 1 and np.any(selected[1:] <= selected[:-1]):
        raise ValueError("selected_sorted must be strictly increasing.")
    positions = np.searchsorted(selected, raw)
    inside = positions < len(selected)
    result = np.zeros(len(raw), dtype=bool)
    result[inside] = selected[positions[inside]] == raw[inside]
    return result




def _final_refit_monthly_partial(
    frame: pd.DataFrame,
    p_retention: Sequence[float] | np.ndarray,
    *,
    cohort_col: str,
    target_col: str,
    selected_training_sorted: Sequence[Any] | np.ndarray,
) -> pd.DataFrame:
    """Aggregate one scored batch while preserving exact history membership."""


    required = {cohort_col, target_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Final-refit trend batch is missing columns: {missing}")
    probability = np.asarray(p_retention, dtype=float)
    if probability.ndim != 1 or len(probability) != len(frame):
        raise ValueError("Final-refit trend probabilities do not match batch rows.")
    if not np.isfinite(probability).all() or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError("Final-refit retention probabilities must be finite in [0, 1].")
    numeric_target = pd.to_numeric(frame[target_col], errors="raise")
    if numeric_target.isna().any() or not set(numeric_target.unique()).issubset(
        {0, 1}
    ):
        raise ValueError("Final-refit trend target must be binary 0/1.")
    target = numeric_target.astype(int)
    cohort = pd.to_datetime(frame[cohort_col], errors="raise")
    if cohort.isna().any():
        raise ValueError("Final-refit trend cohort dates cannot be missing.")
    selected = _sorted_membership_mask(
        frame.index.to_numpy(), selected_training_sorted
    )
    work = pd.DataFrame(
        {
            "YEAR_MONTH": cohort.dt.to_period("M").astype(str).to_numpy(),
            "population_role": np.where(
                selected,
                "FINAL_REFIT_TRAINING_HISTORY",
                "OUTSIDE_FINAL_REFIT_TRAINING_HISTORY",
            ),
            "retained_policies": target.to_numpy(dtype=np.int64),
            "predicted_retained_policies": probability,
        }
    )
    return (
        work.groupby(["YEAR_MONTH", "population_role"], sort=True, observed=True)
        .agg(
            policies=("retained_policies", "size"),
            retained_policies=("retained_policies", "sum"),
            predicted_retained_policies=("predicted_retained_policies", "sum"),
        )
        .reset_index()
    )




def _finalize_final_refit_monthly(
    partials: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Combine batch sums into policy-weighted monthly actual/predicted rates."""


    if not partials:
        raise ValueError("At least one final-refit trend batch is required.")
    combined = pd.concat(partials, ignore_index=True)
    required = {
        "YEAR_MONTH",
        "population_role",
        "policies",
        "retained_policies",
        "predicted_retained_policies",
    }
    missing = sorted(required.difference(combined.columns))
    if missing:
        raise KeyError(f"Final-refit monthly partials are missing: {missing}")
    by_role = (
        combined.groupby(
            ["YEAR_MONTH", "population_role"], sort=True, observed=True
        )[["policies", "retained_policies", "predicted_retained_policies"]]
        .sum()
        .reset_index()
    )
    total = (
        by_role.groupby("YEAR_MONTH", sort=True, observed=True)[
            ["policies", "retained_policies", "predicted_retained_policies"]
        ]
        .sum()
        .reset_index()
    )
    selected = (
        by_role.loc[
            by_role["population_role"].eq("FINAL_REFIT_TRAINING_HISTORY"),
            ["YEAR_MONTH", "policies"],
        ]
        .rename(columns={"policies": "selected_history_training_rows"})
    )
    monthly = total.merge(selected, on="YEAR_MONTH", how="left", validate="one_to_one")
    monthly["selected_history_training_rows"] = (
        monthly["selected_history_training_rows"].fillna(0).astype(np.int64)
    )
    monthly["selected_history_training_fraction"] = (
        monthly["selected_history_training_rows"] / monthly["policies"]
    )
    monthly["in_selected_history_window"] = monthly[
        "selected_history_training_rows"
    ].gt(0)
    monthly["population_role"] = np.select(
        [
            monthly["selected_history_training_fraction"].eq(1.0),
            monthly["selected_history_training_fraction"].eq(0.0),
        ],
        [
            "FINAL_REFIT_TRAINING_HISTORY",
            "OUTSIDE_FINAL_REFIT_TRAINING_HISTORY",
        ],
        default="MIXED_HISTORY_MEMBERSHIP",
    )
    monthly["actual_retention"] = (
        monthly["retained_policies"] / monthly["policies"]
    )
    monthly["predicted_retention"] = (
        monthly["predicted_retained_policies"] / monthly["policies"]
    )
    monthly["actual_churn"] = 1.0 - monthly["actual_retention"]
    monthly["predicted_churn"] = 1.0 - monthly["predicted_retention"]
    monthly["calibration_gap"] = (
        monthly["predicted_retention"] - monthly["actual_retention"]
    )
    for window in (3, 12):
        rolling_policies = monthly["policies"].rolling(
            window, min_periods=1
        ).sum()
        rolling_actual = monthly["retained_policies"].rolling(
            window, min_periods=1
        ).sum()
        rolling_predicted = monthly["predicted_retained_policies"].rolling(
            window, min_periods=1
        ).sum()
        monthly[f"policies_{window}m"] = rolling_policies
        monthly[f"actual_retention_{window}m_weighted"] = (
            rolling_actual / rolling_policies
        )
        monthly[f"predicted_retention_{window}m_weighted"] = (
            rolling_predicted / rolling_policies
        )
    period = pd.PeriodIndex(monthly["YEAR_MONTH"], freq="M")
    expected = pd.period_range(period.min(), period.max(), freq="M")
    missing_months = expected.difference(period)
    if len(missing_months):
        raise ValueError(
            "Final-refit descriptive history has missing cohort months: "
            + ", ".join(missing_months.astype(str))
        )
    return monthly.sort_values("YEAR_MONTH", kind="stable").reset_index(drop=True)




def aggregate_final_refit_descriptive_monthly(
    frame: pd.DataFrame,
    p_retention: Sequence[float] | np.ndarray,
    *,
    cohort_col: str,
    target_col: str,
    selected_training_idx: Sequence[Any] | np.ndarray,
) -> pd.DataFrame:
    """Public pure helper used by tests and single-batch callers."""


    selected_sorted = np.sort(np.asarray(selected_training_idx))
    partial = _final_refit_monthly_partial(
        frame,
        p_retention,
        cohort_col=cohort_col,
        target_col=target_col,
        selected_training_sorted=selected_sorted,
    )
    return _finalize_final_refit_monthly([partial])




def score_full_history_family(
    *,
    prepared: pd.DataFrame,
    final_holdout_idx: Sequence[Any] | np.ndarray,
    selected_training_idx: Sequence[Any] | np.ndarray,
    family: str,
    bundle_path: Path,
    cohort_col: str,
    target_col: str,
    history_audit: Mapping[str, Any],
    deployment_bundle: Any,
    batch_rows: int,
    xgboost_device: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Score one frozen refit on all pre-holdout rows, preserving membership.


    Raw policy predictions are reduced to monthly sums batch by batch. The
    same source population is used for both families; training membership
    differs legitimately with their selected histories. These are fitted /
    backcast diagnostics, not fresh out-of-fold performance observations.
    """
    if family not in MODEL_FAMILIES:
        raise ValueError(f"Unsupported full-history model family: {family!r}")
    if type(batch_rows) is not int or batch_rows < 1:
        raise ValueError("Full-history scoring batch_rows must be a positive integer.")
    descriptive_idx, population = final_refit_descriptive_indices(
        prepared, final_holdout_idx, selected_training_idx, cohort_col=cohort_col,
    )
    # The production population is sorted once and has a RangeIndex. Slice
    # its prefix without allocating another multi-million-row full frame.
    if not isinstance(prepared.index, pd.RangeIndex) or not np.array_equal(
        descriptive_idx, prepared.index[:len(descriptive_idx)].to_numpy()
    ):
        raise ValueError("Full-history scoring requires the sorted pre-holdout prefix.")
    descriptive_frame = prepared.iloc[:len(descriptive_idx)]
    training_sorted = np.sort(np.asarray(selected_training_idx))
    holdout_sorted = np.sort(np.asarray(final_holdout_idx))
    if family == "logistic":
        batches = deployment_bundle.iter_logistic_bundle_scores(
            bundle_path, descriptive_frame, batch_size=batch_rows, verify_hashes=True,
        )
    else:
        batches = _iter_xgboost_scores_on_device(
            bundle_path, descriptive_frame, batch_size=batch_rows,
            deployment_bundle=deployment_bundle, device=xgboost_device,
        )
    partials: list[pd.DataFrame] = []
    next_start = 0
    for raw_start, raw_stop, raw_probability in batches:
        start, stop = int(raw_start), int(raw_stop)
        if start != next_start or stop <= start or stop > len(descriptive_frame):
            raise RuntimeError(
                "Full-history scorer returned incomplete or overlapping batches: "
                f"family={family}; expected_start={next_start}; observed=({start}, {stop})."
            )
        batch = descriptive_frame.iloc[start:stop]
        if _sorted_membership_mask(batch.index.to_numpy(), holdout_sorted).any():
            raise AssertionError("Final holdout entered full-history scoring.")
        probability = np.asarray(raw_probability, dtype=float)
        if family == "xgboost":
            probability = 1.0 - probability
        partials.append(_final_refit_monthly_partial(
            batch, probability, cohort_col=cohort_col, target_col=target_col,
            selected_training_sorted=training_sorted,
        ))
        next_start = stop
        if next_start == len(descriptive_frame) or len(partials) % 10 == 0:
            print(
                f"FULL-HISTORY {family}: scored {next_start:,} / "
                f"{len(descriptive_frame):,} pre-holdout rows", flush=True,
            )
    if next_start != len(descriptive_frame):
        raise RuntimeError(
            f"Full-history {family} scoring did not cover every pre-holdout row: "
            f"expected={len(descriptive_frame)}; scored={next_start}."
        )
    monthly = _finalize_final_refit_monthly(partials)
    if int(monthly["policies"].sum()) != len(descriptive_idx):
        raise AssertionError("Full-history monthly totals do not match scored rows.")
    if int(monthly["selected_history_training_rows"].sum()) != len(training_sorted):
        raise AssertionError("Full-history exact training membership does not match.")
    periods = pd.PeriodIndex(monthly["YEAR_MONTH"], freq="M")
    start = pd.Period(str(history_audit["cutoff_month_inclusive"]), freq="M")
    end = pd.Period(str(history_audit["anchor_month"]), freq="M")
    outside = (periods < start) | (periods > end)
    # The latest mature cohort month can be partial. Exact row membership is
    # authoritative; do not falsely require every row in that month to train.
    if monthly.loc[outside, "selected_history_training_rows"].ne(0).any():
        raise AssertionError("Final-refit training rows fall outside the selected history.")
    by_role = (
        pd.concat(partials, ignore_index=True)
        .groupby(["YEAR_MONTH", "population_role"], sort=True, observed=True)[
            ["policies", "retained_policies", "predicted_retained_policies"]
        ].sum().reset_index()
    )
    by_role.insert(1, "model_family", family)
    population.update({
        "model_family": family,
        "batch_rows": batch_rows,
        "scored_preholdout_rows": int(len(descriptive_idx)),
        "scored_preholdout_months": int(len(monthly)),
        "holdout_rows_scored": 0,
        "performance_evidence": False,
    })
    return monthly, by_role, population




def stable_row_ids(frame: pd.DataFrame, identity_columns: Sequence[str]) -> pd.Series:
    """Create deterministic row IDs from governed source identity columns."""


    columns = [str(column) for column in identity_columns if str(column) in frame]
    if not columns:
        raise ValueError("At least one row-identity column is required.")
    canonical = frame.loc[:, columns].copy()
    for column in canonical:
        if pd.api.types.is_datetime64_any_dtype(canonical[column]):
            canonical[column] = pd.to_datetime(
                canonical[column], errors="raise"
            ).dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        else:
            canonical[column] = canonical[column].astype("string").fillna(
                "__MISSING__"
            )
    hashed = pd.util.hash_pandas_object(canonical, index=False).to_numpy(np.uint64)
    result = pd.Series(
        [f"r{int(value):016x}" for value in hashed], index=frame.index, name="row_id"
    )
    if result.duplicated().any():
        duplicate_rows = int(result.duplicated(False).sum())
        raise ValueError(
            "Governed identity columns do not uniquely identify evaluation rows; "
            f"duplicate_rows={duplicate_rows}; columns={columns}."
        )
    return result




def _paired_standard_error(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    if len(numeric) < 2:
        return 0.0
    return float(numeric.std(ddof=1) / math.sqrt(len(numeric)))




def choose_model_family(
    fold_metrics: pd.DataFrame,
) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    """Choose a family by paired outer-fold AP skill and recall one-SE rules.


    AP skill is primary.  Recall/capture at the fixed 10% capacity is the
    operational guardrail.  When both families are statistically eligible on
    both criteria, Logistic wins the explicit parsimony tie-break.
    """


    required = {
        "fold",
        "model_family",
        "churn_ap_skill",
        "churn_recall_10pct",
    }
    missing = sorted(required.difference(fold_metrics.columns))
    if missing:
        raise KeyError(f"Outer fold metrics are missing: {missing}")
    pivot_ap = fold_metrics.pivot(
        index="fold", columns="model_family", values="churn_ap_skill"
    )
    pivot_recall = fold_metrics.pivot(
        index="fold", columns="model_family", values="churn_recall_10pct"
    )
    if (
        set(pivot_ap.columns) != set(MODEL_FAMILIES)
        or set(pivot_recall.columns) != set(MODEL_FAMILIES)
    ):
        raise ValueError("Family selection requires aligned Logistic and XGBoost folds.")
    if not pivot_ap.index.equals(pivot_recall.index):
        raise ValueError("Family selection metrics are not aligned by outer fold.")
    if (
        not np.isfinite(pivot_ap.to_numpy(dtype=float)).all()
        or not np.isfinite(pivot_recall.to_numpy(dtype=float)).all()
    ):
        raise ValueError("Family selection AP skill and recall metrics must be finite.")
    observed_folds = tuple(int(value) for value in sorted(pivot_ap.index))
    if observed_folds != OUTER_FOLD_IDS:
        raise ValueError(
            "Family selection requires exactly five aligned outer folds "
            f"{list(OUTER_FOLD_IDS)}; observed={list(observed_folds)}."
        )


    ap_means = pivot_ap.mean()
    ap_leader = str(ap_means.sort_values(ascending=False, kind="stable").index[0])
    ap_se: dict[str, float] = {}
    ap_delta: dict[str, float] = {}
    ap_pass: dict[str, bool] = {}
    for family in MODEL_FAMILIES:
        paired = pivot_ap[family] - pivot_ap[ap_leader]
        ap_delta[family] = float(paired.mean())
        ap_se[family] = _paired_standard_error(paired)
        ap_pass[family] = bool(ap_delta[family] >= -ap_se[family] - 1e-15)


    ap_eligible = [family for family in MODEL_FAMILIES if ap_pass[family]]
    recall_means = pivot_recall.mean()
    recall_leader = str(
        recall_means.loc[ap_eligible]
        .sort_values(ascending=False, kind="stable")
        .index[0]
    )
    recall_se: dict[str, float] = {}
    recall_delta: dict[str, float] = {}
    recall_pass: dict[str, bool] = {}
    for family in MODEL_FAMILIES:
        paired = pivot_recall[family] - pivot_recall[recall_leader]
        recall_delta[family] = float(paired.mean())
        recall_se[family] = _paired_standard_error(paired)
        recall_pass[family] = bool(
            family in ap_eligible
            and recall_delta[family] >= -recall_se[family] - 1e-15
        )


    finalists = [family for family in MODEL_FAMILIES if recall_pass[family]]
    if not finalists:
        raise AssertionError("The outer-family selection rule produced no finalist.")
    selected = "logistic" if "logistic" in finalists else finalists[0]
    rows: list[dict[str, Any]] = []
    for family in MODEL_FAMILIES:
        part = fold_metrics.loc[fold_metrics["model_family"].eq(family)]
        rows.append(
            {
                "model_family": family,
                "outer_folds": int(part["fold"].nunique()),
                "mean_churn_ap_skill": float(part["churn_ap_skill"].mean()),
                "std_churn_ap_skill": float(part["churn_ap_skill"].std(ddof=1)),
                "mean_churn_recall_10pct": float(
                    part["churn_recall_10pct"].mean()
                ),
                "std_churn_recall_10pct": float(
                    part["churn_recall_10pct"].std(ddof=1)
                ),
                "ap_skill_delta_vs_leader": ap_delta[family],
                "ap_skill_paired_se": ap_se[family],
                "ap_skill_one_se_eligible": ap_pass[family],
                "recall_delta_vs_eligible_leader": recall_delta[family],
                "recall_paired_se": recall_se[family],
                "recall_one_se_eligible": recall_pass[family],
                "parsimony_finalist": family in finalists,
                "selected": family == selected,
            }
        )
    comparison = pd.DataFrame(rows)
    decision = {
        "selected_model_family": selected,
        "selection_population": "five aligned outer validation folds",
        "primary_metric": "churn_ap_skill",
        "business_guardrail": "churn_recall_10pct",
        "statistical_rule": "paired outer-fold one-standard-error noninferiority",
        "parsimony_tie_break": "prefer Logistic when both families pass",
        "ap_skill_leader": ap_leader,
        "recall_leader_among_ap_eligible": recall_leader,
        "eligible_finalists": finalists,
        "final_holdout_used": False,
    }
    return selected, comparison, decision




def _find_bundle_manifest(bundle_dir: Path) -> Path:
    matches: list[Path] = []
    for path in sorted(bundle_dir.glob("family_manifest_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("contract") == "retention_frozen_source_model_bundle":
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one frozen-family manifest under {bundle_dir}; found {matches}."
        )
    return matches[0]




def _archive_partial_c4_attempt(
    run_root: Path, *, recovery_attempt_id: str
) -> list[dict[str, Any]]:
    """Preserve incomplete c4 trees before a guarded failed-run retry."""


    safe_attempt = str(recovery_attempt_id).strip()
    if not safe_attempt or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in safe_attempt
    ):
        safe_attempt = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    records: list[dict[str, Any]] = []
    for name in ("outer_evaluation", "deployment"):
        source = run_root / name
        if not source.exists():
            continue
        if source.is_symlink() or not source.is_dir():
            raise RuntimeError(
                f"Failed-c4 recovery refuses a linked/non-directory path: {source}"
            )
        if source.resolve().parent != run_root.resolve():
            raise RuntimeError(f"Unsafe failed-c4 recovery source: {source}")
        base = run_root / f"{name}_failed_attempt_{safe_attempt}"
        destination = base
        suffix = 1
        while destination.exists():
            destination = run_root / f"{base.name}_{suffix:02d}"
            suffix += 1
        receipt = {
            "contract": "retention_c4_failed_attempt_archive_v1",
            "run_id": run_root.name,
            "source_directory": str(source),
            "archive_directory": str(destination),
            "recovery_attempt_id": safe_attempt,
            "archived_at": datetime.now().isoformat(timespec="seconds"),
        }
        _atomic_json(source / "c4_resume_archive.json", receipt)
        source.rename(destination)
        records.append(receipt)
    return records




def _c4_io_path(path: Path) -> Path:
    """Support archived bundle paths longer than MAX_PATH on Windows."""
    absolute = str(path.absolute())
    if os.name != "nt" or absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)




def _validated_c4_archive_records(
    run_root: Path, records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Accept only receipt-backed C4 archives belonging to this exact run."""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_record in records:
        record = dict(raw_record)
        source = Path(str(record["source_directory"]))
        archive = Path(str(record["archive_directory"]))
        if (
            record.get("contract") != "retention_c4_failed_attempt_archive_v1"
            or record.get("run_id") != run_root.name
            or source.parent.resolve() != run_root.resolve()
            or source.name not in {"outer_evaluation", "deployment"}
            or archive.parent.resolve() != run_root.resolve()
            or not archive.name.startswith(source.name + "_failed_attempt_")
            or archive.is_symlink()
            or not archive.is_dir()
        ):
            raise RuntimeError("C4 recovery archive does not belong to this run.")
        receipt = _read_json(
            _c4_io_path(archive / "c4_resume_archive.json"), "C4 recovery archive"
        )
        if receipt != record:
            raise RuntimeError("C4 recovery archive receipt differs from its declaration.")
        key = str(archive.resolve())
        if key not in seen:
            result.append(record)
            seen.add(key)
    return result




def _normalized_bundle_reference(value: Any, *, field: str) -> str:
    """Compare portable inventory paths with Windows-native references.


    Older Windows exporters saved runtime_sources using str(Path), while the
    artifact inventory uses forward slashes. Normalize only the lookup value;
    the saved manifest and its identity hash must remain unchanged.
    """
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RuntimeError(f"Saved C4 bundle has an invalid file reference: {field}={value!r}")
    relative = value.replace("\\", "/")
    if (relative.startswith("/") or ":" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise RuntimeError(f"Saved C4 bundle has an unsafe relative file reference: {field}={value!r}")
    return relative




def _validate_reusable_bundle(
    *, source: Path, family: str, recipe: FamilyRecipe,
    training_frame: pd.DataFrame, target_col: str, cohort_col: str,
    identity_columns: Sequence[str], source_metadata: Mapping[str, Any],
    source_code_paths: Mapping[str, Path], deployment_bundle: Any,
) -> tuple[dict[str, Any], Path]:
    """Verify saved fit identity before reuse; never substitute a different fit."""
    root = _c4_io_path(source)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"C4 recovery source is not a regular model directory: {source}")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError(f"C4 recovery refuses linked bundle artifacts: {source}")
    manifest_path = _find_bundle_manifest(root)
    declared = _read_json(manifest_path, "saved C4 family manifest")
    inventory = declared.get("artifact_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise RuntimeError("Saved C4 model has no artifact inventory.")
    inventoried: set[str] = set()
    for item in inventory:
        if not isinstance(item, dict):
            raise RuntimeError("Saved C4 artifact inventory contains an invalid entry.")
        relative = str(item.get("relative_path", ""))
        path = Path(relative)
        if (not relative or "\\" in relative or ":" in relative or path.is_absolute()
                or ".." in path.parts or relative in inventoried):
            raise RuntimeError("Saved C4 artifact inventory has an invalid relative path.")
        inventoried.add(relative)
        digest = item.get("sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise RuntimeError(f"Saved C4 artifact inventory has no valid SHA-256 for {relative!r}.")
    model_files = (
        declared.get("model_files", []) if family == "xgboost"
        else [declared.get("model_file")]
    )
    if not isinstance(model_files, list) or not model_files:
        raise RuntimeError("Saved C4 bundle has no model-file list.")
    declared_runtime = declared.get("runtime_sources")
    if not isinstance(declared_runtime, dict) or not declared_runtime:
        raise RuntimeError("Saved C4 bundle has no runtime-source references.")
    required_files = {
        "preprocessor_file": declared.get("preprocessor_file"),
        **{f"model_files[{index}]": value for index, value in enumerate(model_files)},
        **{f"runtime_sources.{label}": value for label, value in declared_runtime.items()},
    }
    for field, value in required_files.items():
        relative = _normalized_bundle_reference(value, field=field)
        if relative not in inventoried:
            raise RuntimeError(
                f"Saved C4 required file is absent from the integrity inventory: "
                f"{field}={value!r} (inventory path {relative!r})."
            )
    manifest, fitted = deployment_bundle.load_bundle(root, verify_hashes=True)


    def same(label: str, observed: Any, expected: Any) -> None:
        if _json_sha256(observed) != _json_sha256(expected):
            raise RuntimeError(f"Saved C4 bundle differs from the current {label}: {source}")


    same("model family", manifest.get("model_family"), family)
    same("target column", manifest.get("target_column"), target_col)
    same("prediction target", manifest.get("prediction_target"),
         "churn_probability" if family == "xgboost" else "retention_probability")
    same("feature order", manifest.get("features"), list(recipe.features))
    same("selected model recipe", manifest.get("model_recipe"), recipe.model_recipe)
    same("feature routing", fitted.get("feature_types"), recipe.feature_types)
    same("rare-category frequency", fitted.get("rare_min_frequency"), recipe.rare_min_frequency)
    same("training weights", manifest.get("sample_weight"), {"applied": False, "normalization": None})
    expected_population = deployment_bundle._population_summary(
        training_frame, target_col=target_col, cohort_col=cohort_col,
        identity_columns=identity_columns,
    )
    same("exact training population", manifest.get("training_population"), expected_population)
    expected_metadata = dict(source_metadata)
    observed_metadata = dict(manifest.get("source_metadata", {}))
    if observed_metadata.get("pairing_identity") != expected_metadata.get("pairing_identity"):
        # The earlier champion-only C4 omitted fitted_model_family from its
        # final pairing hash. All statistical inputs still have to match.
        if expected_metadata.get("bundle_role") == "FINAL_REFIT":
            legacy_identity = _json_sha256({
                "source_run_id": expected_metadata["source_run_id"],
                "profile_hash": expected_metadata["profile_hash"],
                "source_data_sha256": expected_metadata["source_data_identity"]["sha256"],
                "excluded_holdout": expected_metadata["excluded_holdout"],
                "evaluation_training_population": expected_metadata["evaluation_training_population"],
                "shared_state_mapping_sha256": expected_metadata["shared_state_grouping"]["mapping_sha256"],
                "pairing_scope": {
                    "bundle_role": "FINAL_REFIT",
                    "base_training_index_sha256": expected_metadata["evaluation_training_population"]["index_sha256"],
                    "selected_model_family": family,
                },
            })
            if observed_metadata.get("pairing_identity") == legacy_identity:
                expected_metadata["pairing_identity"] = legacy_identity
    same("fold, history, data and holdout metadata", observed_metadata, expected_metadata)
    for key, expected in deployment_bundle._promoted_source_metadata(expected_metadata).items():
        same(f"promoted {key}", manifest.get(key), expected)
    for key, expected in expected_metadata["shared_state_grouping"].items():
        if key in {"derived_feature", "source_feature"}:
            continue
        same(f"fitted state grouping {key}", fitted["state_contract"].get(key), expected)
    if family == "xgboost":
        same("ensemble seeds", manifest.get("seeds"), list(recipe.seeds))
        same("boosting rounds", manifest.get("boosting_rounds"), recipe.boosting_rounds)
        same("seed model count", len(model_files), len(recipe.seeds))
        same("ensemble aggregation", manifest.get("ensemble_aggregation"), "arithmetic_mean_churn_probability")
    runtime_sources = {"deployment_bundle": Path(deployment_bundle.__file__), **source_code_paths}
    for label, current in runtime_sources.items():
        relative = _normalized_bundle_reference(
            manifest.get("runtime_sources", {}).get(label), field=f"runtime_sources.{label}"
        )
        if relative not in inventoried:
            raise RuntimeError(f"Saved C4 bundle has no verified runtime source: {label}")
        same(f"runtime source {label}", _sha256(root / relative), _sha256(Path(current)))
    return manifest, manifest_path




def _reuse_family_bundle(
    *, destination: Path, recovery_archives: Sequence[Mapping[str, Any]],
    recovery_records: list[dict[str, Any]], **validation_kwargs: Any,
) -> tuple[dict[str, Any], Path] | None:
    """Copy a verified archived fit into the normal pipeline output location."""
    for record in recovery_archives:
        try:
            relative = destination.relative_to(Path(record["source_directory"]))
        except ValueError:
            continue
        source = Path(record["archive_directory"]) / relative
        if not _c4_io_path(source).exists():
            continue
        ancestor = Path(record["archive_directory"])
        for component in relative.parts:
            ancestor = ancestor / component
            if _c4_io_path(ancestor).is_symlink():
                raise RuntimeError("C4 recovery refuses a linked model-directory ancestor.")
        try:
            manifest, source_manifest = _validate_reusable_bundle(
                source=source, **validation_kwargs,
            )
        except Exception as exc:
            # An existing but incompatible fit must not trigger hours of
            # replacement training under a command intended to resume.
            raise RuntimeError(
                f"Saved C4 model failed recovery verification; no replacement "
                f"fit was started for {destination}: {exc}"
            ) from exc
        destination_io = _c4_io_path(destination)
        destination_io.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(destination_io):
            raise FileExistsError(f"C4 recovery destination already exists: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=".r_", dir=destination_io.parent))
        try:
            shutil.copytree(_c4_io_path(source), temporary, dirs_exist_ok=True)
            for item in manifest["artifact_inventory"]:
                if _sha256(temporary / item["relative_path"]) != item["sha256"]:
                    raise RuntimeError("Recovered C4 artifact failed its copy integrity check.")
            if _sha256(temporary / source_manifest.name) != _sha256(source_manifest):
                raise RuntimeError("Recovered C4 manifest failed its copy integrity check.")
            if os.path.lexists(destination_io):
                raise FileExistsError(f"C4 recovery destination already exists: {destination}")
            os.rename(temporary, destination_io)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        manifest_path = destination / source_manifest.name
        recovery_records.append({
            "action": "REUSED_VERIFIED_BUNDLE", "model_family": manifest["model_family"],
            "source_bundle": str(source), "bundle_path": str(destination),
            "manifest_sha256": _sha256(_c4_io_path(manifest_path)),
            "training_rows": manifest["training_population"]["rows"],
            "outer_fold": manifest["source_metadata"]["outer_fold"],
            "bundle_role": manifest["source_metadata"]["bundle_role"],
        })
        print(f"REUSED VERIFIED C4 MODEL (NO FIT): {destination}", flush=True)
        return manifest, manifest_path
    return None




def _write_c4_recovery_report(output_dir: Path, records: Sequence[Mapping[str, Any]]) -> None:
    _atomic_json(output_dir / "c4_recovery_report.json", {
        "contract": "retention_c4_saved_fit_recovery_v1",
        "reused_model_bundles": sum(row["action"] == "REUSED_VERIFIED_BUNDLE" for row in records),
        "new_model_bundles": sum(row["action"] == "FITTED_NEW_BUNDLE" for row in records),
        "predictions_and_metrics_regenerated": True,
        "feature_selection_or_optuna_rerun": False,
        "final_holdout_rows_scored": 0,
        "bundles": list(records),
    })




def _metric_rows(
    predictions: pd.DataFrame, model_diagnostics: Any
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for family in MODEL_FAMILIES:
        family_rows = predictions.loc[predictions["model_family"].eq(family)]
        for fold, part in family_rows.groupby("fold", sort=True, observed=True):
            metric_input = part.rename(
                columns={
                    "y_retention": "actual_retention",
                    "p_retention": "predicted_retention",
                }
            )
            metrics = model_diagnostics.prediction_metrics(
                metric_input, "predicted_retention"
            )
            fold_rows.append(
                {"model_family": family, "fold": int(fold), **metrics}
            )
        family_fold = pd.DataFrame(
            [row for row in fold_rows if row["model_family"] == family]
        )
        pooled = model_diagnostics.prediction_metrics(
            family_rows.rename(
                columns={
                    "y_retention": "actual_retention",
                    "p_retention": "predicted_retention",
                }
            ),
            "predicted_retention",
        )
        row: dict[str, Any] = {
            "model_family": family,
            "outer_folds": int(family_fold["fold"].nunique()),
            "primary_aggregation": "mean_and_sample_std_across_outer_folds",
            "pooled_ranking_metrics_published": False,
        }
        for metric in (
            "retention_roc_auc",
            "churn_average_precision",
            "churn_ap_skill",
            "churn_recall_10pct",
            "top10_lift",
            "log_loss",
            "brier_score",
            "calibration_gap",
        ):
            row[f"mean_{metric}"] = float(family_fold[metric].mean())
            row[f"std_{metric}"] = float(family_fold[metric].std(ddof=1))
        # Proper scoring/calibration summaries can be pooled without ranking
        # scores from independently refitted models against one another.  Raw
        # pooled ROC/AP/Lift are intentionally omitted.
        for metric in ("log_loss", "brier_score", "calibration_gap"):
            row[f"pooled_{metric}"] = float(pooled[metric])
        summary_rows.append(row)
    return pd.DataFrame(fold_rows), pd.DataFrame(summary_rows)




def _tenure_metric_rows(
    predictions: pd.DataFrame, model_diagnostics: Any
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [
        (str(int(fold)), part)
        for fold, part in predictions.groupby("fold", sort=True, observed=True)
    ]
    for scope, scoped in scopes:
        for (tenure, family), part in scoped.groupby(
            ["tenure_group", "model_family"], sort=True, observed=True
        ):
            target = pd.to_numeric(part["y_retention"], errors="raise").astype(int)
            base = {
                "fold": scope,
                "tenure_group": str(tenure),
                "model_family": str(family),
                "rows": int(len(part)),
            }
            if set(target.unique()) != {0, 1}:
                rows.append({**base, "status": "ONE_TARGET_CLASS"})
                continue
            metrics = model_diagnostics.prediction_metrics(
                part.rename(
                    columns={
                        "y_retention": "actual_retention",
                        "p_retention": "predicted_retention",
                    }
                ),
                "predicted_retention",
            )
            rows.append({**base, "status": "SCORED", **metrics})
    result = pd.DataFrame(rows)
    scored = result.loc[result["status"].eq("SCORED")].copy()
    metric_columns = [
        column
        for column in (
            "retention_roc_auc",
            "churn_average_precision",
            "churn_ap_skill",
            "churn_recall_10pct",
            "top10_lift",
            "log_loss",
            "brier_score",
            "calibration_gap",
        )
        if column in scored
    ]
    aggregate_rows: list[dict[str, Any]] = []
    for (tenure, family), part in scored.groupby(
        ["tenure_group", "model_family"], sort=True, observed=True
    ):
        aggregate: dict[str, Any] = {
            "fold": "MEAN_OF_FOLDS",
            "tenure_group": str(tenure),
            "model_family": str(family),
            "rows": int(part["rows"].sum()),
            "status": "AGGREGATED_FOLD_METRICS",
            "outer_folds": int(part["fold"].nunique()),
        }
        for column in metric_columns:
            aggregate[column] = float(part[column].mean())
            aggregate[f"{column}_fold_std"] = float(part[column].std(ddof=1))
        aggregate_rows.append(aggregate)
    if aggregate_rows:
        result = pd.concat([result, pd.DataFrame(aggregate_rows)], ignore_index=True)
    return result




def _population_contract(
    frame: pd.DataFrame,
    *,
    indices: Sequence[Any] | np.ndarray,
    cohort_col: str,
    identity_columns: Sequence[str],
    deployment_bundle: Any,
    definition: str,
) -> dict[str, Any]:
    part = frame.loc[np.asarray(indices)]
    dates = pd.to_datetime(part[cohort_col], errors="raise")
    return {
        "definition": definition,
        "rows": int(len(part)),
        "cohort_start": str(dates.min().date()),
        "cohort_end": str(dates.max().date()),
        "cohort_months": int(dates.dt.to_period("M").nunique()),
        "index_sha256": _index_sha256(part.index.to_numpy()),
        "population_sha256": deployment_bundle.population_sha256(
            part, identity_columns
        ),
        "identity_columns": list(identity_columns),
    }




def _source_metadata(
    *,
    params: Any,
    run_id: str,
    run_root: Path,
    source_data_identity: Mapping[str, Any],
    source_tenure: str,
    excluded_holdout: Mapping[str, Any],
    sealed_final_holdout: Mapping[str, Any],
    evaluation_training_population: Mapping[str, Any],
    cohort_col: str,
    shared_state_contract: Mapping[str, Any],
    pairing_scope: Mapping[str, Any],
    bundle_role: str,
    outer_fold: int | None,
    history_audit: Mapping[str, Any],
) -> dict[str, Any]:
    pairing_payload = {
        "source_run_id": run_id,
        "profile_hash": params.PROFILE_HASH,
        "source_data_sha256": str(source_data_identity["sha256"]),
        "excluded_holdout": excluded_holdout,
        "evaluation_training_population": evaluation_training_population,
        "shared_state_mapping_sha256": shared_state_contract["mapping_sha256"],
        "pairing_scope": pairing_scope,
    }
    return {
        "pairing_identity": _json_sha256(pairing_payload),
        "source_tenure": source_tenure,
        "selected_tenure_groups": list(params.PROFILE_PAYLOAD["selected_tenure_groups"]),
        "profile": dict(params.PROFILE_PAYLOAD),
        "profile_hash": params.PROFILE_HASH,
        "source_run_id": run_id,
        "source_run_path": str(run_root.resolve()),
        "source_data_identity": dict(source_data_identity),
        "excluded_holdout": dict(excluded_holdout),
        "sealed_final_holdout": dict(sealed_final_holdout),
        "evaluation_training_population": dict(evaluation_training_population),
        "cohort_column": cohort_col,
        "shared_state_grouping": dict(shared_state_contract),
        "bundle_role": bundle_role,
        "outer_fold": outer_fold,
        "history_application": dict(history_audit),
        "outer_validation_used_for_tuning_or_early_stopping": False,
        "prior_outer_validation_rows_eligible_for_final_refit": bool(
            bundle_role in {"FINAL_REFIT", "FINAL_REFIT_COMPARISON"}
        ),
        "final_holdout_rows_scored": 0,
    }




def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-gpu", action="store_true",
        help="Test a tiny CUDA fit and prediction, then exit without reading project data or changing outputs.",
    )
    parser.add_argument(
        "--score-final-holdout",
        action="store_true",
        help=(
            "Reserved governance guard. c4 refuses this option because opening "
            "the sealed holdout must be a separate one-time evaluation stage."
        ),
    )
    return parser.parse_args(argv)




def _normalize_xgboost_device(value: str) -> str:
    device = str(value).strip().lower()
    if device == "gpu" or device.startswith("gpu:"):
        device = "cuda" + device[3:]
    if device != "cpu" and re.fullmatch(r"cuda(?::[0-9]+)?", device) is None:
        raise ValueError("RETENTION_XGB_DEVICE must be 'cpu', 'cuda', or 'cuda:<GPU number>'.")
    return device




def _require_booster_device(booster: Any, requested: str) -> str:
    actual = str(json.loads(booster.save_config())["learner"]["generic_param"]["device"])
    if requested.startswith("cuda") and (
        not actual.startswith("cuda") or (":" in requested and actual != requested)
    ):
        raise RuntimeError(
            f"GPU execution requested ({requested}), but XGBoost selected {actual}. "
            "Check the NVIDIA GPU/driver and the installed XGBoost CUDA build. "
            "The pipeline will not silently continue on CPU."
        )
    return actual




def _check_xgboost_device(device: str, *, xgb: Any) -> dict[str, Any]:
    requested = _normalize_xgboost_device(device)
    report = {"requested_device": requested, "xgboost_version": str(xgb.__version__),
              "new_fits_use_requested_device": True, "saved_models_retrained": False}
    if requested == "cpu":
        return {**report, "actual_device": "cpu", "gpu_smoke_test": "NOT_REQUESTED"}
    # Probe both training and prediction before touching any C4 output tree.
    matrix = xgb.DMatrix(np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
                         label=np.asarray([0, 1, 0, 1], dtype=np.float32))
    try:
        booster = xgb.train({"objective": "binary:logistic", "tree_method": "hist",
                             "device": requested, "max_depth": 1}, matrix, num_boost_round=1)
        actual = _require_booster_device(booster, requested)
        prediction = np.asarray(booster.predict(matrix))
        _require_booster_device(booster, requested)
        if prediction.shape != (4,) or not np.isfinite(prediction).all():
            raise RuntimeError("GPU smoke-test predictions have an invalid shape/value.")
    except Exception as exc:
        raise RuntimeError(f"XGBoost GPU preflight failed before project data or outputs were touched: {exc}") from exc
    return {**report, "actual_device": actual, "gpu_smoke_test": "PASSED"}




def _iter_xgboost_scores_on_device(
    bundle_path: Path, frame: pd.DataFrame, *, batch_size: int,
    deployment_bundle: Any, device: str | None,
):
    """Score frozen trees on CUDA, keeping the saved preprocessing unchanged.


    Explicit DMatrix prediction avoids inplace_predict's CPU-input/GPU-device
    fallback. A CPU-trained tree remains the same tree; this does not refit it.
    """
    if device is None or _normalize_xgboost_device(device) == "cpu":
        yield from deployment_bundle.iter_xgboost_bundle_scores(
            bundle_path, frame, batch_size=batch_size, verify_hashes=True,
        )
        return
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("XGBoost scoring batch_size must be a positive integer.")
    import xgboost as xgb
    requested = _normalize_xgboost_device(device)
    root = Path(bundle_path).resolve()
    manifest, fitted = deployment_bundle.load_bundle(root, verify_hashes=True)
    if manifest.get("model_family") != "xgboost" or not manifest.get("model_files"):
        raise ValueError("GPU scoring requires a nonempty XGBoost model bundle.")
    models = []
    for name in manifest["model_files"]:
        model = xgb.Booster()
        model.load_model(str(root / name))
        model.set_param({"device": requested})
        models.append(model)
    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        transformed = deployment_bundle.transform_frame(fitted, frame.iloc[start:stop])
        matrix = xgb.DMatrix(transformed)
        probability_sum = np.zeros(stop - start, dtype=float)
        for model in models:
            probability = np.asarray(model.predict(matrix), dtype=float)
            _require_booster_device(model, requested)
            if probability.shape != (stop - start,) or not np.isfinite(probability).all():
                raise RuntimeError("Saved XGBoost model returned invalid GPU predictions.")
            probability_sum += probability
        yield start, stop, np.clip(probability_sum / len(models), 1e-7, 1.0 - 1e-7)




def _declared_input_sha256(payload: Mapping[str, Any], *, label: str) -> str:
    for key in ("input", "data", "source_data_identity"):
        value = payload.get(key)
        if isinstance(value, Mapping) and str(value.get("sha256", "")).strip():
            return str(value["sha256"]).strip().lower()
    for key in ("input_sha256", "data_sha256", "source_data_sha256"):
        if str(payload.get(key, "")).strip():
            return str(payload[key]).strip().lower()
    raise ValueError(f"{label} does not declare its input snapshot SHA-256.")




def _load_recipes(
    run_root: Path, params: Any, *, expected_input_sha256: str
) -> dict[str, FamilyRecipe]:
    logistic_dir = run_root / "logistic"
    xgboost_dir = run_root / "xgboost"
    logistic = _read_json(logistic_dir / "recommended_model.json", "Logistic recipe")
    xgboost = _read_json(xgboost_dir / "selected_features.json", "XGBoost recipe")
    logistic_types = _read_json(
        logistic_dir / "resolved_logistic_model_feature_types.json",
        "resolved Logistic feature types",
    )
    xgboost_types = _read_json(
        xgboost_dir / "xgb_resolved_model_feature_types.json",
        "resolved XGBoost feature types",
    )
    logistic_run_info = _read_json(logistic_dir / "run_info.json", "Logistic run info")
    xgboost_run_info = _read_json(xgboost_dir / "run_info.json", "XGBoost run info")
    for label, payload in (
        ("Logistic", logistic),
        ("XGBoost", xgboost),
        ("Logistic run info", logistic_run_info),
        ("XGBoost run info", xgboost_run_info),
    ):
        if int(payload.get("final_holdout_rows_scored", -1)) != 0:
            raise RuntimeError(f"{label} recipe does not prove a sealed final holdout.")
    for label, payload in (
        ("Logistic", logistic_run_info),
        ("XGBoost", xgboost_run_info),
    ):
        if str(payload.get("profile_hash", "")) != str(params.PROFILE_HASH):
            raise RuntimeError(f"{label} recipe profile hash does not match runtime.")
        declared_sha = _declared_input_sha256(payload, label=f"{label} run_info")
        if declared_sha != str(expected_input_sha256).strip().lower():
            raise RuntimeError(
                f"{label} recipe input SHA-256 differs from the c4 snapshot: "
                f"declared={declared_sha}; actual={expected_input_sha256}."
            )
    logistic_features = tuple(str(value).strip().upper() for value in logistic["features"])
    xgboost_features = tuple(
        str(value).strip().upper() for value in xgboost["selected_features"]
    )
    for family, features, types in (
        ("Logistic", logistic_features, logistic_types),
        ("XGBoost", xgboost_features, xgboost_types),
    ):
        if not features or len(features) != len(set(features)):
            raise ValueError(f"{family} recipe features must be nonempty and unique.")
        missing = sorted(set(features).difference(types))
        if missing:
            raise ValueError(f"{family} selected features have no resolved types: {missing}")
    logistic_history = dict(logistic["selected_history"])
    xgboost_history = dict(xgboost["selected_history"])
    logistic_months = validated_history_months(logistic_history, family="Logistic")
    xgboost_months = validated_history_months(xgboost_history, family="XGBoost")
    try:
        c_value = float(logistic["C"])
        rounds = int(xgboost["recommended_boosting_rounds_from_inner_confirmation"])
        seeds = tuple(int(value) for value in xgboost["seeds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Selected family recipes are incomplete.") from exc
    if not math.isfinite(c_value) or c_value <= 0 or rounds < 1:
        raise ValueError("Selected C and boosting rounds must be positive and finite.")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Selected XGBoost seeds must be nonempty and unique.")
    logistic_solver = str(logistic.get("solver", "")).strip().lower()
    logistic_class_weight = logistic.get("class_weight")
    if logistic_solver != "lbfgs" or logistic_class_weight is not None:
        raise ValueError(
            "The frozen Logistic recipe must use solver='lbfgs' and "
            f"class_weight=None; observed solver={logistic_solver!r}, "
            f"class_weight={logistic_class_weight!r}."
        )
    try:
        logistic_max_iter = int(logistic_run_info["max_iter_ridge"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Logistic run_info.json must record the selector's max_iter_ridge."
        ) from exc
    if logistic_max_iter < 1:
        raise ValueError("The frozen Logistic max_iter_ridge must be positive.")
    logistic_seed = int(
        logistic_run_info.get("outer_split_settings", {}).get(
            "seed", getattr(params, "SEED", 42)
        )
    )
    rare = float(
        xgboost_run_info.get("arguments", {}).get("rare_min_frequency", 0.001)
    )
    lr_rare = float(logistic.get("rare_min_frequency", 0.001))
    return {
        "logistic": FamilyRecipe(
            family="logistic",
            features=logistic_features,
            feature_types={feature: str(logistic_types[feature]) for feature in logistic_features},
            history=logistic_history,
            history_months=logistic_months,
            rare_min_frequency=lr_rare,
            model_recipe={
                "family": "ridge_logistic",
                "penalty": "l2",
                "C": c_value,
                "solver": logistic_solver,
                "class_weight": logistic_class_weight,
                "max_iter": logistic_max_iter,
                "tol": 1e-4,
                "random_seed": logistic_seed,
                "selected_history": logistic_history,
                "recipe_source": str((logistic_dir / "recommended_model.json").resolve()),
                "recipe_source_sha256": _sha256(
                    logistic_dir / "recommended_model.json"
                ),
                "feature_types_source_sha256": _sha256(
                    logistic_dir / "resolved_logistic_model_feature_types.json"
                ),
            },
        ),
        "xgboost": FamilyRecipe(
            family="xgboost",
            features=xgboost_features,
            feature_types={feature: str(xgboost_types[feature]) for feature in xgboost_features},
            history=xgboost_history,
            history_months=xgboost_months,
            rare_min_frequency=rare,
            model_recipe={
                **dict(xgboost["selected_model_spec"]),
                "selected_history": xgboost_history,
                "boosting_rounds": rounds,
                "early_stopping_rounds": None,
                "outer_validation_used_for_early_stopping": False,
                "recipe_source": str((xgboost_dir / "selected_features.json").resolve()),
                "recipe_source_sha256": _sha256(
                    xgboost_dir / "selected_features.json"
                ),
                "feature_types_source_sha256": _sha256(
                    xgboost_dir / "xgb_resolved_model_feature_types.json"
                ),
            },
            seeds=seeds,
            boosting_rounds=rounds,
        ),
    }




def _logistic_factory(recipe: FamilyRecipe, seed: int) -> Any:
    return LogisticRegression(
        penalty=str(recipe.model_recipe["penalty"]),
        C=float(recipe.model_recipe["C"]),
        solver=str(recipe.model_recipe["solver"]),
        max_iter=int(recipe.model_recipe["max_iter"]),
        tol=float(recipe.model_recipe["tol"]),
        class_weight=recipe.model_recipe["class_weight"],
        random_state=int(recipe.model_recipe.get("random_seed", seed)),
        n_jobs=-1,
    )




def _xgboost_factory(
    recipe: FamilyRecipe, *, device: str, n_jobs: int, seed: int, rounds: int
) -> Any:
    import xgboost as xgb


    spec = recipe.model_recipe
    return xgb.XGBClassifier(
        objective="binary:logistic",
        booster="gbtree",
        tree_method="hist",
        device=str(device),
        n_estimators=int(rounds),
        max_depth=int(spec["max_depth"]),
        learning_rate=float(spec["learning_rate"]),
        min_child_weight=float(spec["min_child_weight"]),
        gamma=float(spec["gamma"]),
        subsample=float(spec["subsample"]),
        colsample_bytree=float(spec["colsample_bytree"]),
        reg_lambda=float(spec["reg_lambda"]),
        reg_alpha=float(spec["reg_alpha"]),
        grow_policy=str(spec.get("grow_policy", "depthwise")),
        max_leaves=int(spec.get("max_leaves", 0)),
        max_bin=int(spec.get("max_bin", 256)),
        eval_metric="aucpr",
        random_state=int(seed),
        n_jobs=int(n_jobs),
        verbosity=0,
    )




def _export_family_bundle(
    *,
    family: str,
    destination: Path,
    training_frame: pd.DataFrame,
    recipe: FamilyRecipe,
    target_col: str,
    cohort_col: str,
    identity_columns: Sequence[str],
    modeling_utils: Any,
    deployment_bundle: Any,
    shared_state_artifact: Any,
    source_metadata: Mapping[str, Any],
    source_code_paths: Mapping[str, Path],
    params: Any,
    recovery_archives: Sequence[Mapping[str, Any]] = (),
    recovery_records: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], Path]:
    if recovery_archives:
        recovered = _reuse_family_bundle(
            destination=destination, recovery_archives=recovery_archives,
            recovery_records=recovery_records if recovery_records is not None else [],
            family=family, recipe=recipe, training_frame=training_frame,
            target_col=target_col, cohort_col=cohort_col, identity_columns=identity_columns,
            source_metadata=source_metadata, source_code_paths=source_code_paths,
            deployment_bundle=deployment_bundle,
        )
        if recovered is not None:
            return recovered
    print(f"FITTING C4 MODEL: {family}; {len(training_frame):,} rows; {destination}", flush=True)
    if family == "logistic":
        manifest = deployment_bundle.export_logistic_source_bundle(
            destination=destination,
            training_frame=training_frame,
            features=recipe.features,
            feature_types=recipe.feature_types,
            rare_min_frequency=recipe.rare_min_frequency,
            target_col=target_col,
            cohort_col=cohort_col,
            identity_columns=identity_columns,
            model_factory=lambda: _logistic_factory(recipe, int(params.SEED)),
            model_recipe=recipe.model_recipe,
            modeling_utils=modeling_utils,
            shared_state_artifact=shared_state_artifact,
            source_metadata=source_metadata,
            source_code_paths=source_code_paths,
            sample_weight=None,
        )
    else:
        import xgboost as xgb


        manifest = deployment_bundle.export_xgboost_source_bundle(
            destination=destination,
            training_frame=training_frame,
            features=recipe.features,
            feature_types=recipe.feature_types,
            rare_min_frequency=recipe.rare_min_frequency,
            target_col=target_col,
            cohort_col=cohort_col,
            identity_columns=identity_columns,
            seeds=recipe.seeds,
            boosting_rounds=int(recipe.boosting_rounds or 0),
            sample_weight=None,
            model_factory=lambda seed, rounds: _xgboost_factory(
                recipe,
                device=params.XGB_DEVICE,
                n_jobs=params.N_JOBS,
                seed=seed,
                rounds=rounds,
            ),
            model_recipe=recipe.model_recipe,
            modeling_utils=modeling_utils,
            shared_state_artifact=shared_state_artifact,
            source_metadata=source_metadata,
            source_code_paths=source_code_paths,
            xgboost_version=str(xgb.__version__),
        )
    manifest_path = _find_bundle_manifest(destination)
    if recovery_records is not None:
        recovery_records.append({
            "action": "FITTED_NEW_BUNDLE", "model_family": family,
            "bundle_path": str(destination), "training_rows": len(training_frame),
            "manifest_sha256": _sha256(manifest_path),
            "outer_fold": source_metadata["outer_fold"],
            "bundle_role": source_metadata["bundle_role"],
            "fit_device": str(params.XGB_DEVICE) if family == "xgboost" else "cpu",
        })
    return manifest, manifest_path




def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.score_final_holdout or os.environ.get(
        "RETENTION_SCORE_FINAL_HOLDOUT", ""
    ).strip().upper() in {"1", "TRUE", "Y", "YES"}:
        raise RuntimeError(
            "c4_evaluate_and_finalize.py never opens the final holdout. Freeze "
            "tenure grouping and model family first, then use a separately "
            "approved one-time final-test scorer."
        )


    if args.check_gpu:
        import xgboost as xgb
        device = _normalize_xgboost_device(os.environ.get("RETENTION_XGB_DEVICE", "cuda"))
        if device == "cpu":
            raise ValueError("--check-gpu requires RETENTION_XGB_DEVICE=cuda (or cuda:<GPU number>).")
        print(json.dumps(_check_xgboost_device(device, xgb=xgb), indent=2), flush=True)
        return


    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import params
    import xgboost as xgb
    from c_modeling import deployment_bundle
    from c_modeling import final_refit_diagnostics
    from c_modeling import model_diagnostics
    from c_modeling import modeling_utils
    from c_modeling import performance_plots


    run_id = os.environ.get("RETENTION_PIPELINE_RUN_ID", "").strip()
    if not run_id:
        raise RuntimeError("RETENTION_PIPELINE_RUN_ID is required for governed c4.")
    run_root = Path(params.PIPELINE_OUTPUT_ROOT) / run_id
    run_root = run_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Pipeline run directory does not exist: {run_root}")
    xgboost_device = _normalize_xgboost_device(params.XGB_DEVICE)
    runtime_execution = _check_xgboost_device(xgboost_device, xgb=xgb)
    diagnostic_batch_rows = int(
        os.environ.get("RETENTION_FINAL_REFIT_DIAGNOSTIC_BATCH_ROWS", "100000")
    )
    if diagnostic_batch_rows < 1:
        raise ValueError("RETENTION_FINAL_REFIT_DIAGNOSTIC_BATCH_ROWS must be positive.")
    print(f"C4 XGBOOST DEVICE: {runtime_execution['actual_device']}; "
          "saved trees are reused; missing models use this device.", flush=True)
    output_dir = run_root / "outer_evaluation"
    deployment_dir = run_root / "deployment"
    # Keep every c4 artifact inside its governed pipeline run so failed-run
    # archival/recovery cannot strand model directories in a second registry.
    fold_bundle_root = (output_dir / "fold_models").resolve()
    final_bundle_root = (deployment_dir / "final_model").resolve()
    failed_run_resume = os.environ.get(
        "RETENTION_RESUME_FAILED_RUN", ""
    ).strip() == "1"
    c4_recovery_archives: list[dict[str, Any]] = []
    recovery_sources: list[dict[str, Any]] = []
    recovery_records: list[dict[str, Any]] = []
    if failed_run_resume:
        prior_sources_path = output_dir / "c4_recovery_sources.json"
        if prior_sources_path.is_file():
            prior_sources = _read_json(prior_sources_path, "prior C4 recovery sources")
            recovery_sources = _validated_c4_archive_records(
                run_root, prior_sources["archives"],
            )
        c4_recovery_archives = _archive_partial_c4_attempt(
            run_root,
            recovery_attempt_id=os.environ.get("RETENTION_RESUME_ATTEMPT_ID", ""),
        )
        recovery_sources = _validated_c4_archive_records(
            run_root, [*c4_recovery_archives, *recovery_sources],
        )
    completion_markers = (
        output_dir / "run_status.json",
        output_dir / "outer_fold_bundle_manifest.json",
        deployment_dir / "final_model_manifest.json",
    )
    if any(path.exists() for path in completion_markers):
        raise FileExistsError(
            "Refusing to overwrite an existing c4 completion artifact. Start a "
            f"new pipeline run or remove only an explicitly abandoned c4 run: {run_id}."
        )
    if fold_bundle_root.exists() or final_bundle_root.exists():
        raise FileExistsError(
            "A c4 model-bundle destination already exists for this run; refusing "
            "to mix attempts."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    deployment_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "execution_device.json", runtime_execution)
    _atomic_json(output_dir / "c4_recovery_sources.json", {
        "run_id": run_id, "archives": recovery_sources,
    })
    if recovery_sources:
        print("C4 RECOVERY: verify and reuse saved fitted models; regenerate predictions and reports.", flush=True)
    _atomic_json(
        output_dir / "run_status.json.incomplete",
        {
            "status": "RUNNING",
            "run_id": run_id,
            "final_holdout_rows_scored": 0,
        },
    )


    input_path = Path(params.DOWNLOAD_PATH).resolve()
    source_data_identity = {
        "path": str(input_path),
        "bytes": int(input_path.stat().st_size),
        "sha256": _sha256(input_path),
    }
    preparation_dir = run_root / "preparation"
    preparation_info = _read_json(
        preparation_dir / "preparation_info.json", "preparation info"
    )
    if preparation_info.get("profile_hash") != params.PROFILE_HASH:
        raise RuntimeError("Preparation profile hash does not match runtime profile.")
    if int(preparation_info.get("final_holdout_rows_scored", -1)) != 0:
        raise RuntimeError("Preparation does not prove a sealed final holdout.")
    preparation_sha = str(preparation_info.get("input_sha256", "")).strip().lower()
    if not preparation_sha or preparation_sha != source_data_identity["sha256"]:
        raise RuntimeError(
            "Preparation input SHA-256 does not match the current c4 snapshot: "
            f"declared={preparation_sha!r}; actual={source_data_identity['sha256']}."
        )
    recipes = _load_recipes(
        run_root,
        params,
        expected_input_sha256=str(source_data_identity["sha256"]),
    )


    if str(getattr(params, "POPULATION_SCOPE_MODE", "")).upper() != "FULL":
        raise RuntimeError(
            "Governed outer performance and pre-holdout refit require the FULL "
            "population; sample and single-policy engineering runs cannot enter c4."
        )


    if int(params.ROLLING_VALIDATION_FOLDS) != len(OUTER_FOLD_IDS):
        raise RuntimeError(
            "Governed outer evaluation requires exactly five folds; "
            f"ROLLING_VALIDATION_FOLDS={params.ROLLING_VALIDATION_FOLDS}."
        )


    raw = modeling_utils.read_table(input_path)
    prepared, outer_folds, final_holdout_idx, split_summary, config = (
        modeling_utils.load_project_population_and_folds(
            raw,
            rolling_validation_folds=int(params.ROLLING_VALIDATION_FOLDS),
            validation_months=int(params.OUTER_VALIDATION_MONTHS),
            test_months=int(params.FINAL_HOLDOUT_MONTHS),
            min_train_months=int(params.MIN_OUTER_TRAIN_MONTHS),
            strict_dataset_contract=bool(params.STRICT_DATASET_CONTRACT),
        )
    )
    del raw
    actual_outer = modeling_utils.fold_definitions_frame(
        prepared,
        outer_folds,
        cohort_col=config.cohort_date_col,
        target_check_col=config.target_check_col,
        target_col=config.target_col,
    )
    modeling_utils.verify_fold_definition_contract(
        actual_outer,
        preparation_dir / "outer_fold_definitions.csv",
        label="c4 outer evaluation",
    )
    reconstructed_fold_ids = tuple(
        int(fold.get("fold", position))
        for position, fold in enumerate(outer_folds, start=1)
    )
    if reconstructed_fold_ids != OUTER_FOLD_IDS:
        raise RuntimeError(
            "Reconstructed outer folds must be exactly [1, 2, 3, 4, 5]; "
            f"observed={list(reconstructed_fold_ids)}."
        )


    identity_candidates = [
        str(getattr(params, "INDEX_COL", "")),
        str(config.policy_id_col),
        str(config.anchor_term_start_col),
        str(config.cohort_date_col),
    ]
    identity_columns = list(
        dict.fromkeys(value for value in identity_candidates if value in prepared)
    )
    all_row_ids = stable_row_ids(prepared, identity_columns)
    holdout = prepared.loc[np.asarray(final_holdout_idx)]
    holdout_dates = pd.to_datetime(holdout[config.cohort_date_col], errors="raise")
    holdout_cohort_months = int(holdout_dates.dt.to_period("M").nunique())
    if int(params.FINAL_HOLDOUT_MONTHS) != 6 or holdout_cohort_months != 6:
        raise RuntimeError(
            "Governed final holdout must contain exactly six cohort months: "
            f"FINAL_HOLDOUT_MONTHS={params.FINAL_HOLDOUT_MONTHS}; "
            f"observed={holdout_cohort_months}."
        )
    excluded_holdout = {
        "definition": "sealed final chronological holdout",
        "rows": int(len(holdout)),
        "cohort_months": holdout_cohort_months,
        "cohort_start": str(holdout_dates.min().date()),
        "cohort_end": str(holdout_dates.max().date()),
        "identity_columns": identity_columns,
        "population_sha256": deployment_bundle.population_sha256(
            holdout, identity_columns
        ),
        "row_index_sha256": _index_sha256(final_holdout_idx),
        "label_availability_buffer_days": int(
            params.LABEL_AVAILABILITY_BUFFER_DAYS
        ),
        "final_training_label_cutoff_exclusive": str(
            holdout_dates.min().date()
        ),
        "pre_holdout_label_unavailable_rows_excluded": int(
            len(prepared)
            - len(final_holdout_idx)
            - int(split_summary["evaluation_train"]["rows"])
        ),
        "policy_cluster_column": str(config.policy_id_col),
        "identity_column_roles": {
            "policy_cluster": str(config.policy_id_col),
            "cohort": str(config.cohort_date_col),
        },
        "cohort_column": str(config.cohort_date_col),
        "target_scored": False,
        "used_for_preprocessing_or_fit": False,
        "used_for_tuning_or_selection": False,
        "target_physically_present_in_snapshot": True,
        "target_used_for_fit_tuning_family_selection_or_performance_metrics": False,
        "predictions_generated": False,
        "seal_type": "LOGICAL_NO_PREDICTIONS_OR_METRICS",
        "rows_scored": 0,
    }


    source_tenure_groups = list(params.PROFILE_PAYLOAD["selected_tenure_groups"])
    source_tenure = (
        str(source_tenure_groups[0])
        if len(source_tenure_groups) == 1
        else "ALL_SELECTED"
    )
    source_code_paths = {
        "modeling_utils": Path(modeling_utils.__file__).resolve(),
    }
    prediction_parts: list[pd.DataFrame] = []
    fold_manifest_rows: list[dict[str, Any]] = []
    history_audit_rows: list[dict[str, Any]] = []


    for position, fold in enumerate(outer_folds, start=1):
        fold_number = int(fold.get("fold", position))
        print(f"OUTER FOLD {fold_number}/{len(outer_folds)}", flush=True)
        base_train_idx = np.asarray(fold["train_idx"])
        validation_idx = np.asarray(fold["validation_idx"])
        if np.intersect1d(validation_idx, final_holdout_idx).size:
            raise AssertionError("Final holdout entered an outer validation fold.")
        base_population = _population_contract(
            prepared,
            indices=base_train_idx,
            cohort_col=config.cohort_date_col,
            identity_columns=identity_columns,
            deployment_bundle=deployment_bundle,
            definition=f"outer fold {fold_number} label-mature base training partition",
        )
        family_train: dict[str, np.ndarray] = {}
        family_audit: dict[str, dict[str, Any]] = {}
        for family, recipe in recipes.items():
            indices, audit = modeling_utils.select_recent_mature_training_indices(
                prepared,
                base_train_idx,
                cohort_col=config.cohort_date_col,
                months=recipe.history_months,
            )
            if not bool(audit["requested_months_fully_available"]):
                raise RuntimeError(
                    f"Outer fold {fold_number} does not contain the selected "
                    f"{recipe.history_months}-month mature history for {family}."
                )
            family_train[family] = indices
            family_audit[family] = {
                **audit,
                "history_id": recipe.history["history_id"],
                "model_family": family,
                "fold": fold_number,
                "sample_weight_applied": False,
            }
            history_audit_rows.append(dict(family_audit[family]))
        shared_idx = np.intersect1d(
            family_train["logistic"], family_train["xgboost"], assume_unique=False
        )
        if not len(shared_idx):
            raise RuntimeError(f"Outer fold {fold_number} has no shared state-fit rows.")
        shared_state, state_contract = deployment_bundle.fit_shared_state_artifact(
            training_frame=prepared.loc[shared_idx],
            target_col=config.target_col,
            modeling_utils=modeling_utils,
        )
        pairing_scope = {
            "bundle_role": "OUTER_FOLD_EVALUATION",
            "outer_fold": fold_number,
            "base_training_index_sha256": base_population["index_sha256"],
            "validation_index_sha256": _index_sha256(validation_idx),
        }
        family_refs: dict[str, Any] = {}
        validation = prepared.loc[validation_idx]
        validation_dates = pd.to_datetime(
            validation[config.cohort_date_col], errors="raise"
        )
        fold_excluded_holdout = {
            "definition": (
                f"outer fold {fold_number} validation window; excluded from "
                "preprocessing, fit, tuning, and early stopping"
            ),
            "rows": int(len(validation)),
            "cohort_start": str(validation_dates.min().date()),
            "cohort_end": str(validation_dates.max().date()),
            "identity_columns": identity_columns,
            "population_sha256": deployment_bundle.population_sha256(
                validation, identity_columns
            ),
            "row_index_sha256": _index_sha256(validation_idx),
            "label_availability_buffer_days": int(
                params.LABEL_AVAILABILITY_BUFFER_DAYS
            ),
            "final_training_label_cutoff_exclusive": str(
                pd.Timestamp(fold["label_cutoff"]).date()
            ),
            "pre_holdout_label_unavailable_rows_excluded": int(
                (
                    pd.to_datetime(
                        prepared.loc[base_train_idx, config.target_check_col],
                        errors="raise",
                    )
                    >= pd.Timestamp(fold["label_cutoff"])
                ).sum()
            ),
            "policy_cluster_column": str(config.policy_id_col),
            "identity_column_roles": {
                "policy_cluster": str(config.policy_id_col),
                "cohort": str(config.cohort_date_col),
            },
            "cohort_column": str(config.cohort_date_col),
            "target_scored": False,
            "used_for_preprocessing_or_fit": False,
            "used_for_tuning_or_selection": False,
        }
        common = pd.DataFrame(
            {
                "row_id": all_row_ids.loc[validation_idx].to_numpy(),
                "fold": fold_number,
                "cohort_date": pd.to_datetime(
                    validation[config.cohort_date_col], errors="raise"
                ).to_numpy(),
                "tenure_group": validation[config.tenure_col].astype(str).to_numpy(),
                "y_retention": pd.to_numeric(
                    validation[config.target_col], errors="raise"
                ).astype(int).to_numpy(),
            }
        )
        for column in identity_columns:
            if column not in common:
                common[column] = validation[column].to_numpy()
        for family in MODEL_FAMILIES:
            recipe = recipes[family]
            metadata = _source_metadata(
                params=params,
                run_id=run_id,
                run_root=run_root,
                source_data_identity=source_data_identity,
                source_tenure=source_tenure,
                excluded_holdout=fold_excluded_holdout,
                sealed_final_holdout=excluded_holdout,
                evaluation_training_population=base_population,
                cohort_col=config.cohort_date_col,
                shared_state_contract=state_contract,
                pairing_scope=pairing_scope,
                bundle_role="OUTER_FOLD_EVALUATION",
                outer_fold=fold_number,
                history_audit=family_audit[family],
            )
            destination = fold_bundle_root / f"fold_{fold_number:02d}" / family
            _, manifest_path = _export_family_bundle(
                family=family,
                destination=destination,
                training_frame=prepared.loc[family_train[family]],
                recipe=recipe,
                target_col=config.target_col,
                cohort_col=config.cohort_date_col,
                identity_columns=identity_columns,
                modeling_utils=modeling_utils,
                deployment_bundle=deployment_bundle,
                shared_state_artifact=shared_state,
                source_metadata=metadata,
                source_code_paths=source_code_paths,
                params=params,
                recovery_archives=recovery_sources,
                recovery_records=recovery_records,
            )
            _write_c4_recovery_report(output_dir, recovery_records)
            if family == "logistic":
                probability = deployment_bundle.score_logistic_bundle(
                    destination, validation, verify_hashes=True
                )
            else:
                probability = 1.0 - np.concatenate([
                    part for _, _, part in _iter_xgboost_scores_on_device(
                        destination, validation,
                        batch_size=diagnostic_batch_rows,
                        deployment_bundle=deployment_bundle, device=xgboost_device,
                    )
                ])
            family_prediction = common.copy()
            family_prediction.insert(4, "model_family", family)
            family_prediction["p_retention"] = np.clip(
                np.asarray(probability, dtype=float), 1e-7, 1.0 - 1e-7
            )
            prediction_parts.append(family_prediction)
            family_refs[family] = {
                "bundle_path": str(destination.resolve()),
                "manifest_path": str(manifest_path.resolve()),
                "sha256": _sha256(manifest_path),
                "training_rows": int(len(family_train[family])),
                "history_id": recipe.history["history_id"],
                "history_months": recipe.history_months,
            }
        fold_row = actual_outer.loc[actual_outer["fold"].eq(fold_number)].iloc[0]
        fold_manifest_rows.append(
            {
                "fold": fold_number,
                "validation_start": str(fold_row["validation_start"]),
                "validation_end": str(fold_row["validation_end"]),
                "label_cutoff": str(fold_row["label_cutoff"]),
                "validation_rows": int(len(validation_idx)),
                "validation_index_sha256": _index_sha256(validation_idx),
                "final_holdout_rows_scored": 0,
                "families": family_refs,
            }
        )


    predictions = pd.concat(prediction_parts, ignore_index=True)
    expected_prediction_rows = 2 * sum(
        len(np.asarray(fold["validation_idx"])) for fold in outer_folds
    )
    if len(predictions) != expected_prediction_rows:
        raise AssertionError("Outer OOF prediction row count is incomplete.")
    if predictions.duplicated(["model_family", "row_id"], keep=False).any():
        raise AssertionError(
            "Outer OOF predictions contain duplicate model-family/row identities."
        )
    alignment_columns = [
        "row_id",
        "fold",
        "cohort_date",
        "tenure_group",
        "y_retention",
        *identity_columns,
    ]
    alignment_columns = list(dict.fromkeys(alignment_columns))
    aligned_frames = {
        family: predictions.loc[
            predictions["model_family"].eq(family), alignment_columns
        ]
        .sort_values("row_id", kind="stable")
        .reset_index(drop=True)
        for family in MODEL_FAMILIES
    }
    if not aligned_frames["logistic"].equals(aligned_frames["xgboost"]):
        raise AssertionError(
            "Logistic and XGBoost outer OOF records differ by row, fold, "
            "cohort, tenure, target, or governed identity."
        )
    probability = pd.to_numeric(predictions["p_retention"], errors="raise")
    if not np.isfinite(probability).all() or not probability.between(0.0, 1.0).all():
        raise AssertionError("Outer OOF probabilities must be finite in [0, 1].")
    outer_prediction_path = output_dir / "outer_oof_predictions.parquet"
    _atomic_parquet(outer_prediction_path, predictions)
    fold_metrics, metric_summary = _metric_rows(predictions, model_diagnostics)
    selected_family, comparison, decision = choose_model_family(fold_metrics)
    tenure_metrics = _tenure_metric_rows(predictions, model_diagnostics)
    _atomic_csv(output_dir / "outer_fold_metrics.csv", fold_metrics)
    _atomic_csv(output_dir / "outer_metric_summary.csv", metric_summary)
    _atomic_csv(output_dir / "outer_tenure_metrics.csv", tenure_metrics)
    _atomic_csv(output_dir / "model_comparison.csv", comparison)
    decision_payload = {
        **decision,
        "contract": "retention_outer_family_selection_v1",
        "run_id": run_id,
        "profile": dict(params.PROFILE_PAYLOAD),
        "profile_hash": params.PROFILE_HASH,
        "population_scope_mode": "FULL",
        "outer_fold_metrics": str((output_dir / "outer_fold_metrics.csv").resolve()),
        "model_comparison": str((output_dir / "model_comparison.csv").resolve()),
        "final_holdout_rows_scored": 0,
    }
    _atomic_json(output_dir / "family_selection.json", decision_payload)
    plot_manifest = performance_plots.generate_performance_plots(
        predictions,
        output_dir / "plots",
        evaluation_scope="outer_oof",
        filename_prefix="outer_oof",
    )
    # The selector's SHAP view explains its newest inner chronological
    # validation refit. It explains the model and is not outer-OOF
    # performance evidence; its evaluation scope must remain explicit.
    inner_shap_source = (
        run_root / "xgboost" / "diagnostics" / "xgboost_shap_beeswarm.png"
    )
    inner_shap_destination = (
        output_dir / "plots" / "xgboost_inner_shap_beeswarm.png"
    )
    _atomic_copy(inner_shap_source, inner_shap_destination)
    plot_manifest = _separate_outer_performance_artifacts(
        plot_manifest,
        interpretability_artifacts=[
            {
                "kind": "shap_beeswarm_interpretability",
                "filename": inner_shap_destination.name,
                "path": str(inner_shap_destination.resolve()),
                "media_type": "image/png",
                "evaluation_scope": "inner_newest_chronological_validation",
                "performance_evidence": False,
                "interpretability_only": True,
                "source_path": str(inner_shap_source.resolve()),
                "source_sha256": _sha256(inner_shap_source),
            }
        ],
    )
    plot_manifest.update(
        {
            "run_id": run_id,
            "profile_hash": params.PROFILE_HASH,
            "prediction_artifact": str(outer_prediction_path.resolve()),
            "prediction_artifact_sha256": _sha256(outer_prediction_path),
            "final_holdout_rows_scored": 0,
        }
    )
    _atomic_json(output_dir / "performance_plots_manifest.json", plot_manifest)
    _atomic_csv(output_dir / "history_application_audit.csv", pd.DataFrame(history_audit_rows))


    print("FINAL PRE-HOLDOUT REFIT (FIXED RECIPES; NO RETUNING)", flush=True)
    final_base_idx, final_base_audit = final_evaluation_training_indices(
        prepared,
        final_holdout_idx,
        cohort_col=config.cohort_date_col,
        target_check_col=config.target_check_col,
        label_buffer_days=int(params.LABEL_AVAILABILITY_BUFFER_DAYS),
    )
    expected_evaluation_rows = int(split_summary["evaluation_train"]["rows"])
    if len(final_base_idx) != expected_evaluation_rows:
        raise RuntimeError(
            "Final evaluation-training reconstruction differs from the saved split "
            f"summary: expected={expected_evaluation_rows}; actual={len(final_base_idx)}."
        )
    final_base_population = _population_contract(
        prepared,
        indices=final_base_idx,
        cohort_col=config.cohort_date_col,
        identity_columns=identity_columns,
        deployment_bundle=deployment_bundle,
        definition="complete label-mature pre-holdout evaluation-training population",
    )
    final_family_train: dict[str, np.ndarray] = {}
    final_family_audit: dict[str, dict[str, Any]] = {}
    for family, recipe in recipes.items():
        indices, audit = modeling_utils.select_recent_mature_training_indices(
            prepared,
            final_base_idx,
            cohort_col=config.cohort_date_col,
            months=recipe.history_months,
        )
        if not bool(audit["requested_months_fully_available"]):
            raise RuntimeError(
                f"Final refit lacks {recipe.history_months} complete mature months "
                f"for {family}."
            )
        final_family_train[family] = indices
        final_family_audit[family] = {
            **audit,
            "history_id": recipe.history["history_id"],
            "model_family": family,
            "sample_weight_applied": False,
        }
    final_bundles: dict[str, dict[str, Any]] = {}
    comparison_role_tables: list[pd.DataFrame] = []
    comparison_populations: dict[str, dict[str, Any]] = {}
    selected_monthly: pd.DataFrame | None = None
    for family in MODEL_FAMILIES:
        recipe = recipes[family]
        is_selected = family == selected_family
        bundle_role = "FINAL_REFIT" if is_selected else "FINAL_REFIT_COMPARISON"
        # The nonselected model is fitted for the standard two-model report.
        # It is not promoted, retuned, or used to alter the outer family choice.
        destination = (
            final_bundle_root / family if is_selected
            else output_dir / "history_models" / family
        )
        print(
            f"FINAL PRE-HOLDOUT {family}: {len(recipe.features)} inputs "
            f"on {len(final_family_train[family]):,} rows; fixed recipe", flush=True,
        )
        final_state, final_state_contract = deployment_bundle.fit_shared_state_artifact(
            training_frame=prepared.loc[final_family_train[family]],
            target_col=config.target_col,
            modeling_utils=modeling_utils,
        )
        metadata = _source_metadata(
            params=params,
            run_id=run_id,
            run_root=run_root,
            source_data_identity=source_data_identity,
            source_tenure=source_tenure,
            excluded_holdout=excluded_holdout,
            sealed_final_holdout=excluded_holdout,
            evaluation_training_population=final_base_population,
            cohort_col=config.cohort_date_col,
            shared_state_contract=final_state_contract,
            pairing_scope={
                "bundle_role": bundle_role,
                "base_training_index_sha256": final_base_population["index_sha256"],
                "selected_model_family": selected_family,
                "fitted_model_family": family,
            },
            bundle_role=bundle_role,
            outer_fold=None,
            history_audit=final_family_audit[family],
        )
        _, manifest_path = _export_family_bundle(
            family=family,
            destination=destination,
            training_frame=prepared.loc[final_family_train[family]],
            recipe=recipe,
            target_col=config.target_col,
            cohort_col=config.cohort_date_col,
            identity_columns=identity_columns,
            modeling_utils=modeling_utils,
            deployment_bundle=deployment_bundle,
            shared_state_artifact=final_state,
            source_metadata=metadata,
            source_code_paths=source_code_paths,
            params=params,
            recovery_archives=recovery_sources,
            recovery_records=recovery_records,
        )
        _write_c4_recovery_report(output_dir, recovery_records)
        final_bundles[family] = {
            "bundle_path": str(destination.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "sha256": _sha256(manifest_path),
            "training_rows": int(len(final_family_train[family])),
            "history_id": recipe.history["history_id"],
            "bundle_role": bundle_role,
            "selected_for_final_test": is_selected,
            "recipe_source": recipe.model_recipe["recipe_source"],
            "recipe_source_sha256": recipe.model_recipe["recipe_source_sha256"],
            "recipe_frozen_before_outer_scoring": True,
            "outer_or_holdout_early_stopping": False,
        }
        monthly, by_role, population = score_full_history_family(
            prepared=prepared,
            final_holdout_idx=final_holdout_idx,
            selected_training_idx=final_family_train[family],
            family=family,
            bundle_path=destination,
            cohort_col=config.cohort_date_col,
            target_col=config.target_col,
            history_audit=final_family_audit[family],
            deployment_bundle=deployment_bundle,
            batch_rows=diagnostic_batch_rows,
            xgboost_device=xgboost_device,
        )
        comparison_role_tables.append(by_role)
        comparison_populations[family] = population
        if is_selected:
            selected_monthly = monthly
        del final_state


    # Both scorers must have consumed the identical ordered source rows, not
    # merely populations of equal size. Each family records that index hash.
    reference_population = comparison_populations[MODEL_FAMILIES[0]]
    for family, population in comparison_populations.items():
        for key in ("index_sha256", "scored_preholdout_rows", "cohort_start", "cohort_end"):
            if population[key] != reference_population[key]:
                raise AssertionError(f"Full-history population differs for {family}: {key}.")
    full_history_comparison = (
        final_refit_diagnostics.write_full_history_model_comparison_artifacts(
            monthly=pd.concat(comparison_role_tables, ignore_index=True),
            output_dir=output_dir / "plots",
            cohort_col=config.cohort_date_col,
            final_holdout_start=reference_population["holdout_start"],
            provenance={
                "run_id": run_id,
                "profile_hash": params.PROFILE_HASH,
                "input_sha256": source_data_identity["sha256"],
                "ordered_scored_index_sha256": reference_population["index_sha256"],
                "model_bundles": final_bundles,
                "model_populations": comparison_populations,
                "recipes_from_normal_pipeline": True,
                "recipe_selection_used_this_chart": False,
            },
        )
    )
    comparison_manifest_path = (
        output_dir / "plots" / str(full_history_comparison["manifest"])
    )
    comparison_inventory = [
        {
            "path": str((output_dir / "plots" / str(name)).resolve()),
            "sha256": _sha256(output_dir / "plots" / str(name)),
        }
        for name in full_history_comparison["artifacts"]
    ]
    comparison_receipt = {
        **full_history_comparison,
        "manifest_path": str(comparison_manifest_path.resolve()),
        "manifest_sha256": _sha256(comparison_manifest_path),
        "artifact_inventory": comparison_inventory,
    }
    # Keep the legacy champion-only data contract for existing test/report
    # readers. The full-history comparison above includes both models.
    assert selected_monthly is not None
    descriptive_monthly = selected_monthly
    descriptive_population = comparison_populations[selected_family]
    selected_bundle = final_bundles[selected_family]
    recipe = recipes[selected_family]
    history_audit = final_family_audit[selected_family]
    training_start = pd.Period(str(history_audit["cutoff_month_inclusive"]), freq="M")
    training_end = pd.Period(str(history_audit["anchor_month"]), freq="M")
    descriptive_trend = (
        final_refit_diagnostics.write_final_refit_descriptive_trend_artifacts(
            monthly=descriptive_monthly,
            output_dir=deployment_dir / "diagnostics",
            model_family=selected_family,
            cohort_col=config.cohort_date_col,
            selected_history=recipe.history,
            training_start=str(training_start),
            training_end=str(training_end),
        )
    )
    descriptive_output_dir = (deployment_dir / "diagnostics").resolve()
    descriptive_manifest_path = descriptive_output_dir / str(descriptive_trend["manifest"])
    selected_bundle["descriptive_actual_vs_predicted_trend"] = {
        **descriptive_trend,
        "writer": {
            "path": str(Path(final_refit_diagnostics.__file__).resolve()),
            "sha256": _sha256(Path(final_refit_diagnostics.__file__).resolve()),
        },
        "manifest_path": str(descriptive_manifest_path.resolve()),
        "manifest_sha256": _sha256(descriptive_manifest_path),
        "artifact_inventory": [
            {
                "path": str((descriptive_output_dir / str(name)).resolve()),
                "sha256": _sha256(descriptive_output_dir / str(name)),
            }
            for name in descriptive_trend["artifacts"]
        ],
        "population": descriptive_population,
        "batch_rows": diagnostic_batch_rows,
        "scored_preholdout_rows": descriptive_population["scored_preholdout_rows"],
        "scored_preholdout_months": descriptive_population["scored_preholdout_months"],
    }
    plot_manifest["full_history_model_comparison"] = comparison_receipt
    plot_manifest["descriptive_artifacts"] = [
        {**item, "evaluation_scope": "full_history_descriptive", "performance_evidence": False}
        for item in comparison_inventory
    ]
    _atomic_json(output_dir / "performance_plots_manifest.json", plot_manifest)


    final_model_manifest = {
        "contract": FINAL_MODEL_CONTRACT,
        "contract_revision": FINAL_MODEL_CONTRACT_REVISION,
        "run_id": run_id,
        "source_run_id": run_id,
        "profile": dict(params.PROFILE_PAYLOAD),
        "profile_hash": params.PROFILE_HASH,
        "population_scope_mode": "FULL",
        "selected_model_family": selected_family,
        "selected_bundle": selected_bundle,
        "artifact_status": "PROVISIONAL_PREHOLDOUT_REFIT",
        "deployment_eligible": False,
        "deployment_blockers": [
            "TENURE_GROUPING_DECISION_NOT_ATTESTED",
            "FINAL_HOLDOUT_EVALUATION_NOT_APPROVED_OR_PASSED",
        ],
        "tenure_grouping_must_be_locked_before_promotion": True,
        "evaluated_candidate_families": list(MODEL_FAMILIES),
        "nonselected_family_refit": True,
        "nonselected_family_refit_purpose": "FULL_HISTORY_DESCRIPTIVE_COMPARISON",
        "full_history_model_comparison": comparison_receipt,
        "comparison_bundles": {
            family: bundle for family, bundle in final_bundles.items()
            if family != selected_family
        },
        "family_selection": str((output_dir / "family_selection.json").resolve()),
        "family_selection_sha256": _sha256(output_dir / "family_selection.json"),
        "final_refit_changes": (
            "fitted coefficients/trees and fold-local preprocessing are relearned "
            "on the final label-mature population; selected features, history, C, "
            "XGBoost hyperparameters, seeds, and boosting rounds remain frozen"
        ),
        "final_training_base": final_base_population,
        "final_training_reconstruction": final_base_audit,
        "excluded_holdout": excluded_holdout,
        "final_holdout_rows_scored": 0,
        "final_holdout_unseal_required": True,
    }
    _atomic_json(deployment_dir / "final_model_manifest.json", final_model_manifest)


    outer_manifest = {
        "contract": OUTER_CONTRACT,
        "contract_revision": OUTER_CONTRACT_REVISION,
        "run_id": run_id,
        "source_run_id": run_id,
        "profile": dict(params.PROFILE_PAYLOAD),
        "profile_hash": params.PROFILE_HASH,
        "population_scope_mode": "FULL",
        "source_data_identity": source_data_identity,
        "input_snapshot": source_data_identity,
        "target_column": config.target_col,
        "cohort_column": config.cohort_date_col,
        "tenure_column": config.tenure_col,
        "identity_columns": identity_columns,
        "evaluation_design": (
            "recipes frozen by inner development; full-row refit on each outer "
            "training partition; each outer validation block scored once"
        ),
        "outer_validation_used_for_tuning_or_early_stopping": False,
        "outer_oof_predictions": {
            "path": str(outer_prediction_path.resolve()),
            "sha256": _sha256(outer_prediction_path),
            "rows": int(len(predictions)),
            "format": "long; one row per source row and model family",
        },
        "outer_fold_metrics": {
            "path": str((output_dir / "outer_fold_metrics.csv").resolve()),
            "sha256": _sha256(output_dir / "outer_fold_metrics.csv"),
        },
        "outer_tenure_metrics": {
            "path": str((output_dir / "outer_tenure_metrics.csv").resolve()),
            "sha256": _sha256(output_dir / "outer_tenure_metrics.csv"),
        },
        "folds": fold_manifest_rows,
        "selected_model_family": selected_family,
        "final_model_manifest": {
            "path": str((deployment_dir / "final_model_manifest.json").resolve()),
            "sha256": _sha256(deployment_dir / "final_model_manifest.json"),
        },
        "excluded_holdout": excluded_holdout,
        "final_holdout_rows_scored": 0,
    }
    _atomic_json(output_dir / "outer_fold_bundle_manifest.json", outer_manifest)
    run_info = {
        **outer_manifest,
        "contract": "retention_outer_evaluation_run_info",
        "input": source_data_identity,
        "selected_recipes": {
            family: {
                "features": list(recipe.features),
                "history": recipe.history,
                "model_recipe": recipe.model_recipe,
                "seeds": list(recipe.seeds),
                "boosting_rounds": recipe.boosting_rounds,
            }
            for family, recipe in recipes.items()
        },
        "software": {"xgboost": str(xgb.__version__)},
        "failed_c4_recovery_archives": c4_recovery_archives,
        "saved_fit_recovery": recovery_records,
        "execution_device": runtime_execution,
    }
    _atomic_json(output_dir / "run_info.json", run_info)
    _atomic_json(
        output_dir / "run_status.json",
        {
            "status": "COMPLETE",
            "run_id": run_id,
            "profile_hash": params.PROFILE_HASH,
            "selected_model_family": selected_family,
            "outer_folds_scored": int(len(outer_folds)),
            "outer_oof_prediction_rows": int(len(predictions)),
            "final_refit_completed": True,
            "final_refit_artifact_status": "PROVISIONAL_PREHOLDOUT_REFIT",
            "descriptive_actual_vs_predicted_trend_generated": True,
            "descriptive_trend_rows": descriptive_population["scored_preholdout_rows"],
            "full_history_comparison_families": list(MODEL_FAMILIES),
            "full_history_comparison_generated": True,
            "descriptive_trend_months": int(len(descriptive_monthly)),
            "deployment_eligible": False,
            "final_holdout_rows_scored": 0,
        },
    )
    (output_dir / "run_status.json.incomplete").unlink(missing_ok=True)
    print(
        "OUTER EVALUATION AND PROVISIONAL PRE-HOLDOUT REFIT COMPLETE\n"
        f"Selected family: {selected_family}\n"
        f"Outer OOF rows: {len(predictions):,}\n"
        "Final holdout rows scored: 0\n"
        f"Outputs: {output_dir}\n"
        f"Full-history comparison: {output_dir / 'plots' / 'model_retention_trend_full_history.png'}",
        flush=True,
    )




if __name__ == "__main__":
    main()
