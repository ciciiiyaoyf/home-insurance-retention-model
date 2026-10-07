#!/usr/bin/env python
"""Evaluate one frozen pooled tenure model on sealed holdout rows.


The evaluator is strictly score-only. In its original comparison mode it loads
the pooled run's sealed holdout once, verifies its exact population identity,
splits those same rows by tenure, and scores both the pooled and matching native
frozen bundles on each subset. With ``--pooled-baseline-only`` it instead scores
the pooled model once on its complete sealed pooled holdout and reports lift
spread using the exact cross-tenure equal-count calculation. No estimator,
preprocessor, state grouping, feature selector, calibrator, or threshold is
fitted here.
"""


from __future__ import annotations


import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence


import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


import cross_tenure_evaluation as cross




COMPONENT_ROLE = "frozen_pooled_vs_native_evaluator"
COMPONENT_CAPABILITIES = ("pooled_baseline",)
EVALUATION_CONTRACT = "frozen_pooled_vs_native_same_holdout_rows"
POOLED_BASELINE_CONTRACT = "frozen_pooled_model_on_pooled_holdout_baseline"
POOLED_BASELINE_MODEL_ROLE = "FROZEN_POOLED_MODEL_ON_POOLED_HOLDOUT"
MODEL_FAMILIES = tuple(cross.MODEL_FAMILIES)




@dataclass(frozen=True)
class BundleContext:
    role: str
    tenure_groups: tuple[str, ...]
    scope_key: str
    source_run: cross.TargetRun
    bundles: Mapping[str, cross.ManifestRecord]
    scoring_columns: frozenset[str]
    target_column: str
    holdout: Mapping[str, Any]
    shared_state: cross.SharedStateContract
    source_snapshot_sha256: str




def _normalize_tenure(value: object) -> str:
    return cross._normalize_tenure(value)




def _normalize_groups(values: Iterable[object]) -> tuple[str, ...]:
    groups: list[str] = []
    for value in values:
        for raw in re.split(r"[,;]", str(value)):
            normalized = _normalize_tenure(raw)
            if normalized and normalized not in groups:
                groups.append(normalized)
    if len(groups) < 2:
        raise ValueError("A pooled tenure scope requires at least two tenure groups.")
    return tuple(groups)




def _scope_key(groups: Sequence[object]) -> str:
    return ",".join(_normalize_tenure(value) for value in groups)




def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))




def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp_", suffix=".json", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                cross._jsonable(payload),
                handle,
                indent=2,
                sort_keys=True,
                default=str,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)




def _parse_assignment(value: str, *, label: str) -> tuple[str, Path]:
    tenure, separator, raw_path = str(value).partition("=")
    tenure = _normalize_tenure(tenure)
    if not separator or not tenure or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            f"{label} must use TENURE=PATH, received {value!r}."
        )
    return tenure, Path(raw_path.strip())




def _assignment_map(
    values: Sequence[str], *, label: str, allowed: Sequence[str]
) -> dict[str, Path]:
    output: dict[str, Path] = {}
    allowed_set = set(allowed)
    for value in values:
        tenure, path = _parse_assignment(value, label=label)
        if tenure not in allowed_set:
            raise ValueError(
                f"{label} tenure {tenure!r} is outside requested groups {list(allowed)}."
            )
        if tenure in output:
            raise ValueError(f"{label} was supplied more than once for tenure {tenure}.")
        output[tenure] = path
    return output




def _resolve_manifest_path(project_root: Path, value: Path) -> Path:
    raw = value.expanduser()
    resolved = raw.resolve() if raw.is_absolute() else (project_root / raw).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Frozen bundle manifest not found: {resolved}")
    return resolved




def _run_completion_timestamp(
    run: Path, profile_groups: Sequence[str], profile_hash: str
) -> pd.Timestamp:
    requested = tuple(profile_groups)
    candidates: list[pd.Timestamp] = []
    for path in sorted(run.rglob("*.json")):
        try:
            payload = _json_load(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if str(payload.get("run_id", "")).strip() != run.name:
            continue
        if str(payload.get("status", "")).strip().upper() != "COMPLETE":
            continue
        if str(payload.get("population_scope_mode", "")).strip().upper() != "FULL":
            continue
        observed_groups = payload.get("selected_tenure_groups")
        if not isinstance(observed_groups, list):
            continue
        observed = tuple(_normalize_tenure(value) for value in observed_groups)
        if observed != requested:
            continue
        if str(payload.get("profile_hash", "")).strip() != str(profile_hash).strip():
            continue
        if any(
            payload.get(key) not in (None, "")
            for key in (
                "test_policy_id",
                "sample_basis_points",
                "sample_percent",
                "policy_sample_salt",
            )
        ):
            continue
        completed = payload.get("completed")
        if completed is None:
            continue
        try:
            candidates.append(pd.Timestamp(completed))
        except (TypeError, ValueError):
            continue
    if len(set(candidates)) != 1:
        raise ValueError(
            f"Run {run} requires one COMPLETE/FULL metadata completion timestamp; "
            f"observed={[str(value) for value in candidates]}."
        )
    return candidates[0]




def resolve_profile_run(
    project_root: Path,
    requested_groups: Sequence[str],
    explicit: Path | None = None,
) -> cross.TargetRun:
    requested_set = set(requested_groups)
    candidates = (
        [cross._resolve_project_or_run_path(project_root, explicit)]
        if explicit is not None
        else cross._pipeline_run_dirs(project_root)
    )
    eligible: list[tuple[cross.TargetRun, pd.Timestamp]] = []
    errors: list[str] = []
    for run in candidates:
        if not run.is_dir():
            continue
        try:
            found = cross._run_profile(run)
            if found is None:
                continue
            profile, profile_hash = found
            observed = tuple(cross._selected_groups(profile))
            if len(observed) != len(requested_groups) or set(observed) != requested_set:
                continue
            record = cross.TargetRun(run, profile, profile_hash)
            eligible.append(
                (
                    record,
                    _run_completion_timestamp(run, observed, profile_hash),
                )
            )
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"{run.name}: {exc}")
    if not eligible:
        qualifier = "explicit " if explicit is not None else ""
        raise FileNotFoundError(
            f"No {qualifier}COMPLETE/FULL run matches tenure groups "
            f"{list(requested_groups)}. inspected_errors={errors[:10]}"
        )
    latest = max(timestamp for _, timestamp in eligible)
    winners = [record for record, timestamp in eligible if timestamp == latest]
    if len(winners) != 1:
        raise RuntimeError(
            "Multiple runs share the latest metadata completion timestamp; pass an "
            f"explicit run. candidates={[str(value.path) for value in winners]}"
        )
    return winners[0]




def _load_bundle_context(
    *,
    project_root: Path,
    role: str,
    tenure_groups: Sequence[str],
    source_run: cross.TargetRun,
    explicit_manifests: Sequence[Path],
    verify_hashes: bool,
) -> BundleContext:
    scope_key = _scope_key(tenure_groups)
    bundles = cross.discover_source_bundle_manifests(
        project_root,
        scope_key,
        explicit_manifests=explicit_manifests,
        explicit_source_run=None if explicit_manifests else source_run.path,
    )
    if verify_hashes:
        for record in bundles.values():
            cross._verify_manifest_inventory(record)
    identities = {
        family: cross._manifest_identity(record.payload)
        for family, record in bundles.items()
    }
    required_identity_fields = (
        "pairing_identity",
        "source_run_id",
        "source_profile_hash",
        "source_snapshot_sha256",
        "excluded_holdout_identity",
        "shared_state_mapping_sha256",
    )
    for family, identity in identities.items():
        missing = [field for field in required_identity_fields if not identity[field]]
        if missing:
            raise ValueError(
                f"Frozen {family} bundle is missing provenance fields: {missing}"
            )
        if identity["source_run_id"] != source_run.path.name:
            raise RuntimeError(
                f"Explicit/resolved {family} bundle belongs to run "
                f"{identity['source_run_id']!r}, not {source_run.path.name!r}."
            )
        if identity["source_profile_hash"] != source_run.profile_hash:
            raise RuntimeError(
                f"Explicit/resolved {family} bundle profile hash does not match "
                f"the selected run {source_run.path.name}."
            )
    snapshot_hashes = {
        identity["source_snapshot_sha256"] for identity in identities.values()
    }
    if len(snapshot_hashes) != 1:
        raise RuntimeError("Frozen Logistic/XGBoost source snapshot hashes disagree.")
    required, target_column, _, shared_state = cross._required_scoring_columns(
        bundles
    )
    holdout = cross._source_holdout_contract(bundles)
    return BundleContext(
        role=role,
        tenure_groups=tuple(tenure_groups),
        scope_key=scope_key,
        source_run=source_run,
        bundles=bundles,
        scoring_columns=frozenset(required),
        target_column=target_column,
        holdout=holdout,
        shared_state=shared_state,
        source_snapshot_sha256=next(iter(snapshot_hashes)),
    )




def _validate_pooled_baseline_context(context: BundleContext) -> None:
    """Require a complete, identical provenance identity for the pooled pair."""


    identities = {
        family: cross._manifest_identity(record.payload)
        for family, record in context.bundles.items()
    }
    required_fields = (
        "pairing_identity",
        "source_tenure",
        "source_run_id",
        "source_profile_hash",
        "cohort_column",
        "source_snapshot_sha256",
        "excluded_holdout_identity",
        "evaluation_training_population_sha256",
        "shared_state_mapping_sha256",
        "source_profile",
    )
    for family, identity in identities.items():
        missing = [field for field in required_fields if not identity.get(field)]
        if missing:
            raise ValueError(
                f"Frozen pooled {family} bundle is missing strict provenance "
                f"fields: {missing}"
            )
    for field in required_fields:
        values = {identity[field] for identity in identities.values()}
        if len(values) != 1:
            raise RuntimeError(
                "Frozen pooled Logistic/XGBoost manifests disagree on "
                f"{field}: {sorted(values)}"
            )


    identity = next(iter(identities.values()))
    if identity["source_tenure"] != context.scope_key:
        raise RuntimeError(
            "Frozen pooled bundle tenure identity does not match the requested "
            f"scope: {identity['source_tenure']!r} != {context.scope_key!r}."
        )
    if identity["source_run_id"] != context.source_run.path.name:
        raise RuntimeError("Frozen pooled bundle source-run identity changed.")
    if identity["source_profile_hash"] != context.source_run.profile_hash:
        raise RuntimeError("Frozen pooled bundle profile-hash identity changed.")
    if identity["cohort_column"] != str(context.holdout["cohort_column"]):
        raise RuntimeError("Frozen pooled bundle cohort-column identity disagrees.")
    if identity["excluded_holdout_identity"] != str(
        context.holdout["population_sha256"]
    ):
        raise RuntimeError("Frozen pooled bundle holdout identity disagrees.")
    if identity["shared_state_mapping_sha256"] != str(
        context.shared_state.mapping_sha256
    ):
        raise RuntimeError("Frozen pooled bundle state-mapping identity disagrees.")
    try:
        recorded_profile = json.loads(identity["source_profile"])
    except json.JSONDecodeError as exc:
        raise ValueError("Frozen pooled bundle source_profile is invalid JSON.") from exc
    if not isinstance(recorded_profile, Mapping) or tuple(
        cross._selected_groups(recorded_profile)
    ) != context.tenure_groups:
        raise RuntimeError(
            "Frozen pooled bundle source_profile does not preserve the exact "
            "ordered pooled tenure scope."
        )




def _validate_contexts(
    pooled: BundleContext, natives: Mapping[str, BundleContext]
) -> tuple[str, tuple[str, ...], str]:
    contexts = [pooled, *natives.values()]
    target_columns = {context.target_column for context in contexts}
    if len(target_columns) != 1:
        raise ValueError(f"Bundle target columns disagree: {sorted(target_columns)}")
    calendar = {
        (
            str(context.holdout["cohort_start"]),
            str(context.holdout["cohort_end"]),
            str(context.holdout["cohort_column"]),
            int(context.holdout["label_availability_buffer_days"]),
            str(context.holdout["final_training_label_cutoff_exclusive"]),
            context.holdout["target_scored"],
            context.holdout["used_for_preprocessing_or_fit"],
        )
        for context in contexts
    }
    if len(calendar) != 1:
        raise ValueError(f"Pooled/native holdout calendars disagree: {sorted(calendar)}")
    identities = {
        tuple(str(value) for value in context.holdout["identity_columns"])
        for context in contexts
    }
    if len(identities) != 1:
        raise ValueError(
            f"Pooled/native ordered row-identity columns disagree: {sorted(identities)}"
        )
    policy_clusters = {
        str(context.holdout["policy_cluster_column"]) for context in contexts
    }
    if len(policy_clusters) != 1:
        raise ValueError(
            "Pooled/native policy-cluster columns disagree: "
            f"{sorted(policy_clusters)}"
        )
    return (
        next(iter(target_columns)),
        next(iter(identities)),
        next(iter(policy_clusters)),
    )




def _population_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = [str(value) for value in columns]
    missing = [column for column in selected if column not in frame]
    if missing:
        raise KeyError(f"Population identity columns are missing: {missing}")
    canonical = frame.loc[:, selected].copy()
    for column in canonical:
        if pd.api.types.is_datetime64_any_dtype(canonical[column]):
            canonical[column] = (
                pd.to_datetime(canonical[column], errors="coerce")
                .dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
                .fillna("__MISSING__")
            )
        else:
            canonical[column] = canonical[column].astype("string").fillna(
                "__MISSING__"
            )
    hashes = pd.util.hash_pandas_object(canonical, index=False).to_numpy(
        dtype=np.uint64
    )
    hashes.sort()
    return hashlib.sha256(hashes.tobytes()).hexdigest()




def _payload_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = sorted({str(value) for value in columns})
    missing = [column for column in selected if column not in frame]
    if missing:
        raise KeyError(f"Payload columns are missing: {missing}")
    canonical = frame.loc[:, selected].copy()
    for column in canonical:
        if pd.api.types.is_datetime64_any_dtype(canonical[column]):
            canonical[column] = (
                pd.to_datetime(canonical[column], errors="coerce")
                .dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
                .fillna("__MISSING__")
            )
        else:
            canonical[column] = canonical[column].astype("string").fillna(
                "__MISSING__"
            )
    row_hashes = pd.util.hash_pandas_object(canonical, index=False).to_numpy(
        dtype=np.uint64
    )
    row_hashes.sort()
    digest = hashlib.sha256()
    digest.update("\n".join(selected).encode("utf-8"))
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()




def _filter_groups(
    frame: pd.DataFrame, tenure_column: str, groups: Sequence[str]
) -> pd.DataFrame:
    raw = frame[tenure_column].astype("string")
    stripped = raw.str.strip()
    missing = raw.isna() | stripped.isin({"", "<NA>", "NAN", "NONE", "NULL"})
    if missing.any():
        raise ValueError(
            f"Scoring snapshot has {int(missing.sum())} missing tenure value(s)."
        )
    observed = stripped.map(_normalize_tenure)
    requested = set(groups)
    found = set(observed.unique().tolist())
    if found != requested:
        raise ValueError(
            f"Scoring data tenure scope mismatch: observed={sorted(found)}; "
            f"expected={list(groups)}."
        )
    return frame.copy().reset_index(drop=True)




def _safe_filename(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9+_-]+", "_", str(value).strip())




def _runtime_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import sklearn


        versions["scikit_learn"] = sklearn.__version__
    except ImportError:
        versions["scikit_learn"] = "NOT_INSTALLED"
    try:
        import xgboost


        versions["xgboost"] = xgboost.__version__
    except ImportError:
        versions["xgboost"] = "NOT_INSTALLED"
    return versions




def _output_inventory(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    excluded = {"evaluation_manifest.json", "output_inventory.csv"}
    for path in sorted(value for value in output_dir.rglob("*") if value.is_file()):
        relative = str(path.relative_to(output_dir)).replace("\\", "/")
        if relative in excluded:
            continue
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": cross.sha256_file(path),
            }
        )
    return pd.DataFrame(rows, columns=["relative_path", "size_bytes", "sha256"])




def _row_id_strings(row_id: pd.Series) -> pd.Series:
    return row_id.astype("uint64").astype("string")




def _cluster_hash(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").fillna("__MISSING__")
    return pd.util.hash_pandas_object(normalized, index=False).astype("uint64").astype(
        "string"
    )




def _top_capacity_mask(
    p_churn: np.ndarray, row_id: np.ndarray, percent: int
) -> np.ndarray:
    n = len(p_churn)
    count = min(n, max(1, int(math.ceil(n * (float(percent) / 100.0)))))
    order = np.lexsort((row_id, -np.asarray(p_churn, dtype=float)))
    mask = np.zeros(n, dtype=bool)
    mask[order[:count]] = True
    return mask




def _capacity_metrics(
    y_retention: np.ndarray,
    p_retention: np.ndarray,
    row_id: np.ndarray,
    capacities: Sequence[int] = (5, 10, 15),
) -> dict[str, float]:
    y_churn = 1 - np.asarray(y_retention, dtype=int)
    p_churn = 1.0 - np.asarray(p_retention, dtype=float)
    prevalence = float(y_churn.mean())
    total_events = float(y_churn.sum())
    output: dict[str, float] = {}
    for percent in capacities:
        mask = _top_capacity_mask(p_churn, row_id, percent)
        precision = float(y_churn[mask].mean())
        output[f"lift_at_{percent}pct"] = (
            precision / prevalence if prevalence > 0 else np.nan
        )
        output[f"capture_at_{percent}pct"] = (
            float(y_churn[mask].sum()) / total_events if total_events > 0 else np.nan
        )
        output[f"precision_at_{percent}pct"] = precision
    return output




def _all_metrics(
    y_retention: np.ndarray, p_retention: np.ndarray, row_id: np.ndarray
) -> dict[str, float]:
    metrics = dict(cross._ranking_metrics(y_retention, p_retention))
    metrics["absolute_calibration_gap"] = abs(float(metrics["calibration_gap"]))
    metrics.update(_capacity_metrics(y_retention, p_retention, row_id))
    return metrics




def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(weights.sum())
    return float(np.dot(values, weights) / denominator) if denominator > 0 else np.nan




def _weighted_metrics(
    y_retention: np.ndarray,
    p_retention: np.ndarray,
    weights: np.ndarray,
    capacity_masks: Mapping[int, np.ndarray],
) -> dict[str, float]:
    y = np.asarray(y_retention, dtype=int)
    p = np.asarray(p_retention, dtype=float)
    w = np.asarray(weights, dtype=float)
    y_churn = 1 - y
    p_churn = 1.0 - p
    if float(w.sum()) <= 0:
        raise ValueError("Bootstrap weights are all zero.")
    class_weight = [float(w[y == value].sum()) for value in (0, 1)]
    if min(class_weight) <= 0:
        raise ValueError("Bootstrap replicate contains only one weighted class.")
    prevalence = _weighted_mean(y_churn, w)
    ap = float(average_precision_score(y_churn, p_churn, sample_weight=w))
    total_events = float(np.dot(y_churn, w))
    calibration_gap = _weighted_mean(p, w) - _weighted_mean(y, w)
    output = {
        "retention_roc_auc": float(roc_auc_score(y, p, sample_weight=w)),
        "churn_average_precision": ap,
        "churn_ap_skill": (ap - prevalence) / (1.0 - prevalence)
        if prevalence < 1.0
        else np.nan,
        "log_loss": float(log_loss(y, p, labels=[0, 1], sample_weight=w)),
        "brier_score": float(brier_score_loss(y, p, sample_weight=w)),
        "calibration_gap": calibration_gap,
        "absolute_calibration_gap": abs(calibration_gap),
    }
    for percent, mask in sorted(capacity_masks.items()):
        selected_weights = w * np.asarray(mask, dtype=float)
        selected_event_rate = _weighted_mean(y_churn, selected_weights)
        selected_events = float(np.dot(y_churn, selected_weights))
        output[f"lift_at_{percent}pct"] = (
            selected_event_rate / prevalence if prevalence > 0 else np.nan
        )
        output[f"capture_at_{percent}pct"] = (
            selected_events / total_events if total_events > 0 else np.nan
        )
        output[f"precision_at_{percent}pct"] = selected_event_rate
    return output




def _bootstrap_pair(
    *,
    tenure: str,
    family: str,
    y_retention: np.ndarray,
    native_probability: np.ndarray,
    pooled_probability: np.ndarray,
    row_id: np.ndarray,
    clusters: pd.Series,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    if reps <= 0:
        return pd.DataFrame()
    codes, unique_clusters = pd.factorize(clusters.astype("string"), sort=True)
    if (codes < 0).any() or len(unique_clusters) == 0:
        raise ValueError("Policy-cluster identity is missing or invalid.")
    seed_material = f"{seed}|{tenure}|{family}".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    rng = np.random.default_rng(local_seed)
    native_capacity_masks = {
        percent: _top_capacity_mask(
            1.0 - native_probability, row_id, percent
        )
        for percent in (5, 10, 15)
    }
    pooled_capacity_masks = {
        percent: _top_capacity_mask(
            1.0 - pooled_probability, row_id, percent
        )
        for percent in (5, 10, 15)
    }
    rows: list[dict[str, Any]] = []
    completed = 0
    attempts = 0
    maximum_attempts = max(reps * 5, reps + 10)
    while completed < reps and attempts < maximum_attempts:
        attempts += 1
        cluster_weights = rng.poisson(1.0, size=len(unique_clusters)).astype(float)
        weights = cluster_weights[codes]
        try:
            native_metrics = _weighted_metrics(
                y_retention, native_probability, weights, native_capacity_masks
            )
            pooled_metrics = _weighted_metrics(
                y_retention, pooled_probability, weights, pooled_capacity_masks
            )
        except ValueError:
            continue
        completed += 1
        for metric in sorted(native_metrics):
            native_value = float(native_metrics[metric])
            pooled_value = float(pooled_metrics[metric])
            rows.append(
                {
                    "tenure": tenure,
                    "model_family": family,
                    "replicate": completed,
                    "metric": metric,
                    "native_value": native_value,
                    "pooled_value": pooled_value,
                    "pooled_minus_native": pooled_value - native_value,
                }
            )
    if completed != reps:
        raise RuntimeError(
            f"Only {completed}/{reps} valid paired bootstrap replicates were produced."
        )
    return pd.DataFrame(rows)




def _rank_group_membership(
    score: np.ndarray, row_id: np.ndarray, n_bins: int
) -> np.ndarray:
    values = np.asarray(score, dtype=float)
    identifiers = np.asarray(row_id)
    if values.shape != identifiers.shape or values.ndim != 1:
        raise ValueError("Score and row identity must have the same 1-D shape.")
    order = np.lexsort((identifiers, -values))
    ranked_groups = pd.qcut(
        np.arange(1, len(values) + 1),
        q=int(n_bins),
        labels=np.arange(1, int(n_bins) + 1),
    ).astype(int)
    membership = np.empty(len(values), dtype=np.int16)
    membership[order] = np.asarray(ranked_groups, dtype=np.int16)
    return membership




def _weighted_lift_spread(
    y_churn: np.ndarray,
    weights: np.ndarray,
    membership: np.ndarray,
    n_bins: int,
) -> dict[str, float]:
    y = np.asarray(y_churn, dtype=float)
    w = np.asarray(weights, dtype=float)
    group = np.asarray(membership, dtype=int) - 1
    total_weight = float(w.sum())
    total_events = float(np.dot(y, w))
    if total_weight <= 0 or total_events <= 0:
        raise ValueError("Bootstrap lift replicate has no weighted rows/events.")
    group_weight = np.bincount(group, weights=w, minlength=n_bins)
    group_events = np.bincount(group, weights=w * y, minlength=n_bins)
    if len(group_weight) != n_bins or np.any(group_weight <= 0):
        raise ValueError("Bootstrap lift replicate has an empty score group.")
    overall_rate = total_events / total_weight
    lift = (group_events / group_weight) / overall_rate
    if not np.isfinite(lift).all() or float(lift.min()) <= 0:
        raise ValueError("Bootstrap lift spread is undefined or non-finite.")
    return {
        "highest_divided_by_lowest_lift": float(lift.max() / lift.min()),
        "ordered_first_to_last_lift_ratio": float(lift[0] / lift[-1]),
    }




def _bootstrap_lift_spread_pair(
    *,
    tenure: str,
    family: str,
    y_retention: np.ndarray,
    native_probability: np.ndarray,
    pooled_probability: np.ndarray,
    row_id: np.ndarray,
    clusters: pd.Series,
    lift_bins: Sequence[int],
    reps: int,
    seed: int,
) -> pd.DataFrame:
    if reps <= 0:
        return pd.DataFrame()
    codes, unique_clusters = pd.factorize(clusters.astype("string"), sort=True)
    if (codes < 0).any() or len(unique_clusters) == 0:
        raise ValueError("Policy-cluster identity is missing or invalid.")
    y_churn = 1 - np.asarray(y_retention, dtype=int)
    native_score = 1.0 - np.asarray(native_probability, dtype=float)
    pooled_score = 1.0 - np.asarray(pooled_probability, dtype=float)
    native_groups = {
        int(bins): _rank_group_membership(native_score, row_id, int(bins))
        for bins in lift_bins
    }
    pooled_groups = {
        int(bins): _rank_group_membership(pooled_score, row_id, int(bins))
        for bins in lift_bins
    }
    seed_material = f"{seed}|{tenure}|{family}|lift_spread".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    rng = np.random.default_rng(local_seed)
    rows: list[dict[str, Any]] = []
    completed = 0
    attempts = 0
    maximum_attempts = max(reps * 5, reps + 10)
    while completed < reps and attempts < maximum_attempts:
        attempts += 1
        cluster_weights = rng.poisson(1.0, size=len(unique_clusters)).astype(float)
        weights = cluster_weights[codes]
        replicate_results: list[tuple[int, dict[str, float], dict[str, float]]] = []
        try:
            for bins in lift_bins:
                native = _weighted_lift_spread(
                    y_churn, weights, native_groups[int(bins)], int(bins)
                )
                pooled = _weighted_lift_spread(
                    y_churn, weights, pooled_groups[int(bins)], int(bins)
                )
                replicate_results.append((int(bins), native, pooled))
        except ValueError:
            continue
        completed += 1
        for bins, native, pooled in replicate_results:
            for metric in sorted(native):
                rows.append(
                    {
                        "tenure": tenure,
                        "model_family": family,
                        "n_bins": bins,
                        "replicate": completed,
                        "spread_metric": metric,
                        "native_value": float(native[metric]),
                        "pooled_value": float(pooled[metric]),
                        "pooled_minus_native": float(
                            pooled[metric] - native[metric]
                        ),
                    }
                )
    if completed != reps:
        raise RuntimeError(
            f"Only {completed}/{reps} valid lift-spread bootstrap replicates "
            f"were produced for tenure={tenure}, family={family}."
        )
    return pd.DataFrame(rows)




def _summarize_lift_spread_comparison(
    point_spreads: pd.DataFrame, replicates: pd.DataFrame
) -> pd.DataFrame:
    keys = ["tenure", "model_family", "n_bins"]
    native = point_spreads.loc[
        point_spreads["model_role"].eq("NATIVE")
    ].set_index(keys)
    pooled = point_spreads.loc[
        point_spreads["model_role"].eq("POOLED")
    ].set_index(keys)
    if not native.index.is_unique or not pooled.index.is_unique:
        raise ValueError("Lift-spread point estimates are not unique by model role.")
    if set(native.index) != set(pooled.index):
        raise ValueError("Native and pooled lift-spread score groups disagree.")
    rows: list[dict[str, Any]] = []
    for key in sorted(native.index):
        for metric in (
            "highest_divided_by_lowest_lift",
            "ordered_first_to_last_lift_ratio",
        ):
            native_value = float(native.loc[key, metric])
            pooled_value = float(pooled.loc[key, metric])
            rows.append(
                {
                    "tenure": key[0],
                    "model_family": key[1],
                    "n_bins": int(key[2]),
                    "spread_metric": metric,
                    "native_value": native_value,
                    "pooled_value": pooled_value,
                    "pooled_minus_native": pooled_value - native_value,
                }
            )
    output = pd.DataFrame(rows)
    if replicates.empty:
        output["bootstrap_replicates"] = 0
        output["ci_2_5pct"] = np.nan
        output["ci_97_5pct"] = np.nan
        output["probability_pooled_higher"] = np.nan
        return output
    summaries: list[dict[str, Any]] = []
    for key, group in replicates.groupby(
        ["tenure", "model_family", "n_bins", "spread_metric"], sort=True
    ):
        delta = group["pooled_minus_native"].to_numpy(dtype=float)
        summaries.append(
            {
                "tenure": key[0],
                "model_family": key[1],
                "n_bins": int(key[2]),
                "spread_metric": key[3],
                "bootstrap_replicates": int(len(delta)),
                "bootstrap_mean_delta": float(delta.mean()),
                "bootstrap_std_delta": float(delta.std(ddof=1)),
                "ci_2_5pct": float(np.quantile(delta, 0.025)),
                "ci_97_5pct": float(np.quantile(delta, 0.975)),
                "probability_pooled_higher": float(np.mean(delta > 0.0)),
            }
        )
    return output.merge(
        pd.DataFrame(summaries),
        on=["tenure", "model_family", "n_bins", "spread_metric"],
        how="left",
        validate="one_to_one",
    ).sort_values(
        ["model_family", "tenure", "n_bins", "spread_metric"],
        kind="mergesort",
    ).reset_index(drop=True)




def _metric_orientation(metric: str) -> str:
    if metric in {"log_loss", "brier_score", "absolute_calibration_gap"}:
        return "LOWER_IS_BETTER"
    if metric in {
        "rows",
        "churn_prevalence_baseline",
        "actual_retention_rate",
        "mean_predicted_retention",
        "calibration_gap",
    }:
        return "NO_DIRECTION"
    return "HIGHER_IS_BETTER"




def _summarize_comparison(
    point_metrics: pd.DataFrame,
    replicates: pd.DataFrame,
    margins: Mapping[tuple[str, str], Mapping[str, Any]],
) -> pd.DataFrame:
    native = point_metrics.loc[point_metrics["model_role"].eq("NATIVE")].copy()
    pooled = point_metrics.loc[point_metrics["model_role"].eq("POOLED")].copy()
    identifiers = ["tenure", "model_family"]
    value_columns = [
        column
        for column in point_metrics.columns
        if column not in {*identifiers, "model_role", "bundle_manifest_sha256"}
        and pd.api.types.is_numeric_dtype(point_metrics[column])
    ]
    point_rows: list[dict[str, Any]] = []
    for key, native_group in native.groupby(identifiers, sort=True):
        pooled_group = pooled.loc[
            pooled["tenure"].eq(key[0]) & pooled["model_family"].eq(key[1])
        ]
        if len(native_group) != 1 or len(pooled_group) != 1:
            raise ValueError(f"Expected one native and pooled metric row for {key}.")
        native_row = native_group.iloc[0]
        pooled_row = pooled_group.iloc[0]
        for metric in value_columns:
            point_rows.append(
                {
                    "tenure": key[0],
                    "model_family": key[1],
                    "metric": metric,
                    "native_value": float(native_row[metric]),
                    "pooled_value": float(pooled_row[metric]),
                    "pooled_minus_native": float(pooled_row[metric] - native_row[metric]),
                }
            )
    output = pd.DataFrame(point_rows)
    if replicates.empty:
        output["bootstrap_replicates"] = 0
        output["bootstrap_mean_delta"] = np.nan
        output["bootstrap_std_delta"] = np.nan
        output["ci_2_5pct"] = np.nan
        output["ci_97_5pct"] = np.nan
        output["probability_pooled_better"] = np.nan
    else:
        summaries: list[dict[str, Any]] = []
        for key, group in replicates.groupby(
            ["tenure", "model_family", "metric"], sort=True
        ):
            delta = group["pooled_minus_native"].to_numpy(dtype=float)
            orientation = _metric_orientation(key[2])
            summaries.append(
                {
                    "tenure": key[0],
                    "model_family": key[1],
                    "metric": key[2],
                    "bootstrap_replicates": int(len(delta)),
                    "bootstrap_mean_delta": float(delta.mean()),
                    "bootstrap_std_delta": float(delta.std(ddof=1)),
                    "ci_2_5pct": float(np.quantile(delta, 0.025)),
                    "ci_97_5pct": float(np.quantile(delta, 0.975)),
                    "probability_pooled_better": float(
                        np.mean(delta < 0.0)
                        if orientation == "LOWER_IS_BETTER"
                        else (
                            np.mean(delta > 0.0)
                            if orientation == "HIGHER_IS_BETTER"
                            else np.nan
                        )
                    ),
                }
            )
        output = output.merge(
            pd.DataFrame(summaries),
            on=["tenure", "model_family", "metric"],
            how="left",
            validate="one_to_one",
        )
    orientations: list[str] = []
    margins_out: list[float] = []
    margin_sources: list[str] = []
    decisions: list[str] = []
    favorable_deltas: list[float] = []
    for row in output.itertuples(index=False):
        orientation = _metric_orientation(row.metric)
        orientations.append(orientation)
        favorable_deltas.append(
            -float(row.pooled_minus_native)
            if orientation == "LOWER_IS_BETTER"
            else (
                float(row.pooled_minus_native)
                if orientation == "HIGHER_IS_BETTER"
                else np.nan
            )
        )
        saved = margins.get((row.model_family, row.metric))
        margin = float(saved["noninferiority_margin"]) if saved else np.nan
        margins_out.append(margin)
        margin_sources.append(str(saved["margin_source"]) if saved else "")
        if saved is None:
            decisions.append("NO_PREDECLARED_POOLING_MARGIN")
        elif int(row.bootstrap_replicates) <= 0:
            decisions.append("NO_BOOTSTRAP_CI")
        elif orientation == "HIGHER_IS_BETTER":
            decisions.append(
                "PASS" if float(row.ci_2_5pct) >= -margin else "FAIL"
            )
        else:
            decisions.append(
                "PASS" if float(row.ci_97_5pct) <= margin else "FAIL"
            )
    output["orientation"] = orientations
    output["favorable_oriented_point_delta"] = favorable_deltas
    output["noninferiority_margin"] = margins_out
    output["margin_source"] = margin_sources
    output["noninferiority_decision"] = decisions
    return output.sort_values(
        ["model_family", "tenure", "metric"], kind="mergesort"
    ).reset_index(drop=True)




def _family_pooling_decisions(
    comparison: pd.DataFrame,
    margins: Mapping[tuple[str, str], Mapping[str, Any]],
    tenure_groups: Sequence[str],
    *,
    integrity_bypass_requested: bool,
) -> dict[str, str]:
    gate_rows = comparison.loc[
        comparison["noninferiority_decision"].isin(["PASS", "FAIL"])
    ]
    decisions: dict[str, str] = {}
    for family in MODEL_FAMILIES:
        if integrity_bypass_requested:
            decisions[family] = "DIAGNOSTIC_ONLY_INTEGRITY_CHECK_BYPASSED"
            continue
        family_margin_keys = [key for key in margins if key[0] == family]
        if not family_margin_keys:
            decisions[family] = "NO_PREDECLARED_POOLING_MARGIN"
            continue
        family_gates = gate_rows.loc[gate_rows["model_family"].eq(family)]
        expected_gate_count = len(family_margin_keys) * len(tenure_groups)
        if len(family_gates) != expected_gate_count:
            decisions[family] = "INCOMPLETE_GATES"
        elif family_gates["noninferiority_decision"].eq("PASS").all():
            decisions[family] = "RANKING_NONINFERIOR"
        else:
            decisions[family] = "KEEP_NATIVE_MODELS"
    return decisions




def _plot_lift_comparison(
    native: pd.DataFrame,
    pooled: pd.DataFrame,
    *,
    title: str,
    destination: Path,
) -> None:
    import matplotlib.pyplot as plt


    native = native.sort_values(["lift", "bin_rank"], kind="mergesort")
    order = native["bin_rank"].tolist()
    pooled = pooled.set_index("bin_rank").loc[order].reset_index()
    x = np.arange(len(native), dtype=float)
    width = 0.4
    fig, axis = plt.subplots(figsize=(12, 7))
    axis.bar(x - width / 2, native["lift"], width, label="Native")
    axis.bar(x + width / 2, pooled["lift"], width, label="Pooled")
    axis.axhline(1.0, linestyle="--", color="gray", linewidth=1.5)
    axis.set_title(title)
    axis.set_xlabel("Percentile")
    axis.set_ylabel("Lift")
    axis.set_xticks(x)
    axis.set_xticklabels(
        [f"{value:g}" for value in native["percentile_upper"].to_numpy()]
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25, linestyle=":")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)




def _plot_interactive_lift_comparison(
    lifts: pd.DataFrame,
    *,
    title: str,
    destination: Path,
) -> None:
    import plotly.graph_objects as go


    available = tuple(sorted(int(value) for value in lifts["n_bins"].unique()))
    if not available:
        raise ValueError("No lift tables were supplied for the interactive chart.")
    initial = 20 if 20 in available else available[0]


    def values_for(bins: int) -> tuple[list[str], list[np.ndarray], list[list[str]]]:
        selected = lifts.loc[lifts["n_bins"].eq(int(bins))]
        native = selected.loc[
            selected["model_role"].eq("NATIVE")
        ].sort_values(["lift", "bin_rank"], kind="mergesort")
        order = native["bin_rank"].tolist()
        pooled = (
            selected.loc[selected["model_role"].eq("POOLED")]
            .set_index("bin_rank")
            .loc[order]
            .reset_index()
        )
        if len(native) != bins or len(pooled) != bins:
            raise ValueError(f"Incomplete native/pooled lift table for {bins} groups.")
        x = [f"{value:g}" for value in native["percentile_upper"].to_numpy()]
        y = [
            native["lift"].to_numpy(dtype=float),
            pooled["lift"].to_numpy(dtype=float),
        ]
        text = [[f"{value:.2f}" for value in values] for values in y]
        return x, y, text


    x0, y0, text0 = values_for(initial)
    figure = go.Figure()
    for role, values, labels, color in (
        ("Native", y0[0], text0[0], "#2f5d8a"),
        ("Pooled", y0[1], text0[1], "#a33b1f"),
    ):
        figure.add_bar(
            name=role,
            x=x0,
            y=values,
            text=labels,
            textposition="outside",
            marker_color=color,
            hovertemplate=(
                f"Model: {role}<br>Percentile: %{{x}}%<br>"
                "Lift: %{y:.3f}<extra></extra>"
            ),
        )
    buttons = []
    for bins in available:
        x, y, labels = values_for(bins)
        buttons.append(
            {
                "label": str(bins),
                "method": "update",
                "args": [
                    {
                        "x": [x, x],
                        "y": y,
                        "text": labels,
                    },
                    {
                        "title": title,
                        "xaxis.title": "Percentile",
                    },
                ],
            }
        )
    maximum = float(lifts["lift"].replace([np.inf, -np.inf], np.nan).max())
    figure.update_layout(
        title=title,
        template="plotly_white",
        barmode="group",
        height=760,
        margin={"l": 70, "r": 40, "t": 100, "b": 60},
        xaxis={"title": "Percentile"},
        yaxis={
            "title": "Lift",
            "range": [0, maximum * 1.10],
            "gridcolor": "rgba(0,0,0,0.15)",
            "griddash": "dot",
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.0,
                "y": 1.10,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": buttons,
                "showactive": True,
            }
        ],
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 1.02},
    )
    figure.add_hline(y=1.0, line_dash="dash", line_color="gray")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(destination, include_plotlyjs=True, full_html=True)




def _parse_lift_bins(raw: str) -> tuple[int, ...]:
    return cross._parse_lift_bins(raw)




def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--tenures", nargs="+", required=True)
    parser.add_argument(
        "--pooled-baseline-only",
        action="store_true",
        help=(
            "Score only the frozen pooled model on its complete exact sealed "
            "pooled holdout; do not resolve or score native tenure models."
        ),
    )
    parser.add_argument("--pooled-run", type=Path)
    parser.add_argument(
        "--native-run",
        action="append",
        default=[],
        metavar="TENURE=PATH",
    )
    parser.add_argument(
        "--pooled-bundle-manifest", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--native-bundle-manifest",
        action="append",
        default=[],
        metavar="TENURE=PATH",
    )
    parser.add_argument("--target-data", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--lift-bins", type=_parse_lift_bins, default=cross.DEFAULT_LIFT_BINS
    )
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-verify-bundle-hashes", action="store_true")
    parser.add_argument("--allow-target-data-hash-mismatch", action="store_true")
    parser.add_argument("--skip-native-snapshot-payload-audit", action="store_true")
    parser.add_argument("--skip-charts", action="store_true")
    args = parser.parse_args(argv)
    args.project_root = args.project_root.expanduser().resolve()
    try:
        args.tenures = _normalize_groups(args.tenures)
    except ValueError as exc:
        parser.error(str(exc))
    if args.bootstrap_reps < 0:
        parser.error("--bootstrap-reps cannot be negative")
    if args.pooled_baseline_only:
        forbidden: list[str] = []
        if args.native_run:
            forbidden.append("--native-run")
        if args.native_bundle_manifest:
            forbidden.append("--native-bundle-manifest")
        if args.no_verify_bundle_hashes:
            forbidden.append("--no-verify-bundle-hashes")
        if args.allow_target_data_hash_mismatch:
            forbidden.append("--allow-target-data-hash-mismatch")
        if args.skip_native_snapshot_payload_audit:
            forbidden.append("--skip-native-snapshot-payload-audit")
        if forbidden:
            parser.error(
                "Pooled baseline mode requires strict pooled bundle/snapshot "
                "identity and has no native inputs; remove: "
                + ", ".join(forbidden)
            )
    return args




def _output_directory(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        raw = args.output_dir.expanduser()
        return raw.resolve() if raw.is_absolute() else (args.project_root / raw).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    scope = "_".join(_safe_filename(value) for value in args.tenures)
    if args.pooled_baseline_only:
        return (
            args.project_root
            / "output"
            / "baseline_lift"
            / f"pooled_tenures_{scope}_{stamp}"
        )
    return (
        args.project_root
        / "output"
        / "pooled_vs_native"
        / f"tenures_{scope}_{stamp}"
    )




def _pooled_baseline_lift_outputs(
    *,
    y_retention: np.ndarray,
    p_retention: np.ndarray,
    row_id: np.ndarray,
    lift_bins: Sequence[int],
    model_family: str,
    tenure_scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the cross-tenure lift and spread functions to one pooled population."""


    y_retention = np.asarray(y_retention, dtype=int)
    p_retention = np.asarray(p_retention, dtype=float)
    row_id = np.asarray(row_id, dtype=np.uint64)
    if not (len(y_retention) == len(p_retention) == len(row_id)):
        raise ValueError("Pooled baseline target, prediction, and row-id lengths differ.")


    lift_frames: list[pd.DataFrame] = []
    spread_records: list[dict[str, Any]] = []
    p_churn = 1.0 - p_retention
    for raw_bins in lift_bins:
        bins = int(raw_bins)
        table = cross.build_lift_table(
            1 - y_retention,
            p_churn,
            n_bins=bins,
            row_id=row_id,
        )
        spread = cross.summarize_lift_spread(table)
        table.insert(0, "n_bins", bins)
        table.insert(0, "target_tenure", tenure_scope)
        table.insert(0, "source_tenure", tenure_scope)
        table.insert(0, "model_family", model_family)
        lift_frames.append(table)
        spread_records.append(
            {
                "model_family": model_family,
                "source_tenure": tenure_scope,
                "target_tenure": tenure_scope,
                "n_bins": bins,
                **spread,
            }
        )
    if not lift_frames:
        raise ValueError("Pooled baseline requires at least one lift-bin count.")
    return (
        pd.concat(lift_frames, ignore_index=True),
        pd.DataFrame(spread_records),
    )




def _run_pooled_baseline(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    verify_hashes: bool,
) -> int:
    """Score one frozen pooled model on its own exact sealed pooled holdout."""


    pooled_run = resolve_profile_run(
        args.project_root,
        args.tenures,
        explicit=args.pooled_run,
    )
    canonical_groups = tuple(cross._selected_groups(pooled_run.profile))
    if set(canonical_groups) != set(args.tenures):
        raise AssertionError(
            "Resolved pooled profile changed the requested tenure scope."
        )
    pooled_manifest_overrides = [
        _resolve_manifest_path(args.project_root, path)
        for path in args.pooled_bundle_manifest
    ]
    context = _load_bundle_context(
        project_root=args.project_root,
        role="POOLED_BASELINE",
        tenure_groups=canonical_groups,
        source_run=pooled_run,
        explicit_manifests=pooled_manifest_overrides,
        verify_hashes=verify_hashes,
    )
    _validate_pooled_baseline_context(context)


    runtime_version_rows: list[dict[str, Any]] = []
    for family in MODEL_FAMILIES:
        record = context.bundles[family]
        for row in cross._validate_bundle_runtime_versions(record):
            runtime_version_rows.append(
                {
                    "bundle_manifest_path": str(record.path),
                    **row,
                }
            )


    tenure_column = str(pooled_run.profile.get("tenure_dimension", "")).strip()
    if not tenure_column:
        raise ValueError("Pooled run profile does not record tenure_dimension.")
    identity_columns = tuple(
        str(value) for value in context.holdout["identity_columns"]
    )
    policy_cluster_column = str(context.holdout["policy_cluster_column"])
    required_columns = set(context.scoring_columns)
    required_columns.update(identity_columns)
    required_columns.update(
        {
            context.target_column,
            tenure_column,
            policy_cluster_column,
            str(context.holdout["cohort_column"]),
        }
    )
    _, _, pooled_data, pooled_data_audit = cross.resolve_target_data(
        pooled_run,
        project_root=args.project_root,
        required_columns=required_columns,
        target_column=context.target_column,
        tenure_column=tenure_column,
        explicit=args.target_data,
        allow_hash_mismatch=False,
    )
    snapshot_match = (
        str(pooled_data_audit["sha256"]) == context.source_snapshot_sha256
    )
    if not snapshot_match:
        raise RuntimeError(
            "Resolved pooled run snapshot does not match the frozen pooled "
            "bundle's source snapshot identity."
        )
    pooled_scope_rows = _filter_groups(
        pooled_data,
        tenure_column,
        canonical_groups,
    )
    del pooled_data
    holdout, month_coverage = cross._filter_evaluation_window(
        pooled_scope_rows,
        cohort_column=str(context.holdout["cohort_column"]),
        holdout_contract=context.holdout,
        evaluation_window="source-holdout",
    )
    del pooled_scope_rows


    observed_population_sha256 = _population_sha256(
        holdout,
        identity_columns,
    )
    expected_population_sha256 = str(context.holdout["population_sha256"])
    population_hash_match = (
        observed_population_sha256 == expected_population_sha256
    )
    row_count_match = len(holdout) == int(context.holdout["rows"])
    if not population_hash_match or not row_count_match:
        raise RuntimeError(
            "Pooled baseline rows do not match the frozen bundle's exact "
            f"sealed holdout: rows={len(holdout)}/{context.holdout['rows']}; "
            f"hash_match={population_hash_match}."
        )


    row_id_series = cross._deterministic_row_id(holdout, identity_columns)
    if row_id_series.nunique(dropna=False) != len(holdout):
        raise RuntimeError("Pooled baseline holdout row identity is not unique.")
    row_id = row_id_series.to_numpy(dtype=np.uint64)
    y_retention = pd.to_numeric(
        holdout[context.target_column], errors="raise"
    ).astype(int).to_numpy()
    if set(np.unique(y_retention)) != {0, 1}:
        raise ValueError(
            "Pooled baseline holdout must contain both retention classes 0 and 1."
        )


    scoring_frame = holdout.loc[:, sorted(context.scoring_columns)].copy()
    predictions: dict[str, np.ndarray] = {}
    for family in MODEL_FAMILIES:
        predictions[family] = cross.score_target_with_frozen_bundle(
            context.bundles[family].path,
            scoring_frame,
            verify_hashes=verify_hashes,
        )


    scope_key = context.scope_key
    state_audit = cross._state_mapping_audit(
        holdout,
        predictions,
        context.shared_state,
        target_column=context.target_column,
        source_tenure=scope_key,
        target_tenure=scope_key,
    )
    state_audit_path = output_dir / "source_state_mapping_applied_to_target.csv"
    state_audit.to_csv(state_audit_path, index=False)
    invalid_state = state_audit.loc[
        ~state_audit["mapping_status"].eq("SOURCE_MAPPING_APPLIED")
    ]
    if not invalid_state.empty:
        raise RuntimeError(
            "Pooled baseline states violate the frozen pooled mapping. See "
            f"{state_audit_path}."
        )


    metric_rows: list[dict[str, Any]] = []
    lift_rows: list[pd.DataFrame] = []
    spread_rows: list[dict[str, Any]] = []
    chart_rows: list[dict[str, Any]] = []
    for family in MODEL_FAMILIES:
        p_retention = predictions[family]
        metric_rows.append(
            {
                "model_family": family,
                "model_role": POOLED_BASELINE_MODEL_ROLE,
                "source_tenure": scope_key,
                "target_tenure": scope_key,
                "source_bundle_manifest_sha256": cross.sha256_file(
                    context.bundles[family].path
                ),
                **cross._ranking_metrics(y_retention, p_retention),
            }
        )
        family_lift_frame, family_spread_frame = _pooled_baseline_lift_outputs(
            y_retention=y_retention,
            p_retention=p_retention,
            row_id=row_id,
            lift_bins=args.lift_bins,
            model_family=family,
            tenure_scope=scope_key,
        )
        lift_rows.append(family_lift_frame)
        spread_rows.extend(family_spread_frame.to_dict(orient="records"))
        family_lifts = {
            int(bins): family_lift_frame.loc[
                family_lift_frame["n_bins"].eq(int(bins))
            ].copy()
            for bins in args.lift_bins
        }
        if args.skip_charts:
            chart_rows.append(
                {
                    "model_family": family,
                    "chart_type": "LIFT_CALCULATIONS",
                    "status": "SKIPPED_BY_REQUEST",
                    "path": "",
                    "detail": "",
                }
            )
        else:
            chart_path = output_dir / "charts" / f"{family}_lift_chart.html"
            try:
                cross.plot_lift_like_graph1(
                    family_lifts,
                    title="All-tenure Lift Calculations",
                    save_html_path=chart_path,
                )
            except ImportError as exc:
                chart_rows.append(
                    {
                        "model_family": family,
                        "chart_type": "LIFT_CALCULATIONS",
                        "status": "SKIPPED_MISSING_PLOTLY",
                        "path": "",
                        "detail": str(exc),
                    }
                )
            else:
                chart_rows.append(
                    {
                        "model_family": family,
                        "chart_type": "LIFT_CALCULATIONS",
                        "status": "WRITTEN",
                        "path": str(chart_path),
                        "detail": "",
                    }
                )


    state_chart_path = output_dir / "charts" / "shared_source_state_mapping_on_target.html"
    if args.skip_charts:
        chart_rows.append(
            {
                "model_family": "shared",
                "chart_type": "SHARED_SOURCE_STATE_MAPPING_ON_TARGET",
                "status": "SKIPPED_BY_REQUEST",
                "path": "",
                "detail": "",
            }
        )
    else:
        try:
            cross.plot_source_state_mapping_choropleth(
                state_audit,
                source_tenure=scope_key,
                target_tenure=scope_key,
                save_html_path=state_chart_path,
            )
        except ImportError as exc:
            chart_rows.append(
                {
                    "model_family": "shared",
                    "chart_type": "SHARED_SOURCE_STATE_MAPPING_ON_TARGET",
                    "status": "SKIPPED_MISSING_PLOTLY",
                    "path": "",
                    "detail": str(exc),
                }
            )
        else:
            chart_rows.append(
                {
                    "model_family": "shared",
                    "chart_type": "SHARED_SOURCE_STATE_MAPPING_ON_TARGET",
                    "status": "WRITTEN",
                    "path": str(state_chart_path),
                    "detail": (
                        "One shared frozen pooled state mapping is used by "
                        "Logistic and XGBoost; no holdout-native regrouping occurs."
                    ),
                }
            )


    model_metrics_path = output_dir / "model_metrics.csv"
    lift_table_path = output_dir / "lift_by_equal_count_group.csv"
    lift_spread_path = output_dir / "lift_spread_summary.csv"
    chart_status_path = output_dir / "chart_generation_status.csv"
    pd.DataFrame(metric_rows).to_csv(model_metrics_path, index=False)
    pd.concat(lift_rows, ignore_index=True).to_csv(lift_table_path, index=False)
    pd.DataFrame(spread_rows).to_csv(lift_spread_path, index=False)
    pd.DataFrame(chart_rows).to_csv(chart_status_path, index=False)


    bundle_identity_rows: list[dict[str, Any]] = []
    for family, record in context.bundles.items():
        bundle_identity_rows.append(
            {
                "model_family": family,
                "manifest_path": str(record.path),
                "manifest_sha256": cross.sha256_file(record.path),
                **cross._manifest_identity(record.payload),
            }
        )
    source_bundle_identity_path = output_dir / "source_bundle_identity.csv"
    runtime_version_validation_path = output_dir / "runtime_version_validation.csv"
    pd.DataFrame(bundle_identity_rows).to_csv(
        source_bundle_identity_path, index=False
    )
    pd.DataFrame(runtime_version_rows).to_csv(
        runtime_version_validation_path, index=False
    )


    inventory = _output_inventory(output_dir)
    inventory_path = output_dir / "output_inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    _write_json(
        output_dir / "evaluation_manifest.json",
        {
            "evaluation_contract": POOLED_BASELINE_CONTRACT,
            "evaluation_mode": "POOLED_BASELINE",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_versions": _runtime_versions(),
            "bundle_runtime_version_validation": {
                "policy": "STRICT_EXACT_MATCH",
                "status": "PASS",
                "override_available": False,
                "audit_csv": runtime_version_validation_path,
                "checks": runtime_version_rows,
            },
            "pooled_run": {
                "path": pooled_run.path,
                "profile": pooled_run.profile,
                "profile_hash": pooled_run.profile_hash,
            },
            "tenure_groups": list(canonical_groups),
            "tenure_scope_key": scope_key,
            "target_column": context.target_column,
            "tenure_column": tenure_column,
            "identity_columns": list(identity_columns),
            "policy_cluster_column": policy_cluster_column,
            "pooled_target_data": pooled_data_audit,
            "pooled_bundle_snapshot_sha256": context.source_snapshot_sha256,
            "pooled_bundle_snapshot_match": snapshot_match,
            "pooled_holdout": dict(context.holdout),
            "pooled_holdout_observed_population_sha256": (
                observed_population_sha256
            ),
            "pooled_holdout_population_hash_match": population_hash_match,
            "pooled_holdout_row_count_match": row_count_match,
            "evaluation_window": "source-holdout",
            "evaluation_month_coverage": month_coverage,
            "source_bundles": {
                family: {
                    "manifest_path": record.path,
                    "manifest_sha256": cross.sha256_file(record.path),
                    "identity": cross._manifest_identity(record.payload),
                }
                for family, record in context.bundles.items()
            },
            "source_bundle_selection": {
                "selection_mode": (
                    "EXPLICIT_MANIFEST"
                    if pooled_manifest_overrides
                    else "EXACT_SOURCE_RUN_IDENTITY"
                ),
                "selected_source_run_id": pooled_run.path.name,
                "selected_source_profile_hash": pooled_run.profile_hash,
                "selected_tenure_scope": scope_key,
                "filesystem_timestamp_used": False,
            },
            "source_state_mapping_diagnostic": {
                "csv_path": state_audit_path,
                "mapping_sha256": context.shared_state.mapping_sha256,
                "categories": list(context.shared_state.categories),
                "derived_feature": context.shared_state.derived_feature,
                "source_feature": context.shared_state.source_feature,
                "holdout_native_grouping_created": False,
            },
            "charts": chart_rows,
            "lift_bins": list(args.lift_bins),
            "lift_definition": (
                "group churn rate divided by pooled all-tenure holdout churn rate"
            ),
            "lift_chart_order": (
                "ascending lift, matching the cross-tenure evaluator"
            ),
            "lift_spread_outputs": {
                "highest_minus_lowest_lift": (
                    "highest group lift minus lowest group lift"
                ),
                "highest_divided_by_lowest_lift": (
                    "highest group lift divided by lowest group lift"
                ),
                "ordered_first_to_last_lift_ratio": (
                    "highest-score group lift divided by lowest-score group lift"
                ),
            },
            "artifacts": {
                "model_metrics_csv": model_metrics_path,
                "lift_by_equal_count_group_csv": lift_table_path,
                "lift_spread_summary_csv": lift_spread_path,
                "chart_generation_status_csv": chart_status_path,
                "source_bundle_identity_csv": source_bundle_identity_path,
                "runtime_version_validation_csv": runtime_version_validation_path,
                "source_state_mapping_applied_to_target_csv": state_audit_path,
                "output_inventory_csv": inventory_path,
            },
            "model_role": POOLED_BASELINE_MODEL_ROLE,
            "models_fitted": 0,
            "target_learning_operations": 0,
            "bundle_hash_verification": verify_hashes,
            "output_inventory": {
                "path": str(inventory_path),
                "sha256": cross.sha256_file(inventory_path),
                "file_count_excluding_manifest_and_inventory": int(len(inventory)),
            },
        },
    )
    print(f"Frozen pooled-baseline evaluation complete: {output_dir}")
    print(f"Lift spread: {lift_spread_path}")
    return 0




def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = _output_directory(args)
    output_dir.mkdir(parents=True, exist_ok=False)
    verify_hashes = not args.no_verify_bundle_hashes
    if args.pooled_baseline_only:
        return _run_pooled_baseline(
            args,
            output_dir,
            verify_hashes=verify_hashes,
        )


    integrity_bypass_requested = bool(
        args.no_verify_bundle_hashes
        or args.allow_target_data_hash_mismatch
        or args.skip_native_snapshot_payload_audit
    )


    pooled_run = resolve_profile_run(
        args.project_root, args.tenures, explicit=args.pooled_run
    )
    canonical_groups = tuple(cross._selected_groups(pooled_run.profile))
    if set(canonical_groups) != set(args.tenures):
        raise AssertionError("Resolved pooled profile changed the requested tenure scope.")


    native_run_overrides = _assignment_map(
        args.native_run, label="--native-run", allowed=canonical_groups
    )
    native_manifest_overrides = _assignment_map(
        args.native_bundle_manifest,
        label="--native-bundle-manifest",
        allowed=canonical_groups,
    )
    native_manifest_overrides = {
        tenure: _resolve_manifest_path(args.project_root, path)
        for tenure, path in native_manifest_overrides.items()
    }
    pooled_manifest_overrides = [
        _resolve_manifest_path(args.project_root, path)
        for path in args.pooled_bundle_manifest
    ]
    native_runs: dict[str, cross.TargetRun] = {
        tenure: resolve_profile_run(
            args.project_root,
            (tenure,),
            explicit=native_run_overrides.get(tenure),
        )
        for tenure in canonical_groups
    }


    pooled_context = _load_bundle_context(
        project_root=args.project_root,
        role="POOLED",
        tenure_groups=canonical_groups,
        source_run=pooled_run,
        explicit_manifests=pooled_manifest_overrides,
        verify_hashes=verify_hashes,
    )
    native_contexts: dict[str, BundleContext] = {}
    for tenure in canonical_groups:
        manifest = native_manifest_overrides.get(tenure)
        native_contexts[tenure] = _load_bundle_context(
            project_root=args.project_root,
            role="NATIVE",
            tenure_groups=(tenure,),
            source_run=native_runs[tenure],
            explicit_manifests=[] if manifest is None else [manifest],
            verify_hashes=verify_hashes,
        )


    target_column, identity_columns, policy_cluster_column = _validate_contexts(
        pooled_context, native_contexts
    )
    tenure_column = str(pooled_run.profile.get("tenure_dimension", "")).strip()
    if not tenure_column:
        raise ValueError("Pooled run profile does not record tenure_dimension.")


    contexts = [pooled_context, *native_contexts.values()]
    runtime_version_rows: list[dict[str, Any]] = []
    for context in contexts:
        for family in cross.MODEL_FAMILIES:
            record = context.bundles[family]
            for row in cross._validate_bundle_runtime_versions(record):
                runtime_version_rows.append(
                    {
                        "bundle_role": context.role,
                        "tenure_scope": ",".join(context.tenure_groups),
                        "bundle_manifest_path": str(record.path),
                        **row,
                    }
                )
    required_columns = set().union(*(set(value.scoring_columns) for value in contexts))
    required_columns.update(identity_columns)
    required_columns.update(
        {
            target_column,
            tenure_column,
            str(pooled_context.holdout["cohort_column"]),
        }
    )
    _, _, pooled_data, pooled_data_audit = cross.resolve_target_data(
        pooled_run,
        project_root=args.project_root,
        required_columns=required_columns,
        target_column=target_column,
        tenure_column=tenure_column,
        explicit=args.target_data,
        allow_hash_mismatch=args.allow_target_data_hash_mismatch,
    )
    pooled_bundle_snapshot_match = (
        str(pooled_data_audit["sha256"])
        == pooled_context.source_snapshot_sha256
    )
    if (
        not pooled_bundle_snapshot_match
        and not args.allow_target_data_hash_mismatch
    ):
        raise RuntimeError(
            "Resolved pooled run snapshot does not match the pooled bundle's "
            "recorded source snapshot identity."
        )
    pooled_scope_rows = _filter_groups(pooled_data, tenure_column, canonical_groups)
    del pooled_data
    pooled_holdout, month_coverage = cross._filter_evaluation_window(
        pooled_scope_rows,
        cohort_column=str(pooled_context.holdout["cohort_column"]),
        holdout_contract=pooled_context.holdout,
        evaluation_window="source-holdout",
    )
    del pooled_scope_rows
    observed_pooled_hash = _population_sha256(pooled_holdout, identity_columns)
    expected_pooled_hash = str(pooled_context.holdout["population_sha256"])
    if observed_pooled_hash != expected_pooled_hash:
        raise RuntimeError(
            "Combined snapshot holdout identity does not match the frozen pooled "
            f"bundle: observed={observed_pooled_hash}; expected={expected_pooled_hash}."
        )
    if len(pooled_holdout) != int(pooled_context.holdout["rows"]):
        raise RuntimeError("Combined holdout row count differs from its frozen seal.")
    pooled_union_row_id = cross._deterministic_row_id(
        pooled_holdout, identity_columns
    )
    if pooled_union_row_id.nunique(dropna=False) != len(pooled_holdout):
        raise RuntimeError("Combined holdout row identity is not unique.")


    # Model-selection tolerances saved by the individual pipeline runs answer a
    # different question and are not a pooling-acceptance contract.  Until a
    # pooling margin is explicitly specified before evaluation, results remain
    # descriptive and cannot approve combining tenure models.
    margin_contract: dict[tuple[str, str], dict[str, Any]] = {}
    metric_rows: list[dict[str, Any]] = []
    lift_rows: list[pd.DataFrame] = []
    spread_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    population_audits: list[dict[str, Any]] = []
    state_audits: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []
    spread_bootstrap_frames: list[pd.DataFrame] = []
    chart_rows: list[dict[str, Any]] = []
    bundle_identity_rows: list[dict[str, Any]] = []
    native_snapshot_audits: dict[str, dict[str, Any]] = {}


    for context in contexts:
        for family, record in context.bundles.items():
            bundle_identity_rows.append(
                {
                    "model_role": context.role,
                    "tenure_scope": context.scope_key,
                    "model_family": family,
                    "source_run_id": context.source_run.path.name,
                    "manifest_path": str(record.path),
                    "manifest_sha256": cross.sha256_file(record.path),
                    "state_mapping_sha256": context.shared_state.mapping_sha256,
                    **cross._manifest_identity(record.payload),
                }
            )


    for tenure in canonical_groups:
        native_context = native_contexts[tenure]
        tenure_mask = pooled_holdout[tenure_column].map(_normalize_tenure).eq(tenure)
        same_rows = pooled_holdout.loc[tenure_mask].copy().reset_index(drop=True)
        if same_rows.empty:
            raise ValueError(f"Combined holdout contains no tenure {tenure!r} rows.")
        row_id_series = cross._deterministic_row_id(same_rows, identity_columns)
        row_id = row_id_series.to_numpy(dtype=np.uint64)
        y_retention = pd.to_numeric(
            same_rows[target_column], errors="raise"
        ).astype(int).to_numpy()
        if set(np.unique(y_retention)) != {0, 1}:
            raise ValueError(f"Tenure {tenure} holdout does not contain both classes.")


        native_hash = _population_sha256(same_rows, identity_columns)
        expected_native_hash = str(native_context.holdout["population_sha256"])
        count_match = len(same_rows) == int(native_context.holdout["rows"])
        hash_match = native_hash == expected_native_hash
        if not count_match or not hash_match:
            raise RuntimeError(
                f"Pooled tenure-{tenure} subset is not the native model's exact "
                f"sealed holdout: rows={len(same_rows)}/"
                f"{native_context.holdout['rows']}; hash_match={hash_match}."
            )


        comparison_columns = sorted(
            set(pooled_context.scoring_columns)
            | set(native_context.scoring_columns)
            | set(identity_columns)
            | {
                target_column,
                tenure_column,
                str(pooled_context.holdout["cohort_column"]),
            }
        )
        pooled_payload_hash = _payload_sha256(same_rows, comparison_columns)
        native_payload_hash = ""
        payload_match: bool | None = None
        native_data_audit: Mapping[str, Any] | None = None
        native_bundle_snapshot_match: bool | None = None
        if not args.skip_native_snapshot_payload_audit:
            _, _, native_data, native_data_audit = cross.resolve_target_data(
                native_runs[tenure],
                project_root=args.project_root,
                required_columns=comparison_columns,
                target_column=target_column,
                tenure_column=tenure_column,
                explicit=None,
                allow_hash_mismatch=args.allow_target_data_hash_mismatch,
            )
            native_bundle_snapshot_match = (
                str(native_data_audit["sha256"])
                == native_context.source_snapshot_sha256
            )
            if (
                not native_bundle_snapshot_match
                and not args.allow_target_data_hash_mismatch
            ):
                raise RuntimeError(
                    f"Resolved native tenure-{tenure} snapshot does not match "
                    "the native bundle's source snapshot identity."
                )
            native_snapshot_audits[tenure] = {
                **dict(native_data_audit),
                "bundle_source_snapshot_sha256": (
                    native_context.source_snapshot_sha256
                ),
                "bundle_source_snapshot_match": native_bundle_snapshot_match,
            }
            native_rows = _filter_groups(
                native_data, tenure_column, (tenure,)
            )
            del native_data
            native_holdout, _ = cross._filter_evaluation_window(
                native_rows,
                cohort_column=str(native_context.holdout["cohort_column"]),
                holdout_contract=native_context.holdout,
                evaluation_window="source-holdout",
            )
            del native_rows
            native_payload_hash = _payload_sha256(
                native_holdout, comparison_columns
            )
            payload_match = native_payload_hash == pooled_payload_hash
            if not payload_match:
                raise RuntimeError(
                    f"Combined and native tenure-{tenure} holdout inputs/outcomes "
                    "differ on the exact comparison payload."
                )
            del native_holdout


        population_audits.append(
            {
                "tenure": tenure,
                "rows": int(len(same_rows)),
                "churn_events": int((1 - y_retention).sum()),
                "actual_retention_rate": float(y_retention.mean()),
                "cohort_start": str(
                    pd.to_datetime(
                        same_rows[pooled_context.holdout["cohort_column"]]
                    ).min()
                ),
                "cohort_end": str(
                    pd.to_datetime(
                        same_rows[pooled_context.holdout["cohort_column"]]
                    ).max()
                ),
                "duplicate_identity_rows": int(
                    same_rows.loc[:, list(identity_columns)]
                    .duplicated(keep=False)
                    .sum()
                ),
                "missing_identity_rows": int(
                    same_rows.loc[:, list(identity_columns)].isna().any(axis=1).sum()
                ),
                "observed_population_sha256": native_hash,
                "native_expected_population_sha256": expected_native_hash,
                "native_population_hash_match": hash_match,
                "native_row_count_match": count_match,
                "pooled_snapshot_payload_sha256": pooled_payload_hash,
                "native_snapshot_payload_sha256": native_payload_hash,
                "native_snapshot_payload_match": payload_match,
                "native_bundle_snapshot_sha256": (
                    native_context.source_snapshot_sha256
                ),
                "native_resolved_snapshot_sha256": (
                    "" if native_data_audit is None else native_data_audit["sha256"]
                ),
                "native_resolved_snapshot_recorded_sha256": (
                    ""
                    if native_data_audit is None
                    else native_data_audit["recorded_sha256"]
                ),
                "native_resolved_snapshot_hash_match": (
                    None if native_data_audit is None else native_data_audit["hash_match"]
                ),
                "native_bundle_snapshot_match": native_bundle_snapshot_match,
                "native_snapshot_payload_audit_skipped": bool(
                    args.skip_native_snapshot_payload_audit
                ),
            }
        )


        predictions: dict[tuple[str, str], np.ndarray] = {}
        for role, context in (("NATIVE", native_context), ("POOLED", pooled_context)):
            scoring_frame = same_rows.loc[:, sorted(context.scoring_columns)].copy()
            role_predictions: dict[str, np.ndarray] = {}
            for family in MODEL_FAMILIES:
                probability = cross.score_target_with_frozen_bundle(
                    context.bundles[family].path,
                    scoring_frame,
                    verify_hashes=verify_hashes,
                )
                predictions[(role, family)] = probability
                role_predictions[family] = probability
                metrics = _all_metrics(y_retention, probability, row_id)
                metric_rows.append(
                    {
                        "tenure": tenure,
                        "model_family": family,
                        "model_role": role,
                        "bundle_manifest_sha256": cross.sha256_file(
                            context.bundles[family].path
                        ),
                        **metrics,
                    }
                )
                p_churn = 1.0 - probability
                for bins in args.lift_bins:
                    table = cross.build_lift_table(
                        1 - y_retention,
                        p_churn,
                        n_bins=bins,
                        row_id=row_id,
                    )
                    table.insert(0, "n_bins", bins)
                    table.insert(0, "model_role", role)
                    table.insert(0, "model_family", family)
                    table.insert(0, "tenure", tenure)
                    lift_rows.append(table)
                    spread_rows.append(
                        {
                            "tenure": tenure,
                            "model_family": family,
                            "model_role": role,
                            "n_bins": bins,
                            **cross.summarize_lift_spread(table),
                        }
                    )
            state = cross._state_mapping_audit(
                same_rows,
                role_predictions,
                context.shared_state,
                target_column=target_column,
                source_tenure=context.scope_key,
                target_tenure=tenure,
            )
            state.insert(0, "model_role", role)
            invalid = ~state["mapping_status"].eq("SOURCE_MAPPING_APPLIED")
            if invalid.any():
                raise RuntimeError(
                    f"{role} state mapping has {int(invalid.sum())} unmapped rows "
                    f"for tenure {tenure}."
                )
            state_audits.append(state)


        prediction_frame = pd.DataFrame(
            {
                "row_id": _row_id_strings(row_id_series),
                "policy_cluster_id": _cluster_hash(
                    same_rows[policy_cluster_column]
                ),
                "cohort_date": pd.to_datetime(
                    same_rows[pooled_context.holdout["cohort_column"]]
                ).dt.strftime("%Y-%m-%d"),
                "tenure": tenure,
                "actual_retention": y_retention,
                "actual_churn": 1 - y_retention,
                "native_logistic_retention_probability": predictions[
                    ("NATIVE", "logistic")
                ],
                "pooled_logistic_retention_probability": predictions[
                    ("POOLED", "logistic")
                ],
                "native_xgboost_retention_probability": predictions[
                    ("NATIVE", "xgboost")
                ],
                "pooled_xgboost_retention_probability": predictions[
                    ("POOLED", "xgboost")
                ],
            }
        )
        prediction_rows.append(prediction_frame)


        for family in MODEL_FAMILIES:
            bootstrap_frames.append(
                _bootstrap_pair(
                    tenure=tenure,
                    family=family,
                    y_retention=y_retention,
                    native_probability=predictions[("NATIVE", family)],
                    pooled_probability=predictions[("POOLED", family)],
                    row_id=row_id,
                    clusters=same_rows[policy_cluster_column],
                    reps=args.bootstrap_reps,
                    seed=args.seed,
                )
            )
            spread_bootstrap_frames.append(
                _bootstrap_lift_spread_pair(
                    tenure=tenure,
                    family=family,
                    y_retention=y_retention,
                    native_probability=predictions[("NATIVE", family)],
                    pooled_probability=predictions[("POOLED", family)],
                    row_id=row_id,
                    clusters=same_rows[policy_cluster_column],
                    lift_bins=args.lift_bins,
                    reps=args.bootstrap_reps,
                    seed=args.seed,
                )
            )


        if args.skip_charts:
            for family in MODEL_FAMILIES:
                for chart_type in ("STATIC_PNG", "INTERACTIVE_HTML"):
                    chart_rows.append(
                        {
                            "tenure": tenure,
                            "model_family": family,
                            "chart_type": chart_type,
                            "status": "SKIPPED_BY_REQUEST",
                            "path": "",
                            "detail": "",
                        }
                    )
        else:
            all_lifts = pd.concat(lift_rows, ignore_index=True)
            for family in MODEL_FAMILIES:
                available = sorted(
                    all_lifts.loc[
                        all_lifts["tenure"].eq(tenure)
                        & all_lifts["model_family"].eq(family),
                        "n_bins",
                    ].unique()
                )
                bins = 10 if 10 in available else max(available)
                selection = all_lifts.loc[
                    all_lifts["tenure"].eq(tenure)
                    & all_lifts["model_family"].eq(family)
                    & all_lifts["n_bins"].eq(bins)
                ]
                destination = (
                    output_dir
                    / "charts"
                    / f"tenure_{_safe_filename(tenure)}_{family}_lift_comparison.png"
                )
                try:
                    _plot_lift_comparison(
                        selection.loc[selection["model_role"].eq("NATIVE")],
                        selection.loc[selection["model_role"].eq("POOLED")],
                        title=(
                            f"Lift Calculations: tenure {tenure} {family.title()} "
                            f"({bins} groups)"
                        ),
                        destination=destination,
                    )
                except ImportError as exc:
                    chart_rows.append(
                        {
                            "tenure": tenure,
                            "model_family": family,
                            "chart_type": "STATIC_PNG",
                            "status": "SKIPPED_MISSING_MATPLOTLIB",
                            "path": "",
                            "detail": str(exc),
                        }
                    )
                else:
                    chart_rows.append(
                        {
                            "tenure": tenure,
                            "model_family": family,
                            "chart_type": "STATIC_PNG",
                            "status": "WRITTEN",
                            "path": str(destination),
                            "detail": f"Static comparison using {bins} equal-count groups.",
                        }
                    )
                family_lifts = all_lifts.loc[
                    all_lifts["tenure"].eq(tenure)
                    & all_lifts["model_family"].eq(family)
                ]
                interactive_destination = (
                    output_dir
                    / "charts"
                    / (
                        f"tenure_{_safe_filename(tenure)}_{family}_"
                        "lift_comparison_interactive.html"
                    )
                )
                try:
                    _plot_interactive_lift_comparison(
                        family_lifts,
                        title=(
                            f"Lift Calculations: tenure {tenure} "
                            f"{family.title()} pooled vs native"
                        ),
                        destination=interactive_destination,
                    )
                except ImportError as exc:
                    chart_rows.append(
                        {
                            "tenure": tenure,
                            "model_family": family,
                            "chart_type": "INTERACTIVE_HTML",
                            "status": "SKIPPED_MISSING_PLOTLY",
                            "path": "",
                            "detail": str(exc),
                        }
                    )
                else:
                    chart_rows.append(
                        {
                            "tenure": tenure,
                            "model_family": family,
                            "chart_type": "INTERACTIVE_HTML",
                            "status": "WRITTEN",
                            "path": str(interactive_destination),
                            "detail": (
                                "Buttons select 3, 4, 5, 10, or 20 "
                                "equal-count groups."
                            ),
                        }
                    )


    point_metrics = pd.DataFrame(metric_rows)
    lifts = pd.concat(lift_rows, ignore_index=True)
    bootstrap = (
        pd.concat(bootstrap_frames, ignore_index=True)
        if bootstrap_frames and any(not value.empty for value in bootstrap_frames)
        else pd.DataFrame()
    )
    spread_bootstrap = (
        pd.concat(spread_bootstrap_frames, ignore_index=True)
        if spread_bootstrap_frames
        and any(not value.empty for value in spread_bootstrap_frames)
        else pd.DataFrame()
    )
    comparison = _summarize_comparison(point_metrics, bootstrap, margin_contract)
    spreads = pd.DataFrame(spread_rows)
    spread_comparison = _summarize_lift_spread_comparison(
        spreads, spread_bootstrap
    )
    if integrity_bypass_requested:
        comparison.loc[
            comparison["noninferiority_decision"].isin(["PASS", "FAIL"]),
            "noninferiority_decision",
        ] = "DIAGNOSTIC_ONLY_INTEGRITY_CHECK_BYPASSED"


    point_metrics.to_csv(output_dir / "model_metrics_by_tenure.csv", index=False)
    lifts.to_csv(output_dir / "lift_by_equal_count_group.csv", index=False)
    spreads.to_csv(output_dir / "lift_spread_summary.csv", index=False)
    spread_comparison.to_csv(
        output_dir / "lift_spread_comparison.csv", index=False
    )
    comparison.to_csv(output_dir / "paired_metric_comparison.csv", index=False)
    if not bootstrap.empty:
        bootstrap.to_csv(
            output_dir / "paired_bootstrap_replicates.csv", index=False
        )
    if not spread_bootstrap.empty:
        spread_bootstrap.to_csv(
            output_dir / "paired_lift_spread_bootstrap_replicates.csv",
            index=False,
        )
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        output_dir / "same_row_predictions.csv", index=False
    )
    pd.DataFrame(population_audits).to_csv(
        output_dir / "same_row_population_audit.csv", index=False
    )
    pd.concat(state_audits, ignore_index=True).to_csv(
        output_dir / "state_mapping_application_audit.csv", index=False
    )
    pd.DataFrame(bundle_identity_rows).to_csv(
        output_dir / "bundle_identity.csv", index=False
    )
    pd.DataFrame(runtime_version_rows).to_csv(
        output_dir / "runtime_version_validation.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "tenure": tenure,
                **{
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in audit.items()
                },
            }
            for tenure, audit in sorted(native_snapshot_audits.items())
        ]
    ).to_csv(output_dir / "native_snapshot_audit.csv", index=False)
    pd.DataFrame(chart_rows).to_csv(
        output_dir / "chart_generation_status.csv", index=False
    )


    family_decisions = _family_pooling_decisions(
        comparison,
        margin_contract,
        canonical_groups,
        integrity_bypass_requested=integrity_bypass_requested,
    )


    pd.DataFrame(
        [
            {
                "model_family": family,
                "tenure_groups": ",".join(canonical_groups),
                "decision": decision,
                "required_predeclared_gate_count": sum(
                    1 for key in margin_contract if key[0] == family
                )
                * len(canonical_groups),
                "integrity_bypass_requested": integrity_bypass_requested,
                "interpretation": (
                    "Descriptive same-row comparison only; no predeclared pooling "
                    "margin is available, so this run cannot approve pooling."
                ),
            }
            for family, decision in family_decisions.items()
        ]
    ).to_csv(output_dir / "family_decision_summary.csv", index=False)


    inventory = _output_inventory(output_dir)
    inventory_path = output_dir / "output_inventory.csv"
    inventory.to_csv(inventory_path, index=False)


    _write_json(
        output_dir / "evaluation_manifest.json",
        {
            "evaluation_contract": EVALUATION_CONTRACT,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_versions": _runtime_versions(),
            "bundle_runtime_version_validation": {
                "policy": "STRICT_EXACT_MATCH",
                "status": "PASS",
                "override_available": False,
                "audit_csv": output_dir / "runtime_version_validation.csv",
                "checks": runtime_version_rows,
            },
            "pooled_run": {
                "path": pooled_run.path,
                "profile": pooled_run.profile,
                "profile_hash": pooled_run.profile_hash,
            },
            "native_runs": {
                tenure: {
                    "path": run.path,
                    "profile": run.profile,
                    "profile_hash": run.profile_hash,
                }
                for tenure, run in native_runs.items()
            },
            "tenure_groups": list(canonical_groups),
            "target_column": target_column,
            "tenure_column": tenure_column,
            "identity_columns": list(identity_columns),
            "policy_cluster_column": policy_cluster_column,
            "pooled_target_data": pooled_data_audit,
            "pooled_bundle_snapshot_sha256": (
                pooled_context.source_snapshot_sha256
            ),
            "pooled_bundle_snapshot_match": pooled_bundle_snapshot_match,
            "pooled_holdout": dict(pooled_context.holdout),
            "pooled_holdout_observed_population_sha256": observed_pooled_hash,
            "evaluation_month_coverage": month_coverage,
            "same_row_population_audit": population_audits,
            "native_target_data": native_snapshot_audits,
            "bundle_identities": bundle_identity_rows,
            "lift_bins": list(args.lift_bins),
            "lift_definition": "group churn rate divided by tenure-population churn rate",
            "lift_spread_definitions": {
                "highest_divided_by_lowest_lift": "maximum group lift divided by minimum group lift",
                "ordered_first_to_last_lift_ratio": "highest-score group lift divided by lowest-score group lift",
            },
            "lift_spread_comparison": {
                "point_estimates": "lift_spread_comparison.csv",
                "paired_bootstrap_replicates": (
                    "paired_lift_spread_bootstrap_replicates.csv"
                    if not spread_bootstrap.empty
                    else None
                ),
                "decision_rule": (
                    "reported with paired confidence intervals; no post-hoc "
                    "noninferiority margin is imposed"
                ),
            },
            "bootstrap": {
                "method": "paired policy-cluster Poisson bootstrap",
                "replicates": args.bootstrap_reps,
                "seed": args.seed,
                "cluster_column_source": policy_cluster_column,
                "top10_membership": "fixed separately from each model's original score ranking",
            },
            "pooling_gate_contract": {
                "status": "NO_PREDECLARED_POOLING_MARGIN",
                "margins": {
                    f"{family}.{metric}": value
                    for (family, metric), value in margin_contract.items()
                },
                "family_decisions": family_decisions,
                "decision_note": (
                    "Results are descriptive only. Pipeline model-selection "
                    "tolerances are not reused as pooled-versus-native acceptance "
                    "margins, so this run cannot approve combining tenure models."
                ),
                "probability_metric_note": (
                    "No saved deployment noninferiority margins exist for log loss, "
                    "Brier score, or absolute calibration gap; those metrics are "
                    "reported but not converted to a post-hoc pass/fail rule."
                ),
            },
            "charts": chart_rows,
            "bundle_hash_verification": verify_hashes,
            "integrity_bypass_requested": integrity_bypass_requested,
            "native_snapshot_payload_audit_skipped": bool(
                args.skip_native_snapshot_payload_audit
            ),
            "models_fitted": 0,
            "target_learning_operations": 0,
            "output_inventory": {
                "path": str(inventory_path),
                "sha256": cross.sha256_file(inventory_path),
                "file_count_excluding_manifest_and_inventory": int(len(inventory)),
            },
            "code": {
                "path": str(Path(__file__).resolve()),
                "sha256": cross.sha256_file(Path(__file__).resolve()),
                "cross_tenure_runtime_path": str(Path(cross.__file__).resolve()),
                "cross_tenure_runtime_sha256": cross.sha256_file(
                    Path(cross.__file__).resolve()
                ),
            },
        },
    )
    print(f"Pooled-versus-native evaluation complete: {output_dir}")
    print(f"Family ranking decisions: {family_decisions}")
    return 0




if __name__ == "__main__":
    raise SystemExit(main())
