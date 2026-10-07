#!/usr/bin/env python
"""Create score-only cross-tenure evidence for tenure-grouping decisions.




The governed default consumes c4 outer-evaluation manifests, applies each
source tenure's frozen outer-fold Logistic/XGBoost bundles to the matching
target-tenure validation window, and compares those scores with target-native
outer-OOF predictions on the exact same rows.  Inner-fold metrics and the final
holdout are never used for tenure-grouping selection.




No estimator, encoder, imputer, scaler, state grouper, binning rule, or other
learned object is fitted here. Target outcomes are attached only after frozen
source models return probabilities. The former final-bundle evaluator remains
available only through an explicit, non-selection diagnostic purpose.
"""




from __future__ import annotations




import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence




import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)








FAMILY_CONTRACT = "retention_frozen_source_model_bundle"
PAIR_CONTRACT = "retention_source_model_pair"
OUTER_EVALUATION_CONTRACT = "retention_outer_evaluation"
SUPPORTED_OUTER_EVALUATION_CONTRACT_REVISION = 1
COMPONENT_ROLE = "frozen_source_evaluator"
MODEL_FAMILIES = ("logistic", "xgboost")
DEFAULT_LIFT_BINS = (3, 4, 5, 10, 20)
CORE_SCORING_RUNTIME_VERSION_KEYS = ("numpy", "pandas", "scikit_learn")
TENURE_GROUPING_PURPOSE = "TENURE_GROUPING_MODEL_STRUCTURE"
NON_SELECTION_DIAGNOSTIC_PURPOSE = "NON_SELECTION_DIAGNOSTIC"








@dataclass(frozen=True)
class OuterFoldBundle:
    fold: int
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    label_cutoff: pd.Timestamp
    families: Mapping[str, ManifestRecord]








@dataclass(frozen=True)
class OuterEvaluationRecord:
    path: Path
    payload: Mapping[str, Any]
    folds: tuple[OuterFoldBundle, ...]
    oof_predictions_path: Path
    identity_columns: tuple[str, ...]
    target_column: str
    cohort_column: str








@dataclass(frozen=True)
class ManifestRecord:
    path: Path
    payload: Mapping[str, Any]




    @property
    def root(self) -> Path:
        return self.path.parent




    @property
    def family(self) -> str:
        return str(self.payload.get("model_family", "")).strip().lower()








@dataclass(frozen=True)
class TargetRun:
    path: Path
    profile: Mapping[str, Any]
    profile_hash: str








@dataclass(frozen=True)
class SharedStateContract:
    mapping: Mapping[str, str]
    categories: tuple[str, ...]
    mapping_sha256: str
    derived_feature: str
    source_feature: str








def _json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload








def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else str(number)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value








def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)








def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()








def _installed_scoring_runtime_versions(family: str) -> dict[str, str]:
    """Read scoring-package versions from the active interpreter."""




    observed = {
        "numpy": str(np.__version__),
        "pandas": str(pd.__version__),
        "scikit_learn": str(sklearn.__version__),
    }
    if family == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Frozen XGBoost scoring requires xgboost, but it is not installed."
            ) from exc
        observed["xgboost"] = str(xgb.__version__)
    return observed








def _validate_bundle_runtime_versions(
    record: ManifestRecord,
) -> list[dict[str, Any]]:
    """Fail closed unless a bundle's recorded scoring versions match exactly."""




    family = record.family
    if family not in MODEL_FAMILIES:
        raise ValueError(
            f"Cannot validate runtime versions for model family {family!r}: "
            f"{record.path}"
        )
    recorded = record.payload.get("runtime_versions")
    if not isinstance(recorded, Mapping):
        raise ValueError(
            f"Frozen {family} bundle has no recorded runtime_versions contract: "
            f"{record.path}"
        )
    required = [*CORE_SCORING_RUNTIME_VERSION_KEYS]
    if family == "xgboost":
        required.append("xgboost")
    missing = [key for key in required if not str(recorded.get(key, "")).strip()]
    if missing:
        raise ValueError(
            f"Frozen {family} bundle runtime_versions is missing required package "
            f"versions {missing}: {record.path}"
        )
    installed = _installed_scoring_runtime_versions(family)
    audit: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for package in required:
        expected = str(recorded[package]).strip()
        observed = str(installed[package]).strip()
        matches = expected == observed
        audit.append(
            {
                "model_family": family,
                "package": package,
                "recorded_version": expected,
                "installed_version": observed,
                "exact_match": matches,
            }
        )
        if not matches:
            mismatches.append(
                f"{package}: recorded={expected!r}, installed={observed!r}"
            )
    if mismatches:
        raise RuntimeError(
            f"Frozen {family} bundle runtime-version mismatch at {record.path}; "
            "scoring is refused: " + "; ".join(mismatches)
        )
    return audit








def sha256_json(payload: Any) -> str:
    rendered = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()








def _normalize_tenure(value: Any) -> str:
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text








def _selected_groups(profile: Mapping[str, Any]) -> tuple[str, ...]:
    raw = profile.get("selected_tenure_groups", [])
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    return tuple(_normalize_tenure(value) for value in raw if str(value).strip())








def _profile_candidates(payload: Any) -> Iterable[tuple[Mapping[str, Any], str]]:
    if isinstance(payload, Mapping):
        profile = payload.get("profile")
        if isinstance(profile, Mapping) and _selected_groups(profile):
            yield profile, str(payload.get("profile_hash", ""))
        source_profile = payload.get("source_profile")
        if isinstance(source_profile, Mapping) and _selected_groups(source_profile):
            yield source_profile, str(
                payload.get("source_profile_hash", payload.get("profile_hash", ""))
            )
        for value in payload.values():
            yield from _profile_candidates(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _profile_candidates(value)








def _run_profile(run: Path) -> tuple[dict[str, Any], str] | None:
    matches: list[tuple[dict[str, Any], str]] = []
    for path in sorted(run.rglob("*.json")):
        try:
            payload = _json_load(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        for profile, profile_hash in _profile_candidates(payload):
            matches.append((dict(profile), profile_hash))
    if not matches:
        return None
    groups = {tuple(_selected_groups(profile)) for profile, _ in matches}
    if len(groups) > 1:
        raise RuntimeError(f"Run contains conflicting tenure profiles: {run}")
    profile_hashes = {value for _, value in matches if value}
    if len(profile_hashes) > 1:
        raise RuntimeError(f"Run contains conflicting profile hashes: {run}")
    for left_index, (left, _) in enumerate(matches):
        for right, _ in matches[left_index + 1 :]:
            if not _profiles_consistent(left, right):
                raise RuntimeError(
                    f"Run contains profiles that agree on tenure but conflict on "
                    f"shared fields: {run}"
                )
    richest = max(matches, key=lambda item: _profile_richness(item[0]))[0]
    return dict(richest), next(iter(profile_hashes), "")








def _profiles_consistent(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return all(
            _profiles_consistent(left[key], right[key])
            for key in set(left).intersection(right)
        )
    return json.dumps(left, sort_keys=True, default=str) == json.dumps(
        right, sort_keys=True, default=str
    )








def _profile_richness(profile: Mapping[str, Any]) -> tuple[int, int, int]:
    def count(value: Any) -> int:
        if isinstance(value, Mapping):
            return len(value) + sum(count(item) for item in value.values())
        if isinstance(value, list):
            return len(value) + sum(count(item) for item in value)
        return 1




    tenure_dimension = str(profile.get("tenure_dimension", "")).strip()
    return (
        int(bool(tenure_dimension)),
        count(profile),
        len(json.dumps(profile, sort_keys=True, default=str)),
    )








def _pipeline_run_dirs(project_root: Path) -> list[Path]:
    root = project_root / "output" / "pipeline_runs"
    if not root.is_dir():
        raise FileNotFoundError(f"Pipeline-run directory does not exist: {root}")
    return sorted(path for path in root.iterdir() if path.is_dir())








def _resolve_project_or_run_path(project_root: Path, value: Path) -> Path:
    expanded = value.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    candidates = (
        project_root / expanded,
        project_root / "output" / "pipeline_runs" / expanded,
    )
    existing = [candidate.resolve() for candidate in candidates if candidate.exists()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise RuntimeError(f"Relative path is ambiguous: {value}; matches={existing}")
    return candidates[0].resolve()








def _complete_full_run_timestamp(
    run: Path, profile: Mapping[str, Any], profile_hash: str
) -> pd.Timestamp:
    expected_groups = _selected_groups(profile)
    candidates: list[pd.Timestamp] = []
    for path in sorted(run.rglob("*.json")):
        try:
            payload = _json_load(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if str(payload.get("run_id", "")).strip() != run.name:
            continue
        if str(payload.get("status", "")).strip().upper() != "COMPLETE":
            continue
        if str(payload.get("population_scope_mode", "")).strip().upper() != "FULL":
            continue
        groups = payload.get("selected_tenure_groups")
        if not isinstance(groups, list):
            continue
        if tuple(_normalize_tenure(value) for value in groups) != expected_groups:
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








def resolve_target_run(
    project_root: Path,
    target_tenure: str,
    explicit: Path | None = None,
) -> TargetRun:
    requested = _normalize_tenure(target_tenure)
    if explicit is not None:
        run = _resolve_project_or_run_path(project_root, explicit)
        candidates = [run]
    else:
        candidates = _pipeline_run_dirs(project_root)
    eligible: list[tuple[TargetRun, pd.Timestamp]] = []
    for run in candidates:
        if not run.is_dir():
            continue
        found = _run_profile(run)
        if found is None:
            continue
        profile, profile_hash = found
        if _selected_groups(profile) == (requested,):
            record = TargetRun(run, profile, profile_hash)
            try:
                completed = _complete_full_run_timestamp(
                    run, profile, profile_hash
                )
            except ValueError:
                continue
            eligible.append((record, completed))
    if not eligible:
        prefix = "Explicit target run is not" if explicit is not None else "No target run is"
        raise FileNotFoundError(
            f"{prefix} a metadata-confirmed single-tenure run for {requested!r}."
        )
    latest = max(timestamp for _, timestamp in eligible)
    winners = [record for record, timestamp in eligible if timestamp == latest]
    if len(winners) != 1:
        raise RuntimeError(
            "Multiple target runs share the latest COMPLETE/FULL metadata "
            "completion timestamp; pass an explicit run. candidates="
            f"{[str(record.path) for record in winners]}"
        )
    return winners[0]








def _walk_json_files(search_roots: Sequence[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in search_roots:
        candidate = root.expanduser().resolve()
        if candidate.is_file():
            paths = [candidate]
        elif candidate.is_dir():
            paths = candidate.rglob("*.json")
        else:
            continue
        for path in paths:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved








def _manifest_source_metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = payload.get("source_metadata")
    return metadata if isinstance(metadata, Mapping) else payload








def _manifest_bundle_role(payload: Mapping[str, Any]) -> str:
    """Return the governed role without treating outer-fold models as final ones."""




    metadata = _manifest_source_metadata(payload)
    return str(
        metadata.get("bundle_role", payload.get("bundle_role", ""))
    ).strip().upper()








def _legacy_final_bundle_is_eligible(payload: Mapping[str, Any]) -> bool:
    """Allow pre-role legacy bundles and explicit final refits only."""




    return _manifest_bundle_role(payload) in {"", "FINAL_REFIT"}








def _manifest_source_tenure(payload: Mapping[str, Any]) -> str:
    metadata = _manifest_source_metadata(payload)
    for key in ("source_tenure", "tenure"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return _normalize_tenure(value)
    for profile_key in ("profile", "source_profile"):
        profile = metadata.get(profile_key)
        if isinstance(profile, Mapping):
            groups = _selected_groups(profile)
            if len(groups) == 1:
                return groups[0]
    groups = metadata.get("selected_tenure_groups")
    if isinstance(groups, (list, tuple)) and len(groups) == 1:
        return _normalize_tenure(groups[0])
    return ""








def _manifest_identity(payload: Mapping[str, Any]) -> dict[str, str]:
    metadata = _manifest_source_metadata(payload)
    profile = metadata.get("profile")
    if not isinstance(profile, Mapping):
        profile = metadata.get("source_profile")
    profile_hash = str(
        metadata.get("source_profile_hash", metadata.get("profile_hash", ""))
    )
    run_id = str(metadata.get("source_run_id", metadata.get("run_id", "")))
    snapshot = metadata.get("source_data_identity")
    if not isinstance(snapshot, Mapping):
        snapshot = metadata.get("source_snapshot")
    if not isinstance(snapshot, Mapping):
        snapshot = metadata.get("data")
    snapshot_sha = str(snapshot.get("sha256", "")) if isinstance(snapshot, Mapping) else str(
        metadata.get("source_snapshot_sha256", "")
    )
    excluded_holdout = payload.get("excluded_holdout")
    if not isinstance(excluded_holdout, Mapping):
        excluded_holdout = metadata.get("excluded_holdout")
    if not isinstance(excluded_holdout, Mapping):
        excluded_holdout = metadata.get("excluded_holdout_identity")
    holdout_sha = ""
    if isinstance(excluded_holdout, Mapping):
        holdout_sha = str(
            excluded_holdout.get(
                "population_sha256",
                excluded_holdout.get(
                    "sha256", excluded_holdout.get("definition_sha256", "")
                ),
            )
        )
    evaluation_training = payload.get("evaluation_training_population")
    if not isinstance(evaluation_training, Mapping):
        evaluation_training = metadata.get("evaluation_training_population")
    shared_state = payload.get("shared_state_grouping")
    if not isinstance(shared_state, Mapping):
        shared_state = metadata.get("shared_state_grouping")
    shared_state_mapping_sha256 = (
        str(shared_state.get("mapping_sha256", ""))
        if isinstance(shared_state, Mapping)
        else ""
    )
    return {
        "pairing_identity": str(
            payload.get("pairing_identity", metadata.get("pairing_identity", ""))
        ),
        "source_tenure": _manifest_source_tenure(payload),
        "source_run_id": run_id,
        "source_profile_hash": profile_hash,
        "cohort_column": str(
            payload.get("cohort_column", metadata.get("cohort_column", ""))
        ),
        "source_snapshot_sha256": snapshot_sha,
        "excluded_holdout_identity": holdout_sha,
        "evaluation_training_population_sha256": hashlib.sha256(
            json.dumps(
                evaluation_training,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        if isinstance(evaluation_training, Mapping)
        else "",
        "shared_state_mapping_sha256": shared_state_mapping_sha256,
        "source_profile": json.dumps(profile, sort_keys=True, default=str)
        if isinstance(profile, Mapping)
        else "",
    }








def _resolve_manifest_reference(pair_path: Path, reference: Any) -> Path | None:
    if isinstance(reference, str) and reference.strip():
        raw = Path(reference).expanduser()
        path = raw if raw.is_absolute() else pair_path.parent / raw
        return path.resolve()
    if isinstance(reference, Mapping):
        for key in ("manifest_relative_path", "manifest_path", "path", "relative_path"):
            raw_value = reference.get(key)
            if raw_value is not None and str(raw_value).strip():
                raw = Path(str(raw_value)).expanduser()
                path = raw if raw.is_absolute() else pair_path.parent / raw
                return path.resolve()
    return None








def _family_references(pair: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
    for container_key in (
        "families",
        "family_manifests",
        "models",
        "bundles",
        "model_bundles",
    ):
        container = pair.get(container_key)
        if isinstance(container, Mapping):
            for family, reference in container.items():
                normalized = str(family).strip().lower()
                if normalized in MODEL_FAMILIES:
                    yield normalized, reference
        elif isinstance(container, list):
            for item in container:
                if not isinstance(item, Mapping):
                    continue
                family = str(item.get("model_family", item.get("family", ""))).lower()
                if family in MODEL_FAMILIES:
                    yield family, item








def _records_from_pair(pair_record: ManifestRecord) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    for expected_family, reference in _family_references(pair_record.payload):
        path = _resolve_manifest_reference(pair_record.path, reference)
        if path is None or not path.is_file():
            raise FileNotFoundError(
                f"Pair manifest references a missing {expected_family} manifest: {reference!r}"
            )
        payload = _json_load(path)
        if str(payload.get("contract", "")) != FAMILY_CONTRACT:
            raise ValueError(f"Referenced family manifest has an invalid contract: {path}")
        observed_family = str(payload.get("model_family", "")).strip().lower()
        if observed_family != expected_family:
            raise ValueError(
                f"Pair manifest labels {path} as {expected_family}, found {observed_family}."
            )
        records.append(ManifestRecord(path, payload))
    observed_families = [record.family for record in records]
    if len(observed_families) != len(set(observed_families)):
        raise ValueError(
            f"Pair manifest references a model family more than once; "
            f"observed={observed_families}"
        )
    pair_identity = _manifest_identity(pair_record.payload)
    for record in records:
        family_identity = _manifest_identity(record.payload)
        for field in (
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
        ):
            pair_value = pair_identity[field]
            family_value = family_identity[field]
            if pair_value and family_value and pair_value != family_value:
                raise RuntimeError(
                    f"Pair and {record.family} manifests disagree on {field}: "
                    f"{pair_value!r} != {family_value!r}"
                )
    return records








def discover_source_bundle_manifests(
    project_root: Path,
    source_tenure: str,
    *,
    explicit_manifests: Sequence[Path] = (),
    explicit_source_run: Path | None = None,
) -> dict[str, ManifestRecord]:
    requested = _normalize_tenure(source_tenure)
    registry_root = project_root / "output" / "frozen_source_models"
    expected_source_run_id: str | None = None
    if explicit_manifests:
        roots = [Path(value) for value in explicit_manifests]
    elif explicit_source_run is not None:
        source_run = _resolve_project_or_run_path(project_root, explicit_source_run)
        expected_source_run_id = source_run.name
        # New bundles live in the short registry to stay below Windows MAX_PATH;
        # retain the exact run as a second root for legacy in-run bundles.
        roots = [registry_root, source_run]
    else:
        # The second root keeps previously materialized, in-run bundles usable.
        roots = [registry_root, project_root / "output" / "pipeline_runs"]




    direct_records: list[ManifestRecord] = []
    pair_records: list[ManifestRecord] = []
    for path in _walk_json_files(roots):
        try:
            payload = _json_load(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        contract = str(payload.get("contract", ""))
        if contract == FAMILY_CONTRACT:
            if _legacy_final_bundle_is_eligible(payload):
                direct_records.append(ManifestRecord(path, payload))
        elif contract == PAIR_CONTRACT and _legacy_final_bundle_is_eligible(payload):
            pair_records.append(ManifestRecord(path, payload))




    candidate_groups: list[dict[str, ManifestRecord]] = []
    for pair in pair_records:
        records = _records_from_pair(pair)
        if not all(
            _legacy_final_bundle_is_eligible(record.payload) for record in records
        ):
            continue
        grouped = {record.family: record for record in records}
        if set(grouped) == set(MODEL_FAMILIES):
            candidate_groups.append(grouped)




    grouped_direct: dict[tuple[str, ...], dict[str, ManifestRecord]] = {}
    for record in direct_records:
        if record.family not in MODEL_FAMILIES:
            continue
        identity = _manifest_identity(record.payload)
        key = tuple(identity[name] for name in sorted(identity))
        grouped_direct.setdefault(key, {})[record.family] = record
    candidate_groups.extend(
        group for group in grouped_direct.values() if set(group) == set(MODEL_FAMILIES)
    )




    eligible: list[dict[str, ManifestRecord]] = []
    for group in candidate_groups:
        identities = [_manifest_identity(group[family].payload) for family in MODEL_FAMILIES]
        tenures = {identity["source_tenure"] for identity in identities}
        if tenures != {requested}:
            continue
        source_run_ids = {
            identity["source_run_id"] for identity in identities
            if identity["source_run_id"]
        }
        if (
            expected_source_run_id is not None
            and source_run_ids != {expected_source_run_id}
        ):
            continue
        for field in (
            "pairing_identity",
            "source_run_id",
            "source_profile_hash",
            "cohort_column",
            "source_snapshot_sha256",
            "excluded_holdout_identity",
            "evaluation_training_population_sha256",
            "shared_state_mapping_sha256",
            "source_profile",
        ):
            values = {identity[field] for identity in identities if identity[field]}
            if len(values) > 1:
                raise RuntimeError(
                    f"Source Logistic/XGBoost manifests disagree on {field}: {sorted(values)}"
                )
        eligible.append(group)




    if not eligible:
        available_final_families = sorted(
            {
                record.family
                for record in direct_records
                if record.family in MODEL_FAMILIES
                and _manifest_source_tenure(record.payload) == requested
                and (
                    expected_source_run_id is None
                    or _manifest_identity(record.payload)["source_run_id"]
                    == expected_source_run_id
                )
            }
        )
        if available_final_families and set(available_final_families) != set(
            MODEL_FAMILIES
        ):
            raise FileNotFoundError(
                "Only a champion-only FINAL_REFIT source bundle is available "
                f"for tenure {requested!r} (families={available_final_families}). "
                "The legacy two-family diagnostic is unavailable. Use --purpose "
                "TENURE_GROUPING_MODEL_STRUCTURE with c4 outer-fold bundles for "
                "governed tenure-grouping evidence."
            )
        raise FileNotFoundError(
            f"No complete, identity-matched Logistic/XGBoost frozen bundle pair was "
            f"found for source tenure {requested!r}. Outer-fold evaluation bundles "
            "are deliberately excluded from legacy discovery; use --purpose "
            "TENURE_GROUPING_MODEL_STRUCTURE for governed grouping evidence."
        )
    # A pair manifest and its referenced family manifests can describe the same
    # physical bundle twice. Collapse only those exact path duplicates. If more
    # than one distinct bundle remains for the resolved source run, fail closed
    # instead of treating filesystem modification time as a model version.
    unique: dict[tuple[str, ...], dict[str, ManifestRecord]] = {}
    for group in eligible:
        key = tuple(
            str(group[family].path.resolve()) for family in MODEL_FAMILIES
        )
        unique[key] = group
    if len(unique) != 1:
        candidates = [list(key) for key in sorted(unique)]
        raise RuntimeError(
            "Multiple distinct frozen bundle pairs match the requested source "
            "run and tenure scope. Pass explicit family manifests; filesystem "
            f"timestamps are not used for model identity. candidates={candidates}"
        )
    return next(iter(unique.values()))








def _runtime_source_path(record: ManifestRecord) -> Path:
    sources = record.payload.get("runtime_sources")
    candidates: list[Any] = []
    if isinstance(sources, Mapping):
        candidates.extend(sources.values())
    elif isinstance(sources, list):
        candidates.extend(sources)
    for item in candidates:
        if isinstance(item, Mapping):
            raw = item.get("relative_path", item.get("path"))
        else:
            raw = item
        if raw is None or not str(raw).strip():
            continue
        path_value = Path(str(raw)).expanduser()
        path = path_value if path_value.is_absolute() else record.root / path_value
        if path.is_file() and path.suffix.lower() == ".py":
            try:
                source = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if "score_logistic_bundle" in source and "score_xgboost_bundle" in source:
                return path.resolve()
    raise FileNotFoundError(
        f"Bundle manifest does not identify its packaged score-only runtime: {record.path}"
    )








def _import_runtime(record: ManifestRecord) -> Any:
    runtime_path = _runtime_source_path(record)
    module_name = f"frozen_bundle_runtime_{sha256_file(runtime_path)[:16]}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, runtime_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import packaged bundle runtime: {runtime_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module








def _verify_manifest_inventory(record: ManifestRecord) -> None:
    inventory = record.payload.get("artifact_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError(f"Bundle manifest has no artifact inventory: {record.path}")
    for item in inventory:
        if not isinstance(item, Mapping):
            raise TypeError(f"Invalid artifact inventory row in {record.path}")
        raw = item.get("relative_path", item.get("path"))
        expected = str(item.get("sha256", ""))
        if raw is None or not str(raw).strip() or not expected:
            raise ValueError(f"Incomplete artifact inventory row in {record.path}: {item}")
        candidate = Path(str(raw)).expanduser()
        path = candidate if candidate.is_absolute() else record.root / candidate
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen bundle artifact hash mismatch: {path}")








def score_target_with_frozen_bundle(
    bundle_manifest_path: str | Path,
    target_frame: pd.DataFrame,
    *,
    verify_hashes: bool = True,
) -> np.ndarray:
    """Return retention probabilities without learning from ``target_frame``."""




    path = Path(bundle_manifest_path).expanduser().resolve()
    record = ManifestRecord(path, _json_load(path))
    if str(record.payload.get("contract", "")) != FAMILY_CONTRACT:
        raise ValueError(f"Unsupported source bundle contract: {path}")
    _validate_bundle_runtime_versions(record)
    if verify_hashes:
        _verify_manifest_inventory(record)
    runtime = _import_runtime(record)
    target_column = str(record.payload.get("target_column", "")).strip()
    features = {str(value) for value in record.payload.get("features", [])}
    if target_column and target_column in features:
        raise ValueError("Frozen bundle illegally includes its outcome as a predictor")
    score_input = (
        target_frame.drop(columns=[target_column])
        if target_column and target_column in target_frame
        else target_frame
    )
    if record.family == "logistic":
        prediction = runtime.score_logistic_bundle(
            record.root, score_input, verify_hashes=verify_hashes
        )
    elif record.family == "xgboost":
        prediction = runtime.score_xgboost_bundle(
            record.root, score_input, verify_hashes=verify_hashes
        )
    else:
        raise ValueError(f"Unsupported source model family: {record.family!r}")
    values = np.asarray(prediction, dtype=float).reshape(-1)
    if values.shape != (len(target_frame),) or not np.isfinite(values).all():
        raise RuntimeError(
            f"Frozen {record.family} scorer returned invalid prediction shape/values."
        )
    if np.any((values < 0.0) | (values > 1.0)):
        observed_min = float(values.min())
        observed_max = float(values.max())
        raise RuntimeError(
            f"Frozen {record.family} scorer returned values outside [0, 1]: "
            f"minimum={observed_min}; maximum={observed_max}."
        )
    prediction_target = str(record.payload.get("prediction_target", "")).lower()
    if "churn" in prediction_target:
        values = 1.0 - values
    elif "retention" not in prediction_target:
        raise ValueError(
            f"Bundle prediction_target must name churn or retention: {prediction_target!r}"
        )
    return np.clip(values, 1e-7, 1.0 - 1e-7)








def _identity_dicts(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if value.get("path") is not None and value.get("sha256") is not None:
            yield value
        for nested in value.values():
            yield from _identity_dicts(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _identity_dicts(nested)








def _resolve_recorded_path(run: Path, identity: Mapping[str, Any]) -> Path | None:
    raw = str(identity.get("path", "")).strip()
    if not raw:
        return None
    direct = Path(raw).expanduser()
    if direct.is_file():
        return direct.resolve()
    normalized = [part for part in raw.replace("\\", "/").split("/") if part]
    if run.name in normalized:
        position = len(normalized) - 1 - normalized[::-1].index(run.name)
        candidate = run.joinpath(*normalized[position + 1 :])
        if candidate.is_file():
            return candidate.resolve()
    basename = PureWindowsPath(raw).name or Path(raw).name
    matches = sorted(path for path in run.rglob(basename) if path.is_file())
    return matches[0].resolve() if len(matches) == 1 else None








def _state_contract_for_bundle(
    record: ManifestRecord,
    runtime: Any,
    preprocessor: Mapping[str, Any],
) -> SharedStateContract:
    state_contract = preprocessor.get("state_contract")
    if not isinstance(state_contract, Mapping):
        raise RuntimeError(
            f"Frozen {record.family} bundle has no shared source-tenure state "
            "contract. Rebuild the source-model pair with the current builder."
        )
    mapping = {
        str(key).strip().upper(): str(value).strip()
        for key, value in dict(state_contract.get("mapping", {})).items()
    }
    categories = tuple(
        str(value).strip() for value in state_contract.get("categories", [])
    )
    if not mapping or not categories or len(categories) != len(set(categories)):
        raise RuntimeError(
            f"Frozen {record.family} shared state contract is empty or invalid. "
            "Rebuild the source-model pair."
        )
    outside = sorted(set(mapping.values()).difference(categories))
    if outside:
        raise RuntimeError(
            f"Frozen {record.family} state mapping uses undeclared groups {outside}. "
            "Rebuild the source-model pair."
        )
    computed_sha = sha256_json(mapping)
    if str(state_contract.get("mapping_sha256", "")) != computed_sha:
        raise RuntimeError(
            f"Frozen {record.family} state mapping hash is invalid. Rebuild the "
            "source-model pair."
        )
    manifest_shared = record.payload.get("shared_state_grouping")
    if not isinstance(manifest_shared, Mapping):
        manifest_shared = _manifest_source_metadata(record.payload).get(
            "shared_state_grouping"
        )
    if not isinstance(manifest_shared, Mapping):
        raise RuntimeError(
            f"Frozen {record.family} manifest has no pair-level shared state "
            "identity. Rebuild the source-model pair."
        )
    if str(manifest_shared.get("mapping_sha256", "")) != computed_sha:
        raise RuntimeError(
            f"Frozen {record.family} manifest and preprocessor disagree on the "
            "shared state mapping. Rebuild the source-model pair."
        )
    manifest_mapping = {
        str(key).strip().upper(): str(value).strip()
        for key, value in dict(manifest_shared.get("mapping", {})).items()
    }
    manifest_categories = tuple(
        str(value).strip() for value in manifest_shared.get("categories", [])
    )
    if manifest_mapping != mapping or manifest_categories != categories:
        raise RuntimeError(
            f"Frozen {record.family} manifest contains a different state mapping or "
            "category order. Rebuild the source-model pair."
        )
    derived_feature, source_feature = _derived_feature_names(
        preprocessor,
        runtime,
        contract_key="state_group",
        legacy_derived_attribute="STATE_GROUP_FEATURE",
        legacy_source_attribute="STATE_SOURCE_FEATURE",
    )
    if (
        str(manifest_shared.get("derived_feature", derived_feature)).strip()
        != derived_feature
        or str(manifest_shared.get("source_feature", source_feature)).strip()
        != source_feature
    ):
        raise RuntimeError(
            f"Frozen {record.family} manifest disagrees on the state feature contract. "
            "Rebuild the source-model pair."
        )
    return SharedStateContract(
        mapping=mapping,
        categories=categories,
        mapping_sha256=computed_sha,
        derived_feature=derived_feature,
        source_feature=source_feature,
    )








def _validate_shared_state_contract(
    loaded: Sequence[tuple[ManifestRecord, Any, Mapping[str, Any]]],
) -> SharedStateContract:
    contracts = [
        _state_contract_for_bundle(record, runtime, preprocessor)
        for record, runtime, preprocessor in loaded
    ]
    if len(contracts) != len(MODEL_FAMILIES):
        raise RuntimeError("Both Logistic and XGBoost bundles are required.")
    reference = contracts[0]
    for observed in contracts[1:]:
        if observed != reference:
            raise RuntimeError(
                "Logistic and XGBoost do not use the identical source-tenure state "
                "mapping, category order, and state feature. Rebuild the source-model "
                "pair with the current builder."
            )
    return reference








def _required_scoring_columns(
    bundles: Mapping[str, ManifestRecord],
) -> tuple[
    set[str],
    str,
    list[tuple[ManifestRecord, Any, Mapping[str, Any]]],
    SharedStateContract,
]:
    required: set[str] = set()
    target_columns: set[str] = set()
    loaded: list[tuple[ManifestRecord, Any, Mapping[str, Any]]] = []
    for family in MODEL_FAMILIES:
        record = bundles[family]
        _validate_bundle_runtime_versions(record)
        runtime = _import_runtime(record)
        manifest, preprocessor = runtime.load_bundle(record.root, verify_hashes=False)
        loaded.append((record, runtime, preprocessor))
        target_columns.add(str(manifest.get("target_column", "")).strip())
        features = [str(value) for value in preprocessor["features"]]
        written_feature, written_source = _derived_feature_names(
            preprocessor,
            runtime,
            contract_key="written_premium",
            legacy_derived_attribute="WRITTEN_PREM_BIN_FEATURE",
            legacy_source_attribute="WRITTEN_PREM_SOURCE_FEATURE",
        )
        state_feature, state_source = _derived_feature_names(
            preprocessor,
            runtime,
            contract_key="state_group",
            legacy_derived_attribute="STATE_GROUP_FEATURE",
            legacy_source_attribute="STATE_SOURCE_FEATURE",
        )
        cohort_year_feature = ""
        cohort_year_source = ""
        derived_contracts = preprocessor.get("derived_feature_contracts")
        if isinstance(derived_contracts, Mapping):
            cohort_year = derived_contracts.get("cohort_calendar_year")
            if isinstance(cohort_year, Mapping):
                cohort_year_feature = str(
                    cohort_year.get("derived_feature", "")
                ).strip()
                cohort_year_source = str(
                    cohort_year.get("source_feature", "")
                ).strip()
        for feature in features:
            if feature == written_feature and written_source:
                required.add(written_source)
            elif feature == state_feature and state_source:
                required.add(state_source)
            elif feature == cohort_year_feature and cohort_year_source:
                required.add(cohort_year_source)
            else:
                required.add(feature)
    shared_state = _validate_shared_state_contract(loaded)
    required.add(shared_state.source_feature)
    target_columns.discard("")
    if len(target_columns) != 1:
        raise RuntimeError(
            f"Source Logistic/XGBoost bundles disagree on target column: {target_columns}"
        )
    return required, next(iter(target_columns)), loaded, shared_state








def _derived_feature_names(
    preprocessor: Mapping[str, Any],
    runtime: Any,
    *,
    contract_key: str,
    legacy_derived_attribute: str,
    legacy_source_attribute: str,
) -> tuple[str, str]:
    contracts = preprocessor.get("derived_feature_contracts")
    if isinstance(contracts, Mapping) and contract_key in contracts:
        contract = contracts[contract_key]
        if not isinstance(contract, Mapping):
            raise TypeError(
                f"Persisted derived-feature contract {contract_key!r} must be a mapping"
            )
        derived = str(contract.get("derived_feature", "")).strip()
        source = str(contract.get("source_feature", "")).strip()
        if not derived or not source:
            raise ValueError(
                f"Persisted derived-feature contract {contract_key!r} is incomplete"
            )
        return derived, source
    # Backward-compatible scoring for bundles created before the persisted
    # derived-feature contract existed.  New bundles never depend on these
    # runtime module constants.
    return (
        str(getattr(runtime, legacy_derived_attribute, "")).strip(),
        str(getattr(runtime, legacy_source_attribute, "")).strip(),
    )








def _source_holdout_contract(
    bundles: Mapping[str, ManifestRecord],
) -> dict[str, Any]:
    contracts: list[dict[str, Any]] = []
    cohort_columns: list[str] = []
    for family in MODEL_FAMILIES:
        payload = bundles[family].payload
        metadata = _manifest_source_metadata(payload)
        holdout = payload.get("excluded_holdout")
        if not isinstance(holdout, Mapping):
            holdout = metadata.get("excluded_holdout")
        if not isinstance(holdout, Mapping):
            raise ValueError(f"Frozen {family} bundle has no excluded_holdout contract")
        start = holdout.get("cohort_start")
        end = holdout.get("cohort_end")
        if start is None or end is None:
            raise ValueError(f"Frozen {family} excluded_holdout has no calendar bounds")
        identity_columns_raw = holdout.get("identity_columns")
        if not isinstance(identity_columns_raw, list) or not identity_columns_raw:
            raise ValueError(
                f"Frozen {family} excluded_holdout has no row identity columns"
            )
        identity_columns = [
            str(value).strip() for value in identity_columns_raw if str(value).strip()
        ]
        if not identity_columns or len(identity_columns) != len(set(identity_columns)):
            raise ValueError(
                f"Frozen {family} excluded_holdout identity columns are invalid"
            )
        policy_cluster_column = str(
            holdout.get("policy_cluster_column", "")
        ).strip()
        policy_cluster_source = "EXPLICIT_HOLDOUT_CONTRACT"
        if not policy_cluster_column:
            roles = holdout.get("identity_column_roles")
            if isinstance(roles, Mapping):
                policy_cluster_column = str(roles.get("policy_cluster", "")).strip()
                policy_cluster_source = "IDENTITY_COLUMN_ROLE_CONTRACT"
        if not policy_cluster_column:
            # Bundles from the immediately preceding materializer revision
            # explicitly defined identity order as policy key, anchor date,
            # cohort date. Preserve compatibility while recording the fallback.
            policy_cluster_column = identity_columns[0]
            policy_cluster_source = "LEGACY_ORDERED_IDENTITY_CONTRACT"
        if policy_cluster_column not in identity_columns:
            raise ValueError(
                f"Frozen {family} policy-cluster column is not an identity field: "
                f"{policy_cluster_column!r}"
            )
        contract = {
            "cohort_start": str(pd.Timestamp(start).date()),
            "cohort_end": str(pd.Timestamp(end).date()),
            "rows": int(holdout.get("rows", 0)),
            "population_sha256": str(holdout.get("population_sha256", "")),
            "row_index_sha256": str(holdout.get("row_index_sha256", "")),
            "label_availability_buffer_days": int(
                holdout.get("label_availability_buffer_days", 0)
            ),
            "final_training_label_cutoff_exclusive": str(
                holdout.get("final_training_label_cutoff_exclusive", "")
            ),
            "pre_holdout_label_unavailable_rows_excluded": int(
                holdout.get("pre_holdout_label_unavailable_rows_excluded", 0)
            ),
            "identity_columns": identity_columns,
            "policy_cluster_column": policy_cluster_column,
            "policy_cluster_column_source": policy_cluster_source,
            "target_scored": holdout.get("target_scored"),
            "used_for_preprocessing_or_fit": holdout.get(
                "used_for_preprocessing_or_fit"
            ),
        }
        if contract["target_scored"] is not False:
            raise ValueError(
                f"Frozen {family} holdout is not sealed as unscored during fitting"
            )
        if contract["used_for_preprocessing_or_fit"] is not False:
            raise ValueError(
                f"Frozen {family} holdout is not sealed as excluded from fitting"
            )
        contracts.append(contract)
        cohort_column = str(
            payload.get(
                "cohort_column",
                metadata.get("cohort_column", holdout.get("cohort_column", "")),
            )
        ).strip()
        if not cohort_column:
            identity_columns = holdout.get("identity_columns")
            if isinstance(identity_columns, list) and identity_columns:
                # The source materializer records identities in the order
                # policy key, anchor date, cohort date.  The explicit
                # cohort_column field takes precedence in newer bundles.
                cohort_column = str(identity_columns[-1]).strip()
        if not cohort_column:
            raise ValueError(
                f"Frozen {family} holdout contract does not identify its cohort column"
            )
        cohort_columns.append(cohort_column)
    if contracts[0] != contracts[1]:
        raise RuntimeError(
            "Source Logistic/XGBoost bundles disagree on the excluded-holdout window"
        )
    if len(set(cohort_columns)) != 1:
        raise RuntimeError(
            f"Source Logistic/XGBoost bundles disagree on cohort column: {cohort_columns}"
        )
    if pd.Timestamp(contracts[0]["cohort_start"]) > pd.Timestamp(
        contracts[0]["cohort_end"]
    ):
        raise ValueError("Frozen source excluded-holdout calendar bounds are reversed")
    return {**contracts[0], "cohort_column": cohort_columns[0]}








def resolve_target_data(
    target_run: TargetRun,
    *,
    project_root: Path,
    required_columns: set[str],
    target_column: str,
    tenure_column: str,
    explicit: Path | None,
    allow_hash_mismatch: bool,
) -> tuple[Path, Mapping[str, Any] | None, pd.DataFrame, dict[str, Any]]:
    required = sorted({*required_columns, target_column, tenure_column})
    candidates: list[tuple[Path, Mapping[str, Any] | None]] = []
    if explicit is not None:
        raw = explicit.expanduser()
        if raw.is_absolute():
            path = raw.resolve()
        else:
            possible = (project_root / raw, target_run.path / raw)
            existing = [value.resolve() for value in possible if value.is_file()]
            if len(existing) != 1:
                raise FileNotFoundError(
                    f"Explicit relative --target-data must resolve uniquely: {explicit}; "
                    f"matches={existing}"
                )
            path = existing[0]
        candidates.append((path, None))
    else:
        seen: set[Path] = set()
        for json_path in sorted(target_run.path.rglob("*.json")):
            try:
                payload = _json_load(json_path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            for identity in _identity_dicts(payload):
                path = _resolve_recorded_path(target_run.path, identity)
                if path is not None and path.suffix.lower() in {".parquet", ".pq"} and path not in seen:
                    seen.add(path)
                    candidates.append((path, identity))
    if not candidates:
        raise FileNotFoundError(
            f"No metadata-recorded target dataset was resolved inside {target_run.path}."
        )




    eligible: list[tuple[Path, Mapping[str, Any] | None, pd.DataFrame, dict[str, Any]]] = []
    errors: list[str] = []
    for path, identity in candidates:
        if not path.is_file():
            errors.append(f"missing: {path}")
            continue
        try:
            frame = pd.read_parquet(path, columns=required)
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        observed_sha = sha256_file(path)
        expected_sha = str(identity.get("sha256", "")) if identity is not None else ""
        hash_match = not expected_sha or observed_sha == expected_sha
        if not hash_match and not allow_hash_mismatch:
            errors.append(f"{path}: recorded SHA-256 mismatch")
            continue
        audit = {
            "path": str(path),
            "sha256": observed_sha,
            "size_bytes": int(path.stat().st_size),
            "recorded_sha256": expected_sha,
            "hash_match": hash_match,
            "columns_loaded": required,
        }
        eligible.append((path, identity, frame, audit))
    if len(eligible) != 1:
        raise RuntimeError(
            "Expected exactly one metadata-identified target dataset matching the "
            f"source bundle schema; found {len(eligible)}. Details: {' | '.join(errors)}"
        )
    return eligible[0]








def _filter_target_tenure(
    frame: pd.DataFrame,
    tenure_column: str,
    target_tenure: str,
) -> pd.DataFrame:
    requested = _normalize_tenure(target_tenure)
    observed = frame[tenure_column].map(_normalize_tenure)
    selected = observed.eq(requested)
    if not selected.any():
        raise ValueError(
            f"Target dataset contains no rows where {tenure_column}={requested!r}. "
            f"Observed values={sorted(observed.dropna().unique().tolist())}."
        )
    output = frame.loc[selected].copy().reset_index(drop=True)
    if not output[tenure_column].map(_normalize_tenure).eq(requested).all():
        raise AssertionError("Target-tenure filter failed")
    return output








def _filter_evaluation_window(
    frame: pd.DataFrame,
    *,
    cohort_column: str,
    holdout_contract: Mapping[str, Any],
    evaluation_window: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cohort = pd.to_datetime(frame[cohort_column], errors="coerce")
    invalid = int(cohort.isna().sum())
    if invalid:
        raise ValueError(
            f"Target cohort column {cohort_column!r} has {invalid} invalid/missing dates"
        )
    if evaluation_window == "all-target":
        observed = sorted(cohort.dt.to_period("M").astype(str).unique().tolist())
        return frame.copy().reset_index(drop=True), {
            "evaluation_window": evaluation_window,
            "cohort_column": cohort_column,
            "expected_months": None,
            "observed_months": observed,
            "missing_months": [],
            "extra_months": [],
            "exact_source_holdout_month_coverage": None,
        }
    if evaluation_window != "source-holdout":
        raise ValueError(f"Unsupported evaluation window: {evaluation_window!r}")
    start = pd.Timestamp(holdout_contract["cohort_start"])
    end = pd.Timestamp(holdout_contract["cohort_end"])
    selected = cohort.between(start, end, inclusive="both")
    if not selected.any():
        raise ValueError(
            "Target tenure has no rows in the source model's excluded-holdout "
            f"calendar window [{start.date()}, {end.date()}]."
        )
    output = frame.loc[selected].copy().reset_index(drop=True)
    output_months = pd.to_datetime(
        output[cohort_column], errors="raise"
    ).dt.to_period("M")
    expected_periods = pd.period_range(
        start=start.to_period("M"), end=end.to_period("M"), freq="M"
    )
    expected = {str(value) for value in expected_periods}
    observed = {str(value) for value in output_months.unique()}
    missing = sorted(expected.difference(observed))
    extra = sorted(observed.difference(expected))
    coverage = {
        "evaluation_window": evaluation_window,
        "cohort_column": cohort_column,
        "source_holdout_start": str(start.date()),
        "source_holdout_end": str(end.date()),
        "expected_months": sorted(expected),
        "observed_months": sorted(observed),
        "missing_months": missing,
        "extra_months": extra,
        "exact_source_holdout_month_coverage": not missing and not extra,
    }
    if missing or extra:
        raise ValueError(
            "Target tenure does not provide exact source-holdout calendar-month "
            f"coverage. missing_months={missing}; extra_months={extra}; "
            f"expected_months={coverage['expected_months']}; "
            f"observed_months={coverage['observed_months']}. "
            "Use --evaluation-window all-target only for an explicitly requested "
            "historical diagnostic, not to bypass an incomplete OOT comparison."
        )
    return output, coverage








def _deterministic_row_id(
    frame: pd.DataFrame, identity_columns: Sequence[str]
) -> pd.Series:
    columns = [str(value) for value in identity_columns]
    missing = [column for column in columns if column not in frame]
    if missing:
        raise KeyError(f"Target data is missing saved row identity columns: {missing}")
    canonical = frame.loc[:, columns].copy()
    missing_identity = canonical.isna().any(axis=1)
    if missing_identity.any():
        raise ValueError(
            f"Saved row identity columns contain missing target values; "
            f"rows={int(missing_identity.sum())}; columns={columns}"
        )
    for column in columns:
        if pd.api.types.is_datetime64_any_dtype(canonical[column]):
            canonical[column] = pd.to_datetime(canonical[column], errors="coerce").astype("string")
        else:
            canonical[column] = canonical[column].astype("string")
    if canonical.duplicated(keep=False).any():
        duplicates = int(canonical.duplicated(keep=False).sum())
        raise ValueError(
            f"Saved row identity columns are not unique on target data; "
            f"duplicate_rows={duplicates}; columns={columns}"
        )
    row_hash = pd.util.hash_pandas_object(canonical, index=False).astype("uint64")
    if row_hash.duplicated(keep=False).any():
        raise RuntimeError("A deterministic target row-identity hash collision occurred")
    return row_hash








def build_lift_table(
    y_true: Sequence[float] | np.ndarray,
    score_sort: Sequence[float] | np.ndarray,
    n_bins: int = 20,
    *,
    row_id: Sequence[Any] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Reproduce the supplied equal-count event-lift calculation.




    Scores are sorted descending, group 1 receives the highest scores, and lift
    is the group's observed event rate divided by the population event rate.
    ``y_true`` must therefore be the event being targeted (churn here).
    """




    y = np.asarray(y_true, dtype=float).reshape(-1)
    score = np.asarray(score_sort, dtype=float).reshape(-1)
    if y.shape != score.shape:
        raise ValueError("y_true and score_sort must have the same one-dimensional shape")
    if len(y) == 0:
        raise ValueError("Cannot calculate lift for an empty population")
    if int(n_bins) < 2 or int(n_bins) > len(y):
        raise ValueError("n_bins must be between 2 and the number of rows")
    if not np.isfinite(y).all() or not np.isfinite(score).all():
        raise ValueError("Lift inputs must be finite")
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("y_true must contain only binary 0/1 events")
    tie_breaker = np.arange(len(y), dtype=np.int64) if row_id is None else np.asarray(row_id)
    if tie_breaker.shape != y.shape:
        raise ValueError("row_id must have the same shape as y_true")
    frame = pd.DataFrame({"y": y, "score_sort": score, "_row_id": tie_breaker})
    frame = frame.sort_values(
        ["score_sort", "_row_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    # This intentionally mirrors the user's supplied implementation: qcut is
    # applied to the one-based rank after descending score order.
    frame["bin_rank"] = pd.qcut(
        frame.index + 1,
        q=int(n_bins),
        labels=np.arange(1, int(n_bins) + 1),
    ).astype(int)
    overall_rate = float(frame["y"].mean())
    output = (
        frame.groupby("bin_rank", as_index=False, sort=True)
        .agg(
            records=("y", "size"),
            actual_events=("y", "sum"),
            actual_event_rate=("y", "mean"),
            minimum_score=("score_sort", "min"),
            maximum_score=("score_sort", "max"),
        )
        .sort_values("bin_rank")
        .reset_index(drop=True)
    )
    output["actual_events"] = output["actual_events"].astype(int)
    output["overall_event_rate"] = overall_rate
    output["lift"] = (
        output["actual_event_rate"] / overall_rate if overall_rate > 0 else np.nan
    )
    output["percentile_upper"] = output["bin_rank"] * (100.0 / int(n_bins))
    output["cumulative_records"] = output["records"].cumsum()
    output["cumulative_events"] = output["actual_events"].cumsum()
    output["cumulative_event_rate"] = (
        output["cumulative_events"] / output["cumulative_records"]
    )
    output["cumulative_lift"] = (
        output["cumulative_event_rate"] / overall_rate if overall_rate > 0 else np.nan
    )
    return output








def summarize_lift_spread(lift_table: pd.DataFrame) -> dict[str, Any]:
    required = {"bin_rank", "lift"}
    missing = required - set(lift_table.columns)
    if missing:
        raise KeyError(f"Lift table is missing {sorted(missing)}")
    finite = lift_table.loc[np.isfinite(pd.to_numeric(lift_table["lift"], errors="coerce"))]
    if finite.empty:
        return {
            "highest_lift": np.nan,
            "highest_lift_group": pd.NA,
            "lowest_lift": np.nan,
            "lowest_lift_group": pd.NA,
            "highest_minus_lowest_lift": np.nan,
            "highest_divided_by_lowest_lift": np.nan,
            "ratio_defined": False,
            "ratio_status": "NO_EVENTS",
            "ordered_first_to_last_lift_ratio": np.nan,
        }
    highest_row = finite.loc[finite["lift"].idxmax()]
    lowest_row = finite.loc[finite["lift"].idxmin()]
    highest = float(highest_row["lift"])
    lowest = float(lowest_row["lift"])
    if lowest == 0.0:
        ratio = math.inf if highest > 0 else np.nan
        status = "ZERO_LOWEST_LIFT"
        defined = False
    else:
        ratio = highest / lowest
        status = "FINITE"
        defined = True
    ordered = finite.sort_values("bin_rank")
    last = float(ordered.iloc[-1]["lift"])
    ordered_ratio = float(ordered.iloc[0]["lift"]) / last if last != 0 else math.inf
    return {
        "highest_lift": highest,
        "highest_lift_group": int(highest_row["bin_rank"]),
        "lowest_lift": lowest,
        "lowest_lift_group": int(lowest_row["bin_rank"]),
        "highest_minus_lowest_lift": highest - lowest,
        "highest_divided_by_lowest_lift": ratio,
        "ratio_defined": defined,
        "ratio_status": status,
        "ordered_first_to_last_lift_ratio": ordered_ratio,
    }








def _ranking_metrics(y_retention: np.ndarray, p_retention: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_retention, dtype=int)
    p = np.asarray(p_retention, dtype=float)
    y_churn = 1 - y
    p_churn = 1.0 - p
    prevalence = float(y_churn.mean())
    ap = float(average_precision_score(y_churn, p_churn))
    return {
        "rows": int(len(y)),
        "retention_roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan,
        "churn_average_precision": ap,
        "churn_prevalence_baseline": prevalence,
        "churn_ap_skill": (ap - prevalence) / (1.0 - prevalence)
        if prevalence < 1.0
        else np.nan,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "actual_retention_rate": float(y.mean()),
        "mean_predicted_retention": float(p.mean()),
        "calibration_gap": float(p.mean() - y.mean()),
    }








def _state_mapping_audit(
    target: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    shared_state: SharedStateContract,
    *,
    target_column: str,
    source_tenure: str,
    target_tenure: str,
) -> pd.DataFrame:
    if shared_state.source_feature not in target:
        raise KeyError(
            "Target scoring frame is missing the shared source-state field "
            f"{shared_state.source_feature!r}."
        )
    state = (
        target[shared_state.source_feature]
        .astype("string")
        .str.strip()
        .str.upper()
    )
    state = state.mask(state.isin({"", "<NA>", "NAN", "NONE", "NULL"}), pd.NA)
    mapped = state.map(dict(shared_state.mapping)).astype("string")
    audit = pd.DataFrame(
        {
            "raw_state": state,
            "source_retention_group": mapped,
            "actual_retention": target[target_column].to_numpy(),
            **{
                f"{family}_predicted_retention": predictions[family]
                for family in MODEL_FAMILIES
            },
        }
    )
    audit["mapping_status"] = np.where(
        audit["raw_state"].isna(),
        "MISSING_RAW_STATE",
        np.where(
            audit["source_retention_group"].isna(),
            "UNSEEN_BY_SOURCE",
            "SOURCE_MAPPING_APPLIED",
        ),
    )
    aggregations: dict[str, tuple[str, str]] = {
        "rows": ("actual_retention", "size"),
        "actual_retention_rate": ("actual_retention", "mean"),
    }
    for family in MODEL_FAMILIES:
        aggregations[f"{family}_mean_predicted_retention"] = (
            f"{family}_predicted_retention",
            "mean",
        )
    grouped = (
        audit.groupby(
            ["raw_state", "source_retention_group", "mapping_status"],
            dropna=False,
            as_index=False,
        )
        .agg(**aggregations)
        .sort_values(["source_retention_group", "raw_state"], kind="mergesort")
    )
    grouped.insert(0, "shared_state_mapping_sha256", shared_state.mapping_sha256)
    grouped.insert(0, "target_tenure", target_tenure)
    grouped.insert(0, "source_tenure", source_tenure)
    grouped["target_volume_share"] = grouped["rows"] / len(target)
    columns = [
        "source_tenure",
        "target_tenure",
        "shared_state_mapping_sha256",
        "raw_state",
        "source_retention_group",
        "mapping_status",
        "rows",
        "target_volume_share",
        "actual_retention_rate",
        *[
            f"{family}_mean_predicted_retention"
            for family in MODEL_FAMILIES
        ],
    ]
    return grouped.reindex(columns=columns).reset_index(drop=True)








def plot_lift_like_graph1(
    lift_by_bins: Mapping[int, pd.DataFrame],
    *,
    title: str,
    save_html_path: Path,
) -> None:
    import plotly.graph_objects as go




    bins_list = tuple(sorted(int(value) for value in lift_by_bins))
    initial = 20 if 20 in lift_by_bins else bins_list[0]




    def prep(frame: pd.DataFrame, bins: int) -> tuple[list[str], np.ndarray, list[str]]:
        display = frame.assign(
            percentile=(frame["bin_rank"] * (100.0 / bins)).round().astype(int)
        ).sort_values(["lift", "bin_rank"], ascending=[True, True])
        x = display["percentile"].astype(str).tolist()
        y = display["lift"].to_numpy(dtype=float)
        text = [f"{value:.1f}" for value in y]
        return x, y, text




    x0, y0, text0 = prep(lift_by_bins[initial], initial)
    finite_maxima = [
        float(frame["lift"].max())
        for frame in lift_by_bins.values()
        if np.isfinite(frame["lift"]).any()
    ]
    maximum = max(finite_maxima, default=1.0)
    figure = go.Figure(
        data=[
            go.Bar(
                x=x0,
                y=y0,
                text=text0,
                textposition="outside",
                marker={"color": y0, "coloraxis": "coloraxis", "line": {"width": 0}},
                hovertemplate="Percentile: %{x}%<br>Lift: %{y:.3f}<extra></extra>",
            )
        ]
    )
    buttons = []
    for bins in bins_list:
        x, y, text = prep(lift_by_bins[bins], bins)
        buttons.append(
            {
                "label": str(bins),
                "method": "update",
                "args": [
                    {"x": [x], "y": [y], "text": [text], "marker.color": [y]},
                    {"title": title, "xaxis.title": "Percentile"},
                ],
            }
        )
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=760,
        margin={"l": 70, "r": 40, "t": 90, "b": 60},
        xaxis={"title": "Percentile"},
        yaxis={
            "title": "Lift",
            "range": [0, maximum * 1.08],
            "gridcolor": "rgba(0,0,0,0.15)",
            "griddash": "dot",
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.0,
                "y": 1.12,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": buttons,
                "showactive": True,
            }
        ],
        coloraxis={
            "cmin": 0,
            "cmax": maximum,
            "colorscale": [[0.0, "#2f5d8a"], [0.5, "#f2e6cf"], [1.0, "#a33b1f"]],
            "colorbar": {
                "title": "Lift",
                "orientation": "h",
                "x": 0.5,
                "xanchor": "center",
                "y": 1.05,
                "yanchor": "bottom",
                "len": 0.35,
                "thickness": 14,
            },
        },
    )
    figure.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Average Lift",
        annotation_position="right",
    )
    save_html_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(save_html_path, include_plotlyjs=True)








def plot_source_state_mapping_choropleth(
    state_audit: pd.DataFrame,
    *,
    source_tenure: str,
    target_tenure: str,
    save_html_path: Path,
) -> None:
    """Map the one shared frozen source grouping applied to target states.




    This visualization consumes the already-produced source-mapping audit.  It
    does not estimate, alter, or display any target-native state grouping.
    """




    import plotly.express as px




    display = state_audit.loc[
        state_audit["mapping_status"].eq("SOURCE_MAPPING_APPLIED")
    ].copy()
    display = display.loc[
        display["raw_state"].astype("string").str.fullmatch(r"[A-Z]{2}", na=False)
    ]
    if display.empty:
        raise ValueError("No source-mapped target states are available.")
    display["source_retention_group"] = display[
        "source_retention_group"
    ].astype(str)
    display = display.sort_values(
        ["source_retention_group", "raw_state"], kind="mergesort"
    )
    groups = sorted(display["source_retention_group"].unique().tolist())
    palette = [
        *px.colors.qualitative.Safe,
        *px.colors.qualitative.Bold,
        *px.colors.qualitative.Set3,
    ]
    color_map = {
        group: palette[index % len(palette)] for index, group in enumerate(groups)
    }
    figure = px.choropleth(
        display,
        locations="raw_state",
        locationmode="USA-states",
        scope="usa",
        color="source_retention_group",
        color_discrete_map=color_map,
        category_orders={"source_retention_group": groups},
        hover_name="raw_state",
        hover_data={
            "raw_state": False,
            "source_retention_group": True,
            "rows": ":,",
            "target_volume_share": ":.2%",
            "actual_retention_rate": ":.2%",
            "logistic_mean_predicted_retention": ":.2%",
            "xgboost_mean_predicted_retention": ":.2%",
        },
        title=(
            f"Tenure {source_tenure} shared source state mapping "
            f"applied to tenure {target_tenure}"
        ),
        labels={
            "source_retention_group": "Source retention group",
            "rows": "Target rows",
            "target_volume_share": "Target volume share",
            "actual_retention_rate": "Target actual retention",
            "logistic_mean_predicted_retention": (
                "Logistic target mean predicted retention"
            ),
            "xgboost_mean_predicted_retention": (
                "XGBoost target mean predicted retention"
            ),
        },
    )
    figure.update_layout(
        template="plotly_white",
        height=720,
        margin={"l": 20, "r": 20, "t": 90, "b": 20},
        legend={"title": {"text": "Shared source group"}},
    )
    figure.add_annotation(
        text=(
            "Diagnostic only: colors are the shared frozen source-tenure groups; "
            "no target-native grouping is created."
        ),
        x=0.5,
        y=0.01,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 12, "color": "#555555"},
    )
    save_html_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(save_html_path, include_plotlyjs=True)








def _outer_reference_path(
    reference: Any,
    *,
    owner_manifest: Path,
    project_root: Path,
) -> Path:
    """Resolve one c4 artifact reference without filesystem-time guessing."""




    raw_value: Any = reference
    if isinstance(reference, Mapping):
        raw_value = next(
            (
                reference.get(key)
                for key in (
                    "manifest_path",
                    "path",
                    "relative_path",
                    "manifest_relative_path",
                )
                if reference.get(key) is not None
                and str(reference.get(key)).strip()
            ),
            None,
        )
    if raw_value is None or not str(raw_value).strip():
        raise ValueError(
            f"Outer-evaluation manifest has an empty artifact reference: {reference!r}"
        )
    raw = Path(str(raw_value)).expanduser()
    candidates = [raw] if raw.is_absolute() else [
        owner_manifest.parent / raw,
        project_root / raw,
    ]
    existing = list(
        dict.fromkeys(candidate.resolve() for candidate in candidates if candidate.exists())
    )
    if len(existing) != 1:
        raise FileNotFoundError(
            "Outer-evaluation artifact reference must resolve exactly once; "
            f"reference={raw_value!r}; matches={[str(value) for value in existing]}."
        )
    return existing[0]








def _family_manifest_at(path: Path, expected_family: str) -> ManifestRecord:
    candidate = path
    if candidate.is_dir():
        matches: list[Path] = []
        for item in sorted(candidate.glob("*.json")):
            try:
                payload = _json_load(item)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if (
                payload.get("contract") == FAMILY_CONTRACT
                and str(payload.get("model_family", "")).strip().lower()
                == expected_family
            ):
                matches.append(item.resolve())
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one frozen {expected_family} family manifest "
                f"under {candidate}; found={[str(value) for value in matches]}."
            )
        candidate = matches[0]
    payload = _json_load(candidate)
    if payload.get("contract") != FAMILY_CONTRACT:
        raise ValueError(f"Unsupported fold-bundle manifest contract: {candidate}")
    observed_family = str(payload.get("model_family", "")).strip().lower()
    if observed_family != expected_family:
        raise ValueError(
            f"Fold-bundle family mismatch at {candidate}: expected "
            f"{expected_family!r}, observed {observed_family!r}."
        )
    return ManifestRecord(candidate.resolve(), payload)








def _outer_manifest_paths_under(root: Path) -> list[Path]:
    if root.is_file():
        candidates = [root]
    elif root.is_dir():
        candidates = sorted(root.rglob("*.json"))
    else:
        return []
    output: list[Path] = []
    for path in candidates:
        try:
            payload = _json_load(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if str(payload.get("contract", "")).strip() == OUTER_EVALUATION_CONTRACT:
            output.append(path.resolve())
    return output








def resolve_outer_evaluation_manifest(
    project_root: Path,
    tenure: str,
    *,
    explicit_manifest: Path | None,
    explicit_run: Path | None,
) -> Path:
    """Resolve one c4 receipt by governed run identity, never file mtime."""




    requested = _normalize_tenure(tenure)
    if explicit_manifest is not None:
        raw = explicit_manifest.expanduser()
        if raw.is_absolute():
            root = raw.resolve()
        else:
            matches = [
                value.resolve()
                for value in (project_root / raw, Path.cwd() / raw)
                if value.exists()
            ]
            if len(matches) != 1:
                raise FileNotFoundError(
                    "Explicit outer-evaluation manifest must resolve exactly once; "
                    f"value={explicit_manifest}; matches={matches}."
                )
            root = matches[0]
        paths = _outer_manifest_paths_under(root)
    elif explicit_run is not None:
        run = _resolve_project_or_run_path(project_root, explicit_run)
        paths = _outer_manifest_paths_under(run)
    else:
        eligible_by_run: list[tuple[Path, pd.Timestamp]] = []
        for run in _pipeline_run_dirs(project_root):
            run_paths = _outer_manifest_paths_under(run)
            for path in run_paths:
                payload = _json_load(path)
                profile = payload.get("profile")
                if not isinstance(profile, Mapping):
                    continue
                if _selected_groups(profile) != (requested,):
                    continue
                try:
                    completed = _complete_full_run_timestamp(
                        run, profile, str(payload.get("profile_hash", ""))
                    )
                except ValueError:
                    continue
                eligible_by_run.append((path, completed))
        if not eligible_by_run:
            paths = []
        else:
            latest = max(value for _, value in eligible_by_run)
            paths = [path for path, value in eligible_by_run if value == latest]
    filtered: list[Path] = []
    for path in paths:
        payload = _json_load(path)
        profile = payload.get("profile")
        if not isinstance(profile, Mapping):
            continue
        if _selected_groups(profile) == (requested,):
            filtered.append(path)
    filtered = list(dict.fromkeys(filtered))
    if len(filtered) != 1:
        raise RuntimeError(
            "Expected exactly one c4 outer-evaluation manifest for tenure "
            f"{requested!r}; found={[str(value) for value in filtered]}. Pass "
            "--source-outer-manifest/--target-outer-manifest or an explicit run."
        )
    return filtered[0]








def load_outer_evaluation_record(
    path: Path,
    *,
    project_root: Path,
    expected_tenure: str,
) -> OuterEvaluationRecord:
    """Load and fail-closed validate one c4 outer-evaluation receipt."""




    manifest_path = path.expanduser().resolve()
    payload = _json_load(manifest_path)
    if str(payload.get("contract", "")).strip() != OUTER_EVALUATION_CONTRACT:
        raise ValueError(
            f"Expected {OUTER_EVALUATION_CONTRACT!r}: {manifest_path}"
        )
    if payload.get("contract_revision") != SUPPORTED_OUTER_EVALUATION_CONTRACT_REVISION:
        raise ValueError(
            "Unsupported c4 outer-evaluation contract revision; expected "
            f"{SUPPORTED_OUTER_EVALUATION_CONTRACT_REVISION}, observed "
            f"{payload.get('contract_revision')!r}: {manifest_path}"
        )
    if payload.get("final_holdout_rows_scored") != 0:
        raise ValueError(
            "Tenure-grouping evidence requires a sealed final holdout; c4 did "
            f"not record final_holdout_rows_scored=0: {manifest_path}"
        )
    profile = payload.get("profile")
    if not isinstance(profile, Mapping):
        raise ValueError(f"c4 manifest has no profile object: {manifest_path}")
    requested = _normalize_tenure(expected_tenure)
    if _selected_groups(profile) != (requested,):
        raise ValueError(
            f"c4 profile does not represent exactly tenure {requested!r}: "
            f"{manifest_path}"
        )
    outer_run_id = str(
        payload.get("run_id", payload.get("source_run_id", ""))
    ).strip()
    outer_profile_hash = str(payload.get("profile_hash", "")).strip()
    if not outer_run_id or not outer_profile_hash:
        raise ValueError("c4 manifest must record run_id and profile_hash.")
    governed_run = project_root / "output" / "pipeline_runs" / outer_run_id
    if not governed_run.is_dir():
        raise FileNotFoundError(
            f"c4 manifest does not resolve to a governed pipeline run: {governed_run}"
        )
    try:
        _complete_full_run_timestamp(governed_run, profile, outer_profile_hash)
    except ValueError as exc:
        raise ValueError(
            "Tenure-grouping evidence requires a metadata-confirmed COMPLETE/FULL "
            f"source run: {governed_run}"
        ) from exc
    source_data_identity = payload.get("source_data_identity")
    if not isinstance(source_data_identity, Mapping):
        source_data_identity = payload.get("input_snapshot")
    outer_snapshot_sha = (
        str(source_data_identity.get("sha256", "")).strip().lower()
        if isinstance(source_data_identity, Mapping)
        else ""
    )
    if not outer_snapshot_sha:
        raise ValueError("c4 manifest must record its input snapshot SHA-256.")
    target_column = str(payload.get("target_column", "")).strip()
    cohort_column = str(payload.get("cohort_column", "")).strip()
    raw_identity = payload.get("identity_columns")
    if not target_column or not cohort_column:
        raise ValueError("c4 manifest must record target_column and cohort_column.")
    if not isinstance(raw_identity, list):
        raise ValueError("c4 manifest identity_columns must be a nonempty list.")
    identity_columns = tuple(
        str(value).strip() for value in raw_identity if str(value).strip()
    )
    if not identity_columns or len(identity_columns) != len(set(identity_columns)):
        raise ValueError("c4 manifest identity_columns must be nonempty and unique.")




    oof_reference = payload.get("outer_oof_predictions")
    if not isinstance(oof_reference, Mapping):
        # Accept the earlier c4 design spelling, but still require the same hash.
        oof_reference = payload.get("oof_predictions")
    if not isinstance(oof_reference, Mapping):
        raise ValueError("c4 manifest has no outer_oof_predictions artifact.")
    oof_path = _outer_reference_path(
        oof_reference, owner_manifest=manifest_path, project_root=project_root
    )
    expected_oof_sha = str(oof_reference.get("sha256", "")).strip().lower()
    if not expected_oof_sha or sha256_file(oof_path) != expected_oof_sha:
        raise RuntimeError(
            f"c4 outer OOF prediction hash is missing or invalid: {oof_path}"
        )




    raw_folds = payload.get("folds")
    if not isinstance(raw_folds, list) or not raw_folds:
        raise ValueError("c4 manifest folds must be a nonempty list.")
    folds: list[OuterFoldBundle] = []
    recipe_identity: dict[str, set[str]] = {
        family: set() for family in MODEL_FAMILIES
    }
    for raw_fold in raw_folds:
        if not isinstance(raw_fold, Mapping):
            raise TypeError("Every c4 fold entry must be a mapping.")
        try:
            fold_number = int(raw_fold["fold"])
            validation_start = pd.Timestamp(raw_fold["validation_start"])
            validation_end = pd.Timestamp(raw_fold["validation_end"])
            label_cutoff = pd.Timestamp(raw_fold["label_cutoff"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid c4 fold definition: {raw_fold}") from exc
        if fold_number < 1 or validation_start > validation_end:
            raise ValueError(f"Invalid c4 fold bounds: {raw_fold}")
        if raw_fold.get("final_holdout_rows_scored") != 0:
            raise ValueError(
                f"c4 fold {fold_number} does not seal the final holdout."
            )
        family_references = raw_fold.get("families")
        if not isinstance(family_references, Mapping):
            raise ValueError(f"c4 fold {fold_number} has no family bundle map.")
        records: dict[str, ManifestRecord] = {}
        for family in MODEL_FAMILIES:
            reference = family_references.get(family)
            if reference is None:
                raise ValueError(
                    f"c4 fold {fold_number} is missing its {family} bundle."
                )
            family_path = _outer_reference_path(
                reference,
                owner_manifest=manifest_path,
                project_root=project_root,
            )
            record = _family_manifest_at(family_path, family)
            expected_sha = (
                str(reference.get("sha256", reference.get("manifest_sha256", "")))
                .strip()
                .lower()
                if isinstance(reference, Mapping)
                else ""
            )
            if not expected_sha or sha256_file(record.path) != expected_sha:
                raise RuntimeError(
                    f"c4 fold {fold_number} {family} bundle manifest hash is "
                    f"missing or invalid: {record.path}"
                )
            metadata = _manifest_source_metadata(record.payload)
            role = str(
                metadata.get("bundle_role", record.payload.get("bundle_role", ""))
            ).strip().upper()
            recorded_fold = metadata.get(
                "outer_fold", record.payload.get("outer_fold")
            )
            if role != "OUTER_FOLD_EVALUATION" or int(recorded_fold) != fold_number:
                raise ValueError(
                    f"c4 fold {fold_number} {family} reference is not an outer-fold "
                    f"evaluation bundle: role={role!r}; outer_fold={recorded_fold!r}."
                )
            if _manifest_source_tenure(record.payload) != requested:
                raise ValueError(
                    f"c4 fold {fold_number} {family} bundle has the wrong source "
                    "tenure."
                )
            family_identity = _manifest_identity(record.payload)
            if family_identity["source_run_id"] != outer_run_id:
                raise ValueError(
                    f"c4 fold {fold_number} {family} bundle source run differs "
                    "from the outer-evaluation manifest."
                )
            if family_identity["source_profile_hash"] != outer_profile_hash:
                raise ValueError(
                    f"c4 fold {fold_number} {family} bundle profile hash differs "
                    "from the outer-evaluation manifest."
                )
            if (
                family_identity["source_snapshot_sha256"].strip().lower()
                != outer_snapshot_sha
            ):
                raise ValueError(
                    f"c4 fold {fold_number} {family} bundle input snapshot differs "
                    "from the outer-evaluation manifest."
                )
            recipe_identity[family].add(
                sha256_json(
                    {
                        "features": record.payload.get("features"),
                        "model_recipe": record.payload.get("model_recipe"),
                        "seeds": record.payload.get("seeds"),
                        "boosting_rounds": record.payload.get("boosting_rounds"),
                    }
                )
            )
            records[family] = record
        holdout = _source_holdout_contract(records)
        if (
            pd.Timestamp(holdout["cohort_start"]) != validation_start
            or pd.Timestamp(holdout["cohort_end"]) != validation_end
        ):
            raise ValueError(
                f"c4 fold {fold_number} bundle validation window disagrees with "
                "the c4 fold definition."
            )
        folds.append(
            OuterFoldBundle(
                fold=fold_number,
                validation_start=validation_start,
                validation_end=validation_end,
                label_cutoff=label_cutoff,
                families=records,
            )
        )
    folds.sort(key=lambda value: value.fold)
    fold_numbers = [value.fold for value in folds]
    if len(fold_numbers) != len(set(fold_numbers)):
        raise ValueError("c4 manifest has duplicate outer fold numbers.")
    if fold_numbers != list(range(1, len(fold_numbers) + 1)):
        raise ValueError(
            f"c4 outer fold numbers must be consecutive from 1: {fold_numbers}"
        )
    inconsistent = {
        family: values
        for family, values in recipe_identity.items()
        if len(values) != 1
    }
    if inconsistent:
        raise ValueError(
            "All outer folds must use one locked inner-selected recipe per model "
            f"family; inconsistent recipe identities={inconsistent}."
        )
    return OuterEvaluationRecord(
        path=manifest_path,
        payload=payload,
        folds=tuple(folds),
        oof_predictions_path=oof_path,
        identity_columns=identity_columns,
        target_column=target_column,
        cohort_column=cohort_column,
    )








def _assert_matching_outer_design(
    source: OuterEvaluationRecord,
    target: OuterEvaluationRecord,
) -> None:
    if source.target_column != target.target_column:
        raise ValueError("Source and target c4 manifests use different outcomes.")
    if source.cohort_column != target.cohort_column:
        raise ValueError("Source and target c4 manifests use different cohort fields.")
    if source.identity_columns != target.identity_columns:
        raise ValueError(
            "Source and target c4 manifests use different row-identity contracts."
        )
    source_folds = {
        item.fold: (
            item.validation_start,
            item.validation_end,
            item.label_cutoff,
        )
        for item in source.folds
    }
    target_folds = {
        item.fold: (
            item.validation_start,
            item.validation_end,
            item.label_cutoff,
        )
        for item in target.folds
    }
    if source_folds != target_folds:
        raise ValueError(
            "Source and target c4 manifests do not use identical outer validation "
            f"windows/label cutoffs: source={source_folds}; target={target_folds}."
        )
    source_profile = source.payload.get("profile", {})
    target_profile = target.payload.get("profile", {})
    # Tenure 0 is new business while 1+ are renewals, so business_scope is
    # intentionally allowed to differ in an all-business transfer study.
    for key in ("tenure_dimension", "tenure_group_domain"):
        if source_profile.get(key) != target_profile.get(key):
            raise ValueError(
                f"Source and target outer evaluations disagree on profile {key}."
            )








def _read_prediction_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported outer OOF prediction type: {path.suffix}")








def _load_target_native_oof(
    record: OuterEvaluationRecord,
    *,
    target_tenure: str,
    explicit_path: Path | None,
    project_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = record.oof_predictions_path
    expected_sha = str(
        (
            record.payload.get("outer_oof_predictions")
            or record.payload.get("oof_predictions")
            or {}
        ).get("sha256", "")
    ).strip().lower()
    if explicit_path is not None:
        candidate = (
            explicit_path.expanduser().resolve()
            if explicit_path.expanduser().is_absolute()
            else (project_root / explicit_path.expanduser()).resolve()
        )
        if not candidate.is_file():
            raise FileNotFoundError(f"Target-native outer OOF file is missing: {candidate}")
        if sha256_file(candidate) != expected_sha:
            raise RuntimeError(
                "An explicit target-native outer OOF file must match the c4 "
                "manifest hash exactly."
            )
        path = candidate
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha:
        raise RuntimeError(f"Target-native outer OOF hash mismatch: {path}")
    frame = _read_prediction_table(path)
    recorded_rows = (
        record.payload.get("outer_oof_predictions")
        or record.payload.get("oof_predictions")
        or {}
    ).get("rows")
    if recorded_rows is not None and int(recorded_rows) != len(frame):
        raise ValueError(
            "Target-native outer OOF row count disagrees with its c4 manifest."
        )
    required = {
        "row_id",
        "fold",
        "cohort_date",
        "model_family",
        "y_retention",
        "p_retention",
        *record.identity_columns,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Target-native outer OOF predictions are missing columns: {missing}"
        )
    tenure_candidates = [
        str(record.payload.get("profile", {}).get("tenure_dimension", "")).strip(),
        "tenure_group",
    ]
    tenure_column = next(
        (value for value in tenure_candidates if value and value in frame), None
    )
    if tenure_column is None:
        raise ValueError(
            "Target-native outer OOF predictions must retain tenure_group or the "
            "profile tenure dimension."
        )
    selected = frame[tenure_column].map(_normalize_tenure).eq(
        _normalize_tenure(target_tenure)
    )
    frame = frame.loc[selected].copy()
    if frame.empty:
        raise ValueError("Target-native outer OOF file has no requested tenure rows.")
    frame["model_family"] = frame["model_family"].astype(str).str.strip().str.lower()
    unexpected = sorted(set(frame["model_family"]).difference(MODEL_FAMILIES))
    if unexpected:
        raise ValueError(f"Unexpected target-native model families: {unexpected}")
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame["cohort_date"] = pd.to_datetime(frame["cohort_date"], errors="raise")
    frame["y_retention"] = pd.to_numeric(
        frame["y_retention"], errors="raise"
    ).astype(int)
    frame["p_retention"] = pd.to_numeric(
        frame["p_retention"], errors="raise"
    ).astype(float)
    if not set(frame["y_retention"].unique()).issubset({0, 1}):
        raise ValueError("Target-native OOF outcomes must be binary 0/1.")
    if (
        not np.isfinite(frame["p_retention"]).all()
        or frame["p_retention"].lt(0).any()
        or frame["p_retention"].gt(1).any()
    ):
        raise ValueError("Target-native OOF probabilities must be finite in [0, 1].")
    key = ["fold", *record.identity_columns, "model_family"]
    if frame.duplicated(key, keep=False).any():
        raise ValueError(
            "Target-native OOF predictions contain duplicate fold/identity/family rows."
        )
    expected_folds = {item.fold for item in record.folds}
    if set(frame["fold"]) != expected_folds:
        raise ValueError(
            "Target-native OOF fold numbers disagree with its c4 manifest: "
            f"expected={sorted(expected_folds)}; observed={sorted(set(frame['fold']))}."
        )
    return frame, {
        "path": str(path),
        "sha256": observed_sha,
        "rows": int(len(frame)),
        "tenure_column": tenure_column,
    }








def _canonical_identity_frame(
    frame: pd.DataFrame, identity_columns: Sequence[str]
) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in identity_columns:
        if column not in frame:
            raise KeyError(f"Identity column is missing: {column}")
        values = frame[column]
        if values.isna().any():
            raise ValueError(f"Identity column {column!r} contains missing values.")
        upper = column.upper()
        if "DATE" in upper or upper.endswith("_DT"):
            parsed = pd.to_datetime(values, errors="coerce")
            if parsed.isna().any():
                raise ValueError(f"Identity date column {column!r} is invalid.")
            output[column] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        else:
            output[column] = values.astype("string")
    return output








def _identity_join_key(
    frame: pd.DataFrame, identity_columns: Sequence[str]
) -> pd.Series:
    canonical = _canonical_identity_frame(frame, identity_columns)
    if canonical.duplicated(keep=False).any():
        raise ValueError(
            f"Row identity is not unique for columns {list(identity_columns)}."
        )
    keys = pd.util.hash_pandas_object(canonical, index=False).astype("uint64")
    if keys.duplicated(keep=False).any():
        raise RuntimeError("A deterministic row-identity hash collision occurred.")
    return keys.astype("string")








def _native_oof_wide(
    native: pd.DataFrame,
    *,
    identity_columns: Sequence[str],
) -> pd.DataFrame:
    row_columns = [
        "fold",
        "row_id",
        "cohort_date",
        "y_retention",
        *identity_columns,
    ]
    grouped = native.groupby(
        ["fold", *identity_columns], dropna=False, sort=False
    )
    for column in ("row_id", "cohort_date", "y_retention"):
        inconsistent = grouped[column].nunique(dropna=False).gt(1)
        if inconsistent.any():
            raise ValueError(
                f"Target-native model families disagree on {column} for the same row."
            )
    base = native.loc[:, row_columns].drop_duplicates(
        ["fold", *identity_columns]
    )
    probabilities = native.pivot(
        index=["fold", *identity_columns],
        columns="model_family",
        values="p_retention",
    )
    missing_families = sorted(set(MODEL_FAMILIES).difference(probabilities.columns))
    if missing_families or probabilities.isna().any().any():
        raise ValueError(
            "Each target-native OOF row must contain both model families; "
            f"missing={missing_families}."
        )
    probabilities = probabilities.rename(
        columns={family: f"native_{family}_p_retention" for family in MODEL_FAMILIES}
    ).reset_index()
    return base.merge(
        probabilities,
        on=["fold", *identity_columns],
        how="inner",
        validate="one_to_one",
    )








def _top_fraction_metrics(
    y_retention: np.ndarray,
    p_retention: np.ndarray,
    *,
    row_id: Sequence[Any],
    fraction: float = 0.10,
) -> dict[str, float]:
    y_churn = 1 - np.asarray(y_retention, dtype=int)
    p_churn = 1.0 - np.asarray(p_retention, dtype=float)
    ranking = pd.DataFrame(
        {"y_churn": y_churn, "p_churn": p_churn, "row_id": list(row_id)}
    ).sort_values(
        ["p_churn", "row_id"], ascending=[False, True], kind="mergesort"
    )
    top_n = max(1, int(math.ceil(len(ranking) * float(fraction))))
    top = ranking.iloc[:top_n]
    prevalence = float(ranking["y_churn"].mean())
    precision = float(top["y_churn"].mean())
    total_events = float(ranking["y_churn"].sum())
    return {
        "top10_lift": precision / prevalence if prevalence > 0 else np.nan,
        "churn_precision_10pct": precision,
        "churn_capture_10pct": (
            float(top["y_churn"].sum()) / total_events
            if total_events > 0
            else np.nan
        ),
        "top_group_rows": int(top_n),
    }








def _outer_metrics(
    y_retention: np.ndarray,
    p_retention: np.ndarray,
    *,
    row_id: Sequence[Any],
) -> dict[str, float]:
    return {
        **_ranking_metrics(y_retention, p_retention),
        **_top_fraction_metrics(
            y_retention, p_retention, row_id=row_id, fraction=0.10
        ),
    }








def _pooled_outer_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    metrics = _ranking_metrics(
        predictions["y_retention"].to_numpy(),
        predictions["p_retention"].to_numpy(),
    )
    selected_parts: list[pd.DataFrame] = []
    for _, fold in predictions.groupby("fold", sort=True):
        ranked = fold.assign(p_churn=1.0 - fold["p_retention"]).sort_values(
            ["p_churn", "row_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        selected_parts.append(ranked.iloc[: max(1, int(math.ceil(len(ranked) * 0.10)))])
    selected = pd.concat(selected_parts, ignore_index=True)
    y_churn = 1 - predictions["y_retention"].to_numpy(dtype=int)
    prevalence = float(y_churn.mean())
    precision = float((1 - selected["y_retention"]).mean())
    total_events = float(y_churn.sum())
    metrics.update(
        {
            "top10_lift": precision / prevalence if prevalence > 0 else np.nan,
            "churn_precision_10pct": precision,
            "churn_capture_10pct": (
                float((1 - selected["y_retention"]).sum()) / total_events
                if total_events > 0
                else np.nan
            ),
            "top_group_rows": int(len(selected)),
            "top10_selection_scope": "TOP_10_PERCENT_WITHIN_EACH_OUTER_FOLD",
        }
    )
    return metrics








METRIC_DIRECTION = {
    "retention_roc_auc": "HIGHER_IS_BETTER",
    "churn_average_precision": "HIGHER_IS_BETTER",
    "churn_ap_skill": "HIGHER_IS_BETTER",
    "top10_lift": "HIGHER_IS_BETTER",
    "churn_precision_10pct": "HIGHER_IS_BETTER",
    "churn_capture_10pct": "HIGHER_IS_BETTER",
    "log_loss": "LOWER_IS_BETTER",
    "brier_score": "LOWER_IS_BETTER",
    "calibration_gap": "CLOSER_TO_ZERO_IS_BETTER",
}








def _metric_delta_rows(
    source_metrics: pd.DataFrame,
    native_metrics: pd.DataFrame,
    *,
    scope: str,
    join_columns: Sequence[str],
) -> list[dict[str, Any]]:
    joined = source_metrics.merge(
        native_metrics,
        on=list(join_columns),
        suffixes=("_source", "_native"),
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for record in joined.to_dict(orient="records"):
        for metric, direction in METRIC_DIRECTION.items():
            source_value = record.get(f"{metric}_source")
            native_value = record.get(f"{metric}_native")
            if source_value is None or native_value is None:
                continue
            if direction == "HIGHER_IS_BETTER":
                performance_delta = source_value - native_value
            elif direction == "LOWER_IS_BETTER":
                performance_delta = native_value - source_value
            else:
                performance_delta = abs(native_value) - abs(source_value)
            rows.append(
                {
                    "scope": scope,
                    **{column: record[column] for column in join_columns},
                    "metric": metric,
                    "direction": direction,
                    "source_transfer_value": source_value,
                    "target_native_value": native_value,
                    "source_minus_native": source_value - native_value,
                    "performance_delta_positive_means_source_better": (
                        performance_delta
                    ),
                }
            )
    return rows








def _run_tenure_grouping_outer_oof(args: argparse.Namespace) -> int:
    """Produce fold-matched, score-only evidence for tenure architecture."""




    output_dir = _output_directory(args)
    output_dir.mkdir(parents=True, exist_ok=False)
    source_manifest_path = resolve_outer_evaluation_manifest(
        args.project_root,
        args.source_tenure,
        explicit_manifest=args.source_outer_manifest,
        explicit_run=args.source_run,
    )
    target_manifest_path = resolve_outer_evaluation_manifest(
        args.project_root,
        args.target_tenure,
        explicit_manifest=args.target_outer_manifest,
        explicit_run=args.target_run,
    )
    source_outer = load_outer_evaluation_record(
        source_manifest_path,
        project_root=args.project_root,
        expected_tenure=args.source_tenure,
    )
    target_outer = load_outer_evaluation_record(
        target_manifest_path,
        project_root=args.project_root,
        expected_tenure=args.target_tenure,
    )
    _assert_matching_outer_design(source_outer, target_outer)
    native_long, native_audit = _load_target_native_oof(
        target_outer,
        target_tenure=args.target_tenure,
        explicit_path=args.target_native_outer_oof,
        project_root=args.project_root,
    )
    native_wide = _native_oof_wide(
        native_long, identity_columns=target_outer.identity_columns
    )




    target_run_reference = args.target_run
    if target_run_reference is None:
        recorded_target_run = str(
            target_outer.payload.get(
                "run_id", target_outer.payload.get("source_run_id", "")
            )
        ).strip()
        if not recorded_target_run:
            raise ValueError(
                "Target c4 manifest must record run_id when --target-run is omitted."
            )
        target_run_reference = Path(recorded_target_run)
    target_run = resolve_target_run(
        args.project_root, args.target_tenure, target_run_reference
    )
    recorded_target_run = str(
        target_outer.payload.get(
            "run_id", target_outer.payload.get("source_run_id", "")
        )
    ).strip()
    if recorded_target_run and target_run.path.name != recorded_target_run:
        raise ValueError(
            "Target c4 manifest run identity disagrees with the resolved target run."
        )
    if target_run.profile_hash != str(target_outer.payload.get("profile_hash", "")):
        raise ValueError(
            "Target c4 manifest profile hash disagrees with the resolved target run."
        )
    tenure_column = str(target_run.profile.get("tenure_dimension", "")).strip()
    if not tenure_column:
        raise ValueError("Target run profile does not record tenure_dimension.")
    required_by_fold: dict[int, set[str]] = {}
    shared_state_by_fold: dict[int, SharedStateContract] = {}
    runtime_version_rows: list[dict[str, Any]] = []
    required_features: set[str] = set(target_outer.identity_columns)
    required_features.add(target_outer.cohort_column)
    for fold in source_outer.folds:
        required, target_column, _, shared_state = _required_scoring_columns(
            fold.families
        )
        if target_column != target_outer.target_column:
            raise ValueError(
                f"Source fold {fold.fold} bundle outcome differs from c4."
            )
        required_by_fold[fold.fold] = required
        shared_state_by_fold[fold.fold] = shared_state
        required_features.update(required)
        for family in MODEL_FAMILIES:
            record = fold.families[family]
            if not args.no_verify_bundle_hashes:
                _verify_manifest_inventory(record)
            for row in _validate_bundle_runtime_versions(record):
                runtime_version_rows.append(
                    {
                        "fold": fold.fold,
                        "bundle_manifest_path": str(record.path),
                        **row,
                    }
                )
    target_path, _, target_data, target_data_audit = resolve_target_data(
        target_run,
        project_root=args.project_root,
        required_columns=required_features,
        target_column=target_outer.target_column,
        tenure_column=tenure_column,
        explicit=args.target_data,
        allow_hash_mismatch=args.allow_target_data_hash_mismatch,
    )
    target_input_contract = target_outer.payload.get("input")
    if not isinstance(target_input_contract, Mapping):
        target_input_contract = target_outer.payload.get("input_snapshot")
    target_input_sha = (
        str(target_input_contract.get("sha256", "")).strip().lower()
        if isinstance(target_input_contract, Mapping)
        else ""
    )
    if target_input_sha:
        if target_data_audit["sha256"] != target_input_sha:
            raise RuntimeError(
                "Resolved target data does not match the c4 input snapshot hash."
            )
    elif args.target_data is not None:
        raise ValueError(
            "An explicit --target-data override is not permitted unless the target "
            "c4 manifest records the immutable input snapshot SHA-256."
        )
    target_rows = _filter_target_tenure(
        target_data, tenure_column, args.target_tenure
    )
    preparation_path = target_run.path / "preparation" / "preparation_info.json"
    if not preparation_path.is_file():
        raise FileNotFoundError(
            f"Target preparation receipt is required to protect the holdout: {preparation_path}"
        )
    preparation = _json_load(preparation_path)
    final_holdout_start = pd.Timestamp(preparation["final_holdout_start"])
    if any(fold.validation_end >= final_holdout_start for fold in source_outer.folds):
        raise ValueError(
            "At least one outer validation window reaches the target final holdout."
        )




    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    lift_rows: list[pd.DataFrame] = []
    spread_rows: list[dict[str, Any]] = []
    state_rows: list[pd.DataFrame] = []
    bundle_rows: list[dict[str, Any]] = []
    for fold in source_outer.folds:
        dates = pd.to_datetime(target_rows[target_outer.cohort_column], errors="raise")
        target_fold = target_rows.loc[
            dates.between(
                fold.validation_start, fold.validation_end, inclusive="both"
            )
        ].copy()
        native_fold = native_wide.loc[native_wide["fold"].eq(fold.fold)].copy()
        if target_fold.empty or native_fold.empty:
            raise ValueError(f"Outer fold {fold.fold} has no target rows.")
        target_fold["_identity_join_key"] = _identity_join_key(
            target_fold, target_outer.identity_columns
        )
        native_fold["_identity_join_key"] = _identity_join_key(
            native_fold, target_outer.identity_columns
        )
        target_keys = set(target_fold["_identity_join_key"])
        native_keys = set(native_fold["_identity_join_key"])
        if target_keys != native_keys:
            raise ValueError(
                f"Outer fold {fold.fold} source-transfer target rows do not exactly "
                "match target-native OOF rows: "
                f"target_only={len(target_keys - native_keys)}; "
                f"native_only={len(native_keys - target_keys)}."
            )
        ordered = target_fold.set_index("_identity_join_key", verify_integrity=True).loc[
            native_fold["_identity_join_key"]
        ].reset_index()
        observed_target = pd.to_numeric(
            ordered[target_outer.target_column], errors="raise"
        ).astype(int).to_numpy()
        if not np.array_equal(
            observed_target, native_fold["y_retention"].to_numpy(dtype=int)
        ):
            raise ValueError(
                f"Outer fold {fold.fold} target outcomes disagree with native OOF."
            )
        observed_cohort = pd.to_datetime(
            ordered[target_outer.cohort_column], errors="raise"
        ).dt.to_period("D")
        native_cohort = native_fold["cohort_date"].dt.to_period("D")
        if not np.array_equal(observed_cohort.to_numpy(), native_cohort.to_numpy()):
            raise ValueError(
                f"Outer fold {fold.fold} cohort dates disagree with native OOF."
            )
        scoring_frame = ordered.loc[:, sorted(required_by_fold[fold.fold])]
        source_predictions: dict[str, np.ndarray] = {}
        for family in MODEL_FAMILIES:
            record = fold.families[family]
            source_predictions[family] = score_target_with_frozen_bundle(
                record.path,
                scoring_frame,
                verify_hashes=not args.no_verify_bundle_hashes,
            )
            bundle_rows.append(
                {
                    "fold": fold.fold,
                    "model_family": family,
                    "manifest_path": str(record.path),
                    "manifest_sha256": sha256_file(record.path),
                    "validation_start": str(fold.validation_start.date()),
                    "validation_end": str(fold.validation_end.date()),
                    "label_cutoff": str(fold.label_cutoff.date()),
                    **_manifest_identity(record.payload),
                }
            )
        state_audit = _state_mapping_audit(
            ordered,
            source_predictions,
            shared_state_by_fold[fold.fold],
            target_column=target_outer.target_column,
            source_tenure=args.source_tenure,
            target_tenure=args.target_tenure,
        )
        state_audit.insert(0, "fold", fold.fold)
        invalid_state = state_audit.loc[
            ~state_audit["mapping_status"].eq("SOURCE_MAPPING_APPLIED")
        ]
        if not invalid_state.empty:
            raise RuntimeError(
                f"Target states violate the frozen source mapping in fold {fold.fold}."
            )
        state_rows.append(state_audit)
        base_columns = [
            "fold",
            "row_id",
            "cohort_date",
            "y_retention",
            *target_outer.identity_columns,
        ]
        base_rows = native_fold.loc[:, base_columns].reset_index(drop=True)
        for family in MODEL_FAMILIES:
            for role, probabilities in (
                ("FROZEN_SOURCE_ON_TARGET", source_predictions[family]),
                (
                    "TARGET_NATIVE_OUTER_OOF",
                    native_fold[f"native_{family}_p_retention"].to_numpy(),
                ),
            ):
                part = base_rows.copy()
                part.insert(4, "model_family", family)
                part.insert(4, "model_role", role)
                part.insert(4, "target_tenure", args.target_tenure)
                part.insert(4, "source_tenure", args.source_tenure)
                part["p_retention"] = np.asarray(probabilities, dtype=float)
                part["p_churn"] = 1.0 - part["p_retention"]
                prediction_parts.append(part)
                metrics = _outer_metrics(
                    part["y_retention"].to_numpy(),
                    part["p_retention"].to_numpy(),
                    row_id=part["row_id"],
                )
                metric_rows.append(
                    {
                        "fold": fold.fold,
                        "model_family": family,
                        "model_role": role,
                        "source_tenure": args.source_tenure,
                        "target_tenure": args.target_tenure,
                        **metrics,
                    }
                )
                for bins in args.lift_bins:
                    table = build_lift_table(
                        1 - part["y_retention"].to_numpy(dtype=int),
                        part["p_churn"].to_numpy(),
                        n_bins=bins,
                        row_id=part["row_id"],
                    )
                    table.insert(0, "n_bins", bins)
                    table.insert(0, "target_tenure", args.target_tenure)
                    table.insert(0, "source_tenure", args.source_tenure)
                    table.insert(0, "model_role", role)
                    table.insert(0, "model_family", family)
                    table.insert(0, "fold", fold.fold)
                    lift_rows.append(table)
                    spread_rows.append(
                        {
                            "fold": fold.fold,
                            "model_family": family,
                            "model_role": role,
                            "source_tenure": args.source_tenure,
                            "target_tenure": args.target_tenure,
                            "n_bins": bins,
                            **summarize_lift_spread(table),
                        }
                    )




    predictions = pd.concat(prediction_parts, ignore_index=True)
    by_fold = pd.DataFrame(metric_rows)
    pooled_rows: list[dict[str, Any]] = []
    for (family, role), group in predictions.groupby(
        ["model_family", "model_role"], sort=True
    ):
        pooled_rows.append(
            {
                "model_family": family,
                "model_role": role,
                "source_tenure": args.source_tenure,
                "target_tenure": args.target_tenure,
                "outer_fold_count": int(group["fold"].nunique()),
                **_pooled_outer_metrics(group),
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    stability_rows: list[dict[str, Any]] = []
    for (family, role), group in by_fold.groupby(
        ["model_family", "model_role"], sort=True
    ):
        for metric, direction in METRIC_DIRECTION.items():
            values = pd.to_numeric(group[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            if finite.empty:
                continue
            worst_index = (
                finite.idxmax()
                if direction == "LOWER_IS_BETTER"
                else finite.abs().idxmax()
                if direction == "CLOSER_TO_ZERO_IS_BETTER"
                else finite.idxmin()
            )
            stability_rows.append(
                {
                    "stability_basis": "MODEL_METRIC_ACROSS_OUTER_FOLDS",
                    "model_family": family,
                    "model_role": role,
                    "source_tenure": args.source_tenure,
                    "target_tenure": args.target_tenure,
                    "metric": metric,
                    "direction": direction,
                    "fold_count": int(len(finite)),
                    "mean": float(finite.mean()),
                    "standard_deviation": float(finite.std(ddof=1))
                    if len(finite) > 1
                    else 0.0,
                    "minimum": float(finite.min()),
                    "maximum": float(finite.max()),
                    "worst_fold": int(group.loc[worst_index, "fold"]),
                    "worst_value": float(group.loc[worst_index, metric]),
                }
            )
    source_by_fold = by_fold.loc[
        by_fold["model_role"].eq("FROZEN_SOURCE_ON_TARGET")
    ].drop(columns="model_role")
    native_by_fold = by_fold.loc[
        by_fold["model_role"].eq("TARGET_NATIVE_OUTER_OOF")
    ].drop(columns="model_role")
    delta_rows = _metric_delta_rows(
        source_by_fold,
        native_by_fold,
        scope="OUTER_FOLD",
        join_columns=("fold", "model_family", "source_tenure", "target_tenure"),
    )
    source_pooled = pooled.loc[
        pooled["model_role"].eq("FROZEN_SOURCE_ON_TARGET")
    ].drop(columns="model_role")
    native_pooled = pooled.loc[
        pooled["model_role"].eq("TARGET_NATIVE_OUTER_OOF")
    ].drop(columns="model_role")
    delta_rows.extend(
        _metric_delta_rows(
            source_pooled,
            native_pooled,
            scope="POOLED_OUTER_OOF",
            join_columns=("model_family", "source_tenure", "target_tenure"),
        )
    )
    fold_delta_frame = pd.DataFrame(delta_rows)
    fold_delta_frame = fold_delta_frame.loc[
        fold_delta_frame["scope"].eq("OUTER_FOLD")
    ]
    for (family, metric, metric_direction), group in fold_delta_frame.groupby(
        ["model_family", "metric", "direction"], sort=True
    ):
        values = pd.to_numeric(
            group["performance_delta_positive_means_source_better"],
            errors="coerce",
        )
        finite = values[np.isfinite(values)]
        if finite.empty:
            continue
        worst_index = finite.idxmin()
        stability_rows.append(
            {
                "stability_basis": "PAIRED_SOURCE_VS_NATIVE_DELTA_ACROSS_OUTER_FOLDS",
                "model_family": family,
                "model_role": "FROZEN_SOURCE_MINUS_TARGET_NATIVE",
                "source_tenure": args.source_tenure,
                "target_tenure": args.target_tenure,
                "metric": metric,
                "direction": "POSITIVE_MEANS_SOURCE_BETTER",
                "metric_direction": metric_direction,
                "fold_count": int(len(finite)),
                "mean": float(finite.mean()),
                "standard_deviation": float(finite.std(ddof=1))
                if len(finite) > 1
                else 0.0,
                "minimum": float(finite.min()),
                "maximum": float(finite.max()),
                "worst_fold": int(group.loc[worst_index, "fold"]),
                "worst_value": float(finite.loc[worst_index]),
            }
        )




    predictions_path = output_dir / "cross_tenure_outer_oof_predictions.parquet"
    by_fold_path = output_dir / "outer_oof_metrics_by_fold.csv"
    pooled_path = output_dir / "outer_oof_metrics_pooled.csv"
    deltas_path = output_dir / "outer_oof_metric_deltas.csv"
    stability_path = output_dir / "outer_oof_metric_stability.csv"
    lift_path = output_dir / "lift_by_outer_fold_equal_count_group.csv"
    spread_path = output_dir / "lift_spread_by_outer_fold.csv"
    state_path = output_dir / "source_state_mapping_by_outer_fold.csv"
    bundles_path = output_dir / "source_outer_fold_bundle_identity.csv"
    runtime_path = output_dir / "runtime_version_validation.csv"
    chart_status_path = output_dir / "chart_generation_status.csv"
    predictions.to_parquet(predictions_path, index=False)
    by_fold.to_csv(by_fold_path, index=False)
    pooled.to_csv(pooled_path, index=False)
    pd.DataFrame(delta_rows).to_csv(deltas_path, index=False)
    pd.DataFrame(stability_rows).to_csv(stability_path, index=False)
    lift_frame = pd.concat(lift_rows, ignore_index=True)
    lift_frame.to_csv(lift_path, index=False)
    pd.DataFrame(spread_rows).to_csv(spread_path, index=False)
    pd.concat(state_rows, ignore_index=True).to_csv(state_path, index=False)
    pd.DataFrame(bundle_rows).to_csv(bundles_path, index=False)
    pd.DataFrame(runtime_version_rows).to_csv(runtime_path, index=False)


    # Restore the historical Logistic/XGBoost HTML lift charts for the governed
    # outer-OOF path.  The bars are built only from the already-computed
    # within-fold lift tables, so no pooled reranking or target-side learning is
    # introduced.  For each score group, fold lift is averaged using the
    # group's record count as the weight.  This preserves the outer-fold
    # ranking contract while giving the same interactive 3/4/5/10/20-group
    # visualization used by the earlier cross-tenure workflow.
    chart_rows: list[dict[str, Any]] = []
    for family in MODEL_FAMILIES:
        chart_path = output_dir / "charts" / f"{family}_lift_chart.html"
        if args.skip_charts:
            chart_rows.append(
                {
                    "model_family": family,
                    "model_role": "FROZEN_SOURCE_ON_TARGET",
                    "chart_type": "LIFT_CALCULATIONS",
                    "status": "SKIPPED_BY_REQUEST",
                    "path": "",
                    "detail": "",
                }
            )
            continue


        family_lifts: dict[int, pd.DataFrame] = {}
        for bins in args.lift_bins:
            source_fold_lift = lift_frame.loc[
                lift_frame["model_family"].eq(family)
                & lift_frame["model_role"].eq("FROZEN_SOURCE_ON_TARGET")
                & lift_frame["n_bins"].eq(bins)
            ].copy()
            if source_fold_lift.empty:
                continue


            def _weighted_fold_lift(group: pd.DataFrame) -> float:
                values = pd.to_numeric(group["lift"], errors="coerce").to_numpy(
                    dtype=float
                )
                weights = pd.to_numeric(
                    group["records"], errors="coerce"
                ).to_numpy(dtype=float)
                valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
                if not valid.any():
                    return np.nan
                return float(np.average(values[valid], weights=weights[valid]))


            chart_rows_for_bins: list[dict[str, Any]] = []
            for bin_rank, group in source_fold_lift.groupby(
                "bin_rank", sort=True
            ):
                chart_rows_for_bins.append(
                    {
                        "bin_rank": int(bin_rank),
                        "lift": _weighted_fold_lift(group),
                        "records": int(
                            pd.to_numeric(
                                group["records"], errors="coerce"
                            ).fillna(0).sum()
                        ),
                    }
                )
            chart_table = pd.DataFrame(chart_rows_for_bins).sort_values(
                "bin_rank", kind="mergesort"
            ).reset_index(drop=True)
            family_lifts[int(bins)] = chart_table


        if not family_lifts:
            chart_rows.append(
                {
                    "model_family": family,
                    "model_role": "FROZEN_SOURCE_ON_TARGET",
                    "chart_type": "LIFT_CALCULATIONS",
                    "status": "SKIPPED_NO_LIFT_DATA",
                    "path": "",
                    "detail": "",
                }
            )
            continue


        try:
            plot_lift_like_graph1(
                family_lifts,
                title="Lift Calculations",
                save_html_path=chart_path,
            )
        except ImportError as exc:
            chart_rows.append(
                {
                    "model_family": family,
                    "model_role": "FROZEN_SOURCE_ON_TARGET",
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
                    "model_role": "FROZEN_SOURCE_ON_TARGET",
                    "chart_type": "LIFT_CALCULATIONS",
                    "status": "WRITTEN",
                    "path": str(chart_path),
                    "detail": (
                        "Historical interactive lift chart restored for the "
                        "governed outer-OOF path; fold lifts are record-weighted "
                        "within each score group."
                    ),
                }
            )
    pd.DataFrame(chart_rows).to_csv(chart_status_path, index=False)
    _write_json(
        output_dir / "evaluation_manifest.json",
        {
            "evaluation_contract": "frozen_source_outer_fold_models_on_target_outer_oof",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": TENURE_GROUPING_PURPOSE,
            "selection_layer": "TENURE_GROUPING_MODEL_STRUCTURE",
            "decision_scope": "EVIDENCE_ONLY",
            "decision_eligible": True,
            "source_tenure": args.source_tenure,
            "target_tenure": args.target_tenure,
            "source_outer_evaluation_manifest": {
                "path": source_outer.path,
                "sha256": sha256_file(source_outer.path),
            },
            "target_outer_evaluation_manifest": {
                "path": target_outer.path,
                "sha256": sha256_file(target_outer.path),
            },
            "target_native_outer_oof": native_audit,
            "target_data": target_data_audit,
            "target_data_path": target_path,
            "outer_fold_count": len(source_outer.folds),
            "outer_validation_windows": [
                {
                    "fold": fold.fold,
                    "validation_start": fold.validation_start,
                    "validation_end": fold.validation_end,
                    "label_cutoff": fold.label_cutoff,
                }
                for fold in source_outer.folds
            ],
            "output_artifacts": {
                "predictions": {
                    "path": predictions_path,
                    "sha256": sha256_file(predictions_path),
                    "rows": len(predictions),
                },
                "metrics_by_fold": {
                    "path": by_fold_path,
                    "sha256": sha256_file(by_fold_path),
                    "rows": len(by_fold),
                },
                "metrics_pooled": {
                    "path": pooled_path,
                    "sha256": sha256_file(pooled_path),
                    "rows": len(pooled),
                },
                "paired_deltas": {
                    "path": deltas_path,
                    "sha256": sha256_file(deltas_path),
                    "rows": len(delta_rows),
                },
                "stability": {
                    "path": stability_path,
                    "sha256": sha256_file(stability_path),
                    "rows": len(stability_rows),
                },
                "lift_by_fold": {
                    "path": lift_path,
                    "sha256": sha256_file(lift_path),
                },
                "lift_spread_by_fold": {
                    "path": spread_path,
                    "sha256": sha256_file(spread_path),
                },
                "chart_generation_status": {
                    "path": chart_status_path,
                    "sha256": sha256_file(chart_status_path),
                    "rows": len(chart_rows),
                },
                "lift_charts": {
                    row["model_family"]: {
                        "status": row["status"],
                        "path": row["path"],
                    }
                    for row in chart_rows
                },
                "state_mapping_by_fold": {
                    "path": state_path,
                    "sha256": sha256_file(state_path),
                },
                "source_fold_bundles": {
                    "path": bundles_path,
                    "sha256": sha256_file(bundles_path),
                },
                "runtime_versions": {
                    "path": runtime_path,
                    "sha256": sha256_file(runtime_path),
                },
            },
            "operational_ranking_scope": "WITHIN_EACH_OUTER_FOLD",
            "target_learning_operations": 0,
            "inner_metrics_used_for_grouping": False,
            "seal_type": "LOGICAL_NO_PREDICTIONS_OR_METRICS",
            "sealed_target_scope": "FINAL_HOLDOUT_ROWS_ONLY",
            "target_snapshot_may_physically_contain_final_holdout_outcomes": True,
            "target_used_for_fit_tuning_grouping_selection_or_metrics": False,
            "final_holdout_rows_scored": 0,
            "final_holdout_start": final_holdout_start,
            "bundle_hash_verification": not args.no_verify_bundle_hashes,
            "interpretation": (
                "One source-to-target transfer cell is grouping evidence, not a "
                "complete tenure partition decision. Compare prespecified grouping "
                "architectures on the same outer folds before locking the design."
            ),
        },
    )
    print(f"Cross-tenure outer-OOF grouping evidence complete: {output_dir}")
    print("Final holdout rows scored: 0")
    return 0








def _parse_lift_bins(raw: str) -> tuple[int, ...]:
    try:
        bins = tuple(sorted({int(value.strip()) for value in raw.split(",") if value.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid --lift-bins: {exc}") from exc
    if not bins or any(value < 2 for value in bins):
        raise argparse.ArgumentTypeError("--lift-bins must contain integers of at least 2")
    return bins








def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-tenure", required=True)
    parser.add_argument("--target-tenure", required=True)
    parser.add_argument(
        "--purpose",
        choices=(TENURE_GROUPING_PURPOSE, NON_SELECTION_DIAGNOSTIC_PURPOSE),
        default=TENURE_GROUPING_PURPOSE,
        help=(
            "TENURE_GROUPING_MODEL_STRUCTURE uses only fold-matched c4 outer-OOF "
            "evidence. The "
            "legacy frozen-final-model path is available only as an explicit "
            "non-selection diagnostic."
        ),
    )
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--target-run", type=Path)
    parser.add_argument("--source-outer-manifest", type=Path)
    parser.add_argument("--target-outer-manifest", type=Path)
    parser.add_argument("--target-native-outer-oof", type=Path)
    parser.add_argument(
        "--source-bundle-manifest",
        type=Path,
        action="append",
        default=[],
        help="Optional manifest file/directory; repeat for multiple search roots.",
    )
    parser.add_argument("--target-data", type=Path)
    parser.add_argument("--lift-bins", type=_parse_lift_bins, default=DEFAULT_LIFT_BINS)
    parser.add_argument(
        "--evaluation-window",
        choices=("outer-oof", "source-holdout", "all-target", "final-test"),
        default=None,
        help=(
            "tenure-grouping permits only outer-oof. Legacy source-holdout/all-target "
            "modes require non-selection-diagnostic purpose and explicit final-test "
            "acknowledgement because they may expose the sealed holdout."
        ),
    )
    parser.add_argument(
        "--confirm-final-holdout-diagnostic",
        action="store_true",
        help=(
            "Acknowledge that an explicit non-selection legacy diagnostic can read "
            "final-holdout outcomes. Such output is never grouping-selection evidence."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-verify-bundle-hashes", action="store_true")
    parser.add_argument("--allow-target-data-hash-mismatch", action="store_true")
    parser.add_argument("--skip-charts", action="store_true")
    args = parser.parse_args(argv)
    args.project_root = args.project_root.expanduser().resolve()
    args.source_tenure = _normalize_tenure(args.source_tenure)
    args.target_tenure = _normalize_tenure(args.target_tenure)
    if not args.source_tenure or not args.target_tenure:
        parser.error("source and target tenure cannot be empty")
    if args.source_tenure == args.target_tenure:
        parser.error("source and target tenure must be different")
    if args.purpose == TENURE_GROUPING_PURPOSE:
        if args.evaluation_window not in (None, "outer-oof"):
            parser.error(
                "Tenure-grouping model-structure selection permits only "
                "--evaluation-window outer-oof. Final-test, source-holdout, and "
                "all-target outcomes must remain outside grouping selection."
            )
        if args.confirm_final_holdout_diagnostic:
            parser.error(
                "--confirm-final-holdout-diagnostic is incompatible with "
                "--purpose TENURE_GROUPING_MODEL_STRUCTURE."
            )
        if args.source_bundle_manifest:
            parser.error(
                "--source-bundle-manifest identifies legacy final bundles. Use "
                "--source-outer-manifest for tenure-grouping evidence."
            )
        if args.no_verify_bundle_hashes or args.allow_target_data_hash_mismatch:
            parser.error(
                "Tenure-grouping evidence requires exact bundle and target-data "
                "hash verification; integrity overrides are diagnostic-only."
            )
        args.evaluation_window = "outer-oof"
    else:
        if args.evaluation_window in (None, "outer-oof", "final-test"):
            if args.evaluation_window == "outer-oof":
                parser.error(
                    "Outer-OOF is the governed tenure-grouping path; use "
                    "--purpose TENURE_GROUPING_MODEL_STRUCTURE."
                )
            args.evaluation_window = "source-holdout"
        if not args.confirm_final_holdout_diagnostic:
            parser.error(
                "Legacy non-selection diagnostics may expose final-holdout outcomes; "
                "add --confirm-final-holdout-diagnostic to acknowledge this."
            )
        if any(
            value is not None
            for value in (
                args.source_outer_manifest,
                args.target_outer_manifest,
                args.target_native_outer_oof,
            )
        ):
            parser.error(
                "c4 outer-evaluation artifacts belong to --purpose "
                "TENURE_GROUPING_MODEL_STRUCTURE, not the legacy diagnostic path."
            )
    return args








def _output_directory(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        raw = args.output_dir.expanduser()
        return raw.resolve() if raw.is_absolute() else (args.project_root / raw).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    source = re.sub(r"[^A-Za-z0-9+_-]+", "_", args.source_tenure)
    target = re.sub(r"[^A-Za-z0-9+_-]+", "_", args.target_tenure)
    return args.project_root / "output" / "cross_tenure" / f"s{source}_to_t{target}_{stamp}"








def _run_non_selection_diagnostic(args: argparse.Namespace) -> int:
    """Run the legacy final-bundle diagnostic after explicit acknowledgement."""




    output_dir = _output_directory(args)
    output_dir.mkdir(parents=True, exist_ok=False)




    bundles = discover_source_bundle_manifests(
        args.project_root,
        args.source_tenure,
        explicit_manifests=args.source_bundle_manifest,
        explicit_source_run=args.source_run,
    )
    verify_hashes = not args.no_verify_bundle_hashes
    if verify_hashes:
        for record in bundles.values():
            _verify_manifest_inventory(record)




    runtime_version_rows: list[dict[str, Any]] = []
    for family in MODEL_FAMILIES:
        record = bundles[family]
        for row in _validate_bundle_runtime_versions(record):
            runtime_version_rows.append(
                {
                    "bundle_manifest_path": str(record.path),
                    **row,
                }
            )




    (
        required_features,
        target_column,
        loaded_bundles,
        shared_state,
    ) = _required_scoring_columns(bundles)
    model_scoring_features = set(required_features)
    holdout_contract = _source_holdout_contract(bundles)
    required_features.add(str(holdout_contract["cohort_column"]))
    required_features.update(
        str(value) for value in holdout_contract["identity_columns"]
    )
    target_run = resolve_target_run(args.project_root, args.target_tenure, args.target_run)
    tenure_column = str(target_run.profile.get("tenure_dimension", "")).strip()
    if not tenure_column:
        raise ValueError("Target run profile does not record tenure_dimension")
    target_path, target_identity, target_data, target_data_audit = resolve_target_data(
        target_run,
        project_root=args.project_root,
        required_columns=required_features,
        target_column=target_column,
        tenure_column=tenure_column,
        explicit=args.target_data,
        allow_hash_mismatch=args.allow_target_data_hash_mismatch,
    )
    target_tenure_rows = _filter_target_tenure(
        target_data, tenure_column, args.target_tenure
    )
    target, evaluation_month_coverage = _filter_evaluation_window(
        target_tenure_rows,
        cohort_column=str(holdout_contract["cohort_column"]),
        holdout_contract=holdout_contract,
        evaluation_window=args.evaluation_window,
    )
    target_cohort_dates = pd.to_datetime(
        target[str(holdout_contract["cohort_column"])], errors="raise"
    )
    final_holdout_rows_scored = int(
        target_cohort_dates.between(
            pd.Timestamp(holdout_contract["cohort_start"]),
            pd.Timestamp(holdout_contract["cohort_end"]),
            inclusive="both",
        ).sum()
    )
    y_retention = pd.to_numeric(target[target_column], errors="raise").astype(int).to_numpy()
    if set(np.unique(y_retention)) != {0, 1}:
        raise ValueError("Target outcome must contain both binary retention classes 0 and 1")




    scoring_columns = sorted(model_scoring_features)
    scoring_frame = target.loc[:, scoring_columns].copy()
    predictions: dict[str, np.ndarray] = {}
    for family in MODEL_FAMILIES:
        predictions[family] = score_target_with_frozen_bundle(
            bundles[family].path,
            scoring_frame,
            verify_hashes=verify_hashes,
        )




    state_audit = _state_mapping_audit(
        target,
        predictions,
        shared_state,
        target_column=target_column,
        source_tenure=args.source_tenure,
        target_tenure=args.target_tenure,
    )
    state_audit.to_csv(output_dir / "source_state_mapping_applied_to_target.csv", index=False)
    invalid_state = state_audit.loc[
        ~state_audit["mapping_status"].eq("SOURCE_MAPPING_APPLIED")
    ]
    if not invalid_state.empty:
        raise RuntimeError(
            "Target states violate the frozen source mapping. See "
            f"{output_dir / 'source_state_mapping_applied_to_target.csv'}"
        )




    row_id = _deterministic_row_id(
        target, holdout_contract["identity_columns"]
    )
    metric_rows: list[dict[str, Any]] = []
    lift_rows: list[pd.DataFrame] = []
    spread_rows: list[dict[str, Any]] = []
    chart_rows: list[dict[str, Any]] = []
    for family in MODEL_FAMILIES:
        p_retention = predictions[family]
        p_churn = 1.0 - p_retention
        metrics = _ranking_metrics(y_retention, p_retention)
        metric_rows.append(
            {
                "model_family": family,
                "model_role": "FROZEN_SOURCE_MODEL_ON_TARGET_TENURE",
                "source_tenure": args.source_tenure,
                "target_tenure": args.target_tenure,
                "source_bundle_manifest_sha256": sha256_file(bundles[family].path),
                **metrics,
            }
        )
        family_lifts: dict[int, pd.DataFrame] = {}
        for bins in args.lift_bins:
            table = build_lift_table(
                1 - y_retention,
                p_churn,
                n_bins=bins,
                row_id=row_id,
            )
            table.insert(0, "n_bins", bins)
            table.insert(0, "target_tenure", args.target_tenure)
            table.insert(0, "source_tenure", args.source_tenure)
            table.insert(0, "model_family", family)
            family_lifts[bins] = table
            lift_rows.append(table)
            spread_rows.append(
                {
                    "model_family": family,
                    "source_tenure": args.source_tenure,
                    "target_tenure": args.target_tenure,
                    "n_bins": bins,
                    **summarize_lift_spread(table),
                }
            )
        if not args.skip_charts:
            chart_path = output_dir / "charts" / f"{family}_lift_chart.html"
            try:
                plot_lift_like_graph1(
                    family_lifts,
                    title="Lift Calculations",
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
        else:
            chart_rows.append(
                {
                    "model_family": family,
                    "chart_type": "LIFT_CALCULATIONS",
                    "status": "SKIPPED_BY_REQUEST",
                    "path": "",
                    "detail": "",
                }
            )




    state_chart_path = (
        output_dir / "charts" / "shared_source_state_mapping_on_target.html"
    )
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
            plot_source_state_mapping_choropleth(
                state_audit,
                source_tenure=args.source_tenure,
                target_tenure=args.target_tenure,
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
                        "Colors are the one shared frozen source-tenure state "
                        "grouping used by Logistic and XGBoost; no target-native "
                        "grouping was created."
                    ),
                }
            )




    pd.DataFrame(metric_rows).to_csv(output_dir / "model_metrics.csv", index=False)
    pd.concat(lift_rows, ignore_index=True).to_csv(
        output_dir / "lift_by_equal_count_group.csv", index=False
    )
    pd.DataFrame(spread_rows).to_csv(output_dir / "lift_spread_summary.csv", index=False)
    pd.DataFrame(chart_rows).to_csv(output_dir / "chart_generation_status.csv", index=False)




    bundle_rows = []
    for family, record in bundles.items():
        identity = _manifest_identity(record.payload)
        bundle_rows.append(
            {
                "model_family": family,
                "manifest_path": str(record.path),
                "manifest_sha256": sha256_file(record.path),
                **identity,
            }
        )
    pd.DataFrame(bundle_rows).to_csv(output_dir / "source_bundle_identity.csv", index=False)
    pd.DataFrame(runtime_version_rows).to_csv(
        output_dir / "runtime_version_validation.csv", index=False
    )




    _write_json(
        output_dir / "evaluation_manifest.json",
        {
            "evaluation_contract": "frozen_source_model_on_target_tenure",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": NON_SELECTION_DIAGNOSTIC_PURPOSE,
            "selection_layer": None,
            "decision_scope": "DIAGNOSTIC_ONLY",
            "decision_eligible": False,
            "final_holdout_rows_scored": final_holdout_rows_scored,
            "source_tenure": args.source_tenure,
            "target_tenure": args.target_tenure,
            "source_bundles": {
                family: {
                    "manifest_path": record.path,
                    "manifest_sha256": sha256_file(record.path),
                    "identity": _manifest_identity(record.payload),
                }
                for family, record in bundles.items()
            },
            "runtime_version_validation": {
                "policy": "STRICT_EXACT_MATCH",
                "status": "PASS",
                "override_available": False,
                "audit_csv": output_dir / "runtime_version_validation.csv",
                "checks": runtime_version_rows,
            },
            "target_run": {
                "path": target_run.path,
                "profile": target_run.profile,
                "profile_hash": target_run.profile_hash,
            },
            "target_data": target_data_audit,
            "target_rows_before_filter": int(len(target_data)),
            "target_rows_after_tenure_filter": int(len(target_tenure_rows)),
            "target_rows_after_evaluation_window_filter": int(len(target)),
            "target_column": target_column,
            "tenure_column": tenure_column,
            "evaluation_window": args.evaluation_window,
            "evaluation_month_coverage": evaluation_month_coverage,
            "source_excluded_holdout": holdout_contract,
            "source_state_mapping_diagnostic": {
                "csv_path": output_dir
                / "source_state_mapping_applied_to_target.csv",
                "definition": (
                    "One shared frozen source-tenure state mapping used by Logistic "
                    "and XGBoost and applied unchanged to target-tenure states; "
                    "target outcomes are evaluation-only hover metrics."
                ),
                "mapping_sha256": shared_state.mapping_sha256,
                "categories": list(shared_state.categories),
                "derived_feature": shared_state.derived_feature,
                "source_feature": shared_state.source_feature,
                "target_native_grouping_created": False,
            },
            "charts": chart_rows,
            "lift_bins": list(args.lift_bins),
            "lift_definition": "group churn rate divided by target-population churn rate",
            "lift_chart_order": "ascending lift, matching the supplied Plotly example",
            "lift_spread_outputs": {
                "highest_minus_lowest_lift": "highest group lift minus lowest group lift",
                "highest_divided_by_lowest_lift": "highest group lift divided by lowest group lift",
                "ordered_first_to_last_lift_ratio": "highest-score group lift divided by lowest-score group lift",
            },
            "target_learning_operations": 0,
            "bundle_hash_verification": verify_hashes,
        },
    )
    print(f"Frozen cross-tenure evaluation complete: {output_dir}")
    print(
        "NON-SELECTION DIAGNOSTIC ONLY; final-holdout rows scored: "
        f"{final_holdout_rows_scored:,}"
    )
    return 0








def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.purpose == TENURE_GROUPING_PURPOSE:
        return _run_tenure_grouping_outer_oof(args)
    return _run_non_selection_diagnostic(args)








if __name__ == "__main__":
    raise SystemExit(main())
