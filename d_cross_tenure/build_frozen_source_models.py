#!/usr/bin/env python
"""Materialize immutable final models from one completed tenure-profile run.




This recovery/deployment step exists because feature-selection runs persist the
winning recipes but do not persist the fitted final estimators.  It reads one
completed source run, verifies its exact snapshot and code identities, rebuilds
the saved chronological split, removes the sealed holdout, and fits final
profile-only Logistic and XGBoost bundles.  A profile may contain one tenure or
multiple tenures.  The sealed holdout is always excluded, and this component
never accepts or reads a separate evaluation population.




All run and artifact selection is contract/content driven.  Run timestamps,
release versions, recipe filenames, and tenure values are never embedded in the
implementation.
"""




from __future__ import annotations




import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence




import numpy as np
import pandas as pd
import joblib
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression




import c_modeling.deployment_bundle as deployment_bundle_runtime
from c_modeling.deployment_bundle import (
    BUNDLE_CONTRACT,
    export_logistic_source_bundle,
    export_xgboost_source_bundle,
    fit_shared_state_artifact,
    load_bundle,
    population_sha256,
    sha256_file,
    sha256_json,
)








COMPONENT_ROLE = "frozen_source_model_builder"
PAIR_CONTRACT = "retention_source_model_pair"
PAIR_CONTRACT_REVISION = 1
SUPPORTED_FAMILIES = ("logistic", "xgboost")




LEGACY_LOGISTIC_ALL_HISTORY = {
    "history_id": "LEGACY_ALL_HISTORY",
    "kind": "legacy_all",
    "compatibility_mode": True,
    "compatibility_reason": (
        "saved Logistic recipe predates the selected_history contract"
    ),
}
CURRENT_LOGISTIC_HISTORY_SCHEMA = 15
CURRENT_LOGISTIC_HISTORY_MODE = (
    "history_tuned_grouped_sequential_floating_forward"
)
ALLOWED_RECENT_HISTORY_MONTHS = frozenset({36, 48, 60})
ALLOWED_DECAY_HALF_LIFE_MONTHS = {
    "logistic": frozenset({24, 36}),
    "xgboost": frozenset({12, 18, 24}),
}








@dataclass(frozen=True)
class JsonDocument:
    path: Path
    payload: dict[str, Any]








@dataclass(frozen=True)
class RunContracts:
    run_root: Path
    documents: tuple[JsonDocument, ...]
    profile: dict[str, Any]
    profile_hash: str
    logistic_recipe: JsonDocument | None
    logistic_run_info: JsonDocument | None
    xgboost_recipe: JsonDocument | None
    xgboost_run_info: JsonDocument | None








@dataclass(frozen=True)
class PythonIdentityCollection:
    """Exact local resolutions plus an audit record for every saved Python identity."""




    resolved_paths: dict[str, Path]
    records: tuple[dict[str, Any], ...]








def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("output") / "pipeline_runs",
        help="Root containing immutable pipeline-run directories.",
    )
    tenure_scope = parser.add_mutually_exclusive_group(required=True)
    tenure_scope.add_argument(
        "--source-tenure",
        help="Backward-compatible single-tenure source profile.",
    )
    tenure_scope.add_argument(
        "--source-tenures",
        nargs="+",
        help=(
            "Exact tenure groups in a saved pooled profile. Values may be supplied "
            "as separate arguments or comma-delimited tokens."
        ),
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        default=None,
        help=(
            "Optional exact source-run path or directory ID. If omitted, the newest "
            "uniquely completed FULL run whose saved profile exactly matches the "
            "requested source tenure group(s) is used."
        ),
    )
    parser.add_argument(
        "--families",
        default=",".join(SUPPORTED_FAMILIES),
        help="Comma-separated subset of logistic,xgboost.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional registry directory. By default the generated pair is published "
            "under output/frozen_source_models."
        ),
    )
    parser.add_argument(
        "--xgb-device",
        default=None,
        help="Explicit device override; omission reuses the saved source recipe setting.",
    )
    parser.add_argument(
        "--xgb-n-jobs",
        type=int,
        default=None,
        help="Explicit thread override; omission reuses the saved source recipe setting.",
    )
    parser.add_argument(
        "--reuse-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse an already verified content-identical pair (default: true).",
    )
    return parser.parse_args(argv)








def _normalize_tenure(value: object) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("--source-tenure cannot be empty.")
    return normalized








def _normalize_tenure_groups(values: Iterable[object]) -> tuple[str, ...]:
    groups: list[str] = []
    for value in values:
        for raw in str(value).split(","):
            normalized = _normalize_tenure(raw)
            if normalized not in groups:
                groups.append(normalized)
    if not groups:
        raise ValueError("At least one source tenure group is required.")
    return tuple(groups)








def _requested_tenure_groups(args: argparse.Namespace) -> tuple[str, ...]:
    raw = (
        [args.source_tenure]
        if args.source_tenure is not None
        else list(args.source_tenures or [])
    )
    return _normalize_tenure_groups(raw)








def _scope_label(groups: Sequence[object]) -> str:
    return ",".join(_normalize_tenure_groups(groups))








def _parse_families(value: str) -> tuple[str, ...]:
    requested: list[str] = []
    for raw in str(value).split(","):
        family = raw.strip().lower()
        if family and family not in requested:
            requested.append(family)
    invalid = sorted(set(requested).difference(SUPPORTED_FAMILIES))
    if invalid or not requested:
        raise ValueError(
            f"--families must contain {SUPPORTED_FAMILIES}; invalid={invalid}."
        )
    return tuple(requested)








def _read_json_documents(root: Path) -> tuple[JsonDocument, ...]:
    documents: list[JsonDocument] = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            documents.append(JsonDocument(path.resolve(), payload))
    return tuple(documents)








def _profile_candidates(documents: Iterable[JsonDocument]) -> list[tuple[dict[str, Any], str]]:
    candidates: list[tuple[dict[str, Any], str]] = []
    for document in documents:
        payload = document.payload
        profile = payload.get("profile")
        profile_hash = payload.get("profile_hash")
        if (
            isinstance(profile, dict)
            and isinstance(profile.get("selected_tenure_groups"), list)
            and isinstance(profile_hash, str)
            and profile_hash.strip()
        ):
            candidates.append((dict(profile), profile_hash.strip()))
        if (
            not isinstance(profile, dict)
            and isinstance(payload.get("selected_tenure_groups"), list)
            and isinstance(payload.get("profile_hash"), str)
            and all(
                key in payload
                for key in (
                    "business_scope",
                    "profile_slug",
                    "tenure_dimension",
                    "tenure_filter_mode",
                    "tenure_group_domain",
                )
            )
        ):
            profile_value = {
                key: payload[key]
                for key in (
                    "business_scope",
                    "profile_slug",
                    "selected_tenure_groups",
                    "tenure_dimension",
                    "tenure_filter_mode",
                    "tenure_group_domain",
                )
                if key in payload
            }
            candidates.append((profile_value, str(payload["profile_hash"]).strip()))
    return candidates








def _canonical_profile(
    documents: Iterable[JsonDocument], source_tenures: Sequence[str]
) -> tuple[dict[str, Any], str]:
    candidates = _profile_candidates(documents)
    requested = tuple(_normalize_tenure_groups(source_tenures))
    requested_set = set(requested)
    matching = [
        (profile, fingerprint)
        for profile, fingerprint in candidates
        if (
            len(profile.get("selected_tenure_groups", [])) == len(requested)
            and {
                _normalize_tenure(value)
                for value in profile.get("selected_tenure_groups", [])
            }
            == requested_set
        )
    ]
    if not matching:
        raise ValueError(
            "Run has no saved profile for the requested source tenure groups "
            f"{list(requested)!r}."
        )
    fingerprints = {fingerprint for _, fingerprint in matching}
    if len(fingerprints) != 1:
        raise ValueError("Run contains conflicting saved profile hashes.")
    fingerprint = fingerprints.pop()
    merged: dict[str, Any] = {}
    for profile, _ in matching:
        for key, value in profile.items():
            if key in merged and sha256_json(merged[key]) != sha256_json(value):
                raise ValueError(
                    f"Run contains conflicting saved profile field {key!r}."
                )
            merged.setdefault(key, value)
    required = (
        "business_scope",
        "profile_slug",
        "selected_tenure_groups",
        "tenure_dimension",
        "tenure_filter_mode",
        "tenure_group_domain",
    )
    richest = sorted(
        (profile for profile, _ in matching),
        key=lambda profile: sum(key in profile for key in required),
        reverse=True,
    )[0]
    canonical = {
        key: richest[key] if key in richest else merged[key]
        for key in required
        if key in richest or key in merged
    }
    missing = [key for key in required if key not in canonical]
    if missing:
        raise ValueError(f"Saved source profile is missing required fields: {missing}")
    return canonical, fingerprint








def _selected_history_contract(
    value: Any,
    *,
    family: str,
    allow_missing_legacy_logistic: bool = False,
) -> dict[str, Any]:
    """Validate a saved family history without silently enabling all history.


    Logistic recommendations created before the history experiment did not
    record a history at all.  Those recipes remain materializable through one
    explicit compatibility sentinel so existing evaluation bundles can be
    recovered.  A newly recorded ``kind=all``/``ALL_HISTORY`` strategy is
    never accepted.
    """


    if value is None:
        if family == "logistic" and allow_missing_legacy_logistic:
            return dict(LEGACY_LOGISTIC_ALL_HISTORY)
        raise ValueError(
            f"Saved {family} recipe is missing its selected_history contract."
        )
    if not isinstance(value, Mapping) or not value:
        raise ValueError(
            f"Saved {family} selected_history must be a nonempty object."
        )
    history = dict(value)
    history_id = str(history.get("history_id", "")).strip()
    kind = str(history.get("kind", "")).strip().lower()
    if not history_id:
        raise ValueError(
            f"Saved {family} selected_history has no nonempty history_id."
        )
    if kind == "all" or history_id.upper() == "ALL_HISTORY":
        raise ValueError(
            f"Saved {family} selected_history requests unweighted ALL_HISTORY, "
            "which is not permitted by the current training-history contract. "
            "Legacy Logistic recipes are supported only when selected_history "
            "is absent, not when an all-history strategy is newly recorded."
        )
    if kind not in {"recent", "decay"}:
        raise ValueError(
            f"Unsupported saved {family} selected_history contract: {history}"
        )
    raw_months = history.get("months")
    if (
        isinstance(raw_months, bool)
        or not isinstance(raw_months, (int, float))
        or not float(raw_months).is_integer()
        or int(raw_months) <= 0
    ):
        raise ValueError(
            f"Saved {family} {kind} history requires positive months: {history}"
        )
    months = int(raw_months)
    history["months"] = months
    if kind == "recent" and months not in ALLOWED_RECENT_HISTORY_MONTHS:
        raise ValueError(
            f"Saved {family} recent history must use one of "
            f"{sorted(ALLOWED_RECENT_HISTORY_MONTHS)} months; observed {months}."
        )
    if kind == "decay":
        allowed_half_lives = ALLOWED_DECAY_HALF_LIFE_MONTHS[family]
        if months not in allowed_half_lives:
            raise ValueError(
                f"Saved {family} decay half-life must use one of "
                f"{sorted(allowed_half_lives)} months; observed {months}."
            )
        if "max_months" not in history:
            raise ValueError(
                f"Saved {family} decay history is an uncapped legacy contract. "
                "Rerun selection so selected_history records max_months=60."
            )
        raw_max_months = history["max_months"]
        if (
            isinstance(raw_max_months, bool)
            or not isinstance(raw_max_months, (int, float))
            or not float(raw_max_months).is_integer()
            or int(raw_max_months) <= 0
        ):
            raise ValueError(
                f"Saved {family} decay history requires positive max_months: "
                f"{history}"
            )
        max_months = int(raw_max_months)
        history["max_months"] = max_months
        if max_months != 60:
            raise ValueError(
                f"Saved {family} decay history must use the prespecified "
                f"max_months=60 cap; observed {max_months}."
            )
    return history




def _logistic_recipe_core(
    payload: Mapping[str, Any],
    *,
    allow_missing_legacy_logistic: bool = False,
) -> dict[str, Any] | None:
    features = payload.get("features")
    if (
        payload.get("model_family") != "ridge_logistic"
        or payload.get("status") != "COMPLETE"
        or not isinstance(features, list)
        or not features
        or "C" not in payload
        or "solver" not in payload
    ):
        return None
    selected_history = _selected_history_contract(
        payload.get("selected_history"),
        family="logistic",
        allow_missing_legacy_logistic=(
            allow_missing_legacy_logistic and "selected_history" not in payload
        ),
    )
    return {
        "model_family": "ridge_logistic",
        "features": [str(value).strip().upper() for value in features],
        "C": float(payload["C"]),
        "solver": str(payload["solver"]),
        "class_weight": payload.get("class_weight"),
        "selected_history": selected_history,
    }








def _xgboost_recipe_core(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    features = payload.get("selected_features")
    if (
        not isinstance(features, list)
        or not features
        or not isinstance(payload.get("selected_model_spec"), dict)
        or not isinstance(payload.get("selected_history"), dict)
        or "recommended_boosting_rounds_from_inner_confirmation" not in payload
        or not isinstance(payload.get("seeds"), list)
        or not payload.get("seeds")
    ):
        return None
    selected_history = _selected_history_contract(
        payload["selected_history"], family="xgboost"
    )
    return {
        "features": [str(value).strip().upper() for value in features],
        "selected_model_spec": dict(payload["selected_model_spec"]),
        "selected_history": selected_history,
        "boosting_rounds": int(
            payload["recommended_boosting_rounds_from_inner_confirmation"]
        ),
        "seeds": [int(value) for value in payload["seeds"]],
    }








def _find_recipe(
    documents: Iterable[JsonDocument],
    core_reader: Callable[[Mapping[str, Any]], dict[str, Any] | None],
    label: str,
) -> JsonDocument | None:
    candidates = [doc for doc in documents if core_reader(doc.payload) is not None]
    if not candidates:
        return None
    cores = {sha256_json(core_reader(doc.payload)) for doc in candidates}
    if len(cores) != 1:
        raise ValueError(f"Run contains conflicting {label} recipes.")
    # A dedicated recipe is the smallest content-equivalent JSON contract.
    return min(candidates, key=lambda doc: (len(doc.payload), doc.path.stat().st_size))








def _is_file_identity(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("path"), str)
        and isinstance(value.get("sha256"), str)
        and len(str(value["sha256"])) == 64
    )








def _find_run_info(
    documents: Iterable[JsonDocument], family: str
) -> JsonDocument | None:
    candidates: list[JsonDocument] = []
    for document in documents:
        payload = document.payload
        if not isinstance(payload.get("profile"), dict):
            continue
        if family == "logistic":
            valid = (
                payload.get("model_family") == "ridge_logistic"
                and _is_file_identity(payload.get("data"))
                and _is_file_identity(payload.get("modeling_utils"))
                and isinstance(payload.get("outer_split_settings"), dict)
            )
        else:
            valid = (
                _is_file_identity(payload.get("input"))
                and _is_file_identity(payload.get("modeling_utils"))
                and _is_file_identity(payload.get("feature_types_file"))
                and isinstance(payload.get("arguments"), dict)
            )
        if valid:
            candidates.append(document)
    if not candidates:
        return None
    fingerprints = {
        sha256_json(
            {
                "profile": doc.payload.get("profile"),
                "profile_hash": doc.payload.get("profile_hash"),
                "data": doc.payload.get("data", doc.payload.get("input")),
                "modeling_utils": doc.payload.get("modeling_utils"),
            }
        )
        for doc in candidates
    }
    if len(fingerprints) != 1:
        raise ValueError(f"Run contains conflicting {family} run-info contracts.")
    return max(candidates, key=lambda doc: len(doc.payload))




def _allows_legacy_logistic_history(
    run_info: JsonDocument | None,
) -> bool:
    """Allow missing history only for a verifiably pre-history Logistic run."""


    if run_info is None:
        return False
    payload = run_info.payload
    raw_schema = payload.get("schema_version")
    if isinstance(raw_schema, bool):
        return False
    try:
        schema = int(raw_schema)
    except (TypeError, ValueError):
        return False
    if schema >= CURRENT_LOGISTIC_HISTORY_SCHEMA:
        return False
    return str(payload.get("mode", "")).strip() != CURRENT_LOGISTIC_HISTORY_MODE








def inspect_run(
    run_root: Path, source_tenures: Sequence[str], families: Sequence[str]
) -> RunContracts:
    documents = _read_json_documents(run_root)
    if not documents:
        raise ValueError(f"No readable JSON contracts found under {run_root}.")
    profile, profile_hash = _canonical_profile(documents, source_tenures)
    logistic_run_info = _find_run_info(documents, "logistic")
    xgboost_run_info = _find_run_info(documents, "xgboost")
    allow_legacy_logistic = _allows_legacy_logistic_history(logistic_run_info)
    logistic_recipe = _find_recipe(
        documents,
        lambda payload: _logistic_recipe_core(
            payload,
            allow_missing_legacy_logistic=allow_legacy_logistic,
        ),
        "Logistic",
    )
    xgboost_recipe = _find_recipe(documents, _xgboost_recipe_core, "XGBoost")
    missing: list[str] = []
    if "logistic" in families and (logistic_recipe is None or logistic_run_info is None):
        missing.append("complete Logistic recipe/run-info")
    if "xgboost" in families and (xgboost_recipe is None or xgboost_run_info is None):
        missing.append("complete XGBoost recipe/run-info")
    if missing:
        raise ValueError(f"Run is not materializable: missing {', '.join(missing)}.")
    return RunContracts(
        run_root=run_root.resolve(),
        documents=documents,
        profile=profile,
        profile_hash=profile_hash,
        logistic_recipe=logistic_recipe,
        logistic_run_info=logistic_run_info,
        xgboost_recipe=xgboost_recipe,
        xgboost_run_info=xgboost_run_info,
    )








def _completed_full_run_timestamp(contracts: RunContracts) -> pd.Timestamp:
    candidates: list[pd.Timestamp] = []
    expected_groups = tuple(
        _normalize_tenure_groups(contracts.profile["selected_tenure_groups"])
    )
    for document in contracts.documents:
        payload = document.payload
        if str(payload.get("run_id", "")).strip() != contracts.run_root.name:
            continue
        if str(payload.get("status", "")).strip().upper() != "COMPLETE":
            continue
        if str(payload.get("population_scope_mode", "")).strip().upper() != "FULL":
            continue
        observed_groups = payload.get("selected_tenure_groups")
        if not isinstance(observed_groups, list):
            continue
        if tuple(_normalize_tenure_groups(observed_groups)) != expected_groups:
            continue
        if str(payload.get("profile_hash", "")).strip() != contracts.profile_hash:
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
            "Run requires one unambiguous COMPLETE/FULL completion timestamp; "
            f"observed={[str(value) for value in candidates]}."
        )
    return candidates[0]








def _resolve_run_root(
    *,
    project_root: Path,
    runs_root: Path,
    source_run: Path | None,
    source_tenures: Sequence[str],
    families: Sequence[str],
) -> RunContracts:
    resolved_runs = runs_root if runs_root.is_absolute() else project_root / runs_root
    resolved_runs = resolved_runs.expanduser().resolve()
    if source_run is not None:
        direct = source_run.expanduser()
        if not direct.is_absolute():
            project_candidate = (project_root / direct).resolve()
            id_candidate = (resolved_runs / direct).resolve()
            direct = project_candidate if project_candidate.is_dir() else id_candidate
        direct = direct.resolve()
        if not direct.is_dir():
            raise FileNotFoundError(f"Source run is not a directory: {direct}")
        contracts = inspect_run(direct, source_tenures, families)
        _completed_full_run_timestamp(contracts)
        return contracts
    if not resolved_runs.is_dir():
        raise FileNotFoundError(f"Pipeline-runs root not found: {resolved_runs}")
    candidates: list[RunContracts] = []
    errors: list[str] = []
    for child in sorted(path for path in resolved_runs.iterdir() if path.is_dir()):
        try:
            candidate = inspect_run(child, source_tenures, families)
            _completed_full_run_timestamp(candidate)
            candidates.append(candidate)
        except (OSError, ValueError) as exc:
            errors.append(f"{child.name}: {exc}")
    if not candidates:
        raise FileNotFoundError(
            "No completed run for source tenure groups "
            f"{list(source_tenures)!r} provides {families}. "
            f"Inspected {len(errors)} run(s)."
        )
    completion = {
        item.run_root: _completed_full_run_timestamp(item) for item in candidates
    }
    latest = max(completion.values())
    winners = [item for item in candidates if completion[item.run_root] == latest]
    if len(winners) != 1:
        raise RuntimeError(
            "Multiple completed runs share the latest metadata completion time; "
            "pass --source-run explicitly. candidates="
            f"{[str(item.run_root) for item in winners]}"
        )
    return winners[0]








def _descriptor_suffix_candidates(raw_path: str, run_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for parts in (Path(raw_path).parts, PureWindowsPath(raw_path).parts):
        normalized = [str(value).rstrip("\\/") for value in parts]
        positions = [
            index
            for index, value in enumerate(normalized)
            if value.casefold() == run_root.name.casefold()
        ]
        for position in positions:
            suffix = normalized[position + 1 :]
            if suffix:
                candidates.append(run_root.joinpath(*suffix))
    return candidates








def _verify_descriptor(path: Path, descriptor: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    expected_size = descriptor.get("size")
    if expected_size is not None and int(path.stat().st_size) != int(expected_size):
        return False
    return sha256_file(path) == str(descriptor["sha256"]).lower()








def _recorded_python_stem(raw_path: str) -> str:
    """Return the logical module stem without depending on the host OS."""




    windows_stem = PureWindowsPath(raw_path).stem
    native_stem = Path(raw_path).stem
    return str(windows_stem or native_stem).strip()








def _path_priority(path: Path, *, run_root: Path, project_root: Path) -> tuple[Any, ...]:
    resolved = path.resolve()
    for priority, root in enumerate((run_root.resolve(), project_root.resolve())):
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        return (priority, len(relative.parts), str(relative).casefold())
    return (2, len(resolved.parts), str(resolved).casefold())








def _source_cache_path(
    *, descriptor: Mapping[str, Any], project_root: Path, raw_path: str
) -> Path:
    """Use a short content-addressed locator to avoid Windows MAX_PATH failures."""




    expected_sha = str(descriptor["sha256"]).lower()
    recorded_name = PureWindowsPath(raw_path).name or Path(raw_path).name
    cache_dir = (
        project_root
        / "output"
        / "frozen_source_models"
        / "_src"
        / expected_sha[:20]
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / recorded_name








def _publish_verified_source_bytes(
    data: bytes,
    *,
    descriptor: Mapping[str, Any],
    project_root: Path,
    raw_path: str,
) -> Path | None:
    expected_size = descriptor.get("size")
    expected_sha = str(descriptor["sha256"]).lower()
    if expected_size is not None and len(data) != int(expected_size):
        return None
    if hashlib.sha256(data).hexdigest() != expected_sha:
        return None
    destination = _source_cache_path(
        descriptor=descriptor, project_root=project_root, raw_path=raw_path
    )
    if destination.exists():
        if _verify_descriptor(destination, descriptor):
            return destination.resolve()
        raise RuntimeError(
            "Content-addressed source cache collision at " f"{destination}."
        )
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=destination.parent, prefix=".source_", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    if not _verify_descriptor(destination, descriptor):
        raise RuntimeError(f"Published source identity failed verification: {destination}")
    return destination.resolve()








def _recover_exact_newline_variant(
    candidate: Path,
    *,
    descriptor: Mapping[str, Any],
    project_root: Path,
    raw_path: str,
) -> Path | None:
    """Recover exact recorded bytes after transparent Git/Library newline conversion."""




    try:
        original = candidate.read_bytes()
    except OSError:
        return None
    if b"\x00" in original:
        return None
    canonical_lf = original.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    variants = (canonical_lf, canonical_lf.replace(b"\n", b"\r\n"))
    for data in variants:
        recovered = _publish_verified_source_bytes(
            data,
            descriptor=descriptor,
            project_root=project_root,
            raw_path=raw_path,
        )
        if recovered is not None:
            return recovered
    return None








def _descriptor_project_relatives(raw_path: str, project_root: Path) -> tuple[Path, ...]:
    candidates: list[Path] = []
    anchor = project_root.name.casefold()
    for path_class in (Path, PureWindowsPath):
        parsed = path_class(raw_path)
        normalized = [str(value).rstrip("\\/") for value in parsed.parts]
        for index, value in enumerate(normalized):
            if value.casefold() == anchor and normalized[index + 1 :]:
                candidates.append(Path(*normalized[index + 1 :]))
        if not parsed.is_absolute() and normalized:
            candidates.append(Path(*normalized))
    unique: dict[str, Path] = {}
    for candidate in candidates:
        normalized_key = candidate.as_posix().casefold()
        if normalized_key not in {"", "."}:
            unique.setdefault(normalized_key, candidate)
    return tuple(unique[key] for key in sorted(unique))








def _recover_from_local_git(
    descriptor: Mapping[str, Any],
    *,
    project_root: Path,
    raw_path: str,
) -> Path | None:
    """Recover an exact historical source blob from the local repository only."""




    try:
        root_result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if root_result.returncode != 0:
        return None
    try:
        git_root = Path(os.fsdecode(root_result.stdout).strip()).resolve()
        project_prefix = project_root.resolve().relative_to(git_root)
    except (OSError, ValueError):
        return None




    seen_commits: set[str] = set()
    for project_relative in _descriptor_project_relatives(raw_path, project_root):
        repository_relative = (project_prefix / project_relative).as_posix()
        commit_ids: list[str] = []
        for history_args in (
            ["log", "--all", "--format=%H", "--", repository_relative],
            ["log", "--follow", "--format=%H", "--", repository_relative],
        ):
            result = subprocess.run(
                ["git", "-C", str(git_root), *history_args],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode == 0:
                commit_ids.extend(os.fsdecode(result.stdout).splitlines())
        for commit_id in commit_ids:
            commit_id = commit_id.strip()
            if not commit_id or commit_id in seen_commits:
                continue
            seen_commits.add(commit_id)
            blob = subprocess.run(
                [
                    "git",
                    "-C",
                    str(git_root),
                    "show",
                    "--no-textconv",
                    f"{commit_id}:{repository_relative}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if blob.returncode != 0:
                continue
            recovered = _publish_verified_source_bytes(
                blob.stdout,
                descriptor=descriptor,
                project_root=project_root,
                raw_path=raw_path,
            )
            if recovered is not None:
                return recovered
            # Git commonly normalizes text blobs to LF. Reconstruct CRLF only
            # when the resulting bytes exactly satisfy the recorded identity.
            canonical_lf = blob.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            recovered = _publish_verified_source_bytes(
                canonical_lf.replace(b"\n", b"\r\n"),
                descriptor=descriptor,
                project_root=project_root,
                raw_path=raw_path,
            )
            if recovered is not None:
                return recovered
    return None








def resolve_identity_file(
    descriptor: Mapping[str, Any], *, run_root: Path, project_root: Path
) -> Path:
    if not _is_file_identity(descriptor):
        raise TypeError(f"Invalid saved file identity: {descriptor}")
    raw_path = str(descriptor["path"])
    suffix = (PureWindowsPath(raw_path).suffix or Path(raw_path).suffix).lower()
    direct_candidates = [Path(raw_path), *_descriptor_suffix_candidates(raw_path, run_root)]
    for candidate in direct_candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if _verify_descriptor(resolved, descriptor):
            return resolved
        if suffix == ".py" and resolved.is_file():
            recovered = _recover_exact_newline_variant(
                resolved,
                descriptor=descriptor,
                project_root=project_root,
                raw_path=raw_path,
            )
            if recovered is not None:
                return recovered
    expected_size = descriptor.get("size")
    dynamic_name = PureWindowsPath(raw_path).name or Path(raw_path).name
    search_roots = [run_root, project_root]
    seen: set[Path] = set()
    matches: list[Path] = []
    for root in search_roots:
        for candidate in root.rglob(dynamic_name):
            resolved = candidate.resolve()
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            if _verify_descriptor(resolved, descriptor):
                matches.append(resolved)
                continue
            if suffix == ".py":
                recovered = _recover_exact_newline_variant(
                    resolved,
                    descriptor=descriptor,
                    project_root=project_root,
                    raw_path=raw_path,
                )
                if recovered is not None:
                    matches.append(recovered)
    if matches:
        return min(
            set(matches),
            key=lambda path: _path_priority(
                path, run_root=run_root, project_root=project_root
            ),
        )




    # A saved source may have been renamed into an archive. Search by exact
    # content rather than basename; size is only a prefilter, never identity.
    if suffix == ".py":
        for root in search_roots:
            try:
                candidates = root.rglob("*.py")
                for candidate in candidates:
                    resolved = candidate.resolve()
                    if resolved in seen or not resolved.is_file():
                        continue
                    seen.add(resolved)
                    if expected_size is None or resolved.stat().st_size == int(expected_size):
                        if _verify_descriptor(resolved, descriptor):
                            matches.append(resolved)
                            continue
                    recovered = _recover_exact_newline_variant(
                        resolved,
                        descriptor=descriptor,
                        project_root=project_root,
                        raw_path=raw_path,
                    )
                    if recovered is not None:
                        matches.append(recovered)
            except OSError:
                continue
        if matches:
            return min(
                set(matches),
                key=lambda path: _path_priority(
                    path, run_root=run_root, project_root=project_root
                ),
            )
        recovered = _recover_from_local_git(
            descriptor, project_root=project_root, raw_path=raw_path
        )
        if recovered is not None:
            return recovered




    raise FileNotFoundError(
        "Could not resolve saved artifact identity "
        f"path={raw_path!r}, size={descriptor.get('size')}, "
        f"sha256={descriptor['sha256']}; no exact verified bytes were found."
    )








def _iter_file_identities(
    value: Any, *, prefix: str = "contract"
) -> Iterable[tuple[str, dict[str, Any]]]:
    if _is_file_identity(value):
        yield prefix, dict(value)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_file_identities(item, prefix=f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_file_identities(item, prefix=f"{prefix}[{index}]")








def _collect_python_identities(
    payload: Mapping[str, Any],
    *,
    run_root: Path,
    project_root: Path,
    provenance_only_roles: Sequence[str] = ("script",),
) -> PythonIdentityCollection:
    """Resolve executed dependencies exactly and retain generator-code provenance."""




    output: dict[str, Path] = {}
    records: list[dict[str, Any]] = []
    provenance_roles = {str(value).strip().casefold() for value in provenance_only_roles}
    for label, descriptor in _iter_file_identities(payload):
        raw_path = str(descriptor["path"])
        suffix = (PureWindowsPath(raw_path).suffix or Path(raw_path).suffix).lower()
        if suffix != ".py":
            continue
        logical_stem = _recorded_python_stem(raw_path)
        terminal_role = label.rsplit(".", 1)[-1].casefold()
        provenance_only = terminal_role in provenance_roles
        identity_label = f"{label}:{logical_stem}"
        base_record: dict[str, Any] = {
            "identity_label": identity_label,
            "logical_stem": logical_stem,
            "recorded_role": terminal_role,
            "execution_role": (
                "provenance_only" if provenance_only else "runtime_dependency"
            ),
            "required_for_runtime": not provenance_only,
            "sha256": str(descriptor["sha256"]).lower(),
            "size": (
                None if descriptor.get("size") is None else int(descriptor["size"])
            ),
        }
        try:
            resolved = resolve_identity_file(
                descriptor, run_root=run_root, project_root=project_root
            )
        except FileNotFoundError as exc:
            if not provenance_only:
                raise FileNotFoundError(
                    "Required runtime Python identity is unavailable: "
                    f"role={terminal_role!r}, logical_module={logical_stem!r}, "
                    f"size={descriptor.get('size')}, sha256={descriptor['sha256']}. "
                    "An exact local, newline-reconstructed, or local-Git copy is required."
                ) from exc
            records.append(
                {
                    **base_record,
                    "resolution_status": "recorded_identity_not_locally_available",
                    "resolved_path": None,
                }
            )
            continue
        output[identity_label] = resolved
        records.append(
            {
                **base_record,
                "resolution_status": "exact_bytes_resolved",
                "resolved_path": str(resolved),
            }
        )
    return PythonIdentityCollection(output, tuple(records))








def _runtime_versions(*, include_xgboost: bool) -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    if include_xgboost:
        import xgboost as xgb




        versions["xgboost"] = xgb.__version__
    return versions








def _require_recorded_runtime_version(
    *,
    source_contract: Mapping[str, Any],
    contract_key: str,
    installed_version: str,
    package_label: str,
) -> None:
    """Fail closed when final materialization cannot match the source runtime.




    The feature-selection run did not save a final estimator, so this recovery
    operation must rebuild it.  Rebuilding with a different learning-library
    version could change preprocessing, optimization, or serialization even
    when every seed and parameter is identical.
    """




    saved = str(source_contract.get(contract_key, "")).strip()
    if not saved:
        raise ValueError(
            f"The source run did not record {package_label} under "
            f"{contract_key!r}; exact source-model materialization is unsafe."
        )
    observed = str(installed_version).strip()
    if saved != observed:
        raise RuntimeError(
            f"{package_label} version mismatch: source run={saved}; "
            f"current environment={observed}. Activate the source run's "
            "recorded environment before materializing the bundle."
        )








def _unique_module_path(
    paths: Mapping[str, Path], module_stem: str
) -> Path | None:
    """Resolve a runtime module by its recorded logical name, not archive filename."""




    matches = sorted(
        {
            path.resolve()
            for label, path in paths.items()
            if (
                label.rsplit(":", 1)[-1]
                if ":" in label
                else path.stem
            )
            == module_stem
        }
    )
    if len(matches) > 1:
        identities = {sha256_file(path) for path in matches}
        if len(identities) > 1:
            raise ValueError(
                f"Conflicting saved identities found for module {module_stem}: "
                f"{[str(path) for path in matches]}."
            )
        # Same content in multiple recorded locations; use deterministic path.
    return matches[0] if matches else None








def _load_feature_types(
    path: Path,
    selected_features: Sequence[str],
    *,
    derived_feature_types: Mapping[str, str] | None = None,
) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    mappings: list[dict[str, str]] = []
    allowed_derived = {
        str(feature).strip().upper(): str(kind).strip().lower()
        for feature, kind in dict(derived_feature_types or {}).items()
    }
    if any(kind not in {"continuous", "discrete"} for kind in allowed_derived.values()):
        raise ValueError("Derived feature-type contract contains an invalid type.")




    def visit(value: Any) -> None:
        if isinstance(value, dict):
            normalized = {
                str(key).strip().upper(): str(item).strip().lower()
                for key, item in value.items()
                if isinstance(item, str)
                and str(item).strip().lower() in {"continuous", "discrete"}
            }
            if normalized:
                mappings.append(normalized)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)




    visit(payload)
    if not mappings:
        raise ValueError(f"No feature-type mapping found in {path}.")
    selected = [str(value).strip().upper() for value in selected_features]
    augmented_mappings: list[dict[str, str]] = []
    for mapping in mappings:
        augmented = dict(mapping)
        for feature, kind in allowed_derived.items():
            previous = augmented.setdefault(feature, kind)
            if previous != kind:
                raise ValueError(
                    f"Conflicting prepared/derived types for {feature} in {path}."
                )
        augmented_mappings.append(augmented)
    eligible = [
        mapping for mapping in augmented_mappings
        if set(selected).issubset(mapping)
    ]
    if not eligible:
        union: dict[str, str] = {}
        for mapping in augmented_mappings:
            for feature, kind in mapping.items():
                previous = union.setdefault(feature, kind)
                if previous != kind:
                    raise ValueError(f"Conflicting types for {feature} in {path}.")
        eligible = [union] if set(selected).issubset(union) else []
    if not eligible:
        missing = sorted(
            set(selected).difference(
                set().union(*(set(mapping) for mapping in augmented_mappings))
            )
        )
        raise ValueError(f"Feature-type contract is missing selected features: {missing}")
    best = max(eligible, key=len)
    return {feature: best[feature] for feature in selected}








def _saved_holdout_contract(documents: Iterable[JsonDocument]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for document in documents:
        payload = document.payload
        if all(key in payload for key in ("row_count", "cohort_start", "cohort_end")):
            rows = payload.get("row_count")
            start = payload.get("cohort_start")
            end = payload.get("cohort_end")
        elif all(
            key in payload
            for key in (
                "final_holdout_rows_identified",
                "final_holdout_start",
                "final_holdout_end",
            )
        ):
            rows = payload.get("final_holdout_rows_identified")
            start = payload.get("final_holdout_start")
            end = payload.get("final_holdout_end")
        else:
            continue
        try:
            normalized = {
                "rows": int(rows),
                "cohort_start": str(pd.Timestamp(start).date()),
                "cohort_end": str(pd.Timestamp(end).date()),
            }
        except (TypeError, ValueError):
            continue
        if normalized["rows"] > 0:
            candidates.append(normalized)
    unique = {json.dumps(value, sort_keys=True) for value in candidates}
    if len(unique) != 1:
        raise ValueError(
            "A unique saved final-holdout seal is required; "
            f"observed={sorted(unique)}."
        )
    return json.loads(unique.pop())








@contextmanager
def _saved_profile_environment(profile: Mapping[str, Any], profile_hash: str):
    assignments = {
        "RETENTION_BUSINESS_SCOPE": str(profile["business_scope"]),
        "RETENTION_TENURE_GROUPS": ",".join(
            str(value) for value in profile["selected_tenure_groups"]
        ),
        "RETENTION_TENURE_FILTER_MODE": str(profile["tenure_filter_mode"]),
        "RETENTION_PROFILE_SLUG": str(profile["profile_slug"]),
        "RETENTION_PROFILE_HASH": str(profile_hash),
        "RETENTION_PROFILE_PAYLOAD_JSON": json.dumps(
            dict(profile), sort_keys=True, separators=(",", ":")
        ),
    }
    legacy = (
        "RETENTION_TENURE_MIN",
        "RETENTION_TENURE_MAX",
        "RETENTION_TENURE_VALUES",
    )
    previous = {key: os.environ.get(key) for key in (*assignments, *legacy)}
    try:
        for key in legacy:
            os.environ.pop(key, None)
        os.environ.update(assignments)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value








@contextmanager
def _isolated_profile_modules():
    """Prevent current/target profile modules from contaminating source fitting."""




    exact = {
        "params",
        "tenure_profile",
        "func.tenure_profile",
        "func.retention_population",
        "state_grouping",
        "c_modeling.state_grouping",
        "c_modeling.modeling_utils",
    }
    saved = {name: module for name, module in sys.modules.items() if name in exact}
    parent_attributes: dict[tuple[str, str], tuple[bool, Any]] = {}
    for parent_name, attribute in (
        ("func", "tenure_profile"),
        ("func", "retention_population"),
        ("c_modeling", "state_grouping"),
        ("c_modeling", "modeling_utils"),
    ):
        parent = sys.modules.get(parent_name)
        if parent is not None:
            parent_attributes[(parent_name, attribute)] = (
                hasattr(parent, attribute),
                getattr(parent, attribute, None),
            )
    for name in saved:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in exact:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        for (parent_name, attribute), (existed, value) in parent_attributes.items():
            parent = sys.modules.get(parent_name)
            if parent is None:
                continue
            if existed:
                setattr(parent, attribute, value)
            elif hasattr(parent, attribute):
                delattr(parent, attribute)








@contextmanager
def _working_directory(path: Path):
    """Use ``path`` as both the working directory and an import root.




    When this builder is launched from ``d_evaluation``, Python sets
    ``sys.path[0]`` to that subdirectory. Changing the process working
    directory alone does not make project packages such as ``c_modeling`` and
    ``func`` importable, so temporarily place the verified project root at the
    front of ``sys.path`` as well.
    """




    resolved = path.expanduser().resolve()
    previous = Path.cwd()
    import_root = str(resolved)
    added_to_import_path = import_root not in sys.path
    if added_to_import_path:
        sys.path.insert(0, import_root)
    os.chdir(resolved)
    try:
        yield
    finally:
        os.chdir(previous)
        if added_to_import_path:
            try:
                sys.path.remove(import_root)
            except ValueError:
                pass








def _import_module(path: Path, role: str) -> Any:
    identity = sha256_file(path)
    name = f"_{role}_{identity[:16]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {role} from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module








def _import_module_aliases(path: Path, aliases: Sequence[str]) -> Any:
    normalized = [str(value).strip() for value in aliases if str(value).strip()]
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError("Module aliases must be nonempty and unique.")
    spec = importlib.util.spec_from_file_location(normalized[0], path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import saved module from {path}.")
    module = importlib.util.module_from_spec(spec)
    for alias in normalized:
        sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
        for alias in normalized:
            if "." not in alias:
                continue
            parent_name, attribute = alias.rsplit(".", 1)
            parent = importlib.import_module(parent_name)
            setattr(parent, attribute, module)
    except Exception:
        for alias in normalized:
            if sys.modules.get(alias) is module:
                sys.modules.pop(alias, None)
        raise
    return module








def _verify_active_module(alias: str, expected_path: Path) -> None:
    module = sys.modules.get(alias)
    observed_file = getattr(module, "__file__", None) if module is not None else None
    if observed_file is None:
        raise ImportError(f"Expected saved module alias is not active: {alias}")
    observed_path = Path(observed_file).resolve()
    if sha256_file(observed_path) != sha256_file(expected_path):
        raise ValueError(
            f"Active module {alias} differs from its saved identity: "
            f"observed={observed_path}; expected={expected_path}."
        )








def _outer_settings(run_info: Mapping[str, Any]) -> dict[str, Any]:
    value = run_info.get("outer_split_settings")
    if not isinstance(value, dict):
        raise ValueError("Logistic run-info has no saved outer split settings.")
    required = ("folds", "validation_months", "holdout_months", "minimum_train_months")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"Saved outer split settings are missing: {missing}")
    return {
        "rolling_validation_folds": int(value["folds"]),
        "validation_months": int(value["validation_months"]),
        "test_months": int(value["holdout_months"]),
        "min_train_months": int(value["minimum_train_months"]),
        "strict_dataset_contract": bool(value.get("strict_dataset_contract", True)),
    }








def _index_sha256(values: Iterable[Any]) -> str:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="raise").to_numpy(dtype=np.int64)
    numeric.sort()
    return hashlib.sha256(numeric.astype("<i8", copy=False).tobytes()).hexdigest()








def _prepare_source_population(
    *,
    modeling_utils: Any,
    data_path: Path,
    run_info: Mapping[str, Any],
    outer_contract_path: Path,
    saved_holdout: Mapping[str, Any],
    profile: Mapping[str, Any],
    source_tenures: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], Any]:
    raw = modeling_utils.read_table(data_path)
    prepared, folds, holdout_idx, _, config = modeling_utils.load_project_population_and_folds(
        raw, **_outer_settings(run_info)
    )
    del raw
    actual_outer = modeling_utils.fold_definitions_frame(
        prepared,
        folds,
        cohort_col=config.cohort_date_col,
        target_check_col=config.target_check_col,
        target_col=config.target_col,
    )
    modeling_utils.verify_fold_definition_contract(
        actual_outer,
        outer_contract_path,
        label="source-model materialization outer folds",
    )
    holdout_index = np.asarray(holdout_idx)
    if len(holdout_index) != int(saved_holdout["rows"]):
        raise ValueError(
            "Reconstructed holdout row count differs from its seal: "
            f"{len(holdout_index)} != {saved_holdout['rows']}."
        )
    holdout = prepared.loc[holdout_index].copy()
    holdout_dates = pd.to_datetime(holdout[config.cohort_date_col], errors="raise")
    observed_dates = (str(holdout_dates.min().date()), str(holdout_dates.max().date()))
    expected_dates = (saved_holdout["cohort_start"], saved_holdout["cohort_end"])
    if observed_dates != expected_dates:
        raise ValueError(
            f"Reconstructed holdout dates {observed_dates} differ from seal {expected_dates}."
        )
    pre_holdout = prepared.drop(index=holdout_index).copy()
    if pre_holdout.empty or np.intersect1d(pre_holdout.index, holdout_index).size:
        raise AssertionError("The sealed holdout was not fully excluded.")
    saved_label_buffer = run_info["outer_split_settings"].get(
        "label_availability_buffer_days"
    )
    if saved_label_buffer is None:
        raise ValueError(
            "Saved source split contract is missing label_availability_buffer_days."
        )
    label_buffer_days = int(config.label_availability_buffer_days)
    if label_buffer_days != int(saved_label_buffer):
        raise ValueError(
            "Runtime/source label-availability buffer differs from the saved run: "
            f"runtime={label_buffer_days}; saved={saved_label_buffer}."
        )
    label_cutoff = pd.Timestamp(saved_holdout["cohort_start"]) - pd.Timedelta(
        days=label_buffer_days
    )
    target_check = pd.to_datetime(
        pre_holdout[config.target_check_col], errors="raise"
    )
    label_available = target_check.lt(label_cutoff)
    development = pre_holdout.loc[label_available].copy()
    embargoed = pre_holdout.loc[~label_available].copy()
    if development.empty:
        raise ValueError(
            "The saved holdout/label-availability boundary produced an empty "
            "source evaluation-training population."
        )
    development_target = pd.to_numeric(
        development[config.target_col], errors="raise"
    ).astype(int)
    if set(development_target.unique()) != {0, 1}:
        raise ValueError(
            "Source evaluation-training population must contain both target classes."
        )
    if not pd.to_datetime(
        development[config.target_check_col], errors="raise"
    ).lt(label_cutoff).all():
        raise AssertionError("A label-unavailable row entered source final fitting.")
    tenure_dimension = str(profile["tenure_dimension"])
    if tenure_dimension not in prepared:
        raise KeyError(f"Saved tenure dimension is missing: {tenure_dimension}")
    expected_tenures = set(_normalize_tenure_groups(source_tenures))




    def verify_tenure_scope(
        frame: pd.DataFrame, *, label: str, require_all_groups: bool
    ) -> None:
        values = frame[tenure_dimension].astype("string")
        stripped = values.str.strip()
        missing = values.isna() | stripped.isin(
            {"", "<NA>", "NAN", "NONE", "NULL"}
        )
        if missing.any():
            raise ValueError(
                f"{label} has {int(missing.sum())} missing tenure value(s); "
                "missing tenure rows cannot enter a frozen model population."
            )
        observed = {
            _normalize_tenure(value) for value in stripped.unique().tolist()
        }
        unexpected = observed.difference(expected_tenures)
        if unexpected or (require_all_groups and observed != expected_tenures):
            raise ValueError(
                f"{label} violates the saved tenure scope: "
                f"observed={sorted(observed)}; expected={list(source_tenures)}."
            )




    verify_tenure_scope(prepared, label="Prepared source population", require_all_groups=True)
    verify_tenure_scope(holdout, label="Sealed source holdout", require_all_groups=True)
    verify_tenure_scope(pre_holdout, label="Pre-holdout source population", require_all_groups=True)
    verify_tenure_scope(development, label="Source development population", require_all_groups=True)
    if not embargoed.empty:
        verify_tenure_scope(
            embargoed,
            label="Label-embargoed source population",
            require_all_groups=False,
        )
    identity_columns = [
        value
        for value in (
            getattr(config, "policy_id_col", None),
            getattr(config, "anchor_term_start_col", None),
            getattr(config, "cohort_date_col", None),
        )
        if isinstance(value, str) and value in prepared
    ]
    policy_cluster_column = getattr(config, "policy_id_col", None)
    if (
        not isinstance(policy_cluster_column, str)
        or policy_cluster_column not in identity_columns
    ):
        raise ValueError(
            "The saved population contract does not expose its policy-cluster "
            "column as an identity field."
        )
    excluded_holdout = {
        **dict(saved_holdout),
        "row_index_sha256": _index_sha256(holdout_index),
        "population_sha256": population_sha256(holdout, identity_columns),
        "identity_columns": identity_columns,
        "policy_cluster_column": policy_cluster_column,
        "identity_column_roles": {
            "policy_cluster": policy_cluster_column,
            "anchor_term_start": getattr(config, "anchor_term_start_col", None),
            "cohort_date": getattr(config, "cohort_date_col", None),
        },
        "target_scored": False,
        "used_for_preprocessing_or_fit": False,
        "label_availability_buffer_days": label_buffer_days,
        "final_training_label_cutoff_exclusive": str(label_cutoff),
        "pre_holdout_label_unavailable_rows_excluded": int(len(embargoed)),
    }
    development_dates = pd.to_datetime(
        development[config.cohort_date_col], errors="raise"
    )
    evaluation_training_population = {
        "definition": (
            "non-holdout source rows with target-check date strictly before "
            "the holdout-start label-availability cutoff"
        ),
        "rows": int(len(development)),
        "cohort_start": str(development_dates.min()),
        "cohort_end": str(development_dates.max()),
        "target_check_max": str(
            pd.to_datetime(
                development[config.target_check_col], errors="raise"
            ).max()
        ),
        "label_cutoff_exclusive": str(label_cutoff),
        "label_availability_buffer_days": label_buffer_days,
        "row_index_sha256": _index_sha256(development.index),
        "population_sha256": population_sha256(development, identity_columns),
        "identity_columns": identity_columns,
        "retention_rate": float(development_target.mean()),
        "both_target_classes_present": True,
        "sealed_holdout_rows_excluded": int(len(holdout)),
        "pre_holdout_label_unavailable_rows_excluded": int(len(embargoed)),
    }
    return (
        development,
        holdout,
        excluded_holdout,
        evaluation_training_population,
        config,
    )








def _history_training_population(
    development: pd.DataFrame,
    *,
    cohort_col: str,
    holdout_start: str,
    history: Mapping[str, Any],
    family: str,
) -> tuple[pd.DataFrame, np.ndarray | None, dict[str, Any]]:
    kind = str(history.get("kind", "")).strip().lower()
    history_id = str(history.get("history_id", "")).strip()
    anchor = pd.Timestamp(holdout_start)
    development_dates = pd.to_datetime(development[cohort_col], errors="raise")
    if kind == "recent":
        months = int(history["months"])
        if months <= 0:
            raise ValueError(f"Saved {family} recent months must be positive.")
        cutoff = anchor - pd.DateOffset(months=months)
        training = development.loc[development_dates.ge(cutoff)].copy()
        weights = None
        detail = {
            "history_id": history_id,
            "kind": kind,
            "months": months,
            "anchor_exclusive": str(anchor),
            "cutoff_inclusive": str(cutoff),
        }
    elif kind == "decay":
        months = int(history["months"])
        max_months = int(history["max_months"])
        if months <= 0 or max_months != 60:
            raise ValueError(
                f"Saved {family} decay history requires positive half-life "
                f"months and max_months=60: {dict(history)}"
            )
        cutoff = anchor - pd.DateOffset(months=max_months)
        training = development.loc[development_dates.ge(cutoff)].copy()
        dates = pd.to_datetime(training[cohort_col], errors="raise")
        age_days = (dates.max() - dates).dt.total_seconds().to_numpy() / 86_400.0
        weights = np.power(0.5, age_days / (months * 30.4375)).astype(np.float64)
        weights /= float(np.mean(weights))
        detail = {
            "history_id": history_id,
            "kind": kind,
            "months": months,
            "max_months": max_months,
            "anchor_exclusive": str(anchor),
            "cutoff_inclusive": str(cutoff),
            "weight_mean": float(weights.mean()),
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
            "weight_effective_rows": float(
                np.square(weights.sum()) / np.square(weights).sum()
            ),
        }
    elif family == "logistic" and kind == "legacy_all":
        training = development.copy()
        weights = None
        detail = {
            **dict(history),
            "history_id": history_id or "LEGACY_ALL_HISTORY",
            "kind": kind,
            "legacy_compatibility": True,
        }
    else:
        raise ValueError(
            f"Unsupported saved {family} history contract: {dict(history)}"
        )
    if training.empty:
        raise ValueError(
            f"Saved {family} history contract removed every source row."
        )
    training_dates = pd.to_datetime(training[cohort_col], errors="raise")
    detail.update(
        {
            "development_rows_before_history": int(len(development)),
            "training_rows_after_history": int(len(training)),
            "older_rows_excluded": int(len(development) - len(training)),
            "training_cohort_start": str(training_dates.min()),
            "training_cohort_end": str(training_dates.max()),
            "sample_weight_applied": weights is not None,
        }
    )
    return training, weights, detail




def _logistic_training_population(
    development: pd.DataFrame,
    *,
    cohort_col: str,
    holdout_start: str,
    history: Mapping[str, Any],
) -> tuple[pd.DataFrame, np.ndarray | None, dict[str, Any]]:
    return _history_training_population(
        development,
        cohort_col=cohort_col,
        holdout_start=holdout_start,
        history=history,
        family="logistic",
    )




def _xgboost_training_population(
    development: pd.DataFrame,
    *,
    cohort_col: str,
    holdout_start: str,
    history: Mapping[str, Any],
) -> tuple[pd.DataFrame, np.ndarray | None, dict[str, Any]]:
    return _history_training_population(
        development,
        cohort_col=cohort_col,
        holdout_start=holdout_start,
        history=history,
        family="xgboost",
    )




def _shared_state_training_population(
    development: pd.DataFrame,
    family_training: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return the common rows allowed by every selected family history.


    The pair contract intentionally shares one source-state map across model
    families.  When histories differ, fitting that artifact on the intersection
    is the only way to ensure no family's excluded older rows affect the map.
    """


    if not family_training:
        raise ValueError("No family training populations were supplied.")
    keep = np.ones(len(development), dtype=bool)
    for family, frame in family_training.items():
        if frame.empty:
            raise ValueError(f"{family} selected an empty training population.")
        keep &= development.index.isin(frame.index)
    shared = development.loc[keep].copy()
    if shared.empty:
        raise ValueError(
            "Selected family histories have no common rows for the canonical "
            "shared state artifact."
        )
    return shared








def _xgboost_factory(
    spec: Mapping[str, Any], *, device: str, n_jobs: int
) -> Callable[[int, int], Any]:
    import xgboost as xgb




    required = (
        "max_depth",
        "learning_rate",
        "min_child_weight",
        "gamma",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
    )
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"Saved XGBoost model specification is missing: {missing}")




    def factory(seed: int, rounds: int) -> Any:
        return xgb.XGBClassifier(
            objective="binary:logistic",
            booster="gbtree",
            tree_method="hist",
            device=device,
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




    return factory








def _find_family_manifest(bundle_dir: Path) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in bundle_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("contract") == BUNDLE_CONTRACT:
            matches.append((path, payload))
    if len(matches) != 1:
        raise ValueError(f"Expected one family manifest under {bundle_dir}.")
    return matches[0]








def _find_pair_manifest(pair_dir: Path) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in pair_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("contract") == PAIR_CONTRACT:
            matches.append((path, payload))
    if len(matches) != 1:
        raise ValueError(f"Expected one source-model pair manifest under {pair_dir}.")
    return matches[0]








def _directory_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": str(path.relative_to(root)).replace("\\", "/"),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in sorted(value for value in root.rglob("*") if value.is_file())
    ]








def _verified_pair_manifest(pair_dir: Path) -> tuple[Path, dict[str, Any]]:
    path, manifest = _find_pair_manifest(pair_dir)
    observed = str(manifest.get("manifest_identity", ""))
    expected = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_identity"}
    )
    if observed != expected:
        raise ValueError(
            f"Source-model pair manifest identity mismatch: {path}; "
            f"observed={observed!r}; expected={expected!r}."
        )
    inventory = manifest.get("artifact_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Source-model pair manifest contains no artifact inventory.")
    root = pair_dir.resolve()
    for item in inventory:
        if not isinstance(item, dict):
            raise ValueError("Invalid source-model pair inventory row.")
        artifact = (root / str(item["relative_path"])).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise ValueError("Pair inventory path escapes its pair directory.") from exc
        if not artifact.is_file():
            raise ValueError(f"Pair artifact is missing: {artifact}")
        if artifact.stat().st_size != int(item["bytes"]):
            raise ValueError(f"Pair artifact size mismatch: {artifact}")
        if sha256_file(artifact) != str(item["sha256"]):
            raise ValueError(f"Pair artifact hash mismatch: {artifact}")
    family_rows = manifest.get("families")
    if not isinstance(family_rows, list) or not family_rows:
        raise ValueError("Source-model pair manifest contains no family bundles.")
    shared_state = manifest.get("shared_state_grouping")
    if not isinstance(shared_state, dict):
        raise ValueError(
            "Source-model pair has no canonical shared state grouping. Rebuild it "
            "with the current source-bundle builder."
        )
    expected_mapping = {
        str(key).strip().upper(): str(value).strip()
        for key, value in dict(shared_state.get("mapping", {})).items()
    }
    expected_categories = [
        str(value).strip() for value in shared_state.get("categories", [])
    ]
    expected_mapping_sha = sha256_json(expected_mapping)
    if (
        not expected_mapping
        or not expected_categories
        or str(shared_state.get("mapping_sha256", "")) != expected_mapping_sha
    ):
        raise ValueError("Pair-level shared state-grouping contract is invalid.")
    for row in family_rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid source-model family registry row.")
        family_manifest_path = (root / str(row["manifest_relative_path"])).resolve()
        try:
            family_manifest_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Family manifest escapes its source-pair directory.") from exc
        family_path, family_manifest = _find_family_manifest(family_manifest_path.parent)
        if family_path != family_manifest_path:
            raise ValueError("Family registry does not point to its unique manifest.")
        if family_manifest.get("manifest_identity") != row.get("manifest_identity"):
            raise ValueError("Family registry manifest identity mismatch.")
        if family_manifest.get("pairing_identity") != manifest.get("pairing_identity"):
            raise ValueError("Family bundle pairing identity mismatch.")
        _, preprocessor = load_bundle(
            family_manifest_path.parent, verify_hashes=True
        )
        family_shared = family_manifest.get("shared_state_grouping")
        state_contract = preprocessor.get("state_contract")
        if not isinstance(family_shared, dict) or not isinstance(state_contract, dict):
            raise ValueError(
                "A family bundle has no shared state-grouping contract. Rebuild the "
                "source-model pair with the current builder."
            )
        observed_mapping = {
            str(key).strip().upper(): str(value).strip()
            for key, value in dict(state_contract.get("mapping", {})).items()
        }
        observed_categories = [
            str(value).strip() for value in state_contract.get("categories", [])
        ]
        if (
            observed_mapping != expected_mapping
            or observed_categories != expected_categories
            or str(state_contract.get("mapping_sha256", ""))
            != expected_mapping_sha
            or str(family_shared.get("mapping_sha256", ""))
            != expected_mapping_sha
        ):
            raise ValueError(
                "Logistic/XGBoost state groupings are not identical to the pair-level "
                "source-tenure mapping. Rebuild the source-model pair."
            )
    return path, manifest








def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp_", suffix=".json", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)








def materialize(args: argparse.Namespace) -> Path:
    project_root = args.project_root.expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"Project root not found: {project_root}")
    requested_tenures = _requested_tenure_groups(args)
    families = _parse_families(args.families)
    contracts = _resolve_run_root(
        project_root=project_root,
        runs_root=args.runs_root,
        source_run=args.source_run,
        source_tenures=requested_tenures,
        families=families,
    )
    source_tenures = _normalize_tenure_groups(
        contracts.profile["selected_tenure_groups"]
    )
    source_scope = _scope_label(source_tenures)
    logistic_info = contracts.logistic_run_info.payload if contracts.logistic_run_info else None
    if logistic_info is None:
        # Population/split materialization requires the shared Logistic run contract,
        # even when only XGBoost is requested.
        raise ValueError("The source run needs its shared Logistic population/split contract.")
    _require_recorded_runtime_version(
        source_contract=logistic_info,
        contract_key="sklearn_version",
        installed_version=sklearn.__version__,
        package_label="scikit-learn",
    )
    if "xgboost" in families:
        if contracts.xgboost_run_info is None:
            raise ValueError("The source run has no completed XGBoost run contract.")
        import xgboost as xgb




        _require_recorded_runtime_version(
            source_contract=contracts.xgboost_run_info.payload,
            contract_key="xgboost_version",
            installed_version=xgb.__version__,
            package_label="XGBoost",
        )




    data_descriptor = dict(logistic_info["data"])
    data_path = resolve_identity_file(
        data_descriptor, run_root=contracts.run_root, project_root=project_root
    )
    modeling_descriptor = dict(logistic_info["modeling_utils"])
    modeling_path = resolve_identity_file(
        modeling_descriptor, run_root=contracts.run_root, project_root=project_root
    )
    outer_path = resolve_identity_file(
        logistic_info["outer_fold_definitions"],
        run_root=contracts.run_root,
        project_root=project_root,
    )
    preparation_payload: dict[str, Any] = {}
    if _is_file_identity(logistic_info.get("preparation_info")):
        preparation_path = resolve_identity_file(
            logistic_info["preparation_info"],
            run_root=contracts.run_root,
            project_root=project_root,
        )
        loaded_preparation = json.loads(
            preparation_path.read_text(encoding="utf-8-sig")
        )
        if not isinstance(loaded_preparation, dict):
            raise ValueError("Saved preparation-info contract must be a JSON object.")
        preparation_payload = loaded_preparation
    if contracts.xgboost_run_info is not None:
        xgb_input = contracts.xgboost_run_info.payload["input"]
        if str(xgb_input["sha256"]) != str(data_descriptor["sha256"]):
            raise ValueError("Logistic and XGBoost did not use the same source snapshot.")




    saved_holdout = _saved_holdout_contract(contracts.documents)
    source_code_paths: dict[str, Path] = {
        "frozen_source_model_builder": Path(__file__).resolve(),
        "deployment_bundle": Path(deployment_bundle_runtime.__file__).resolve(),
        "modeling_utils": modeling_path,
    }
    python_identity_records: list[dict[str, Any]] = []
    dependency_payloads: list[tuple[str, Mapping[str, Any]]] = [
        ("logistic_run", logistic_info),
        ("preparation", preparation_payload),
    ]
    if contracts.xgboost_run_info is not None:
        dependency_payloads.append(
            ("xgboost_run", contracts.xgboost_run_info.payload)
        )
    for scope, payload in dependency_payloads:
        collection = _collect_python_identities(
            payload,
            run_root=contracts.run_root,
            project_root=project_root,
        )
        for label, path in collection.resolved_paths.items():
            source_code_paths[f"{scope}.{label}"] = path
        for record in collection.records:
            python_identity_records.append(
                {
                    **record,
                    "scope": scope,
                    "identity_label": f"{scope}.{record['identity_label']}",
                }
            )




    state_grouping_path = _unique_module_path(source_code_paths, "state_grouping")
    if _is_file_identity(logistic_info.get("state_grouping")) and state_grouping_path is None:
        raise ValueError("Saved state-grouping identity could not be activated.")
    tenure_profile_path = _unique_module_path(source_code_paths, "tenure_profile")
    params_path = _unique_module_path(source_code_paths, "params")
    retention_population_path = _unique_module_path(
        source_code_paths, "retention_population"
    )




    with (
        _saved_profile_environment(contracts.profile, contracts.profile_hash),
        _working_directory(project_root),
        _isolated_profile_modules(),
    ):
        if tenure_profile_path is not None:
            _import_module_aliases(
                tenure_profile_path,
                ("func.tenure_profile", "tenure_profile"),
            )
        if params_path is not None:
            _import_module_aliases(params_path, ("params",))
        if retention_population_path is not None:
            _import_module_aliases(
                retention_population_path, ("func.retention_population",)
            )
        if state_grouping_path is not None:
            _import_module_aliases(
                state_grouping_path,
                ("c_modeling.state_grouping", "state_grouping"),
            )
        for alias, expected in (
            ("func.tenure_profile", tenure_profile_path),
            ("params", params_path),
            ("func.retention_population", retention_population_path),
            ("c_modeling.state_grouping", state_grouping_path),
        ):
            if expected is not None:
                _verify_active_module(alias, expected)
        modeling_utils = _import_module(modeling_path, "saved_modeling_utils")
        cohort_control_contract = (
            modeling_utils.cohort_calendar_year_contract()
            if hasattr(modeling_utils, "cohort_calendar_year_contract")
            else None
        )
        derived_feature_types = (
            {
                str(cohort_control_contract["derived_feature"]).strip().upper():
                str(cohort_control_contract["configured_type"]).strip().lower()
            }
            if cohort_control_contract is not None
            else {}
        )
        (
            development,
            _,
            excluded_holdout,
            evaluation_training_population,
            config,
        ) = _prepare_source_population(
            modeling_utils=modeling_utils,
            data_path=data_path,
            run_info=logistic_info,
            outer_contract_path=outer_path,
            saved_holdout=saved_holdout,
            profile=contracts.profile,
            source_tenures=source_tenures,
        )




        identity_columns = [
            str(value) for value in excluded_holdout["identity_columns"]
        ]
        family_recipe_cores: dict[str, dict[str, Any]] = {}
        feature_type_paths: dict[str, Path] = {}
        feature_types: dict[str, dict[str, str]] = {}
        if "logistic" in families:
            assert contracts.logistic_recipe is not None
            lr_core = _logistic_recipe_core(
                contracts.logistic_recipe.payload,
                allow_missing_legacy_logistic=(
                    _allows_legacy_logistic_history(
                        contracts.logistic_run_info
                    )
                ),
            )
            assert lr_core is not None
            family_recipe_cores["logistic"] = lr_core
            lr_types_path = resolve_identity_file(
                logistic_info["feature_types_file"],
                run_root=contracts.run_root,
                project_root=project_root,
            )
            feature_type_paths["logistic"] = lr_types_path
            if derived_feature_types.keys() & set(lr_core["features"]):
                saved_contract = contracts.logistic_recipe.payload.get(
                    "cohort_calendar_year_contract"
                )
                if saved_contract != cohort_control_contract:
                    raise ValueError(
                        "Logistic recipe time-control contract differs from its "
                        "saved modeling runtime."
                    )
            feature_types["logistic"] = _load_feature_types(
                lr_types_path,
                lr_core["features"],
                derived_feature_types=derived_feature_types,
            )
        if "xgboost" in families:
            assert contracts.xgboost_recipe is not None
            assert contracts.xgboost_run_info is not None
            xgb_core = _xgboost_recipe_core(contracts.xgboost_recipe.payload)
            assert xgb_core is not None
            family_recipe_cores["xgboost"] = xgb_core
            xgb_types_path = resolve_identity_file(
                contracts.xgboost_run_info.payload["feature_types_file"],
                run_root=contracts.run_root,
                project_root=project_root,
            )
            feature_type_paths["xgboost"] = xgb_types_path
            if derived_feature_types.keys() & set(xgb_core["features"]):
                saved_contract = contracts.xgboost_recipe.payload.get(
                    "cohort_calendar_year_contract"
                )
                if saved_contract != cohort_control_contract:
                    raise ValueError(
                        "XGBoost recipe time-control contract differs from its "
                        "saved modeling runtime."
                    )
            feature_types["xgboost"] = _load_feature_types(
                xgb_types_path,
                xgb_core["features"],
                derived_feature_types=derived_feature_types,
            )




        family_training: dict[str, pd.DataFrame] = {}
        family_sample_weights: dict[str, np.ndarray | None] = {}
        family_history_application: dict[str, dict[str, Any]] = {}
        if "logistic" in family_recipe_cores:
            logistic_training, logistic_weights, logistic_history_detail = (
                _logistic_training_population(
                    development,
                    cohort_col=config.cohort_date_col,
                    holdout_start=excluded_holdout["cohort_start"],
                    history=family_recipe_cores["logistic"]["selected_history"],
                )
            )
            family_training["logistic"] = logistic_training
            family_sample_weights["logistic"] = logistic_weights
            family_history_application["logistic"] = logistic_history_detail
        if "xgboost" in family_recipe_cores:
            xgb_training, xgb_weights, xgb_history_detail = (
                _xgboost_training_population(
                    development,
                    cohort_col=config.cohort_date_col,
                    holdout_start=excluded_holdout["cohort_start"],
                    history=family_recipe_cores["xgboost"]["selected_history"],
                )
            )
            family_training["xgboost"] = xgb_training
            family_sample_weights["xgboost"] = xgb_weights
            family_history_application["xgboost"] = xgb_history_detail




        shared_state_training = _shared_state_training_population(
            development, family_training
        )
        shared_state_artifact, shared_state_contract = fit_shared_state_artifact(
            training_frame=shared_state_training,
            target_col=config.target_col,
            modeling_utils=modeling_utils,
        )
        shared_state_grouping = {
            **shared_state_contract,
            "source_tenure": source_scope,
            "source_tenure_groups": list(source_tenures),
            "fit_scope": "intersection_of_family_selected_training_populations",
            "fit_rows": int(len(shared_state_training)),
            "fit_population_identity_columns": identity_columns,
            "fit_population_sha256": population_sha256(
                shared_state_training, identity_columns
            ),
            "family_selected_history": {
                family: dict(core["selected_history"])
                for family, core in family_recipe_cores.items()
            },
            "family_selected_training_population_sha256": {
                family: population_sha256(frame, identity_columns)
                for family, frame in family_training.items()
            },
            "target_tenure_rows_used": 0,
        }
        family_final_fit_populations = {
            family: {
                "rows": int(len(frame)),
                "population_sha256": population_sha256(frame, identity_columns),
                "identity_columns": list(identity_columns),
                "history_application": family_history_application[family],
            }
            for family, frame in family_training.items()
        }




        saved_default_preprocessing = modeling_utils.ModelSpec()
        if "seed" not in logistic_info["outer_split_settings"]:
            raise ValueError("Saved Logistic split contract is missing its random seed.")
        if "max_iter_ridge" not in logistic_info:
            raise ValueError("Saved Logistic run contract is missing max_iter_ridge.")
        logistic_fit_controls = {
            "random_seed": int(logistic_info["outer_split_settings"]["seed"]),
            "max_iter": int(logistic_info["max_iter_ridge"]),
            "tol": float(saved_default_preprocessing.tol),
            "rare_min_frequency": saved_default_preprocessing.rare_min_frequency,
        }
        xgboost_fit_controls: dict[str, Any] | None = None
        if "xgboost" in families:
            assert contracts.xgboost_run_info is not None
            saved_arguments = dict(contracts.xgboost_run_info.payload["arguments"])
            missing_controls = [
                key
                for key in ("device", "n_jobs", "rare_min_frequency")
                if key not in saved_arguments
            ]
            if missing_controls:
                raise ValueError(
                    "Saved XGBoost run arguments are missing final-fit controls: "
                    f"{missing_controls}."
                )
            xgboost_fit_controls = {
                "device": str(args.xgb_device or saved_arguments["device"]),
                "n_jobs": int(
                    args.xgb_n_jobs
                    if args.xgb_n_jobs is not None
                    else saved_arguments["n_jobs"]
                ),
                "rare_min_frequency": float(saved_arguments["rare_min_frequency"]),
            }
        materialization_runtime_versions = _runtime_versions(
            include_xgboost="xgboost" in families
        )
        portable_python_identity_records = sorted(
            (
                {
                    "scope": record["scope"],
                    "identity_label": record["identity_label"],
                    "logical_stem": record["logical_stem"],
                    "recorded_role": record["recorded_role"],
                    "execution_role": record["execution_role"],
                    "required_for_runtime": bool(record["required_for_runtime"]),
                    "sha256": record["sha256"],
                    "size": record["size"],
                    "resolution_status": record["resolution_status"],
                }
                for record in python_identity_records
            ),
            key=lambda record: (
                str(record["identity_label"]),
                str(record["sha256"]),
            ),
        )
        pairing_payload = {
            "contract": PAIR_CONTRACT,
            "contract_revision": PAIR_CONTRACT_REVISION,
            "source_run_id": contracts.run_root.name,
            "source_tenure": source_scope,
            "source_tenure_groups": list(source_tenures),
            "selected_tenure_groups": list(source_tenures),
            "profile_hash": contracts.profile_hash,
            "source_data_sha256": data_descriptor["sha256"],
            "source_data_size": int(data_path.stat().st_size),
            "excluded_holdout": excluded_holdout,
            "evaluation_training_population": evaluation_training_population,
            "cohort_column": config.cohort_date_col,
            "cohort_calendar_year_contract": cohort_control_contract,
            "shared_state_grouping": shared_state_grouping,
            "recipes": family_recipe_cores,
            "family_final_fit_populations": family_final_fit_populations,
            "feature_type_contract_sha256": {
                family: sha256_file(path) for family, path in feature_type_paths.items()
            },
            "modeling_utils_sha256": sha256_file(modeling_path),
            "source_code_sha256": {
                label: sha256_file(path)
                for label, path in sorted(source_code_paths.items())
            },
            "recorded_python_identities": portable_python_identity_records,
            "activated_source_modules": {
                "tenure_profile": (
                    None
                    if tenure_profile_path is None
                    else sha256_file(tenure_profile_path)
                ),
                "params": (
                    None
                    if params_path is None
                    else sha256_file(params_path)
                ),
                "retention_population": (
                    None
                    if retention_population_path is None
                    else sha256_file(retention_population_path)
                ),
                "state_grouping": (
                    None
                    if state_grouping_path is None
                    else sha256_file(state_grouping_path)
                ),
            },
            "materialization_runtime_versions": materialization_runtime_versions,
            "final_fit_controls": {
                "logistic": logistic_fit_controls,
                "xgboost": xgboost_fit_controls,
            },
        }
        pairing_identity = sha256_json(pairing_payload)
        output_root = (
            project_root / "output" / "frozen_source_models"
            if args.output_root is None
            else (
                args.output_root.expanduser().resolve()
                if args.output_root.is_absolute()
                else (project_root / args.output_root).resolve()
            )
        )
        output_root.mkdir(parents=True, exist_ok=True)
        # Keep published and temporary paths safely below the traditional
        # Windows MAX_PATH boundary while retaining full identities inside the
        # manifests. Directory names are locators, never the identity source.
        pair_dir = output_root / f"sp_{pairing_identity[:16]}"
        if pair_dir.exists():
            if not args.reuse_existing:
                raise FileExistsError(
                    f"Content-identical source-model pair already exists: {pair_dir}"
                )
            _, existing = _verified_pair_manifest(pair_dir)
            if existing.get("pairing_identity") != pairing_identity:
                raise ValueError(f"Existing pair identity mismatch: {pair_dir}")
            return pair_dir




        temporary_pair = Path(tempfile.mkdtemp(prefix=".p_", dir=output_root))
        family_records: list[dict[str, Any]] = []
        try:
            shared_state_relative_dir = Path("shared_preprocessing") / "state_grouping"
            shared_state_artifact.save(temporary_pair / shared_state_relative_dir)
            source_data_identity = {
                **data_descriptor,
                "resolved_path": str(data_path),
                "verified_sha256": sha256_file(data_path),
            }
            shared_metadata = {
                "pairing_identity": pairing_identity,
                "source_tenure": source_scope,
                "source_tenure_groups": list(source_tenures),
                "selected_tenure_groups": list(source_tenures),
                "profile": dict(contracts.profile),
                "profile_hash": contracts.profile_hash,
                "source_run_id": contracts.run_root.name,
                "source_run_path": str(contracts.run_root),
                "source_data_identity": source_data_identity,
                "excluded_holdout": excluded_holdout,
                "evaluation_training_population": evaluation_training_population,
                "family_final_fit_populations": family_final_fit_populations,
                "cohort_column": config.cohort_date_col,
                "cohort_calendar_year_contract": cohort_control_contract,
                "shared_state_grouping": shared_state_grouping,
                "recorded_python_identities": portable_python_identity_records,
            }
            if "logistic" in families:
                core = family_recipe_cores["logistic"]
                logistic_training = family_training["logistic"]
                logistic_weights = family_sample_weights["logistic"]
                logistic_history_detail = family_history_application["logistic"]
                recipe_identity = sha256_json(core)
                destination = temporary_pair / f"lr_{recipe_identity[:16]}"
                seed = int(logistic_fit_controls["random_seed"])
                max_iter = int(logistic_fit_controls["max_iter"])




                def logistic_factory() -> LogisticRegression:
                    return LogisticRegression(
                        penalty="l2",
                        C=float(core["C"]),
                        solver=str(core["solver"]),
                        max_iter=max_iter,
                        tol=float(logistic_fit_controls["tol"]),
                        class_weight=core.get("class_weight"),
                        random_state=seed,
                        n_jobs=-1,
                    )




                export_logistic_source_bundle(
                    destination=destination,
                    training_frame=logistic_training,
                    features=core["features"],
                    feature_types=feature_types["logistic"],
                    rare_min_frequency=float(
                        logistic_fit_controls["rare_min_frequency"]
                    ),
                    target_col=config.target_col,
                    cohort_col=config.cohort_date_col,
                    identity_columns=identity_columns,
                    model_factory=logistic_factory,
                    sample_weight=logistic_weights,
                    model_recipe={
                        **core,
                        "max_iter": max_iter,
                        "tol": float(logistic_fit_controls["tol"]),
                        "random_seed": seed,
                        "final_history_application": logistic_history_detail,
                        "final_fit_scope": (
                            "all_source_development_rows_legacy_compatibility"
                            if logistic_history_detail.get("legacy_compatibility")
                            else "source_development_rows_after_saved_history_rule"
                        ),
                    },
                    modeling_utils=modeling_utils,
                    shared_state_artifact=shared_state_artifact,
                    source_metadata=shared_metadata,
                    source_code_paths=source_code_paths,
                )
                manifest_path, manifest = _find_family_manifest(destination)
                family_records.append(
                    {
                        "model_family": "logistic",
                        "bundle_relative_path": str(destination.relative_to(temporary_pair)).replace(
                            "\\", "/"
                        ),
                        "manifest_relative_path": str(
                            manifest_path.relative_to(temporary_pair)
                        ).replace("\\", "/"),
                        "manifest_identity": manifest["manifest_identity"],
                    }
                )




            if "xgboost" in families:
                assert contracts.xgboost_run_info is not None
                core = family_recipe_cores["xgboost"]
                recipe_identity = sha256_json(core)
                destination = temporary_pair / f"xgb_{recipe_identity[:16]}"
                assert xgboost_fit_controls is not None
                device = str(xgboost_fit_controls["device"])
                n_jobs = int(xgboost_fit_controls["n_jobs"])
                rare_frequency = float(xgboost_fit_controls["rare_min_frequency"])
                xgb_training = family_training["xgboost"]
                weights = family_sample_weights["xgboost"]
                history_detail = family_history_application["xgboost"]
                import xgboost as xgb




                export_xgboost_source_bundle(
                    destination=destination,
                    training_frame=xgb_training,
                    features=core["features"],
                    feature_types=feature_types["xgboost"],
                    rare_min_frequency=rare_frequency,
                    target_col=config.target_col,
                    cohort_col=config.cohort_date_col,
                    identity_columns=identity_columns,
                    seeds=core["seeds"],
                    boosting_rounds=core["boosting_rounds"],
                    sample_weight=weights,
                    model_factory=_xgboost_factory(
                        core["selected_model_spec"], device=device, n_jobs=n_jobs
                    ),
                    model_recipe={
                        **core,
                        "device": device,
                        "n_jobs": n_jobs,
                        "rare_min_frequency": rare_frequency,
                        "final_history_application": history_detail,
                        "final_fit_scope": "source_development_rows_after_saved_history_rule",
                    },
                    modeling_utils=modeling_utils,
                    shared_state_artifact=shared_state_artifact,
                    source_metadata=shared_metadata,
                    source_code_paths=source_code_paths,
                    xgboost_version=xgb.__version__,
                )
                manifest_path, manifest = _find_family_manifest(destination)
                family_records.append(
                    {
                        "model_family": "xgboost",
                        "bundle_relative_path": str(destination.relative_to(temporary_pair)).replace(
                            "\\", "/"
                        ),
                        "manifest_relative_path": str(
                            manifest_path.relative_to(temporary_pair)
                        ).replace("\\", "/"),
                        "manifest_identity": manifest["manifest_identity"],
                    }
                )




            pair_manifest = {
                **pairing_payload,
                "pairing_identity": pairing_identity,
                "profile": dict(contracts.profile),
                "source_run_path": str(contracts.run_root),
                "source_data_identity": source_data_identity,
                "shared_state_grouping": {
                    **shared_state_grouping,
                    "artifact_relative_directory": str(
                        shared_state_relative_dir
                    ).replace("\\", "/"),
                },
                "families": family_records,
                "artifact_inventory": _directory_inventory(temporary_pair),
                "builder": {
                    "component_role": COMPONENT_ROLE,
                    "path": str(Path(__file__).resolve()),
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
            }
            pair_manifest_identity = sha256_json(pair_manifest)
            pair_manifest["manifest_identity"] = pair_manifest_identity
            _atomic_json(
                temporary_pair / f"pair_manifest_{pair_manifest_identity[:20]}.json",
                pair_manifest,
            )
            os.replace(temporary_pair, pair_dir)
            return pair_dir
        except Exception:
            shutil.rmtree(temporary_pair, ignore_errors=True)
            raise








def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    pair_dir = materialize(args)
    pair_manifest_path, pair_manifest = _verified_pair_manifest(pair_dir)
    print("FROZEN SOURCE MODEL BUILD COMPLETE", flush=True)
    print(f"Source run: {pair_manifest['source_run_id']}", flush=True)
    print(
        f"Source tenure groups: {pair_manifest['selected_tenure_groups']}",
        flush=True,
    )
    print(f"Pairing identity: {pair_manifest['pairing_identity']}", flush=True)
    print(f"Manifest: {pair_manifest_path}", flush=True)
    return 0








if __name__ == "__main__":
    raise SystemExit(main())
