#!/usr/bin/env python
"""Fit, persist, verify, load, and score frozen source-model bundles.




The bundle contract is deliberately target-agnostic at scoring time.  Every
learned preprocessing object, derived-feature artifact, feature order, and
fitted estimator is learned from the source training population and persisted.
Calling :func:`transform_frame` never fits and never reads a target column.
"""




from __future__ import annotations




import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
from typing import Any, Callable, Iterator, Mapping, Sequence
import uuid




import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler








BUNDLE_CONTRACT = "retention_frozen_source_model_bundle"
BUNDLE_CONTRACT_REVISION = 1
CORE_SCORING_RUNTIME_VERSION_KEYS = ("numpy", "pandas", "scikit_learn")
BUNDLE_RENAME_RETRY_DELAYS_SECONDS = (0.25, 0.5, 1.0, 2.0, 4.0, 4.0, 4.0)




class BundlePublicationError(RuntimeError):
    """A completed bundle remains staged because its final rename failed."""


    def __init__(self, temporary: Path, destination: Path, error: OSError):
        self.temporary_dir = temporary
        self.destination = destination
        super().__init__(
            f"Could not publish completed model bundle: {error}. "
            f"Staged files have been retained at: {temporary}. "
            f"Intended destination: {destination}. "
            "No existing destination was deliberately replaced. Close file or "
            "preview processes holding these paths before retrying the failed "
            "pipeline stage. Persistent access-denied errors require checking "
            "the folder's rename permissions; retries do not change permissions."
        )








def _installed_scoring_runtime_versions(*, include_xgboost: bool) -> dict[str, str]:
    """Return versions from the active scoring interpreter.




    Values are read from the imported packages at runtime.  There are no
    accepted-version constants: a frozen bundle must match the environment in
    which it was created exactly.
    """




    observed = {
        "numpy": str(np.__version__),
        "pandas": str(pd.__version__),
        "scikit_learn": str(sklearn.__version__),
    }
    if include_xgboost:
        try:
            import xgboost as xgb
        except ImportError as exc:  # pragma: no cover - exercised without xgboost
            raise RuntimeError(
                "Frozen XGBoost scoring requires xgboost, but it is not installed."
            ) from exc
        observed["xgboost"] = str(xgb.__version__)
    return observed








def validate_runtime_versions(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Fail closed unless recorded and installed scoring versions match exactly."""




    family = str(manifest.get("model_family", "")).strip().lower()
    if family not in {"logistic", "xgboost"}:
        raise ValueError(
            f"Cannot validate runtime versions for model family {family!r}."
        )
    recorded = manifest.get("runtime_versions")
    if not isinstance(recorded, Mapping):
        raise ValueError("Frozen bundle has no recorded runtime_versions contract.")
    required = [*CORE_SCORING_RUNTIME_VERSION_KEYS]
    if family == "xgboost":
        required.append("xgboost")
    missing = [key for key in required if not str(recorded.get(key, "")).strip()]
    if missing:
        raise ValueError(
            "Frozen bundle runtime_versions is missing required package versions: "
            f"{missing}."
        )
    installed = _installed_scoring_runtime_versions(
        include_xgboost=family == "xgboost"
    )
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
            "Frozen bundle runtime-version mismatch; scoring is refused: "
            + "; ".join(mismatches)
        )
    return audit








def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)








def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()








def sha256_json(payload: Any) -> str:
    rendered = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()








def feature_order_sha256(features: Sequence[str]) -> str:
    normalized = [str(value).strip().upper() for value in features]
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()








def population_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Order-independent identity for the exact rows used to fit the model."""




    selected = [str(value) for value in columns if str(value) in frame]
    if not selected:
        raise ValueError("At least one training-population identity column is required.")
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
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()








def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp_", suffix=".json", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)








def _atomic_joblib(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp_", suffix=".joblib", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)








def _one_hot_encoder(
    min_frequency: float | int | None,
    *,
    categories: Sequence[str] | None = None,
) -> OneHotEncoder:
    kwargs: dict[str, Any] = {
        "handle_unknown": "ignore",
        "dtype": np.float32,
    }
    if min_frequency is not None:
        kwargs["min_frequency"] = min_frequency
    if categories is not None:
        normalized = [str(value) for value in categories]
        if not normalized or len(normalized) != len(set(normalized)):
            raise ValueError("Explicit one-hot categories must be nonempty and unique.")
        kwargs["categories"] = [np.asarray(normalized, dtype=object)]
    try:
        return OneHotEncoder(sparse_output=True, **kwargs)
    except TypeError:  # pragma: no cover - old scikit-learn compatibility
        # scikit-learn <1.2 spells only the sparse-output argument
        # differently.  Keep min_frequency so an older, unsupported runtime
        # fails closed instead of silently changing the frozen rare-level rule.
        return OneHotEncoder(sparse=True, **kwargs)








def _categorical_array(series: pd.Series) -> np.ndarray:
    values = series.astype("string").astype(object).to_numpy(copy=True)
    values[pd.isna(values)] = np.nan
    return values.reshape(-1, 1)








def _numeric_array(series: pd.Series) -> np.ndarray:
    return (
        pd.to_numeric(series, errors="coerce")
        .to_numpy(dtype=np.float64, na_value=np.nan)
        .reshape(-1, 1)
    )








def _virtual_feature_contract(modeling_utils: Any) -> dict[str, dict[str, Any]]:
    attributes = {
        "written_premium": ("WRITTEN_PREM_BIN_FEATURE", "WRITTEN_PREM_SOURCE_FEATURE"),
        "state_group": ("STATE_GROUP_FEATURE", "STATE_SOURCE_FEATURE"),
    }
    contract: dict[str, dict[str, Any]] = {}
    for role, (derived_attr, source_attr) in attributes.items():
        if not hasattr(modeling_utils, derived_attr) or not hasattr(
            modeling_utils, source_attr
        ):
            raise AttributeError(
                "Saved modeling_utils is missing its virtual-feature contract: "
                f"{derived_attr}, {source_attr}."
            )
        derived = str(getattr(modeling_utils, derived_attr)).strip().upper()
        source = str(getattr(modeling_utils, source_attr)).strip().upper()
        if not derived or not source or derived == source:
            raise ValueError(
                f"Invalid saved virtual-feature contract for {role}: "
                f"derived={derived!r}; source={source!r}."
            )
        contract[role] = {
            "derived_feature": derived,
            "source_feature": source,
        }
    cohort_attributes = (
        "COHORT_CALENDAR_YEAR_FEATURE",
        "COHORT_CALENDAR_YEAR_SOURCE_FEATURE",
        "COHORT_CALENDAR_YEAR_TYPE",
        "COHORT_CALENDAR_YEAR_FORMULA",
        "COHORT_CALENDAR_YEAR_CONTRACT_VERSION",
    )
    present = [hasattr(modeling_utils, name) for name in cohort_attributes]
    if any(present) and not all(present):
        missing = [
            name for name, exists in zip(cohort_attributes, present) if not exists
        ]
        raise AttributeError(
            "Saved modeling_utils has a partial cohort-calendar-year contract; "
            f"missing={missing}."
        )
    if all(present):
        derived = str(
            getattr(modeling_utils, "COHORT_CALENDAR_YEAR_FEATURE")
        ).strip().upper()
        source = str(
            getattr(modeling_utils, "COHORT_CALENDAR_YEAR_SOURCE_FEATURE")
        ).strip().upper()
        configured_type = str(
            getattr(modeling_utils, "COHORT_CALENDAR_YEAR_TYPE")
        ).strip().lower()
        if (
            not derived
            or not source
            or derived == source
            or configured_type != "continuous"
        ):
            raise ValueError("Invalid saved cohort-calendar-year contract.")
        contract["cohort_calendar_year"] = {
            "contract_version": int(
                getattr(modeling_utils, "COHORT_CALENDAR_YEAR_CONTRACT_VERSION")
            ),
            "derived_feature": derived,
            "source_feature": source,
            "configured_type": configured_type,
            "unit": "calendar_year",
            "encoding": "continuous_integer",
            "formula": str(
                getattr(modeling_utils, "COHORT_CALENDAR_YEAR_FORMULA")
            ),
            "known_at_scoring_time": True,
            "target_used": False,
            "month_component_used": False,
        }
    return contract








def state_contract_from_artifact(fitted_state: Any) -> dict[str, Any]:
    """Return and validate the canonical, model-family-independent state map."""




    mapping = {
        str(key).strip().upper(): str(value).strip()
        for key, value in fitted_state.mapping.items()
    }
    categories = [str(value).strip() for value in fitted_state.categories]
    if not mapping:
        raise ValueError("The shared source-tenure state mapping is empty.")
    if not categories or len(categories) != len(set(categories)):
        raise ValueError(
            "The shared source-tenure state categories must be nonempty and unique."
        )
    if any(not key or not value for key, value in mapping.items()):
        raise ValueError("The shared source-tenure state mapping contains blank values.")
    outside = sorted(set(mapping.values()).difference(categories))
    if outside:
        raise ValueError(
            "The shared source-tenure state mapping uses undeclared categories: "
            f"{outside}."
        )
    return {
        "mapping": mapping,
        "categories": categories,
        "mapping_sha256": sha256_json(mapping),
        "scope": "shared_source_tenure",
    }








def fit_shared_state_artifact(
    *,
    training_frame: pd.DataFrame,
    target_col: str,
    modeling_utils: Any,
) -> tuple[Any, dict[str, Any]]:
    """Fit one source-tenure state artifact for every model family to reuse.




    This function is called once by the source-bundle builder on rows admitted
    by every selected family history.  Family-specific preprocessing and model
    fitting may use larger family-local selected populations, but rows excluded
    by a family's saved history must never influence this shared artifact.
    """




    virtual = _virtual_feature_contract(modeling_utils)
    state_feature = virtual["state_group"]["derived_feature"]
    derived = modeling_utils.prefit_derived_feature_artifacts(
        training_frame,
        {state_feature: "discrete"},
        target_col=target_col,
    )
    fitted_state = derived.get(state_feature)
    if fitted_state is None:
        raise ValueError(
            f"Saved modeling_utils did not fit required shared feature {state_feature}."
        )
    contract = state_contract_from_artifact(fitted_state)
    contract.update(
        {
            "derived_feature": state_feature,
            "source_feature": virtual["state_group"]["source_feature"],
        }
    )
    return fitted_state, contract








def _apply_written_premium(
    series: pd.Series, artifact: Mapping[str, Any], *, derived_feature: str
) -> pd.Series:
    edges = np.asarray(artifact["edges"], dtype=float)
    labels = [str(value) for value in artifact["labels"]]
    if len(edges) != len(labels) + 1 or not np.all(np.diff(edges) > 0):
        raise ValueError("Frozen written-premium bin edges are invalid.")
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    numeric = numeric.where(np.isfinite(numeric), np.nan)
    output = pd.cut(
        numeric,
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
        ordered=True,
    ).astype("string")
    return output.fillna(str(artifact["missing_label"])).rename(
        derived_feature
    )








def _apply_state_mapping(
    series: pd.Series,
    state_contract: Mapping[str, Any],
    *,
    derived_feature: str,
) -> pd.Series:
    mapping = {str(key): str(value) for key, value in state_contract["mapping"].items()}
    categories = {str(value) for value in state_contract["categories"]}
    state = series.astype("string").str.strip().str.upper()
    missing_tokens = {"", "<NA>", "NAN", "NONE", "NULL"}
    state = state.mask(state.isin(missing_tokens), pd.NA)
    malformed = state.notna() & ~state.str.fullmatch(r"[A-Z]{2}", na=False)
    outside = state.notna() & ~malformed & ~state.isin(mapping)
    if state.isna().any() or malformed.any() or outside.any():
        raise ValueError(
            "Scoring rows violate the frozen source-state contract: "
            f"missing_rows={int(state.isna().sum())}; "
            f"malformed={sorted(state.loc[malformed].astype(str).unique())}; "
            f"outside_contract={sorted(state.loc[outside].astype(str).unique())}."
        )
    output = state.map(mapping).astype("string")
    if output.isna().any() or not output.isin(categories).all():
        raise AssertionError("Frozen source-state mapping produced an invalid category.")
    return output.rename(derived_feature)








def _raw_series_for_feature(
    frame: pd.DataFrame,
    feature: str,
    fitted: Mapping[str, Any],
) -> pd.Series:
    virtual = fitted["derived_feature_contracts"]
    written = virtual["written_premium"]
    state_group = virtual["state_group"]
    cohort_year = virtual.get("cohort_calendar_year")
    if feature == written["derived_feature"]:
        source = written["source_feature"]
        if source not in frame:
            raise KeyError(
                f"{written['derived_feature']} requires {source}."
            )
        return _apply_written_premium(
            frame[source],
            fitted["written_premium_artifact"],
            derived_feature=written["derived_feature"],
        )
    if feature == state_group["derived_feature"]:
        source = state_group["source_feature"]
        if source not in frame:
            raise KeyError(f"{state_group['derived_feature']} requires {source}.")
        return _apply_state_mapping(
            frame[source],
            fitted["state_contract"],
            derived_feature=state_group["derived_feature"],
        )
    if cohort_year is not None and feature == cohort_year["derived_feature"]:
        source = cohort_year["source_feature"]
        if source not in frame:
            raise KeyError(f"{cohort_year['derived_feature']} requires {source}.")
        parsed = pd.to_datetime(frame[source], errors="coerce")
        invalid = parsed.isna()
        if invalid.any():
            raise ValueError(
                f"{cohort_year['derived_feature']} requires nonmissing valid "
                f"calendar dates; invalid_rows={int(invalid.sum())}."
            )
        values = parsed.dt.year.astype(np.int64)
        return values.rename(cohort_year["derived_feature"])
    if feature not in frame:
        raise KeyError(f"Scoring frame is missing source-model feature {feature!r}.")
    return frame[feature]








def fit_preprocessor(
    *,
    training_frame: pd.DataFrame,
    features: Sequence[str],
    feature_types: Mapping[str, str],
    rare_min_frequency: float | int | None,
    target_col: str,
    modeling_utils: Any,
    state_artifact_dir: Path,
    shared_state_artifact: Any,
) -> tuple[dict[str, Any], sparse.csr_matrix, list[dict[str, Any]]]:
    """Fit family-local preprocessing while reusing one shared state artifact."""




    ordered_features = [str(value).strip().upper() for value in features]
    if not ordered_features or len(ordered_features) != len(set(ordered_features)):
        raise ValueError("Source-model features must be nonempty and unique.")
    selected_types = {
        feature: str(feature_types.get(feature, "discrete")).strip().lower()
        for feature in ordered_features
    }
    invalid_types = {
        feature: kind
        for feature, kind in selected_types.items()
        if kind not in {"continuous", "discrete"}
    }
    if invalid_types:
        raise ValueError(f"Unsupported source feature types: {invalid_types}")




    virtual_feature_contracts = _virtual_feature_contract(modeling_utils)
    written_feature = virtual_feature_contracts["written_premium"]["derived_feature"]
    state_feature = virtual_feature_contracts["state_group"]["derived_feature"]
    cohort_year_contract = virtual_feature_contracts.get("cohort_calendar_year")
    cohort_year_feature = (
        None
        if cohort_year_contract is None
        else cohort_year_contract["derived_feature"]
    )
    if cohort_year_feature in ordered_features and (
        selected_types.get(cohort_year_feature) != "continuous"
    ):
        raise ValueError(
            f"{cohort_year_feature} must use continuous preprocessing."
        )
    if shared_state_artifact is None:
        raise ValueError(
            "A canonical shared source-tenure state artifact is required. Rebuild "
            "the source bundle with the current builder."
        )
    family_derived_types = dict(selected_types)
    family_derived_types.pop(state_feature, None)
    derived = modeling_utils.prefit_derived_feature_artifacts(
        training_frame,
        family_derived_types,
        target_col=target_col,
    )
    written_artifact = derived.get(written_feature)
    if state_feature in derived:
        raise AssertionError(
            "Family-local preprocessing attempted to refit the shared state artifact."
        )
    fitted_state = shared_state_artifact
    if written_feature in ordered_features and written_artifact is None:
        raise ValueError(
            f"Saved modeling_utils did not fit requested virtual feature {written_feature}."
        )
    if state_feature in ordered_features and fitted_state is None:
        raise ValueError(
            f"Saved modeling_utils did not fit requested virtual feature {state_feature}."
        )
    state_contract = state_contract_from_artifact(fitted_state)
    fitted_state.save(state_artifact_dir)
    state_artifact_files: list[str] = []
    for state_path in sorted(
        value for value in state_artifact_dir.rglob("*") if value.is_file()
    ):
        addressed = _content_addressed_file(
            state_path, label=state_path.stem or "state_grouping_artifact"
        )
        state_artifact_files.append(
            str(addressed.relative_to(state_artifact_dir.parent.parent)).replace(
                "\\", "/"
            )
        )
    state_contract["artifact_files"] = state_artifact_files




    fitted: dict[str, Any] = {
        "contract": BUNDLE_CONTRACT,
        "contract_revision": BUNDLE_CONTRACT_REVISION,
        "features": ordered_features,
        "feature_types": selected_types,
        "rare_min_frequency": rare_min_frequency,
        "derived_feature_contracts": virtual_feature_contracts,
        "transformers": {},
        "written_premium_artifact": written_artifact,
        "state_contract": state_contract,
    }
    blocks: list[sparse.csr_matrix] = []
    ranges: list[dict[str, Any]] = []
    start = 0
    for feature in ordered_features:
        series = _raw_series_for_feature(training_frame, feature, fitted)
        kind = selected_types[feature]
        if kind == "discrete":
            categories: Sequence[str] | None = None
            feature_frequency = rare_min_frequency
            if feature == written_feature:
                categories = [
                    *[str(value) for value in written_artifact["labels"]],
                    str(written_artifact["missing_label"]),
                ]
                feature_frequency = None
            elif feature == state_feature:
                categories = list(state_contract["categories"])
                feature_frequency = None
            transformer = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="__MISSING__"),
                    ),
                    (
                        "onehot",
                        _one_hot_encoder(feature_frequency, categories=categories),
                    ),
                ]
            )
            values = _categorical_array(series)
        else:
            transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler(with_mean=False)),
                ]
            )
            values = _numeric_array(series)
        matrix = sparse.csr_matrix(
            transformer.fit_transform(values), dtype=np.float32
        )
        if matrix.shape[1] < 1:
            raise RuntimeError(f"{feature} produced no transformed columns.")
        fitted["transformers"][feature] = transformer
        stop = start + int(matrix.shape[1])
        ranges.append(
            {
                "raw_feature": feature,
                "configured_type": kind,
                "start_column": start,
                "end_column_exclusive": stop,
                "transformed_column_count": int(matrix.shape[1]),
            }
        )
        blocks.append(matrix)
        start = stop
    training_matrix = sparse.hstack(blocks, format="csr", dtype=np.float32)
    fitted["transformed_column_count"] = int(training_matrix.shape[1])
    fitted["feature_ranges"] = ranges
    return fitted, training_matrix, ranges








def transform_frame(
    fitted_preprocessor: Mapping[str, Any], scoring_frame: pd.DataFrame
) -> sparse.csr_matrix:
    """Apply source-fitted preprocessing without fitting or target access."""




    blocks: list[sparse.csr_matrix] = []
    for feature in fitted_preprocessor["features"]:
        series = _raw_series_for_feature(scoring_frame, feature, fitted_preprocessor)
        kind = fitted_preprocessor["feature_types"][feature]
        values = _categorical_array(series) if kind == "discrete" else _numeric_array(series)
        transformer = fitted_preprocessor["transformers"][feature]
        matrix = sparse.csr_matrix(transformer.transform(values), dtype=np.float32)
        blocks.append(matrix)
    output = sparse.hstack(blocks, format="csr", dtype=np.float32)
    expected = int(fitted_preprocessor["transformed_column_count"])
    if output.shape != (len(scoring_frame), expected):
        raise RuntimeError(
            "Frozen preprocessing width mismatch: "
            f"observed={output.shape}; expected_rows={len(scoring_frame)}; "
            f"expected_columns={expected}."
        )
    return output








def _runtime_versions(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    if extra:
        payload.update(dict(extra))
    return payload








def _copy_runtime_sources(
    bundle_dir: Path, source_code_paths: Mapping[str, str | Path]
) -> dict[str, str]:
    runtime_dir = bundle_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    sources = {"deployment_bundle": Path(__file__).resolve(), **source_code_paths}
    copied: dict[str, str] = {}
    for label, source_value in sources.items():
        source = Path(source_value).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Bundle runtime source is missing: {source}")
        safe_label = "".join(
            character if character.isalnum() or character in "_-" else "_"
            for character in str(label)
        ).strip("_")
        if not safe_label:
            raise ValueError(f"Invalid runtime-source label: {label!r}")
        label_identity = hashlib.sha256(str(label).encode("utf-8")).hexdigest()[:8]
        destination = runtime_dir / (
            f"src_{label_identity}_{sha256_file(source)[:16]}"
            f"{source.suffix or '.py'}"
        )
        shutil.copy2(source, destination)
        copied[str(label)] = str(destination.relative_to(bundle_dir))
    return copied








def _inventory(bundle_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(value for value in bundle_dir.rglob("*") if value.is_file()):
        if path.name == "bundle_manifest.json":
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(bundle_dir)).replace("\\", "/"),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return rows








def _bundle_destination_path(destination: str | Path) -> Path:
    """Resolve parents without following an existing destination symlink."""


    path = Path(destination).expanduser().absolute()
    return path.parent.resolve() / path.name




def _new_temporary_bundle(destination: Path) -> Path:
    destination = _bundle_destination_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise FileExistsError(
            f"Refusing to overwrite an existing source-model bundle: {destination}"
        )
    return Path(tempfile.mkdtemp(prefix=".b_", dir=destination.parent))








def _content_addressed_file(path: Path, *, label: str) -> Path:
    """Rename one completed artifact to a deterministic content-addressed name."""




    digest = sha256_file(path)
    safe_label = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in str(label)
    ).strip("_")
    if not safe_label:
        raise ValueError("Content-addressed artifact labels cannot be empty.")
    if len(safe_label) > 20:
        label_identity = hashlib.sha256(safe_label.encode("utf-8")).hexdigest()[:8]
        safe_label = f"{safe_label[:11]}_{label_identity}"
    destination = path.with_name(f"{safe_label}_{digest[:20]}{path.suffix}")
    if destination.exists():
        raise FileExistsError(f"Content-addressed artifact already exists: {destination}")
    os.replace(path, destination)
    return destination








def _publish_bundle(
    temporary: Path, destination: Path, manifest: dict[str, Any]
) -> str:
    manifest_identity = sha256_json(
        {
            key: value
            for key, value in manifest.items()
            if key != "manifest_identity"
        }
    )
    manifest["manifest_identity"] = manifest_identity
    manifest_name = f"family_manifest_{manifest_identity[:20]}.json"
    _atomic_json(temporary / manifest_name, manifest)
    destination = _bundle_destination_path(destination)
    # Publication is a single same-filesystem directory rename. A transient
    # Windows scanner/file-preview lock must not immediately discard the fit.
    # Do not fall back to copying a partially visible bundle or overwriting an
    # existing destination. os.rename also refuses a racing destination on
    # Windows, unlike os.replace for existing files.
    delays = BUNDLE_RENAME_RETRY_DELAYS_SECONDS
    for attempt in range(len(delays) + 1):
        if os.path.lexists(destination):
            error = FileExistsError(
                f"Refusing to overwrite an existing model bundle: {destination}"
            )
            raise BundlePublicationError(temporary, destination, error) from error
        try:
            os.rename(temporary, destination)
            break
        except OSError as error:
            retryable = getattr(error, "winerror", None) in {5, 32, 33}
            if not retryable or attempt == len(delays):
                raise BundlePublicationError(
                    temporary, destination, error
                ) from error
            delay = delays[attempt]
            print(
                f"Bundle publication blocked by Windows (WinError "
                f"{error.winerror}); retry {attempt + 1}/{len(delays)} "
                f"in {delay:g}s: {destination}",
                flush=True,
            )
            time.sleep(delay)
    return manifest_name








def _find_bundle_manifest(root: Path) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("contract") == BUNDLE_CONTRACT:
            matches.append((path, payload))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {BUNDLE_CONTRACT!r} manifest under {root}; "
            f"found {[str(path) for path, _ in matches]}."
        )
    return matches[0]








def _population_summary(
    frame: pd.DataFrame,
    *,
    target_col: str,
    cohort_col: str,
    identity_columns: Sequence[str],
) -> dict[str, Any]:
    target = pd.to_numeric(frame[target_col], errors="raise").astype(int)
    if set(target.unique()) != {0, 1}:
        raise ValueError("Source model training target must contain both classes.")
    cohort = pd.to_datetime(frame[cohort_col], errors="raise")
    present_identity = [value for value in identity_columns if value in frame]
    return {
        "rows": int(len(frame)),
        "retention_rate": float(target.mean()),
        "cohort_start": str(cohort.min()),
        "cohort_end": str(cohort.max()),
        "identity_columns": present_identity,
        "population_sha256": population_sha256(frame, present_identity),
    }




def _validated_sample_weight(
    sample_weight: np.ndarray | Sequence[float] | None,
    *,
    rows: int,
    family: str,
) -> np.ndarray | None:
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != (int(rows),):
        raise ValueError(f"{family} sample weights do not match training rows.")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError(f"{family} sample weights must be finite and positive.")
    if not np.isclose(float(weights.mean()), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError(
            f"{family} sample weights must be mean-normalized to one."
        )
    return weights




def _sample_weight_summary(sample_weight: np.ndarray | None) -> dict[str, Any]:
    if sample_weight is None:
        return {
            "applied": False,
            "normalization": None,
        }
    return {
        "applied": True,
        "normalization": "mean_one",
        "mean": float(sample_weight.mean()),
        "minimum": float(sample_weight.min()),
        "maximum": float(sample_weight.max()),
        "effective_rows": float(
            np.square(sample_weight.sum()) / np.square(sample_weight).sum()
        ),
    }








def _promoted_source_metadata(source_metadata: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "pairing_identity",
        "source_tenure",
        "selected_tenure_groups",
        "profile",
        "profile_hash",
        "source_run_id",
        "source_run_path",
        "source_data_identity",
        "excluded_holdout",
        "evaluation_training_population",
        "cohort_column",
        "shared_state_grouping",
    )
    missing = [key for key in required if key not in source_metadata]
    if missing:
        raise ValueError(f"Source bundle metadata is missing required fields: {missing}")
    return {key: source_metadata[key] for key in required}








def export_logistic_source_bundle(
    *,
    destination: str | Path,
    training_frame: pd.DataFrame,
    features: Sequence[str],
    feature_types: Mapping[str, str],
    rare_min_frequency: float | int | None,
    target_col: str,
    cohort_col: str,
    identity_columns: Sequence[str],
    model_factory: Callable[[], Any],
    model_recipe: Mapping[str, Any],
    modeling_utils: Any,
    shared_state_artifact: Any,
    source_metadata: Mapping[str, Any],
    source_code_paths: Mapping[str, str | Path],
    sample_weight: np.ndarray | Sequence[float] | None = None,
) -> dict[str, Any]:
    sample_weight = _validated_sample_weight(
        sample_weight,
        rows=len(training_frame),
        family="Logistic",
    )
    destination_path = _bundle_destination_path(destination)
    temporary = _new_temporary_bundle(destination_path)
    try:
        fitted, matrix, ranges = fit_preprocessor(
            training_frame=training_frame,
            features=features,
            feature_types=feature_types,
            rare_min_frequency=rare_min_frequency,
            target_col=target_col,
            modeling_utils=modeling_utils,
            state_artifact_dir=temporary / "preprocessing" / "state_grouping",
            shared_state_artifact=shared_state_artifact,
        )
        preprocessor_path = temporary / "preprocessing" / "preprocessor.joblib"
        _atomic_joblib(preprocessor_path, fitted)
        preprocessor_path = _content_addressed_file(
            preprocessor_path, label="fitted_preprocessor"
        )
        written_premium_path: Path | None = None
        if fitted.get("written_premium_artifact") is not None:
            written_premium_path = temporary / "preprocessing" / "written_premium_artifact.json"
            _atomic_json(written_premium_path, fitted["written_premium_artifact"])
            written_premium_path = _content_addressed_file(
                written_premium_path, label="written_premium_artifact"
            )
        ranges_path = temporary / "preprocessing" / "feature_ranges.csv"
        pd.DataFrame(ranges).to_csv(ranges_path, index=False)
        ranges_path = _content_addressed_file(ranges_path, label="feature_ranges")
        model = model_factory()
        target = pd.to_numeric(training_frame[target_col], errors="raise").astype(int)
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(matrix, target.to_numpy(), **fit_kwargs)
        model_path = temporary / "models" / "logistic_model.joblib"
        _atomic_joblib(model_path, model)
        model_path = _content_addressed_file(model_path, label="logistic_estimator")
        probe = np.asarray(
            model.predict_proba(matrix[: min(1000, int(matrix.shape[0]))])
        )
        if probe.ndim != 2 or not np.isfinite(probe).all():
            raise RuntimeError("Saved Logistic model failed the source-training smoke test.")
        runtime_sources = _copy_runtime_sources(temporary, source_code_paths)
        population = _population_summary(
            training_frame,
            target_col=target_col,
            cohort_col=cohort_col,
            identity_columns=identity_columns,
        )
        manifest: dict[str, Any] = {
            "contract": BUNDLE_CONTRACT,
            "contract_revision": BUNDLE_CONTRACT_REVISION,
            **_promoted_source_metadata(source_metadata),
            "model_family": "logistic",
            "prediction_target": "retention_probability",
            "target_column": target_col,
            "features": list(fitted["features"]),
            "feature_order_sha256": feature_order_sha256(fitted["features"]),
            "raw_feature_count": int(len(fitted["features"])),
            "transformed_column_count": int(matrix.shape[1]),
            "derived_feature_contracts": dict(fitted["derived_feature_contracts"]),
            "model_recipe": dict(model_recipe),
            "training_population": population,
            "sample_weight": _sample_weight_summary(sample_weight),
            "source_metadata": dict(source_metadata),
            "runtime_versions": _runtime_versions(),
            "runtime_sources": runtime_sources,
            "preprocessor_file": str(preprocessor_path.relative_to(temporary)).replace(
                "\\", "/"
            ),
            "model_file": str(model_path.relative_to(temporary)).replace("\\", "/"),
            "feature_ranges_file": str(ranges_path.relative_to(temporary)).replace(
                "\\", "/"
            ),
            "written_premium_artifact_file": (
                None
                if written_premium_path is None
                else str(written_premium_path.relative_to(temporary)).replace("\\", "/")
            ),
            "artifact_inventory": _inventory(temporary),
        }
        _publish_bundle(temporary, destination_path, manifest)
        return manifest
    except BundlePublicationError:
        # The fit and manifest are complete. Keep them for inspection/recovery;
        # the stage still fails and cannot present this directory as published.
        raise
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise








def export_xgboost_source_bundle(
    *,
    destination: str | Path,
    training_frame: pd.DataFrame,
    features: Sequence[str],
    feature_types: Mapping[str, str],
    rare_min_frequency: float | int | None,
    target_col: str,
    cohort_col: str,
    identity_columns: Sequence[str],
    seeds: Sequence[int],
    boosting_rounds: int,
    sample_weight: np.ndarray | None,
    model_factory: Callable[[int, int], Any],
    model_recipe: Mapping[str, Any],
    modeling_utils: Any,
    shared_state_artifact: Any,
    source_metadata: Mapping[str, Any],
    source_code_paths: Mapping[str, str | Path],
    xgboost_version: str,
) -> dict[str, Any]:
    normalized_seeds = [int(value) for value in seeds]
    if not normalized_seeds or len(normalized_seeds) != len(set(normalized_seeds)):
        raise ValueError("XGBoost bundle seeds must be nonempty and unique.")
    if int(boosting_rounds) < 1:
        raise ValueError("XGBoost boosting rounds must be positive.")
    sample_weight = _validated_sample_weight(
        sample_weight,
        rows=len(training_frame),
        family="XGBoost",
    )




    destination_path = _bundle_destination_path(destination)
    temporary = _new_temporary_bundle(destination_path)
    try:
        fitted, matrix, ranges = fit_preprocessor(
            training_frame=training_frame,
            features=features,
            feature_types=feature_types,
            rare_min_frequency=rare_min_frequency,
            target_col=target_col,
            modeling_utils=modeling_utils,
            state_artifact_dir=temporary / "preprocessing" / "state_grouping",
            shared_state_artifact=shared_state_artifact,
        )
        preprocessor_path = temporary / "preprocessing" / "preprocessor.joblib"
        _atomic_joblib(preprocessor_path, fitted)
        preprocessor_path = _content_addressed_file(
            preprocessor_path, label="fitted_preprocessor"
        )
        written_premium_path: Path | None = None
        if fitted.get("written_premium_artifact") is not None:
            written_premium_path = temporary / "preprocessing" / "written_premium_artifact.json"
            _atomic_json(written_premium_path, fitted["written_premium_artifact"])
            written_premium_path = _content_addressed_file(
                written_premium_path, label="written_premium_artifact"
            )
        ranges_path = temporary / "preprocessing" / "feature_ranges.csv"
        pd.DataFrame(ranges).to_csv(ranges_path, index=False)
        ranges_path = _content_addressed_file(ranges_path, label="feature_ranges")
        y_churn = 1 - pd.to_numeric(
            training_frame[target_col], errors="raise"
        ).astype(int).to_numpy()
        model_files: list[str] = []
        for seed in normalized_seeds:
            model = model_factory(seed, int(boosting_rounds))
            fit_kwargs: dict[str, Any] = {}
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            model.fit(matrix, y_churn, **fit_kwargs)
            model_path = temporary / "models" / f"xgboost_seed_{seed}.ubj"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_model = model_path.with_name(
                f".m_{uuid.uuid4().hex[:12]}.tmp{model_path.suffix}"
            )
            model.save_model(str(temporary_model))
            os.replace(temporary_model, model_path)
            model_path = _content_addressed_file(
                model_path, label=f"xgboost_seed_{seed}"
            )
            probe = np.asarray(
                model.predict_proba(
                    matrix[: min(1000, int(matrix.shape[0]))]
                )[:, 1],
                dtype=float,
            )
            if probe.ndim != 1 or not np.isfinite(probe).all():
                raise RuntimeError(
                    f"Saved XGBoost seed {seed} failed the source-training smoke test."
                )
            model_files.append(str(model_path.relative_to(temporary)).replace("\\", "/"))
        runtime_sources = _copy_runtime_sources(temporary, source_code_paths)
        population = _population_summary(
            training_frame,
            target_col=target_col,
            cohort_col=cohort_col,
            identity_columns=identity_columns,
        )
        manifest = {
            "contract": BUNDLE_CONTRACT,
            "contract_revision": BUNDLE_CONTRACT_REVISION,
            **_promoted_source_metadata(source_metadata),
            "model_family": "xgboost",
            "prediction_target": "churn_probability",
            "target_column": target_col,
            "features": list(fitted["features"]),
            "feature_order_sha256": feature_order_sha256(fitted["features"]),
            "raw_feature_count": int(len(fitted["features"])),
            "transformed_column_count": int(matrix.shape[1]),
            "derived_feature_contracts": dict(fitted["derived_feature_contracts"]),
            "model_recipe": dict(model_recipe),
            "seeds": normalized_seeds,
            "boosting_rounds": int(boosting_rounds),
            "model_files": model_files,
            "ensemble_aggregation": "arithmetic_mean_churn_probability",
            "training_population": population,
            "sample_weight": _sample_weight_summary(sample_weight),
            "source_metadata": dict(source_metadata),
            "runtime_versions": _runtime_versions({"xgboost": xgboost_version}),
            "runtime_sources": runtime_sources,
            "preprocessor_file": str(preprocessor_path.relative_to(temporary)).replace(
                "\\", "/"
            ),
            "feature_ranges_file": str(ranges_path.relative_to(temporary)).replace(
                "\\", "/"
            ),
            "written_premium_artifact_file": (
                None
                if written_premium_path is None
                else str(written_premium_path.relative_to(temporary)).replace("\\", "/")
            ),
            "artifact_inventory": _inventory(temporary),
        }
        _publish_bundle(temporary, destination_path, manifest)
        return manifest
    except BundlePublicationError:
        # A failed final rename must not delete already-fitted seed estimators.
        raise
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise








def load_bundle(bundle_dir: str | Path, *, verify_hashes: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(bundle_dir).expanduser().resolve()
    _, manifest = _find_bundle_manifest(root)
    if manifest.get("contract") != BUNDLE_CONTRACT:
        raise ValueError(f"Unsupported source-model bundle contract: {manifest.get('contract')!r}")
    if int(manifest.get("contract_revision", -1)) != BUNDLE_CONTRACT_REVISION:
        raise ValueError(
            "Unsupported source-model bundle revision: "
            f"{manifest.get('contract_revision')!r}"
        )
    observed_manifest_identity = str(manifest.get("manifest_identity", ""))
    expected_manifest_identity = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_identity"}
    )
    if observed_manifest_identity != expected_manifest_identity:
        raise ValueError(
            "Source-model bundle manifest identity mismatch: "
            f"observed={observed_manifest_identity!r}; "
            f"expected={expected_manifest_identity!r}."
        )
    if feature_order_sha256(manifest["features"]) != manifest["feature_order_sha256"]:
        raise ValueError("Source-model bundle feature-order hash mismatch.")
    validate_runtime_versions(manifest)
    if verify_hashes:
        for item in manifest["artifact_inventory"]:
            path = root / item["relative_path"]
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise ValueError(f"Source-model bundle artifact hash mismatch: {path}")
    fitted = joblib.load(root / manifest["preprocessor_file"])
    if list(fitted["features"]) != list(manifest["features"]):
        raise ValueError("Preprocessor feature order differs from bundle manifest.")
    return manifest, fitted








def score_logistic_bundle(
    bundle_dir: str | Path,
    scoring_frame: pd.DataFrame,
    *,
    verify_hashes: bool = True,
) -> np.ndarray:
    root = Path(bundle_dir).expanduser().resolve()
    manifest, fitted = load_bundle(root, verify_hashes=verify_hashes)
    if manifest["model_family"] != "logistic":
        raise ValueError("Requested Logistic scoring from a non-Logistic bundle.")
    matrix = transform_frame(fitted, scoring_frame)
    model = joblib.load(root / manifest["model_file"])
    classes = {int(value): index for index, value in enumerate(model.classes_)}
    if 1 not in classes:
        raise ValueError("Saved Logistic model has no retention-positive class 1.")
    return np.clip(
        np.asarray(model.predict_proba(matrix)[:, classes[1]], dtype=float),
        1e-7,
        1.0 - 1e-7,
    )








def _validated_batch_size(batch_size: int) -> int:
    if (
        isinstance(batch_size, (bool, np.bool_))
        or not isinstance(batch_size, (int, np.integer))
        or int(batch_size) <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    return int(batch_size)








def iter_logistic_bundle_scores(
    bundle_dir: str | Path,
    scoring_frame: pd.DataFrame,
    *,
    batch_size: int = 100_000,
    verify_hashes: bool = True,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield bounded batches of Logistic retention probabilities.




    The bundle, fitted preprocessor, and estimator are loaded exactly once.
    Each yielded tuple contains the positional ``start`` and exclusive ``stop``
    offsets into ``scoring_frame`` plus the corresponding retention
    probabilities.  The scoring frame does not need to contain a target.
    """




    rows_per_batch = _validated_batch_size(batch_size)
    if not isinstance(scoring_frame, pd.DataFrame):
        raise TypeError("scoring_frame must be a pandas DataFrame.")
    root = Path(bundle_dir).expanduser().resolve()
    manifest, fitted = load_bundle(root, verify_hashes=verify_hashes)
    if manifest["model_family"] != "logistic":
        raise ValueError("Requested Logistic scoring from a non-Logistic bundle.")
    model = joblib.load(root / manifest["model_file"])
    classes = {int(value): index for index, value in enumerate(model.classes_)}
    if 1 not in classes:
        raise ValueError("Saved Logistic model has no retention-positive class 1.")
    retention_class = classes[1]




    def batches() -> Iterator[tuple[int, int, np.ndarray]]:
        for start in range(0, len(scoring_frame), rows_per_batch):
            stop = min(start + rows_per_batch, len(scoring_frame))
            matrix = transform_frame(fitted, scoring_frame.iloc[start:stop])
            probability = np.clip(
                np.asarray(
                    model.predict_proba(matrix)[:, retention_class], dtype=float
                ),
                1e-7,
                1.0 - 1e-7,
            )
            if probability.shape != (stop - start,):
                raise RuntimeError(
                    "Saved Logistic model returned an invalid batch shape: "
                    f"observed={probability.shape}; expected={(stop - start,)}."
                )
            yield start, stop, probability




    return batches()








def score_xgboost_bundle(
    bundle_dir: str | Path,
    scoring_frame: pd.DataFrame,
    *,
    verify_hashes: bool = True,
) -> np.ndarray:
    import xgboost as xgb




    root = Path(bundle_dir).expanduser().resolve()
    manifest, fitted = load_bundle(root, verify_hashes=verify_hashes)
    if manifest["model_family"] != "xgboost":
        raise ValueError("Requested XGBoost scoring from a non-XGBoost bundle.")
    matrix = transform_frame(fitted, scoring_frame)
    predictions: list[np.ndarray] = []
    for relative_path in manifest["model_files"]:
        model = xgb.XGBClassifier()
        model.load_model(str(root / relative_path))
        predictions.append(np.asarray(model.predict_proba(matrix)[:, 1], dtype=float))
    if not predictions:
        raise ValueError("XGBoost source bundle contains no model files.")
    return np.clip(np.mean(np.vstack(predictions), axis=0), 1e-7, 1.0 - 1e-7)








def iter_xgboost_bundle_scores(
    bundle_dir: str | Path,
    scoring_frame: pd.DataFrame,
    *,
    batch_size: int = 100_000,
    verify_hashes: bool = True,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield bounded batches of mean-ensemble XGBoost churn probabilities.




    The bundle, fitted preprocessor, and every seed estimator are loaded exactly
    once.  The returned probability meaning intentionally matches
    :func:`score_xgboost_bundle`: class 1 is churn, not retention.  The scoring
    frame does not need to contain a target.
    """




    rows_per_batch = _validated_batch_size(batch_size)
    if not isinstance(scoring_frame, pd.DataFrame):
        raise TypeError("scoring_frame must be a pandas DataFrame.")
    import xgboost as xgb




    root = Path(bundle_dir).expanduser().resolve()
    manifest, fitted = load_bundle(root, verify_hashes=verify_hashes)
    if manifest["model_family"] != "xgboost":
        raise ValueError("Requested XGBoost scoring from a non-XGBoost bundle.")
    models: list[Any] = []
    for relative_path in manifest["model_files"]:
        model = xgb.XGBClassifier()
        model.load_model(str(root / relative_path))
        models.append(model)
    if not models:
        raise ValueError("XGBoost source bundle contains no model files.")




    def batches() -> Iterator[tuple[int, int, np.ndarray]]:
        for start in range(0, len(scoring_frame), rows_per_batch):
            stop = min(start + rows_per_batch, len(scoring_frame))
            matrix = transform_frame(fitted, scoring_frame.iloc[start:stop])
            probability_sum = np.zeros(stop - start, dtype=float)
            for model in models:
                seed_probability = np.asarray(
                    model.predict_proba(matrix)[:, 1], dtype=float
                )
                if seed_probability.shape != (stop - start,):
                    raise RuntimeError(
                        "Saved XGBoost model returned an invalid batch shape: "
                        f"observed={seed_probability.shape}; "
                        f"expected={(stop - start,)}."
                    )
                probability_sum += seed_probability
            yield start, stop, np.clip(
                probability_sum / float(len(models)),
                1e-7,
                1.0 - 1e-7,
            )




    return batches()








__all__ = [
    "BUNDLE_CONTRACT",
    "BUNDLE_CONTRACT_REVISION",
    "export_logistic_source_bundle",
    "export_xgboost_source_bundle",
    "fit_shared_state_artifact",
    "state_contract_from_artifact",
    "feature_order_sha256",
    "load_bundle",
    "iter_logistic_bundle_scores",
    "iter_xgboost_bundle_scores",
    "population_sha256",
    "validate_runtime_versions",
    "score_logistic_bundle",
    "score_xgboost_bundle",
    "sha256_file",
    "transform_frame",
]
