#!/usr/bin/env python
"""Runtime utilities for Logistic Regression floating-forward selection.


This module contains only the preprocessing re-exports, sparse matrix cache,
Ridge fitting, metrics, and atomic checkpoint helpers used by
``logistic_feature_selection.py``. It contains no forward-selection stages and
has no dependency on an earlier model-selection workflow.
"""


from __future__ import annotations


import hashlib
import json
import math
import os
import sqlite3
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, TextIO


import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression




def required_path(name: str) -> Path:
    """Return a required path supplied by the production launcher."""


    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(f"Required environment variable {name} is not set.")
    return Path(value).expanduser().resolve()




CODE = required_path("RETENTION_LR_CODE_DIR")
DATA = required_path("RETENTION_LR_DATA")
MANIFEST = required_path("RETENTION_LR_MANIFEST")
POOL = required_path("RETENTION_LR_POOL")
FEATURE_TYPES_CONTRACT = required_path("RETENTION_LR_FEATURE_TYPES")
ROUTING_AUDIT = required_path("RETENTION_FEATURE_ROUTING_AUDIT")
PREPARATION_INFO = required_path("RETENTION_LR_PREPARATION_INFO")
OUT = required_path("RETENTION_LR_OUTPUT_DIR")
OUT.mkdir(parents=True, exist_ok=True)
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))


from modeling_utils import (  # noqa: E402
    ModelSpec,
    PreparedFold,
    build_candidate_audit,
    build_expanding_folds,
    compute_metrics,
    export_fold_derived_artifacts,
    fold_definitions_frame,
    load_project_population_and_folds,
    prepare_fold_objects,
    read_table,
)




SEED = int(os.environ.get("RETENTION_LR_SEED", "42"))
PRIMARY_METRIC = os.environ.get(
    "RETENTION_LR_PRIMARY_METRIC", "ap_skill"
).strip().lower()
TRAIN_MAX_ROWS = int(os.environ.get("RETENTION_LR_TRAIN_MAX_ROWS", "100000"))
VALIDATION_MAX_ROWS = int(
    os.environ.get("RETENTION_LR_VALIDATION_MAX_ROWS", "50000")
)
MAX_ITER_RIDGE = int(os.environ.get("RETENTION_LR_MAX_ITER_RIDGE", "2000"))
LIFT_SPREAD_BIN_COUNTS = (3, 4, 5, 10, 20)
# The SQLite table layout has not changed, but the payload contract has.  A
# versioned result payload lets an interrupted run distinguish reusable fits
# from pre-fix rows that do not contain the lift-spread diagnostics required by
# the history-screen reports.
FIT_RESULT_SCHEMA_VERSION = 2
LIFT_SPREAD_METRIC_COLUMNS = tuple(
    name
    for bins in LIFT_SPREAD_BIN_COUNTS
    for name in (
        f"highest_lift_{bins}bins",
        f"highest_lift_group_{bins}bins",
        f"lowest_lift_{bins}bins",
        f"lowest_lift_group_{bins}bins",
        f"highest_divided_by_lowest_lift_{bins}bins",
        f"lift_spread_ratio_defined_{bins}bins",
        f"lift_spread_ratio_status_{bins}bins",
    )
)
CHECKPOINT_REQUIRED_RESULT_COLUMNS = (
    "fold",
    "retention_roc_auc",
    "churn_average_precision",
    "churn_ap_skill",
    "top10_lift",
    *LIFT_SPREAD_METRIC_COLUMNS,
)


if PRIMARY_METRIC != "ap_skill":
    raise ValueError("RETENTION_LR_PRIMARY_METRIC must be ap_skill.")
if TRAIN_MAX_ROWS < 1000 or VALIDATION_MAX_ROWS < 1000:
    raise ValueError("Inner sample caps must each be at least 1,000 rows.")
if MAX_ITER_RIDGE < 1:
    raise ValueError("RETENTION_LR_MAX_ITER_RIDGE must be positive.")




def jsonable(value: Any) -> Any:
    """Convert pandas/numpy/path values into deterministic JSON values."""


    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, pd.DataFrame):
        return [jsonable(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.Index):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, set):
        return sorted(jsonable(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime, date, pd.Period)):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if not isinstance(value, (str, bytes)):
        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)) and missing:
                return None
        except (TypeError, ValueError):
            pass
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)




_ATOMIC_REPLACE_ATTEMPTS = 12
_ATOMIC_REPLACE_INITIAL_DELAY_SECONDS = 0.05
_ATOMIC_REPLACE_MAX_DELAY_SECONDS = 0.50




def _replace_with_retry(temporary: Path, destination: Path) -> None:
    """Atomically replace a file despite short-lived Windows sharing locks."""


    delay = _ATOMIC_REPLACE_INITIAL_DELAY_SECONDS
    for attempt in range(_ATOMIC_REPLACE_ATTEMPTS):
        try:
            os.replace(temporary, destination)
            return
        except OSError as exc:
            retryable = isinstance(exc, PermissionError) or getattr(
                exc, "winerror", None
            ) in {5, 32, 33}
            if not retryable:
                raise
            if attempt + 1 == _ATOMIC_REPLACE_ATTEMPTS:
                raise PermissionError(
                    "Windows kept the output file locked after bounded atomic-"
                    f"replace retries: {destination}. Close Excel, Explorer "
                    "Preview, or another program reading this file and rerun."
                ) from exc
            time.sleep(delay)
            delay = min(delay * 2.0, _ATOMIC_REPLACE_MAX_DELAY_SECONDS)




def _atomic_write(path: Path, writer: Callable[[TextIO], object]) -> None:
    """Write through a unique same-directory temporary and publish atomically."""


    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".atomic-", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer(handle)
            handle.flush()
        _replace_with_retry(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)




def atomic_write_text(path: Path, value: str) -> None:
    _atomic_write(path, lambda handle: handle.write(value))




def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_text(path, json.dumps(jsonable(payload), indent=2, sort_keys=True))




def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write(path, lambda handle: frame.to_csv(handle, index=False))




def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()




def feature_hash(features: list[str]) -> str:
    return sha256_text("\n".join(features))[:20]




def file_identity(path: Path, hash_content: bool) -> dict:
    stat = path.stat()
    identity = {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if hash_content:
        identity["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return identity




def read_pool(path: Path) -> list[str]:
    values = [
        value.strip().upper()
        for value in path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError("Screened pool is empty or contains duplicates.")
    return values




def oldest_complete_train(
    df: pd.DataFrame,
    fold: dict,
    cohort_col: str,
) -> pd.DataFrame:
    idx = np.asarray(fold["train_idx"])
    month = pd.to_datetime(df[cohort_col]).dt.to_period("M")
    in_train = pd.Series(df.index.isin(idx), index=df.index)
    complete = in_train.groupby(month).all()
    out = df.loc[month.isin(complete[complete].index)].copy()
    if out.empty or np.setdiff1d(out.index, idx).size:
        raise AssertionError("Invalid oldest-fold development population.")
    return out




def ridge_spec(c_value: float) -> dict:
    return {
        "id": f"RIDGE_C{c_value:g}",
        "family": "ridge",
        "C": float(c_value),
        "l1_ratio": None,
        "solver": "lbfgs",
        "class_weight": None,
    }




def make_ridge_model(spec: dict) -> LogisticRegression:
    if spec.get("family") != "ridge":
        raise ValueError("The production Logistic model accepts Ridge settings only.")
    return LogisticRegression(
        penalty="l2",
        C=float(spec["C"]),
        solver="lbfgs",
        max_iter=MAX_ITER_RIDGE,
        tol=1e-4,
        class_weight=None,
        random_state=SEED,
        n_jobs=-1,
    )




def capacity_metrics(
    y_retention: np.ndarray,
    p_retention: np.ndarray,
) -> dict:
    y_churn = 1 - np.asarray(y_retention, dtype=int)
    p_churn = 1.0 - np.asarray(p_retention, dtype=float)
    prevalence = float(y_churn.mean())
    order = np.argsort(-p_churn, kind="stable")
    lifts = []
    result = {}
    for fraction in (0.05, 0.10, 0.15, 0.20):
        count = max(1, int(math.ceil(len(y_churn) * fraction)))
        positives = int(y_churn[order[:count]].sum())
        precision = positives / count
        capture = positives / max(1, int(y_churn.sum()))
        lift = precision / prevalence if prevalence > 0 else np.nan
        label = int(round(fraction * 100))
        result[f"churn_precision_{label}pct"] = precision
        result[f"churn_capture_{label}pct"] = capture
        result[f"lift_{label}pct"] = lift
        lifts.append(lift)
    result["mean_capacity_lift_5_20"] = float(np.nanmean(lifts))
    return result




def lift_spread_metrics(
    y_retention: np.ndarray,
    p_retention: np.ndarray,
    *,
    bin_counts: tuple[int, ...] = LIFT_SPREAD_BIN_COUNTS,
    row_id: np.ndarray | None = None,
) -> dict[str, Any]:
    """Report equal-count churn-lift spread for every governed bin count.


    These fields are descriptive diagnostics only; Logistic history and
    feature selection do not optimize them.  Stable row IDs make the result
    deterministic when predicted probabilities tie.
    """


    retention = np.asarray(y_retention, dtype=int).reshape(-1)
    probabilities = np.asarray(p_retention, dtype=float).reshape(-1)
    if retention.shape != probabilities.shape or not len(retention):
        raise ValueError("Lift-spread inputs must be non-empty and aligned.")
    if (
        not np.isfinite(probabilities).all()
        or not np.isin(retention, [0, 1]).all()
    ):
        raise ValueError(
            "Lift-spread inputs must be finite with a binary target."
        )
    tie_breaker = (
        np.arange(len(retention), dtype=np.int64)
        if row_id is None
        else np.asarray(row_id).reshape(-1)
    )
    if tie_breaker.shape != retention.shape:
        raise ValueError("row_id must align to the lift-spread inputs.")


    ranked = pd.DataFrame(
        {
            "y": 1 - retention,
            "score": 1.0 - probabilities,
            "row_id": tie_breaker,
        }
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
            lifts = (
                ranked.groupby("bin_rank", sort=True)["y"].mean()
                / prevalence
            )
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




def add_derived_metrics(metrics: dict) -> None:
    prevalence = float(metrics["churn_prevalence_baseline"])
    average_precision = float(metrics["churn_average_precision"])
    metrics["churn_ap_skill"] = (average_precision - prevalence) / max(
        1e-12,
        1.0 - prevalence,
    )




def calibration_parameters(
    y: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    outcome = np.asarray(y, dtype=int)
    if np.unique(outcome).size < 2:
        return np.nan, np.nan
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    for position, penalty in enumerate((None, "none")):
        try:
            model = LogisticRegression(
                penalty=penalty,
                solver="lbfgs",
                max_iter=1000,
                tol=1e-7,
            )
            model.fit(logits, outcome)
            return float(model.intercept_[0]), float(model.coef_[0, 0])
        except Exception:
            if position == 1:
                return np.nan, np.nan
    return np.nan, np.nan




def validate_sample_weight(
    sample_weight: np.ndarray | None,
    expected_rows: int,
) -> np.ndarray | None:
    """Return a safe one-dimensional training-weight array or ``None``."""


    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if len(weights) != int(expected_rows):
        raise ValueError(
            "Sample-weight length does not match the fold training rows: "
            f"{len(weights)} != {int(expected_rows)}."
        )
    if not np.isfinite(weights).all():
        raise ValueError("Sample weights must all be finite.")
    if (weights < 0.0).any() or not (weights > 0.0).any():
        raise ValueError(
            "Sample weights must be nonnegative with at least one positive value."
        )
    return weights.copy()




@dataclass
class FoldMatrixCache:
    fold: PreparedFold
    seed_features: list[str]
    train_matrix: sparse.csr_matrix
    validation_matrix: sparse.csr_matrix
    slices: dict[str, tuple[int, int]]
    sample_weight: np.ndarray | None = None
    history_identity: str = ""


    @classmethod
    def build(
        cls,
        fold: PreparedFold,
        seed_features: list[str],
        *,
        sample_weight: np.ndarray | None = None,
        history_identity: str = "",
    ) -> "FoldMatrixCache":
        train_blocks = []
        validation_blocks = []
        slices = {}
        start = 0
        for feature in seed_features:
            block = fold.get_feature_block(feature)
            train_block = block.train_matrix.tocsr()
            validation_block = block.validation_matrix.tocsr()
            if train_block.shape[1] != validation_block.shape[1]:
                raise AssertionError(f"Encoded column mismatch for {feature}.")
            stop = start + train_block.shape[1]
            slices[feature] = (start, stop)
            start = stop
            train_blocks.append(train_block)
            validation_blocks.append(validation_block)
        normalized_weight = validate_sample_weight(sample_weight, len(fold.train))
        return cls(
            fold=fold,
            seed_features=seed_features.copy(),
            train_matrix=sparse.hstack(train_blocks, format="csr"),
            validation_matrix=sparse.hstack(validation_blocks, format="csr"),
            slices=slices,
            sample_weight=normalized_weight,
            history_identity=str(history_identity).strip(),
        )


    def matrices(
        self,
        features: list[str],
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        if features == self.seed_features:
            return self.train_matrix, self.validation_matrix
        if not features:
            return (
                sparse.csr_matrix((self.train_matrix.shape[0], 0), dtype=np.float32),
                sparse.csr_matrix(
                    (self.validation_matrix.shape[0], 0), dtype=np.float32
                ),
            )
        missing = [feature for feature in features if feature not in self.slices]
        if missing:
            raise KeyError(f"Features are outside the matrix cache: {missing}")
        columns = np.concatenate(
            [
                np.arange(*self.slices[feature], dtype=np.int64)
                for feature in features
            ]
        )
        return self.train_matrix[:, columns], self.validation_matrix[:, columns]




def evaluate_model(
    cache: FoldMatrixCache,
    features: list[str],
    spec: dict,
    include_train_diagnostics: bool,
) -> dict:
    started = time.perf_counter()
    x_train, x_validation = cache.matrices(features)
    y_train = np.asarray(cache.fold.y_train, dtype=int)
    y_validation = (
        cache.fold.validation[cache.fold.target_col].astype(int).to_numpy()
    )
    if np.unique(y_train).size < 2 or np.unique(y_validation).size < 2:
        raise RuntimeError(
            f"Fold {cache.fold.fold_number} does not contain both target classes."
        )
    model: LogisticRegression | None = None
    if features:
        model = make_ridge_model(spec)
        fit_kwargs: dict[str, Any] = {}
        if cache.sample_weight is not None:
            fit_kwargs["sample_weight"] = cache.sample_weight
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x_train, cache.fold.y_train, **fit_kwargs)
        convergence_messages = [
            str(item.message)
            for item in caught
            if issubclass(item.category, ConvergenceWarning)
        ]
        class_index = {int(value): idx for idx, value in enumerate(model.classes_)}
        p_validation = model.predict_proba(x_validation)[:, class_index[1]]
    else:
        # Preserve the selector's genuine intercept-only audit without adding a
        # fake transformed feature.  The maximum-likelihood intercept predicts
        # the (optionally weighted) training retention prevalence everywhere.
        prior = float(
            np.average(y_train, weights=cache.sample_weight)
            if cache.sample_weight is not None
            else np.mean(y_train)
        )
        p_validation = np.full(len(y_validation), prior, dtype=float)
        convergence_messages = []
    metrics = compute_metrics(
        cache.fold.validation,
        p_validation,
        target_col=cache.fold.target_col,
        policy_id_col=cache.fold.policy_id_col,
        policy_inception_col=cache.fold.policy_inception_col,
    )
    add_derived_metrics(metrics)
    metrics.update(
        capacity_metrics(
            cache.fold.validation[cache.fold.target_col].astype(int).to_numpy(),
            p_validation,
        )
    )
    metrics.update(
        lift_spread_metrics(
            y_validation,
            p_validation,
            row_id=pd.util.hash_pandas_object(
                cache.fold.validation.index,
                index=False,
            ).to_numpy(dtype=np.uint64),
        )
    )
    calibration_intercept, calibration_slope = calibration_parameters(
        y_validation,
        p_validation,
    )
    metrics.update(
        calibration_intercept=calibration_intercept,
        calibration_slope=calibration_slope,
        mean_predicted_retention=float(np.mean(p_validation)),
        actual_retention_rate=float(np.mean(y_validation)),
        predicted_minus_actual_retention=(
            float(np.mean(p_validation) - np.mean(y_validation))
        ),
    )


    if include_train_diagnostics:
        p_train = (
            model.predict_proba(x_train)[:, class_index[1]]
            if model is not None
            else np.full(len(y_train), prior, dtype=float)
        )
        train_metrics = compute_metrics(
            cache.fold.train,
            p_train,
            target_col=cache.fold.target_col,
            policy_id_col=cache.fold.policy_id_col,
            policy_inception_col=cache.fold.policy_inception_col,
        )
        add_derived_metrics(train_metrics)
        train_capacity = capacity_metrics(
            cache.fold.train[cache.fold.target_col].astype(int).to_numpy(),
            p_train,
        )
        for name in (
            "retention_roc_auc",
            "churn_average_precision",
            "churn_ap_skill",
            "top10_lift",
            "log_loss",
            "brier_score",
            "calibration_gap",
        ):
            metrics[f"train_{name}"] = train_metrics[name]
        metrics["train_mean_capacity_lift_5_20"] = train_capacity[
            "mean_capacity_lift_5_20"
        ]
        metrics["train_minus_validation_auc_gap"] = (
            float(train_metrics["retention_roc_auc"])
            - float(metrics["retention_roc_auc"])
        )
        metrics["train_minus_validation_ap_skill_gap"] = (
            float(train_metrics["churn_ap_skill"])
            - float(metrics["churn_ap_skill"])
        )
        metrics["train_minus_validation_lift10_gap"] = (
            float(train_metrics["top10_lift"])
            - float(metrics["top10_lift"])
        )


    coefficients = (
        np.asarray(model.coef_).ravel()
        if model is not None
        else np.asarray([], dtype=float)
    )
    sample_weight = cache.sample_weight
    metrics.update(
        fold=int(cache.fold.fold_number),
        train_rows=len(cache.fold.train),
        validation_rows=len(cache.fold.validation),
        raw_feature_count=len(features),
        transformed_column_count=x_train.shape[1],
        nonzero_coefficient_count=int((np.abs(coefficients) > 1e-10).sum()),
        maximum_absolute_coefficient=(
            float(np.max(np.abs(coefficients))) if len(coefficients) else 0.0
        ),
        candidate_id=spec["id"],
        model_family="ridge",
        C=float(spec["C"]),
        l1_ratio=np.nan,
        solver="lbfgs",
        class_weight="null",
        training_weighted=sample_weight is not None,
        sample_weight_min=(
            float(np.min(sample_weight)) if sample_weight is not None else np.nan
        ),
        sample_weight_max=(
            float(np.max(sample_weight)) if sample_weight is not None else np.nan
        ),
        sample_weight_mean=(
            float(np.mean(sample_weight)) if sample_weight is not None else np.nan
        ),
        history_identity=cache.history_identity,
        converged=not convergence_messages,
        convergence_message=" | ".join(convergence_messages),
        elapsed_seconds=time.perf_counter() - started,
    )
    del x_train, x_validation, model
    return metrics




class FitCheckpoint:
    """Durable, constant-write checkpoint storage for individual model fits.


    Each fit is stored as one JSON payload behind a unique ``fit_key``.  This
    avoids a wide, evolving SQL schema while making every save a single keyed
    SQLite upsert instead of an O(number-of-fits) full-CSV rewrite.


    CSV checkpoints are intentionally not migrated: the immutable run
    fingerprint contains the code identity, so a run created by the former
    checkpoint implementation is not compatible with this one.
    """


    _SCHEMA_VERSION = "1"
    _DATABASE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


    def __init__(self, path: Path):
        self.path = Path(path).expanduser().resolve()
        if self.path.suffix.lower() not in self._DATABASE_SUFFIXES:
            raise ValueError(
                "FitCheckpoint requires a .db, .sqlite, or .sqlite3 path; "
                f"received {self.path}."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)


        try:
            self._connection = sqlite3.connect(
                self.path,
                timeout=30.0,
                isolation_level=None,
            )
            self._configure_database()
            self._initialize_schema()
        except Exception:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise


    def _configure_database(self) -> None:
        self._connection.execute("PRAGMA busy_timeout = 30000")
        # Pipeline outputs are run-local files. WAL avoids creating/deleting a
        # rollback journal for every one of the thousands of tiny fit records.
        # Keep the project output on a local drive; SQLite WAL is not intended
        # for a database opened concurrently across different network hosts.
        self._connection.execute("PRAGMA journal_mode = WAL")
        # NORMAL keeps the database transactionally consistent across process
        # interruption without forcing a disk sync after every tiny row. A
        # machine power loss may require repeating the most recent fits, which
        # is preferable to adding hours of checkpoint I/O to a smoke run.
        self._connection.execute("PRAGMA synchronous = NORMAL")
        self._connection.execute("PRAGMA wal_autocheckpoint = 1000")


    def _initialize_schema(self) -> None:
        self._connection.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS checkpoint_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS fit_results (
                fit_key TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL
            );
            INSERT INTO checkpoint_metadata(key, value)
                VALUES ('schema_version', '1')
                ON CONFLICT(key) DO NOTHING;
            COMMIT;
            """
        )
        row = self._connection.execute(
            "SELECT value FROM checkpoint_metadata WHERE key = 'schema_version'"
        ).fetchone()
        if row is None or str(row[0]) != self._SCHEMA_VERSION:
            found = None if row is None else row[0]
            raise RuntimeError(
                f"Unsupported fit checkpoint schema {found!r} at {self.path}."
            )


    @staticmethod
    def _serialized_row(row: dict) -> tuple[str, str, str]:
        if "fit_key" not in row or not str(row["fit_key"]).strip():
            raise ValueError("Checkpoint rows require a non-empty fit_key.")
        if "status" not in row or not str(row["status"]).strip():
            raise ValueError("Checkpoint rows require a non-empty status.")
        payload = jsonable(row)
        if not isinstance(payload, dict):
            raise TypeError("Checkpoint payload must serialize to a dictionary.")
        status = str(payload["status"])
        if status not in {"completed", "failed"}:
            raise ValueError(
                "Checkpoint status must be 'completed' or 'failed'; "
                f"received {status!r}."
            )
        return (
            str(payload["fit_key"]),
            status,
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )


    def _upsert_many(self, rows: list[dict]) -> None:
        serialized = [self._serialized_row(row) for row in rows]
        if not serialized:
            return
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._connection.executemany(
                """
                INSERT INTO fit_results(
                    fit_key, status, payload_json, updated_at_utc
                ) VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                ON CONFLICT(fit_key) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    updated_at_utc = excluded.updated_at_utc
                """,
                serialized,
            )
            self._connection.execute("COMMIT")
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise


    def completed(self, fit_key_value: str) -> dict | None:
        match = self._connection.execute(
            """
            SELECT payload_json
            FROM fit_results
            WHERE fit_key = ? AND status = 'completed'
            """,
            (str(fit_key_value),),
        ).fetchone()
        if match is None:
            return None
        try:
            payload = json.loads(match[0])
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Invalid checkpoint payload for fit {fit_key_value}."
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Checkpoint payload for fit {fit_key_value} is not an object."
            )
        if (
            str(payload.get("fit_key", "")) != str(fit_key_value)
            or str(payload.get("status", "")) != "completed"
        ):
            raise RuntimeError(
                f"Checkpoint payload identity is invalid for fit {fit_key_value}."
            )
        raw_result_version = payload.get("result_schema_version")
        if raw_result_version is None:
            # A legitimate pre-fix checkpoint row.  Its model fit may have
            # succeeded, but it cannot populate the governed history-screen
            # lift-spread outputs, so it must be recomputed.
            return None
        try:
            result_version = int(raw_result_version)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Checkpoint payload for fit {fit_key_value} has an invalid "
                "result_schema_version."
            ) from exc
        if result_version < FIT_RESULT_SCHEMA_VERSION:
            return None
        if result_version > FIT_RESULT_SCHEMA_VERSION:
            raise RuntimeError(
                f"Checkpoint payload for fit {fit_key_value} uses newer result "
                f"schema {result_version}; this code supports "
                f"{FIT_RESULT_SCHEMA_VERSION}."
            )
        missing = [
            column
            for column in CHECKPOINT_REQUIRED_RESULT_COLUMNS
            if column not in payload
        ]
        if missing:
            # Treat an incomplete current-version result like any other stale
            # fit so the caller's normal fit budget and atomic upsert repair it.
            return None
        return payload


    def upsert(self, row: dict) -> None:
        self._upsert_many([row])


    def close(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is None:
            return
        self._connection = None
        connection.close()


    def remove(self) -> list[Path]:
        """Best-effort cleanup after success; return paths still on disk.


        A transient Windows scanner/preview lock must not turn an otherwise
        durable, successfully completed model run into a failed run.  Callers
        can report the returned paths as harmless internal leftovers.
        """


        try:
            self.close()
        except Exception:
            # Continue trying every file even if a transient sharing lock
            # interfered with connection cleanup.
            pass
        candidates = (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
            Path(f"{self.path}-journal"),
        )
        for checkpoint_file in candidates:
            try:
                checkpoint_file.unlink(missing_ok=True)
            except OSError:
                pass
        return [path for path in candidates if path.exists()]


    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Interpreter shutdown and exception unwinding must not be masked
            # by best-effort connection cleanup.
            pass




def fit_key(
    context: str,
    fold_number: int,
    features: list[str],
    spec: dict,
    include_train_diagnostics: bool,
    history_identity: str = "",
) -> str:
    value = {
        "context": context,
        "fold": int(fold_number),
        "feature_hash": feature_hash(features),
        "candidate_id": spec["id"],
        "diagnostics": bool(include_train_diagnostics),
        "history_identity": str(history_identity).strip(),
    }
    return sha256_text(json.dumps(value, sort_keys=True))




def checkpointed_fit(
    store: FitCheckpoint,
    context: str,
    phase: str,
    plan_label: str,
    cache: FoldMatrixCache,
    features: list[str],
    spec: dict,
    include_train_diagnostics: bool = False,
) -> dict:
    key = fit_key(
        context,
        cache.fold.fold_number,
        features,
        spec,
        include_train_diagnostics,
        cache.history_identity,
    )
    cached = store.completed(key)
    if cached is not None:
        cached_diagnostics = str(cached.get("diagnostics", "")).strip().lower()
        expected_identity = {
            "fit_key": key,
            "context": context,
            "feature_hash": feature_hash(features),
            "features": ";".join(features),
            "fold": int(cache.fold.fold_number),
            "candidate_id": str(spec["id"]),
            "diagnostics": bool(include_train_diagnostics),
            "history_identity": cache.history_identity,
        }
        actual_identity = {
            "fit_key": str(cached.get("fit_key", "")),
            "context": str(cached.get("context", "")),
            "feature_hash": str(cached.get("feature_hash", "")),
            "features": str(cached.get("features", "")),
            "fold": int(cached.get("fold", -1)),
            "candidate_id": str(cached.get("candidate_id", "")),
            "diagnostics": cached_diagnostics in {"1", "true", "t", "yes"},
            "history_identity": str(cached.get("history_identity", "")),
        }
        mismatches = {
            name: (expected_identity[name], actual_identity[name])
            for name in expected_identity
            if expected_identity[name] != actual_identity[name]
        }
        if mismatches:
            raise RuntimeError(
                "Checkpoint fit identity does not exactly match the requested "
                f"fit for key {key}: {mismatches}"
            )
        print(
            f"RESUME {phase} {plan_label} fold {cache.fold.fold_number} "
            f"{spec['id']}",
            flush=True,
        )
        return cached
    metadata = {
        "fit_key": key,
        "status": "failed",
        "result_schema_version": FIT_RESULT_SCHEMA_VERSION,
        "context": context,
        "phase": phase,
        "plan_label": plan_label,
        "feature_hash": feature_hash(features),
        "features": ";".join(features),
        "fold": int(cache.fold.fold_number),
        "candidate_id": spec["id"],
        "diagnostics": bool(include_train_diagnostics),
        "history_identity": cache.history_identity,
    }
    try:
        result = evaluate_model(
            cache,
            features,
            spec,
            include_train_diagnostics,
        )
    except Exception as exc:
        store.upsert({**metadata, "error": f"{type(exc).__name__}: {exc}"})
        raise
    row = {**metadata, **result, "status": "completed", "error": ""}
    store.upsert(row)
    print(
        f"SAVED {phase} {plan_label} fold {cache.fold.fold_number} "
        f"{spec['id']} in {float(result['elapsed_seconds']):.1f}s",
        flush=True,
    )
    return row




SUMMARY_METRICS = (
    "retention_roc_auc",
    "churn_average_precision",
    "churn_ap_skill",
    "top10_lift",
    "mean_capacity_lift_5_20",
    "log_loss",
    "brier_score",
    "calibration_gap",
)




def summarize_folds(results: pd.DataFrame) -> dict:
    row = {"valid_folds": len(results)}
    for column in SUMMARY_METRICS:
        values = pd.to_numeric(results[column], errors="coerce")
        row[f"mean_{column}"] = float(values.mean())
        row[f"min_{column}"] = float(values.min())
        row[f"std_{column}"] = float(values.std(ddof=1))
    row["converged_folds"] = int(
        results.converged.astype(str).str.lower().isin({"true", "1"}).sum()
    )
    return row
