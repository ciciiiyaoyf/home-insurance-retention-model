"""Descriptive actual-versus-predicted trends for a provisional final refit.


This module only writes artifacts from an already aggregated monthly table. It
does not fit or score a model. Its outputs are retrospective diagnostics and
must never be presented as chronological out-of-fold performance evidence.
"""


from __future__ import annotations


import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


import numpy as np
import pandas as pd




YEAR_MONTH_COLUMN = "YEAR_MONTH"




def _pyplot() -> Any:
    try:
        import matplotlib


        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise ImportError(
            "Final-refit trend plots require matplotlib. Run: "
            "python -m pip install 'matplotlib>=3.8,<4'"
        ) from exc
    return plt




def _strict_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    return value




def _selected_history_window(
    selected_history: Mapping[str, Any],
) -> tuple[int, str]:
    kind = str(selected_history.get("kind", "")).strip().lower()
    history_id = str(selected_history.get("history_id", "")).strip()
    if not history_id:
        raise ValueError("Selected history must contain a non-empty history_id.")
    try:
        months = int(selected_history["months"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Selected history must contain positive integer months."
        ) from exc
    if months < 1:
        raise ValueError("Selected history months must be positive.")
    if kind == "recent":
        return months, f"most recent {months} months"
    if kind == "decay":
        try:
            maximum = int(selected_history["max_months"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Selected decay history must contain positive max_months."
            ) from exc
        if maximum < 1 or maximum < months:
            raise ValueError(
                "Selected decay max_months must be positive and at least its "
                "half-life."
            )
        return maximum, f"{maximum}-month cap; {months}-month half-life"
    raise ValueError(f"Unsupported selected history kind: {kind!r}")




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




def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp_", suffix=".json", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                _strict_json(payload),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)




def _save_figure_atomic(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp_", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        fig.savefig(temporary, dpi=180, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)




def _plot_final_refit_descriptive_trend(
    monthly: pd.DataFrame,
    destination: Path,
    *,
    family: str,
    scope_label: str,
    training_start: pd.Period,
    training_end: pd.Period,
) -> None:
    """Plot fitted/backcast values without presenting them as validation."""


    plt = _pyplot()
    from matplotlib.ticker import PercentFormatter


    x = pd.to_datetime(monthly[YEAR_MONTH_COLUMN].astype(str) + "-01")
    fig, (trend_axis, volume_axis) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    trend_axis.plot(
        x,
        monthly["actual_retention"],
        color="#005ea8",
        linewidth=2.1,
        label="Actual retention",
    )
    trend_axis.plot(
        x,
        monthly["predicted_retention"],
        color="#ef6c00",
        linewidth=2.1,
        label="Final-refit predicted retention",
    )
    if monthly[YEAR_MONTH_COLUMN].iloc[0] < str(training_start):
        trend_axis.axvline(
            training_start.start_time,
            color="#555555",
            linestyle="--",
            linewidth=1.4,
            label="Final-refit training window begins",
        )
    if monthly[YEAR_MONTH_COLUMN].iloc[-1] > str(training_end):
        trend_axis.axvline(
            (training_end + 1).start_time,
            color="#555555",
            linestyle=":",
            linewidth=1.4,
            label="Final-refit training window ends",
        )
    trend_axis.set_ylabel("Retention rate")
    trend_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    trend_axis.set_title(
        f"{family.title()} final-refit descriptive trend - {scope_label}\n"
        "NOT OOF PERFORMANCE"
    )
    trend_axis.grid(alpha=0.25)
    trend_axis.legend(loc="best")
    volume_axis.bar(
        x,
        monthly["policies"],
        width=24,
        color="#8da0cb",
        alpha=0.75,
    )
    volume_axis.set_ylabel("Policies")
    volume_axis.set_xlabel("Month-1 evaluation cohort")
    volume_axis.grid(axis="y", alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save_figure_atomic(fig, destination)
    plt.close(fig)




def write_final_refit_descriptive_trend_artifacts(
    *,
    monthly: pd.DataFrame,
    output_dir: str | Path,
    model_family: str,
    cohort_col: str,
    selected_history: Mapping[str, Any],
    training_start: str | pd.Timestamp | pd.Period,
    training_end: str | pd.Timestamp | pd.Period,
) -> dict[str, Any]:
    """Write long fitted/backcast trends that are explicitly not performance.


    ``monthly`` must already contain aggregated predictions from the provisional
    final-refit model. Rows before ``training_start`` are retrospective
    backcasts, rows within the bounds belong to selected training-window
    cohorts, and later rows are post-training pre-holdout diagnostics. A
    boundary cohort may include rows excluded from fitting by label maturity;
    calendar bounds do not prove that every row was a training observation.
    Exact membership columns supplied by the caller are preserved in the CSV.
    This function does not score data.
    """


    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    family = str(model_family).strip().lower()
    if not family or any(
        not (character.isalnum() or character in {"_", "-"})
        for character in family
    ):
        raise ValueError(f"Invalid model-family artifact prefix: {model_family!r}")
    required = {
        YEAR_MONTH_COLUMN,
        "policies",
        "actual_retention",
        "predicted_retention",
    }
    missing = sorted(required.difference(monthly.columns))
    if missing:
        raise KeyError(f"Final-refit monthly summary is missing: {missing}")
    if monthly.empty:
        raise ValueError("Final-refit monthly summary cannot be empty.")


    frame = monthly.copy()
    try:
        periods = pd.PeriodIndex(frame[YEAR_MONTH_COLUMN].astype(str), freq="M")
        start_period = pd.Period(training_start, freq="M")
        end_period = pd.Period(training_end, freq="M")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Final-refit trend months and training bounds must be valid months."
        ) from exc
    if periods.isna().any() or pd.isna(start_period) or pd.isna(end_period):
        raise ValueError(
            "Final-refit trend months and training bounds must be valid months."
        )
    if start_period > end_period:
        raise ValueError("Final-refit training_start must not follow training_end.")
    frame[YEAR_MONTH_COLUMN] = periods.astype(str)
    frame = frame.sort_values(YEAR_MONTH_COLUMN, kind="stable").reset_index(drop=True)
    if frame[YEAR_MONTH_COLUMN].duplicated().any():
        raise ValueError(
            "Final-refit monthly summary must contain unique cohort months."
        )
    sorted_periods = pd.PeriodIndex(frame[YEAR_MONTH_COLUMN], freq="M")
    expected_months = pd.period_range(
        sorted_periods.min(), sorted_periods.max(), freq="M"
    )
    if not sorted_periods.equals(expected_months):
        raise ValueError(
            "Final-refit monthly summary must contain a continuous month history."
        )


    policies = pd.to_numeric(frame["policies"], errors="raise").astype(float)
    if (
        not np.isfinite(policies).all()
        or (policies <= 0).any()
        or not np.equal(policies, np.floor(policies)).all()
    ):
        raise ValueError("Final-refit monthly policies must be positive integers.")
    frame["policies"] = policies.astype("int64")
    for column in ("actual_retention", "predicted_retention"):
        values = pd.to_numeric(frame[column], errors="raise").astype(float)
        if not np.isfinite(values).all() or not values.between(0.0, 1.0).all():
            raise ValueError(f"{column} must contain finite probabilities in [0, 1].")
        frame[column] = values


    if start_period not in sorted_periods or end_period not in sorted_periods:
        raise ValueError(
            "Final-refit training bounds must both exist in the monthly summary."
        )
    window_months, history_description = _selected_history_window(selected_history)
    selected_mask = (sorted_periods >= start_period) & (sorted_periods <= end_period)
    selected = frame.loc[selected_mask].copy()
    if len(selected) != window_months:
        raise ValueError(
            "Exact final-refit training bounds do not match selected history: "
            f"expected={window_months} months; observed={len(selected)}."
        )
    frame["prediction_role"] = np.select(
        [sorted_periods < start_period, sorted_periods <= end_period],
        ["BACKCAST_PRE_TRAINING", "FINAL_REFIT_TRAINING_WINDOW"],
        default="POST_TRAINING_PREHOLDOUT_DIAGNOSTIC",
    )
    selected = frame.loc[selected_mask].copy()


    prefix = f"{family}_final_refit_actual_vs_predicted"
    full_csv = output / f"{prefix}_full_history.csv"
    full_plot = output / f"{prefix}_full_history.png"
    selected_csv = output / f"{prefix}_selected_training_history.csv"
    selected_plot = output / f"{prefix}_selected_training_history.png"
    manifest_path = output / "final_refit_actual_vs_predicted_manifest.json"
    _atomic_csv(full_csv, frame)
    _atomic_csv(selected_csv, selected)
    _plot_final_refit_descriptive_trend(
        frame,
        full_plot,
        family=family,
        scope_label=(
            "full pre-holdout history (backcasts, training values, and later "
            "diagnostics)"
        ),
        training_start=start_period,
        training_end=end_period,
    )
    _plot_final_refit_descriptive_trend(
        selected,
        selected_plot,
        family=family,
        scope_label=(
            f"selected training-window cohorts ({history_description})\n"
            "May include non-training boundary rows"
        ),
        training_start=start_period,
        training_end=end_period,
    )
    artifacts = [full_csv, full_plot, selected_csv, selected_plot]
    manifest = {
        "contract": "retention_final_refit_descriptive_trend_v1",
        "model_family": family,
        "scope": "provisional_final_refit_retrospective_preholdout_description",
        "evaluation_scope": "FINAL_REFIT_DESCRIPTIVE_NOT_OOF",
        "performance_evidence": False,
        "official_performance_source": "outer_oof_temporal_stability",
        "predictions_included": True,
        "prediction_interpretation": (
            "selected training-window cohorts can include both fitted observations "
            "and non-training boundary rows; calendar bounds alone do not establish "
            "exact training membership; retrospective backcasts before that window; "
            "post-training pre-holdout diagnostic scores after that window; "
            "never OOF performance"
        ),
        "sealed_holdout_scored": False,
        "final_holdout_rows_scored": 0,
        "cohort_column": str(cohort_col),
        "cohort_month_definition": f"calendar month of {cohort_col}",
        "full_history_start": str(frame[YEAR_MONTH_COLUMN].iloc[0]),
        "full_history_end": str(frame[YEAR_MONTH_COLUMN].iloc[-1]),
        "full_history_months": int(len(frame)),
        "selected_history": dict(selected_history),
        "training_start": str(start_period),
        "training_end": str(end_period),
        "training_months": int(len(selected)),
        "training_bounds_inclusive": True,
        "plot_warning": "NOT OOF PERFORMANCE",
        "artifacts": [path.name for path in artifacts],
    }
    _atomic_json(manifest_path, manifest)
    return {**manifest, "manifest": manifest_path.name}




def _validated_full_history_comparison(
    monthly: pd.DataFrame,
    *,
    final_holdout_start: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Validate monthly evidence before combining either model's role strata.


    A month may include training and non-training observations for the same
    family, particularly in a partially mature boundary month. The caller must
    classify actual row membership before aggregation; calendar bounds alone
    are not enough to determine whether an observation was used for fitting.
    """


    required = {
        YEAR_MONTH_COLUMN,
        "model_family",
        "population_role",
        "policies",
        "retained_policies",
        "predicted_retained_policies",
    }
    missing = sorted(required.difference(monthly.columns))
    if missing:
        raise KeyError(f"Full-history model comparison is missing: {missing}")
    if monthly.empty:
        raise ValueError("Full-history model comparison cannot be empty.")
    frame = monthly.loc[:, sorted(required)].copy()
    try:
        periods = pd.PeriodIndex(frame[YEAR_MONTH_COLUMN].astype(str), freq="M")
        holdout_start = pd.Timestamp(final_holdout_start)
    except (TypeError, ValueError) as exc:
        raise ValueError("Comparison months and final_holdout_start must be valid dates.") from exc
    if periods.isna().any() or pd.isna(holdout_start):
        raise ValueError("Comparison months and final_holdout_start must be valid dates.")
    # This plot is deliberately pre-holdout only. Even part of the holdout's
    # first calendar month is refused because a monthly table cannot establish
    # the eligibility of individual dates in that month.
    if (periods >= holdout_start.to_period("M")).any():
        raise ValueError("Full-history comparison must exclude all final-holdout months.")
    frame[YEAR_MONTH_COLUMN] = periods.astype(str)
    if frame["model_family"].isna().any() or frame["population_role"].isna().any():
        raise ValueError("Model family and population role cannot be missing.")
    frame["model_family"] = frame["model_family"].astype(str).str.strip().str.lower()
    frame["population_role"] = frame["population_role"].astype(str).str.strip()
    if set(frame["model_family"]) != {"logistic", "xgboost"}:
        raise ValueError("Full-history comparison requires exactly Logistic and XGBoost.")
    if frame["population_role"].eq("").any():
        raise ValueError("Population roles must be non-empty.")
    keys = [YEAR_MONTH_COLUMN, "model_family", "population_role"]
    if frame.duplicated(keys).any():
        raise ValueError("Monthly model/population-role strata must be unique.")
    for column in ("policies", "retained_policies", "predicted_retained_policies"):
        values = pd.to_numeric(frame[column], errors="raise").astype(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must be finite.")
        if column != "predicted_retained_policies":
            if not np.equal(values, np.floor(values)).all():
                raise ValueError(f"{column} must contain integer counts.")
            frame[column] = values.astype("int64")
        else:
            frame[column] = values
    if (frame["policies"] <= 0).any():
        raise ValueError("Every monthly role stratum must have a positive row count.")
    for column in ("retained_policies", "predicted_retained_policies"):
        if ((frame[column] < 0) | (frame[column] > frame["policies"])).any():
            raise ValueError(f"{column} must lie between zero and its row count.")
    frame = frame.sort_values(keys, kind="stable").reset_index(drop=True)
    totals = frame.groupby([YEAR_MONTH_COLUMN, "model_family"], sort=True)[
        ["policies", "retained_policies", "predicted_retained_policies"]
    ].sum()
    logistic = totals.xs("logistic", level="model_family")
    xgboost = totals.xs("xgboost", level="model_family")
    if not logistic.index.equals(xgboost.index):
        raise ValueError("Both models must score exactly the same observed cohort months.")
    if not np.array_equal(
        logistic[["policies", "retained_policies"]].to_numpy(),
        xgboost[["policies", "retained_policies"]].to_numpy(),
    ):
        raise ValueError("Both models must have identical monthly row and actual-retained counts.")
    combined = logistic[["policies", "retained_policies"]].copy()
    combined["actual_retention"] = combined["retained_policies"] / combined["policies"]
    for family, source in (("logistic", logistic), ("xgboost", xgboost)):
        combined[f"{family}_predicted_retained_policies"] = source["predicted_retained_policies"]
        combined[f"{family}_predicted_retention"] = (
            source["predicted_retained_policies"] / combined["policies"]
        )
    return combined.reset_index(), frame, holdout_start




def _plot_full_history_model_comparison(monthly: pd.DataFrame, destination: Path) -> None:
    """Plot two frozen final-fit models on one shared historical population."""


    plt = _pyplot()
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter, PercentFormatter


    periods = pd.PeriodIndex(monthly[YEAR_MONTH_COLUMN], freq="M")
    # Missing calendar months have no observations: leave a break, rather than
    # interpolate rates or treat a missing month as a zero-retention month.
    expected = pd.period_range(periods.min(), periods.max(), freq="M")
    plotted = monthly.set_index(periods).reindex(expected)
    x = expected.to_timestamp()
    fig, (trend_axis, volume_axis) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    try:
        for column, label, color in (
            ("actual_retention", "Actual retention", "#252525"),
            ("logistic_predicted_retention", "Logistic predicted", "#1267b1"),
            ("xgboost_predicted_retention", "XGBoost predicted", "#ed6a24"),
        ):
            trend_axis.plot(x, plotted[column], label=label, color=color, linewidth=2.0)
        fig.suptitle("Actual vs predicted retention - full history", fontsize=15, fontweight="bold")
        trend_axis.set_title("Historical fit/backcast; not out-of-fold validation", fontsize=10, color="#555555")
        trend_axis.set_ylabel("Retention rate")
        trend_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
        trend_axis.grid(alpha=0.22)
        trend_axis.legend(loc="best")
        volume_axis.bar(x, plotted["policies"], width=24, color="#9ebbd4", alpha=0.8)
        volume_axis.set_ylabel("Policy-term rows")
        volume_axis.set_xlabel("Month-1 cohort month")
        volume_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        volume_axis.grid(axis="y", alpha=0.2)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=12, interval_multiples=True)
        volume_axis.xaxis.set_major_locator(locator)
        volume_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        volume_axis.set_xlim(x.min() - pd.Timedelta(days=18), x.max() + pd.Timedelta(days=18))
        fig.autofmt_xdate(rotation=35, ha="right")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _save_figure_atomic(fig, destination)
    finally:
        plt.close(fig)




def write_full_history_model_comparison_artifacts(
    *,
    monthly: pd.DataFrame,
    output_dir: str | Path,
    cohort_col: str,
    final_holdout_start: str | pd.Timestamp,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a full-history Logistic/XGBoost comparison from monthly sums.


    The caller must score the same pre-holdout rows with both already fitted
    models and verify row identities before aggregation. ``monthly`` contains
    one row per month, model family and actual training-membership role, with
    ``policies``, ``retained_policies`` and summed retention probabilities in
    ``predicted_retained_policies``. This writer validates monthly populations,
    but matching aggregate counts alone cannot prove identical row identities.


    Counts, rather than precomputed stratum rates, ensure partially mature
    months and different family-specific training windows are weighted exactly.
    No refitting, scoring, smoothing, model reselection or holdout access occurs.
    """


    combined, by_role, holdout_start = _validated_full_history_comparison(
        monthly, final_holdout_start=final_holdout_start,
    )
    output = Path(output_dir).expanduser().resolve()
    prefix = "model_retention_trend_full_history"
    plot_path = output / f"{prefix}.png"
    csv_path = output / f"{prefix}.csv"
    roles_path = output / f"{prefix}_by_population_role.csv"
    manifest_path = output / f"{prefix}_manifest.json"
    _atomic_csv(csv_path, combined)
    _atomic_csv(roles_path, by_role)
    _plot_full_history_model_comparison(combined, plot_path)
    periods = pd.PeriodIndex(combined[YEAR_MONTH_COLUMN], freq="M")
    missing_months = pd.period_range(periods.min(), periods.max(), freq="M").difference(periods)
    manifest = {
        "contract": "retention_full_history_model_comparison_v1",
        "model_families": ["logistic", "xgboost"],
        "evaluation_scope": "full_history_descriptive",
        "performance_evidence": False,
        "prediction_interpretation": (
            "Frozen final-fit models describe identical pre-holdout observations; "
            "training-membership roles are recorded separately for each model. "
            "These fitted values and retrospective scores are not out-of-fold validation."
        ),
        "plot_title": "Actual vs predicted retention - full history",
        "plot_subtitle": "Historical fit/backcast; not out-of-fold validation",
        "aggregation": "sum retention probabilities / shared monthly policy-term row count",
        "smoothing_applied": False,
        "calendar_months_without_rows": missing_months.astype(str).tolist(),
        "missing_month_display": "line break; no interpolation",
        "cohort_column": str(cohort_col),
        "full_history_start": str(periods.min()),
        "full_history_end": str(periods.max()),
        "full_history_observed_months": int(len(combined)),
        "rows_per_model": int(combined["policies"].sum()),
        "monthly_population_counts_match": True,
        "row_identity_verification": "caller responsibility before monthly aggregation",
        "final_holdout_start": holdout_start.isoformat(),
        "sealed_holdout_scored": False,
        "final_holdout_rows_scored": 0,
        "population_role_rows": {
            family: {
                str(role): int(rows)
                for role, rows in by_role.loc[by_role["model_family"].eq(family)]
                .groupby("population_role")["policies"].sum().items()
            }
            for family in ("logistic", "xgboost")
        },
        "provenance": dict(provenance or {}),
        "artifacts": [plot_path.name, csv_path.name, roles_path.name],
    }
    _atomic_json(manifest_path, manifest)
    return {**manifest, "manifest": manifest_path.name}




__all__ = [
    "YEAR_MONTH_COLUMN",
    "write_final_refit_descriptive_trend_artifacts",
    "write_full_history_model_comparison_artifacts",
]
