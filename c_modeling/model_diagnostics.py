"""
Model diagnostics consume chronological out-of-fold predictions. Separate
actual-only business-trend helpers consume the contract-filtered modeling
population and exclude the sealed holdout by cohort boundary. Nothing in this
module fits a model or scores the sealed final holdout.
"""


from __future__ import annotations


import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)




YEAR_CONTROL_FEATURE = "COHORT_CALENDAR_YEAR"
WITH_YEAR_COLUMN = "predicted_retention_with_year"
ACTUAL_COLUMN = "actual_retention"
YEAR_MONTH_COLUMN = "YEAR_MONTH"




def _pyplot() -> Any:
    try:
        import matplotlib


        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise ImportError(
            "python -m pip install 'matplotlib>=3.8,<4'"
        ) from exc
    return plt




def _shap_package() -> Any:
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise ImportError(
            "XGBoost explanations require the SHAP package. Run: "
            "python -m pip install 'shap>=0.46,<1'"
        ) from exc
    return shap




def _strict_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value




def _stable_top_decile(
    actual_retention: np.ndarray,
    predicted_retention: np.ndarray,
    row_id: np.ndarray,
) -> tuple[float, float, float, int]:
    y_churn = 1 - np.asarray(actual_retention, dtype=int)
    p_churn = 1.0 - np.asarray(predicted_retention, dtype=float)
    ranked = pd.DataFrame(
        {"y": y_churn, "score": p_churn, "row_id": row_id}
    ).sort_values(
        ["score", "row_id"], ascending=[False, True], kind="mergesort"
    )
    top_n = max(1, int(math.ceil(len(ranked) * 0.10)))
    top = ranked.iloc[:top_n]
    prevalence = float(ranked["y"].mean())
    precision = float(top["y"].mean())
    recall = float(top["y"].sum() / max(float(ranked["y"].sum()), 1.0))
    lift = precision / prevalence if prevalence > 0.0 else np.nan
    return precision, recall, float(lift), top_n




def prediction_metrics(
    frame: pd.DataFrame,
    prediction_column: str,
) -> dict[str, Any]:
    """Compute the shared Logistic/XGBoost scorecard on aligned OOF rows."""


    required = {ACTUAL_COLUMN, prediction_column, "row_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Prediction frame is missing required columns: {missing}")
    y_retention = pd.to_numeric(frame[ACTUAL_COLUMN], errors="raise").astype(int)
    p_retention = np.clip(
        pd.to_numeric(frame[prediction_column], errors="raise").to_numpy(float),
        1e-7,
        1.0 - 1e-7,
    )
    if set(y_retention.unique()) != {0, 1}:
        raise ValueError("Diagnostic population must contain both target classes.")
    y_churn = 1 - y_retention.to_numpy()
    p_churn = 1.0 - p_retention
    prevalence = float(y_churn.mean())
    ap = float(average_precision_score(y_churn, p_churn))
    ap_skill = (ap - prevalence) / max(1e-12, 1.0 - prevalence)
    precision, recall, lift, top_n = _stable_top_decile(
        y_retention.to_numpy(),
        p_retention,
        frame["row_id"].astype(str).to_numpy(),
    )
    return {
        "rows": int(len(frame)),
        "retention_roc_auc": float(roc_auc_score(y_retention, p_retention)),
        "churn_average_precision": ap,
        "churn_prevalence_baseline": prevalence,
        "churn_ap_skill": float(ap_skill),
        "churn_precision_10pct": precision,
        "churn_capture_10pct": recall,
        "churn_recall_10pct": recall,
        "top10_lift": lift,
        "top_group_rows": int(top_n),
        "log_loss": float(log_loss(y_retention, p_retention, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_retention, p_retention)),
        "actual_retention_rate": float(y_retention.mean()),
        "mean_predicted_retention": float(p_retention.mean()),
        "calibration_gap": float(p_retention.mean() - y_retention.mean()),
    }




def validate_diagnostic_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "fold",
        "row_id",
        ACTUAL_COLUMN,
        WITH_YEAR_COLUMN,
        YEAR_CONTROL_FEATURE,
        YEAR_MONTH_COLUMN,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Calendar-year diagnostic frame is missing: {missing}")
    result = frame.copy()
    result[YEAR_CONTROL_FEATURE] = pd.to_numeric(
        result[YEAR_CONTROL_FEATURE], errors="raise"
    ).astype(int)
    result[YEAR_MONTH_COLUMN] = pd.PeriodIndex(
        result[YEAR_MONTH_COLUMN].astype(str), freq="M"
    ).astype(str)
    values = pd.to_numeric(result[WITH_YEAR_COLUMN], errors="raise")
    if values.isna().any() or not values.between(0.0, 1.0).all():
        raise ValueError(f"{WITH_YEAR_COLUMN} must contain finite probabilities.")
    result[WITH_YEAR_COLUMN] = values.astype(float)
    if result["row_id"].duplicated().any():
        raise ValueError("Chronological diagnostic predictions contain duplicate rows.")
    return result.sort_values([YEAR_MONTH_COLUMN, "row_id"], kind="stable")




def calendar_control_performance(frame: pd.DataFrame) -> pd.DataFrame:
    validated = validate_diagnostic_frame(frame)
    rows: list[dict[str, Any]] = []
    pooled = prediction_metrics(validated, WITH_YEAR_COLUMN)
    rows.append({"scope": "POOLED_OOF", "fold": 0, **pooled})
    for fold, part in validated.groupby("fold", sort=True):
        rows.append(
            {
                "scope": "FOLD",
                "fold": int(fold),
                **prediction_metrics(part, WITH_YEAR_COLUMN),
            }
        )
    return pd.DataFrame(rows)




def _group_summary(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    grouped = frame.groupby(group_column, observed=True, sort=True)
    summary = grouped.agg(
        rows=(ACTUAL_COLUMN, "size"),
        actual_retention=(ACTUAL_COLUMN, "mean"),
        predicted_retention=(WITH_YEAR_COLUMN, "mean"),
    ).reset_index()
    summary["actual_churn"] = 1.0 - summary["actual_retention"]
    summary["predicted_churn"] = 1.0 - summary["predicted_retention"]
    summary["calibration_gap"] = (
        summary["predicted_retention"] - summary["actual_retention"]
    )
    summary["population_fraction"] = summary["rows"] / float(summary["rows"].sum())
    return summary




def _is_continuous_plot_feature(
    values: pd.Series, configured_type: str | None = None
) -> bool:
    """Honor the feature contract before inferring numeric display values."""
    configured = str(configured_type or "").lower()
    if configured in {"discrete", "categorical", "category", "binary"}:
        return False
    if configured in {"continuous", "numeric"}:
        return True
    if isinstance(values.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(values):
        return False
    if pd.api.types.is_numeric_dtype(values):
        return True
    present = values.dropna()
    return bool(len(present) and pd.to_numeric(present, errors="coerce").notna().all())




def _numeric_plot_values(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )




def _format_numeric_axis(axis: Any, *, feature: str, values: np.ndarray) -> None:
    """At most a handful of numeric ticks; never one tick per observation."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator


    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    extent = float(np.max(np.abs(finite))) if finite.size else 0.0
    decimals = 0 if extent >= 100 else (1 if extent >= 10 else 2)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=3))
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{value:,.{decimals}f}")
    )
    axis.tick_params(axis="x", labelrotation=0, labelsize=9)
    if feature.upper() == "WRITTEN_PREM":
        # Presentation only: do not clip, replace, or otherwise mutate model inputs.
        right = max(float(finite.max()) * 1.05, 1.0) if finite.size else 1.0
        axis.set_xlim(0.0, right)




def _one_way_bins(
    values: pd.Series, *, bins: int = 10, continuous: bool = True
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    nonmissing = numeric.notna()
    if continuous and nonmissing.sum() and numeric[nonmissing].nunique() > min(12, bins):
        quantiles = pd.qcut(
            numeric[nonmissing],
            q=min(bins, int(numeric[nonmissing].nunique())),
            duplicates="drop",
        )
        interval_labels = [
            f"Q{position:02d}: {interval}"
            for position, interval in enumerate(quantiles.cat.categories, start=1)
        ]
        label_by_code = dict(enumerate(interval_labels))
        result = pd.Series("__MISSING__", index=values.index, dtype="string")
        result.loc[nonmissing] = (
            quantiles.cat.codes.map(label_by_code).astype("string").to_numpy()
        )
        category_order = [*interval_labels]
        if (~nonmissing).any():
            category_order.append("__MISSING__")
        return pd.Series(
            pd.Categorical(
                result.astype(str),
                categories=category_order,
                ordered=True,
            ),
            index=values.index,
        )
    categorical = values.astype("string").fillna("__MISSING__")
    counts = categorical.value_counts(dropna=False)
    keep = set(counts.head(20).index.astype(str))
    return categorical.map(lambda value: str(value) if str(value) in keep else "__OTHER__")




def one_way_summaries(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    feature_types: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validated = validate_diagnostic_frame(frame)
    outputs: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    ordered = list(dict.fromkeys([YEAR_CONTROL_FEATURE, *features]))
    for feature in ordered:
        if feature not in validated:
            audit.append({"feature": feature, "status": "MISSING_FROM_OOF_FRAME"})
            continue
        working = validated.copy()
        group_column = "_one_way_level"
        numeric_feature = feature != YEAR_CONTROL_FEATURE and _is_continuous_plot_feature(
            working[feature], (feature_types or {}).get(feature)
        )
        negative = np.zeros(len(working), dtype=bool)
        nonfinite = np.zeros(len(working), dtype=bool)
        if feature == YEAR_CONTROL_FEATURE:
            working[group_column] = working[feature].astype(str)
        elif numeric_feature:
            numeric = _numeric_plot_values(working[feature])
            nonfinite = ~np.isfinite(numeric)
            if feature.upper() == "WRITTEN_PREM":
                negative = np.isfinite(numeric) & (numeric < 0)
            visible_values = pd.Series(numeric, index=working.index).mask(nonfinite | negative)
            working[group_column] = _one_way_bins(visible_values).astype("string")
            working.loc[negative, group_column] = "__NEGATIVE_DISPLAY_EXCLUDED__"
            working.loc[nonfinite, group_column] = "__MISSING__"
            working["_numeric_display_value"] = visible_values
        else:
            working[group_column] = _one_way_bins(working[feature], continuous=False)
        summary = _group_summary(working, group_column).rename(
            columns={group_column: "level"}
        )
        summary["axis_kind"] = "numeric" if numeric_feature else "categorical"
        if numeric_feature:
            coordinates = working.groupby(group_column, observed=True)["_numeric_display_value"].agg(
                plot_x="median", plot_x_min="min", plot_x_max="max"
            ).reset_index().rename(columns={group_column: "level"})
            summary = summary.merge(coordinates, on="level", how="left", validate="one_to_one")
        summary.insert(0, "feature", feature)
        outputs.append(summary)
        audit.append(
            {
                "feature": feature,
                "status": "WRITTEN",
                "levels": int(len(summary)),
                "axis_kind": "numeric" if numeric_feature else "categorical",
                "input_rows": int(len(working)),
                "plot_rows": int(len(working) - negative.sum() - nonfinite.sum()),
                "negative_rows_hidden_from_plot": int(negative.sum()),
                "nonfinite_rows_hidden_from_plot": int(nonfinite.sum()),
                "raw_values_modified": False,
            }
        )
    table = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    return table, pd.DataFrame(audit)




def _artifact_slug(value: str) -> str:
    characters: list[str] = []
    separator_pending = False
    for character in str(value).strip().lower():
        if character.isalnum():
            if separator_pending and characters:
                characters.append("_")
            characters.append(character)
            separator_pending = False
        else:
            separator_pending = True
    return "".join(characters).strip("_") or "feature"




def _plot_one_way_summary(
    summary: pd.DataFrame,
    destination: Path,
    title: str,
) -> None:
    plt = _pyplot()
    from matplotlib.ticker import PercentFormatter


    feature = str(summary["feature"].iloc[0]) if "feature" in summary else ""
    numeric = "axis_kind" in summary and summary["axis_kind"].eq("numeric").all()
    hidden_rows = 0
    plotted = summary.copy()
    if numeric:
        visible = np.isfinite(pd.to_numeric(plotted["plot_x"], errors="coerce"))
        if feature.upper() == "WRITTEN_PREM":
            visible &= plotted["plot_x"].ge(0)
        hidden_rows = int(plotted.loc[~visible, "rows"].sum())
        plotted = plotted.loc[visible].sort_values("plot_x", kind="stable")
    levels = plotted["level"].astype(str).tolist()
    positions = plotted["plot_x"].to_numpy(float) if numeric else np.arange(len(levels))
    width = max(10.0, min(18.0, 6.0 + 0.55 * len(levels)))
    fig, (axis, volume_axis) = plt.subplots(
        2,
        1,
        figsize=(width, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axis.plot(
        positions,
        plotted["actual_retention"],
        marker="o",
        linewidth=2.1,
        label="Actual retention",
    )
    axis.plot(
        positions,
        plotted["predicted_retention"],
        marker="o",
        linewidth=1.9,
        label="Predicted retention",
    )
    axis.set_ylabel("Retention rate")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    bar_width = 0.8
    if numeric and len(positions) > 1:
        spacing = np.diff(positions)
        bar_width = 0.7 * np.minimum(np.r_[spacing[0], spacing], np.r_[spacing, spacing[-1]])
    volume_axis.bar(positions, plotted["rows"], width=bar_width, color="#8da0cb", alpha=0.8)
    volume_axis.set_ylabel("Policies")
    volume_axis.set_xlabel(f"{feature} (bin median)" if numeric else "Feature level")
    if numeric:
        range_values = np.r_[positions, plotted["plot_x_min"].to_numpy(float), plotted["plot_x_max"].to_numpy(float)]
        _format_numeric_axis(volume_axis, feature=feature, values=range_values)
        if hidden_rows:
            axis.text(0.01, 0.02, f"Display excludes {hidden_rows:,} rows with missing/nonfinite values"
                      + (" or negative premium." if feature.upper() == "WRITTEN_PREM" else "."),
                      transform=axis.transAxes, fontsize=8)
    else:
        volume_axis.set_xticks(positions)
        volume_axis.set_xticklabels(levels, rotation=40, ha="right", fontsize=8)
    volume_axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)




def _write_one_way_plot_artifacts(
    *,
    year: pd.DataFrame,
    one_way: pd.DataFrame,
    ordered_features: Sequence[str],
    output: Path,
    model_family: str,
    top_n: int,
) -> tuple[Path, list[Path], list[str]]:
    year_plot = output / f"{model_family}_calendar_year_control_one_way.png"
    year_plot_frame = year.rename(columns={YEAR_CONTROL_FEATURE: "level"})
    _plot_one_way_summary(
        year_plot_frame,
        year_plot,
        f"{model_family.title()} calendar-year control audit - inner OOF",
    )


    limit = max(0, int(top_n))
    candidates = [
        feature
        for feature in dict.fromkeys(str(value) for value in ordered_features)
        if feature != YEAR_CONTROL_FEATURE
    ]
    paths: list[Path] = []
    plotted_features: list[str] = []
    for feature in candidates:
        if len(plotted_features) >= limit:
            break
        summary = one_way.loc[one_way["feature"].eq(feature)]
        if summary.empty:
            continue
        destination = output / (
            f"{model_family}_one_way_{len(plotted_features) + 1:02d}_"
            f"{_artifact_slug(feature)}.png"
        )
        _plot_one_way_summary(
            summary,
            destination,
            f"{model_family.title()} one-way retention - {feature} (inner OOF)",
        )
        paths.append(destination)
        plotted_features.append(feature)
    return year_plot, paths, plotted_features




def _plot_time_trend(monthly: pd.DataFrame, destination: Path, title: str) -> None:
    plt = _pyplot()
    x = pd.to_datetime(monthly[YEAR_MONTH_COLUMN].astype(str) + "-01")
    fig, (axis, volume_axis) = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axis.plot(x, monthly["actual_retention_3m"], label="Actual retention", linewidth=2.4)
    axis.plot(
        x,
        monthly["predicted_retention_3m"],
        label="Predicted retention (calendar-year control)",
        linewidth=2.0,
    )
    axis.set_ylabel("Retention rate (3-month rolling)")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    volume_axis.bar(x, monthly["rows"], width=24, color="#8da0cb", alpha=0.75)
    volume_axis.set_ylabel("Policies")
    volume_axis.set_xlabel("Policy-start year-month")
    volume_axis.grid(axis="y", alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)




def business_retention_monthly_summary(
    *,
    population: pd.DataFrame,
    target_col: str,
    cohort_col: str,
    end_month_exclusive: str | pd.Timestamp | pd.Period,
) -> pd.DataFrame:
    """Aggregate actual retention for complete pre-holdout cohort months.


    The input is the already contract-filtered modeling population. The sealed
    holdout is excluded by cohort month before aggregation. Rolling retention
    is policy weighted: rolling retained policies divided by rolling policies.
    """


    required = {target_col, cohort_col}
    missing = sorted(required.difference(population.columns))
    if missing:
        raise KeyError(f"Business-trend population is missing: {missing}")
    if isinstance(end_month_exclusive, pd.Period):
        end_period = end_month_exclusive.asfreq("M")
    else:
        end_timestamp = pd.Timestamp(end_month_exclusive)
        if pd.isna(end_timestamp):
            raise ValueError("Business-trend holdout boundary is missing.")
        end_period = end_timestamp.to_period("M")


    cohort_dates = pd.to_datetime(population[cohort_col], errors="raise")
    if cohort_dates.isna().any():
        raise ValueError(f"{cohort_col} contains missing cohort dates.")
    cohort_months = cohort_dates.dt.to_period("M")
    included = cohort_months.lt(end_period)
    if not included.any():
        raise ValueError("No complete pre-holdout cohort months are available.")
    target = pd.to_numeric(population.loc[included, target_col], errors="raise")
    if target.isna().any():
        raise ValueError(f"{target_col} contains missing pre-holdout outcomes.")
    observed_values = set(target.unique().tolist())
    if not observed_values or not observed_values.issubset({0, 1}):
        raise ValueError(f"{target_col} must be binary 0/1 for business trends.")


    working = pd.DataFrame(
        {
            YEAR_MONTH_COLUMN: cohort_months.loc[included],
            "_retained": target.astype(int),
        }
    )
    monthly = (
        working.groupby(YEAR_MONTH_COLUMN, observed=True, sort=True)
        .agg(
            policies=("_retained", "size"),
            retained_policies=("_retained", "sum"),
        )
        .reset_index()
    )
    expected_months = pd.period_range(
        monthly[YEAR_MONTH_COLUMN].min(),
        monthly[YEAR_MONTH_COLUMN].max(),
        freq="M",
    )
    missing_months = expected_months.difference(
        pd.PeriodIndex(monthly[YEAR_MONTH_COLUMN], freq="M")
    )
    if len(missing_months):
        raise ValueError(
            "Business-trend cohort history has missing months: "
            + ", ".join(missing_months.astype(str))
        )


    monthly["actual_retention"] = (
        monthly["retained_policies"] / monthly["policies"]
    )
    monthly["actual_churn"] = 1.0 - monthly["actual_retention"]
    monthly["population_fraction"] = (
        monthly["policies"] / float(monthly["policies"].sum())
    )
    for window in (3, 12):
        rolling_policies = monthly["policies"].rolling(
            window, min_periods=window
        ).sum()
        rolling_retained = monthly["retained_policies"].rolling(
            window, min_periods=window
        ).sum()
        monthly[f"policies_{window}m"] = rolling_policies
        monthly[f"retained_policies_{window}m"] = rolling_retained
        monthly[f"actual_retention_{window}m_weighted"] = (
            rolling_retained / rolling_policies
        )
    monthly[YEAR_MONTH_COLUMN] = monthly[YEAR_MONTH_COLUMN].astype(str)
    return monthly




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
        raise ValueError("Selected history must contain positive integer months.") from exc
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
                "Selected decay max_months must be positive and at least its half-life."
            )
        return maximum, f"{maximum}-month cap; {months}-month half-life"
    raise ValueError(f"Unsupported selected history kind: {kind!r}")




def _plot_business_retention_trend(
    monthly: pd.DataFrame,
    destination: Path,
    title: str,
    *,
    show_three_month: bool,
) -> None:
    plt = _pyplot()
    from matplotlib.ticker import PercentFormatter


    x = pd.to_datetime(monthly[YEAR_MONTH_COLUMN].astype(str) + "-01")
    fig, (axis, volume_axis) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axis.plot(
        x,
        monthly["actual_retention"],
        label="Monthly actual retention",
        color="#90a4ae",
        linewidth=1.0,
        alpha=0.65,
    )
    if show_three_month and monthly[
        "actual_retention_3m_weighted"
    ].notna().any():
        axis.plot(
            x,
            monthly["actual_retention_3m_weighted"],
            label="Actual retention (policy-weighted 3-month rolling)",
            color="#ef6c00",
            linewidth=1.9,
        )
    if monthly["actual_retention_12m_weighted"].notna().any():
        axis.plot(
            x,
            monthly["actual_retention_12m_weighted"],
            label="Actual retention (policy-weighted 12-month rolling)",
            color="#005ea8",
            linewidth=2.5,
        )
    axis.set_ylabel("Actual retention rate")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    volume_axis.bar(x, monthly["policies"], width=24, color="#8da0cb", alpha=0.75)
    volume_axis.set_ylabel("Policies")
    volume_axis.set_xlabel("Month-1 evaluation cohort")
    volume_axis.grid(axis="y", alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)




def write_business_retention_trend_artifacts(
    *,
    monthly: pd.DataFrame,
    output_dir: str | Path,
    model_family: str,
    cohort_col: str,
    selected_history: Mapping[str, Any],
) -> dict[str, Any]:
    """Write actual-only business context without scoring the sealed holdout."""


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
        "retained_policies",
        "actual_retention",
        "actual_retention_3m_weighted",
        "actual_retention_12m_weighted",
    }
    missing = sorted(required.difference(monthly.columns))
    if missing:
        raise KeyError(f"Business monthly summary is missing: {missing}")
    frame = monthly.copy()
    frame[YEAR_MONTH_COLUMN] = pd.PeriodIndex(
        frame[YEAR_MONTH_COLUMN].astype(str), freq="M"
    ).astype(str)
    frame = frame.sort_values(YEAR_MONTH_COLUMN, kind="stable").reset_index(drop=True)
    if frame.empty or frame[YEAR_MONTH_COLUMN].duplicated().any():
        raise ValueError("Business monthly summary must contain unique cohort months.")
    window_months, history_description = _selected_history_window(selected_history)
    recent = frame.tail(min(window_months, len(frame))).copy()


    full_csv = output / f"{family}_business_retention_trend_full_history.csv"
    full_plot = output / f"{family}_business_retention_trend_full_history.png"
    selected_csv = output / f"{family}_business_retention_trend_selected_history.csv"
    selected_plot = output / f"{family}_business_retention_trend_selected_history.png"
    frame.to_csv(full_csv, index=False)
    recent.to_csv(selected_csv, index=False)
    _plot_business_retention_trend(
        frame,
        full_plot,
        f"{family.title()} actual retention - full mature pre-holdout history",
        show_three_month=False,
    )
    _plot_business_retention_trend(
        recent,
        selected_plot,
        (
            f"{family.title()} actual retention - selected history window "
            f"({history_description})"
        ),
        show_three_month=True,
    )
    artifacts = [full_csv, full_plot, selected_csv, selected_plot]
    manifest = {
        "contract": "retention_business_trend_v1",
        "model_family": family,
        "scope": "actual_only_complete_pre_holdout_modeling_population",
        "sealed_holdout_scored": False,
        "predictions_included": False,
        "cohort_column": str(cohort_col),
        "cohort_month_definition": f"calendar month of {cohort_col}",
        "rolling_definition": (
            "rolling retained policies divided by rolling policies; rates are "
            "policy weighted and computed before the selected-history tail is sliced"
        ),
        "full_history_start": str(frame[YEAR_MONTH_COLUMN].iloc[0]),
        "full_history_end": str(frame[YEAR_MONTH_COLUMN].iloc[-1]),
        "full_history_months": int(len(frame)),
        "selected_history": dict(selected_history),
        "selected_history_window_rule": (
            "months for recent history; max_months cap for decay history"
        ),
        "selected_history_window_months_requested": int(window_months),
        "selected_history_window_months_plotted": int(len(recent)),
        "selected_history_start": str(recent[YEAR_MONTH_COLUMN].iloc[0]),
        "selected_history_end": str(recent[YEAR_MONTH_COLUMN].iloc[-1]),
        "artifacts": [path.name for path in artifacts],
    }
    manifest_path = output / f"{family}_business_retention_trend_manifest.json"
    manifest_path.write_text(
        json.dumps(_strict_json(manifest), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return {**manifest, "manifest": manifest_path.name}




def write_calendar_control_artifacts(
    *,
    predictions: pd.DataFrame,
    output_dir: str | Path,
    model_family: str,
    one_way_features: Sequence[str],
    one_way_plot_top_n: int = 10,
    feature_types: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = validate_diagnostic_frame(predictions)
    predictions_path = output / f"{model_family}_calendar_year_oof_predictions.csv"
    frame.to_csv(predictions_path, index=False)
    performance = calendar_control_performance(frame)
    performance_path = output / f"{model_family}_calendar_year_performance.csv"
    performance.to_csv(performance_path, index=False)
    year = _group_summary(frame, YEAR_CONTROL_FEATURE)
    year_path = output / f"{model_family}_calendar_year_one_way_summary.csv"
    year.to_csv(year_path, index=False)
    monthly = _group_summary(frame, YEAR_MONTH_COLUMN)[
        [
            YEAR_MONTH_COLUMN,
            "rows",
            "population_fraction",
            "actual_retention",
            "predicted_retention",
            "actual_churn",
            "predicted_churn",
            "calibration_gap",
        ]
    ].copy()
    monthly["actual_retention_3m"] = monthly["actual_retention"].rolling(
        3, min_periods=1
    ).mean()
    monthly["predicted_retention_3m"] = monthly["predicted_retention"].rolling(
        3, min_periods=1
    ).mean()
    monthly_path = output / f"{model_family}_policy_start_year_month_trend.csv"
    monthly.to_csv(monthly_path, index=False)
    plot_path = output / f"{model_family}_policy_start_year_month_trend.png"
    _plot_time_trend(
        monthly,
        plot_path,
        (
            f"{model_family.title()} inner-OOF retention trend with "
            "calendar-year control"
        ),
    )
    one_way, audit = one_way_summaries(frame, one_way_features, feature_types=feature_types)
    one_way_path = output / f"{model_family}_selected_feature_one_way_summaries.csv"
    audit_path = output / f"{model_family}_one_way_summary_audit.csv"
    one_way.to_csv(one_way_path, index=False)
    audit.to_csv(audit_path, index=False)
    year_plot_path, one_way_plot_paths, one_way_plot_features = (
        _write_one_way_plot_artifacts(
            year=year,
            one_way=one_way,
            ordered_features=one_way_features,
            output=output,
            model_family=model_family,
            top_n=one_way_plot_top_n,
        )
    )
    manifest = {
        "contract": "retention_calendar_year_diagnostics_v3",
        "model_family": model_family,
        "scope": "chronological_out_of_fold_development_only",
        "sealed_holdout_scored": False,
        "year_control": YEAR_CONTROL_FEATURE,
        "year_month_definition": "calendar month of saved policy/anchor term start date",
        "model_variant": "WITH_MANDATORY_CALENDAR_YEAR_CONTROL",
        "time_trend_scope": (
            "actual retention and the production candidate with the mandatory "
            "calendar-year control"
        ),
        "one_way_scope": (
            "calendar year plus selected model features; actual and predicted "
            "retention are from chronological out-of-fold development rows"
        ),
        "one_way_plot_feature_limit": int(max(0, one_way_plot_top_n)),
        "one_way_plot_features": one_way_plot_features,
        "calendar_year_control_plot": year_plot_path.name,
        "metric_hierarchy": {
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
        },
        "artifacts": [
            path.name
            for path in (
                predictions_path,
                performance_path,
                year_path,
                monthly_path,
                plot_path,
                one_way_path,
                audit_path,
                year_plot_path,
                *one_way_plot_paths,
            )
        ],
    }
    manifest_path = output / f"{model_family}_diagnostics_manifest.json"
    manifest_path.write_text(
        json.dumps(_strict_json(manifest), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return {**manifest, "manifest": manifest_path.name}




def aggregate_raw_shap(
    transformed_shap: np.ndarray,
    ranges: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(transformed_shap, dtype=float)
    if values.ndim != 2:
        raise ValueError("SHAP values must be a two-dimensional matrix.")
    output: list[np.ndarray] = []
    features: list[str] = []
    for item in ranges:
        start = int(item["start_column"])
        stop = int(item["end_column_exclusive"])
        if start < 0 or stop <= start or stop > values.shape[1]:
            raise ValueError(f"Invalid transformed-column range: {item}")
        features.append(str(item["raw_feature"]))
        output.append(values[:, start:stop].sum(axis=1))
    if not output:
        raise ValueError("No raw feature ranges were provided for SHAP aggregation.")
    return np.column_stack(output), features




def write_xgboost_shap_artifacts(
    *,
    raw_shap: np.ndarray,
    feature_values: pd.DataFrame,
    features: Sequence[str],
    output_dir: str | Path,
    top_n: int = 15,
    feature_types: Mapping[str, str] | None = None,
) -> list[str]:
    """Write raw-feature SHAP tables and plots with the official SHAP package."""


    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    shap_values = np.asarray(raw_shap, dtype=float)
    ordered_features = [str(value) for value in features]
    if shap_values.shape != (len(feature_values), len(ordered_features)):
        raise ValueError(
            "Raw SHAP matrix shape does not match its feature-value frame."
        )
    importance = pd.DataFrame(
        {
            "feature": ordered_features,
            "mean_absolute_shap_log_odds": np.mean(np.abs(shap_values), axis=0),
            "mean_shap_log_odds": np.mean(shap_values, axis=0),
        }
    ).sort_values("mean_absolute_shap_log_odds", ascending=False, kind="stable")
    importance_path = output / "xgboost_shap_importance.csv"
    importance.to_csv(importance_path, index=False)
    top = importance.head(max(1, min(int(top_n), len(importance))))[
        "feature"
    ].tolist()
    positions = [ordered_features.index(feature) for feature in top]
    sample_long = pd.concat(
        [
            pd.DataFrame(
                {
                    "feature": feature,
                    "shap_log_odds_churn": shap_values[:, position],
                    "feature_value": feature_values[feature].astype(str).to_numpy(),
                }
            )
            for feature, position in zip(top, positions)
        ],
        ignore_index=True,
    )
    sample_path = output / "xgboost_shap_values_top_features.csv"
    sample_long.to_csv(sample_path, index=False)


    shap = _shap_package()
    plt = _pyplot()
    plot_values = pd.DataFrame(index=feature_values.index)
    for feature in ordered_features:
        raw = feature_values[feature]
        numeric = pd.to_numeric(raw, errors="coerce")
        if _is_continuous_plot_feature(raw, (feature_types or {}).get(feature)):
            finite = numeric.replace([np.inf, -np.inf], np.nan)
            fill = finite.median()
            plot_values[feature] = finite.fillna(0.0 if pd.isna(fill) else fill)
        else:
            codes, _ = pd.factorize(raw.astype("string").fillna("__MISSING__"))
            plot_values[feature] = codes.astype(float)
    explanation = shap.Explanation(
        values=shap_values,
        data=plot_values[ordered_features].to_numpy(float),
        display_data=feature_values[ordered_features]
        .astype("string")
        .fillna("__MISSING__")
        .to_numpy(),
        feature_names=ordered_features,
    )
    shap.plots.beeswarm(
        explanation,
        max_display=len(top),
        show=False,
        plot_size=(11, max(6, 0.48 * len(top) + 2)),
    )
    axis = plt.gca()
    axis.set_xlabel("SHAP value (positive increases predicted churn risk)")
    axis.set_title("XGBoost SHAP beeswarm - chronological validation sample")
    beeswarm_figure = plt.gcf()
    beeswarm_figure.tight_layout()
    beeswarm_path = output / "xgboost_shap_beeswarm.png"
    beeswarm_figure.savefig(beeswarm_path, dpi=180, bbox_inches="tight")
    plt.close(beeswarm_figure)


    scatter_features = [
        feature
        for feature in top
        if feature != YEAR_CONTROL_FEATURE
        if _is_continuous_plot_feature(
            feature_values[feature], (feature_types or {}).get(feature)
        )
    ][:6]
    scatter_paths: list[Path] = []
    scatter_audit: list[dict[str, Any]] = []
    plotted_features: list[str] = []
    for feature in scatter_features:
        rank = top.index(feature) + 1
        position = ordered_features.index(feature)
        numeric = _numeric_plot_values(feature_values[feature])
        values = shap_values[:, position]
        finite_x = np.isfinite(numeric)
        finite_shap = np.isfinite(values)
        negative = finite_x & (numeric < 0) if feature.upper() == "WRITTEN_PREM" else np.zeros(len(numeric), dtype=bool)
        visible = finite_x & finite_shap & ~negative
        detail = {
            "feature": feature,
            "input_rows": int(len(numeric)),
            "plot_rows": int(visible.sum()),
            "negative_rows_hidden_from_plot": int(negative.sum()),
            "nonfinite_feature_rows_hidden_from_plot": int((~finite_x).sum()),
            "nonfinite_shap_rows_hidden_from_plot": int((~finite_shap).sum()),
            "total_rows_hidden_from_plot": int((~visible).sum()),
            "x_axis_lower_bound": 0.0 if feature.upper() == "WRITTEN_PREM" else None,
            "positive_values_capped": False,
            "raw_values_modified": False,
            "status": "WRITTEN" if visible.any() else "NO_FINITE_VISIBLE_ROWS",
        }
        scatter_audit.append(detail)
        if not visible.any():
            continue
        # SHAP uses display_data for ticks. Passing strings here makes every
        # distinct continuous value a category, even when data itself is numeric.
        numeric_explanation = shap.Explanation(
            values=values[visible, np.newaxis],
            data=numeric[visible, np.newaxis],
            display_data=numeric[visible, np.newaxis],
            feature_names=[feature],
        )
        fig, axis = plt.subplots(figsize=(8.5, 5.5))
        shap.plots.scatter(
            numeric_explanation[:, feature],
            color="#1f77b4",
            hist=True,
            alpha=0.45,
            ax=axis,
            show=False,
        )
        axis.set_ylabel("SHAP log-odds contribution to churn")
        axis.set_title(f"XGBoost SHAP dependence - {feature}")
        axis.set_xlabel(feature)
        _format_numeric_axis(axis, feature=feature, values=numeric[visible])
        axis.grid(alpha=0.2)
        if (~visible).any():
            fig.text(0.02, 0.01,
                     f"Display only: {(~visible).sum():,} rows excluded"
                     + (f"; {negative.sum():,} negative premium values" if negative.any() else "")
                     + ". Raw values are unchanged.", fontsize=8)
            fig.subplots_adjust(bottom=0.16)
        fig.tight_layout(rect=(0, 0.055, 1, 1) if (~visible).any() else None)
        destination = output / (
            f"xgboost_shap_scatter_{rank:02d}_{_artifact_slug(feature)}.png"
        )
        fig.savefig(destination, dpi=180, bbox_inches="tight")
        plt.close(fig)
        scatter_paths.append(destination)
        plotted_features.append(feature)


    plot_manifest_path = output / "xgboost_shap_plot_manifest.json"
    plot_manifest = {
        "contract": "xgboost_shap_plots_v3_numeric_axes",
        "top_features": top,
        "scatter_features": plotted_features,
        "scatter_display_audit": scatter_audit,
        "numeric_axis_max_intervals": 7,
        "feature_types": dict(feature_types or {}),
        "scatter_control_feature_excluded": YEAR_CONTROL_FEATURE,
        "scatter_control_exclusion_reason": (
            "calendar year is a mandatory control and is kept in governance "
            "diagnostics rather than explanatory dependence plots"
        ),
        "artifacts": [
            importance_path.name,
            sample_path.name,
            beeswarm_path.name,
            *[path.name for path in scatter_paths],
            plot_manifest_path.name,
        ],
    }
    plot_manifest_path.write_text(
        json.dumps(_strict_json(plot_manifest), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return top




__all__ = [
    "ACTUAL_COLUMN",
    "WITH_YEAR_COLUMN",
    "YEAR_CONTROL_FEATURE",
    "YEAR_MONTH_COLUMN",
    "aggregate_raw_shap",
    "business_retention_monthly_summary",
    "calendar_control_performance",
    "one_way_summaries",
    "prediction_metrics",
    "validate_diagnostic_frame",
    "write_business_retention_trend_artifacts",
    "write_calendar_control_artifacts",
    "write_xgboost_shap_artifacts",
]
