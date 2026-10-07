"""Presentation-ready performance plots for retention evaluation sets.


The public entry point consumes row-level outer-OOF or final-test predictions.
All score-ranking charts rank policies within their chronological fold before
pooling results, which avoids treating scores from different refits as though
they came from one fitted model.
"""


from __future__ import annotations


import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)




REQUIRED_PREDICTION_COLUMNS = (
    "model_family",
    "y_retention",
    "p_retention",
    "fold",
    "cohort_date",
)
EVALUATION_SCOPE_LABELS = {
    "outer_oof": "Outer out-of-fold validation",
    "final_test": "Final sealed test",
}
_PALETTE = (
    "#1769AA",
    "#E66C37",
    "#2E8B57",
    "#7B61A8",
    "#D4A017",
    "#3A9199",
)




def _pyplot() -> Any:
    try:
        import matplotlib


        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise ImportError(
            "Performance plots require matplotlib. Install matplotlib>=3.8."
        ) from exc
    return plt




def _percent_formatter() -> Any:
    try:
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise ImportError(
            "Performance plots require matplotlib. Install matplotlib>=3.8."
        ) from exc
    return PercentFormatter




def _shap_package() -> Any:
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The SHAP beeswarm plot requires shap. Install shap>=0.46."
        ) from exc
    return shap




def _scope_label(evaluation_scope: str) -> str:
    normalized = str(evaluation_scope).strip().lower()
    if normalized not in EVALUATION_SCOPE_LABELS:
        raise ValueError(
            "evaluation_scope must be 'outer_oof' or 'final_test'; "
            f"received {evaluation_scope!r}."
        )
    return EVALUATION_SCOPE_LABELS[normalized]




def _display_model(model_family: str) -> str:
    normalized = str(model_family).strip().lower()
    known = {"logistic": "Logistic", "xgboost": "XGBoost"}
    return known.get(normalized, normalized.replace("_", " ").title())




def _model_colors(model_families: Sequence[str]) -> dict[str, str]:
    ordered = sorted({str(value) for value in model_families})
    return {
        family: _PALETTE[position % len(_PALETTE)]
        for position, family in enumerate(ordered)
    }




def _artifact(kind: str, destination: Path, **metadata: Any) -> dict[str, Any]:
    return {
        "kind": str(kind),
        "filename": destination.name,
        "path": str(destination.resolve()),
        "media_type": "image/png",
        **metadata,
    }




def _save_figure_atomic(fig: Any, destination: str | Path, *, dpi: int = 180) -> Path:
    """Write one PNG completely before replacing the declared destination."""


    plt = _pyplot()
    output = Path(destination).expanduser().resolve()
    if output.suffix.lower() != ".png":
        raise ValueError(f"Plot destination must end in .png: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        fig.savefig(
            temporary,
            format="png",
            dpi=int(dpi),
            bbox_inches="tight",
            facecolor="white",
            metadata={"Software": "Home Retention 13/1 performance plots"},
        )
        os.replace(temporary, output)
    finally:
        plt.close(fig)
        if temporary.exists():
            temporary.unlink()
    return output




def validate_prediction_frame(
    predictions: pd.DataFrame,
    *,
    evaluation_scope: str,
    require_aligned_models: bool = True,
) -> pd.DataFrame:
    """Validate and normalize one row-level evaluation prediction table."""


    _scope_label(evaluation_scope)
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame.")
    missing = sorted(set(REQUIRED_PREDICTION_COLUMNS).difference(predictions.columns))
    if missing:
        raise KeyError(f"Prediction frame is missing required columns: {missing}")
    if predictions.empty:
        raise ValueError("Prediction frame cannot be empty.")


    result = predictions.copy()
    result["model_family"] = result["model_family"].astype("string").str.strip()
    if result["model_family"].isna().any() or result["model_family"].eq("").any():
        raise ValueError("model_family must be nonmissing and nonempty.")


    target = pd.to_numeric(result["y_retention"], errors="raise")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("y_retention must contain only binary 0/1 values.")
    result["y_retention"] = target.astype(np.int8)


    probability = pd.to_numeric(result["p_retention"], errors="raise").astype(float)
    if not np.isfinite(probability).all() or not probability.between(0.0, 1.0).all():
        raise ValueError("p_retention must contain finite probabilities in [0, 1].")
    result["p_retention"] = probability


    if result["fold"].isna().any() or result["fold"].astype(str).str.strip().eq("").any():
        raise ValueError("fold must be nonmissing and nonempty.")
    dates = pd.to_datetime(result["cohort_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("cohort_date must contain valid, nonmissing dates.")
    result["cohort_date"] = dates
    result["_source_order"] = np.arange(len(result), dtype=np.int64)


    if "row_id" in result:
        result["row_id"] = result["row_id"].astype("string").str.strip()
        if result["row_id"].isna().any() or result["row_id"].eq("").any():
            raise ValueError("row_id must be nonmissing and nonempty when supplied.")
        duplicate_identity = result.duplicated(
            ["model_family", "row_id"], keep=False
        )
        if duplicate_identity.any():
            raise ValueError(
                "Each model family must score every outer-OOF row exactly once; "
                f"duplicate_rows={int(duplicate_identity.sum())}."
            )


    model_families = sorted(result["model_family"].astype(str).unique())
    for family, part in result.groupby("model_family", sort=True, observed=True):
        if set(part["y_retention"].unique()) != {0, 1}:
            raise ValueError(
                f"Model family {family!r} must contain both target classes."
            )
        for fold, fold_part in part.groupby("fold", sort=False, observed=True):
            if set(fold_part["y_retention"].unique()) != {0, 1}:
                raise ValueError(
                    f"Model family {family!r}, fold {fold!r} must contain both "
                    "target classes."
                )
        if (
            str(evaluation_scope).strip().lower() == "final_test"
            and part["fold"].nunique() != 1
        ):
            raise ValueError(
                "Each final-test model family must contain exactly one evaluation "
                f"partition; {family!r} has {part['fold'].nunique()}."
            )


    if require_aligned_models and len(model_families) > 1:
        comparison_columns = ["fold", "cohort_date", "y_retention"]
        if "row_id" in result:
            comparison_columns.insert(0, "row_id")
        reference: pd.DataFrame | None = None
        reference_family = ""
        for family in model_families:
            counts = (
                result.loc[result["model_family"].eq(family), comparison_columns]
                .value_counts(sort=False)
                .rename("rows")
                .sort_index()
                .reset_index()
            )
            if reference is None:
                reference = counts
                reference_family = family
            elif not counts.equals(reference):
                raise ValueError(
                    "Model families must score aligned fold/cohort/target "
                    f"populations; {family!r} differs from {reference_family!r}."
                )


    return result.sort_values(
        ["model_family", "fold", "cohort_date", "_source_order"],
        kind="stable",
    ).reset_index(drop=True)




def _risk_rank_summary(predictions: pd.DataFrame, bins: int) -> pd.DataFrame:
    """Aggregate actual churn by fold-local predicted-risk rank."""


    if bins < 2 or bins > 100:
        raise ValueError("bins must be between 2 and 100.")
    ranked_parts: list[pd.DataFrame] = []
    group_columns = ["model_family", "fold"]
    for _, part in predictions.groupby(group_columns, sort=True, observed=True):
        ranked = part.assign(
            y_churn=1 - part["y_retention"],
            p_churn=1.0 - part["p_retention"],
        )
        tie_breakers = (
            ["row_id"]
            if "row_id" in ranked
            else ["cohort_date", "_source_order"]
        )
        ranked = ranked.sort_values(
            ["p_churn", *tie_breakers],
            ascending=[False, *([True] * len(tie_breakers))],
            kind="mergesort",
        )
        rank = np.arange(len(ranked), dtype=np.int64)
        ranked["risk_bin"] = np.minimum((rank * bins // len(ranked)) + 1, bins)
        ranked_parts.append(ranked)
    ranked_all = pd.concat(ranked_parts, ignore_index=True)
    summary = (
        ranked_all.groupby(["model_family", "risk_bin"], sort=True, observed=True)
        .agg(
            rows=("y_churn", "size"),
            churn_events=("y_churn", "sum"),
            churn_rate=("y_churn", "mean"),
            mean_predicted_churn=("p_churn", "mean"),
        )
        .reset_index()
    )
    summary["population_rows_cumulative"] = summary.groupby(
        "model_family", observed=True
    )["rows"].cumsum()
    summary["churn_events_cumulative"] = summary.groupby(
        "model_family", observed=True
    )["churn_events"].cumsum()
    totals = (
        ranked_all.groupby("model_family", observed=True)["y_churn"]
        .agg(total_rows="size", total_churn="sum", prevalence="mean")
        .reset_index()
    )
    summary = summary.merge(totals, on="model_family", how="left", validate="many_to_one")
    summary["lift"] = summary["churn_rate"] / summary["prevalence"]
    summary["population_fraction_cumulative"] = (
        summary["population_rows_cumulative"] / summary["total_rows"]
    )
    summary["churn_capture_cumulative"] = (
        summary["churn_events_cumulative"] / summary["total_churn"]
    )
    return summary




def _fold_discrimination_curves(
    predictions: pd.DataFrame,
    *,
    grid_size: int = 201,
) -> dict[str, Any]:
    """Return fold metrics and curves interpolated onto common display grids."""


    fpr_grid = np.linspace(0.0, 1.0, int(grid_size))
    recall_grid = np.linspace(0.0, 1.0, int(grid_size))
    fold_rows: list[dict[str, Any]] = []
    interpolated_tpr: list[np.ndarray] = []
    interpolated_precision: list[np.ndarray] = []
    raw_curves: list[dict[str, np.ndarray]] = []


    for fold, part in predictions.groupby("fold", sort=True, observed=True):
        y_churn = 1 - part["y_retention"].to_numpy(dtype=int)
        p_churn = 1.0 - part["p_retention"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y_churn, p_churn)
        precision, recall, _ = precision_recall_curve(y_churn, p_churn)


        fold_tpr = np.interp(fpr_grid, fpr, tpr)
        fold_tpr[0] = 0.0
        fold_tpr[-1] = 1.0


        # precision_recall_curve returns decreasing recall and may repeat recall
        # values.  Use the best precision at each recall before interpolation.
        pr_points = pd.DataFrame(
            {"recall": recall, "precision": precision}
        ).groupby("recall", sort=True, as_index=False)["precision"].max()
        fold_precision = np.interp(
            recall_grid,
            pr_points["recall"].to_numpy(dtype=float),
            pr_points["precision"].to_numpy(dtype=float),
        )
        interpolated_tpr.append(fold_tpr)
        interpolated_precision.append(fold_precision)
        raw_curves.append(
            {
                "fpr": fpr,
                "tpr": tpr,
                "recall": recall,
                "precision": precision,
            }
        )
        fold_rows.append(
            {
                "fold": str(fold),
                "rows": int(len(part)),
                "auc": float(roc_auc_score(y_churn, p_churn)),
                "average_precision": float(
                    average_precision_score(y_churn, p_churn)
                ),
                "churn_prevalence": float(y_churn.mean()),
            }
        )


    metrics = pd.DataFrame(fold_rows)
    tpr_matrix = np.vstack(interpolated_tpr)
    precision_matrix = np.vstack(interpolated_precision)
    ddof = 1 if len(metrics) > 1 else 0
    return {
        "fpr_grid": fpr_grid,
        "recall_grid": recall_grid,
        "mean_tpr": tpr_matrix.mean(axis=0),
        "sd_tpr": tpr_matrix.std(axis=0, ddof=ddof),
        "mean_precision": precision_matrix.mean(axis=0),
        "sd_precision": precision_matrix.std(axis=0, ddof=ddof),
        "raw_curves": raw_curves,
        "fold_metrics": metrics,
        "summary": {
            "fold_count": int(len(metrics)),
            "mean_fold_auc": float(metrics["auc"].mean()),
            "sd_fold_auc": float(metrics["auc"].std(ddof=ddof)),
            "mean_fold_average_precision": float(
                metrics["average_precision"].mean()
            ),
            "sd_fold_average_precision": float(
                metrics["average_precision"].std(ddof=ddof)
            ),
            "mean_fold_churn_prevalence": float(
                metrics["churn_prevalence"].mean()
            ),
            "sd_fold_churn_prevalence": float(
                metrics["churn_prevalence"].std(ddof=ddof)
            ),
        },
    }




def plot_roc_pr(
    predictions: pd.DataFrame,
    destination: str | Path,
    *,
    evaluation_scope: str,
) -> dict[str, Any]:
    """Plot fold-macro outer curves or direct single-set final-test curves."""


    normalized_scope = str(evaluation_scope).strip().lower()
    scope = _scope_label(normalized_scope)
    plt = _pyplot()
    PercentFormatter = _percent_formatter()
    models = sorted(predictions["model_family"].astype(str).unique())
    colors = _model_colors(models)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    metric_summary: dict[str, dict[str, Any]] = {}


    for family in models:
        part = predictions.loc[predictions["model_family"].eq(family)]
        curves = _fold_discrimination_curves(part)
        summary = curves["summary"]
        metric_summary[family] = summary
        label = _display_model(family)
        if normalized_scope == "outer_oof":
            auc_label = (
                f"{label} (mean fold AUC "
                f"{summary['mean_fold_auc']:.3f} +/- {summary['sd_fold_auc']:.3f})"
            )
            ap_label = (
                f"{label} (mean fold AP "
                f"{summary['mean_fold_average_precision']:.3f} +/- "
                f"{summary['sd_fold_average_precision']:.3f})"
            )
            axes[0].plot(
                curves["fpr_grid"],
                curves["mean_tpr"],
                color=colors[family],
                linewidth=2.2,
                label=auc_label,
            )
            axes[0].fill_between(
                curves["fpr_grid"],
                np.clip(curves["mean_tpr"] - curves["sd_tpr"], 0.0, 1.0),
                np.clip(curves["mean_tpr"] + curves["sd_tpr"], 0.0, 1.0),
                color=colors[family],
                alpha=0.12,
                linewidth=0,
            )
            axes[1].plot(
                curves["recall_grid"],
                curves["mean_precision"],
                color=colors[family],
                linewidth=2.2,
                label=ap_label,
            )
            axes[1].fill_between(
                curves["recall_grid"],
                np.clip(
                    curves["mean_precision"] - curves["sd_precision"],
                    0.0,
                    1.0,
                ),
                np.clip(
                    curves["mean_precision"] + curves["sd_precision"],
                    0.0,
                    1.0,
                ),
                color=colors[family],
                alpha=0.12,
                linewidth=0,
            )
        else:
            raw = curves["raw_curves"][0]
            axes[0].plot(
                raw["fpr"],
                raw["tpr"],
                color=colors[family],
                linewidth=2.2,
                label=f"{label} (AUC {summary['mean_fold_auc']:.3f})",
            )
            axes[1].plot(
                raw["recall"],
                raw["precision"],
                color=colors[family],
                linewidth=2.2,
                label=(
                    f"{label} (AP "
                    f"{summary['mean_fold_average_precision']:.3f})"
                ),
            )


    first_summary = metric_summary[models[0]]
    prevalence = first_summary["mean_fold_churn_prevalence"]
    prevalence_sd = first_summary["sd_fold_churn_prevalence"]
    axes[0].plot(
        [0, 1],
        [0, 1],
        color="#777777",
        linestyle="--",
        linewidth=1.3,
        label="Random",
    )
    baseline_label = (
        f"Mean fold churn baseline ({prevalence:.1%})"
        if normalized_scope == "outer_oof"
        else f"Churn baseline ({prevalence:.1%})"
    )
    axes[1].axhline(
        prevalence,
        color="#777777",
        linestyle="--",
        linewidth=1.3,
        label=baseline_label,
    )
    if normalized_scope == "outer_oof" and prevalence_sd > 0.0:
        axes[1].axhspan(
            max(0.0, prevalence - prevalence_sd),
            min(1.0, prevalence + prevalence_sd),
            color="#777777",
            alpha=0.06,
            linewidth=0,
        )
    curve_aggregation = (
        "macro_average_equal_fold_weight"
        if normalized_scope == "outer_oof"
        else "single_final_test_set"
    )
    axes[0].set(
        title=(
            "ROC - macro-average across folds"
            if normalized_scope == "outer_oof"
            else "ROC curve"
        ),
        xlabel="False-positive rate",
        ylabel="True-positive rate",
    )
    axes[1].set(
        title=(
            "Precision-recall - macro-average across folds"
            if normalized_scope == "outer_oof"
            else "Precision-recall curve"
        ),
        xlabel="Recall / churn capture",
        ylabel="Precision / churn rate",
    )
    for axis in axes:
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.xaxis.set_major_formatter(PercentFormatter(1.0))
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(alpha=0.22)
        axis.legend(loc="best", frameon=False)
    fig.suptitle(f"Model performance - {scope}", fontsize=15, fontweight="bold")
    fig.tight_layout()
    output = _save_figure_atomic(fig, destination)
    return _artifact(
        "roc_pr",
        output,
        evaluation_scope=normalized_scope,
        model_families=models,
        curve_aggregation=curve_aggregation,
        variability_band=(
            "plus_minus_one_fold_standard_deviation"
            if normalized_scope == "outer_oof"
            else "not_applicable"
        ),
        fold_metric_summary=metric_summary,
    )




def plot_percentile_lift_gains(
    predictions: pd.DataFrame,
    destination: str | Path,
    *,
    evaluation_scope: str,
    percentile_bins: int = 20,
) -> dict[str, Any]:
    """Plot fold-local risk-percentile lift and cumulative churn gains."""


    scope = _scope_label(evaluation_scope)
    summary = _risk_rank_summary(predictions, percentile_bins)
    plt = _pyplot()
    PercentFormatter = _percent_formatter()
    models = sorted(summary["model_family"].astype(str).unique())
    colors = _model_colors(models)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    lift_axis = axes[0].twinx()


    for family in models:
        part = summary.loc[summary["model_family"].eq(family)].sort_values("risk_bin")
        x = part["population_fraction_cumulative"].to_numpy(float)
        label = _display_model(family)
        axes[0].plot(x, part["churn_rate"], color=colors[family], marker="o", markersize=3.5, linewidth=2.0, label=f"{label} churn rate")
        lift_axis.plot(x, part["lift"], color=colors[family], linestyle="--", linewidth=1.6, alpha=0.82, label=f"{label} lift")
        axes[1].plot(x, part["churn_capture_cumulative"], color=colors[family], linewidth=2.3, label=label)


    axes[0].set(
        xlabel="Risk-score percentile band (highest risk first)",
        ylabel="Observed churn rate",
        title="Churn rate and lift by score percentile",
    )
    lift_axis.set_ylabel("Lift multiplier")
    axes[0].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].grid(alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    lift_handles, lift_labels = lift_axis.get_legend_handles_labels()
    axes[0].legend(handles + lift_handles, labels + lift_labels, loc="best", frameon=False, fontsize=8.5)


    axes[1].plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.3, label="Random targeting")
    axes[1].set(xlabel="Risk-ranked population targeted", ylabel="Cumulative churn captured", title="Cumulative gains")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].grid(alpha=0.22)
    axes[1].legend(loc="lower right", frameon=False)
    fig.suptitle(f"Lift and gains - {scope}", fontsize=15, fontweight="bold")
    fig.tight_layout()
    output = _save_figure_atomic(fig, destination)
    return _artifact(
        "percentile_lift_gains",
        output,
        evaluation_scope=evaluation_scope,
        model_families=models,
        ranking_scope="within_fold",
        percentile_bins=int(percentile_bins),
    )




def plot_decile_lift(
    predictions: pd.DataFrame,
    destination: str | Path,
    *,
    evaluation_scope: str,
) -> dict[str, Any]:
    """Plot observed churn and lift from highest- to lowest-risk decile."""


    scope = _scope_label(evaluation_scope)
    summary = _risk_rank_summary(predictions, 10)
    plt = _pyplot()
    PercentFormatter = _percent_formatter()
    models = sorted(summary["model_family"].astype(str).unique())
    colors = _model_colors(models)
    fig, raw_axes = plt.subplots(1, len(models), figsize=(7.2 * len(models), 5.8), squeeze=False)
    axes = raw_axes.ravel()


    for axis, family in zip(axes, models):
        part = summary.loc[summary["model_family"].eq(family)].sort_values("risk_bin")
        deciles = part["risk_bin"].to_numpy(int)
        axis.bar(deciles, part["churn_rate"], color=colors[family], alpha=0.82, label="Observed churn rate")
        prevalence = float(part["prevalence"].iloc[0])
        axis.axhline(prevalence, color="#555555", linestyle="--", linewidth=1.3, label=f"Average churn ({prevalence:.1%})")
        axis.set(xlabel="Risk decile (1 = highest risk)", ylabel="Observed churn rate", title=_display_model(family), xticks=deciles)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(axis="y", alpha=0.22)
        lift_axis = axis.twinx()
        lift_axis.plot(deciles, part["lift"], color="#202020", marker="o", linewidth=1.8, label="Lift")
        lift_axis.set_ylabel("Lift multiplier")
        handles, labels = axis.get_legend_handles_labels()
        lift_handles, lift_labels = lift_axis.get_legend_handles_labels()
        axis.legend(handles + lift_handles, labels + lift_labels, loc="best", frameon=False, fontsize=8.5)


    fig.suptitle(f"Lift by score decile - {scope}", fontsize=15, fontweight="bold")
    fig.tight_layout()
    output = _save_figure_atomic(fig, destination)
    return _artifact(
        "decile_lift",
        output,
        evaluation_scope=evaluation_scope,
        model_families=models,
        ranking_scope="within_fold",
    )




def _calibration_summary(predictions: pd.DataFrame, bins: int) -> pd.DataFrame:
    ranked = _risk_rank_summary(predictions, bins)
    return ranked.loc[
        :, [
            "model_family",
            "risk_bin",
            "rows",
            "mean_predicted_churn",
            "churn_rate",
        ]
    ].copy()




def plot_calibration(
    predictions: pd.DataFrame,
    destination: str | Path,
    *,
    evaluation_scope: str,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Plot predicted versus observed churn across fold-local risk bins."""


    scope = _scope_label(evaluation_scope)
    summary = _calibration_summary(predictions, calibration_bins)
    plt = _pyplot()
    PercentFormatter = _percent_formatter()
    models = sorted(summary["model_family"].astype(str).unique())
    colors = _model_colors(models)
    fig, axis = plt.subplots(figsize=(7.6, 6.5))
    maximum = max(float(summary["mean_predicted_churn"].max()), float(summary["churn_rate"].max()))
    upper = min(1.0, max(0.10, math.ceil(maximum * 10.0) / 10.0))
    axis.plot([0.0, upper], [0.0, upper], color="#666666", linestyle="--", linewidth=1.4, label="Perfect calibration")
    for family in models:
        part = summary.loc[summary["model_family"].eq(family)].sort_values("mean_predicted_churn")
        axis.plot(part["mean_predicted_churn"], part["churn_rate"], color=colors[family], marker="o", linewidth=2.0, label=_display_model(family))
    axis.set(xlabel="Mean predicted churn", ylabel="Observed churn", title=f"Calibration - {scope}", xlim=(0.0, upper), ylim=(0.0, upper))
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(alpha=0.22)
    axis.legend(loc="best", frameon=False)
    fig.tight_layout()
    output = _save_figure_atomic(fig, destination)
    return _artifact(
        "calibration",
        output,
        evaluation_scope=evaluation_scope,
        model_families=models,
        calibration_bins=int(calibration_bins),
        ranking_scope="within_fold",
    )




def plot_temporal_stability(
    predictions: pd.DataFrame,
    destination: str | Path,
    *,
    evaluation_scope: str,
) -> dict[str, Any]:
    """Plot monthly actual/predicted retention and evaluation volume."""


    scope = _scope_label(evaluation_scope)
    work = predictions.assign(cohort_month=predictions["cohort_date"].dt.to_period("M").dt.to_timestamp())
    monthly = (
        work.groupby(["model_family", "cohort_month"], sort=True, observed=True)
        .agg(rows=("y_retention", "size"), actual_retention=("y_retention", "mean"), predicted_retention=("p_retention", "mean"))
        .reset_index()
    )
    models = sorted(monthly["model_family"].astype(str).unique())
    colors = _model_colors(models)
    plt = _pyplot()
    PercentFormatter = _percent_formatter()
    fig, (trend_axis, volume_axis) = plt.subplots(
        2,
        1,
        figsize=(13.5, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]},
    )


    reference = monthly.loc[monthly["model_family"].eq(models[0])]
    trend_axis.plot(reference["cohort_month"], reference["actual_retention"], color="#202020", marker="o", linewidth=2.3, label="Actual retention")
    for family in models:
        part = monthly.loc[monthly["model_family"].eq(family)]
        trend_axis.plot(part["cohort_month"], part["predicted_retention"], color=colors[family], marker="o", linewidth=1.9, label=f"{_display_model(family)} predicted")
    trend_axis.set(ylabel="Retention rate", title=f"Temporal performance stability - {scope}")
    trend_axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    trend_axis.grid(alpha=0.22)
    trend_axis.legend(loc="best", frameon=False)
    volume_axis.bar(reference["cohort_month"], reference["rows"], width=22, color="#AFC6DC")
    volume_axis.set(xlabel="Month-1 cohort month", ylabel="Rows")
    volume_axis.grid(axis="y", alpha=0.18)
    fig.autofmt_xdate(rotation=35, ha="right")
    fig.tight_layout()
    output = _save_figure_atomic(fig, destination)
    return _artifact("temporal_stability", output, evaluation_scope=evaluation_scope, model_families=models)




def plot_shap_beeswarm(
    shap_values: Any,
    feature_values: pd.DataFrame,
    destination: str | Path,
    *,
    evaluation_scope: str,
    model_family: str = "xgboost",
    max_display: int = 15,
    output_label: str = "predicted churn risk",
) -> dict[str, Any]:
    """Plot one SHAP beeswarm for supplied row-by-feature explanations."""


    scope = _scope_label(evaluation_scope)
    if not isinstance(feature_values, pd.DataFrame) or feature_values.empty:
        raise ValueError("feature_values must be a nonempty pandas DataFrame.")
    raw_values = getattr(shap_values, "values", shap_values)
    values = np.asarray(raw_values, dtype=float)
    if values.ndim != 2:
        raise ValueError(
            "shap_values must be a two-dimensional row-by-feature array; "
            "select the churn output before calling this helper."
        )
    if values.shape != feature_values.shape:
        raise ValueError(
            "shap_values and feature_values must have the same row/feature shape; "
            f"received {values.shape} and {feature_values.shape}."
        )
    if not np.isfinite(values).all():
        raise ValueError("shap_values must be finite.")
    if max_display < 1:
        raise ValueError("max_display must be positive.")


    encoded = pd.DataFrame(index=feature_values.index)
    display = feature_values.astype("string").fillna("__MISSING__")
    for feature in feature_values.columns:
        numeric = pd.to_numeric(feature_values[feature], errors="coerce")
        if numeric.notna().mean() >= 0.80:
            finite = numeric.replace([np.inf, -np.inf], np.nan)
            median = finite.median()
            encoded[feature] = finite.fillna(0.0 if pd.isna(median) else median)
        else:
            codes, _ = pd.factorize(display[feature], sort=True)
            encoded[feature] = codes.astype(float)


    shap = _shap_package()
    plt = _pyplot()
    explanation = shap.Explanation(
        values=values,
        data=encoded.to_numpy(dtype=float),
        display_data=display.to_numpy(),
        feature_names=[str(value) for value in feature_values.columns],
    )
    shap.plots.beeswarm(
        explanation,
        max_display=min(int(max_display), feature_values.shape[1]),
        show=False,
        plot_size=(11, max(6.0, 0.46 * min(max_display, feature_values.shape[1]) + 2.2)),
    )
    axis = plt.gca()
    axis.set_xlabel(f"SHAP value (positive increases {output_label})")
    axis.set_title(f"{_display_model(model_family)} SHAP - {scope}")
    figure = plt.gcf()
    figure.tight_layout()
    output = _save_figure_atomic(figure, destination)
    return _artifact(
        "shap_beeswarm",
        output,
        evaluation_scope=evaluation_scope,
        model_families=[str(model_family)],
        explained_output=str(output_label),
        rows=int(values.shape[0]),
        features=int(values.shape[1]),
        max_display=min(int(max_display), int(values.shape[1])),
    )




def generate_performance_plots(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    *,
    evaluation_scope: str,
    filename_prefix: str | None = None,
    percentile_bins: int = 20,
    calibration_bins: int = 10,
    shap_values: Any | None = None,
    shap_feature_values: pd.DataFrame | None = None,
    shap_model_family: str = "xgboost",
    shap_max_display: int = 15,
    shap_output_label: str = "predicted churn risk",
) -> dict[str, Any]:
    """Generate the governed evaluation plot set and return a manifest payload."""


    normalized_scope = str(evaluation_scope).strip().lower()
    scope_label = _scope_label(normalized_scope)
    validated = validate_prediction_frame(
        predictions,
        evaluation_scope=normalized_scope,
        require_aligned_models=True,
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = str(filename_prefix or normalized_scope).strip()
    if not prefix or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in prefix):
        raise ValueError(
            "filename_prefix must contain only letters, numbers, '_' or '-'."
        )


    artifacts = [
        plot_roc_pr(validated, output / f"{prefix}_roc_pr.png", evaluation_scope=normalized_scope),
        plot_percentile_lift_gains(
            validated,
            output / f"{prefix}_percentile_lift_gains.png",
            evaluation_scope=normalized_scope,
            percentile_bins=percentile_bins,
        ),
        plot_decile_lift(validated, output / f"{prefix}_decile_lift.png", evaluation_scope=normalized_scope),
        plot_calibration(
            validated,
            output / f"{prefix}_calibration.png",
            evaluation_scope=normalized_scope,
            calibration_bins=calibration_bins,
        ),
        plot_temporal_stability(
            validated,
            output / f"{prefix}_temporal_stability.png",
            evaluation_scope=normalized_scope,
        ),
    ]


    shap_inputs = (shap_values is not None, shap_feature_values is not None)
    if any(shap_inputs) and not all(shap_inputs):
        raise ValueError(
            "shap_values and shap_feature_values must be supplied together."
        )
    if all(shap_inputs):
        artifacts.append(
            plot_shap_beeswarm(
                shap_values,
                shap_feature_values,
                output / f"{prefix}_shap_beeswarm.png",
                evaluation_scope=normalized_scope,
                model_family=shap_model_family,
                max_display=shap_max_display,
                output_label=shap_output_label,
            )
        )


    families = sorted(validated["model_family"].astype(str).unique())
    return {
        "contract": "retention_performance_plots_v1",
        "evaluation_scope": normalized_scope,
        "evaluation_scope_label": scope_label,
        "prediction_rows": int(len(validated)),
        "model_families": families,
        "required_prediction_columns": list(REQUIRED_PREDICTION_COLUMNS),
        "ranking_scope": "within_fold",
        "artifacts": artifacts,
    }




__all__ = [
    "EVALUATION_SCOPE_LABELS",
    "REQUIRED_PREDICTION_COLUMNS",
    "generate_performance_plots",
    "plot_calibration",
    "plot_decile_lift",
    "plot_percentile_lift_gains",
    "plot_roc_pr",
    "plot_shap_beeswarm",
    "plot_temporal_stability",
    "validate_prediction_frame",
]
