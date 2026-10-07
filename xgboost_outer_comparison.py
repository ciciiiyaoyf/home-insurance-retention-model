"""Report an aligned, post-hoc comparison of two frozen XGBoost recipes.


This module consumes predictions only. It never fits, retunes, selects a winner,
opens a final test, or replaces the model used by the original pipeline run.
"""


from __future__ import annotations


import hashlib
import math
from pathlib import Path
from typing import Any
from uuid import uuid4


import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve




VARIANTS = ("baseline", "challenger")
REQUIRED_COLUMNS = {
    "variant", "model_family", "row_id", "fold", "cohort_date",
    "tenure_group", "y_retention", "p_retention",
}
HIGHER_IS_BETTER = {
    "retention_roc_auc", "churn_average_precision", "churn_ap_skill",
    "churn_precision_10pct", "churn_capture_10pct", "churn_recall_10pct",
    "top10_lift",
}
LOWER_IS_BETTER = {"log_loss", "brier_score", "absolute_calibration_gap"}
METRICS = (
    "retention_roc_auc", "churn_average_precision", "churn_ap_skill",
    "churn_prevalence_baseline", "churn_precision_10pct",
    "churn_capture_10pct", "churn_recall_10pct", "top10_lift",
    "log_loss", "brier_score", "actual_retention_rate",
    "mean_predicted_retention", "calibration_gap", "absolute_calibration_gap",
)




def validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Require the same five outer-fold populations and outcomes for both recipes."""
    if predictions.columns.duplicated().any():
        raise ValueError("Prediction column names must be unique.")
    missing = sorted(REQUIRED_COLUMNS.difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction input is missing columns: {missing}")
    frame = predictions.copy()
    if frame.empty or frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required prediction fields must be nonempty and nonmissing.")
    if set(frame["variant"]) != set(VARIANTS):
        raise ValueError("Prediction variants must be exactly baseline and challenger.")
    if set(frame["model_family"]) != {"xgboost"}:
        raise ValueError("This comparison accepts model_family=xgboost only.")
    for column in ("row_id", "tenure_group"):
        frame[column] = frame[column].astype(str)
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"{column} contains empty identities.")
    folds = pd.to_numeric(frame["fold"], errors="raise")
    if not np.isfinite(folds).all() or not folds.eq(np.floor(folds)).all():
        raise ValueError("Fold values must be finite integers.")
    frame["fold"] = folds.astype(int)
    frame["cohort_date"] = pd.to_datetime(frame["cohort_date"], errors="raise")
    if frame["cohort_date"].isna().any():
        raise ValueError("cohort_date contains missing dates after conversion.")
    targets = pd.to_numeric(frame["y_retention"], errors="raise")
    if not targets.isin([0, 1]).all():
        raise ValueError("y_retention must contain binary 0/1 outcomes.")
    frame["y_retention"] = targets.astype(int)
    probabilities = pd.to_numeric(frame["p_retention"], errors="raise")
    if not np.isfinite(probabilities).all() or not probabilities.between(0, 1).all():
        raise ValueError("p_retention must contain finite probabilities in [0, 1].")
    frame["p_retention"] = probabilities.astype(float)
    if frame.duplicated(["variant", "row_id"]).any():
        raise ValueError("A row_id must appear once per variant, including across folds.")
    aligned = {}
    for variant in VARIANTS:
        part = frame.loc[frame["variant"].eq(variant)]
        if set(part["fold"]) != set(range(1, 6)):
            raise ValueError(f"{variant} must contain exactly outer folds 1 through 5.")
        for fold, group in part.groupby("fold", sort=True):
            if set(group["y_retention"]) != {0, 1}:
                raise ValueError(f"{variant} fold {fold} must contain both target classes.")
        aligned[variant] = part.set_index("row_id").sort_index()
    baseline, challenger = (aligned[variant] for variant in VARIANTS)
    if not baseline.index.equals(challenger.index):
        raise ValueError("Baseline and challenger row identities do not match.")
    # All supplied identity/context columns must agree, including optional policy
    # keys. Only the variant name and its predicted probability may differ.
    comparable = [c for c in baseline.columns if c not in {"variant", "p_retention"}]
    for column in comparable:
        left, right = baseline[column], challenger[column]
        equal = (left.eq(right) | (left.isna() & right.isna())).fillna(False)
        if not bool(equal.all()):
            raise ValueError(f"Baseline/challenger alignment differs for {column}.")
    return frame.sort_values(["variant", "fold", "row_id"], kind="stable").reset_index(drop=True)




def _metrics(part: pd.DataFrame, model_diagnostics: Any) -> dict[str, Any]:
    result = dict(model_diagnostics.prediction_metrics(
        part.rename(columns={"y_retention": "actual_retention", "p_retention": "predicted_retention"}),
        "predicted_retention",
    ))
    result["absolute_calibration_gap"] = abs(float(result["calibration_gap"]))
    for metric in METRICS:
        if metric not in result or not math.isfinite(float(result[metric])):
            raise ValueError(f"Shared metric implementation returned invalid {metric}.")
    return result




def comparison_tables(
    frame: pd.DataFrame, model_diagnostics: Any, labels: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build macro-fold summaries and paired differences, without choosing a winner."""
    fold_rows, tenure_rows = [], []
    for (variant, fold), part in frame.groupby(["variant", "fold"], sort=True, observed=True):
        base = {"variant": variant, "model_label": labels[variant], "model_family": "xgboost", "fold": int(fold)}
        fold_rows.append({**base, **_metrics(part, model_diagnostics)})
        for tenure, group in part.groupby("tenure_group", sort=True, observed=True):
            row = {**base, "tenure_group": str(tenure), "rows": int(len(group))}
            if set(group["y_retention"]) != {0, 1}:
                tenure_rows.append({**row, "status": "ONE_TARGET_CLASS"})
            else:
                tenure_rows.append({**row, "status": "SCORED", **_metrics(group, model_diagnostics)})
    folds = pd.DataFrame(fold_rows)
    summaries = []
    for variant in VARIANTS:
        part = folds.loc[folds["variant"].eq(variant)]
        row = {
            "variant": variant, "model_label": labels[variant],
            "model_family": "xgboost", "outer_folds": int(len(part)),
            "rows": int(part["rows"].sum()),
            "aggregation": "equal_weight_mean_and_sample_sd_across_folds",
        }
        for metric in METRICS:
            row[f"mean_{metric}"] = float(part[metric].mean())
            row[f"sd_{metric}"] = float(part[metric].std(ddof=1))
        summaries.append(row)
    paired = []
    left = folds.loc[folds["variant"].eq("baseline")].set_index("fold").sort_index()
    right = folds.loc[folds["variant"].eq("challenger")].set_index("fold").sort_index()
    for metric in METRICS:
        differences = right[metric] - left[metric]
        if metric in HIGHER_IS_BETTER:
            direction, interpretation = "higher_is_better", "positive difference favors challenger"
        elif metric in LOWER_IS_BETTER:
            direction, interpretation = "lower_is_better", "negative difference favors challenger"
        elif metric == "calibration_gap":
            direction, interpretation = "signed_bias_only", "compare absolute_calibration_gap; signed change alone is not improvement"
        else:
            direction, interpretation = "descriptive", "no universal improvement direction"
        row = {
            "metric": metric, "difference_definition": "challenger_minus_baseline",
            "direction": direction, "interpretation": interpretation,
            "outer_folds": int(len(differences)),
            "baseline_macro_mean": float(left[metric].mean()),
            "challenger_macro_mean": float(right[metric].mean()),
            "mean_difference": float(differences.mean()),
            "sample_sd_difference": float(differences.std(ddof=1)),
            "paired_standard_error": float(differences.std(ddof=1) / math.sqrt(len(differences))),
        }
        for fold, value in differences.items():
            row[f"fold_{int(fold)}_difference"] = float(value)
        paired.append(row)
    return folds, pd.DataFrame(summaries), pd.DataFrame(paired), pd.DataFrame(tenure_rows)




def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)




def _atomic_text(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)




def _plots(frame: pd.DataFrame, folds: pd.DataFrame, labels: dict[str, str], output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter


    colors = {"baseline": "#1967A5", "challenger": "#E67326"}
    files = []


    def save(fig: Any, name: str) -> None:
        destination = output / name
        temporary = destination.with_name(f".{destination.stem}.{uuid4().hex}.tmp")
        try:
            fig.tight_layout()
            fig.savefig(temporary, format="png", dpi=160, bbox_inches="tight", facecolor="white")
            temporary.replace(destination)
            files.append(destination)
        finally:
            plt.close(fig)
            temporary.unlink(missing_ok=True)


    grid = np.linspace(0.0, 1.0, 501)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for variant in VARIANTS:
        roc_lines, pr_lines = [], []
        for _, group in frame.loc[frame["variant"].eq(variant)].groupby("fold", sort=True):
            y, p = 1 - group["y_retention"].to_numpy(), 1 - group["p_retention"].to_numpy()
            fpr, tpr, _ = roc_curve(y, p)
            precision, recall, _ = precision_recall_curve(y, p)
            roc_lines.append(np.interp(grid, fpr, tpr))
            pr_lines.append(np.interp(grid, recall[::-1], precision[::-1]))
        part = folds.loc[folds["variant"].eq(variant)]
        for axis, curves, metric, short_name in (
            (axes[0], roc_lines, "retention_roc_auc", "AUC"),
            (axes[1], pr_lines, "churn_average_precision", "AP"),
        ):
            values = np.asarray(curves)
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            label = f"{labels[variant]}: {short_name} {part[metric].mean():.3f}; fold SD {part[metric].std(ddof=1):.3f}"
            axis.plot(grid, mean, color=colors[variant], label=label, linewidth=2)
            axis.fill_between(grid, np.clip(mean-sd, 0, 1), np.clip(mean+sd, 0, 1), color=colors[variant], alpha=0.10)
    axes[0].plot([0, 1], [0, 1], "--", color="#777777", label="Random ranking")
    prevalence = folds.loc[folds["variant"].eq("baseline"), "churn_prevalence_baseline"].mean()
    axes[1].axhline(prevalence, color="#777777", linestyle="--", label=f"Mean fold churn prevalence: {prevalence:.1%}")
    axes[0].set(title="ROC: equal-weight average of fold curves", xlabel="False-positive rate", ylabel="True-positive rate / churn")
    axes[1].set(title="Precision-recall: equal-weight average of fold curves", xlabel="Recall / churn capture", ylabel="Precision / churn rate")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="best")
    fig.suptitle("XGBoost recipe comparison - outer validation\nShading is between-fold SD, not a confidence interval", fontsize=13)
    save(fig, "outer_roc_pr_comparison.png")


    working = frame.assign(month=frame["cohort_date"].dt.to_period("M").astype(str))
    monthly = working.groupby(["variant", "month"], sort=True, observed=True).agg(
        actual=("y_retention", "mean"), predicted=("p_retention", "mean"), rows=("row_id", "size")
    ).reset_index()
    baseline = monthly.loc[monthly["variant"].eq("baseline")]
    months = baseline["month"].tolist()
    x = np.arange(len(months))
    fig, (upper, lower) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    upper.plot(x, baseline["actual"], "o-", color="#252525", label="Actual retention", markersize=4)
    for variant in VARIANTS:
        part = monthly.loc[monthly["variant"].eq(variant)].set_index("month").loc[months]
        upper.plot(x, part["predicted"], "o-", color=colors[variant], label=labels[variant], markersize=4)
    upper.set(title="Actual versus predicted retention - outer validation months", ylabel="Retention rate")
    upper.yaxis.set_major_formatter(PercentFormatter(1))
    upper.legend(fontsize=9)
    upper.grid(alpha=0.2)
    lower.bar(x, baseline["rows"], color="#A9C5DB")
    stride = max(1, math.ceil(len(months) / 18))
    tick_indices = x[::stride]
    lower.set_xticks(tick_indices, [months[i] for i in tick_indices], rotation=45, ha="right")
    lower.set(ylabel="Rows", xlabel="Month-1 cohort month")
    save(fig, "outer_temporal_comparison.png")


    fig, axis = plt.subplots(figsize=(10, 5))
    for variant in VARIANTS:
        part = folds.loc[folds["variant"].eq(variant)].sort_values("fold")
        axis.plot(part["fold"], part["churn_capture_10pct"], "o-", color=colors[variant], label=labels[variant], linewidth=2)
    axis.set_xticks(sorted(folds["fold"].unique()))
    axis.set(title="Churn captured in the highest-risk 10% - by outer fold", xlabel="Outer fold", ylabel="Fraction of all churn captured")
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.grid(alpha=0.2)
    axis.legend()
    save(fig, "outer_capture_comparison.png")
    return files




def write_comparison(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    *,
    model_diagnostics: Any,
    baseline_label: str,
    challenger_label: str,
) -> dict[str, Any]:
    """Write paired scorecards and plots for already generated outer predictions."""
    labels = {"baseline": str(baseline_label).strip(), "challenger": str(challenger_label).strip()}
    if not all(labels.values()) or len(set(labels.values())) != len(labels):
        raise ValueError("Baseline and challenger labels must be nonempty and distinct.")
    frame = validate_predictions(predictions)
    folds, summary, paired, tenure = comparison_tables(frame, model_diagnostics, labels)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for name, table in (
        ("comparison_fold_metrics.csv", folds),
        ("comparison_summary.csv", summary),
        ("paired_metric_differences.csv", paired),
        ("comparison_tenure_metrics.csv", tenure),
    ):
        path = output / name
        _atomic_csv(table, path)
        files.append(path)
    files.extend(_plots(frame, folds, labels, output))
    report = (
        "# XGBoost outer-validation recipe comparison\n\n"
        f"Baseline: {labels['baseline']}\n\nChallenger: {labels['challenger']}\n\n"
        f"Both recipes score the same {int(summary.iloc[0]['rows']):,} rows across "
        f"{int(summary.iloc[0]['outer_folds'])} chronological outer validation folds. "
        "This is a post-hoc comparison on existing outer validation periods, "
        "not a fresh independent test or proof that one recipe generalizes better.\n\n"
        "## Read the results\n\n"
        "- comparison_fold_metrics.csv gives each recipe's complete scorecard per fold.\n"
        "- comparison_summary.csv gives equal-weight fold means and sample SDs; "
        "ranking metrics are not calculated after pooling predictions from different fold models.\n"
        "- paired_metric_differences.csv subtracts baseline from challenger within each fold. "
        "Positive AP skill, capture, precision, lift, or AUC changes favor the challenger. "
        "Negative log loss, Brier score, or absolute calibration-gap changes favor it. "
        "A signed calibration-gap change alone does not indicate improvement.\n"
        "- Paired standard errors summarize variability in fold differences. Chronological "
        "folds can be dependent; these values are not a formal noninferiority test or "
        "a guarantee of statistical significance.\n"
        "- comparison_tenure_metrics.csv checks whether aggregate results conceal "
        "weaker performance in particular tenure groups. A single-class group is "
        "marked ONE_TARGET_CLASS rather than assigned a fabricated ranking score.\n\n"
        "## Decision boundary\n\n"
        "The report does not automatically choose a winner or replace an existing "
        "deployment artifact. Assess the paired performance differences, period and "
        "tenure stability, calibration, and implementation cost together. A smaller "
        "recipe can be preferable with acceptably similar performance, but fewer "
        "features do not by themselves imply better performance.\n\n"
        "No final-test predictions are required or produced by this reporting module. "
        "If a final test was already inspected while changing the model, it cannot "
        "be treated as untouched confirmation of the revised choice.\n"
    )
    report_path = output / "report.md"
    _atomic_text(report, report_path)
    files.append(report_path)
    return {
        "contract": "xgboost_frozen_recipe_outer_comparison_report_v1",
        "evaluation_scope": "posthoc_outer_validation_comparison",
        "baseline_label": labels["baseline"],
        "challenger_label": labels["challenger"],
        "outer_folds": sorted(frame["fold"].unique().astype(int).tolist()),
        "rows_per_variant": int(summary.iloc[0]["rows"]),
        "automatic_winner_selected": False,
        "final_test_scored": False,
        "artifacts": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in files
        ],
    }
