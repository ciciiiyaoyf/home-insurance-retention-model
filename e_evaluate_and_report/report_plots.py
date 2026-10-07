"""Plots from supplied saved-model predictions; no fitting or model scoring."""
from __future__ import annotations


from pathlib import Path
from typing import Any


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve




_COLORS = {"train": "#1765ad", "validation": "#e87527"}
_FAMILY_LABELS = {"logistic": "Logistic", "xgboost": "XGBoost"}




def _positive_integer(value: Any, name: str) -> int:
    number = float(value)
    if not np.isfinite(number) or number < 1 or number != int(number):
        raise ValueError(f"{name} must be a positive integer.")
    return int(number)




def _validated_record(record: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    family, split = str(record["model_family"]), str(record["split"])
    if family not in {"logistic", "xgboost"} or split not in _COLORS:
        raise ValueError("Only Logistic/XGBoost train and validation records are supported.")
    y = np.asarray(record["y_retention"], dtype=float)
    p = np.asarray(record["p_retention"], dtype=float)
    if y.ndim != 1 or p.shape != y.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("Targets must be a binary vector containing both classes; predictions must align.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Predictions must be finite probabilities in [0, 1].")
    rows = _positive_integer(record["rows"], "rows")
    if rows != len(y):
        raise ValueError("Declared row count does not match predictions.")
    population = _positive_integer(record.get("training_population_rows", rows), "population rows")
    if split == "train" and population < rows:
        raise ValueError("Training population cannot be smaller than scored training rows.")
    result.update(y_retention=y, p_retention=p, rows=rows, training_population_rows=population,
                  fold=_positive_integer(record["fold"], "fold"))
    return result




def write_split_curves(records: list[dict], output_dir: Path) -> dict[str, Any]:
    """Compare apparent training and outer OOF validation, macro-averaged over five folds.


    The sealed final test is explicitly annotated as unavailable, never drawn as
    a dummy curve. Each scalar AUC/AP is calculated on its own fold's rows.
    """
    validated = [_validated_record(record) for record in records]
    if not validated:
        raise ValueError("No prediction records were supplied.")
    keys = [(r["model_family"], r["split"], r["fold"]) for r in validated]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate model/split/fold records.")
    families = sorted({r["model_family"] for r in validated})
    for family in families:
        for split in _COLORS:
            if {r["fold"] for r in validated if r["model_family"] == family and r["split"] == split} != set(range(1, 6)):
                raise ValueError("Each family requires identical five folds for train and validation.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0, 1, 501)
    metrics, points, artifacts = [], [], []
    curves = {}
    for r in sorted(validated, key=lambda v: (v["model_family"], v["split"], v["fold"])):
        y, p = r["y_retention"], r["p_retention"]
        churn, churn_p = 1 - y, 1 - p
        fpr, tpr, _ = roc_curve(churn, churn_p)
        precision, recall, _ = precision_recall_curve(churn, churn_p)
        roc_y = np.interp(grid, fpr, tpr)
        pr_y = np.interp(grid, recall[::-1], precision[::-1])
        key = (r["model_family"], r["split"])
        curves.setdefault(key, []).append((roc_y, pr_y))
        metrics.append({"model_family": key[0], "split": key[1], "fold": r["fold"],
                        "rows": r["rows"], "training_population_rows": r["training_population_rows"] if key[1] == "train" else None,
                        "retention_roc_auc": roc_auc_score(y, p),
                        "churn_average_precision": average_precision_score(churn, churn_p),
                        "churn_prevalence": churn.mean(), "final_test_evaluated": False})
        for i, x in enumerate(grid):
            points.append({"model_family": key[0], "split": key[1], "fold": r["fold"],
                           "grid_value": x, "roc_true_positive_rate": roc_y[i], "pr_precision": pr_y[i]})
    metric_frame = pd.DataFrame(metrics)
    metric_path, point_path = output / "split_curve_metrics.csv", output / "split_curve_points.csv"
    metric_frame.to_csv(metric_path, index=False)
    pd.DataFrame(points).to_csv(point_path, index=False)
    for family in families:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for split, color in _COLORS.items():
            selected = metric_frame.loc[(metric_frame.model_family == family) & (metric_frame.split == split)]
            sampled = split == "train" and bool((selected.rows < selected.training_population_rows).any())
            label = "Train (in-sample" + ("; sampled)" if sampled else ")") if split == "train" else "Validation (outer OOF)"
            for panel, column, metric_name in ((0, "retention_roc_auc", "AUC"), (1, "churn_average_precision", "AP")):
                matrix = np.stack([values[panel] for values in curves[(family, split)]])
                mean, spread = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
                axes[panel].plot(grid, mean, color=color, linewidth=2,
                                 label=f"{label}: mean {metric_name} {selected[column].mean():.3f}; fold SD {selected[column].std(ddof=1):.3f}")
                axes[panel].fill_between(grid, np.clip(mean - spread, 0, 1), np.clip(mean + spread, 0, 1), color=color, alpha=.08)
            axes[1].axhline(selected.churn_prevalence.mean(), color=color, linestyle=":", linewidth=1.2,
                            label=f"{split.title()} mean churn baseline: {selected.churn_prevalence.mean():.1%}")
        axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random ranking")
        for ax, title, xlabel, ylabel in zip(axes, ["ROC - mean of fold curves", "Precision-recall - mean of fold curves"],
                                             ["False-positive rate (churn)", "Recall / churn capture"], ["True-positive rate (churn)", "Precision / churn rate"]):
            ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1))
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=.15)
            ax.legend(loc="best", fontsize=8)
        fig.suptitle(f"{_FAMILY_LABELS[family]} - Train / Validation / Test status", fontsize=15, weight="bold")
        fig.text(.5, .025, "Final test: not evaluated (sealed). Shading = between-fold SD. Training curves are apparent fit, not independent performance.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .06, 1, .94))
        path = output / f"{family}_train_validation_test_roc_pr.png"
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)
        artifacts.append(str(path))
    return {"artifacts": artifacts, "metric_csv": str(metric_path), "curve_csv": str(point_path),
            "final_test_evaluated": False, "aggregation": "equal-weight mean of five fold metrics; curves separately interpolated"}




def _history_frame(monthly: pd.DataFrame) -> pd.DataFrame:
    required = {"YEAR_MONTH", "policies", "actual_retention", "predicted_retention", "selected_history_training_rows"}
    if not required.issubset(monthly.columns) or monthly.empty:
        raise ValueError(f"Monthly table must be nonempty and contain {sorted(required)}.")
    frame = monthly.copy()
    frame["YEAR_MONTH"] = pd.PeriodIndex(frame.YEAR_MONTH.astype(str), freq="M").astype(str)
    if frame.YEAR_MONTH.duplicated().any():
        raise ValueError("Monthly table contains duplicate months.")
    for column in ("policies", "selected_history_training_rows"):
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values).all() or (values != np.floor(values)).any() or (values < 0).any():
            raise ValueError("Monthly counts must be finite nonnegative integers.")
        frame[column] = values.astype("int64")
    if (frame.policies < 1).any() or (frame.selected_history_training_rows > frame.policies).any():
        raise ValueError("Monthly training counts must be within nonempty population counts.")
    mask = frame.selected_history_training_rows.gt(0)
    if not mask.any():
        raise ValueError("No final-refit training rows are identified.")
    for column in ("actual_retention", "predicted_retention", "training_actual_retention", "training_predicted_retention"):
        if column not in frame:
            raise ValueError(f"Missing monthly rates: {column}.")
        values = pd.to_numeric(frame.loc[mask if column.startswith("training_") else slice(None), column], errors="raise")
        if not np.isfinite(values).all() or not values.between(0, 1).all():
            raise ValueError(f"{column} must contain finite probabilities for its populated months.")
    return frame.sort_values("YEAR_MONTH", kind="stable").reset_index(drop=True)




def write_history_trends(monthly: pd.DataFrame, output_dir: Path, family: str,
                         training_start: str, training_end: str) -> dict[str, Any]:
    """Write all-pre-holdout backcast and exact training-membership trend views.


    Training-subset rates are mandatory: a partial boundary month must not
    borrow the actual or predicted rate of its nontraining rows.
    """
    if family not in {"logistic", "xgboost"}:
        raise ValueError("Unknown model family.")
    frame = _history_frame(monthly)
    start, end = pd.Period(training_start, freq="M"), pd.Period(training_end, freq="M")
    if start > end:
        raise ValueError("Training window is reversed.")
    train_mask = frame.selected_history_training_rows.gt(0)
    periods = pd.PeriodIndex(frame.YEAR_MONTH, freq="M")
    if ((periods[train_mask] < start) | (periods[train_mask] > end)).any():
        raise ValueError("Training membership lies outside the saved training window.")
    selected = frame.loc[train_mask].copy()
    selected["policies"] = selected.selected_history_training_rows
    selected["actual_retention"] = selected.training_actual_retention
    selected["predicted_retention"] = selected.training_predicted_retention
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for name, part, title in (("full_history", frame, "All observed pre-holdout history"),
                              ("selected_training_history", selected, "Exact selected training population")):
        csv_path = output / f"{family}_final_refit_actual_vs_predicted_{name}.csv"
        image_path = csv_path.with_suffix(".png")
        part.to_csv(csv_path, index=False)
        dates = pd.PeriodIndex(part.YEAR_MONTH, freq="M").to_timestamp()
        fig, (ax, volume) = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
        ax.plot(dates, part.actual_retention, color="#222222", linewidth=2, label="Actual retention")
        ax.plot(dates, part.predicted_retention, color="#1765ad", linewidth=2, label=f"Predicted retention ({_FAMILY_LABELS[family]} final refit)")
        ax.axvspan(start.to_timestamp(), (end + 1).to_timestamp(), color="#e87527", alpha=.08,
                   label="Selected training months (boundary month may be partial)")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set(ylabel="Retention rate", title=title)
        ax.grid(alpha=.15)
        ax.legend(loc="best", fontsize=8)
        volume.bar(dates, part.policies, width=24, color="#b9c8db", label="Rows in this view")
        volume.bar(dates, part.selected_history_training_rows, width=24, color="#6b8bb4", label="Final-refit training rows")
        volume.set(ylabel="Policy-term rows", xlabel="Month-1 cohort month")
        volume.set_xlim(dates.min() - pd.Timedelta(days=15), dates.max() + pd.Timedelta(days=15))
        volume.legend(loc="best", fontsize=8)
        volume.tick_params(axis="x", rotation=35)
        fig.suptitle(f"{_FAMILY_LABELS[family]} - Actual vs. predicted retention", fontsize=15, weight="bold")
        fig.text(.5, .02, "Retrospective fitted/backcast diagnostic, not OOF validation. Sealed final holdout excluded.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .95))
        fig.savefig(image_path, dpi=180, facecolor="white")
        plt.close(fig)
        artifacts.extend([str(image_path), str(csv_path)])
    return {"artifacts": artifacts, "model_family": family, "final_holdout_scored": False,
            "scope": "retrospective_fitted_backcast_not_oof", "history_months": len(frame),
            "selected_training_rows": int(selected.policies.sum())}




def _validated_final_test_record(record: dict[str, Any], expected_split: str) -> dict[str, Any]:
    """Validate one prediction set without implying permission to score a test."""
    if record.get("split") != expected_split:
        raise ValueError(f"Expected a {expected_split!r} prediction record.")
    # Reuse the strict binary/probability/count checks already used by the
    # five-fold reports. Single-model training and test sets have no fold.
    checked = dict(record, split="train" if expected_split == "train" else "validation",
                   fold=record.get("fold", 1) if expected_split == "validation" else 1)
    result = _validated_record(checked)
    result["split"] = expected_split
    if expected_split != "validation":
        result["fold"] = None
    return result




def write_final_test_comparison(training_record: dict, validation_records: list[dict],
                                test_record: dict, output_dir: Path) -> dict[str, Any]:
    """Draw the selected family's train, outer-validation, and final-test curves.


    Train and test are predictions from the same saved final model. Validation
    consists of five separately refitted outer models. Validation curves and
    scalar metrics are equal-weight fold means; test and train are single-set
    curves/metrics with no fabricated between-fold standard deviation. This
    function accepts predictions only; it never loads a model or opens data.
    """
    train = _validated_final_test_record(training_record, "train")
    test = _validated_final_test_record(test_record, "test")
    validation = [_validated_final_test_record(record, "validation") for record in validation_records]
    folds = [record["fold"] for record in validation]
    if len(folds) != 5 or set(folds) != set(range(1, 6)):
        raise ValueError("Final-test comparison requires exactly validation folds 1 through 5.")
    records = [train, *sorted(validation, key=lambda record: record["fold"]), test]
    family = train["model_family"]
    if any(record["model_family"] != family for record in records):
        raise ValueError("Train, validation, and test must use the same selected model family.")


    grid = np.linspace(0, 1, 501)
    metrics, points = [], []
    curves: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {"train": [], "validation": [], "test": []}
    for record in records:
        y, p = record["y_retention"], record["p_retention"]
        churn, churn_p = 1 - y, 1 - p
        fpr, tpr, _ = roc_curve(churn, churn_p)
        precision, recall, _ = precision_recall_curve(churn, churn_p)
        roc_y = np.interp(grid, fpr, tpr)
        pr_y = np.interp(grid, recall[::-1], precision[::-1])
        split = record["split"]
        curves[split].append((roc_y, pr_y))
        metrics.append({"model_family": family, "split": split,
                        "record_kind": "fold" if split == "validation" else "single_dataset",
                        "fold": record["fold"], "rows": record["rows"],
                        "training_population_rows": record["training_population_rows"] if split == "train" else None,
                        "retention_roc_auc": float(roc_auc_score(y, p)),
                        "churn_average_precision": float(average_precision_score(churn, churn_p)),
                        "churn_prevalence": float(churn.mean()),
                        "retention_roc_auc_fold_sd": None, "churn_average_precision_fold_sd": None,
                        "final_test_evaluated": True})
        for i, x in enumerate(grid):
            points.append({"model_family": family, "split": split,
                           "record_kind": "fold" if split == "validation" else "single_dataset",
                           "fold": record["fold"], "grid_value": x,
                           "roc_true_positive_rate": roc_y[i], "pr_precision": pr_y[i]})
    fold_metrics = pd.DataFrame([row for row in metrics if row["split"] == "validation"])
    validation_summary = {"model_family": family, "split": "validation", "record_kind": "macro_mean",
                          "fold": None, "rows": int(fold_metrics.rows.sum()), "training_population_rows": None,
                          "retention_roc_auc": float(fold_metrics.retention_roc_auc.mean()),
                          "churn_average_precision": float(fold_metrics.churn_average_precision.mean()),
                          "churn_prevalence": float(fold_metrics.churn_prevalence.mean()),
                          "retention_roc_auc_fold_sd": float(fold_metrics.retention_roc_auc.std(ddof=1)),
                          "churn_average_precision_fold_sd": float(fold_metrics.churn_average_precision.std(ddof=1)),
                          "final_test_evaluated": True}
    metrics.append(validation_summary)
    validation_mean = np.stack(curves["validation"]).mean(axis=0)
    for i, x in enumerate(grid):
        points.append({"model_family": family, "split": "validation", "record_kind": "macro_mean",
                       "fold": None, "grid_value": x, "roc_true_positive_rate": validation_mean[0, i],
                       "pr_precision": validation_mean[1, i]})
    summaries = {"train": metrics[0], "validation": validation_summary, "test": metrics[-2]}
    sampled = train["rows"] < train["training_population_rows"]
    labels = {"train": "Train (in-sample" + ("; sampled)" if sampled else ")"),
              "validation": "Validation (outer fold mean)", "test": "Test (final holdout)"}
    colors = {"train": "#1765ad", "validation": "#e87527", "test": "#29955c"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for split, color in colors.items():
        matrix = np.stack(curves[split])
        mean = matrix.mean(axis=0)
        summary = summaries[split]
        for panel, column, name in ((0, "retention_roc_auc", "AUC"), (1, "churn_average_precision", "AP")):
            metric_label = f"{labels[split]}: {name} {summary[column]:.3f}"
            if split == "validation":
                metric_label += f"; fold SD {summary[column + '_fold_sd']:.3f}"
                spread = matrix[:, panel, :].std(axis=0, ddof=1)
                axes[panel].fill_between(grid, np.clip(mean[panel] - spread, 0, 1),
                                         np.clip(mean[panel] + spread, 0, 1), color=color, alpha=.08)
            axes[panel].plot(grid, mean[panel], color=color, linewidth=2, label=metric_label)
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random ranking")
    baselines = " | ".join(f"{split.title()}: {summaries[split]['churn_prevalence']:.1%}" for split in colors)
    for ax, title, xlabel, ylabel in zip(axes, ["ROC", "Precision-recall"],
                                         ["False-positive rate (churn)", "Recall / churn capture"],
                                         ["True-positive rate (churn)", "Precision / churn rate"]):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.15)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"{_FAMILY_LABELS[family]} - Train / Validation / Test", fontsize=15, weight="bold")
    fig.text(.5, .075, "Train and test use the same frozen final model. Validation uses five separate outer-fold refits.",
             ha="center", fontsize=9)
    fig.text(.5, .045, "Training is apparent fit. Shading = validation between-fold SD, not a confidence interval.",
             ha="center", fontsize=9)
    fig.text(.5, .015, "Churn prevalence (validation = mean fold baseline): " + baselines, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .94))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    image_path, metric_path, point_path = (output / name for name in
                                          ("train_validation_test_roc_pr.png", "split_metrics.csv", "curve_points.csv"))
    fig.savefig(image_path, dpi=180, facecolor="white")
    plt.close(fig)
    pd.DataFrame(metrics).to_csv(metric_path, index=False)
    pd.DataFrame(points).to_csv(point_path, index=False)
    return {"artifacts": [str(image_path), str(metric_path), str(point_path)], "model_family": family,
            "final_test_evaluated": True, "train_sampled": sampled,
            "aggregation": {"train": "single final-model dataset", "validation": "equal-weight mean of five outer folds",
                            "test": "single final-model held-out dataset"}}
