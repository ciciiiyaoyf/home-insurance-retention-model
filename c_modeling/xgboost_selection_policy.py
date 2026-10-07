"""Practical parsimony for the final, already-retuned XGBoost feature sets.


Only inner-fold metrics enter the decision. Seeds must have been combined
before scoring each fold. These are explicit engineering tolerances, not a
formal equivalence test or a claim that examined test data remain untouched.
"""
from __future__ import annotations


import hashlib
import json
from typing import Any, Mapping


import numpy as np
import pandas as pd




FINAL_FEATURE_SELECTION_RULE = "PRACTICAL_AP_RECALL_LIMITS_WITH_FOLD_GUARDS_V1"
DEFAULT_POLICY = {
    "version": 1,
    "mean_ap_skill_loss": 0.001,
    "mean_recall_loss": 0.001,
    "max_fold_ap_skill_loss": 0.003,
    "max_fold_recall_loss": 0.005,
}




def validated_policy(policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate explicit absolute-score losses; recall 0.001 means 0.1 pp."""
    result = dict(DEFAULT_POLICY)
    if policy is not None:
        if not isinstance(policy, Mapping):
            raise ValueError("Selection policy must be a JSON object.")
        unknown = set(policy) - set(result)
        if unknown:
            raise ValueError(f"Unknown selection policy fields: {sorted(unknown)}")
        result.update(policy)
    if isinstance(result["version"], bool) or result["version"] != 1:
        raise ValueError("Only practical feature-selection policy version 1 is supported.")
    result["version"] = 1
    for key in set(result) - {"version"}:
        value = result[key]
        if isinstance(value, (bool, str)):
            raise ValueError(f"Policy {key} must be a finite numeric loss.")
        try:
            value = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid policy value: {key}") from exc
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Policy {key} must lie between 0 and 1.")
        result[key] = value
    for metric in ("ap_skill", "recall"):
        if result[f"max_fold_{metric}_loss"] < result[f"mean_{metric}_loss"]:
            raise ValueError(f"The {metric} fold limit cannot be below its mean limit.")
    return result




def policy_sha256(policy: Mapping[str, Any] | None = None) -> str:
    payload = {"rule": FINAL_FEATURE_SELECTION_RULE, "limits": validated_policy(policy)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()




def select_practical_feature_group(
    long_frame: pd.DataFrame,
    *,
    id_col: str = "group_id",
    policy: Mapping[str, Any] | None = None,
) -> tuple[str, pd.DataFrame, str]:
    """Prefer the smallest finalist meeting mean-loss and worst-fold limits.


    AP reference: highest mean AP skill. Recall reference: highest mean recall
    among finalists that pass BOTH AP limits. Check mean and worst-fold recall
    against that single reference. Retain paired SEs as diagnostics only; a
    noisy candidate never receives a larger practical loss allowance.
    """
    limits = validated_policy(policy)
    frame = long_frame.copy()
    if "churn_recall_10pct" not in frame and "churn_capture_10pct" in frame:
        frame["churn_recall_10pct"] = frame["churn_capture_10pct"]
    required = {id_col, "fold", "raw_feature_count", "churn_ap_skill", "churn_recall_10pct"}
    missing = required - set(frame)
    if frame.empty or missing:
        raise ValueError(f"Finalist metrics are empty or missing fields: {sorted(missing)}")
    if frame[id_col].isna().any() or frame[id_col].astype(str).str.strip().eq("").any():
        raise ValueError("Feature-group identifiers must be nonempty.")
    frame[id_col] = frame[id_col].astype(str)
    for key in ("fold", "raw_feature_count"):
        values = pd.to_numeric(frame[key], errors="raise")
        if not np.isfinite(values).all() or not values.eq(np.floor(values)).all() or values.le(0).any():
            raise ValueError(f"{key} must contain positive integers.")
        frame[key] = values.astype("int64")
    for key in ("churn_ap_skill", "churn_recall_10pct"):
        frame[key] = pd.to_numeric(frame[key], errors="raise").astype(float)
        if not np.isfinite(frame[key]).all():
            raise ValueError(f"Nonfinite finalist metric: {key}")
    if not frame["churn_recall_10pct"].between(0, 1).all() or frame["churn_ap_skill"].gt(1).any():
        raise ValueError("Recall must be in [0,1] and AP skill cannot exceed 1.")
    if "churn_capture_10pct" in frame:
        capture = pd.to_numeric(frame["churn_capture_10pct"], errors="raise")
        if not np.allclose(capture, frame["churn_recall_10pct"], atol=1e-12, rtol=0):
            raise ValueError("Capture and recall aliases disagree.")
    if frame.duplicated([id_col, "fold"]).any():
        raise ValueError("Expected one ensemble metric row per finalist and fold.")
    parts = {key: part.sort_values("fold").set_index("fold")
             for key, part in frame.groupby(id_col, sort=True)}
    folds = next(iter(parts.values())).index.tolist()
    if len(folds) < 3 or any(part.index.tolist() != folds for part in parts.values()):
        raise ValueError("Finalists must share at least three identical chronological folds.")
    if any(part["raw_feature_count"].nunique() != 1 for part in parts.values()):
        raise ValueError("A finalist's feature count changed between folds.")
    for key in ("validation_rows", "churn_prevalence_baseline"):
        if key in frame:
            baseline = pd.to_numeric(next(iter(parts.values()))[key], errors="raise")
            for part in parts.values():
                values = pd.to_numeric(part[key], errors="raise")
                if not np.isfinite(values).all() or not np.allclose(values, baseline, rtol=0, atol=1e-12):
                    raise ValueError(f"Finalists have different per-fold {key}.")


    rows = []
    diagnostic_metrics = ("retention_roc_auc", "churn_average_precision", "top10_lift",
                          "churn_precision_10pct", "log_loss", "brier_score", "calibration_gap")
    for key, part in parts.items():
        row = {id_col: key, "raw_feature_count": int(part["raw_feature_count"].iloc[0]),
               "fold_count": len(folds), "mean_churn_ap_skill": float(part["churn_ap_skill"].mean()),
               "mean_churn_recall_10pct": float(part["churn_recall_10pct"].mean())}
        for metric in diagnostic_metrics:
            if metric in part:
                values = pd.to_numeric(part[metric], errors="raise")
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite finalist diagnostic: {metric}")
                row[f"mean_{metric}"] = float(values.mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    ap_ref = summary.sort_values(["mean_churn_ap_skill", "mean_churn_recall_10pct",
                                  "raw_feature_count", id_col],
                                 ascending=[False, False, True, True]).iloc[0][id_col]
    epsilon = 16 * np.finfo(float).eps


    def statistics(candidate: str, reference: str, metric: str) -> tuple[float, float, float]:
        delta = (parts[reference][metric] - parts[candidate][metric]).to_numpy(dtype=float)
        return float(delta.mean()), float(delta.max()), float(delta.std(ddof=1) / np.sqrt(len(delta)))


    ap_stats = {key: statistics(key, ap_ref, "churn_ap_skill") for key in parts}
    ap_pass = {key for key, (mean, worst, _) in ap_stats.items()
               if mean <= limits["mean_ap_skill_loss"] + epsilon
               and worst <= limits["max_fold_ap_skill_loss"] + epsilon}
    recall_ref = summary.loc[summary[id_col].isin(ap_pass)].sort_values(
        ["mean_churn_recall_10pct", "mean_churn_ap_skill", "raw_feature_count", id_col],
        ascending=[False, False, True, True]).iloc[0][id_col]
    annotations = []
    for key in parts:
        ap_mean, ap_worst, ap_se = ap_stats[key]
        rec_mean, rec_worst, rec_se = statistics(key, recall_ref, "churn_recall_10pct")
        ap_mean_pass = ap_mean <= limits["mean_ap_skill_loss"] + epsilon
        ap_fold_pass = ap_worst <= limits["max_fold_ap_skill_loss"] + epsilon
        rec_mean_pass = rec_mean <= limits["mean_recall_loss"] + epsilon
        rec_fold_pass = rec_worst <= limits["max_fold_recall_loss"] + epsilon
        checks = {"mean_ap_skill": ap_mean_pass, "worst_fold_ap_skill": ap_fold_pass,
                  "mean_recall": rec_mean_pass, "worst_fold_recall": rec_fold_pass}
        annotations.append({id_col: key, "selection_rule": FINAL_FEATURE_SELECTION_RULE,
            "ap_reference_id": ap_ref, "recall_reference_id": recall_ref,
            "mean_ap_skill_loss_vs_reference": ap_mean, "max_fold_ap_skill_loss_vs_reference": ap_worst,
            "se_ap_skill_loss_vs_reference": ap_se, "mean_recall_10pct_loss_vs_reference": rec_mean,
            "max_fold_recall_10pct_loss_vs_reference": rec_worst, "se_recall_10pct_loss_vs_reference": rec_se,
            "passes_mean_ap_limit": ap_mean_pass, "passes_fold_ap_limit": ap_fold_pass,
            "passes_mean_recall_limit": rec_mean_pass, "passes_fold_recall_limit": rec_fold_pass,
            "eligible_performance_group": all(checks.values()),
            "failed_checks": ";".join(name for name, passed in checks.items() if not passed),
            **{f"policy_{name}": value for name, value in limits.items()}})
    annotated = summary.merge(pd.DataFrame(annotations), on=id_col, validate="one_to_one")
    eligible = annotated.loc[annotated["eligible_performance_group"]]
    if eligible.empty:
        # At least the AP-qualified recall leader must qualify against itself.
        raise RuntimeError("No finalist passed the practical selection limits.")
    winner = str(eligible.sort_values(
        ["raw_feature_count", "mean_churn_ap_skill", "mean_churn_recall_10pct", id_col],
        ascending=[True, False, False, True]).iloc[0][id_col])
    annotated["selected_by_rule"] = annotated[id_col].eq(winner)
    annotated = annotated.sort_values(["selected_by_rule", "eligible_performance_group", "raw_feature_count", id_col],
                                     ascending=[False, False, True, True]).reset_index(drop=True)
    reason = ("Selected the fewest raw features among already-retuned finalists passing explicit "
              f"mean AP-skill loss <= {limits['mean_ap_skill_loss']:g}, mean capture@10% loss <= "
              f"{limits['mean_recall_loss']:g}, worst-fold AP-skill loss <= {limits['max_fold_ap_skill_loss']:g}, "
              f"and worst-fold capture loss <= {limits['max_fold_recall_loss']:g}. "
              "Paired standard errors are diagnostics, not acceptance thresholds. "
              "The decision uses inner folds only and is not a formal equivalence claim.")
    return winner, annotated, reason
