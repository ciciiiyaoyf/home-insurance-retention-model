"""Validate completed selections for C4 without rerunning their searches.


Search-code/plotting edits do not invalidate an already chosen recipe. Data,
folds, preprocessing, routing, recipe agreement and the resolved XGBoost
finalist evidence still have to match. This contract authorizes C4 only; it
must never be used to resume an unfinished feature-selection search.
"""
from __future__ import annotations


import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any




class CompletedSelectionError(RuntimeError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__(
            "Completed-selection validation failed; no selection was rerun or archived:\n"
            + "\n".join(f"- {error}" for error in report["errors"])
        )




def has_completed_selection_receipts(run_root: Path) -> bool:
    """A published result must be checked, not silently archived as a retry."""
    return (run_root / "xgboost/run_info.json").exists()




def validate_completed_selections(*, run_root: Path, logistic: Any, xgboost: Any) -> dict[str, Any]:
    report: dict[str, Any] = {
        "contract": "completed_selections_for_c4_v1", "run_id": run_root.name,
        "status": "CHECKING", "errors": [], "source_differences": [],
        "execution_differences": [], "validated_sha256": {},
        "stages_authorized": ["c4_evaluate_and_finalize.py"],
        "selection_rerun": False, "final_holdout_rows_scored": 0,
    }
    errors = report["errors"]
    cache: dict[Path, str] = {}


    def check(ok: bool, label: str) -> None:
        if not ok:
            errors.append(label)


    def finish_if_invalid() -> None:
        if errors:
            report["status"] = "BLOCKED"
            raise CompletedSelectionError(report)


    def safe_file(path: Path) -> bool:
        # Also reject directory symlinks, not just a linked final filename.
        return (not any(p.is_symlink() for p in (path, *path.parents))
                and path.is_file() and path.stat().st_size > 0)


    def digest(path: Path) -> str:
        if path not in cache:
            if not safe_file(path):
                raise ValueError(f"Missing, empty or linked file: {path}")
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    h.update(block)
            cache[path] = h.hexdigest()
            report["validated_sha256"][str(path)] = cache[path]
        return cache[path]


    def read(path: Path) -> dict[str, Any]:
        try:
            digest(path)
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(value, dict):
                raise ValueError("expected a JSON object")
            return value
        except (OSError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
            return {}


    def verify_hash(info: dict[str, Any], key: str, path: Path, *, family: str,
                    search_only: bool = False) -> None:
        receipt = info.get(key)
        expected = receipt.get("sha256") if isinstance(receipt, dict) else None
        try:
            current = digest(path)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            return
        if expected != current:
            if search_only and isinstance(expected, str) and len(expected) == 64:
                report["source_differences"].append({
                    "family": family, "field": key, "path": str(path),
                    "saved_sha256": expected, "current_sha256": current,
                    "reason": "Completed recipe retained; selector/diagnostic code is not rerun in C4.",
                })
            else:
                errors.append(f"{family}.{key}: SHA-256 mismatch for {path}; saved={expected}; current={current}")


    for family, runner in (("logistic", logistic), ("xgboost", xgboost)):
        check(Path(runner.OUTPUT_DIR).resolve() == (run_root / family).resolve(),
              f"{family}: runner points to a different run directory")
        try:
            runner._verify_preparation_profile()
            runner._read_feature_types(runner._read_pool())
        except Exception as exc:
            errors.append(f"{family}: preparation/routing validation failed: {exc}")
    lr_dir, xgb_dir = run_root / "logistic", run_root / "xgboost"
    lr_info, xgb_info = read(lr_dir / "run_info.json"), read(xgb_dir / "run_info.json")
    lr_recipe, xgb_recipe = read(lr_dir / "recommended_model.json"), read(xgb_dir / "selected_features.json")
    lr_status = read(lr_dir / "run_status.json")
    history = read(lr_dir / "history_selection_decision.json")
    checkpoint_path = xgb_dir / xgboost.STAGE4_SELECTION_CHECKPOINT_NAME
    checkpoint = read(checkpoint_path)
    finish_if_invalid()
    for family, info, recipe, runner in (
        ("logistic", lr_info, lr_recipe, logistic),
        ("xgboost", xgb_info, xgb_recipe, xgboost),
    ):
        check(info.get("profile_hash") == runner.PROFILE_HASH and info.get("profile") == runner.PROFILE_PAYLOAD,
              f"{family}: selected result belongs to a different profile")
        for label, payload in (("run_info", info), ("recipe", recipe)):
            check(payload.get("final_holdout_rows_scored") == 0, f"{family}.{label}: holdout seal is not proven")
        for key, attr in (
            ("data" if family == "logistic" else "input", "DATA"),
            ("manifest", "MANIFEST"), ("preparation_info", "PREPARATION_INFO"),
            ("outer_fold_definitions", "OUTER_FOLD_DEFINITIONS"),
            ("inner_fold_definitions", "INNER_FOLD_DEFINITIONS"),
            ("feature_types_file", "FEATURE_TYPES"), ("routing_audit", "ROUTING_AUDIT"),
            ("pool_file" if family == "logistic" else "starting_feature_pool",
             "POOL" if family == "logistic" else "STARTING_POOL"),
            ("modeling_utils", "MODELING_UTILS"), ("feature_routing", "FEATURE_ROUTING"),
        ):
            verify_hash(info, key, Path(getattr(runner, attr)), family=family)
        if family == "logistic":
            verify_hash(info, "state_grouping", Path(runner.STATE_GROUPING), family=family)
            verify_hash(info, "logistic_utils", Path(runner.SUPPORT), family=family, search_only=True)
        for key, attr in (("script", "SCRIPT"), ("model_diagnostics", "MODEL_DIAGNOSTICS")):
            verify_hash(info, key, Path(getattr(runner, attr)), family=family, search_only=True)
        feature_key = "features" if family == "logistic" else "selected_features"
        features = recipe.get(feature_key)
        check(isinstance(features, list) and bool(features)
              and all(isinstance(f, str) and f.strip() == f and f.upper() == f for f in features),
              f"{family}: selected feature list is invalid")
        if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
            continue
        check(len(features) == len(set(features)), f"{family}: duplicate selected features")
        features_path = lr_dir / "recommended_features.txt" if family == "logistic" else xgb_dir / "selected_features.txt"
        try:
            digest(features_path)
            lines = [line.strip().upper() for line in features_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            check(lines == features, f"{family}: selected JSON and text feature lists disagree")
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
        types_path = lr_dir / "resolved_logistic_model_feature_types.json" if family == "logistic" else xgb_dir / "xgb_resolved_model_feature_types.json"
        feature_types = read(types_path)
        check(all(feature_types.get(f) in {"continuous", "discrete"} for f in features),
              f"{family}: selected features have missing/invalid resolved types")
        prepared_types = read(Path(runner.FEATURE_TYPES))
        check(all(feature_types[f] == prepared_types[f] for f in features if f in prepared_types and f in feature_types),
              f"{family}: selected feature routing differs from preparation")
        check(recipe.get("selected_history") == info.get("selected_history"),
              f"{family}: selected history differs between recipe and receipt")


    check(lr_status.get("status") == lr_recipe.get("status") == "COMPLETE", "logistic: selection is not COMPLETE")
    check(lr_status.get("final_holdout_rows_scored") == 0, "logistic: status does not prove a sealed holdout")
    for field in ("features", "C"):
        check(lr_recipe.get(field) == lr_info.get(field), f"logistic: recipe {field} differs from run_info")
    check(history.get("selected_history") == lr_status.get("selected_history") == lr_info.get("selected_history"),
          "logistic: history decision/status differs from recipe")
    check(lr_info.get("sklearn_version") == logistic.sklearn.__version__,
          "logistic: installed scikit-learn differs from the completed selection")
    check(xgb_info.get("outer_validation_rows_scored") == xgb_recipe.get("outer_validation_rows_scored") == 0,
          "xgboost: selection used outer-validation rows")
    try:
        resolved = xgboost._read_resolved_stage4_checkpoint()
        check(resolved is not None and xgboost._selected_finalist_matches_checkpoint(resolved),
              "xgboost: resolved finalist evidence/current selection policy or sole winner marker does not match")
    except Exception as exc:
        errors.append(f"xgboost: finalist evidence cannot be verified: {exc}")
    verify_hash(xgb_info, "stage4_selection_checkpoint", checkpoint_path, family="xgboost")
    for field in ("selected_group_id", "selected_history", "selected_model_spec", "selected_raw_feature_count"):
        check(xgb_recipe.get(field) == xgb_info.get(field), f"xgboost: recipe {field} differs from run_info")
    for field in ("selected_history", "selected_model_spec"):
        check(xgb_recipe.get(field) == checkpoint.get("selected_configuration", {}).get(field),
              f"xgboost: recipe {field} differs from the resolved finalist")
    check(xgb_recipe.get("selected_group_id") == checkpoint.get("selected_group_id"),
          "xgboost: selected group differs from checkpoint")
    features = xgb_recipe.get("selected_features", [])
    if isinstance(features, list) and all(isinstance(f, str) for f in features):
        feature_hash = hashlib.sha256(("\n".join(features) + "\n").encode("utf-8")).hexdigest()
        check(feature_hash == checkpoint.get("selected_feature_sha256"), "xgboost: selected feature sequence differs from checkpoint")
        check(len(features) == checkpoint.get("selected_raw_feature_count") == xgb_recipe.get("selected_raw_feature_count"),
              "xgboost: selected feature counts disagree")
    check(xgb_recipe.get("seeds") == checkpoint.get("seeds"), "xgboost: seeds differ from resolved checkpoint")
    for name, receipt in checkpoint.get("bound_artifacts", {}).items():
        if Path(name).name != name or "/" in name or "\\" in name:
            errors.append("xgboost: unsafe bound-artifact path")
            continue
        verify_hash({name: receipt}, name, xgb_dir / name, family="xgboost")
    # The final recipe's number of trees is derived from this signed evidence.
    try:
        with (xgb_dir / "xgb_finalist_tuned_group_confirmation_long.csv").open(encoding="utf-8-sig", newline="") as stream:
            rounds = [float(row["best_rounds"]) for row in csv.DictReader(stream)
                      if row["group_id"] == xgb_recipe["selected_group_id"]]
        check(bool(rounds) and all(math.isfinite(v) and v >= 1 for v in rounds)
              and int(round(median(rounds))) == xgb_recipe.get("recommended_boosting_rounds_from_inner_confirmation"),
              "xgboost: boosting rounds do not match completed inner confirmation")
    except (OSError, KeyError, ValueError) as exc:
        errors.append(f"xgboost: cannot verify boosting rounds: {exc}")
    for key, current in xgboost._expected_arguments().items():
        saved = xgb_info.get("arguments", {}).get(key)
        if saved != current:
            if key in {"device", "n_jobs"}:
                report["execution_differences"].append({"field": key, "saved": saved, "requested": current})
            else:
                errors.append(f"xgboost.arguments.{key}: saved={saved!r}, current={current!r}")
    finish_if_invalid()
    report.update(status="VALIDATED", selected_group_id=xgb_recipe["selected_group_id"],
                  selected_feature_count=len(features), final_holdout_rows_scored=0)
    return report
