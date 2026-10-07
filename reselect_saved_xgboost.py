"""Export a challenger from recorded, already-retuned inner-fold finalists.

This recovery utility was reconstructed from the retained selector's artifact
writers and selection-policy API. It never runs acquisition, fitting, tuning,
outer scoring, final-test scoring, or promotion of the original selected model.
Only the modern finalist-tuning contract (version 3) is supported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from c_modeling.xgboost_selection_policy import (
    FINAL_FEATURE_SELECTION_RULE, policy_sha256, select_practical_feature_group,
    validated_policy,
)

EXPORT_CONTRACT = "retention_saved_xgboost_reselection_v1"
FINALIST_VERSION = 3
METRICS_FILE = "xgb_finalist_tuned_group_confirmation_long.csv"
CONFIGURATIONS_FILE = "xgb_finalist_selected_configurations.json"
INVENTORY_FILE = "xgb_confirmation_group_inventory.csv"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} does not match the recorded contract.")


def checked_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact must be a regular file beneath {root}: {relative}")
    return path


def verify_sha(path: Path, expected: Any) -> None:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError(f"Missing valid recorded SHA-256 for {path.name}")
    require_equal(sha256(path), expected.lower(), f"SHA-256 for {path.name}")


def _sealed(payload: Mapping[str, Any], label: str, *, inner_only: bool = False) -> None:
    require_equal(payload.get("final_holdout_rows_scored"), 0, f"{label} final-test exclusion")
    if inner_only:
        require_equal(payload.get("outer_validation_rows_scored"), 0, f"{label} outer exclusion")


def load_saved_inputs(run_dir: Path) -> dict[str, Any]:
    """Read recorded selection artifacts, with no population or estimator load."""
    run = run_dir.expanduser().resolve(strict=True)
    relatives = ["pipeline_status.json", "preparation/preparation_info.json",
                 "xgboost/run_info.json", "xgboost/selected_features.json",
                 "xgboost/xgb_resolved_model_feature_types.json",
                 f"xgboost/{METRICS_FILE}", f"xgboost/{CONFIGURATIONS_FILE}",
                 f"xgboost/{INVENTORY_FILE}"]
    paths = {name: checked_file(run, name) for name in relatives}
    status = read_json(paths["pipeline_status.json"])
    prep = read_json(paths["preparation/preparation_info.json"])
    info = read_json(paths["xgboost/run_info.json"])
    selected = read_json(paths["xgboost/selected_features.json"])
    types = read_json(paths["xgboost/xgb_resolved_model_feature_types.json"])
    configs = read_json(paths[f"xgboost/{CONFIGURATIONS_FILE}"])
    require_equal(status.get("status"), "COMPLETE", "Pipeline completion")
    require_equal(status.get("run_id"), run.name, "Pipeline run identity")
    profile_hash = prep["profile_hash"]
    for label, payload in (("Pipeline", status), ("XGBoost", info), ("Finalists", configs)):
        require_equal(payload.get("profile_hash"), profile_hash, f"{label} profile")
    if "pipeline_run_id" in prep:
        require_equal(prep["pipeline_run_id"], run.name, "Preparation run identity")
    for label, payload in (("Preparation", prep), ("Selected recipe", selected),
                           ("XGBoost run", info), ("Finalist configurations", configs)):
        _sealed(payload, label, inner_only=label != "Preparation")
    require_equal(configs.get("contract_version"), FINALIST_VERSION, "Finalist contract version")
    require_equal(info.get("finalist_tuning_contract_version"), FINALIST_VERSION, "Run finalist version")
    require_equal(info["input"]["sha256"], prep["input_sha256"], "Selection input")
    verify_sha(paths["preparation/preparation_info.json"], info["preparation_info"]["sha256"])
    types_relative = "preparation/" + PureWindowsPath(str(info["feature_types_file"]["path"])).name
    paths[types_relative] = checked_file(run, types_relative)
    verify_sha(paths[types_relative], info["feature_types_file"]["sha256"])
    prepared_types = read_json(paths[types_relative])
    if any(value not in {"continuous", "discrete"} for value in types.values()):
        raise ValueError("Resolved XGBoost feature types must be continuous or discrete.")
    for name in set(types).intersection(prepared_types):
        require_equal(types[name], prepared_types[name], f"Prepared feature type {name}")
    for name in (METRICS_FILE, CONFIGURATIONS_FILE):
        recorded = info["finalist_artifacts"][name]
        verify_sha(paths[f"xgboost/{name}"], recorded["sha256"])
        verify_sha(paths[f"xgboost/{name}"], selected["finalist_tuning_policy"]["artifacts"][name]["sha256"])
    features = selected["selected_features"]
    if not isinstance(features, list) or not features or len(features) != len(set(features)):
        raise ValueError("Original selected features must be nonempty and unique.")
    seeds = selected["seeds"]
    if (not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds))
            or any(type(value) is not int or value < 0 for value in seeds)):
        raise ValueError("Recorded seeds must be unique nonnegative integers.")
    rare = info["arguments"]["rare_min_frequency"]
    if isinstance(rare, bool) or not isinstance(rare, (int, float)) or not math.isfinite(rare) or rare < 0:
        raise ValueError("Recorded rare-category frequency is invalid.")
    require_equal(selected["xgboost_version"], info["xgboost_version"], "Recorded XGBoost version")
    return {"run": run, "paths": paths, "prep": prep, "info": info,
            "selected": selected, "feature_types": types, "configurations": configs,
            "seeds": seeds, "rare_min_frequency": rare,
            "source_hashes": {path: sha256(path) for path in paths.values()}}


def build_recipe(saved: Mapping[str, Any], policy: Mapping[str, Any] | None = None):
    """Reapply the retained policy solely to saved ensemble inner-fold metrics."""
    from c_modeling.c4_evaluate_and_finalize import validated_history_months

    limits = validated_policy(policy)
    run = saved["run"]
    frame = pd.read_csv(saved["paths"][f"xgboost/{METRICS_FILE}"])
    inventory = pd.read_csv(saved["paths"][f"xgboost/{INVENTORY_FILE}"])
    needed = {"group_id", "raw_feature_count", "features"}
    if not needed.issubset(inventory) or inventory["group_id"].duplicated().any():
        raise ValueError("Invalid or duplicate confirmation feature inventory.")
    groups = {}
    for row in inventory.itertuples(index=False):
        values = [value.strip().upper() for value in str(row.features).split(";") if value.strip()]
        if not values or len(values) != len(set(values)) or int(row.raw_feature_count) != len(values):
            raise ValueError("Invalid feature order/count in confirmation inventory.")
        groups[str(row.group_id)] = values
    configs = saved["configurations"]
    config_by_group = configs["configurations"]
    finalists = configs["finalist_group_ids"]
    if len(finalists) != len(set(finalists)) or set(config_by_group) != set(finalists):
        raise ValueError("Finalist configuration identities disagree.")
    require_equal(set(frame["group_id"].astype(str)), set(finalists), "Complete finalist metric grid")
    require_equal(set(saved["info"]["finalist_group_ids"]), set(finalists), "Recorded finalist universe")
    required_columns = {"seed_count", "seeds", "selected_candidate_id", "selected_history_id",
                        "selected_model_id", "ordered_feature_sha256", "best_rounds"}
    if not required_columns.issubset(frame):
        raise ValueError(f"Missing frozen finalist fields: {sorted(required_columns-set(frame))}")
    for group_id in finalists:
        config = config_by_group[group_id]
        validated_history_months(config["selected_history"], family="saved finalist XGBoost")
        features = groups[group_id]
        feature_hash = hashlib.sha256(("\n".join(features)+"\n").encode()).hexdigest()
        require_equal(config["group_id"], group_id, "Configuration group")
        require_equal(config["raw_feature_count"], len(features), "Configuration feature count")
        require_equal(config["ordered_feature_sha256"], feature_hash, "Configuration feature order")
        part = frame.loc[frame["group_id"].astype(str).eq(group_id)]
        expected = {"raw_feature_count": len(features), "seed_count": len(saved["seeds"]),
                    "seeds": ";".join(str(seed) for seed in saved["seeds"]),
                    "selected_candidate_id": config["selected_candidate_id"],
                    "selected_history_id": config["selected_history"]["history_id"],
                    "selected_model_id": config["selected_model_spec"]["model_id"],
                    "ordered_feature_sha256": feature_hash}
        for column, value in expected.items():
            if not part[column].astype(str).eq(str(value)).all():
                raise ValueError(f"Finalist {group_id} has inconsistent {column}.")
        rounds = pd.to_numeric(part["best_rounds"], errors="raise")
        if not np.isfinite(rounds).all() or (rounds <= 0).any():
            raise ValueError("Inner-confirmation boosting rounds must be positive and finite.")
        missing = set(features) - set(saved["feature_types"])
        if missing:
            raise ValueError(f"Missing recorded feature types: {sorted(missing)}")
        if not set(saved["selected"].get("mandatory_control_features", [])).issubset(features):
            raise ValueError("Finalist lost a mandatory calendar control.")
    winner, summary, reason = select_practical_feature_group(frame, policy=limits)
    config = config_by_group[winner]
    features = groups[winner]
    rows = frame.loc[frame["group_id"].astype(str).eq(winner)]
    rounds = int(round(float(pd.to_numeric(rows["best_rounds"]).median())))
    if rounds < 1:
        raise ValueError("Recommended boosting rounds must be positive.")
    baseline = saved["selected"]
    # Establish the original model as a genuine member of the retained grid.
    original_id = baseline["selected_group_id"]
    original_config = config_by_group[original_id]
    require_equal(groups[original_id], baseline["selected_features"], "Original feature inventory")
    for key, config_key in (("selected_configuration_id", "selected_candidate_id"),
                            ("selected_history", "selected_history"),
                            ("selected_model_spec", "selected_model_spec")):
        require_equal(baseline[key], original_config[config_key], f"Original {key}")
    original_rounds = int(round(float(pd.to_numeric(
        frame.loc[frame["group_id"].astype(str).eq(original_id), "best_rounds"]).median())))
    require_equal(original_rounds, baseline["recommended_boosting_rounds_from_inner_confirmation"],
                  "Original inner-confirmation rounds")
    payload = {"contract": EXPORT_CONTRACT, "source_run_id": run.name,
               "profile_hash": saved["prep"]["profile_hash"],
               "source_input": {"filename": PureWindowsPath(str(saved["prep"]["input"])).name,
                                "sha256": saved["prep"]["input_sha256"]},
               "selected_group_id": winner, "selected_features": features,
               "selected_configuration_id": config["selected_candidate_id"],
               "selected_history": config["selected_history"],
               "selected_model_spec": config["selected_model_spec"],
               "recommended_boosting_rounds_from_inner_confirmation": rounds,
               "feature_types": {name: saved["feature_types"][name] for name in features},
               "seeds": saved["seeds"], "rare_min_frequency": saved["rare_min_frequency"],
               "xgboost_version": baseline["xgboost_version"],
               "selection_rule": FINAL_FEATURE_SELECTION_RULE, "selection_policy": limits,
               "selection_policy_sha256": policy_sha256(limits), "selection_reason": reason,
               "selection_uses_inner_folds_only": True, "outer_validation_rows_scored": 0,
               "final_holdout_rows_scored": 0, "original_model_modified": False,
               "source_artifacts": {str(path.relative_to(run)): digest
                                    for path, digest in saved["source_hashes"].items()}}
    return payload, summary


def export_reselection(run_dir: Path, *, policy=None, output_dir: Path | None = None) -> Path:
    saved = load_saved_inputs(run_dir)
    recipe, summary = build_recipe(saved, policy)
    run = saved["run"]
    export_root = run / "xgboost_reselections"
    identity = hashlib.sha256(json.dumps(recipe, sort_keys=True, allow_nan=False).encode()).hexdigest()[:12]
    destination = ((output_dir.expanduser().resolve() if output_dir is not None else
                    export_root / f"{recipe['selected_group_id']}_{identity}")).resolve()
    if destination == export_root.resolve() or not destination.is_relative_to(export_root.resolve()):
        raise ValueError("Exports must be children of this run's xgboost_reselections directory.")
    for path, digest in saved["source_hashes"].items():
        verify_sha(path, digest)
    destination.mkdir(parents=True, exist_ok=False)
    recipe_path = destination / "recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    summary.to_csv(destination / "selection_summary.csv", index=False)
    receipt = {"contract": EXPORT_CONTRACT, "status": "COMPLETE", "source_run_id": run.name,
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "output_sha256": {name: sha256(destination / name)
                                  for name in ("recipe.json", "selection_summary.csv")},
               "final_holdout_rows_scored": 0, "outer_validation_rows_scored": 0}
    # Recheck source identity immediately before marking an export complete.
    for path, digest in saved["source_hashes"].items():
        verify_sha(path, digest)
    (destination / "run_status.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return recipe_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--policy-json", type=Path, help="Optional JSON object of explicit practical selection limits.")
    parser.add_argument("--output-dir", type=Path, help="New child of RUN/xgboost_reselections; never overwritten.")
    args = parser.parse_args(argv)
    policy = read_json(args.policy_json) if args.policy_json else None
    path = export_reselection(args.run_dir, policy=policy, output_dir=args.output_dir)
    print(f"Saved inner-fold challenger recipe: {path}")
    print("Original models unchanged. Models fitted: 0. Final-test rows scored: 0.")


if __name__ == "__main__":
    main()
