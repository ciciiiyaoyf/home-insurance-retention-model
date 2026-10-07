"""Add reports to a completed retention run using saved models only.


Install beside report_plots.py in e_evaluate_and_report. Example:
python e_evaluate_and_report/make_reports.py --run-id RUN_ID


No acquisition, model fitting, feature selection, tuning, or final-test scoring.
Completed artifacts are read only. New reports go to a timestamped directory.
"""


from __future__ import annotations


import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import sys
from typing import Any, Mapping


import numpy as np
import pandas as pd




FAMILIES = ("logistic", "xgboost")




def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value




def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: observed={actual!r}; expected={expected!r}")




def verify_digest(path: Path, expected: Any, bundle_api: Any) -> None:
    digest = str(expected).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"Missing or invalid recorded SHA-256 for {path}")
    require_equal(bundle_api.sha256_file(path), digest, f"SHA-256 {path.name}")




def activate_profile(project_root: Path, info: Mapping[str, Any], run_id: str) -> Any:
    """Reuse the project's profile activation, without invoking either launcher."""
    os.chdir(project_root)
    from func.tenure_profile import profile_from_values
    from a_business_understanding import a0_one_click


    groups = info["profile"]["selected_tenure_groups"]
    if not groups:
        raise ValueError("Preparation has no selected tenure groups.")
    profile = profile_from_values(tenure_values=groups)
    # Exposure year is an optional profile dimension. Match the saved identity;
    # never guess based on a default profile or force the saved hash into params.
    for include_year in (False, True):
        params, _ = a0_one_click._activate_profile(
            profile,
            include_exposure_year_predictor=include_year,
            test_policy=None,
            sample_percent=None,
        )
        if str(params.PROFILE_HASH) == str(info["profile_hash"]):
            os.environ["RETENTION_PIPELINE_RUN_ID"] = run_id
            params = importlib.reload(params)
            require_equal(params.PROFILE_HASH, info["profile_hash"], "Runtime profile")
            require_equal(params.POPULATION_SCOPE_MODE, "FULL", "Population scope")
            return params
    raise ValueError(
        "Current project parameters cannot reproduce this saved FULL profile. "
        "Restore the run's matching parameters before producing reports."
    )




def verify_training_population(
    frame: pd.DataFrame,
    indices: np.ndarray,
    manifest: Mapping[str, Any],
    bundle_api: Any,
    *,
    cohort_col: str,
    target_check_col: str,
    label_cutoff: Any,
    holdout_start: Any,
) -> None:
    population = manifest["training_population"]
    columns = list(population["identity_columns"])
    if not columns or set(columns) - set(frame.columns):
        raise ValueError("Every saved training identity column must be present.")
    part = frame.loc[indices, list(dict.fromkeys([*columns, cohort_col, target_check_col]))]
    if not len(part) or not part.index.is_unique:
        raise ValueError("Training membership must be nonempty and unique.")
    if not pd.to_datetime(part[cohort_col], errors="raise").lt(holdout_start).all():
        raise ValueError("Final holdout entered training reconstruction.")
    if not pd.to_datetime(part[target_check_col], errors="raise").lt(label_cutoff).all():
        raise ValueError("Training reconstruction violates the strict label cutoff.")
    require_equal(len(part), int(population["rows"]), "Saved training rows")
    require_equal(
        bundle_api.population_sha256(part, columns),
        population["population_sha256"],
        "Saved training population identity",
    )




def selected_training_indices(
    frame: pd.DataFrame,
    base_indices: np.ndarray,
    manifest: Mapping[str, Any],
    modeling_utils: Any,
    *,
    cohort_col: str,
) -> np.ndarray:
    saved = manifest["source_metadata"]["history_application"]
    months = int(saved["requested_months"])
    if months not in (36, 48, 60):
        raise ValueError("Only the saved governed 36/48/60-month histories are supported.")
    indices, observed = modeling_utils.select_recent_mature_training_indices(
        frame, base_indices, cohort_col=cohort_col, months=months
    )
    for name in ("anchor_month", "cutoff_month_inclusive", "selected_rows", "requested_months"):
        require_equal(observed[name], saved[name], f"History {name}")
    if not observed["requested_months_fully_available"]:
        raise ValueError("The saved training window cannot be reconstructed completely.")
    return indices




def load_saved_bundle(
    run_root: Path,
    reference: Mapping[str, Any],
    canonical_dir: Path,
    *,
    family: str,
    run_id: str,
    profile_hash: str,
    input_sha: str,
    bundle_api: Any,
) -> tuple[Path, dict[str, Any]]:
    """Use the canonical run-local location, including after moving a project."""
    destination = canonical_dir.resolve()
    if not destination.is_relative_to(run_root.resolve()):
        raise ValueError("A model bundle must remain inside its source run.")
    name = PureWindowsPath(str(reference["manifest_path"])).name
    if not name or name in (".", ".."):
        raise ValueError("Invalid saved bundle manifest filename.")
    manifest_path = destination / name
    verify_digest(manifest_path, reference["sha256"], bundle_api)
    manifest, _ = bundle_api.load_bundle(destination, verify_hashes=True)
    require_equal(manifest, read_json(manifest_path), "Referenced bundle manifest")
    require_equal(manifest["model_family"], family, "Bundle family")
    metadata = manifest["source_metadata"]
    require_equal(metadata["source_run_id"], run_id, "Bundle source run")
    require_equal(metadata["profile_hash"], profile_hash, "Bundle profile")
    require_equal(metadata["source_data_identity"]["sha256"], input_sha, "Bundle input")
    return destination, manifest




def raw_scoring_columns(manifest: Mapping[str, Any]) -> list[str]:
    contracts = manifest["derived_feature_contracts"]
    derived = {
        value["derived_feature"]: value["source_feature"]
        for value in contracts.values()
        if isinstance(value, Mapping) and "derived_feature" in value and "source_feature" in value
    }
    columns = list(dict.fromkeys(derived.get(feature, feature) for feature in manifest["features"]))
    if manifest["target_column"] in columns:
        raise ValueError("Saved predictor contract contains the target.")
    return columns




def score_batches(
    frame: pd.DataFrame,
    indices: np.ndarray,
    destination: Path,
    manifest: Mapping[str, Any],
    bundle_api: Any,
    *,
    holdout_start: Any,
    cohort_col: str,
    batch_rows: int,
):
    """Reject held-out cohorts before preprocessing; pass predictors only."""
    indices = np.asarray(indices)
    dates = pd.to_datetime(frame.loc[indices, cohort_col], errors="raise")
    if dates.isna().any() or not dates.lt(pd.Timestamp(holdout_start)).all():
        raise ValueError("Final test rows cannot be scored by this reporting command.")
    del dates
    family = manifest["model_family"]
    columns = raw_scoring_columns(manifest)
    verified_manifest, fitted = bundle_api.load_bundle(destination, verify_hashes=True)
    require_equal(verified_manifest, manifest, "Scoring bundle identity")
    if family == "logistic":
        import joblib
        model = joblib.load(destination / manifest["model_file"])
        positions = {int(value): index for index, value in enumerate(model.classes_)}
        if 1 not in positions:
            raise ValueError("The saved Logistic model has no retention class 1.")
        models = [model]
        positive_column = positions[1]
    elif family == "xgboost":
        import xgboost
        models = []
        for relative in manifest["model_files"]:
            model = xgboost.XGBClassifier()
            model.load_model(destination / relative)
            models.append(model)
        if not models:
            raise ValueError("No saved XGBoost estimators found.")
        positive_column = 1
    else:
        raise ValueError("Unsupported saved model family.")
    # Copy only one predictor batch, never all 18M+ history rows at once.
    # Estimators and fitted preprocessing are loaded once per model bundle.
    for start in range(0, len(indices), batch_rows):
        stop = min(start + batch_rows, len(indices))
        batch_indices = indices[start:stop]
        matrix = bundle_api.transform_frame(fitted, frame.loc[batch_indices, columns])
        values = np.zeros(stop - start, dtype=float)
        for model in models:
            values += np.asarray(model.predict_proba(matrix)[:, positive_column], dtype=float)
        values /= len(models)
        if values.shape != (stop - start,) or not np.isfinite(values).all():
            raise ValueError("Invalid saved-model prediction array.")
        if ((values < 0) | (values > 1)).any():
            raise ValueError("Saved-model probabilities must be in [0, 1].")
        values = np.clip(values, 1e-7, 1.0 - 1e-7)
        yield batch_indices, values if family == "logistic" else 1.0 - values




def validated_outer_rows(
    predictions: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    family: str,
    fold: int,
    target_col: str,
    cohort_col: str,
    identity_columns: list[str],
    c4: Any,
) -> pd.DataFrame:
    if set(identity_columns) - set(validation.columns):
        raise ValueError("Missing outer row identity fields.")
    expected_ids = c4.stable_row_ids(validation, identity_columns).astype(str)
    part = predictions.loc[
        predictions["model_family"].eq(family) & predictions["fold"].eq(fold)
    ].copy()
    if part["row_id"].duplicated().any() or len(part) != len(validation):
        raise ValueError("Saved OOF rows are duplicate or incomplete.")
    require_equal(set(part["row_id"].astype(str)), set(expected_ids), "Outer row identities")
    part["row_id"] = part["row_id"].astype(str)
    part = part.set_index("row_id").loc[expected_ids.tolist()]
    target = pd.to_numeric(validation[target_col], errors="raise").to_numpy()
    if not np.array_equal(part["y_retention"].to_numpy(), target):
        raise ValueError("Saved OOF outcomes differ from the frozen population.")
    if not np.array_equal(
        pd.to_datetime(part["cohort_date"], errors="raise").to_numpy(),
        pd.to_datetime(validation[cohort_col], errors="raise").to_numpy(),
    ):
        raise ValueError("Saved OOF cohort dates differ from the frozen population.")
    return part




def monthly_partial(
    frame: pd.DataFrame,
    indices: np.ndarray,
    probability: np.ndarray,
    selected_indices: np.ndarray,
    *,
    cohort_col: str,
    target_col: str,
) -> pd.DataFrame:
    values = pd.to_numeric(frame.loc[indices, target_col], errors="raise").to_numpy()
    if not np.isin(values, [0, 1]).all():
        raise ValueError("Retention outcomes must be binary.")
    positions = np.searchsorted(selected_indices, indices)
    in_training = positions < len(selected_indices)
    in_training[in_training] = selected_indices[positions[in_training]] == indices[in_training]
    part = pd.DataFrame({
        "YEAR_MONTH": pd.to_datetime(frame.loc[indices, cohort_col]).dt.to_period("M").astype(str).to_numpy(),
        "policies": 1,
        "actual_sum": values,
        "predicted_sum": probability,
        "selected_history_training_rows": in_training.astype(int),
        "training_actual_sum": np.where(in_training, values, 0),
        "training_predicted_sum": np.where(in_training, probability, 0),
    })
    return part.groupby("YEAR_MONTH", sort=True).sum().reset_index()




def finalize_monthly(parts: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(parts, ignore_index=True).groupby("YEAR_MONTH", sort=True).sum()
    combined["actual_retention"] = combined["actual_sum"] / combined["policies"]
    combined["predicted_retention"] = combined["predicted_sum"] / combined["policies"]
    count = combined["selected_history_training_rows"].replace(0, np.nan)
    combined["training_actual_retention"] = combined["training_actual_sum"] / count
    combined["training_predicted_retention"] = combined["training_predicted_sum"] / count
    return combined.reset_index()




def run(args: argparse.Namespace) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from c_modeling import deployment_bundle as bundles
    from c_modeling import modeling_utils
    from c_modeling import c4_evaluate_and_finalize as c4
    from e_evaluate_and_report import report_plots as charts


    run_root = project_root / "output" / "pipeline_runs" / args.run_id
    outer_dir = run_root / "outer_evaluation"
    preparation = read_json(run_root / "preparation" / "preparation_info.json")
    outer = read_json(outer_dir / "outer_fold_bundle_manifest.json")
    status = read_json(outer_dir / "run_status.json")
    require_equal(status["status"], "COMPLETE", "C4 status")
    require_equal(outer["run_id"], args.run_id, "Outer run ID")
    require_equal(outer["profile_hash"], preparation["profile_hash"], "Outer profile")
    require_equal(int(outer["final_holdout_rows_scored"]), 0, "Sealed holdout")
    require_equal([int(fold["fold"]) for fold in outer["folds"]], [1, 2, 3, 4, 5], "Outer folds")
    final_path = run_root / "deployment" / "final_model_manifest.json"
    verify_digest(final_path, outer["final_model_manifest"]["sha256"], bundles)
    final = read_json(final_path)
    require_equal(final["run_id"], args.run_id, "Final run ID")
    require_equal(final["profile_hash"], outer["profile_hash"], "Final profile")
    require_equal(int(final["final_holdout_rows_scored"]), 0, "Final test seal")
    input_path = run_root / "input_snapshot" / "retention_13_1_population.parquet"
    input_sha = outer["source_data_identity"]["sha256"]
    require_equal(input_sha, preparation["input_sha256"], "Snapshot identity")
    print("Verifying the frozen snapshot; no training or test scoring will run.", flush=True)
    verify_digest(input_path, input_sha, bundles)
    params = activate_profile(project_root, preparation, args.run_id)
    settings = preparation["split_settings"]
    for name, key in (
        ("ROLLING_VALIDATION_FOLDS", "rolling_validation_folds"),
        ("OUTER_VALIDATION_MONTHS", "outer_validation_months"),
        ("FINAL_HOLDOUT_MONTHS", "final_holdout_months"),
        ("LABEL_AVAILABILITY_BUFFER_DAYS", "label_availability_buffer_days"),
    ):
        require_equal(getattr(params, name), settings[key], name)
    print("Reconstructing and checking the saved populations and folds.", flush=True)
    raw = modeling_utils.read_table(input_path)
    prepared, folds, holdout_idx, _, config = modeling_utils.load_project_population_and_folds(
        raw,
        rolling_validation_folds=int(settings["rolling_validation_folds"]),
        validation_months=int(settings["outer_validation_months"]),
        test_months=int(settings["final_holdout_months"]),
        min_train_months=int(settings["min_outer_train_months"]),
        strict_dataset_contract=True,
    )
    del raw
    definitions = modeling_utils.fold_definitions_frame(
        prepared, folds, cohort_col=config.cohort_date_col,
        target_check_col=config.target_check_col, target_col=config.target_col,
    )
    modeling_utils.verify_fold_definition_contract(
        definitions, run_root / "preparation" / "outer_fold_definitions.csv",
        label="Saved-model reporting",
    )
    holdout_start = pd.to_datetime(prepared.loc[holdout_idx, config.cohort_date_col]).min()
    require_equal(holdout_start, pd.Timestamp(final["excluded_holdout"]["cohort_start"]), "Holdout boundary")
    require_equal(len(holdout_idx), int(final["excluded_holdout"]["rows"]), "Holdout rows")
    identities = list(outer["identity_columns"])
    if not identities or set(identities) - set(prepared.columns):
        raise ValueError("All recorded row-identity columns are required.")
    require_equal(
        bundles.population_sha256(prepared.loc[holdout_idx, identities], identities),
        final["excluded_holdout"]["population_sha256"], "Holdout membership",
    )
    destination = run_root / "reporting" / (
        "saved_model_reports_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    destination.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {
        "contract": "retention_saved_model_reports_v1", "run_id": args.run_id,
        "profile_hash": outer["profile_hash"], "input_sha256": input_sha,
        "training_performed": False, "tuning_performed": False,
        "final_holdout_rows_scored": 0, "final_test_curve_available": False,
        "source_outer_manifest_sha256": bundles.sha256_file(outer_dir / "outer_fold_bundle_manifest.json"),
        "source_final_manifest_sha256": bundles.sha256_file(final_path),
        "train_sample_cap_per_fold": args.train_sample_rows,
        "training_membership_verified": [], "artifacts": [],
    }
    if args.only != "history":
        oof_path = outer_dir / "outer_oof_predictions.parquet"
        verify_digest(oof_path, outer["outer_oof_predictions"]["sha256"], bundles)
        predictions = pd.read_parquet(oof_path)
        require_equal(len(predictions), int(outer["outer_oof_predictions"]["rows"]), "OOF rows")
        require_equal(set(predictions["model_family"]), set(FAMILIES), "OOF families")
        if not pd.to_datetime(predictions["cohort_date"]).lt(holdout_start).all():
            raise ValueError("Saved validation predictions contain final test cohorts.")
        records = []
        for fold, saved_fold in zip(folds, outer["folds"]):
            number = int(fold["fold"])
            require_equal(number, int(saved_fold["fold"]), "Fold number")
            for family in FAMILIES:
                print(f"Scoring saved {family} fold {number} on its training sample.", flush=True)
                model_dir, manifest = load_saved_bundle(
                    run_root, saved_fold["families"][family],
                    outer_dir / "fold_models" / f"fold_{number:02d}" / family,
                    family=family, run_id=args.run_id, profile_hash=outer["profile_hash"],
                    input_sha=input_sha, bundle_api=bundles,
                )
                require_equal(manifest["source_metadata"]["bundle_role"], "OUTER_FOLD_EVALUATION", "Bundle role")
                require_equal(int(manifest["source_metadata"]["outer_fold"]), number, "Bundle fold")
                train_idx = selected_training_indices(
                    prepared, np.asarray(fold["train_idx"]), manifest, modeling_utils,
                    cohort_col=config.cohort_date_col,
                )
                verify_training_population(
                    prepared, train_idx, manifest, bundles,
                    cohort_col=config.cohort_date_col, target_check_col=config.target_check_col,
                    label_cutoff=fold["label_cutoff"], holdout_start=holdout_start,
                )
                sample_idx = train_idx
                if args.train_sample_rows and len(sample_idx) > args.train_sample_rows:
                    sample_idx = np.sort(np.random.default_rng(42 + number).choice(
                        sample_idx, args.train_sample_rows, replace=False
                    ))
                train_probability = np.concatenate([
                    probability for _, probability in score_batches(
                        prepared, sample_idx, model_dir, manifest, bundles,
                        holdout_start=holdout_start, cohort_col=config.cohort_date_col,
                        batch_rows=args.batch_rows,
                    )
                ])
                records.append({
                    "model_family": family, "fold": number, "split": "train",
                    "y_retention": prepared.loc[sample_idx, config.target_col].to_numpy(),
                    "p_retention": train_probability, "rows": len(sample_idx),
                    "training_population_rows": len(train_idx),
                })
                validation = prepared.loc[
                    np.asarray(fold["validation_idx"]),
                    list(dict.fromkeys([*identities, config.target_col, config.cohort_date_col])),
                ]
                valid = validated_outer_rows(
                    predictions, validation, family=family, fold=number,
                    target_col=config.target_col, cohort_col=config.cohort_date_col,
                    identity_columns=identities, c4=c4,
                )
                records.append({
                    "model_family": family, "fold": number, "split": "validation",
                    "y_retention": valid["y_retention"].to_numpy(),
                    "p_retention": valid["p_retention"].to_numpy(), "rows": len(valid),
                })
                receipt["training_membership_verified"].append({
                    "model_family": family, "fold": number, "training_rows": len(train_idx),
                    "sample_rows": len(sample_idx),
                    "population_sha256": manifest["training_population"]["population_sha256"],
                })
        receipt["split_reports"] = charts.write_split_curves(records, destination)
        del predictions, records
    if args.only != "curves":
        family = str(final["selected_model_family"])
        if family not in FAMILIES:
            raise ValueError("Unsupported final champion family.")
        model_dir, manifest = load_saved_bundle(
            run_root, final["selected_bundle"], run_root / "deployment" / "final_model" / family,
            family=family, run_id=args.run_id, profile_hash=outer["profile_hash"],
            input_sha=input_sha, bundle_api=bundles,
        )
        require_equal(manifest["source_metadata"]["bundle_role"], "FINAL_REFIT", "Final bundle role")
        base_idx, audit = c4.final_evaluation_training_indices(
            prepared, holdout_idx, cohort_col=config.cohort_date_col,
            target_check_col=config.target_check_col,
            label_buffer_days=int(settings["label_availability_buffer_days"]),
        )
        train_idx = selected_training_indices(
            prepared, base_idx, manifest, modeling_utils, cohort_col=config.cohort_date_col
        )
        train_idx = np.sort(train_idx)
        verify_training_population(
            prepared, train_idx, manifest, bundles,
            cohort_col=config.cohort_date_col, target_check_col=config.target_check_col,
            label_cutoff=audit["label_cutoff_exclusive"], holdout_start=holdout_start,
        )
        descriptive_idx = prepared.index[
            pd.to_datetime(prepared[config.cohort_date_col]).lt(holdout_start)
            & ~prepared.index.isin(holdout_idx)
        ].to_numpy()
        print(f"Scoring the saved final {family} model on {len(descriptive_idx):,} pre-test rows.", flush=True)
        parts = []
        for number, (indices, probability) in enumerate(score_batches(
            prepared, descriptive_idx, model_dir, manifest, bundles,
            holdout_start=holdout_start, cohort_col=config.cohort_date_col,
            batch_rows=args.batch_rows,
        ), start=1):
            parts.append(monthly_partial(
                prepared, indices, probability, train_idx,
                cohort_col=config.cohort_date_col, target_col=config.target_col,
            ))
            if number == 1 or number % 10 == 0:
                print(f"Full-history prediction batches completed: {number}", flush=True)
        monthly = finalize_monthly(parts)
        require_equal(int(monthly["policies"].sum()), len(descriptive_idx), "History coverage")
        require_equal(int(monthly["selected_history_training_rows"].sum()), len(train_idx), "Final training coverage")
        history = manifest["source_metadata"]["history_application"]
        receipt["history_reports"] = charts.write_history_trends(
            monthly, destination, family=family,
            training_start=history["cutoff_month_inclusive"], training_end=history["anchor_month"],
        )
        receipt["history_interpretation"] = "Frozen final-model backcasts and fitted values; NOT OOF performance"
        receipt["history_rows"] = len(descriptive_idx)
        receipt["history_months"] = len(monthly)
        receipt["selected_model_family"] = family
    receipt["status"] = "COMPLETE"
    receipt["artifacts"] = sorted(path.name for path in destination.iterdir() if path.is_file())
    (destination / "saved_model_reports_manifest.json").write_text(
        json.dumps(receipt, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"REPORTS COMPLETE: {destination}\nModel fits: 0\nFinal test rows scored: 0", flush=True)
    return destination




def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--only", choices=("both", "history", "curves"), default="both")
    parser.add_argument("--batch-rows", type=int, default=100_000)
    parser.add_argument("--train-sample-rows", type=int, default=100_000,
                        help="Training curve sample per fold; 0 means all training rows. Validation/history use all rows.")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}", args.run_id):
        parser.error("Invalid run ID.")
    if args.batch_rows < 1 or args.train_sample_rows < 0:
        parser.error("batch-rows must be positive and train-sample-rows nonnegative.")
    return args




if __name__ == "__main__":
    run(parse_args())
