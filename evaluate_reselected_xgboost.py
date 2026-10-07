#!/usr/bin/env python
"""Fit a saved XGBoost challenger on existing outer folds and compare predictions.


Run from the project root. No acquisition, feature search, Optuna, final refit,
family promotion, or final-test scoring occurs. Completed folds are reusable
only when the data, recipe, source files and execution settings still match.
"""
from __future__ import annotations


import argparse
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import re
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


import numpy as np
import pandas as pd




CONTRACT = "retention_reselected_xgboost_outer_comparison_v1"




def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: observed={actual!r}; expected={expected!r}")




def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()




def verify_file(path: Path, expected: Any) -> str:
    expected = str(expected).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"No valid recorded SHA-256 for {path}")
    require_equal(digest(path), expected, f"SHA-256 {path.name}")
    return expected




def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload




class ComparisonLock(AbstractContextManager):
    """OS-owned lock: automatically released even after Ctrl+C or process death."""


    def __init__(self, path: Path):
        self.path = path
        self.handle = None


    def __enter__(self):
        self.handle = self.path.open("a+b")
        self.handle.seek(0, 2)
        if self.handle.tell() == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError("This comparison is already running in another process.") from exc
        return self


    def __exit__(self, *args):
        if self.handle is not None:
            try:
                self.handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            finally:
                self.handle.close()
        return False




def make_recipe(payload: Mapping[str, Any], c4: Any) -> Any:
    history = dict(payload["selected_history"])
    months = c4.validated_history_months(history, family="challenger XGBoost")
    rounds = payload["recommended_boosting_rounds_from_inner_confirmation"]
    if isinstance(rounds, bool) or int(rounds) != rounds or int(rounds) < 1:
        raise ValueError("Recorded boosting rounds must be a positive integer.")
    features = tuple(payload["selected_features"])
    seeds = tuple(payload["seeds"])
    if not features or len(features) != len(set(features)):
        raise ValueError("Recorded features must be nonempty and unique.")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Recorded seeds must be nonempty and unique.")
    return c4.FamilyRecipe(
        family="xgboost", features=features,
        feature_types=dict(payload["feature_types"]), history=history,
        history_months=months, rare_min_frequency=payload["rare_min_frequency"],
        model_recipe={
            **dict(payload["selected_model_spec"]), "selected_history": history,
            "boosting_rounds": int(rounds), "early_stopping_rounds": None,
            "outer_validation_used_for_early_stopping": False,
            "selected_configuration_id": payload["selected_configuration_id"],
        }, seeds=seeds, boosting_rounds=int(rounds),
    )




def validate_fold_membership(frame, fold, holdout_idx, *, config, buffer_days):
    train = np.asarray(fold["train_idx"])
    valid = np.asarray(fold["validation_idx"])
    if not len(train) or not len(valid) or not frame.index.is_unique:
        raise ValueError("Nonempty folds and unique population indices are required.")
    for name, values in (("training", train), ("validation", valid)):
        if len(np.unique(values)) != len(values):
            raise ValueError(f"Duplicate {name} fold indices.")
        if np.intersect1d(values, holdout_idx).size:
            raise ValueError(f"Final holdout entered outer {name}.")
    if np.intersect1d(train, valid).size:
        raise ValueError("Outer training and validation overlap.")
    cutoff = pd.Timestamp(fold["label_cutoff"])
    available = pd.to_datetime(frame.loc[train, config.target_check_col], errors="raise")
    available = available + pd.to_timedelta(int(buffer_days), unit="D")
    if not available.lt(cutoff).all():
        raise ValueError("Outer training violates the strict label-availability cutoff.")
    train_months = pd.to_datetime(frame.loc[train, config.cohort_date_col]).dt.to_period("M")
    valid_dates = pd.to_datetime(frame.loc[valid, config.cohort_date_col])
    if not valid_dates.ge(cutoff).all() or set(train_months) & set(valid_dates.dt.to_period("M")):
        raise ValueError("Outer cohort chronology is invalid.")
    return train, valid




def aligned_baseline(oof, validation, *, fold_number, config, identities, c4):
    """Return original predictions in precisely the reconstructed validation order."""
    if not identities or set(identities) - set(validation.columns):
        raise ValueError("Missing governed identity fields.")
    row_ids = c4.stable_row_ids(validation, identities).astype(str)
    expected = pd.DataFrame({
        "row_id": row_ids.to_numpy(), "fold": int(fold_number),
        "cohort_date": pd.to_datetime(validation[config.cohort_date_col]).to_numpy(),
        "tenure_group": validation[config.tenure_col].astype(str).to_numpy(),
        "y_retention": pd.to_numeric(validation[config.target_col], errors="raise").astype(int).to_numpy(),
    })
    for col in identities:
        if col not in expected:
            expected[col] = validation[col].to_numpy()
    observed = oof.loc[oof["fold"].eq(fold_number)].copy()
    if observed["row_id"].duplicated().any() or len(observed) != len(expected):
        raise ValueError(f"Baseline fold {fold_number} has incomplete or duplicate rows.")
    if set(observed["row_id"]) != set(expected["row_id"]):
        raise ValueError(f"Baseline fold {fold_number} row identities differ.")
    observed = observed.set_index("row_id").loc[row_ids.tolist()].reset_index()
    for col in expected:
        if col not in observed:
            raise ValueError(f"Baseline is missing alignment field {col}.")
        a, b = observed[col], expected[col]
        if col == "cohort_date" or pd.api.types.is_datetime64_any_dtype(b):
            equal = np.array_equal(pd.to_datetime(a).to_numpy(), pd.to_datetime(b).to_numpy())
        elif col in ("fold", "y_retention"):
            equal = np.array_equal(pd.to_numeric(a).to_numpy(), pd.to_numeric(b).to_numpy())
        else:
            equal = a.astype("string").fillna("__MISSING__").equals(
                b.astype("string").fillna("__MISSING__"))
        if not equal:
            raise ValueError(f"Baseline fold {fold_number} {col} differs from the saved population.")
    p = pd.to_numeric(observed["p_retention"], errors="raise").to_numpy()
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Baseline probabilities must be finite and within [0, 1].")
    expected["model_family"] = "xgboost"
    expected["p_retention"] = p
    return expected




def baseline_bundle(run, reference, family, number, *, outer, bundles):
    directory = run / "outer_evaluation" / "fold_models" / f"fold_{number:02d}" / family
    filename = PureWindowsPath(str(reference["manifest_path"])).name
    path = directory / filename
    verify_file(path, reference["sha256"])
    manifest = read_json(path)
    require_equal(manifest["model_family"], family, "Baseline family")
    metadata = manifest["source_metadata"]
    require_equal(metadata["source_run_id"], run.name, "Baseline run")
    require_equal(metadata["profile_hash"], outer["profile_hash"], "Baseline profile")
    require_equal(metadata["outer_fold"], number, "Baseline fold")
    require_equal(metadata["source_data_identity"]["sha256"],
                  outer["source_data_identity"]["sha256"], "Baseline input")
    require_equal(metadata["bundle_role"], "OUTER_FOLD_EVALUATION", "Baseline bundle role")
    if family == "xgboost":
        loaded, fitted = bundles.load_bundle(directory, verify_hashes=True)
        require_equal(loaded, manifest, "Referenced baseline manifest")
    else:
        fitted = None
    return directory, manifest, fitted, path




def reconstruct_history(prepared, base_idx, manifest, mu, bundles, *, config):
    saved = manifest["source_metadata"]["history_application"]
    indices, audit = mu.select_recent_mature_training_indices(
        prepared, base_idx, cohort_col=config.cohort_date_col,
        months=int(saved["requested_months"]))
    for key in ("anchor_month", "cutoff_month_inclusive", "selected_rows", "requested_months"):
        require_equal(audit[key], saved[key], f"Baseline history {key}")
    if not audit["requested_months_fully_available"]:
        raise ValueError("Incomplete baseline training history.")
    population = manifest["training_population"]
    require_equal(len(indices), int(population["rows"]), "Baseline training rows")
    cols = list(population["identity_columns"])
    require_equal(bundles.population_sha256(prepared.loc[indices, cols], cols),
                  population["population_sha256"], "Baseline training membership")
    return indices




def raw_predictors(manifest):
    derived = {value["derived_feature"]: value["source_feature"]
               for value in manifest["derived_feature_contracts"].values()
               if isinstance(value, Mapping) and "source_feature" in value and "derived_feature" in value}
    columns = list(dict.fromkeys(derived.get(name, name) for name in manifest["features"]))
    if manifest["target_column"] in columns:
        raise ValueError("A predictor dependency contains the target.")
    return columns




def verify_fitted_recipe(manifest, fitted, recipe, expected_contract):
    """Require the same complete frozen recipe for both fresh and resumed fits."""
    require_equal(manifest["model_family"], "xgboost", "Fitted challenger family")
    require_equal(manifest["source_metadata"]["comparison_fold_contract"],
                  expected_contract, "Published model resume contract")
    require_equal(manifest["features"], list(recipe.features), "Fitted challenger features")
    require_equal(manifest["seeds"], list(recipe.seeds), "Fitted challenger seeds")
    require_equal(manifest["boosting_rounds"], recipe.boosting_rounds, "Fitted challenger rounds")
    require_equal(manifest["model_recipe"], recipe.model_recipe, "Fitted challenger model recipe")
    require_equal(fitted["feature_types"], recipe.feature_types, "Fitted challenger feature types")
    require_equal(fitted["rare_min_frequency"], recipe.rare_min_frequency, "Fitted rare category setting")
    require_equal(manifest["ensemble_aggregation"], "arithmetic_mean_churn_probability",
                  "Fitted ensemble aggregation")




def validate_cached_prediction(part, baseline):
    """A resumed prediction must retain every saved validation-row field."""
    if set(part.columns) != set(baseline.columns):
        raise ValueError("Resumed prediction columns differ from the baseline contract.")
    observed = part.loc[:, baseline.columns].reset_index(drop=True)
    expected = baseline.reset_index(drop=True)
    if not observed.drop(columns="p_retention").equals(expected.drop(columns="p_retention")):
        raise ValueError("Resumed fold identities, labels or validation fields differ.")
    probability = pd.to_numeric(observed["p_retention"], errors="raise").to_numpy(float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Resumed probabilities must be finite and within [0, 1].")
    return observed




def fit_or_score_fold(directory, expected_contract, baseline, training, validation, *,
                      recipe, config, identities, mu, bundles, c4, state, metadata,
                      source_code_paths, runtime, batch_rows):
    """Resume without fitting; materialize a callable training input only if needed."""
    directory.mkdir(parents=True, exist_ok=True)
    contract_path = directory / "fold_contract.json"
    if contract_path.exists():
        require_equal(read_json(contract_path), expected_contract, "Fold resume contract")
    else:
        if any(directory.iterdir()):
            raise ValueError("Refusing to adopt a fold directory without its contract.")
        c4._atomic_json(contract_path, expected_contract)
    receipt_path = directory / "receipt.json"
    prediction_path = directory / "predictions.parquet"
    destination = directory / "model"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        require_equal(receipt["contract"], expected_contract, "Completed fold contract")
        require_equal(receipt["status"], "COMPLETE", "Fold status")
        verify_file(prediction_path, receipt["prediction_sha256"])
        manifest_path = destination / receipt["manifest_filename"]
        verify_file(manifest_path, receipt["manifest_sha256"])
        manifest, fitted = bundles.load_bundle(destination, verify_hashes=True)
        require_equal(manifest, read_json(manifest_path), "Resumed referenced manifest")
        verify_fitted_recipe(manifest, fitted, recipe, expected_contract)
        part = validate_cached_prediction(pd.read_parquet(prediction_path), baseline)
        require_equal(int(receipt["rows"]), len(part), "Resumed row count")
        require_equal(receipt["final_holdout_rows_scored"], 0, "Resumed test exclusion")
        print(f"  Reused completed fold {expected_contract['fold']}", flush=True)
        return part, receipt


    if destination.exists():
        # A valid published model may survive an interruption before prediction.
        manifest, fitted = bundles.load_bundle(destination, verify_hashes=True)
        manifest_path = c4._find_bundle_manifest(destination)
        print("  Reusing the published fold model; finishing validation scoring", flush=True)
    else:
        print(f"  Fitting {len(recipe.seeds)} seed models, {recipe.boosting_rounds} rounds each", flush=True)
        training_frame = training() if callable(training) else training
        manifest, manifest_path = c4._export_family_bundle(
            family="xgboost", destination=destination, training_frame=training_frame,
            recipe=recipe, target_col=config.target_col, cohort_col=config.cohort_date_col,
            identity_columns=identities, modeling_utils=mu, deployment_bundle=bundles,
            shared_state_artifact=state,
            source_metadata={**metadata, "comparison_fold_contract": expected_contract},
            source_code_paths=source_code_paths, params=runtime)
        del training_frame
        loaded, fitted = bundles.load_bundle(destination, verify_hashes=True)
        require_equal(loaded, manifest, "Newly published challenger manifest")


    verify_fitted_recipe(manifest, fitted, recipe, expected_contract)
    predictors = validation.loc[:, raw_predictors(manifest)]
    probability = np.empty(len(validation), dtype=float)
    next_start = 0
    for start, stop, churn in bundles.iter_xgboost_bundle_scores(
            destination, predictors, batch_size=batch_rows, verify_hashes=True):
        if start != next_start or stop <= start or stop > len(validation):
            raise ValueError("Invalid validation scoring batch boundaries.")
        values = np.asarray(churn, dtype=float)
        if values.shape != (stop-start,) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Invalid challenger churn probabilities.")
        probability[start:stop] = 1.0 - values
        next_start = stop
    require_equal(next_start, len(validation), "Complete validation scoring")
    part = baseline.copy()
    part["p_retention"] = np.clip(probability, 1e-7, 1.0-1e-7)
    c4._atomic_parquet(prediction_path, part)
    receipt = {"status": "COMPLETE", "contract": expected_contract,
               "prediction_sha256": digest(prediction_path),
               "manifest_filename": manifest_path.name, "manifest_sha256": digest(manifest_path),
               "rows": len(part), "final_holdout_rows_scored": 0}
    c4._atomic_json(receipt_path, receipt)
    return part, receipt




def _positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number




def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Existing completed pipeline run.")
    parser.add_argument("--recipe", type=Path, help="Exported recipe.json; inferred only if one completed export exists.")
    parser.add_argument("--output-dir", type=Path, help="Optional new directory under the run's xgboost_outer_comparisons/.")
    parser.add_argument("--device", help="Execution device; defaults to current project XGB_DEVICE and is recorded.")
    parser.add_argument("--n-jobs", type=_positive_integer, help="Thread count; default is the recorded project setting.")
    parser.add_argument("--batch-rows", type=_positive_integer, default=100000, help="Validation scoring batch size.")
    parser.add_argument("--check-only", action="store_true", help="Validate source artifacts and recipe without loading/fitting the population.")
    return parser.parse_args(argv)




def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    from reselected_outer_support import load_reselection, activate_profile, load_shared_state
    from xgboost_outer_comparison import write_comparison
    from c_modeling import c4_evaluate_and_finalize as c4


    run = args.run_dir.expanduser().resolve(strict=True)
    saved = load_reselection(run, args.recipe)
    payload = saved["recipe"]
    prep_path = run / "preparation" / "preparation_info.json"
    prep = read_json(prep_path)
    outer_path = run / "outer_evaluation" / "outer_fold_bundle_manifest.json"
    outer = read_json(outer_path)
    require_equal(outer["contract"], c4.OUTER_CONTRACT, "Original outer contract")
    require_equal(outer["contract_revision"], c4.OUTER_CONTRACT_REVISION, "Original outer revision")
    require_equal(outer["run_id"], run.name, "Original outer run")
    require_equal(outer["profile_hash"], payload["profile_hash"], "Challenger profile")
    require_equal(prep["profile_hash"], payload["profile_hash"], "Preparation profile")
    require_equal(outer["population_scope_mode"], "FULL", "Outer population scope")
    require_equal(outer["final_holdout_rows_scored"], 0, "Original outer test exclusion")
    require_equal(prep["final_holdout_rows_scored"], 0, "Preparation test exclusion")
    params = activate_profile(root, prep, run.name)
    from c_modeling import deployment_bundle as bundles, modeling_utils as mu, model_diagnostics as diagnostics
    import xgboost as xgb
    import scipy
    import sklearn
    import joblib
    import pyarrow


    recipe = make_recipe(payload, c4)
    require_equal(str(xgb.__version__), str(payload["xgboost_version"]), "Selected XGBoost version")
    settings = prep["split_settings"]
    expected_folds = tuple(c4.OUTER_FOLD_IDS)
    require_equal(int(settings["rolling_validation_folds"]), len(expected_folds), "Saved fold count")
    require_equal(int(params.LABEL_AVAILABILITY_BUFFER_DAYS), int(settings["label_availability_buffer_days"]),
                  "Label availability buffer")
    fold_refs = {int(row["fold"]): row for row in outer["folds"]}
    require_equal(tuple(sorted(fold_refs)), expected_folds, "Baseline fold IDs")
    require_equal(len(fold_refs), len(outer["folds"]), "Unique baseline folds")
    identities = list(outer["identity_columns"])
    input_name = PureWindowsPath(str(prep["input"])).name
    input_path = run / "input_snapshot" / input_name
    input_sha = prep["input_sha256"]
    require_equal(payload["source_input"]["sha256"], input_sha, "Recipe input")
    require_equal(outer["source_data_identity"]["sha256"], input_sha, "Outer input")
    oof_path = run / "outer_evaluation" / "outer_oof_predictions.parquet"
    metrics_path = run / "outer_evaluation" / "outer_fold_metrics.csv"
    folds_path = run / "preparation" / "outer_fold_definitions.csv"
    print(f"Recipe: {saved['recipe_path']}", flush=True)
    print(f"Challenger: {len(recipe.features)} inputs; {recipe.history_months} mature months; "
          f"{recipe.boosting_rounds} rounds; seeds {list(recipe.seeds)}", flush=True)
    print("Verifying saved data, outer predictions and baseline bundles...", flush=True)
    verify_file(input_path, input_sha)
    verify_file(oof_path, outer["outer_oof_predictions"]["sha256"])
    verify_file(metrics_path, outer["outer_fold_metrics"]["sha256"])


    originals = {}
    sources = {str(p): digest(p) for p in (prep_path, outer_path, folds_path, oof_path, metrics_path)}
    sources.update({str(p): h for p, h in saved["source_hashes"].items()})
    for number in expected_folds:
        originals[number] = {}
        for family in ("logistic", "xgboost"):
            item = baseline_bundle(run, fold_refs[number]["families"][family], family, number,
                                   outer=outer, bundles=bundles)
            originals[number][family] = item
            sources[str(item[3])] = digest(item[3])
        manifest = originals[number]["xgboost"][1]
        require_equal(manifest["features"], saved["saved"]["selected"]["selected_features"], "Baseline selected features")
        require_equal(manifest["boosting_rounds"], saved["saved"]["selected"]["recommended_boosting_rounds_from_inner_confirmation"],
                      "Baseline selected rounds")
        require_equal(manifest["seeds"], saved["saved"]["seeds"], "Baseline selected seeds")
        original_recipe = saved["saved"]["selected"]
        for name, expected in original_recipe["selected_model_spec"].items():
            require_equal(manifest["model_recipe"].get(name), expected, f"Baseline selected setting {name}")
        require_equal(manifest["model_recipe"]["selected_history"], original_recipe["selected_history"],
                      "Baseline selected history")
        require_equal(originals[number]["xgboost"][2]["rare_min_frequency"],
                      saved["saved"]["rare_min_frequency"], "Baseline rare category setting")


    baseline_features = list(originals[expected_folds[0]]["xgboost"][1]["features"])
    device = args.device or str(params.XGB_DEVICE)
    n_jobs = int(args.n_jobs if args.n_jobs is not None else params.N_JOBS)
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be positive or the project setting -1.")
    runtime = SimpleNamespace(XGB_DEVICE=device, N_JOBS=n_jobs, SEED=int(params.SEED))
    code_modules = [c4, bundles, mu, diagnostics, importlib.import_module("reselected_outer_support"),
                    importlib.import_module("xgboost_outer_comparison"), params,
                    importlib.import_module("func.retention_population"),
                    importlib.import_module("func.tenure_profile")]
    code_files = [Path(__file__).resolve(), *(Path(module.__file__).resolve() for module in code_modules)]
    for candidate in (root / "c_modeling" / "state_grouping.py", root / "c_modeling" / "feature_routing.py",
                      root / "c_modeling" / "feature_variant_map.json", root / "a_business_understanding" / "a0_one_click.py"):
        if candidate.is_file():
            code_files.append(candidate)
    code_hashes = {str(path.relative_to(root)): digest(path) for path in code_files}
    code_sources = {str(path): digest(path) for path in code_files}
    # Export/selection helpers are part of verification even though no tuning is
    # performed. Changes to them must invalidate a resumed comparison too.
    for module_name in ("reselect_saved_xgboost", "c_modeling.xgboost_selection_policy"):
        module_path = Path(importlib.import_module(module_name).__file__).resolve()
        code_hashes[str(module_path.relative_to(root))] = digest(module_path)
        code_sources[str(module_path)] = digest(module_path)
    contract = {"contract": CONTRACT, "source_run_id": run.name, "profile_hash": payload["profile_hash"],
                "recipe_sha256": digest(saved["recipe_path"]), "input_sha256": input_sha,
                "source_sha256": {str(Path(p).relative_to(root)) if Path(p).is_relative_to(root) else str(p): h
                                  for p, h in sources.items()},
                "code_sha256": code_hashes, "folds": list(expected_folds), "device": device, "n_jobs": n_jobs,
                "batch_rows": args.batch_rows,
                "runtime": {"python": platform.python_version(), "pandas": pd.__version__,
                            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                            "xgboost": xgb.__version__, "joblib": joblib.__version__, "pyarrow": pyarrow.__version__},
                "outer_comparison_is_post_hoc": True, "final_test_scored": False}
    comparison_root = run / "xgboost_outer_comparisons"
    output = (args.output_dir.expanduser().resolve() if args.output_dir else
              comparison_root / f"{payload['selected_group_id']}_{contract['recipe_sha256'][:12]}")
    if not output.resolve().is_relative_to(comparison_root.resolve()) or output.resolve() == comparison_root.resolve():
        raise ValueError("--output-dir must be a child of this run's xgboost_outer_comparisons/.")
    print(f"Comparison outputs: {output}", flush=True)
    if args.check_only:
        print("SOURCE CHECK PASSED. Full row/fold reconstruction is checked when evaluation starts. No models fitted.")
        return


    output.mkdir(parents=True, exist_ok=True)
    with ComparisonLock(output / "comparison.lock"):
        contract_path = output / "comparison_contract.json"
        if contract_path.exists():
            require_equal(read_json(contract_path), contract, "Comparison resume contract (use a new --output-dir if settings changed)")
        else:
            if any(p.name != "comparison.lock" for p in output.iterdir()):
                raise ValueError("Refusing to adopt a comparison directory without its contract.")
            c4._atomic_json(contract_path, contract)
        complete_path = output / "run_status.json"
        if complete_path.exists() and read_json(complete_path).get("status") == "COMPLETE":
            status = read_json(complete_path)
            require_equal(status["comparison_contract_sha256"], digest(contract_path), "Completed comparison identity")
            for relative, sha in status["output_sha256"].items():
                verify_file(output / relative, sha)
            print(f"Already complete; results reused: {output / 'comparison_summary.csv'}")
            return
        c4._atomic_json(output / "run_status.json.incomplete", {"status": "RUNNING", "contract": contract,
                                                              "final_holdout_rows_scored": 0})
        print("Reconstructing the saved population and five outer folds...", flush=True)
        raw = mu.read_table(input_path)
        prepared, folds, holdout_idx, split_summary, config = mu.load_project_population_and_folds(
            raw, rolling_validation_folds=int(settings["rolling_validation_folds"]),
            validation_months=int(settings["outer_validation_months"]), test_months=int(settings["final_holdout_months"]),
            min_train_months=int(settings["min_outer_train_months"]), strict_dataset_contract=bool(settings["strict_dataset_contract"]))
        del raw
        gc.collect()
        actual = mu.fold_definitions_frame(prepared, folds, cohort_col=config.cohort_date_col,
                                          target_check_col=config.target_check_col, target_col=config.target_col)
        mu.verify_fold_definition_contract(actual, folds_path, label="reselected XGBoost outer evaluation")
        require_equal(tuple(int(fold["fold"]) for fold in folds), expected_folds, "Reconstructed folds")
        for key, value in (("target_column", config.target_col), ("cohort_column", config.cohort_date_col),
                           ("tenure_column", config.tenure_col)):
            require_equal(value, outer[key], key)
        holdout = prepared.loc[np.asarray(holdout_idx), identities]
        require_equal(len(holdout), int(outer["excluded_holdout"]["rows"]), "Excluded holdout rows")
        require_equal(bundles.population_sha256(holdout, identities), outer["excluded_holdout"]["population_sha256"],
                      "Excluded holdout identity")
        del holdout
        oof = pd.read_parquet(oof_path, filters=[("model_family", "==", "xgboost")])
        require_equal(set(oof["model_family"]), {"xgboost"}, "OOF baseline family")
        require_equal(tuple(sorted(oof["fold"].unique())), expected_folds, "OOF baseline folds")
        if oof["row_id"].duplicated().any():
            raise ValueError("Baseline OOF rows are duplicated across folds.")
        require_equal(len(oof), sum(len(fold["validation_idx"]) for fold in folds), "Baseline OOF total rows")
        original_metrics = pd.read_csv(metrics_path)
        original_metrics = original_metrics.loc[original_metrics["model_family"].eq("xgboost")].set_index("fold")
        predictions, history_rows, receipts = [], [], []
        source_code_paths = {"modeling_utils": Path(mu.__file__).resolve()}
        for fold in folds:
            number = int(fold["fold"])
            print(f"OUTER FOLD {number}/{len(folds)}", flush=True)
            base, valid_idx = validate_fold_membership(prepared, fold, holdout_idx, config=config,
                                                       buffer_days=int(settings["label_availability_buffer_days"]))
            require_equal(c4._index_sha256(valid_idx), fold_refs[number]["validation_index_sha256"], "Baseline validation membership")
            validation = prepared.loc[valid_idx]
            baseline = aligned_baseline(oof, validation, fold_number=number, config=config, identities=identities, c4=c4)
            computed = diagnostics.prediction_metrics(baseline.rename(columns={"y_retention": "actual_retention",
                                                                                "p_retention": "predicted_retention"}),
                                                     "predicted_retention")
            for metric in ("retention_roc_auc", "churn_ap_skill", "top10_lift", "log_loss", "brier_score", "calibration_gap"):
                if not np.isclose(computed[metric], original_metrics.loc[number, metric], rtol=1e-9, atol=1e-10):
                    raise ValueError(f"Baseline fold {number} {metric} does not reproduce its recorded metric.")
            selected_idx, history = mu.select_recent_mature_training_indices(
                prepared, base, cohort_col=config.cohort_date_col, months=recipe.history_months)
            if not history["requested_months_fully_available"]:
                raise ValueError("The full selected mature-history window is not available.")
            history = {**history, "fold": number, "model_family": "xgboost", "variant": "challenger",
                       "history_id": recipe.history["history_id"], "sample_weight_applied": False}
            history_rows.append(history)
            old_dir, old_manifest, fitted, _ = originals[number]["xgboost"]
            old_xgb_idx = reconstruct_history(prepared, base, old_manifest, mu, bundles, config=config)
            old_lr_idx = reconstruct_history(prepared, base, originals[number]["logistic"][1], mu, bundles, config=config)
            original_shared = np.intersect1d(old_xgb_idx, old_lr_idx)
            challenger_shared = np.intersect1d(selected_idx, old_lr_idx)
            if not len(challenger_shared):
                raise ValueError("No eligible rows for the shared state grouping.")
            state_reused_from_baseline = np.array_equal(original_shared, challenger_shared)
            fold_directory = output / "folds" / f"fold_{number:02d}"
            preliminary_contract = {
                "comparison_contract_sha256": digest(contract_path), "fold": number,
                "training_index_sha256": c4._index_sha256(selected_idx),
                "validation_index_sha256": c4._index_sha256(valid_idx),
                "shared_state_training_index_sha256": c4._index_sha256(challenger_shared),
            }
            state_mode = ("REUSED_IDENTICAL_TRAINING_ONLY_STATE_ARTIFACT" if state_reused_from_baseline
                          else "REFITTED_FOR_CHANGED_CHALLENGER_HISTORY")
            published_directory = fold_directory / "model"
            if published_directory.exists():
                published_manifest, published_fitted = bundles.load_bundle(published_directory, verify_hashes=True)
                published_contract = published_manifest["source_metadata"]["comparison_fold_contract"]
                for key, expected in preliminary_contract.items():
                    require_equal(published_contract.get(key), expected, f"Published fold {key}")
                state = load_shared_state(published_directory, published_manifest, published_fitted)
                require_equal(published_manifest["source_metadata"]["history_application"]["state_artifact_mode"],
                              state_mode, "Published state-fit history")
            elif state_reused_from_baseline:
                state = load_shared_state(old_dir, old_manifest, fitted)
            else:
                state, _ = bundles.fit_shared_state_artifact(training_frame=prepared.loc[challenger_shared],
                                                              target_col=config.target_col, modeling_utils=mu)
            state_contract = bundles.state_contract_from_artifact(state)
            if state_reused_from_baseline:
                require_equal(state_contract["mapping_sha256"], fitted["state_contract"]["mapping_sha256"], "Reused state mapping")
            history["state_artifact_mode"] = state_mode
            # Identity metadata needs only its identity/date columns. Avoid an
            # otherwise unnecessary copy of every training predictor on resume.
            population_columns = list(dict.fromkeys([*identities, config.cohort_date_col]))
            population = c4._population_contract(prepared.loc[:, population_columns], indices=selected_idx,
                                                 cohort_col=config.cohort_date_col,
                                                 identity_columns=identities, deployment_bundle=bundles,
                                                 definition="challenger selected mature outer training population")
            fold_contract = {**preliminary_contract,
                             "state_mapping_sha256": state_contract["mapping_sha256"]}
            metadata = {**old_manifest["source_metadata"], "bundle_role": "CHALLENGER_OUTER_FOLD_EVALUATION",
                        "history_application": history, "shared_state_grouping": state_contract,
                        "evaluation_training_population": population, "comparison_recipe_sha256": contract["recipe_sha256"],
                        "comparison_is_post_hoc": True, "deployment_eligible": False,
                        "pairing_identity": c4._json_sha256(fold_contract), "final_holdout_rows_scored": 0}
            for path, sha in {**sources, **code_sources}.items():
                verify_file(Path(path), sha)
            challenger, receipt = fit_or_score_fold(
                fold_directory, fold_contract, baseline,
                lambda: prepared.loc[selected_idx], validation, recipe=recipe, config=config, identities=identities,
                mu=mu, bundles=bundles, c4=c4, state=state, metadata=metadata,
                source_code_paths=source_code_paths, runtime=runtime, batch_rows=args.batch_rows)
            receipts.append(receipt)
            baseline["variant"] = "baseline"
            challenger["variant"] = "challenger"
            predictions.extend([baseline, challenger])
            del validation, state
            gc.collect()
        result = pd.concat(predictions, ignore_index=True)
        c4._atomic_parquet(output / "outer_comparison_predictions.parquet", result)
        c4._atomic_csv(output / "history_application_audit.csv", pd.DataFrame(history_rows))
        report = write_comparison(result, output, model_diagnostics=diagnostics,
                                  baseline_label=f"Original XGBoost ({len(baseline_features)} inputs)",
                                  challenger_label=f"Challenger XGBoost ({len(recipe.features)} inputs)")
        for path, sha in {**sources, **code_sources}.items():
            verify_file(Path(path), sha)
        c4._atomic_json(output / "comparison_manifest.json", {
            "contract": contract, "recipe": payload, "report": report,
            "outer_validation_rows_per_variant": len(oof), "fold_receipts": receipts,
            "original_model_modified": False, "optuna_performed": False,
            "final_refit_performed": False, "final_holdout_rows_scored": 0,
        })
        output_hashes = {str(p.relative_to(output)): digest(p) for p in output.rglob("*")
                         if p.is_file() and p.name not in ("comparison.lock", "run_status.json", "run_status.json.incomplete")
                         and not any(part.startswith(".") for part in p.relative_to(output).parts)}
        c4._atomic_json(complete_path, {"status": "COMPLETE", "completed_utc": datetime.now(timezone.utc).isoformat(),
                                      "comparison_contract_sha256": digest(contract_path), "output_sha256": output_hashes,
                                      "final_holdout_rows_scored": 0})
        (output / "run_status.json.incomplete").unlink(missing_ok=True)
        print(f"OUTER COMPARISON COMPLETE: {len(oof):,} validation rows per variant", flush=True)
        print(f"Summary: {output / 'comparison_summary.csv'}")
        print(f"Paired changes: {output / 'paired_metric_differences.csv'}")
        print(f"Report: {output / 'report.md'}")
        print("Original model retained. Final test rows scored: 0.")




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Repeat the same command to reuse completed folds and published models.", file=sys.stderr)
        raise SystemExit(130)
