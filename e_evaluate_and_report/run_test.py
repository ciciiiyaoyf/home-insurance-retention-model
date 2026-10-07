"""Evaluate the saved final champion on its six-month chronological test.


Run from the project root:
    python e_evaluate_and_report/run_test.py --run-id RUN_ID


This explicitly opens the final test. No fitting, selection, tuning, or
deployment occurs. Finish tenure/grouping choices before using it as a final
assessment. Repeating the command reuses committed predictions for the same
frozen model and regenerates its reports.
"""
from __future__ import annotations


import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping


import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from e_evaluate_and_report import report_plots as charts  # noqa: E402




def _reports():
    from e_evaluate_and_report import make_reports
    return make_reports




def now() -> str:
    return datetime.now(timezone.utc).isoformat()




def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)




def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)




@contextmanager
def scoring_lock(output: Path):
    """Prevent concurrent writers to this run's test results."""
    path = output / "scoring.lock"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(now())
    except FileExistsError as exc:
        raise RuntimeError(
            f"A final-test operation already holds {path}. If its process was "
            "interrupted, confirm it has stopped before removing only this lock file."
        ) from exc
    try:
        yield
    finally:
        path.unlink(missing_ok=True)




def verify_holdout(frame, indices, excluded, bundle_manifest, bundle_api, c4, *, cohort_col):
    """Require exactly the original test, before any test transformation."""
    reports = _reports()
    indices = np.asarray(indices)
    if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices):
        raise ValueError("Test membership must be a nonempty unique index vector.")
    columns = list(excluded["identity_columns"])
    if not columns or set(columns) - set(frame.columns):
        raise ValueError("Every saved test identity column is required.")
    part = frame.loc[indices, list(dict.fromkeys([*columns, cohort_col]))]
    dates = pd.to_datetime(part[cohort_col], errors="raise")
    if dates.isna().any():
        raise ValueError("Test cohort dates cannot be missing.")
    months = pd.PeriodIndex(sorted(dates.dt.to_period("M").unique()))
    if len(months) != 6 or not months.equals(pd.period_range(months[0], periods=6, freq="M")):
        raise ValueError("The final test must cover six consecutive cohort months.")
    for name, actual in (
        ("rows", len(part)), ("cohort_months", len(months)),
        ("cohort_start", str(dates.min().date())), ("cohort_end", str(dates.max().date())),
        ("population_sha256", bundle_api.population_sha256(part, columns)),
        ("row_index_sha256", c4._index_sha256(indices)),
    ):
        reports.require_equal(actual, excluded[name], f"Final test {name}")
        reports.require_equal(bundle_manifest["source_metadata"]["sealed_final_holdout"][name],
                              excluded[name], f"Bundle test {name}")
    reports.require_equal(bundle_manifest["source_metadata"]["bundle_role"], "FINAL_REFIT", "Test model role")
    return part




def score_test(frame, indices, excluded, model_dir, manifest, bundle_api, c4, *, cohort_col, batch_rows, on_batch=None):
    """Explicit final-test scoring; preserve the ordinary reporter's seal guard."""
    reports = _reports()
    verify_holdout(frame, indices, excluded, manifest, bundle_api, c4, cohort_col=cohort_col)
    columns = reports.raw_scoring_columns(manifest)
    predictors = frame.loc[indices, columns]
    family = manifest["model_family"]
    if family == "logistic":
        iterator = bundle_api.iter_logistic_bundle_scores
    elif family == "xgboost":
        iterator = bundle_api.iter_xgboost_bundle_scores
    else:
        raise ValueError("Unsupported final model family.")
    probability = np.empty(len(indices), dtype=float)
    cursor = 0
    for start, stop, values in iterator(model_dir, predictors, batch_size=batch_rows, verify_hashes=True):
        values = np.asarray(values, dtype=float)
        if start != cursor or stop <= start or stop > len(indices) or values.shape != (stop - start,):
            raise ValueError("Test prediction batches must cover each row exactly once, in order.")
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Test probabilities must be finite and within [0, 1].")
        probability[start:stop] = values if family == "logistic" else 1.0 - values
        cursor = stop
        if on_batch is not None:
            on_batch(cursor)
        print(f"Final test rows predicted: {cursor:,} of {len(indices):,}", flush=True)
    reports.require_equal(cursor, len(indices), "Test prediction coverage")
    return probability




def prediction_frame(frame, indices, probability, *, family, fold, config, identities, c4):
    columns = list(dict.fromkeys([*identities, config.cohort_date_col, config.target_col, config.tenure_col]))
    part = frame.loc[indices, columns]
    result = pd.DataFrame({
        "row_id": c4.stable_row_ids(part, identities).astype(str).to_numpy(),
        "model_family": family, "fold": fold,
        "cohort_date": pd.to_datetime(part[config.cohort_date_col]).to_numpy(),
        "y_retention": part[config.target_col].to_numpy(),
        "p_retention": probability,
        "tenure_group": part[config.tenure_col].astype(str).to_numpy(),
    })
    if result.row_id.duplicated().any() or set(result.y_retention.unique()) != {0, 1}:
        raise ValueError("Prediction rows must have unique identities and both target classes.")
    return result




def cached_predictions(output, status, source_identity, bundles):
    """Reuse committed scores, but never substitute a changed final model."""
    reports = _reports()
    reports.require_equal(status["source_identity"], source_identity, "Previously evaluated frozen model")
    if not status.get("scoring_complete", False):
        return None
    files = status["prediction_files"]
    result = {}
    for split in ("train", "validation", "test"):
        item = files[split]
        name = item["filename"]
        if Path(name).name != name:
            raise ValueError("Cached prediction filenames must be run-local.")
        path = output / name
        reports.verify_digest(path, item["sha256"], bundles)
        part = pd.read_parquet(path)
        reports.require_equal(len(part), int(item["rows"]), f"Cached {split} rows")
        result[split] = part
    return result




def write_outputs(predictions, output, status, charts, diagnostics, performance_plots):
    """Plot only committed predictions; a plotting failure cannot erase exposure."""
    status.update(status="REPORTING", plots_complete=False)
    atomic_json(output / "test_status.json", status)
    try:
        return _write_outputs(predictions, output, status, charts, diagnostics, performance_plots)
    except Exception as exc:
        status.update(status="FAILED", plots_complete=False, failed_at=now(), error=f"{type(exc).__name__}: {exc}")
        atomic_json(output / "test_status.json", status)
        raise




def _write_outputs(predictions, output, status, charts, diagnostics, performance_plots):
    family = status["selected_model_family"]
    records = {}
    for split in ("train", "test"):
        part = predictions[split]
        records[split] = {
            "model_family": family, "split": split, "rows": len(part),
            "y_retention": part.y_retention.to_numpy(), "p_retention": part.p_retention.to_numpy(),
        }
    records["train"]["training_population_rows"] = status["final_training_rows"]
    validation = [dict(
        model_family=family, split="validation", fold=int(number), rows=len(part),
        y_retention=part.y_retention.to_numpy(), p_retention=part.p_retention.to_numpy(),
    ) for number, part in predictions["validation"].groupby("fold", sort=True)]
    # The first plot is the requested three-line view. All test-only plots
    # below use the single final model, never a pooled collection of refits.
    comparison = charts.write_final_test_comparison(records["train"], validation, records["test"], output)
    test = predictions["test"]
    metric_input = test.rename(columns={"y_retention": "actual_retention", "p_retention": "predicted_retention"})
    metrics = diagnostics.prediction_metrics(metric_input, "predicted_retention")
    pd.DataFrame([dict(model_family=family, evaluation_scope="final_test", **metrics)]).to_csv(output / "test_metrics.csv", index=False)
    plot_manifest = performance_plots.generate_performance_plots(test, output / "plots", evaluation_scope="final_test")
    status.update(status="COMPLETE", completed_at=now(), plots_complete=True,
                  comparison_plot=comparison, performance_plots=plot_manifest, test_metrics=metrics)
    atomic_json(output / "test_status.json", status)
    print(f"FINAL TEST COMPLETE: {output}\nModel: {family}\nTest rows scored: {len(test):,}\n"
          f"ROC AUC: {metrics['retention_roc_auc']:.4f}\nChurn AP: {metrics['churn_average_precision']:.4f}\n"
          f"Lift@10%: {metrics['top10_lift']:.4f}\nModel fits: 0", flush=True)




def run(args):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from c_modeling import deployment_bundle as bundles, modeling_utils, c4_evaluate_and_finalize as c4
    from c_modeling import model_diagnostics as diagnostics, performance_plots
    reports = _reports()
    if not hasattr(charts, "write_final_test_comparison"):
        raise RuntimeError("Install the report_plots.py supplied with run_test.py before running this command.")
    run_root = root / "output" / "pipeline_runs" / args.run_id
    outer_dir = run_root / "outer_evaluation"
    preparation = reports.read_json(run_root / "preparation" / "preparation_info.json")
    outer = reports.read_json(outer_dir / "outer_fold_bundle_manifest.json")
    c4_status = reports.read_json(outer_dir / "run_status.json")
    reports.require_equal(c4_status["status"], "COMPLETE", "C4 status")
    reports.require_equal(outer["run_id"], args.run_id, "Outer run")
    reports.require_equal(outer["profile_hash"], preparation["profile_hash"], "Preparation profile")
    final_path = run_root / "deployment" / "final_model_manifest.json"
    reports.verify_digest(final_path, outer["final_model_manifest"]["sha256"], bundles)
    final = reports.read_json(final_path)
    reports.require_equal(final["run_id"], args.run_id, "Final run")
    reports.require_equal(final["profile_hash"], outer["profile_hash"], "Final profile")
    # These are C4's historical zero-scoring receipts. New exposure is recorded
    # in final_test/test_status.json, never written into hash-bound C4 artifacts.
    for source in (final, outer):
        reports.require_equal(int(source["final_holdout_rows_scored"]), 0, "C4 historical test scoring")
    selection_path = outer_dir / "family_selection.json"
    reports.verify_digest(selection_path, final["family_selection_sha256"], bundles)
    family = final["selected_model_family"]
    if family not in ("logistic", "xgboost"):
        raise ValueError("Unsupported selected family.")
    reports.require_equal(family, outer["selected_model_family"], "Outer selected family")
    reports.require_equal(family, reports.read_json(selection_path)["selected_model_family"], "Frozen family choice")
    input_sha = outer["source_data_identity"]["sha256"]
    reports.require_equal(input_sha, preparation["input_sha256"], "Input identity")
    identity = dict(run_id=args.run_id, profile_hash=outer["profile_hash"], input_sha256=input_sha,
                    final_manifest_sha256=bundles.sha256_file(final_path),
                    selected_bundle_sha256=final["selected_bundle"]["sha256"],
                    outer_manifest_sha256=bundles.sha256_file(outer_dir / "outer_fold_bundle_manifest.json"))
    output = run_root / "final_test"
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "test_status.json"
    with scoring_lock(output):
        status = reports.read_json(status_path) if status_path.exists() else {
            "contract": "retention_final_test_v1", "source_identity": identity,
            "run_id": args.run_id, "profile_hash": outer["profile_hash"], "selected_model_family": family,
            "created_at": now(), "scoring_complete": False, "plots_complete": False,
            "training_performed": False, "tuning_performed": False, "deployment_eligible": False,
            "test_exposure_possible": False, "final_holdout_rows_scored": 0, "attempts": [],
            "interpretation": "Evaluation of this frozen model. If results inform later choices, the holdout is reused development evidence.",
        }
        cached = cached_predictions(output, status, identity, bundles)
        if cached is not None:
            print("Using saved test predictions; no model will be rescored.", flush=True)
            write_outputs(cached, output, status, charts, diagnostics, performance_plots)
            return output
        status["attempts"].append({"started_at": now(), "same_frozen_model": True})
        status["status"] = "VERIFYING"
        atomic_json(status_path, status)
        try:
            params = reports.activate_profile(root, preparation, args.run_id)
            settings = preparation["split_settings"]
            for attr, key in (("ROLLING_VALIDATION_FOLDS", "rolling_validation_folds"),
                              ("OUTER_VALIDATION_MONTHS", "outer_validation_months"),
                              ("FINAL_HOLDOUT_MONTHS", "final_holdout_months"),
                              ("LABEL_AVAILABILITY_BUFFER_DAYS", "label_availability_buffer_days")):
                reports.require_equal(getattr(params, attr), settings[key], attr)
            input_path = run_root / "input_snapshot" / "retention_13_1_population.parquet"
            print("Verifying snapshot and reconstructing saved training/test membership.", flush=True)
            reports.verify_digest(input_path, input_sha, bundles)
            raw = modeling_utils.read_table(input_path)
            prepared, folds, test_idx, _, config = modeling_utils.load_project_population_and_folds(
                raw, rolling_validation_folds=int(settings["rolling_validation_folds"]),
                validation_months=int(settings["outer_validation_months"]), test_months=int(settings["final_holdout_months"]),
                min_train_months=int(settings["min_outer_train_months"]), strict_dataset_contract=True)
            del raw
            definitions = modeling_utils.fold_definitions_frame(prepared, folds, cohort_col=config.cohort_date_col,
                target_check_col=config.target_check_col, target_col=config.target_col)
            modeling_utils.verify_fold_definition_contract(definitions, run_root / "preparation" / "outer_fold_definitions.csv", label="Final test")
            model_dir, manifest = reports.load_saved_bundle(run_root, final["selected_bundle"],
                run_root / "deployment" / "final_model" / family, family=family, run_id=args.run_id,
                profile_hash=outer["profile_hash"], input_sha=input_sha, bundle_api=bundles)
            excluded = final["excluded_holdout"]
            verify_holdout(prepared, test_idx, excluded, manifest, bundles, c4, cohort_col=config.cohort_date_col)
            identities = list(excluded["identity_columns"])
            reports.require_equal(identities, list(outer["identity_columns"]), "Identity columns")
            base_idx, audit = c4.final_evaluation_training_indices(prepared, test_idx, cohort_col=config.cohort_date_col,
                target_check_col=config.target_check_col, label_buffer_days=int(settings["label_availability_buffer_days"]))
            reports.require_equal(audit, final["final_training_reconstruction"], "Final training reconstruction")
            train_idx = reports.selected_training_indices(prepared, base_idx, manifest, modeling_utils, cohort_col=config.cohort_date_col)
            reports.verify_training_population(prepared, train_idx, manifest, bundles, cohort_col=config.cohort_date_col,
                target_check_col=config.target_check_col, label_cutoff=audit["label_cutoff_exclusive"], holdout_start=excluded["cohort_start"])
            if np.intersect1d(train_idx, test_idx).size:
                raise ValueError("Training and final test rows overlap.")
            oof_path = outer_dir / "outer_oof_predictions.parquet"
            reports.verify_digest(oof_path, outer["outer_oof_predictions"]["sha256"], bundles)
            oof = pd.read_parquet(oof_path)
            reports.require_equal(len(oof), int(outer["outer_oof_predictions"]["rows"]), "OOF rows")
            validation_parts = []
            for fold in folds:
                indices = np.asarray(fold["validation_idx"])
                if np.intersect1d(indices, test_idx).size:
                    raise ValueError("Outer validation overlaps the final test.")
                part = prepared.loc[indices, list(dict.fromkeys([*identities, config.target_col, config.cohort_date_col]))]
                validation_parts.append(reports.validated_outer_rows(oof, part, family=family, fold=int(fold["fold"]),
                    target_col=config.target_col, cohort_col=config.cohort_date_col, identity_columns=identities, c4=c4).reset_index())
            validation_predictions = pd.concat(validation_parts, ignore_index=True)
            del oof, validation_parts
            sample = np.asarray(train_idx)
            if args.train_sample_rows and len(sample) > args.train_sample_rows:
                sample = np.sort(np.random.default_rng(42).choice(sample, args.train_sample_rows, replace=False))
            print(f"Scoring {len(sample):,} training rows with the saved final model for the comparison curve.", flush=True)
            train_p = np.concatenate([p for _, p in reports.score_batches(prepared, sample, model_dir, manifest, bundles,
                holdout_start=excluded["cohort_start"], cohort_col=config.cohort_date_col, batch_rows=args.batch_rows)])
            training_predictions = prediction_frame(prepared, sample, train_p, family=family, fold="FINAL_TRAIN",
                config=config, identities=identities, c4=c4)
            status.update(status="SCORING_TEST", test_exposure_possible=True, excluded_holdout=excluded,
                          test_rows_scored_count_is_complete=False,
                          final_training_rows=len(train_idx), training_curve_rows=len(sample))
            atomic_json(status_path, status)
            def record_test_batch(rows):
                status["final_holdout_rows_scored"] = int(rows)
                atomic_json(status_path, status)
            test_p = score_test(prepared, test_idx, excluded, model_dir, manifest, bundles, c4,
                                cohort_col=config.cohort_date_col, batch_rows=args.batch_rows, on_batch=record_test_batch)
            # Preserve exposure even if a later output/plot operation fails.
            status["final_holdout_rows_scored"] = int(len(test_idx))
            status["test_rows_scored_count_is_complete"] = True
            atomic_json(status_path, status)
            test_predictions = prediction_frame(prepared, test_idx, test_p, family=family, fold="FINAL_TEST",
                config=config, identities=identities, c4=c4)
            predictions = {"train": training_predictions, "validation": validation_predictions, "test": test_predictions}
            inventory = {}
            for split, part in predictions.items():
                path = output / f"{split}_predictions.parquet"
                atomic_parquet(path, part)
                inventory[split] = dict(filename=path.name, sha256=bundles.sha256_file(path), rows=len(part))
            status.update(status="SCORED", scoring_complete=True, prediction_files=inventory, scored_at=now())
            atomic_json(status_path, status)
            del prepared
            write_outputs(predictions, output, status, charts, diagnostics, performance_plots)
            return output
        except Exception as exc:
            status.update(status="FAILED", failed_at=now(), error=f"{type(exc).__name__}: {exc}")
            atomic_json(status_path, status)
            raise




def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-rows", type=int, default=100_000)
    parser.add_argument("--train-sample-rows", type=int, default=100_000,
                        help="Final-model training curve sample; 0 scores all training rows. Test always uses all rows.")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}", args.run_id):
        parser.error("Invalid run ID.")
    if args.batch_rows < 1 or args.train_sample_rows < 0:
        parser.error("batch-rows must be positive and train-sample-rows nonnegative.")
    return args




if __name__ == "__main__":
    run(parse_args())
