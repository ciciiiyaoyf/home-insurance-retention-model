# -*- coding: utf-8 -*-
"""Run the current 13/1 retention model-development stages in one click.
The ordered list is intentionally explicit. It prepares the temporal modeling
population, candidate pool, and validation folds, runs Logistic Regression and
XGBoost against the same downloaded snapshot and feature contract, scores the
reserved outer folds, and refits one provisional pre-holdout model. Deployment
eligibility remains blocked until tenure grouping is locked and the separately
governed final test is approved and passed. The sealed final holdout remains
unscored throughout model-development and tenure-grouping decisions.
"""
from __future__ import annotations
from datetime import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from func.modeling import run_scripts  # noqa: E402
from func.operation_lock import exclusive_operation_lock  # noqa: E402
from params import (  # noqa: E402
    ACTIVE_PIPELINE_STATE_PATH,
    PIPELINE_OPERATION_LOCK_PATH,
    PIPELINE_OUTPUT_ROOT,
    PROFILE_HASH,
    PROFILE_LABEL,
    new_run_id,
    profile_env,
    profile_state_payload,
    setup_params,
)
SCRIPTS = [
    "c1_prepare_modeling_data.py",
    "c2_run_logistic_regression.py",
    "c3_run_xgboost.py",
    "c4_evaluate_and_finalize.py",
]
STEP_NAME = Path(__file__).stem
STATE_PATH = Path(ACTIVE_PIPELINE_STATE_PATH)
ROOT_ORCHESTRATED = (
    os.environ.get("RETENTION_ROOT_ORCHESTRATED", "").strip() == "1"
)
FAILED_RUN_RESUME_ENV = "RETENTION_RESUME_FAILED_RUN"
FAILED_RUN_RESUME = os.environ.get(FAILED_RUN_RESUME_ENV, "").strip() == "1"
FAILED_RUN_RESUME_RUN_ID_ENV = "RETENTION_RESUME_FAILED_RUN_ID"
FAILED_RUN_RESUME_ATTEMPT_ID_ENV = "RETENTION_RESUME_ATTEMPT_ID"
FAILED_RUN_RESUME_STAGE_ENV = "RETENTION_RESUME_FAILED_STAGE"
FAILED_RUN_LOGISTIC_RETRY = "FAILED_RUN_LOGISTIC_RETRY"
FAILED_RUN_XGBOOST_RETRY = "FAILED_RUN_XGBOOST_RETRY"
FAILED_RUN_RESUME_RUN_ID = os.environ.get(
    FAILED_RUN_RESUME_RUN_ID_ENV, ""
).strip()
FAILED_RUN_RESUME_ATTEMPT_ID = os.environ.get(
    FAILED_RUN_RESUME_ATTEMPT_ID_ENV, ""
).strip()
FAILED_RUN_RESUME_STAGE = os.environ.get(
    FAILED_RUN_RESUME_STAGE_ENV, ""
).strip().lower()
XGB_RECONFIRMATION_ENV = "RETENTION_XGB_RECONFIRMATION"
XGB_RECONFIRMATION = os.environ.get(XGB_RECONFIRMATION_ENV, "").strip() == "1"
XGB_RECONFIRMATION_RUN_ID_ENV = "RETENTION_XGB_RECONFIRMATION_RUN_ID"
XGB_RECONFIRMATION_OUTPUT_NAME_ENV = "RETENTION_XGB_RECONFIRMATION_OUTPUT_NAME"
XGB_RECONFIRMATION_RUN_ID = os.environ.get(
    XGB_RECONFIRMATION_RUN_ID_ENV, ""
).strip()
XGB_RECONFIRMATION_OUTPUT_NAME = os.environ.get(
    XGB_RECONFIRMATION_OUTPUT_NAME_ENV, ""
).strip()
MODEL_RESTART_ENV = "RETENTION_CANONICAL_MODEL_RESTART"
MODEL_RESTART_KIND = "CANONICAL_XGBOOST_RESTART"
MODEL_RESTART_STAGE = os.environ.get(MODEL_RESTART_ENV, "").strip()
FINALIZATION_REQUIRED_RELATIVE_PATHS = (
    "outer_evaluation/run_info.json",
    "outer_evaluation/run_status.json",
    "outer_evaluation/outer_oof_predictions.parquet",
    "outer_evaluation/outer_fold_metrics.csv",
    "outer_evaluation/outer_metric_summary.csv",
    "outer_evaluation/outer_tenure_metrics.csv",
    "outer_evaluation/history_application_audit.csv",
    "outer_evaluation/model_comparison.csv",
    "outer_evaluation/family_selection.json",
    "outer_evaluation/performance_plots_manifest.json",
    "outer_evaluation/outer_fold_bundle_manifest.json",
    "deployment/final_model_manifest.json",
)
def _new_run_id() -> str:
    """Build the organization-standard stage run ID."""
    return new_run_id(STEP_NAME)


def _create_setup(run_id: str):
    from func.setup.setup import Setup
    return Setup(**setup_params(run_id))


def _read_state() -> dict[str, object]:
    if not STATE_PATH.is_file():
        return {}
    try:
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "MALFORMED"}
    if not isinstance(payload, dict):
        return {"status": "MALFORMED"}
    if not str(payload.get("status", "")).strip():
        return {**payload, "status": "UNKNOWN"}
    return payload


def _write_state(payload: dict[str, object]) -> None:
    destinations = [STATE_PATH]
    run_id = str(payload.get("run_id", "")).strip()
    if run_id:
        destinations.append(
            Path(PIPELINE_OUTPUT_ROOT) / run_id / "pipeline_status.json"
        )
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, destination)


def _run_id() -> str:
    supplied = os.environ.get("RETENTION_PIPELINE_RUN_ID", "").strip()
    state = _read_state()
    status = str(state.get("status", "")).upper()
    active_run_id = str(state.get("run_id", "")).strip()
    active_profile_hash = str(state.get("profile_hash", "")).strip()
    if ROOT_ORCHESTRATED:
        if not supplied:
            raise RuntimeError(
                "Root-orchestrated modeling requires RETENTION_PIPELINE_RUN_ID."
            )
        if (
            status != "RUNNING"
            or active_run_id != supplied
            or active_profile_hash != PROFILE_HASH
        ):
            raise RuntimeError(
                "Root pipeline state no longer matches the supplied run/profile; "
                "modeling will not take over global state ownership."
            )
        if FAILED_RUN_RESUME and (
            state.get("resume_kind")
            not in {FAILED_RUN_LOGISTIC_RETRY, FAILED_RUN_XGBOOST_RETRY}
            or str(state.get("resume_requested_run_id", "")).strip() != supplied
            or FAILED_RUN_RESUME_RUN_ID != supplied
            or not FAILED_RUN_RESUME_ATTEMPT_ID
            or str(state.get("recovery_attempt_id", "")).strip()
            != FAILED_RUN_RESUME_ATTEMPT_ID
            or FAILED_RUN_RESUME_STAGE
            != (
                "logistic"
                if state.get("resume_kind") == FAILED_RUN_LOGISTIC_RETRY
                else "xgboost"
            )
        ):
            raise RuntimeError(
                "Failed-run recovery environment does not match the guarded "
                "resume receipt owned by root a_business_understanding/a0_one_click.py."
            )
        if MODEL_RESTART_STAGE and (
            MODEL_RESTART_STAGE != "xgboost" or FAILED_RUN_RESUME or XGB_RECONFIRMATION
            or state.get("resume_kind") != MODEL_RESTART_KIND
            or state.get("resume_requested_run_id") != supplied
            or not state.get("recovery_archive")
        ):
            raise RuntimeError("Canonical restart environment does not match the root-owned receipt.")
        return supplied
    if status in {"RUNNING", "PAUSED", "DATA_READY"} and (
        active_profile_hash != PROFILE_HASH
    ):
        raise RuntimeError(
            "The active model controller belongs to a different tenure profile. "
            "Repeat its original command; profile-crossed resume is blocked."
        )
    if status == "RUNNING":
        raise RuntimeError(
            f"Pipeline run {active_run_id or '<unknown>'} is already RUNNING. "
            "Do not start a second modeling controller."
        )
    if status == "PAUSED" and active_run_id:
        if supplied and supplied != active_run_id:
            raise RuntimeError(
                f"Paused run {active_run_id} cannot be resumed as {supplied}."
            )
        return active_run_id
    if status == "DATA_READY" and active_run_id:
        if supplied and supplied != active_run_id:
            raise RuntimeError(
                f"Prepared run {active_run_id} cannot be modeled as {supplied}."
            )
        return active_run_id
    if state and status not in {"COMPLETE", "FAILED"}:
        raise RuntimeError(
            f"Pipeline state is {status or 'UNKNOWN'}. Repair or restore "
            "output/pipeline_state/active_run.json before modeling."
        )
    if supplied:
        return supplied
    raise RuntimeError(
        "No DATA_READY or PAUSED run is available for standalone modeling. "
        "Run b_data_acquisition/b0_one_click.py first, or use root a_business_understanding/a0_one_click.py."
    )


def _run_paths(run_id: str) -> dict[str, str]:
    root = Path(PIPELINE_OUTPUT_ROOT) / run_id
    return {
        "snapshot_path": str(
            root / "input_snapshot" / "retention_13_1_population.parquet"
        ),
        "snapshot_metadata_path": str(
            root
            / "input_snapshot"
            / "retention_13_1_population.metadata.json"
        ),
        "feature_contract_path": str(root / "model_inputs" / "feature_contract.csv"),
        "candidate_pool_path": str(root / "model_inputs" / "candidate_pool.txt"),
    }


def _state_payload(run_id: str, status: str, **values: object) -> dict[str, object]:
    prior = _read_state()
    if str(prior.get("run_id", "")).strip() != run_id:
        prior = {}
    return {
        **prior,
        "run_id": run_id,
        "status": status,
        **profile_state_payload(),
        **_run_paths(run_id),
        **values,
    }




def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload




def _require_declared_artifacts(manifest_path: Path) -> None:
    payload = _read_json_object(manifest_path, "diagnostic manifest")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError(
            f"Diagnostic manifest has no nonempty artifacts list: {manifest_path}"
        )
    invalid: list[str] = []
    for value in artifacts:
        if isinstance(value, dict):
            name = str(value.get("path") or value.get("filename") or "").strip()
        else:
            name = str(value).strip()
        declared = Path(name)
        candidate = (
            declared if declared.is_absolute() else manifest_path.parent / declared
        )
        try:
            within_manifest_tree = candidate.resolve().is_relative_to(
                manifest_path.parent.resolve()
            )
        except OSError:
            within_manifest_tree = False
        if (
            not name
            or not within_manifest_tree
            or candidate.is_symlink()
            or not candidate.is_file()
            or candidate.stat().st_size <= 0
        ):
            invalid.append(name or "<empty>")
    if invalid:
        raise RuntimeError(
            f"Diagnostic manifest {manifest_path} has missing or unsafe "
            "declared artifacts: " + ", ".join(invalid)
        )




def _validate_finalization_outputs(run_id: str) -> dict[str, object]:
    run_root = Path(PIPELINE_OUTPUT_ROOT) / run_id
    invalid: list[str] = []
    for relative in FINALIZATION_REQUIRED_RELATIVE_PATHS:
        path = run_root / relative
        try:
            valid = (
                not path.is_symlink()
                and path.is_file()
                and path.stat().st_size > 0
            )
        except OSError:
            valid = False
        if not valid:
            invalid.append(relative)
    if invalid:
        raise RuntimeError(
            "Outer evaluation/final refit did not produce its complete artifact "
            "contract:\n- " + "\n- ".join(invalid)
        )
    status = _read_json_object(
        run_root / "outer_evaluation" / "run_status.json",
        "outer-evaluation status",
    )
    scored = int(status.get("final_holdout_rows_scored", 0))
    if str(status.get("status", "")).upper() != "COMPLETE":
        raise RuntimeError("Outer evaluation did not report COMPLETE status.")
    if str(status.get("profile_hash", "")).strip() != PROFILE_HASH:
        raise RuntimeError("Outer evaluation profile does not match this run.")
    if scored != 0:
        raise RuntimeError(
            "c4 scored the sealed final holdout during model development."
        )
    _require_declared_artifacts(
        run_root / "outer_evaluation" / "performance_plots_manifest.json"
    )
    return status
def _archive_xgboost_attempt(
    run_id: str,
    *,
    output_root: Path = Path(PIPELINE_OUTPUT_ROOT),
    timestamp: str | None = None,
    recovery_attempt_id: str = "",
) -> Path | None:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise RuntimeError("Unsafe run ID for XGBoost recovery archive.")
    run_root = Path(output_root) / run_id
    source = run_root / "xgboost"
    if not source.exists():
        return None
    if source.is_symlink() or not source.is_dir():
        raise RuntimeError(f"XGBoost attempt is not a safe directory: {source}")
    if source.resolve().parent != run_root.resolve():
        raise RuntimeError(
            "XGBoost recovery refuses a linked directory outside the run root."
        )
    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    base = run_root / f"xgboost_failed_attempt_{stamp}"
    destination = base
    suffix = 1
    while destination.exists():
        destination = run_root / f"{base.name}_{suffix:02d}"
        suffix += 1
    receipt = {
        "run_id": run_id,
        "source_directory": str(source),
        "archive_directory": str(destination),
        "archive_reason": "FAILED_RUN_XGBOOST_RETRY",
        "profile_hash": PROFILE_HASH,
        "recovery_attempt_id": recovery_attempt_id,
        "archived_at": datetime.now().isoformat(timespec="seconds"),
    }
    receipt_path = source / "resume_archive.json"
    temporary = receipt_path.with_name(receipt_path.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    os.replace(temporary, receipt_path)
    source.rename(destination)
    return destination
def _validate_completed_logistic_for_recovery() -> None:
    try:
        runner = importlib.import_module(
            "c_modeling.c2_run_logistic_regression"
        )
        runner._verify_preparation_profile()
        reusable = bool(runner._can_reuse())
    except Exception as exc:
        raise RuntimeError(
            "Could not prove that preparation and Logistic are safely reusable; "
            "no XGBoost output was archived."
        ) from exc
    if not reusable:
        raise RuntimeError(
            "Preparation or Logistic no longer matches its completed run "
            "contract. Recovery will not rerun Logistic or archive XGBoost."
        )




def _validate_completed_selections_for_c4(run_id: str, xgboost_runner) -> dict:
    """Validate frozen completed recipes before C4-only failed-run recovery."""
    from c_modeling.completed_selection_recovery import (
        CompletedSelectionError, validate_completed_selections,
    )
    run_root = Path(PIPELINE_OUTPUT_ROOT) / run_id
    report_path = run_root / "completed_selection_resume_report.json"
    try:
        report = validate_completed_selections(
            run_root=run_root,
            logistic=importlib.import_module("c_modeling.c2_run_logistic_regression"),
            xgboost=xgboost_runner,
        )
    except CompletedSelectionError as exc:
        report_path.write_text(json.dumps(exc.report, indent=2), encoding="utf-8")
        print(f"Resume mismatch report: {report_path}", flush=True)
        raise
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for difference in report["source_differences"]:
        print(
            f"FROZEN SELECTION: {difference['family']}.{difference['field']} "
            "changed since selection; the completed recipe is retained.", flush=True,
        )
    for difference in report["execution_differences"]:
        print(f"C4 EXECUTION: {difference['field']} {difference['saved']} -> "
              f"{difference['requested']}; selection is not repeated.", flush=True)
    print(f"COMPLETED SELECTIONS VERIFIED: {report['selected_group_id']}; "
          f"{report['selected_feature_count']} XGBoost inputs. Running C4 only.", flush=True)
    return report




def _validate_frozen_logistic_for_stage4_recovery(run_id: str) -> None:
    """Validate the frozen Logistic receipt without current-source equality.


    This narrower contract is allowed only after the same run's XGBoost output
    has passed the Stage-4 recovery preflight.  Root a0 already validated these
    files before issuing the guarded recovery environment; repeating the core
    profile/status/holdout checks here prevents this exception from becoming a
    general bypass of Logistic reuse validation.
    """


    run_root = Path(PIPELINE_OUTPUT_ROOT) / run_id
    required = (
        run_root / "logistic" / "run_status.json",
        run_root / "logistic" / "run_info.json",
        run_root / "logistic" / "recommended_model.json",
        run_root / "logistic" / "recommended_features.txt",
        run_root / "logistic" / "resolved_logistic_model_feature_types.json",
        run_root / "logistic" / "diagnostics" / "logistic_diagnostics_manifest.json",
        run_root / "logistic" / "diagnostics" / "logistic_calendar_year_performance.csv",
        run_root / "logistic" / "diagnostics" / "logistic_calendar_year_one_way_summary.csv",
        run_root / "logistic" / "diagnostics" / "logistic_selected_feature_one_way_summaries.csv",
        run_root / "logistic" / "diagnostics" / "logistic_policy_start_year_month_trend.csv",
        run_root / "logistic" / "diagnostics" / "logistic_policy_start_year_month_trend.png",
        run_root / "logistic" / "diagnostics" / "logistic_calendar_year_control_one_way.png",
        run_root / "logistic" / "diagnostics" / "logistic_business_retention_trend_manifest.json",
        run_root / "logistic" / "diagnostics" / "logistic_business_retention_trend_full_history.png",
        run_root / "logistic" / "diagnostics" / "logistic_business_retention_trend_selected_history.png",
    )
    if any(not path.is_file() or path.stat().st_size <= 0 for path in required):
        raise RuntimeError(
            "Frozen Logistic artifacts are incomplete; Stage-4 XGBoost recovery "
            "was refused without archiving any XGBoost output."
        )
    _require_declared_artifacts(
        run_root / "logistic" / "diagnostics" / "logistic_diagnostics_manifest.json"
    )
    _require_declared_artifacts(
        run_root
        / "logistic"
        / "diagnostics"
        / "logistic_business_retention_trend_manifest.json"
    )
    try:
        status = json.loads(required[0].read_text(encoding="utf-8"))
        info = json.loads(required[1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Frozen Logistic receipts are invalid JSON.") from exc
    if not isinstance(status, dict) or not isinstance(info, dict):
        raise RuntimeError("Frozen Logistic receipts must be JSON objects.")
    if (
        str(status.get("status", "")).upper() != "COMPLETE"
        or status.get("final_holdout_rows_scored") != 0
        or info.get("final_holdout_rows_scored") != 0
        or str(info.get("profile_hash", "")).strip() != PROFILE_HASH
    ):
        raise RuntimeError(
            "Frozen Logistic receipt does not prove COMPLETE status, the active "
            "profile, and an unscored final holdout."
        )
def _xgboost_runner_for_recovery():
    try:
        runner = importlib.import_module("c_modeling.c3_run_xgboost")
        runner._verify_preparation_profile()
        pool = runner._read_pool()
        runner._read_feature_types(pool)
    except Exception as exc:
        raise RuntimeError(
            "Current XGBoost code cannot validate the frozen preparation "
            "contract; no XGBoost output was archived."
        ) from exc
    return runner




def _validated_xgb_reconfirmation_paths(
    run_id: str,
    output_name: str,
    *,
    output_root: Path = Path(PIPELINE_OUTPUT_ROOT),
) -> tuple[Path, Path, Path]:
    """Resolve a completed source run and a new direct-child destination."""


    if (
        not run_id
        or Path(run_id).name != run_id
        or run_id in {".", ".."}
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}", run_id)
    ):
        raise RuntimeError("Unsafe source run ID for XGBoost reconfirmation.")
    if (
        not output_name
        or Path(output_name).name != output_name
        or not re.fullmatch(
            r"xgboost_reconfirmation_[0-9]{8}-[0-9]{6}-[0-9]{6}(?:_[0-9]{2,})?",
            output_name,
        )
    ):
        raise RuntimeError("Unsafe XGBoost reconfirmation output name.")
    output_root = Path(output_root)
    run_root = output_root / run_id
    if run_root.is_symlink() or not run_root.is_dir():
        raise RuntimeError(
            f"Completed source run directory does not exist: {run_root}"
        )
    if run_root.resolve().parent != output_root.resolve():
        raise RuntimeError(
            "XGBoost reconfirmation source must be directly under pipeline_runs."
        )
    canonical = run_root / "xgboost"
    if canonical.is_symlink() or not canonical.is_dir():
        raise RuntimeError(
            f"Canonical XGBoost output is missing or linked: {canonical}"
        )
    if canonical.resolve().parent != run_root.resolve():
        raise RuntimeError(
            "Canonical XGBoost output is not a direct child of its run."
        )
    destination = run_root / output_name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"XGBoost reconfirmation destination already exists: {destination}"
        )
    if destination.resolve().parent != run_root.resolve():
        raise RuntimeError(
            "XGBoost reconfirmation destination must remain inside its source run."
        )
    return run_root, canonical, destination




def _validate_complete_reconfirmation_source(run_root: Path, run_id: str) -> None:
    status_path = run_root / "pipeline_status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Invalid completed source pipeline status: {status_path}"
        ) from exc
    if not isinstance(status, dict):
        raise RuntimeError("Completed source pipeline status must be a JSON object.")
    if (
        str(status.get("run_id", "")).strip() != run_id
        or str(status.get("status", "")).upper() != "COMPLETE"
        or str(status.get("profile_hash", "")).strip() != PROFILE_HASH
    ):
        raise RuntimeError(
            "XGBoost reconfirmation requires a COMPLETE source run matching "
            "the active run ID and tenure profile."
        )




def _regular_file_tree_manifest(directory: Path) -> tuple[tuple[str, int, str], ...]:
    """Return a deterministic, content-bound manifest for a regular file tree."""


    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError(
            f"Protected XGBoost tree is missing, linked, or not a directory: {directory}"
        )
    manifest: list[tuple[str, int, str]] = []
    try:
        entries = sorted(
            directory.rglob("*"),
            key=lambda path: path.relative_to(directory).as_posix(),
        )
        for path in entries:
            relative = path.relative_to(directory).as_posix()
            if path.is_symlink():
                raise RuntimeError(
                    "Protected XGBoost tree contains a symbolic link: "
                    f"{relative}"
                )
            if path.is_dir():
                continue
            if not path.is_file():
                raise RuntimeError(
                    "Protected XGBoost tree contains a non-regular entry: "
                    f"{relative}"
                )
            size = int(path.stat().st_size)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if int(path.stat().st_size) != size:
                raise RuntimeError(
                    "Protected XGBoost file changed while it was being hashed: "
                    f"{relative}"
                )
            manifest.append((relative, size, digest.hexdigest()))
    except OSError as exc:
        raise RuntimeError(
            f"Could not inventory protected XGBoost tree: {directory}"
        ) from exc
    return tuple(manifest)




def _run_xgb_reconfirmation() -> None:
    """Run current XGBoost code against frozen artifacts into a sibling."""


    if not ROOT_ORCHESTRATED:
        raise RuntimeError(
            f"{XGB_RECONFIRMATION_ENV}=1 is accepted only from root "
            "a_business_understanding/a0_one_click.py."
        )
    if FAILED_RUN_RESUME:
        raise RuntimeError(
            "XGBoost reconfirmation and failed-run recovery are mutually exclusive."
        )
    active_state = _read_state()
    active_status = str(active_state.get("status", "")).upper()
    if active_status in {"RUNNING", "PAUSED", "DATA_READY"}:
        raise RuntimeError(
            "An active pipeline controller is still marked "
            f"{active_status}; XGBoost reconfirmation was refused."
        )
    supplied_run_id = os.environ.get("RETENTION_PIPELINE_RUN_ID", "").strip()
    if not XGB_RECONFIRMATION_RUN_ID or supplied_run_id != XGB_RECONFIRMATION_RUN_ID:
        raise RuntimeError(
            "XGBoost reconfirmation source identity does not match the supplied "
            "pipeline run ID."
        )
    run_root, canonical, destination = _validated_xgb_reconfirmation_paths(
        XGB_RECONFIRMATION_RUN_ID,
        XGB_RECONFIRMATION_OUTPUT_NAME,
    )
    _validate_complete_reconfirmation_source(run_root, XGB_RECONFIRMATION_RUN_ID)
    _validate_frozen_logistic_for_stage4_recovery(XGB_RECONFIRMATION_RUN_ID)
    status_path = run_root / "pipeline_status.json"
    if (
        status_path.is_symlink()
        or not status_path.is_file()
        or status_path.stat().st_size <= 0
    ):
        raise RuntimeError(
            f"Protected pipeline status is missing or unsafe: {status_path}"
        )
    protected_status = status_path.read_bytes()
    protected_canonical_manifest = _regular_file_tree_manifest(canonical)
    canonical_manifest_paths = {
        relative for relative, _, _ in protected_canonical_manifest
    }
    required_canonical_paths = {
        "run_info.json",
        "selected_features.json",
        "selected_features.txt",
        "xgb_resolved_model_feature_types.json",
        "feature_selection_summary.md",
        "diagnostics/xgboost_diagnostics_manifest.json",
        "diagnostics/xgboost_business_retention_trend_manifest.json",
        "diagnostics/xgboost_business_retention_trend_full_history.png",
        "diagnostics/xgboost_business_retention_trend_selected_history.png",
        "diagnostics/xgboost_calendar_year_control_one_way.png",
        "diagnostics/xgboost_shap_beeswarm.png",
        "diagnostics/xgboost_shap_plot_manifest.json",
    }
    missing_canonical_paths = sorted(
        required_canonical_paths.difference(canonical_manifest_paths)
    )
    if missing_canonical_paths:
        raise RuntimeError(
            "Protected canonical XGBoost tree is incomplete; missing: "
            + ", ".join(missing_canonical_paths)
        )
    for manifest_name in (
        "xgboost_diagnostics_manifest.json",
        "xgboost_business_retention_trend_manifest.json",
        "xgboost_shap_plot_manifest.json",
    ):
        _require_declared_artifacts(canonical / "diagnostics" / manifest_name)
    runner = _xgboost_runner_for_recovery()
    if str(getattr(runner, "RUN_ID", "")).strip() != XGB_RECONFIRMATION_RUN_ID:
        raise RuntimeError(
            "Imported XGBoost runner does not match the reconfirmation source run."
        )
    expected_preparation = run_root / "preparation"
    if Path(runner.PREPARATION_DIR).resolve() != expected_preparation.resolve():
        raise RuntimeError(
            "Imported XGBoost runner does not point to the frozen preparation."
        )
    if Path(runner.OUTPUT_DIR).resolve() != canonical.resolve():
        raise RuntimeError(
            "Imported XGBoost runner canonical output does not match the source run."
        )
    runner.OUTPUT_DIR = destination
    print("HOME RETENTION 13/1 - XGBOOST-ONLY RECONFIRMATION", flush=True)
    print(f"Source run ID: {XGB_RECONFIRMATION_RUN_ID}", flush=True)
    print(f"Frozen preparation: {expected_preparation}", flush=True)
    print(f"Canonical XGBoost preserved: {canonical}", flush=True)
    print(f"New XGBoost output: {destination}", flush=True)
    try:
        runner.main()
    finally:
        status_changed = (
            status_path.is_symlink()
            or not status_path.is_file()
            or status_path.read_bytes() != protected_status
        )
        try:
            canonical_changed = (
                _regular_file_tree_manifest(canonical)
                != protected_canonical_manifest
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "XGBoost reconfirmation changed or invalidated the protected "
                f"canonical XGBoost tree: {canonical}"
            ) from exc
        if status_changed:
            raise RuntimeError(
                "XGBoost reconfirmation changed the protected source pipeline "
                f"status: {status_path}"
            )
        if canonical_changed:
            raise RuntimeError(
                "XGBoost reconfirmation changed the protected canonical "
                f"XGBoost tree: {canonical}"
            )
    if not runner._can_reuse():
        raise RuntimeError(
            "XGBoost reconfirmation exited without a complete reusable artifact set."
        )
    receipt = {
        "reconfirmation_kind": "COMPLETED_RUN_XGBOOST_ONLY",
        "source_run_id": XGB_RECONFIRMATION_RUN_ID,
        "source_pipeline_status": "COMPLETE",
        "profile_hash": PROFILE_HASH,
        "canonical_xgboost_directory": str(canonical.resolve()),
        "reconfirmation_output_directory": str(destination.resolve()),
        "data_acquisition_rerun": False,
        "preparation_rerun": False,
        "logistic_rerun": False,
        "final_holdout_rows_scored": 0,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    receipt_path = destination / "xgboost_reconfirmation_receipt.json"
    temporary = receipt_path.with_name(receipt_path.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    os.replace(temporary, receipt_path)
    print(f"XGBOOST RECONFIRMATION COMPLETE: {destination}", flush=True)




def main() -> None:
    phase_dir = Path(__file__).resolve().parent
    if MODEL_RESTART_STAGE and (not ROOT_ORCHESTRATED or XGB_RECONFIRMATION or FAILED_RUN_RESUME):
        raise RuntimeError("Canonical model restart requires the normal root controller and cannot be combined with other recovery modes.")
    missing_scripts = [name for name in SCRIPTS if not (phase_dir / name).is_file()]
    if missing_scripts and not XGB_RECONFIRMATION:
        raise FileNotFoundError(
            "Required modeling stage script(s) missing:\n- "
            + "\n- ".join(missing_scripts)
        )
    if XGB_RECONFIRMATION:
        _run_xgb_reconfirmation()
        return
    if FAILED_RUN_RESUME and not ROOT_ORCHESTRATED:
        raise RuntimeError(
            f"{FAILED_RUN_RESUME_ENV}=1 is accepted only from root a0_one_click.py."
        )
    if ROOT_ORCHESTRATED:
        run_id = _run_id()
        started = str(
            _read_state().get(
                "started", datetime.now().isoformat(timespec="seconds")
            )
        )
    else:
        with exclusive_operation_lock(Path(PIPELINE_OPERATION_LOCK_PATH)):
            run_id = _run_id()
            prior = _read_state()
            started = (
                str(prior.get("started"))
                if str(prior.get("status", "")).upper() == "PAUSED"
                and str(prior.get("run_id", "")) == run_id
                and prior.get("started")
                else datetime.now().isoformat(timespec="seconds")
            )
            _write_state(_state_payload(run_id, "RUNNING", started=started))
    print("HOME RETENTION 13/1 - MODEL DEVELOPMENT", flush=True)
    print(f"Profile: {PROFILE_LABEL}", flush=True)
    print(f"Pipeline run ID: {run_id}", flush=True)
    print("Final sealed holdout will be scored: NO", flush=True)
    resume_kind = str(_read_state().get("resume_kind", "")).strip()
    published_selection_resume = (
        FAILED_RUN_RESUME and resume_kind == FAILED_RUN_XGBOOST_RETRY
        and (Path(PIPELINE_OUTPUT_ROOT) / run_id / "xgboost/run_info.json").exists()
    )
    if FAILED_RUN_RESUME:
        if resume_kind == FAILED_RUN_LOGISTIC_RETRY:
            print(
                "Failed-run recovery: frozen acquisition/preparation were "
                "validated by the root controller. Logistic resumes from its "
                "compatible per-fit checkpoint before XGBoost and c4 run.",
                flush=True,
            )
        elif published_selection_resume:
            print(
                "Failed-run recovery: completed Logistic and XGBoost selections "
                "will be verified as frozen recipes. C4 alone will run if valid; "
                "a mismatch stops recovery without archiving either selection.", flush=True,
            )
        else:
            print(
                "Failed-run recovery: preparation and Logistic were validated by "
                "the root controller. A complete Stage-4 XGBoost boundary will be "
                "reused for diagnostics/finalization; otherwise the failed attempt "
                "is archived before XGBoost is retried.",
                flush=True,
            )
    print("Execution order:", flush=True)
    displayed_scripts = (
        SCRIPTS[2:]
        if MODEL_RESTART_STAGE == "xgboost"
        else SCRIPTS[3:]
        if published_selection_resume
        else SCRIPTS[1:]
        if FAILED_RUN_RESUME and resume_kind == FAILED_RUN_LOGISTIC_RETRY
        else SCRIPTS
    )
    for index, script in enumerate(displayed_scripts, start=1):
        print(f"  {index}. {script}", flush=True)
    setup = None
    try:
        setup = _create_setup(run_id)
        child_environment = {
            **profile_env(),
            "RETENTION_PIPELINE_RUN_ID": run_id,
            "RETENTION_ROOT_ORCHESTRATED": (
                "1" if ROOT_ORCHESTRATED else "0"
            ),
            FAILED_RUN_RESUME_ENV: "1" if FAILED_RUN_RESUME else "0",
            FAILED_RUN_RESUME_RUN_ID_ENV: FAILED_RUN_RESUME_RUN_ID,
            FAILED_RUN_RESUME_ATTEMPT_ID_ENV: FAILED_RUN_RESUME_ATTEMPT_ID,
            FAILED_RUN_RESUME_STAGE_ENV: FAILED_RUN_RESUME_STAGE,
            MODEL_RESTART_ENV: MODEL_RESTART_STAGE,
        }
        if MODEL_RESTART_STAGE == "xgboost":
            print("Frozen acquisition, preparation and Logistic retained; running normal c3 and c4 in their canonical output locations.", flush=True)
            run_scripts(scripts=SCRIPTS[2:], path=phase_dir, env=child_environment)
        elif FAILED_RUN_RESUME:
            if resume_kind == FAILED_RUN_LOGISTIC_RETRY:
                print(
                    "Skipping c1 preparation; running c2, c3, and c4 against "
                    "the same frozen run.",
                    flush=True,
                )
                run_scripts(
                    scripts=SCRIPTS[1:],
                    path=phase_dir,
                    env=child_environment,
                )
            else:
                xgboost_runner = _xgboost_runner_for_recovery()
                from c_modeling.completed_selection_recovery import has_completed_selection_receipts
                completed_selection = has_completed_selection_receipts(
                    Path(PIPELINE_OUTPUT_ROOT) / run_id
                )
                post_selection_recovery = (
                    not completed_selection
                    and not xgboost_runner._can_reuse()
                    and xgboost_runner._can_resume_post_selection()
                )
                if completed_selection:
                    _validate_completed_selections_for_c4(run_id, xgboost_runner)
                elif post_selection_recovery:
                    _validate_frozen_logistic_for_stage4_recovery(run_id)
                else:
                    _validate_completed_logistic_for_recovery()
                if completed_selection or xgboost_runner._can_reuse():
                    print(
                        "XGBoost output is already complete and matches the current "
                        "contract; running only outer evaluation/final refit.",
                        flush=True,
                    )
                    run_scripts(
                        scripts=SCRIPTS[3:],
                        path=phase_dir,
                        env=child_environment,
                    )
                elif post_selection_recovery:
                    print(
                        "Validated XGBoost Stage-4 artifacts found; preserving the "
                        "selection results and resuming diagnostics/finalization only.",
                        flush=True,
                    )
                    recovery_environment = {
                        **child_environment,
                        xgboost_runner.POST_SELECTION_RECOVERY_ENV: "1",
                    }
                    run_scripts(
                        scripts=SCRIPTS[2:],
                        path=phase_dir,
                        env=recovery_environment,
                    )
                    if not xgboost_runner._can_reuse():
                        raise RuntimeError(
                            "Post-selection XGBoost recovery exited without a "
                            "complete, reusable artifact set; the original Stage-4 "
                            "artifacts were left in place and the pipeline will "
                            "remain FAILED."
                        )
                else:
                    archived = _archive_xgboost_attempt(
                        run_id,
                        recovery_attempt_id=FAILED_RUN_RESUME_ATTEMPT_ID,
                    )
                    if archived is None:
                        print(
                            "No prior XGBoost directory was present; starting "
                            "XGBoost fresh.",
                            flush=True,
                        )
                    else:
                        print(
                            f"Prior XGBoost attempt preserved at: {archived}",
                            flush=True,
                        )
                    run_scripts(
                        scripts=SCRIPTS[2:],
                        path=phase_dir,
                        env=child_environment,
                    )
                    if not xgboost_runner._can_reuse():
                        raise RuntimeError(
                            "XGBoost exited without producing a complete, reusable "
                            "artifact set; the pipeline will remain FAILED."
                        )
        else:
            run_scripts(
                scripts=SCRIPTS,
                path=phase_dir,
                env=child_environment,
            )
        finalization_status = _validate_finalization_outputs(run_id)
    except subprocess.CalledProcessError as exc:
        status = "PAUSED" if exc.returncode == 75 else "FAILED"
        if setup is not None:
            setup.shutdown()
            setup = None
        if not ROOT_ORCHESTRATED:
            _write_state(
                _state_payload(
                    run_id,
                    status,
                    exit_code=exc.returncode,
                    updated=datetime.now().isoformat(timespec="seconds"),
                )
            )
        if status == "PAUSED":
            print(
                "The Logistic Regression search paused at a safe checkpoint. "
                "Run c0_one_click.py again; it will reuse this run ID.",
                flush=True,
            )
            raise SystemExit(75) from exc
        raise SystemExit(exc.returncode) from exc
    except Exception:
        if setup is not None:
            setup.shutdown()
            setup = None
        if not ROOT_ORCHESTRATED:
            _write_state(
                _state_payload(
                    run_id,
                    "FAILED",
                    updated=datetime.now().isoformat(timespec="seconds"),
                )
            )
        raise
    finally:
        if setup is not None:
            setup.shutdown()
    if not ROOT_ORCHESTRATED:
        _write_state(
            _state_payload(
                run_id,
                "COMPLETE",
                started=started,
                final_holdout_rows_scored=int(
                    finalization_status.get("final_holdout_rows_scored", 0)
                ),
                final_holdout_scoring_requested=False,
                completed=datetime.now().isoformat(timespec="seconds"),
            )
        )
    print(f"MODEL DEVELOPMENT COMPLETE: output/pipeline_runs/{run_id}", flush=True)
def _entrypoint() -> None:
    arguments = sys.argv[1:]
    if any(value in {"-h", "--help"} for value in arguments):
        print(
            "This stage controller is normally launched by root "
            "a_business_understanding/a0_one_click.py. Run `python a_business_understanding/a0_one_click.py --help` from the "
            "project root for supported options."
        )
        return
    if arguments:
        raise SystemExit(
            "c0_one_click.py accepts no direct arguments; use root "
            "a_business_understanding/a0_one_click.py."
        )
    main()
if __name__ == "__main__":
    _entrypoint()
