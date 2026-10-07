# -*- coding: utf-8 -*-
"""Run one profile-scoped Home 13/1 development pipeline.
Examples::
    python a_business_understanding/a0_one_click.py
    python a_business_understanding/a0_one_click.py --tenures 1 3 4+
    python a_business_understanding/a0_one_click.py --tenures 2 --rerun-xgboost-run-id RUN_ID
    python a_business_understanding/a0_one_click.py --tenures 2 --resume-run-id RUN_ID --rerun-preparation
    python a_business_understanding/a0_one_click.py --tenures ALL --resume-run-id RUN_ID --restart-modeling-from xgboost
    python a_business_understanding/a0_one_click.py --renewal
    python a_business_understanding/a0_one_click.py --new-business
    python a_business_understanding/a0_one_click.py --tenure 4+
    python a_business_understanding/a0_one_click.py --tenure 7
    python a_business_understanding/a0_one_click.py --tenures 1-3
Use ``--refresh-source`` only when the shared physical all-tenure Snowflake
table must be rebuilt.  Repeat the identical profile command after a safe
Logistic pause.
"""
from __future__ import annotations
import argparse
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from func.modeling import run_scripts  # noqa: E402
from func.operation_lock import exclusive_operation_lock  # noqa: E402
from func.tenure_profile import (  # noqa: E402
    BUSINESS_SCOPE_ENV,
    LEGACY_TENURE_ENV_VARS,
    PROFILE_HASH_ENV,
    PROFILE_PAYLOAD_ENV,
    PROFILE_SLUG_ENV,
    TENURE_FILTER_MODE_ENV,
    TENURE_GROUPS_ENV,
    TenureProfile,
    profile_from_values,
)
STAGES = [
    PROJECT_ROOT / "b_data_acquisition" / "b0_one_click.py",
    PROJECT_ROOT / "c_modeling" / "c0_one_click.py",
]
STATE_PATH = PROJECT_ROOT / "output" / "pipeline_state" / "active_run.json"
OPERATION_LOCK_PATH = PROJECT_ROOT / "output" / "pipeline_state" / "operation.lock"
PIPELINE_OUTPUT_ROOT = PROJECT_ROOT / "output" / "pipeline_runs"
# Easy VS Code default.  Edit only this value when clicking Run Python File.
# Accepted examples: "0", "4+", "7", "1-7", "1,3,10", or "ALL".
# Any whole-number selector >=4 resolves to the complete top-coded 4+ group.
# Explicit command-line selectors always override it.
DEFAULT_TENURE_FILTER = "ALL"
TEST_POLICY_ID_ENV = "RETENTION_TEST_POLICY_ID"
SAMPLE_PERCENT_ENV = "RETENTION_SAMPLE_PERCENT"
FAILED_RUN_RESUME_ENV = "RETENTION_RESUME_FAILED_RUN"
FAILED_RUN_RESUME_RUN_ID_ENV = "RETENTION_RESUME_FAILED_RUN_ID"
FAILED_RUN_RESUME_ATTEMPT_ID_ENV = "RETENTION_RESUME_ATTEMPT_ID"
FAILED_RUN_RESUME_STAGE_ENV = "RETENTION_RESUME_FAILED_STAGE"
FAILED_RUN_LOGISTIC_RETRY = "FAILED_RUN_LOGISTIC_RETRY"
FAILED_RUN_XGBOOST_RETRY = "FAILED_RUN_XGBOOST_RETRY"
XGB_RECONFIRMATION_ENV = "RETENTION_XGB_RECONFIRMATION"
XGB_RECONFIRMATION_RUN_ID_ENV = "RETENTION_XGB_RECONFIRMATION_RUN_ID"
XGB_RECONFIRMATION_OUTPUT_NAME_ENV = "RETENTION_XGB_RECONFIRMATION_OUTPUT_NAME"
LEGACY_INNER_SELECTION_OUTER_FOLD_ENV = "RETENTION_INNER_SELECTION_OUTER_FOLD"
MODEL_RESTART_ENV = "RETENTION_CANONICAL_MODEL_RESTART"
MODEL_RESTART_KIND = "CANONICAL_XGBOOST_RESTART"
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
def _sample_percent(value: str) -> str:
    try:
        percent = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(
            "sample percent must be a number such as 2"
        ) from exc
    if not percent.is_finite() or percent < Decimal("0.01") or percent >= 100:
        raise argparse.ArgumentTypeError(
            "sample percent must be at least 0.01 and less than 100"
        )
    quantized = percent.quantize(Decimal("0.01"))
    if quantized != percent:
        raise argparse.ArgumentTypeError(
            "sample percent supports at most two decimal places"
        )
    return format(quantized, "f")
def _test_policy(value: str) -> str:
    policy_id = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", policy_id):
        raise argparse.ArgumentTypeError(
            "test policy must contain 1-128 letters, digits, '.', '_', or '-'"
        )
    return policy_id
def _resume_run_id(value: str) -> str:
    run_id = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}", run_id):
        raise argparse.ArgumentTypeError(
            "resume run ID must contain 1-192 letters, digits, '.', '_', or '-'"
        )
    return run_id




def _rerun_xgboost_run_id(value: str) -> str:
    try:
        return _resume_run_id(value)
    except argparse.ArgumentTypeError as exc:
        raise argparse.ArgumentTypeError(
            "XGBoost rerun ID must contain 1-192 letters, digits, '.', '_', or '-'"
        ) from exc
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    business_scope_group = parser.add_mutually_exclusive_group()
    business_scope_group.add_argument(
        "--new-business",
        action="store_true",
        help="Select new-business scope (group 0).",
    )
    business_scope_group.add_argument(
        "--renewal",
        action="store_true",
        help="Select renewal scope (groups 1, 2, 3, and top-coded 4+).",
    )
    business_scope_group.add_argument(
        "--all",
        action="store_true",
        help="Select all business scopes.",
    )
    parser.add_argument(
        "--tenure",
        default=None,
        help=(
            "One anchor tenure selector. Whole numbers 4 or greater (for "
            "example 7 or 10) resolve to the complete top-coded 4+ group."
        ),
    )
    parser.add_argument(
        "--tenures",
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "Selected anchor tenure values/groups. Examples: --tenures 1-3, "
            "--tenures 7, --tenures 1,3,10, or --tenures ALL. Every whole "
            "number >=4 resolves to 4+, which selects all actual tenures of "
            "four years or more. The filter runs in Snowflake before download."
        ),
    )
    parser.add_argument(
        "--tenure-min",
        default=None,
        help=(
            "Inclusive lower selector; omit max for the remaining finite "
            "groups through 4+. Values >=4 resolve to 4+."
        ),
    )
    parser.add_argument(
        "--tenure-max",
        default=None,
        help="Inclusive upper selector; whole numbers >=4 resolve to 4+.",
    )
    parser.add_argument(
        "--refresh-source",
        action="store_true",
        help="Refresh/QA the existing CY table, then run the selected model profile.",
    )
    parser.add_argument(
        "--refresh-source-only",
        action="store_true",
        help="Refresh/QA the existing CY table and stop before download/modeling.",
    )
    parser.add_argument(
        "--refresh-preflight-only",
        action="store_true",
        help=(
            "Run every live source/CY candidate check and write a receipt, "
            "but do not replace the existing table."
        ),
    )
    parser.add_argument(
        "--include-exposure-year-predictor",
        action="store_true",
        help=(
            "Allow EXPOSURE_CALENDAR_YEAR to compete as a discrete challenger "
            "feature. It remains QA-only by default."
        ),
    )
    parser.add_argument(
        "--sql-dry-run",
        action="store_true",
        help="Print the tenure-filtered download SELECT without connecting.",
    )
    test_scope = parser.add_mutually_exclusive_group()
    test_scope.add_argument(
        "--test-policy",
        type=_test_policy,
        default=None,
        metavar="CONTRACT",
        help=(
            "Engineering smoke test for one adjusted policy contract. Use with "
            "--acquisition-only; one policy is not sufficient for model folds."
        ),
    )
    test_scope.add_argument(
        "--sample-percent",
        type=_sample_percent,
        default=None,
        metavar="PCT",
        help=(
            "Deterministically sample this percent of policy contracts before "
            "download (for example 2). All selected terms for sampled policies "
            "are retained."
        ),
    )
    parser.add_argument(
        "--acquisition-only",
        action="store_true",
        help=(
            "Run Snowflake QA/download and input-contract preparation, then stop "
            "before Logistic/XGBoost. Recommended with --test-policy."
        ),
    )
    parser.add_argument(
        "--recover-stale-run",
        action="store_true",
        help=(
            "Replace a persisted RUNNING marker with a recoverable FAILED "
            "receipt before starting a fresh run. Use only after confirming "
            "that the previous project Python process has stopped."
        ),
    )
    parser.add_argument(
        "--resume-run-id",
        type=_resume_run_id,
        default=None,
        metavar="RUN_ID",
        help=(
            "Resume one validated FAILED run without repeating acquisition or "
            "preparation. An incomplete Logistic search resumes its compatible "
            "fit checkpoints; a Logistic-complete run resumes XGBoost or "
            "finalization at the latest provable boundary."
        ),
    )
    parser.add_argument(
        "--rerun-preparation",
        action="store_true",
        help=(
            "Use only with --resume-run-id when a FAILED run's preparation "
            "contract must be rebuilt. The frozen download/model-input files "
            "are verified and reused; prior preparation, Logistic, XGBoost, "
            "and finalization directories are archived before c1 through c4 rerun."
        ),
    )
    parser.add_argument(
        "--restart-modeling-from",
        choices=("xgboost",),
        help=(
            "With --resume-run-id and an explicit original profile, archive the "
            "prior XGBoost and dependent evaluations, then run normal XGBoost "
            "selection and outer evaluation in their canonical locations. Reuse "
            "verified acquisition, preparation and completed Logistic. Accepts "
            "COMPLETE or FAILED runs; repeating after failure starts XGBoost fresh."
        ),
    )
    parser.add_argument(
        "--rerun-xgboost-run-id",
        type=_rerun_xgboost_run_id,
        default=None,
        metavar="RUN_ID",
        help=(
            "Rerun only XGBoost for one validated COMPLETE run, using that "
            "run's frozen input and preparation artifacts. Logistic and data "
            "acquisition are skipped. The canonical xgboost directory and "
            "pipeline status are preserved; results are written to a sibling "
            "xgboost_reconfirmation_<timestamp> directory. Supply the original "
            "tenure/profile selector (for example --tenures 2)."
        ),
    )
    args = parser.parse_args(argv)
    # Map convenience flags to business_scope
    if args.new_business:
        args.business_scope = "new-business"
    elif args.renewal:
        args.business_scope = "renewal"
    elif args.all:
        args.business_scope = "all"
    else:
        args.business_scope = None
    profile_selector_supplied = any(
        value is not None
        for value in (
            args.business_scope,
            args.tenure,
            args.tenures,
            args.tenure_min,
            args.tenure_max,
        )
    )
    try:
        default_selection = (
            DEFAULT_TENURE_FILTER
            if not profile_selector_supplied
            else None
        )
        args.profile = profile_from_values(
            business_scope=args.business_scope,
            tenure=args.tenure,
            tenure_values=(
                args.tenures if args.tenures is not None else default_selection
            ),
            tenure_min=args.tenure_min,
            tenure_max=args.tenure_max,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if sum(
        bool(value)
        for value in (
            args.refresh_source,
            args.refresh_source_only,
            args.refresh_preflight_only,
        )
    ) > 1:
        parser.error(
            "Choose only one of --refresh-source, --refresh-source-only, or "
            "--refresh-preflight-only."
        )
    if (
        args.refresh_source
        or args.refresh_source_only
        or args.refresh_preflight_only
    ) and args.sql_dry_run:
        parser.error("Source refresh cannot be combined with --sql-dry-run.")
    if (
        args.refresh_source
        or args.refresh_source_only
        or args.refresh_preflight_only
    ) and (args.test_policy is not None or args.sample_percent is not None):
        parser.error(
            "Policy/sample test scope cannot be combined with any shared-CY "
            "refresh flag. The test scope applies only to extraction."
        )
    if (args.refresh_source_only or args.refresh_preflight_only) and args.acquisition_only:
        parser.error(
            "Refresh-only/preflight-only operates on the complete shared CY and "
            "cannot be combined with --acquisition-only."
        )
    if args.test_policy is not None and not (
        args.acquisition_only or args.sql_dry_run
    ):
        parser.error(
            "--test-policy is an acquisition smoke test; add "
            "--acquisition-only (or --sql-dry-run)."
        )
    if args.sample_percent is not None and not (
        args.acquisition_only or args.sql_dry_run
    ):
        parser.error(
            "--sample-percent is an acquisition smoke test; add "
            "--acquisition-only (or --sql-dry-run). Full outer evaluation and "
            "deployment require the complete selected population."
        )
    if args.resume_run_id is not None:
        incompatible = []
        for enabled, flag in (
            (args.recover_stale_run, "--recover-stale-run"),
            (args.refresh_source, "--refresh-source"),
            (args.refresh_source_only, "--refresh-source-only"),
            (args.refresh_preflight_only, "--refresh-preflight-only"),
            (args.include_exposure_year_predictor, "--include-exposure-year-predictor"),
            (args.sql_dry_run, "--sql-dry-run"),
            (args.acquisition_only, "--acquisition-only"),
            (args.test_policy is not None, "--test-policy"),
            (args.sample_percent is not None, "--sample-percent"),
        ):
            if enabled:
                incompatible.append(flag)
        if incompatible:
            parser.error(
                "--resume-run-id cannot be combined with "
                + ", ".join(incompatible)
                + ". A failed-run resume must reuse its original inputs and profile."
            )
    elif args.rerun_preparation:
        parser.error("--rerun-preparation requires --resume-run-id RUN_ID.")
    if args.restart_modeling_from is not None:
        if args.resume_run_id is None or not profile_selector_supplied:
            parser.error(
                "--restart-modeling-from requires --resume-run-id RUN_ID and "
                "the original explicit tenure/profile selector."
            )
        if args.rerun_preparation or args.rerun_xgboost_run_id is not None:
            parser.error(
                "--restart-modeling-from cannot be combined with "
                "--rerun-preparation or --rerun-xgboost-run-id."
            )
    if args.rerun_xgboost_run_id is not None:
        if not profile_selector_supplied:
            parser.error(
                "--rerun-xgboost-run-id requires the completed run's original "
                "profile selector (for example --tenures 2)."
            )
        incompatible = []
        for enabled, flag in (
            (args.resume_run_id is not None, "--resume-run-id"),
            (args.rerun_preparation, "--rerun-preparation"),
            (args.recover_stale_run, "--recover-stale-run"),
            (args.refresh_source, "--refresh-source"),
            (args.refresh_source_only, "--refresh-source-only"),
            (args.refresh_preflight_only, "--refresh-preflight-only"),
            (args.include_exposure_year_predictor, "--include-exposure-year-predictor"),
            (args.sql_dry_run, "--sql-dry-run"),
            (args.acquisition_only, "--acquisition-only"),
            (args.test_policy is not None, "--test-policy"),
            (args.sample_percent is not None, "--sample-percent"),
        ):
            if enabled:
                incompatible.append(flag)
        if incompatible:
            parser.error(
                "--rerun-xgboost-run-id cannot be combined with "
                + ", ".join(incompatible)
                + ". Reconfirmation must reuse the completed run's frozen inputs."
            )
    return args
def _activate_profile(
    profile: TenureProfile,
    *,
    include_exposure_year_predictor: bool,
    test_policy: str | None,
    sample_percent: str | None,
):
    os.environ.pop("RETENTION_PIPELINE_RUN_ID", None)
    os.environ.pop("RETENTION_ROOT_ORCHESTRATED", None)
    os.environ.pop(MODEL_RESTART_ENV, None)
    for name in LEGACY_TENURE_ENV_VARS:
        os.environ.pop(name, None)
    os.environ.pop(TEST_POLICY_ID_ENV, None)
    os.environ.pop(SAMPLE_PERCENT_ENV, None)
    os.environ.pop(FAILED_RUN_RESUME_ENV, None)
    os.environ.pop(FAILED_RUN_RESUME_RUN_ID_ENV, None)
    os.environ.pop(FAILED_RUN_RESUME_ATTEMPT_ID_ENV, None)
    os.environ.pop(FAILED_RUN_RESUME_STAGE_ENV, None)
    os.environ.pop(XGB_RECONFIRMATION_ENV, None)
    os.environ.pop(XGB_RECONFIRMATION_RUN_ID_ENV, None)
    os.environ.pop(XGB_RECONFIRMATION_OUTPUT_NAME_ENV, None)
    os.environ.pop("RETENTION_SCORE_FINAL_HOLDOUT", None)
    # This legacy selector once allowed preparation to disagree with the two
    # model engines. Inner feature discovery is now governed in params.py.
    os.environ.pop(LEGACY_INNER_SELECTION_OUTER_FOLD_ENV, None)
    os.environ[BUSINESS_SCOPE_ENV] = profile.business_scope
    os.environ[TENURE_GROUPS_ENV] = ",".join(profile.tenure_groups)
    os.environ[TENURE_FILTER_MODE_ENV] = profile.filter_mode
    os.environ[PROFILE_SLUG_ENV] = profile.slug
    os.environ[PROFILE_PAYLOAD_ENV] = json.dumps(
        profile.payload, sort_keys=True, separators=(",", ":")
    )
    os.environ.pop(PROFILE_HASH_ENV, None)
    os.environ["RETENTION_INCLUDE_EXPOSURE_YEAR_AS_PREDICTOR"] = (
        "1" if include_exposure_year_predictor else "0"
    )
    if test_policy is not None:
        os.environ[TEST_POLICY_ID_ENV] = str(test_policy).strip()
    if sample_percent is not None:
        os.environ[SAMPLE_PERCENT_ENV] = str(sample_percent)
    import params
    params = importlib.reload(params)
    environment = params.profile_env()
    os.environ.update(environment)
    return params, environment
def _read_state() -> dict[str, Any]:
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
def _write_state(payload: dict[str, Any]) -> None:
    destinations = [STATE_PATH]
    run_id = str(payload.get("run_id", "")).strip()
    if run_id:
        destinations.append(PIPELINE_OUTPUT_ROOT / run_id / "pipeline_status.json")
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, destination)
def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload




def _require_declared_artifacts(
    run_root: Path,
    manifest_relative_path: str,
) -> None:
    """Require every regular file declared by one diagnostic manifest."""


    manifest_path = run_root / manifest_relative_path
    payload = _read_json_object(manifest_path, manifest_relative_path)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError(
            f"Diagnostic manifest has no nonempty artifacts list: {manifest_path}"
        )
    base = manifest_path.parent
    invalid: list[str] = []
    for value in artifacts:
        if isinstance(value, dict):
            name = str(value.get("path") or value.get("filename") or "").strip()
        else:
            name = str(value).strip()
        declared = Path(name)
        candidate = declared if declared.is_absolute() else base / declared
        try:
            within_manifest_tree = candidate.resolve().is_relative_to(
                base.resolve()
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




def _require_model_diagnostics(run_root: Path, model_family: str) -> None:
    family = str(model_family).strip().lower()
    manifests = [
        f"{family}/diagnostics/{family}_diagnostics_manifest.json",
        f"{family}/diagnostics/{family}_business_retention_trend_manifest.json",
    ]
    if family == "xgboost":
        manifests.append(
            "xgboost/diagnostics/xgboost_shap_plot_manifest.json"
        )
    for relative in manifests:
        _require_declared_artifacts(run_root, relative)




def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()




def _validate_failed_run_for_preparation_rebuild(
    *,
    run_id: str,
    profile_hash: str,
    profile_payload: dict[str, Any],
    output_root: Path = PIPELINE_OUTPUT_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prove that a FAILED run's immutable acquisition can be reused."""


    try:
        _resume_run_id(run_id)
    except argparse.ArgumentTypeError as exc:
        raise RuntimeError(str(exc)) from exc
    output_root = Path(output_root)
    run_root = output_root / run_id
    if run_root.is_symlink() or not run_root.is_dir():
        raise RuntimeError(f"Preparation-rebuild run does not exist: {run_root}")
    if run_root.resolve().parent != output_root.resolve():
        raise RuntimeError(
            "Preparation-rebuild run must be directly under pipeline_runs."
        )


    paths = {
        "pipeline_status": run_root / "pipeline_status.json",
        "input": run_root / "input_snapshot" / "retention_13_1_population.parquet",
        "input_metadata": (
            run_root
            / "input_snapshot"
            / "retention_13_1_population.metadata.json"
        ),
        "feature_contract": run_root / "model_inputs" / "feature_contract.csv",
        "candidate_pool": run_root / "model_inputs" / "candidate_pool.txt",
        "preparation_info": run_root / "preparation" / "preparation_info.json",
    }
    invalid = []
    for label, path in paths.items():
        try:
            valid = (
                not path.is_symlink()
                and path.is_file()
                and path.stat().st_size > 0
            )
        except OSError:
            valid = False
        if not valid:
            invalid.append(f"{label}: {path.relative_to(run_root)}")
    if invalid:
        raise RuntimeError(
            "Preparation rebuild requires complete regular acquisition and "
            "prior-preparation receipts. Missing, empty, or linked:\n- "
            + "\n- ".join(invalid)
        )


    status = _read_json_object(paths["pipeline_status"], "pipeline status")
    if str(status.get("run_id", "")).strip() != run_id:
        raise RuntimeError("Preparation-rebuild run ID does not match its status.")
    if str(status.get("status", "")).strip().upper() != "FAILED":
        raise RuntimeError("--rerun-preparation accepts only a FAILED run.")
    if str(status.get("profile_hash", "")).strip() != profile_hash:
        raise RuntimeError(
            "Preparation-rebuild run does not match the requested tenure profile."
        )
    status_profile = status.get("profile")
    if status_profile is not None and status_profile != profile_payload:
        raise RuntimeError(
            "Preparation-rebuild status profile payload does not match the request."
        )


    preparation = _read_json_object(
        paths["preparation_info"], "preparation info"
    )
    if (
        str(preparation.get("profile_hash", "")).strip() != profile_hash
        or preparation.get("profile") != profile_payload
        or preparation.get("final_holdout_rows_scored") != 0
    ):
        raise RuntimeError(
            "Prior preparation does not prove the requested profile and a sealed "
            "final holdout."
        )
    for label, receipt_key in (
        ("input", "input_sha256"),
        ("input_metadata", "input_metadata_sha256"),
        ("feature_contract", "feature_contract_sha256"),
    ):
        expected = str(preparation.get(receipt_key, "")).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RuntimeError(
                f"Prior preparation has no valid {receipt_key} receipt."
            )
        actual = _sha256_file(paths[label])
        if actual != expected:
            raise RuntimeError(
                f"Frozen {label} content no longer matches prior preparation; "
                "modeling rebuild was refused."
            )
    return status, preparation




def _archive_modeling_outputs_for_preparation_rebuild(
    *,
    run_id: str,
    recovery_attempt_id: str,
    output_root: Path = PIPELINE_OUTPUT_ROOT,
) -> Path:
    """Archive stale modeling outputs without deleting or overwriting them."""


    output_root = Path(output_root)
    run_root = output_root / run_id
    if run_root.is_symlink() or run_root.resolve().parent != output_root.resolve():
        raise RuntimeError("Unsafe preparation-rebuild run directory.")
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9]{6}", recovery_attempt_id):
        raise RuntimeError("Unsafe preparation-rebuild attempt ID.")


    sources: list[Path] = []
    for name in (
        "preparation",
        "logistic",
        "xgboost",
        "outer_evaluation",
        "deployment",
    ):
        source = run_root / name
        if not source.exists():
            continue
        if (
            source.is_symlink()
            or not source.is_dir()
            or source.resolve().parent != run_root.resolve()
        ):
            raise RuntimeError(f"Unsafe modeling output cannot be archived: {source}")
        sources.append(source)
    if run_root / "preparation" not in sources:
        raise RuntimeError("No prior preparation directory is available to rebuild.")


    archive = run_root / f"modeling_rebuild_archive_{recovery_attempt_id}"
    if archive.exists() or archive.is_symlink():
        raise FileExistsError(f"Preparation-rebuild archive exists: {archive}")
    archive.mkdir()
    moved: list[str] = []
    receipt_path = archive / "modeling_rebuild_receipt.json"
    try:
        for source in sources:
            source.rename(archive / source.name)
            moved.append(source.name)
        receipt = {
            "status": "COMPLETE",
            "run_id": run_id,
            "archive_reason": "FAILED_RUN_PREPARATION_REBUILD",
            "recovery_attempt_id": recovery_attempt_id,
            "archived_directories": moved,
            "archived_at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        receipt = {
            "status": "INCOMPLETE",
            "run_id": run_id,
            "archive_reason": "FAILED_RUN_PREPARATION_REBUILD",
            "recovery_attempt_id": recovery_attempt_id,
            "archived_directories": moved,
            "error": f"{type(exc).__name__}: {exc}",
            "archived_at": datetime.now().isoformat(timespec="seconds"),
        }
        temporary = receipt_path.with_name(receipt_path.name + ".tmp")
        temporary.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        os.replace(temporary, receipt_path)
        raise
    temporary = receipt_path.with_name(receipt_path.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    os.replace(temporary, receipt_path)
    return archive




def _validate_failed_run_resume(
    *,
    run_id: str,
    profile_hash: str,
    profile_payload: dict[str, Any],
    output_root: Path = PIPELINE_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Classify and validate a FAILED modeling run at its latest safe stage.


    Acquisition and preparation are immutable for both recovery paths.  A
    completed Logistic receipt continues to use the existing XGBoost recovery
    contract.  Otherwise, a durable Logistic fit checkpoint can be resumed in
    place after its saved statistical identities and sealed holdout are
    verified.
    """


    try:
        _resume_run_id(run_id)
    except argparse.ArgumentTypeError as exc:
        raise RuntimeError(str(exc)) from exc
    output_root = Path(output_root)
    run_root = output_root / run_id
    if run_root.is_symlink() or not run_root.is_dir():
        raise RuntimeError(f"Resume run directory does not exist: {run_root}")
    if run_root.resolve().parent != output_root.resolve():
        raise RuntimeError("Resume run directory must be directly under pipeline_runs.")
    pipeline_status, preparation_info = _validate_failed_run_for_preparation_rebuild(
        run_id=run_id,
        profile_hash=profile_hash,
        profile_payload=profile_payload,
        output_root=output_root,
    )
    if not isinstance(preparation_info.get("profile"), dict):
        raise RuntimeError("Preparation profile payload is missing or invalid.")


    preparation_required = (
        "preparation/quality_screened_logistic_candidate_pool.txt",
        "preparation/logistic_feature_types.json",
        "preparation/quality_screened_xgboost_candidate_pool.txt",
        "preparation/xgboost_feature_types.json",
        "preparation/feature_routing_audit.csv",
        "preparation/outer_fold_definitions.csv",
        "preparation/inner_fold_definitions.csv",
    )
    missing = []
    for relative in preparation_required:
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
            missing.append(relative)
    if missing:
        raise RuntimeError(
            "Failed-run resume requires complete preparation artifacts. "
            "Missing, empty, or linked:\n- " + "\n- ".join(missing)
        )


    logistic_status_path = run_root / "logistic" / "run_status.json"
    logistic_status: dict[str, Any] = {}
    if logistic_status_path.is_file():
        logistic_status = _read_json_object(
            logistic_status_path, "Logistic run status"
        )
    if str(logistic_status.get("status", "")).upper() != "COMPLETE":
        _validate_incomplete_logistic_checkpoint(
            run_root=run_root,
            profile_hash=profile_hash,
            preparation_info=preparation_info,
        )
        return {
            **pipeline_status,
            "validated_resume_kind": FAILED_RUN_LOGISTIC_RETRY,
        }


    logistic_required = (
        "logistic/run_status.json",
        "logistic/run_info.json",
        "logistic/recommended_model.json",
        "logistic/recommended_features.txt",
        "logistic/resolved_logistic_model_feature_types.json",
        "logistic/diagnostics/logistic_diagnostics_manifest.json",
        "logistic/diagnostics/logistic_calendar_year_performance.csv",
        "logistic/diagnostics/logistic_calendar_year_one_way_summary.csv",
        "logistic/diagnostics/logistic_selected_feature_one_way_summaries.csv",
        "logistic/diagnostics/logistic_policy_start_year_month_trend.csv",
        "logistic/diagnostics/logistic_policy_start_year_month_trend.png",
        "logistic/diagnostics/logistic_calendar_year_control_one_way.png",
        "logistic/diagnostics/logistic_business_retention_trend_manifest.json",
        "logistic/diagnostics/logistic_business_retention_trend_full_history.png",
        "logistic/diagnostics/logistic_business_retention_trend_selected_history.png",
    )
    missing = []
    for relative in logistic_required:
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
            missing.append(relative)
    if missing:
        raise RuntimeError(
            "XGBoost recovery requires complete Logistic artifacts. Missing, "
            "empty, or linked:\n- " + "\n- ".join(missing)
        )
    _require_model_diagnostics(run_root, "logistic")
    logistic_info = _read_json_object(
        run_root / "logistic" / "run_info.json", "Logistic run info"
    )
    for label, payload in (
        ("preparation", preparation_info),
        ("Logistic", logistic_info),
    ):
        if str(payload.get("profile_hash", "")).strip() != profile_hash:
            raise RuntimeError(
                f"{label} profile hash does not match the requested profile."
            )
    if logistic_status.get("final_holdout_rows_scored") != 0:
        raise RuntimeError("Logistic run status does not confirm a sealed holdout.")
    if logistic_info.get("final_holdout_rows_scored") != 0:
        raise RuntimeError("Logistic run info does not confirm a sealed holdout.")
    return {
        **pipeline_status,
        "validated_resume_kind": FAILED_RUN_XGBOOST_RETRY,
    }




def _validate_incomplete_logistic_checkpoint(
    *,
    run_root: Path,
    profile_hash: str,
    preparation_info: dict[str, Any],
) -> None:
    """Prove that an interrupted Logistic search is safe to resume in place."""


    logistic_dir = run_root / "logistic"
    if logistic_dir.is_symlink() or not logistic_dir.is_dir():
        raise RuntimeError(
            "The FAILED run has no regular Logistic output directory to resume."
        )
    downstream = [
        name
        for name in ("xgboost", "outer_evaluation", "deployment")
        if (run_root / name).exists()
    ]
    if downstream:
        raise RuntimeError(
            "Incomplete Logistic output is inconsistent with downstream model "
            "artifacts; recovery was refused: " + ", ".join(downstream)
        )


    required = {
        "config": logistic_dir / "run_config.json",
        "metric_contract": logistic_dir / "metric_contract.json",
        "history_identity": logistic_dir / "history_population_identity.json",
        "holdout_seal": logistic_dir / "holdout_seal.json",
        "resolved_types": (
            logistic_dir / "resolved_logistic_model_feature_types.json"
        ),
        "inner_folds": logistic_dir / "inner_fold_definitions.csv",
        "checkpoint": logistic_dir / ".fit_checkpoint.sqlite3",
    }
    invalid = []
    for label, path in required.items():
        try:
            valid = (
                not path.is_symlink()
                and path.is_file()
                and path.stat().st_size > 0
            )
        except OSError:
            valid = False
        if not valid:
            invalid.append(f"{label}: {path.relative_to(run_root)}")
    if invalid:
        raise RuntimeError(
            "Incomplete Logistic recovery requires its durable checkpoint "
            "contract. Missing, empty, or linked:\n- " + "\n- ".join(invalid)
        )


    config = _read_json_object(required["config"], "Logistic run config")
    if (
        str(config.get("profile_hash", "")).strip() != profile_hash
        or config.get("profile") != preparation_info.get("profile")
        or config.get("final_holdout_scored") is not False
    ):
        raise RuntimeError(
            "Incomplete Logistic config does not match the prepared profile and "
            "sealed-holdout contract."
        )
    invocation_keys = {
        "invocation_runtime_budget_hours",
        "invocation_max_new_fits",
        "invocation_max_actions_safety_cap",
    }
    immutable = {
        key: value
        for key, value in config.items()
        if key != "config_fingerprint" and key not in invocation_keys
    }
    actual_fingerprint = hashlib.sha256(
        json.dumps(immutable, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if config.get("config_fingerprint") != actual_fingerprint:
        raise RuntimeError("Saved Logistic run-config fingerprint is invalid.")


    identity_contract = {
        "data": (
            run_root / "input_snapshot" / "retention_13_1_population.parquet",
            "input_sha256",
        ),
        "manifest": (
            run_root / "model_inputs" / "feature_contract.csv",
            "feature_contract_sha256",
        ),
        "pool_file": (
            run_root
            / "preparation"
            / "quality_screened_logistic_candidate_pool.txt",
            None,
        ),
        "feature_types_file": (
            run_root / "preparation" / "logistic_feature_types.json",
            None,
        ),
        "routing_audit": (
            run_root / "preparation" / "feature_routing_audit.csv",
            None,
        ),
        "preparation_info": (
            run_root / "preparation" / "preparation_info.json",
            None,
        ),
        "outer_fold_definitions": (
            run_root / "preparation" / "outer_fold_definitions.csv",
            None,
        ),
        "inner_fold_definitions": (
            run_root / "preparation" / "inner_fold_definitions.csv",
            None,
        ),
    }
    for key, (path, preparation_hash_key) in identity_contract.items():
        identity = config.get(key)
        expected = (
            str(identity.get("sha256", "")).strip().lower()
            if isinstance(identity, dict)
            else ""
        )
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RuntimeError(f"Logistic config has no valid {key} SHA-256.")
        if preparation_hash_key is not None:
            prepared_hash = str(
                preparation_info.get(preparation_hash_key, "")
            ).strip().lower()
            if expected != prepared_hash:
                raise RuntimeError(
                    f"Logistic {key} identity differs from preparation."
                )
        elif _sha256_file(path) != expected:
            raise RuntimeError(
                f"Logistic checkpoint input changed after it was fitted: {key}."
            )


    history = _read_json_object(
        required["history_identity"], "Logistic history-population identity"
    )
    history_digest = str(history.get("identity_hash", "")).strip().lower()
    history_payload = {
        key: value for key, value in history.items() if key != "identity_hash"
    }
    if history_digest != hashlib.sha256(
        json.dumps(history_payload, sort_keys=True).encode("utf-8")
    ).hexdigest():
        raise RuntimeError("Logistic history-population identity is invalid.")


    seal = _read_json_object(required["holdout_seal"], "Logistic holdout seal")
    if (
        seal.get("predictions_created") is not False
        or seal.get("target_metrics_created") is not False
        or int(seal.get("row_count", 0)) <= 0
    ):
        raise RuntimeError(
            "Incomplete Logistic output does not prove an untouched final holdout."
        )


    checkpoint_uri = required["checkpoint"].resolve().as_uri() + "?mode=ro"
    try:
        with sqlite3.connect(checkpoint_uri, uri=True, timeout=30.0) as connection:
            quick_check = connection.execute("PRAGMA quick_check").fetchone()
            schema = connection.execute(
                "SELECT value FROM checkpoint_metadata WHERE key='schema_version'"
            ).fetchone()
            completed = connection.execute(
                "SELECT COUNT(*) FROM fit_results WHERE status='completed'"
            ).fetchone()
    except sqlite3.Error as exc:
        raise RuntimeError("Logistic fit checkpoint is unreadable or invalid.") from exc
    if (
        quick_check is None
        or str(quick_check[0]).lower() != "ok"
        or schema is None
        or str(schema[0]) != "1"
        or completed is None
        or int(completed[0]) < 1
    ):
        raise RuntimeError(
            "Logistic fit checkpoint failed integrity, schema, or completed-fit checks."
        )




def _validate_completed_run_for_xgb_reconfirmation(
    *,
    run_id: str,
    profile_hash: str,
    profile_payload: dict[str, Any],
    output_root: Path = PIPELINE_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Prove that a completed run can be reused without changing it."""


    try:
        _rerun_xgboost_run_id(run_id)
    except argparse.ArgumentTypeError as exc:
        raise RuntimeError(str(exc)) from exc
    output_root = Path(output_root)
    run_root = output_root / run_id
    if run_root.is_symlink() or not run_root.is_dir():
        raise RuntimeError(
            f"Completed XGBoost rerun directory does not exist: {run_root}"
        )
    if run_root.resolve().parent != output_root.resolve():
        raise RuntimeError(
            "Completed XGBoost rerun directory must be directly under pipeline_runs."
        )
    canonical_xgboost = run_root / "xgboost"
    if canonical_xgboost.is_symlink() or not canonical_xgboost.is_dir():
        raise RuntimeError(
            "A completed, regular canonical xgboost directory is required for "
            "reconfirmation."
        )
    if canonical_xgboost.resolve().parent != run_root.resolve():
        raise RuntimeError(
            "Canonical XGBoost output must be a direct child of the completed run."
        )
    required = (
        "pipeline_status.json",
        "input_snapshot/retention_13_1_population.parquet",
        "input_snapshot/retention_13_1_population.metadata.json",
        "model_inputs/feature_contract.csv",
        "preparation/preparation_info.json",
        "preparation/quality_screened_logistic_candidate_pool.txt",
        "preparation/logistic_feature_types.json",
        "preparation/quality_screened_xgboost_candidate_pool.txt",
        "preparation/xgboost_feature_types.json",
        "preparation/feature_routing_audit.csv",
        "preparation/outer_fold_definitions.csv",
        "preparation/inner_fold_definitions.csv",
        "logistic/run_status.json",
        "logistic/run_info.json",
        "logistic/recommended_model.json",
        "logistic/recommended_features.txt",
        "logistic/resolved_logistic_model_feature_types.json",
        "logistic/diagnostics/logistic_diagnostics_manifest.json",
        "logistic/diagnostics/logistic_calendar_year_performance.csv",
        "logistic/diagnostics/logistic_calendar_year_one_way_summary.csv",
        "logistic/diagnostics/logistic_selected_feature_one_way_summaries.csv",
        "logistic/diagnostics/logistic_policy_start_year_month_trend.csv",
        "logistic/diagnostics/logistic_policy_start_year_month_trend.png",
        "logistic/diagnostics/logistic_calendar_year_control_one_way.png",
        "logistic/diagnostics/logistic_business_retention_trend_manifest.json",
        "logistic/diagnostics/logistic_business_retention_trend_full_history.png",
        "logistic/diagnostics/logistic_business_retention_trend_selected_history.png",
        "xgboost/run_info.json",
        "xgboost/selected_features.json",
        "xgboost/selected_features.txt",
        "xgboost/xgb_resolved_model_feature_types.json",
        "xgboost/feature_selection_summary.md",
        "xgboost/diagnostics/xgboost_diagnostics_manifest.json",
        "xgboost/diagnostics/xgboost_business_retention_trend_manifest.json",
        "xgboost/diagnostics/xgboost_business_retention_trend_full_history.png",
        "xgboost/diagnostics/xgboost_business_retention_trend_selected_history.png",
        "xgboost/diagnostics/xgboost_calendar_year_control_one_way.png",
        "xgboost/diagnostics/xgboost_shap_beeswarm.png",
        "xgboost/diagnostics/xgboost_shap_plot_manifest.json",
        *FINALIZATION_REQUIRED_RELATIVE_PATHS,
    )
    invalid = []
    for relative in required:
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
            "Completed-run XGBoost reconfirmation requires regular, non-empty "
            "frozen acquisition, preparation, Logistic, and canonical XGBoost "
            "artifacts. Missing, empty, or linked:\n- "
            + "\n- ".join(invalid)
        )
    _require_model_diagnostics(run_root, "logistic")
    _require_model_diagnostics(run_root, "xgboost")
    _require_declared_artifacts(
        run_root,
        "outer_evaluation/performance_plots_manifest.json",
    )
    outer_status = _read_json_object(
        run_root / "outer_evaluation" / "run_status.json",
        "outer-evaluation status",
    )
    if (
        str(outer_status.get("status", "")).upper() != "COMPLETE"
        or str(outer_status.get("profile_hash", "")).strip() != profile_hash
    ):
        raise RuntimeError(
            "Completed run does not contain a COMPLETE outer-evaluation receipt "
            "for the requested profile."
        )
    pipeline_status = _read_json_object(
        run_root / "pipeline_status.json", "pipeline status"
    )
    if str(pipeline_status.get("run_id", "")).strip() != run_id:
        raise RuntimeError(
            "XGBoost reconfirmation run ID does not match pipeline_status.json."
        )
    if str(pipeline_status.get("status", "")).upper() != "COMPLETE":
        raise RuntimeError(
            "--rerun-xgboost-run-id accepts only a COMPLETE pipeline run."
        )
    if str(pipeline_status.get("profile_hash", "")).strip() != profile_hash:
        raise RuntimeError(
            "Completed run profile does not match the requested tenure/profile."
        )
    status_profile = pipeline_status.get("profile")
    if status_profile is not None and status_profile != profile_payload:
        raise RuntimeError(
            "Completed run profile payload does not match the requested profile."
        )
    # Parsing the snapshot metadata now catches a truncated/faulty receipt before
    # the long XGBoost subprocess is started. The XGBoost runner performs the
    # authoritative content-hash checks against preparation and current code.
    _read_json_object(
        run_root
        / "input_snapshot"
        / "retention_13_1_population.metadata.json",
        "input snapshot metadata",
    )
    preparation_info = _read_json_object(
        run_root / "preparation" / "preparation_info.json", "preparation info"
    )
    logistic_status = _read_json_object(
        run_root / "logistic" / "run_status.json", "Logistic run status"
    )
    logistic_info = _read_json_object(
        run_root / "logistic" / "run_info.json", "Logistic run info"
    )
    xgboost_info = _read_json_object(
        run_root / "xgboost" / "run_info.json", "canonical XGBoost run info"
    )
    for label, payload in (
        ("preparation", preparation_info),
        ("Logistic", logistic_info),
        ("canonical XGBoost", xgboost_info),
    ):
        if str(payload.get("profile_hash", "")).strip() != profile_hash:
            raise RuntimeError(
                f"{label} profile hash does not match the requested profile."
            )
        if payload.get("profile") != profile_payload:
            raise RuntimeError(
                f"{label} profile payload does not match the requested profile."
            )
    if str(logistic_status.get("status", "")).upper() != "COMPLETE":
        raise RuntimeError(
            "Logistic must be COMPLETE before XGBoost-only reconfirmation."
        )
    if logistic_status.get("final_holdout_rows_scored") != 0:
        raise RuntimeError("Logistic does not confirm a sealed final holdout.")
    if logistic_info.get("final_holdout_rows_scored") != 0:
        raise RuntimeError("Logistic run info does not confirm a sealed holdout.")
    if xgboost_info.get("final_holdout_rows_scored") != 0:
        raise RuntimeError(
            "Canonical XGBoost run info does not confirm a sealed holdout."
        )
    return pipeline_status




def _xgb_reconfirmation_output_name(
    run_root: Path,
    *,
    timestamp: str | None = None,
) -> str:
    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9]{6}", stamp):
        raise RuntimeError("Unsafe XGBoost reconfirmation timestamp.")
    base = f"xgboost_reconfirmation_{stamp}"
    candidate = base
    suffix = 1
    while (Path(run_root) / candidate).exists():
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    return candidate




def _optional_file_bytes(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None




def _run_completed_xgb_reconfirmation(
    *,
    params_module,
    profile_environment: dict[str, str],
    run_id: str,
) -> Path:
    """Run only XGBoost into a new sibling folder without owning state."""


    active_state = _read_state()
    active_status = str(active_state.get("status", "")).upper()
    active_run_id = str(active_state.get("run_id", "")).strip()
    if active_status in {"RUNNING", "PAUSED", "DATA_READY"}:
        raise RuntimeError(
            f"Pipeline run {active_run_id or '<unknown>'} is marked "
            f"{active_status}. Finish or intentionally resolve that active "
            "controller before starting XGBoost reconfirmation."
        )
    if active_state and active_status not in {
        "COMPLETE",
        "FAILED",
    }:
        raise RuntimeError(
            f"Pipeline state is {active_status or 'UNKNOWN'}. Repair "
            f"{STATE_PATH} before XGBoost reconfirmation."
        )
    _validate_completed_run_for_xgb_reconfirmation(
        run_id=run_id,
        profile_hash=params_module.PROFILE_HASH,
        profile_payload=params_module.PROFILE_PAYLOAD,
        output_root=PIPELINE_OUTPUT_ROOT,
    )
    run_root = PIPELINE_OUTPUT_ROOT / run_id
    output_name = _xgb_reconfirmation_output_name(run_root)
    destination = run_root / output_name
    if destination.exists():
        raise FileExistsError(
            f"XGBoost reconfirmation destination already exists: {destination}"
        )
    # The special controller path is designed not to write either status file.
    # Compare their exact bytes after the child exits so that this preservation
    # contract cannot silently regress later.
    protected_statuses = {
        STATE_PATH: _optional_file_bytes(STATE_PATH),
        run_root / "pipeline_status.json": _optional_file_bytes(
            run_root / "pipeline_status.json"
        ),
    }
    print("HOME RETENTION 13/1 - XGBOOST RECONFIRMATION", flush=True)
    print(f"Profile:         {params_module.PROFILE_LABEL}", flush=True)
    print(f"Source run ID:   {run_id}", flush=True)
    print(f"Output folder:   {destination}", flush=True)
    print("Data acquisition: SKIPPED", flush=True)
    print("Preparation:      REUSED (frozen)", flush=True)
    print("Logistic:         SKIPPED (validated COMPLETE)", flush=True)
    print("Canonical XGBoost: PRESERVED", flush=True)
    child_environment = {
        **profile_environment,
        "RETENTION_PIPELINE_RUN_ID": run_id,
        "RETENTION_ROOT_ORCHESTRATED": "1",
        "RETENTION_REFRESH_ALL_TENURE_TABLE": "0",
        "RETENTION_REFRESH_PREFLIGHT_ONLY": "0",
        FAILED_RUN_RESUME_ENV: "0",
        FAILED_RUN_RESUME_RUN_ID_ENV: "",
        FAILED_RUN_RESUME_ATTEMPT_ID_ENV: "",
        XGB_RECONFIRMATION_ENV: "1",
        XGB_RECONFIRMATION_RUN_ID_ENV: run_id,
        XGB_RECONFIRMATION_OUTPUT_NAME_ENV: output_name,
    }
    try:
        run_scripts(
            scripts=[STAGES[1].relative_to(PROJECT_ROOT)],
            path=PROJECT_ROOT,
            env=child_environment,
        )
    finally:
        for path, expected in protected_statuses.items():
            if _optional_file_bytes(path) != expected:
                raise RuntimeError(
                    "XGBoost reconfirmation changed a protected pipeline "
                    f"status file unexpectedly: {path}"
                )
    if not destination.is_dir() or destination.is_symlink():
        raise RuntimeError(
            "XGBoost reconfirmation finished without a safe sibling output "
            f"directory: {destination}"
        )
    print(f"XGBOOST RECONFIRMATION COMPLETE: {destination}", flush=True)
    return destination
def _validate_canonical_model_restart(
    *, run_id: str, profile_hash: str, profile_payload: dict[str, Any],
) -> dict[str, Any]:
    """Read-only proof that retained inputs and Logistic describe this run."""
    _resume_run_id(run_id)
    active = _read_state()
    if active and str(active.get("status", "")).upper() not in {"COMPLETE", "FAILED"}:
        raise RuntimeError(
            "An active or unresolved pipeline exists; no output was archived. "
            "Wait for the current process to finish before restarting modeling."
        )
    run_root = PIPELINE_OUTPUT_ROOT / run_id
    if (run_root.is_symlink() or not run_root.is_dir()
            or run_root.resolve().parent != PIPELINE_OUTPUT_ROOT.resolve()):
        raise RuntimeError("Restart requires a regular run directly under pipeline_runs.")
    status = _read_json_object(run_root / "pipeline_status.json", "pipeline status")
    if (str(status.get("status", "")).upper() not in {"COMPLETE", "FAILED"}
            or status.get("run_id") != run_id
            or status.get("profile_hash") != profile_hash
            or status.get("profile", profile_payload) != profile_payload):
        raise RuntimeError("Restart requires a COMPLETE/FAILED run with the exact requested profile.")
    for directory in ("input_snapshot", "model_inputs", "preparation", "logistic"):
        path = run_root / directory
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError(f"Retained run directory is missing or linked: {path}")
    preparation = _read_json_object(
        run_root / "preparation/preparation_info.json", "preparation receipt"
    )
    logistic = _read_json_object(run_root / "logistic/run_info.json", "Logistic receipt")
    logistic_status = _read_json_object(
        run_root / "logistic/run_status.json", "Logistic status"
    )
    for label, payload in (("Preparation", preparation), ("Logistic", logistic)):
        if (payload.get("profile_hash") != profile_hash
                or payload.get("profile") != profile_payload
                or payload.get("final_holdout_rows_scored") != 0):
            raise RuntimeError(f"{label} does not match the frozen profile/holdout contract.")
    if (str(logistic_status.get("status", "")).upper() != "COMPLETE"
            or logistic_status.get("final_holdout_rows_scored") != 0):
        raise RuntimeError("A complete Logistic selection is required before XGBoost restart.")
    # Compare data content, not timestamps. Changed plotting code is allowed;
    # the saved Logistic model recipe remains a frozen development result.
    frozen = (
        ("input_snapshot/retention_13_1_population.parquet", "input_sha256", "data"),
        ("input_snapshot/retention_13_1_population.metadata.json", "input_metadata_sha256", None),
        ("model_inputs/feature_contract.csv", "feature_contract_sha256", "manifest"),
    )
    for relative, preparation_key, logistic_key in frozen:
        path = run_root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Frozen input is missing, empty or linked: {path}")
        digest = _sha256_file(path)
        if preparation.get(preparation_key) != digest:
            raise RuntimeError(f"Frozen input no longer matches preparation: {relative}")
        if logistic_key is not None and logistic.get(logistic_key, {}).get("sha256") != digest:
            raise RuntimeError(f"Frozen input no longer matches Logistic: {relative}")
    for relative, key in (
        ("preparation/outer_fold_definitions.csv", "outer_fold_definitions"),
        ("preparation/inner_fold_definitions.csv", "inner_fold_definitions"),
        ("preparation/quality_screened_logistic_candidate_pool.txt", "pool_file"),
        ("preparation/logistic_feature_types.json", "feature_types_file"),
        ("preparation/feature_routing_audit.csv", "routing_audit"),
        ("preparation/preparation_info.json", "preparation_info"),
    ):
        path = run_root / relative
        if path.is_symlink() or not path.is_file() or logistic.get(key, {}).get("sha256") != _sha256_file(path):
            raise RuntimeError(f"Frozen Logistic input receipt does not match: {relative}")
    for relative in (
        "logistic/recommended_model.json", "logistic/recommended_features.txt",
        "logistic/resolved_logistic_model_feature_types.json",
    ):
        path = run_root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Frozen Logistic recipe is missing, empty or linked: {path}")
    recipe = _read_json_object(run_root / "logistic/recommended_model.json", "Logistic recommendation")
    types = _read_json_object(run_root / "logistic/resolved_logistic_model_feature_types.json", "Logistic feature types")
    features = recipe.get("features")
    feature_lines = [value.strip().upper() for value in
                     (run_root / "logistic/recommended_features.txt").read_text(encoding="utf-8-sig").splitlines()
                     if value.strip()]
    if (recipe.get("status") != "COMPLETE" or recipe.get("final_holdout_rows_scored") != 0
            or not isinstance(features, list) or not features
            or features != logistic.get("features")
            or recipe.get("C") != logistic.get("C")
            or recipe.get("selected_history") != logistic.get("selected_history")
            or feature_lines != [str(feature).strip().upper() for feature in features]
            or len(feature_lines) != len(set(feature_lines))
            or any(str(types.get(feature, "")).lower() not in {"continuous", "discrete"}
                   for feature in feature_lines)):
        raise RuntimeError("Frozen Logistic recommendation, selected features and run receipt disagree.")
    return status




def _archive_canonical_model_outputs(run_root: Path, *, attempt_id: str) -> Path:
    """Move stale model-dependent trees out of canonical output, with rollback."""
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9]{6}", attempt_id):
        raise ValueError("Invalid model-restart attempt ID.")
    names = {
        "xgboost", "outer_evaluation", "deployment", "final_test", "reports",
        "report", "reporting", "performance_reports", "model_reports", "final_refit_diagnostics",
        "xgboost_reselection", "xgboost_outer_comparisons",
    }
    sources = sorted(
        (path for path in run_root.iterdir()
         if path.name in names or path.name.startswith("xgboost_reconfirmation_")),
        key=lambda path: path.name,
    )
    # Validate every path before the first move, including nested links.
    for source in sources:
        if source.is_symlink() or not source.is_dir():
            raise RuntimeError(f"Dependent output is not a regular directory: {source}")
        if any(path.is_symlink() or not (path.is_dir() or path.is_file())
               for path in source.rglob("*")):
            raise RuntimeError(f"Dependent output contains a linked/nonregular entry: {source}")
    archive = run_root / f"model_restart_archive_{attempt_id}"
    if archive.exists() or archive.is_symlink():
        raise FileExistsError(f"Model restart archive already exists: {archive}")
    prior_status = (run_root / "pipeline_status.json").read_bytes()
    archive.mkdir()
    moved: list[Path] = []
    try:
        (archive / "prior_pipeline_status.json").write_bytes(prior_status)
        for source in sources:
            source.rename(archive / source.name)
            moved.append(source)
        receipt = {
            "status": "COMPLETE", "restart_kind": MODEL_RESTART_KIND,
            "run_id": run_root.name, "attempt_id": attempt_id,
            "archived_directories": [source.name for source in moved],
            "retained_directories": ["input_snapshot", "model_inputs", "preparation", "logistic"],
            "prior_final_test_exists": "final_test" in [source.name for source in moved],
            "archived_at": datetime.now().isoformat(timespec="seconds"),
        }
        (archive / "restart_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    except BaseException:
        for source in reversed(moved):
            (archive / source.name).rename(source)
        raise
    return archive




def _run_canonical_model_restart(*, params_module, profile_environment: dict[str, str], run_id: str) -> None:
    """Restart normal c3/c4; never import a reselection recipe or force a size."""
    environment = {
        **profile_environment, "RETENTION_PIPELINE_RUN_ID": run_id,
        "RETENTION_ROOT_ORCHESTRATED": "1", MODEL_RESTART_ENV: "xgboost",
        FAILED_RUN_RESUME_ENV: "0", FAILED_RUN_RESUME_RUN_ID_ENV: "",
        FAILED_RUN_RESUME_ATTEMPT_ID_ENV: "", "RETENTION_RESUME_FAILED_STAGE": "",
        XGB_RECONFIRMATION_ENV: "0", XGB_RECONFIRMATION_RUN_ID_ENV: "",
        XGB_RECONFIRMATION_OUTPUT_NAME_ENV: "", "RETENTION_XGB_RESUME_POST_SELECTION": "0",
        "RETENTION_REFRESH_ALL_TENURE_TABLE": "0", "RETENTION_REFRESH_PREFLIGHT_ONLY": "0",
    }
    started = datetime.now().isoformat(timespec="seconds")
    required_stages = [STAGES[1], PROJECT_ROOT / "c_modeling/c3_run_xgboost.py",
                       PROJECT_ROOT / "c_modeling/c4_evaluate_and_finalize.py"]
    if any(not path.is_file() for path in required_stages):
        raise FileNotFoundError("Canonical model restart requires the normal c0, c3 and c4 scripts.")
    with exclusive_operation_lock(OPERATION_LOCK_PATH):
        prior = _validate_canonical_model_restart(
            run_id=run_id, profile_hash=params_module.PROFILE_HASH,
            profile_payload=params_module.PROFILE_PAYLOAD,
        )
        # Preflight uses a clean process so params resolves the requested run.
        # This must finish successfully BEFORE any completed output is moved.
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "c_modeling/c3_run_xgboost.py"),
             "--validate-preparation-only"],
            cwd=PROJECT_ROOT, env={**os.environ, **environment}, check=True,
        )
        run_root = PIPELINE_OUTPUT_ROOT / run_id
        attempt = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        prior_test_seen = bool(prior.get("prior_final_test_exists")) or (run_root / "final_test").exists()
        archive = _archive_canonical_model_outputs(run_root, attempt_id=attempt)
        metadata = {
            "resume_kind": MODEL_RESTART_KIND, "resume_requested_run_id": run_id,
            "recovery_attempt_id": attempt, "recovery_archive": str(archive),
            "prior_final_test_exists": prior_test_seen,
            "acquisition_reused": True, "preparation_reused": True, "logistic_reused": True,
        }
        _write_state(_state_payload(
            params_module=params_module, run_id=run_id, status="RUNNING",
            stage="xgboost", started=started, **metadata,
        ))
    print(f"NORMAL PIPELINE RESTART FROM XGBOOST: {run_id}", flush=True)
    print(f"Prior model/evaluation outputs preserved at: {archive}", flush=True)
    print("Reusing verified download, preparation and Logistic. Running normal XGBoost selection and outer evaluation.", flush=True)
    try:
        run_scripts(scripts=[STAGES[1].relative_to(PROJECT_ROOT)], path=PROJECT_ROOT, env=environment)
    except BaseException as exc:
        _write_state(_state_payload(
            params_module=params_module, run_id=run_id, status="FAILED",
            stage="modeling", started=started, error=f"{type(exc).__name__}: {exc}",
            **metadata,
        ))
        raise
    _write_state(_state_payload(
        params_module=params_module, run_id=run_id, status="COMPLETE",
        started=started, completed=datetime.now().isoformat(timespec="seconds"),
        final_holdout_rows_scored=0, **metadata,
    ))
    print(f"NORMAL PIPELINE RESTART COMPLETE: {run_id}", flush=True)




def _execution_plan(
    *,
    profile_hash: str,
    fresh_run_id: str,
    recover_stale_run: bool = False,
    resume_run_id: str | None = None,
    rerun_preparation: bool = False,
    profile_payload: dict[str, Any] | None = None,
) -> tuple[str, list[Path], bool, dict[str, Any]]:
    state = _read_state()
    status = str(state.get("status", "")).upper()
    prior_run_id = str(state.get("run_id", "")).strip()
    prior_profile_hash = str(state.get("profile_hash", "")).strip()
    if status == "RUNNING":
        if resume_run_id is not None:
            raise RuntimeError(
                f"Pipeline run {prior_run_id or '<unknown>'} is still marked "
                "RUNNING. Stop that process first. Failed-run recovery will not "
                "take over while another controller may still be active."
            )
        if not recover_stale_run:
            raise RuntimeError(
                f"Pipeline run {prior_run_id or '<unknown>'} is marked RUNNING. "
                "If you have confirmed that its Python process has stopped, "
                "repeat the command with --recover-stale-run."
            )
        recovered = {
            **state,
            "status": "FAILED",
            "termination_kind": "RECOVERED_STALE_RUNNING_MARKER",
            "error": (
                "Operator confirmed that the prior process had stopped; its "
                "stale RUNNING marker was recovered before a fresh run."
            ),
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        _write_state(recovered)
        state = recovered
        status = "FAILED"
    if resume_run_id is not None:
        if status == "PAUSED":
            raise RuntimeError(
                "A PAUSED Logistic checkpoint is active. Resume it with the "
                "identical profile command before recovering another failed run."
            )
        if state and status not in {"COMPLETE", "FAILED"}:
            raise RuntimeError(
                f"Pipeline state is {status or 'UNKNOWN'}. Repair {STATE_PATH} first."
            )
        if rerun_preparation:
            if profile_payload is None:
                raise RuntimeError(
                    "Preparation rebuild requires the active profile payload."
                )
            target_state, _ = _validate_failed_run_for_preparation_rebuild(
                run_id=resume_run_id,
                profile_hash=profile_hash,
                profile_payload=profile_payload,
                output_root=PIPELINE_OUTPUT_ROOT,
            )
            recovery_attempt_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            archive = _archive_modeling_outputs_for_preparation_rebuild(
                run_id=resume_run_id,
                recovery_attempt_id=recovery_attempt_id,
                output_root=PIPELINE_OUTPUT_ROOT,
            )
            target_state = {
                **target_state,
                "modeling_rebuild_attempt_id": recovery_attempt_id,
                "modeling_rebuild_archive": str(archive),
            }
        else:
            if profile_payload is None:
                raise RuntimeError(
                    "Failed-run recovery requires the active profile payload."
                )
            target_state = _validate_failed_run_resume(
                run_id=resume_run_id,
                profile_hash=profile_hash,
                profile_payload=profile_payload,
                output_root=PIPELINE_OUTPUT_ROOT,
            )
        return resume_run_id, [STAGES[1]], True, target_state
    if status == "PAUSED":
        if not prior_run_id or not prior_profile_hash:
            raise RuntimeError(
                "PAUSED state is missing run/profile identity and cannot resume."
            )
        if prior_profile_hash != profile_hash:
            prior_label = state.get("profile_label", state.get("profile_slug"))
            raise RuntimeError(
                f"The paused run belongs to {prior_label}, not the requested "
                "profile. Repeat the paused run's identical command first."
            )
        return prior_run_id, [STAGES[1]], True, state
    if state and status not in {"COMPLETE", "FAILED"}:
        raise RuntimeError(
            f"Pipeline state is {status or 'UNKNOWN'}. Repair {STATE_PATH} first."
        )
    return fresh_run_id, list(STAGES), False, state
def _run_paths(run_id: str) -> dict[str, str]:
    root = PIPELINE_OUTPUT_ROOT / run_id
    return {
        "snapshot_path": str(
            root / "input_snapshot" / "retention_13_1_population.parquet"
        ),
        "snapshot_metadata_path": str(
            root / "input_snapshot" / "retention_13_1_population.metadata.json"
        ),
        "feature_contract_path": str(root / "model_inputs" / "feature_contract.csv"),
        "candidate_pool_path": str(root / "model_inputs" / "candidate_pool.txt"),
    }
def _run_started_at(resuming: bool, prior_state: dict[str, Any]) -> str:
    if resuming and prior_state.get("started"):
        return str(prior_state["started"])
    return datetime.now().isoformat(timespec="seconds")
def _state_payload(
    *, params_module, run_id: str, status: str, **values: Any
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": status,
        **params_module.profile_state_payload(),
        **_run_paths(run_id),
        **values,
    }
def main(argv: list[str] | None = None) -> None:
    effective_argv = (
        sys.argv[1:] if argv is None and __name__ == "__main__" else (argv or [])
    )
    args = parse_args(effective_argv)
    params_module, profile_environment = _activate_profile(
        args.profile,
        include_exposure_year_predictor=(
            args.include_exposure_year_predictor
        ),
        test_policy=args.test_policy,
        sample_percent=args.sample_percent,
    )
    if args.restart_modeling_from is not None:
        _run_canonical_model_restart(
            params_module=params_module, profile_environment=profile_environment,
            run_id=args.resume_run_id,
        )
        return
    if args.rerun_xgboost_run_id is not None:
        if not STAGES[1].is_file():
            raise FileNotFoundError(
                f"Missing modeling controller required for XGBoost "
                f"reconfirmation: {STAGES[1]}"
            )
        with exclusive_operation_lock(OPERATION_LOCK_PATH):
            _run_completed_xgb_reconfirmation(
                params_module=params_module,
                profile_environment=profile_environment,
                run_id=args.rerun_xgboost_run_id,
            )
        return
    if args.refresh_source_only or args.refresh_preflight_only:
        refresh_script = (
            PROJECT_ROOT
            / "b_data_acquisition"
            / "b0_refresh_all_tenure_table.py"
        )
        if not refresh_script.is_file():
            raise FileNotFoundError(f"Missing source refresh step: {refresh_script}")
        run_id = (
            f"{params_module.TIMESTAMP}_{params_module.LOB}_"
            f"{params_module.COV}_ALL-TENURES_refresh-all-tenure"
        )
        mode_label = (
            "PREFLIGHT ONLY"
            if args.refresh_preflight_only
            else "ALL-TENURE CY REFRESH ONLY"
        )
        print(f"HOME RETENTION 13/1 - {mode_label}", flush=True)
        print(f"Existing table: {params_module.MAIN_TABLE_FQN}", flush=True)
        print(f"Refresh run ID: {run_id}", flush=True)
        with exclusive_operation_lock(OPERATION_LOCK_PATH):
            active = _read_state()
            if str(active.get("status", "")).upper() == "RUNNING":
                raise RuntimeError(
                    "A model pipeline is RUNNING. Wait for it to finish or pause "
                    "before replacing the shared CY table."
                )
            run_scripts(
                scripts=[refresh_script.relative_to(PROJECT_ROOT)],
                path=PROJECT_ROOT,
                env={
                    **profile_environment,
                    "RETENTION_PIPELINE_RUN_ID": run_id,
                    "RETENTION_REFRESH_ALL_TENURE_TABLE": "1",
                    "RETENTION_REFRESH_PREFLIGHT_ONLY": (
                        "1" if args.refresh_preflight_only else "0"
                    ),
                },
            )
        print(
            "ALL-TENURE PREFLIGHT COMPLETE; CY UNCHANGED"
            if args.refresh_preflight_only
            else "ALL-TENURE CY REFRESH COMPLETE",
            flush=True,
        )
        return
    if args.sql_dry_run:
        from b_data_acquisition.b1_download_retention_13_1 import main as b1_main
        b1_main(["--dry-run"])
        return
    missing = [path for path in STAGES if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required one-click phase(s) missing:\n"
            + "\n".join(f"- {path}" for path in missing)
        )
    with exclusive_operation_lock(OPERATION_LOCK_PATH):
        run_id, selected_stages, resuming, prior_state = _execution_plan(
            profile_hash=params_module.PROFILE_HASH,
            profile_payload=params_module.PROFILE_PAYLOAD,
            fresh_run_id=params_module.new_run_id(Path(__file__).stem),
            recover_stale_run=bool(args.recover_stale_run),
            resume_run_id=args.resume_run_id,
            rerun_preparation=bool(args.rerun_preparation),
        )
        if args.acquisition_only:
            if resuming:
                raise RuntimeError(
                    "--acquisition-only cannot resume a PAUSED modeling run. "
                    "Complete or intentionally archive that run first."
                )
            selected_stages = [STAGES[0]]
        os.environ["RETENTION_PIPELINE_RUN_ID"] = run_id
        started = _run_started_at(resuming, prior_state)
        modeling_rebuild = bool(args.rerun_preparation)
        failed_run_resume = args.resume_run_id is not None and not modeling_rebuild
        failed_resume_kind = (
            str(prior_state.get("validated_resume_kind", "")).strip()
            if failed_run_resume
            else ""
        )
        resume_metadata: dict[str, Any] = {"resumed": resuming}
        if modeling_rebuild:
            resume_metadata.update(
                {
                    "resume_kind": "FAILED_RUN_PREPARATION_REBUILD",
                    "resume_requested_run_id": args.resume_run_id,
                    "resume_source_status": "FAILED",
                    "recovery_attempt_id": prior_state.get(
                        "modeling_rebuild_attempt_id"
                    ),
                    "recovery_archive": prior_state.get(
                        "modeling_rebuild_archive"
                    ),
                    "acquisition_reused": True,
                    "preparation_rerun": True,
                    "logistic_rerun": True,
                    "xgboost_rerun": True,
                    "outer_evaluation_rerun": True,
                    "final_refit_rerun": True,
                }
            )
        elif failed_run_resume:
            if failed_resume_kind not in {
                FAILED_RUN_LOGISTIC_RETRY,
                FAILED_RUN_XGBOOST_RETRY,
            }:
                raise RuntimeError(
                    "Failed-run validation did not identify a safe recovery stage."
                )
            recovery_attempt_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            resume_metadata.update(
                {
                    "resume_kind": failed_resume_kind,
                    "resume_requested_run_id": args.resume_run_id,
                    "resume_source_status": "FAILED",
                    "recovery_attempt_id": recovery_attempt_id,
                    "acquisition_reused": True,
                    "preparation_reused": True,
                    "logistic_checkpoint_recovery": (
                        failed_resume_kind == FAILED_RUN_LOGISTIC_RETRY
                    ),
                }
            )
        _write_state(
            _state_payload(
                params_module=params_module,
                run_id=run_id,
                status="RUNNING",
                stage=(
                    "modeling_rebuild"
                    if modeling_rebuild
                    else "logistic_recovery"
                    if failed_resume_kind == FAILED_RUN_LOGISTIC_RETRY
                    else "xgboost_recovery"
                    if failed_resume_kind == FAILED_RUN_XGBOOST_RETRY
                    else "modeling" if resuming else "data_acquisition"
                ),
                started=started,
                **resume_metadata,
                refresh_source_requested=bool(args.refresh_source),
                final_holdout_scoring_requested=False,
            )
        )
    try:
        print("HOME RETENTION 13/1 - TENURE PROFILE PIPELINE", flush=True)
        print(f"Profile:         {params_module.PROFILE_LABEL}", flush=True)
        print(f"Profile ID:      {params_module.PROFILE_SLUG}", flush=True)
        print(f"Pipeline run ID: {run_id}", flush=True)
        print("Final sealed holdout will be scored: NO", flush=True)
        if modeling_rebuild:
            print(
                "Failed-run modeling rebuild: the frozen download and model "
                "inputs were verified and reused. Prior preparation/model "
                "outputs were archived; c1 through c4 will rerun.",
                flush=True,
            )
        elif failed_run_resume:
            if failed_resume_kind == FAILED_RUN_LOGISTIC_RETRY:
                print(
                    "Failed-run resume: acquisition and preparation are reused. "
                    "The compatible Logistic fit checkpoint resumes first; "
                    "XGBoost and outer evaluation follow after Logistic completes.",
                    flush=True,
                )
            else:
                print(
                    "Failed-run resume: acquisition is skipped; preparation and "
                    "Logistic completion were validated. A recoverable XGBoost "
                    "Stage-4 boundary will be finalized in place; otherwise the "
                    "prior attempt will be archived before XGBoost is retried.",
                    flush=True,
                )
        elif resuming:
            print("Resume mode: acquisition is skipped.", flush=True)
        else:
            if args.refresh_source:
                print(
                    "1. Refresh and QA the shared all-tenure Snowflake table",
                    flush=True,
                )
            print(
                "2. Filter the requested tenure in Snowflake and download",
                flush=True,
            )
            print(
                "3. Prepare features/folds; run Logistic and XGBoost",
                flush=True,
            )
            print(
                "4. Score the five outer folds, create performance plots, and "
                "refit one provisional pre-holdout model",
                flush=True,
            )
            if args.acquisition_only:
                print(
                    "Test mode: stop after data acquisition/input contract.",
                    flush=True,
                )
        child_environment = {
            **profile_environment,
            "RETENTION_PIPELINE_RUN_ID": run_id,
            "RETENTION_ROOT_ORCHESTRATED": "1",
            "RETENTION_REFRESH_ALL_TENURE_TABLE": (
                "1" if args.refresh_source and not resuming else "0"
            ),
            "RETENTION_REFRESH_PREFLIGHT_ONLY": "0",
            FAILED_RUN_RESUME_ENV: "1" if failed_run_resume else "0",
            FAILED_RUN_RESUME_RUN_ID_ENV: (
                run_id if failed_run_resume else ""
            ),
            FAILED_RUN_RESUME_ATTEMPT_ID_ENV: (
                str(resume_metadata.get("recovery_attempt_id", ""))
            ),
            FAILED_RUN_RESUME_STAGE_ENV: (
                "logistic"
                if failed_resume_kind == FAILED_RUN_LOGISTIC_RETRY
                else "xgboost"
                if failed_resume_kind == FAILED_RUN_XGBOOST_RETRY
                else ""
            ),
        }
        for stage in selected_stages:
            if stage == STAGES[1]:
                _write_state(
                    _state_payload(
                        params_module=params_module,
                        run_id=run_id,
                        status="RUNNING",
                        stage=(
                            "modeling_rebuild"
                            if modeling_rebuild
                            else "logistic_recovery"
                            if failed_resume_kind == FAILED_RUN_LOGISTIC_RETRY
                            else "xgboost_recovery"
                            if failed_resume_kind == FAILED_RUN_XGBOOST_RETRY
                            else "modeling"
                        ),
                        started=started,
                        **resume_metadata,
                        final_holdout_scoring_requested=False,
                    )
                )
            run_scripts(
                scripts=[stage.relative_to(PROJECT_ROOT)],
                path=PROJECT_ROOT,
                env=child_environment,
            )
    except KeyboardInterrupt as exc:
        _write_state(
            _state_payload(
                params_module=params_module,
                run_id=run_id,
                status="FAILED",
                termination_kind="OPERATOR_INTERRUPTED",
                error="KeyboardInterrupt: operator stopped the pipeline",
                exit_code=130,
                started=started,
                **resume_metadata,
                updated=datetime.now().isoformat(timespec="seconds"),
            )
        )
        print(
            "\nPipeline stopped by operator; run state recorded as FAILED.",
            flush=True,
        )
        raise SystemExit(130) from exc
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 75 and (
            not failed_run_resume
            or failed_resume_kind == FAILED_RUN_LOGISTIC_RETRY
        ):
            _write_state(
                _state_payload(
                    params_module=params_module,
                    run_id=run_id,
                    status="PAUSED",
                    stage="logistic",
                    started=started,
                    **resume_metadata,
                    updated=datetime.now().isoformat(timespec="seconds"),
                )
            )
            if modeling_rebuild:
                print(
                    "Pipeline paused safely. Resume with the same tenure/profile "
                    "selector, but omit --resume-run-id and --rerun-preparation.",
                    flush=True,
                )
            elif failed_resume_kind == FAILED_RUN_LOGISTIC_RETRY:
                print(
                    "Pipeline paused safely after failed-run Logistic recovery. "
                    "Repeat the same tenure/profile selector without "
                    "--resume-run-id.",
                    flush=True,
                )
            else:
                print(
                    "Pipeline paused safely. Repeat the identical profile command "
                    "to resume.",
                    flush=True,
                )
            raise SystemExit(75)
        _write_state(
            _state_payload(
                params_module=params_module,
                run_id=run_id,
                status="FAILED",
                exit_code=exc.returncode,
                started=started,
                **resume_metadata,
                updated=datetime.now().isoformat(timespec="seconds"),
            )
        )
        raise SystemExit(exc.returncode) from exc
    except Exception as exc:
        _write_state(
            _state_payload(
                params_module=params_module,
                run_id=run_id,
                status="FAILED",
                error=f"{type(exc).__name__}: {exc}",
                started=started,
                **resume_metadata,
                updated=datetime.now().isoformat(timespec="seconds"),
            )
        )
        raise
    final_holdout_rows_scored = 0
    try:
        if not args.acquisition_only:
            finalization_status = _read_json_object(
                PIPELINE_OUTPUT_ROOT
                / run_id
                / "outer_evaluation"
                / "run_status.json",
                "outer-evaluation status",
            )
            final_holdout_rows_scored = int(
                finalization_status.get("final_holdout_rows_scored", 0)
            )
            if final_holdout_rows_scored != 0:
                raise RuntimeError(
                    "The sealed final holdout was scored during model development."
                )
    except Exception as exc:
        _write_state(
            _state_payload(
                params_module=params_module,
                run_id=run_id,
                status="FAILED",
                error=f"{type(exc).__name__}: {exc}",
                started=started,
                **resume_metadata,
                updated=datetime.now().isoformat(timespec="seconds"),
            )
        )
        raise
    _write_state(
        _state_payload(
            params_module=params_module,
            run_id=run_id,
            status="COMPLETE",
            started=started,
            **resume_metadata,
            final_holdout_scoring_requested=False,
            final_holdout_rows_scored=final_holdout_rows_scored,
            completed=datetime.now().isoformat(timespec="seconds"),
        )
    )
    print(
        "\nRETENTION 13/1 ACQUISITION TEST COMPLETE"
        if args.acquisition_only
        else "\nRETENTION 13/1 PROFILE PIPELINE COMPLETE",
        flush=True,
    )
if __name__ == "__main__":
    main()
