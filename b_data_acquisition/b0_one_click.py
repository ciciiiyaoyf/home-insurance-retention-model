# -*- coding: utf-8 -*-
"""Run the tenure-aware retention data-acquisition phase."""


from __future__ import annotations


from datetime import datetime
import json
import logging
import os
from pathlib import Path
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
    PROFILE_LABEL,
    REFRESH_ALL_TENURE_TABLE,
    new_run_id,
    profile_env,
    profile_state_payload,
    setup_params,
)




logger = logging.getLogger(__name__)
STEP_NAME = Path(__file__).stem
STATE_PATH = Path(ACTIVE_PIPELINE_STATE_PATH)
ROOT_ORCHESTRATED = os.getenv("RETENTION_ROOT_ORCHESTRATED", "").strip() == "1"
BASE_SCRIPTS = [
    "b1_download_retention_13_1.py",
    "b2_prepare_model_inputs.py",
]




def acquisition_scripts(refresh_source: bool = REFRESH_ALL_TENURE_TABLE) -> list[str]:
    scripts = list(BASE_SCRIPTS)
    if refresh_source:
        scripts.insert(0, "b0_refresh_all_tenure_table.py")
    return scripts




SCRIPTS = acquisition_scripts()




def _new_run_id() -> str:
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
    return payload if isinstance(payload, dict) else {"status": "MALFORMED"}




def _run_paths(run_id: str) -> dict[str, str]:
    root = Path(PIPELINE_OUTPUT_ROOT) / run_id
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




def _state_payload(run_id: str, status: str, **values: object) -> dict[str, object]:
    return {
        "run_id": run_id,
        "status": status,
        **profile_state_payload(),
        **_run_paths(run_id),
        **values,
    }




def _write_state(payload: dict[str, object]) -> None:
    run_id = str(payload.get("run_id", "")).strip()
    destinations = [STATE_PATH]
    if run_id:
        destinations.append(Path(PIPELINE_OUTPUT_ROOT) / run_id / "pipeline_status.json")
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, destination)




def _claim_standalone_run(supplied_run_id: str) -> tuple[str, str]:
    with exclusive_operation_lock(Path(PIPELINE_OPERATION_LOCK_PATH)):
        state = _read_state()
        status = str(state.get("status", "")).upper()
        if status in {"RUNNING", "PAUSED", "DATA_READY"}:
            raise RuntimeError(
                f"Pipeline run {state.get('run_id', '<unknown>')} is {status}."
            )
        if state and status not in {"COMPLETE", "FAILED"}:
            raise RuntimeError(f"Pipeline state is {status or 'UNKNOWN'}.")
        run_id = supplied_run_id or _new_run_id()
        started = datetime.now().isoformat(timespec="seconds")
        _write_state(
            _state_payload(
                run_id,
                "RUNNING",
                stage="data_acquisition",
                started=started,
            )
        )
    return run_id, started




def main() -> None:
    supplied_run_id = os.getenv("RETENTION_PIPELINE_RUN_ID", "").strip()
    if ROOT_ORCHESTRATED:
        if not supplied_run_id:
            raise RuntimeError(
                "Root-orchestrated acquisition requires RETENTION_PIPELINE_RUN_ID."
            )
        run_id = supplied_run_id
        started = str(
            _read_state().get(
                "started", datetime.now().isoformat(timespec="seconds")
            )
        )
    else:
        run_id, started = _claim_standalone_run(supplied_run_id)


    phase_dir = Path(__file__).resolve().parent
    scripts = acquisition_scripts()
    print("HOME RETENTION 13/1 - DATA ACQUISITION", flush=True)
    print(f"Profile: {PROFILE_LABEL}", flush=True)
    print(f"Run ID: {run_id}", flush=True)
    print("Execution order:", flush=True)
    for index, script in enumerate(scripts, start=1):
        print(f"  {index}. {script}", flush=True)


    setup = None
    try:
        setup = _create_setup(run_id)
        run_scripts(
            scripts=scripts,
            path=phase_dir,
            env={
                **profile_env(),
                "RETENTION_PIPELINE_RUN_ID": run_id,
                "RETENTION_ROOT_ORCHESTRATED": (
                    "1" if ROOT_ORCHESTRATED else "0"
                ),
                "RETENTION_REFRESH_ALL_TENURE_TABLE": (
                    "1" if REFRESH_ALL_TENURE_TABLE else "0"
                ),
                "RETENTION_REFRESH_PREFLIGHT_ONLY": "0",
            },
        )
    except subprocess.CalledProcessError as exc:
        if setup is not None:
            setup.shutdown()
            setup = None
        if not ROOT_ORCHESTRATED:
            _write_state(
                _state_payload(
                    run_id,
                    "FAILED",
                    stage="data_acquisition",
                    started=started,
                    exit_code=exc.returncode,
                    updated=datetime.now().isoformat(timespec="seconds"),
                )
            )
        logger.exception("Retention data acquisition failed.")
        raise SystemExit(exc.returncode) from exc
    except Exception as exc:
        if setup is not None:
            setup.shutdown()
            setup = None
        if not ROOT_ORCHESTRATED:
            _write_state(
                _state_payload(
                    run_id,
                    "FAILED",
                    stage="data_acquisition",
                    started=started,
                    error=f"{type(exc).__name__}: {exc}",
                    updated=datetime.now().isoformat(timespec="seconds"),
                )
            )
        logger.exception("Retention data acquisition failed.")
        raise
    finally:
        if setup is not None:
            setup.shutdown()


    if not ROOT_ORCHESTRATED:
        _write_state(
            _state_payload(
                run_id,
                "DATA_READY",
                stage="modeling",
                started=started,
                data_ready=datetime.now().isoformat(timespec="seconds"),
            )
        )
        print(
            "DATA ACQUISITION COMPLETE. Run c_modeling/c0_one_click.py next.",
            flush=True,
        )




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
            "b0_one_click.py accepts no direct arguments; use root "
            "a_business_understanding/a0_one_click.py."
        )
    main()




if __name__ == "__main__":
    _entrypoint()
