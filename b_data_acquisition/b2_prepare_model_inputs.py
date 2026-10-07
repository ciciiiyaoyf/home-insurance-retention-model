# -*- coding: utf-8 -*-
"""Derive model inputs from the ROC config and verified profile CTE extract.


This stage never rewrites or reduces the source config.  It creates a separate,
run-specific model contract for the columns actually queried by B1 and a text
pool containing the config rows whose ``cv_ind`` equals 1.
"""


from __future__ import annotations


import argparse
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


import pandas as pd




PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from b_data_acquisition.b1_download_retention_13_1 import (  # noqa: E402
    _sha256,
    load_download_contract,
    verify_published_artifacts,
)
from params import (  # noqa: E402
    DOWNLOAD_METADATA_PATH,
    DOWNLOAD_PATH,
    MODEL_INPUT_DIR,
    RAW_STATE_COLUMN,
    REQUIRED_QUERY_COLUMNS,
    TARGET,
    resolve_data_config_path,
)




DEFAULT_OUTPUT_DIR = Path(MODEL_INPUT_DIR)
MODEL_CONTRACT_FILENAME = "feature_contract.csv"
CANDIDATE_POOL_FILENAME = "candidate_pool.txt"


MODEL_CONTRACT_COLUMNS = (
    "column",
    "role",
    "enabled_for_model",
    "availability_in_source",
    "var_type",
    "source_config_row",
    "source_config_cv_ind",
    "source_config_segment",
    "description",
    "source_full_config_sha256",
)




def derive_model_contract(
    download_contract: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    """Build a run-specific contract without reducing the full config."""


    required_keys = {
        "frame",
        "path",
        "projected_columns",
        "model_features",
        "var_types",
    }
    missing_keys = sorted(required_keys.difference(download_contract))
    if missing_keys:
        raise ValueError(f"Download contract is missing keys: {missing_keys}")


    full = download_contract["frame"].copy().reset_index(drop=True)
    full["source_config_row"] = full.index + 2  # header occupies CSV row 1
    projected_columns = [
        str(value).strip().upper()
        for value in download_contract["projected_columns"]
    ]
    candidate_pool = [
        str(value).strip().upper() for value in download_contract["model_features"]
    ]
    if not projected_columns or not candidate_pool:
        raise ValueError("The SQL projection or config-derived candidate pool is empty.")
    if len(projected_columns) != len(set(projected_columns)):
        raise ValueError("The SQL projection contains duplicate column names.")
    if len(candidate_pool) != len(set(candidate_pool)):
        raise ValueError("Enabled ROC config rows contain duplicate column names.")
    missing_required = sorted(set(REQUIRED_QUERY_COLUMNS).difference(projected_columns))
    if missing_required:
        raise ValueError(f"Required SQL fields are missing: {missing_required}")
    missing_candidates = sorted(set(candidate_pool).difference(projected_columns))
    if missing_candidates:
        raise ValueError(
            f"Config candidates are missing from the SQL projection: {missing_candidates}"
        )


    # Disabled duplicate legacy rows may remain in the config. Prefer an
    # enabled row when a selected candidate also has an earlier disabled row.
    lookup: dict[str, pd.Series] = {}
    lookup_rows = full.sort_values(
        "cv_enabled", ascending=False, kind="stable"
    )
    for _, row in lookup_rows.iterrows():
        lookup.setdefault(str(row["normalized_column"]), row)


    config_hash = _sha256(Path(download_contract["path"]))
    candidate_set = set(candidate_pool)
    # The SQL extract puts required structural fields first.  Raw state is both
    # structurally required and a config-enabled model candidate, so scanning
    # SQL order would move it ahead of the ROC ``cv_ind`` sequence.  Keep the
    # maintained CSV as the feature-order authority, then retain SQL order for
    # all non-candidate fields.
    contract_columns = candidate_pool + [
        column for column in projected_columns if column not in candidate_set
    ]
    rows: list[dict[str, Any]] = []
    for column in contract_columns:
        source = lookup.get(column)


        def source_value(field: str) -> str:
            if source is None or field not in source.index:
                return ""
            return str(source[field]).strip()


        if column == TARGET:
            role = "TARGET"
        elif column in candidate_set:
            role = "CANDIDATE"
        else:
            role = "STRUCTURAL"
        rows.append(
            {
                "column": column,
                "role": role,
                "enabled_for_model": "Y" if role == "CANDIDATE" else "N",
                "availability_in_source": "PRESENT",
                "var_type": download_contract["var_types"][column],
                "source_config_row": (
                    int(source["source_config_row"]) if source is not None else ""
                ),
                "source_config_cv_ind": (
                    source_value("cv_ind")
                ),
                "source_config_segment": (
                    source_value("seg")
                ),
                "description": (
                    source_value("description")
                ),
                "source_full_config_sha256": config_hash,
            }
        )


    model_contract = pd.DataFrame(rows, columns=MODEL_CONTRACT_COLUMNS)
    derived_pool = model_contract.loc[
        model_contract["role"].eq("CANDIDATE")
        & model_contract["enabled_for_model"].eq("Y"),
        "column",
    ].tolist()
    if derived_pool != candidate_pool:
        raise ValueError("Derived candidate order differs from the ROC cv_ind order.")
    if model_contract.loc[model_contract["role"].eq("TARGET"), "column"].tolist() != [
        TARGET
    ]:
        raise ValueError(f"The model contract must contain one target: {TARGET}.")
    return model_contract, candidate_pool




def _csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")




def _atomic_write_if_changed(path: Path, content: bytes) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() == content:
        return "unchanged"
    if path.exists() and not path.is_file():
        raise IsADirectoryError(f"Expected a file output path; found: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return "written"




def prepare_model_inputs(
    *,
    input_path: str | Path = DOWNLOAD_PATH,
    input_metadata_path: str | Path = DOWNLOAD_METADATA_PATH,
    config_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Verify B1 output and publish the dynamic model contract and pool."""


    resolved_config_path = resolve_data_config_path(config_path)
    download_contract = load_download_contract(resolved_config_path)
    source_metadata = verify_published_artifacts(
        input_path, input_metadata_path, download_contract
    )
    model_contract, candidate_pool = derive_model_contract(download_contract)


    destination = Path(output_dir).resolve()
    contract_path = destination / MODEL_CONTRACT_FILENAME
    pool_path = destination / CANDIDATE_POOL_FILENAME
    contract_action = _atomic_write_if_changed(
        contract_path, _csv_bytes(model_contract)
    )
    pool_action = _atomic_write_if_changed(
        pool_path, ("\n".join(candidate_pool) + "\n").encode("utf-8")
    )
    return {
        "source_parquet": str(Path(input_path).resolve()),
        "source_metadata": str(Path(input_metadata_path).resolve()),
        "source_row_count": int(source_metadata["row_count"]),
        "source_full_config_rows": len(download_contract["frame"]),
        "source_full_config_sha256": _sha256(download_contract["path"]),
        "profile": source_metadata.get(
            "profile", source_metadata.get("profile_label")
        ),
        "profile_hash": source_metadata["profile_hash"],
        "profile_slug": source_metadata["profile_slug"],
        "model_contract": str(contract_path),
        "model_contract_rows": int(len(model_contract)),
        "model_contract_action": contract_action,
        "candidate_pool": str(pool_path),
        "candidate_pool_count": int(len(candidate_pool)),
        "candidate_pool_action": pool_action,
        "raw_state_queried": RAW_STATE_COLUMN in model_contract["column"].tolist(),
        "raw_state_enabled_by_roc_cv_ind": RAW_STATE_COLUMN in candidate_pool,
    }




def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(DOWNLOAD_PATH))
    parser.add_argument(
        "--input-metadata", type=Path, default=Path(DOWNLOAD_METADATA_PATH)
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "ROC config CSV. If omitted, use RETENTION_DATA_CONFIG_PATH or "
            "the only CSV beneath the active HOME config directory."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)




def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    summary = prepare_model_inputs(
        input_path=args.input,
        input_metadata_path=args.input_metadata,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print("RETENTION 13/1 MODEL INPUTS PREPARED")
    print(json.dumps(summary, indent=2))
    return summary




if __name__ == "__main__":
    main()
