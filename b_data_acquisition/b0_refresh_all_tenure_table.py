# -*- coding: utf-8 -*-
"""Explicitly rebuild the existing CY table with all mature tenure terms.


Routine model runs never execute this file. Root ``a_business_understanding/a0_one_click.py`` adds it
only when ``--refresh-source`` is supplied. The checked-in SQL owns the full
``CREATE OR REPLACE`` statement; this runner extracts its candidate relation
for read-only preflight and executes that exact rendered DDL only after every
guard passes.
"""


from __future__ import annotations


from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable


import pandas as pd




PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from params import (  # noqa: E402
    ALL_TENURE_BUILD_SQL_PATH,
    MAIN_TABLE_FQN,
    MAX_ANNUAL_TERM_DAYS,
    MIN_ANNUAL_TERM_DAYS,
    LATEST_ANCHOR_TERM_START_DATE_EXCLUSIVE,
    PIPELINE_OUTPUT_ROOT,
    SF_CONN,
    SOURCE_TABLE_FQN,
    TRUSTED_DATA_AS_OF_DATE,
    new_run_id,
    setup_params,
)




REQUIRED_ZERO_COLUMNS = (
    "NULL_POLICY_KEY_COUNT",
    "NULL_SELECTED_INCEPTION_COUNT",
    "NULL_TECHNICAL_INDEX_COUNT",
    "NULL_ANCHOR_START_COUNT",
    "NULL_ANCHOR_TENURE_COUNT",
    "NULL_ANCHOR_TERM_END_COUNT",
    "NULL_MONTH1_DATE_COUNT",
    "NULL_MONTH13_DATE_COUNT",
    "TENURE_AMBIGUITY_COUNT",
    "TERM_END_AMBIGUITY_COUNT",
    "REPRESENTATIVE_TENURE_FAILURE_COUNT",
    "REPRESENTATIVE_TERM_FAILURE_COUNT",
    "MONTH1_DATE_FAILURE_COUNT",
    "MONTH13_DATE_FAILURE_COUNT",
    "FEATURE_AT_MONTH1_FAILURE_COUNT",
    "MONTH1_ELIGIBILITY_FAILURE_COUNT",
    "TARGET_FAILURE_COUNT",
    "TARGET_INTERVAL_RECOMPUTATION_FAILURE_COUNT",
    "FIXED_HORIZON_FAILURE_COUNT",
    "MATURITY_FAILURE_COUNT",
    "INVALID_TENURE_GROUP_COUNT",
    "SPECIAL_TENURE_CROSSWALK_FAILURE_COUNT",
    "INVALID_DED_FINAL_COUNT",
    "DUPLICATE_KEY_GROUPS",
    "DUPLICATE_TECHNICAL_INDEX_GROUPS",
    "TENURE0_TARGET_REGRESSION_COUNT",
)


SOURCE_ZERO_COLUMNS = (
    "NULL_POLICY_KEY_COUNT",
    "NULL_ANCHOR_START_COUNT",
    "NULL_ANCHOR_TENURE_COUNT",
    "NULL_ANCHOR_TERM_END_COUNT",
    "INVALID_OR_NULL_SOURCE_TENURE_GROUP_COUNT",
    "TENURE_AMBIGUITY_COUNT",
    "TERM_END_AMBIGUITY_COUNT",
    "INVALID_TENURE_GROUP_COUNT",
    "SPECIAL_TENURE_CROSSWALK_FAILURE_COUNT",
    "FIXED_HORIZON_FAILURE_COUNT",
    "FEATURE_ROW_TIE_GROUPS",
    "TERM_SOURCE_ROW_TIE_GROUPS",
    "TERM_DEDUCTIBLE_PARSE_FAILURE_COUNT",
)


INFORMATIONAL_QA_COLUMNS = (
    "CROSS_INCEPTION_RESCUE_COUNT",
    "MULTIPLE_INCEPTION_COVERAGE_COUNT",
    "TENURE0_OLD_DUPLICATE_GROUPS",
    "TENURE0_NEW_DUPLICATE_GROUPS",
    "TENURE0_OLD_ONLY_ROWS",
    "TENURE0_NEW_ONLY_ROWS",
    "TENURE0_TECHNICAL_INDEX_MISMATCH_COUNT",
    "TENURE0_TERM_START_MISMATCH_COUNT",
    "TENURE0_TERM_END_MISMATCH_COUNT",
    "TENURE0_MONTH1_MISMATCH_COUNT",
    "TENURE0_MONTH13_MISMATCH_COUNT",
    "TENURE0_TARGET_RESCUE_COUNT",
    "TENURE0_NB_MAX_EXPOSURE_END_MISMATCH_COUNT",
    "TENURE0_NB_MIN_EXPOSURE_START_MISMATCH_COUNT",
    "TENURE0_NB_TERM_START_MISMATCH_COUNT",
    "TENURE0_NB_TERM_END_MISMATCH_COUNT",
)


# Only candidate-intrinsic diagnostics are stable before and after replacement.
# TENURE0_* compares against the table that exists at query time, so it cannot
# enter the exact preflight/postbuild identity summary after same-name CTAS.
STABLE_INFORMATIONAL_QA_COLUMNS = (
    "CROSS_INCEPTION_RESCUE_COUNT",
    "MULTIPLE_INCEPTION_COVERAGE_COUNT",
)


RAW_PROJECTION_MARKER = "/*__RAW_HISTORY_PROJECTION__*/"
FINAL_OUTPUT_PROJECTION_MARKER = "/*__FINAL_OUTPUT_PROJECTION__*/"
RETENTION_TIME_MARKER = "/*__DATA_RETENTION_TIME_IN_DAYS__*/"
_COLUMN_RE = re.compile(r"^[A-Z][A-Z0-9_]*")
LEGACY_COMPATIBILITY_ALIASES = {
    "NB_MAX_EXPSR_END_EFF_DT": "TENURE0_POLICY_MAX_EXPOSURE_END",
    "NB_MIN_EXPSR_EFF_DT": "TENURE0_POLICY_MIN_EXPOSURE_START",
    "NB_POLICY_TERM_END_DATE": "TENURE0_POLICY_TERM_END",
    "NB_POLICY_TERM_START_DATE": "TENURE0_POLICY_TERM_START",
}
DERIVED_BUILD_COLUMNS = frozenset(
    {
        "RETENTION_13_1_IND",
        "MONTH_1_EVALUATION_DATE",
        "MONTH_13_EVALUATION_DATE",
        "ACTIVE_AT_MONTH_1_IND",
        "ACTIVE_AT_MONTH_13_IND",
        "ANCHOR_TERM_START_DATE",
        "ANCHOR_TERM_END_DATE",
        "ANCHOR_TENURE_YEARS_GRP",
        "ANCHOR_TENURE_YEARS_INT",
        "ANCHOR_TENURE_VALUE_COUNT",
        "ANCHOR_TERM_END_VALUE_COUNT",
        "ANCHOR_MIN_EXPSR_EFF_DT",
        "POLICY_MAX_EXPSR_END_EFF_DT",
        "DED_RAW",
        "DED_PROCESS",
        "DED_FINAL",
        "WRITTEN_PREM",
        "PREV_WRITTEN_PREM",
        "RATE_CHANGE",
        "RATE_CHANGE2",
        *LEGACY_COMPATIBILITY_ALIASES,
        "__RAW_TENURE_GROUP_VALUE",
        "__RAW_TENURE_YEARS_INT",
        "__CANONICAL_TENURE_GROUP",
        "__TERM_SOURCE_ROW_NUMBER",
        "__ANCHOR_ROW_NUMBER",
    }
)
DERIVED_OUTPUT_TYPES = {
    "ANCHOR_TERM_START_DATE": "DATE",
    "ANCHOR_TERM_END_DATE": "DATE",
    "ANCHOR_TENURE_YEARS_GRP": "VARCHAR",
    # Passive compatibility output only; group selection never reads it.
    "ANCHOR_TENURE_YEARS_INT": "FLOAT",
    "ANCHOR_TENURE_VALUE_COUNT": "NUMBER",
    "ANCHOR_TERM_END_VALUE_COUNT": "NUMBER",
    "ANCHOR_MIN_EXPSR_EFF_DT": "DATE",
    "POLICY_MAX_EXPSR_END_EFF_DT": "DATE",
    "DED_RAW": "VARCHAR",
    "DED_PROCESS": "NUMBER",
    "DED_FINAL": "VARCHAR",
    "WRITTEN_PREM": "NUMBER",
    "PREV_WRITTEN_PREM": "NUMBER",
    "RATE_CHANGE": "NUMBER",
    "RATE_CHANGE2": "NUMBER",
    "MONTH_1_EVALUATION_DATE": "DATE",
    "MONTH_13_EVALUATION_DATE": "DATE",
    "NB_MAX_EXPSR_END_EFF_DT": "DATE",
    "NB_MIN_EXPSR_EFF_DT": "DATE",
    "NB_POLICY_TERM_END_DATE": "DATE",
    "NB_POLICY_TERM_START_DATE": "DATE",
    "ACTIVE_AT_MONTH_1_IND": "NUMBER",
    "ACTIVE_AT_MONTH_13_IND": "NUMBER",
    "RETENTION_13_1_IND": "NUMBER",
}
DERIVED_OUTPUT_COLUMNS = tuple(DERIVED_OUTPUT_TYPES)
OPTIONAL_LEGACY_OUTPUT_COLUMNS = frozenset({"ANCHOR_TENURE_YEARS_INT"})
BUILD_SOURCE_REQUIRED_COLUMNS = frozenset(
    {
        "EXPOSURE_ROW_ID",
        "POLICY_CONTRACT_ID",
        "POLICY_INCEPTION_DATE",
        "POLICY_TERM_START_DATE",
        "POLICY_TERM_END_DATE",
        "EXPOSURE_START_DATE",
        "EXPOSURE_END_DATE",
        "TENURE_YEARS_GRP",
        "TENURE_YEARS_INT",
        "DED_ALLPERIL_OFST_NOTREND",
        "COV_A_DWELLING_CVG_AMT",
        "EARNED_EXPOSURE_VALUE",
        "TOT_ERND_PREM_AMT",
    }
)




def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()




def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()




def _strip_leading_sql_comments(value: str) -> str:
    result = value.lstrip()
    while True:
        block = re.match(r"/\*.*?\*/", result, flags=re.DOTALL)
        if block is not None:
            result = result[block.end() :].lstrip()
            continue
        line = re.match(r"--[^\r\n]*(?:\r?\n|$)", result)
        if line is not None:
            result = result[line.end() :].lstrip()
            continue
        return result




def _safe_column(value: object) -> str:
    column = str(value).strip().upper()
    if not _COLUMN_RE.fullmatch(column):
        raise ValueError(f"Unsafe Snowflake column name in schema: {value!r}.")
    return column




def _schema_records(frame: pd.DataFrame, label: str) -> list[dict[str, str]]:
    result = _normalized(frame, label)
    required = {"COLUMN_NAME", "DATA_TYPE"}
    if not required.issubset(result.columns):
        raise ValueError(f"{label} is missing {sorted(required)}.")
    if "ORDINAL_POSITION" in result.columns:
        result = result.sort_values("ORDINAL_POSITION", kind="stable")
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for _, row in result.iterrows():
        column = _safe_column(row["COLUMN_NAME"])
        if column in seen:
            raise ValueError(f"{label} contains duplicate column {column}.")
        seen.add(column)
        records.append(
            {"column": column, "data_type": str(row["DATA_TYPE"]).strip().upper()}
        )
    if not records:
        raise ValueError(f"{label} contains no columns.")
    return records




def raw_projection_columns(
    source_schema: pd.DataFrame | None = None,
) -> list[str]:
    """Return every safe frozen-source column except fields recomputed here.


    A live refresh always supplies ``source_schema``.  The config fallback is
    only for offline rendering/tests; it is never used to decide a destructive
    live projection.
    """


    if source_schema is not None:
        candidates = [
            record["column"]
            for record in _schema_records(source_schema, "source schema")
        ]
    else:
        from b_data_acquisition.b1_download_retention_13_1 import (
            load_download_contract,
        )
        from params import resolve_data_config_path


        contract = load_download_contract(resolve_data_config_path())
        candidates = list(contract["all_config_columns"])
        # Offline rendering must still carry physical-build inputs that are
        # structural or used only to derive governed outputs. Live refreshes
        # always use the actual source schema above.
        for required_column in sorted(BUILD_SOURCE_REQUIRED_COLUMNS):
            if required_column not in candidates:
                candidates.append(required_column)


    columns: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        column = _safe_column(value)
        if column in DERIVED_BUILD_COLUMNS or column in seen:
            continue
        columns.append(column)
        seen.add(column)
    missing = sorted(BUILD_SOURCE_REQUIRED_COLUMNS.difference(columns))
    if missing:
        raise ValueError(f"Raw source projection is missing build fields: {missing}.")
    return columns




def final_output_columns(
    raw_columns: Iterable[str],
    current_schema: pd.DataFrame | None = None,
) -> list[str]:
    """Preserve live column order, then append genuinely new build outputs."""


    raw = [_safe_column(column) for column in raw_columns]
    available = [*raw, *DERIVED_OUTPUT_COLUMNS]
    if len(available) != len(set(available)):
        duplicates = sorted(
            column for column in set(available) if available.count(column) > 1
        )
        raise ValueError(f"Candidate output columns are duplicated: {duplicates}.")
    if current_schema is None:
        return [
            column
            for column in available
            if column not in OPTIONAL_LEGACY_OUTPUT_COLUMNS
        ]
    current = [
        record["column"]
        for record in _described_schema_records(current_schema, "current CY schema")
    ]
    missing = sorted(set(current).difference(available))
    if missing:
        raise ValueError(
            "The candidate relation cannot preserve current CY columns: "
            f"{missing}."
        )
    current_set = set(current)
    return [
        *current,
        *(
            column
            for column in available
            if column not in current_set
            and column not in OPTIONAL_LEGACY_OUTPUT_COLUMNS
        ),
    ]




def load_build_sql(
    path: str | Path = ALL_TENURE_BUILD_SQL_PATH,
    *,
    raw_columns: Iterable[str] | None = None,
    output_columns: Iterable[str] | None = None,
) -> str:
    sql_path = Path(path).resolve()
    if not sql_path.is_file():
        raise FileNotFoundError(f"All-tenure build SQL is missing: {sql_path}")
    source_text = sql_path.read_text(encoding="utf-8").strip()
    source_text = source_text.replace(
        "INSURANCE_ANALYTICS.RETENTION.POLICY_EXPOSURE_HISTORY", SOURCE_TABLE_FQN
    ).replace(
        "INSURANCE_ANALYTICS.RETENTION.POLICY_RETENTION_13_1", MAIN_TABLE_FQN
    )
    if source_text.count(RAW_PROJECTION_MARKER) != 1:
        raise ValueError(
            "Build SQL must contain exactly one raw-history projection marker."
        )
    if source_text.count(FINAL_OUTPUT_PROJECTION_MARKER) != 1:
        raise ValueError(
            "Build SQL must contain exactly one final-output projection marker."
        )
    if source_text.count(RETENTION_TIME_MARKER) != 1:
        raise ValueError(
            "Build SQL must contain exactly one data-retention marker."
        )
    projection_columns = (
        raw_projection_columns()
        if raw_columns is None
        else [_safe_column(column) for column in raw_columns]
    )
    if not projection_columns or len(projection_columns) != len(
        set(projection_columns)
    ):
        raise ValueError("Rendered raw source projection is empty or duplicated.")
    forbidden = sorted(set(projection_columns).intersection(DERIVED_BUILD_COLUMNS))
    if forbidden:
        raise ValueError(f"Recomputed columns cannot enter RAW_HISTORY: {forbidden}.")
    projection = ",\n        ".join(
        f'"{column}"' for column in projection_columns
    )
    source_text = source_text.replace(RAW_PROJECTION_MARKER, projection)
    final_columns = (
        final_output_columns(projection_columns)
        if output_columns is None
        else [_safe_column(column) for column in output_columns]
    )
    expected_outputs = set(final_output_columns(projection_columns))
    if (
        not final_columns
        or len(final_columns) != len(set(final_columns))
        or set(final_columns) != expected_outputs
    ):
        raise ValueError(
            "Final output projection must contain every raw and derived output "
            "exactly once."
        )
    final_projection = ",\n    ".join(f'"{column}"' for column in final_columns)
    source_text = source_text.replace(
        FINAL_OUTPUT_PROJECTION_MARKER, final_projection
    )
    sql = _strip_leading_sql_comments(source_text)
    relation = candidate_relation(sql)
    normalized = " ".join(relation.upper().split())
    required = (
        f"FROM {SOURCE_TABLE_FQN}",
        "ACTIVE_AT_MONTH_13_IND AS RETENTION_13_1_IND",
        "DATEADD(DAY, 30, T.ANCHOR_TERM_START_DATE)",
        "DATEADD(DAY, 30, T.ANCHOR_TERM_END_DATE)",
    )
    missing = [token for token in required if token not in normalized]
    if missing:
        raise ValueError(f"Build SQL is missing required contract text: {missing}")
    required_group_contract = (
        "ANCHOR_TENURE_YEARS_GRP IN ('0', '1', '2', '3', '4+')"
    )
    if required_group_contract not in normalized:
        raise ValueError(
            "The physical build must validate the exact tenure-group domain."
        )
    required_tenure_normalization = (
        "T.__RAW_TENURE_GROUP_VALUE AS __CANONICAL_TENURE_GROUP",
        "S.__CANONICAL_TENURE_GROUP AS TENURE_YEARS_GRP",
    )
    missing_normalization = [
        token for token in required_tenure_normalization if token not in normalized
    ]
    if missing_normalization:
        raise ValueError(
            "The physical build is missing the governed tenure-group "
            f"normalization: {missing_normalization}."
        )
    forbidden_integer_overrides = (
        "WHEN T.__RAW_TENURE_YEARS_INT = -1 THEN '0'",
        "WHEN T.__RAW_TENURE_YEARS_INT >= 4 THEN '4+'",
    )
    present_integer_overrides = [
        token for token in forbidden_integer_overrides if token in normalized
    ]
    if present_integer_overrides:
        raise ValueError(
            "TENURE_YEARS_GRP must remain authoritative; integer tenure may "
            "be audited but cannot construct the modeling group: "
            f"{present_integer_overrides}."
        )
    # Resolve internal SQL names from the contract-and-term join.
    # Validate the feature rules independently of internal CTE names.
    term_feature_joins = re.findall(
        r"\bLEFT JOIN ([A-Z][A-Z0-9_$]*) AS F ON "
        r"F\.POLICY_CONTRACT_ID = A\.POLICY_CONTRACT_ID "
        r"AND F\.([A-Z][A-Z0-9_$]*) = A\.ANCHOR_TERM_START_DATE\b",
        normalized,
    )
    if len(term_feature_joins) != 1:
        raise ValueError(
            "The table build must join one term-feature relation to the "
            "anchor rows by contract number and term start date."
        )
    term_features_cte, term_start_column = term_feature_joins[0]
    required_term_feature_logic = (
        "PARTITION BY R.POLICY_CONTRACT_ID, R.POLICY_TERM_START_DATE::DATE",
        "R.EXPOSURE_START_DATE::DATE ASC NULLS LAST",
        "SPLIT_PART(M.DED_ALLPERIL_OFST_NOTREND, '_', 1) AS DED_RAW",
        "WHEN CONTAINS(B.DED_RAW, '%')",
        "B.COV_A_DWELLING_CVG_AMT * TO_NUMBER(REPLACE(B.DED_RAW, '%', '')) / 100",
        "ELSE TO_NUMBER(B.DED_RAW) END AS DED_PROCESS",
        "WHEN P.DED_PROCESS <= 500 THEN '000500'",
        "WHEN P.DED_PROCESS <= 10000 THEN '010000'",
        "WHEN P.DED_PROCESS > 10000 THEN '10000+'",
        "WHEN M.EARNED_EXPOSURE_VALUE <= 0 THEN 0",
        "ELSE M.TOT_ERND_PREM_AMT / M.EARNED_EXPOSURE_VALUE END AS WRITTEN_PREM",
        "LAG(P.WRITTEN_PREM) OVER",
        f"PARTITION BY P.POLICY_CONTRACT_ID ORDER BY P.{term_start_column}",
        "WHEN G.PREV_WRITTEN_PREM IS NULL THEN NULL",
        "WHEN G.PREV_WRITTEN_PREM IS NULL THEN -100",
        "WHEN G.WRITTEN_PREM = 0 THEN NULL",
        "(G.WRITTEN_PREM - G.PREV_WRITTEN_PREM) / G.WRITTEN_PREM * 100",
        "F.RATE_CHANGE",
        "F.RATE_CHANGE2",
        f"LEFT JOIN {term_features_cte} AS F",
    )
    missing_term_feature_logic = [
        token for token in required_term_feature_logic if token not in normalized
    ]
    if missing_term_feature_logic:
        raise ValueError(
            "The table build is missing required deductible, premium, or rate-change "
            f"logic: {missing_term_feature_logic}."
        )
    term_features_position = normalized.find(f"{term_features_cte} AS (")
    if term_features_position < 0:
        raise ValueError("The joined term-feature CTE is not defined in the build SQL.")
    if term_features_position > normalized.index("VALID_MATURE_ANCHOR_TERMS AS"):
        raise ValueError(
            "PREV_WRITTEN_PREM must be constructed across all source terms "
            "before target-maturity filtering."
        )
    if "TENURE_YEARS_GRP NOT IN" in normalized:
        raise ValueError("The physical all-tenure build must not exclude a profile.")
    if "PROPERTY_STATE NOT IN" in normalized:
        raise ValueError("State scope must be inherited from the frozen source.")
    expected_literal_counts = {
        f"DATE '{TRUSTED_DATA_AS_OF_DATE}'": 3,
        f"DATE '{LATEST_ANCHOR_TERM_START_DATE_EXCLUSIVE}'": 1,
    }
    for literal, expected_count in expected_literal_counts.items():
        observed_count = normalized.count(literal)
        if observed_count != expected_count:
            raise ValueError(
                f"Build SQL date contract drift for {literal}: expected "
                f"{expected_count}, found {observed_count}."
            )
    return sql




def build_source_schema_sql() -> str:
    database, schema, table = SOURCE_TABLE_FQN.split(".")
    return f"""
SELECT COLUMN_NAME, DATA_TYPE, ORDINAL_POSITION
FROM {database}.INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_CATALOG = '{database}'
  AND TABLE_SCHEMA = '{schema}'
  AND TABLE_NAME = '{table}'
ORDER BY ORDINAL_POSITION
""".strip()




def build_target_schema_sql() -> str:
    """Describe the live table with full type details and physical order."""


    return f"DESCRIBE TABLE {MAIN_TABLE_FQN}"




def build_candidate_schema_probe_sql(build_sql: str) -> str:
    """Compile the candidate relation without scanning or returning rows."""


    return f"{candidate_relation(build_sql)}\nLIMIT 0"




def build_describe_result_sql(query_id: object) -> str:
    """Describe one explicitly identified probe result, never LAST_QUERY_ID."""


    value = str(query_id).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"Unsafe Snowflake query ID: {query_id!r}.")
    return f"DESCRIBE RESULT '{value}'"




def validate_source_schema(frame: pd.DataFrame) -> dict[str, int]:
    records = _schema_records(frame, "source schema")
    available = {record["column"] for record in records}
    missing = sorted(BUILD_SOURCE_REQUIRED_COLUMNS.difference(available))
    if missing:
        raise ValueError(
            "The frozen source is missing fields required to construct the "
            f"all-tenure population: {missing}."
        )
    unsupported_hash_types = sorted(
        record["column"]
        for record in records
        if record["data_type"] in {"GEOGRAPHY", "GEOMETRY"}
    )
    if unsupported_hash_types:
        raise ValueError(
            "The content-fingerprint QA cannot hash source geospatial fields: "
            f"{unsupported_hash_types}. Add an explicitly reviewed exclusion "
            "or canonical representation before refresh."
        )
    projected = raw_projection_columns(frame)
    return {
        "available_column_count": len(available),
        "projected_raw_column_count": len(projected),
    }




def _described_schema_records(
    frame: pd.DataFrame, label: str
) -> list[dict[str, str]]:
    """Normalize DESCRIBE TABLE / DESCRIBE RESULT output losslessly."""


    result = _normalized(frame, label)
    required = {"NAME", "TYPE", "KIND", "NULL?"}
    if not required.issubset(result.columns):
        raise ValueError(f"{label} is missing {sorted(required)}.")
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for _, row in result.iterrows():
        column = _safe_column(row["NAME"])
        if column in seen:
            raise ValueError(f"{label} contains duplicate column {column}.")
        seen.add(column)
        data_type = re.sub(r"\s+", "", str(row["TYPE"]).strip().upper())
        kind = str(row["KIND"]).strip().upper()
        nullable = str(row["NULL?"]).strip().upper()
        if not data_type or kind != "COLUMN" or nullable not in {"Y", "N"}:
            raise ValueError(
                f"{label} has an invalid descriptor for {column}: "
                f"type={data_type!r}, kind={kind!r}, nullable={nullable!r}."
            )
        records.append(
            {
                "column": column,
                "data_type": data_type,
                "nullable": nullable,
            }
        )
    if not records:
        raise ValueError(f"{label} contains no columns.")
    return records




def _schema_signature(records: Iterable[dict[str, str]]) -> str:
    return _sha256_text(
        json.dumps(list(records), sort_keys=True, separators=(",", ":"))
    )




def validate_schema_preservation(
    current_schema: pd.DataFrame,
    candidate_schema: pd.DataFrame,
) -> dict[str, Any]:
    """Require exact type/nullability and order for every existing column.


    New all-tenure structural columns are intentional and may be inserted, but
    the ordered sequence of all pre-existing columns must remain unchanged.
    The candidate metadata comes from Snowflake's own LIMIT 0 result, not a
    Python prediction of CTAS inference.
    """


    current = _described_schema_records(current_schema, "current CY schema")
    candidate = _described_schema_records(candidate_schema, "candidate schema")
    current_by_name = {record["column"]: record for record in current}
    candidate_by_name = {record["column"]: record for record in candidate}
    removed = sorted(set(current_by_name).difference(candidate_by_name))
    changed = {
        column: {
            "current": current_by_name[column],
            "candidate": candidate_by_name[column],
        }
        for column in sorted(set(current_by_name).intersection(candidate_by_name))
        if current_by_name[column] != candidate_by_name[column]
    }
    current_order = [record["column"] for record in current]
    candidate_existing_order = [
        record["column"]
        for record in candidate
        if record["column"] in current_by_name
    ]
    reordered = candidate_existing_order != current_order
    if removed or changed or reordered:
        raise ValueError(
            "Same-name refresh would not preserve the exact current CY "
            "schema/type/order: "
            + json.dumps(
                {
                    "removed_columns": removed,
                    "type_or_nullability_changes": changed,
                    "current_column_order": current_order if reordered else [],
                    "candidate_existing_column_order": (
                        candidate_existing_order if reordered else []
                    ),
                },
                sort_keys=True,
            )
        )
    added = [
        record["column"]
        for record in candidate
        if record["column"] not in current_by_name
    ]
    return {
        "current_column_count": len(current),
        "candidate_column_count": len(candidate),
        "preserved_column_count": len(current),
        "removed_column_count": 0,
        "type_change_count": 0,
        "reordered_existing_column_count": 0,
        "added_column_count": len(added),
        "added_columns": added,
        "candidate_schema_sha256": _schema_signature(candidate),
    }




def validate_schema_identity(
    expected_schema: pd.DataFrame,
    observed_schema: pd.DataFrame,
) -> dict[str, Any]:
    """Require the postbuild table to equal the compiled candidate exactly."""


    expected = _described_schema_records(expected_schema, "expected schema")
    observed = _described_schema_records(observed_schema, "observed schema")
    if expected != observed:
        raise ValueError(
            "Postbuild CY schema does not exactly match the compiled candidate: "
            + json.dumps({"expected": expected, "observed": observed})
        )
    return {
        "column_count": len(observed),
        "schema_sha256": _schema_signature(observed),
    }




def candidate_relation(build_sql: str) -> str:
    """Extract and validate the read-only relation from the approved CTAS."""


    sql = _strip_leading_sql_comments(str(build_sql)).strip()
    if sql.endswith(";"):
        sql = sql[:-1].rstrip()
    retention_value = rf"(?:{re.escape(RETENTION_TIME_MARKER)}|[1-9][0-9]*)"
    prefix = (
        rf"\ACREATE\s+OR\s+REPLACE\s+TABLE\s+{re.escape(MAIN_TABLE_FQN)}"
        rf"\s+DATA_RETENTION_TIME_IN_DAYS\s*=\s*{retention_value}"
        rf"\s+COPY\s+GRANTS\s+COPY\s+TAGS\s+AS\s+"
    )
    match = re.match(prefix, sql, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        raise ValueError(
            "Build SQL must own exactly the approved CREATE OR REPLACE TABLE "
            f"{MAIN_TABLE_FQN} CTAS header."
        )
    relation = sql[match.end() :].strip()
    if not relation.upper().startswith("WITH "):
        raise ValueError("The CTAS candidate relation must start with WITH.")
    if ";" in relation:
        raise ValueError("Build SQL must contain exactly one statement.")
    if re.search(
        r"\b(?:CREATE|ALTER|DROP|TRUNCATE|INSERT|UPDATE|DELETE|MERGE)\b",
        relation,
        flags=re.IGNORECASE,
    ):
        raise ValueError("The CTAS candidate must be a read-only WITH/SELECT relation.")
    return relation




def build_replace_sql(
    build_sql: str | None = None,
    *,
    retention_time_days: int,
) -> str:
    """Render the checked-in CTAS after Python preflight passes."""


    template = build_sql or load_build_sql()
    candidate_relation(template)
    value = str(retention_time_days).strip()
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError("Replacement retention must be at least one day.")
    if template.count(RETENTION_TIME_MARKER) != 1:
        raise ValueError("Rendered build SQL requires exactly one retention marker.")
    rendered = template.replace(RETENTION_TIME_MARKER, value)
    unresolved = re.findall(r"/\*__[A-Z0-9_]+__\*/", rendered)
    if unresolved:
        raise ValueError(f"Rendered build SQL has unresolved markers: {unresolved}.")
    candidate_relation(rendered)
    normalized = " ".join(rendered.upper().split())
    target = f"CREATE OR REPLACE TABLE {MAIN_TABLE_FQN}"
    if normalized.count(target) != 1:
        raise ValueError("Rendered build SQL must replace the approved CY exactly once.")
    return rendered




def _candidate_qa_with_clause(relation: str) -> str:
    """Return a Snowflake-safe WITH clause exposing ``CANDIDATE``.


    The physical build relation is itself a WITH query.  Flatten its final
    ``SELECT * FROM ELIGIBLE_ANCHOR_TERMS`` into one additional CTE instead of
    nesting a WITH query inside another CTE.  Post-build QA uses a plain table
    SELECT and takes the simpler wrapping branch.
    """


    if relation.lstrip().upper().startswith("WITH "):
        final_select_start = relation.upper().rfind("\nSELECT\n")
        if final_select_start < 0:
            raise ValueError(
                "The all-tenure build relation must end with "
                "an explicit SELECT from ELIGIBLE_ANCHOR_TERMS."
            )
        final_select = relation[final_select_start:].strip()
        if not re.fullmatch(
            r"SELECT\s+.+?\s+FROM\s+ELIGIBLE_ANCHOR_TERMS",
            final_select,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            raise ValueError(
                "The all-tenure build relation must end with "
                "an explicit SELECT from ELIGIBLE_ANCHOR_TERMS."
            )
        existing_ctes = relation[:final_select_start].rstrip()
        return f"""{existing_ctes},
CANDIDATE AS
(
    {final_select}
),"""
    return f"""WITH CANDIDATE AS
(
{relation}
),"""




def _qa_select(relation: str) -> str:
    with_clause = _candidate_qa_with_clause(relation)
    return f"""
{with_clause}
DUPLICATES AS
(
    SELECT COUNT(*) AS KEY_COUNT
    FROM CANDIDATE
    GROUP BY
        POLICY_CONTRACT_ID,
        ANCHOR_TERM_START_DATE
    HAVING COUNT(*) > 1
),
TECHNICAL_INDEX_DUPLICATES AS
(
    SELECT COUNT(*) AS KEY_COUNT
    FROM CANDIDATE
    GROUP BY EXPOSURE_ROW_ID
    HAVING COUNT(*) > 1
),
FROZEN_INTERVALS AS
(
    SELECT
        POLICY_CONTRACT_ID,
        POLICY_INCEPTION_DATE::DATE AS POLICY_INCEPTION_DATE,
        EXPOSURE_START_DATE::DATE AS EXPOSURE_START_DATE,
        COALESCE(
            EXPOSURE_END_DATE::DATE,
            DATE '{TRUSTED_DATA_AS_OF_DATE}'
        ) AS EXPOSURE_END_DATE
    FROM {SOURCE_TABLE_FQN}
),
MONTH13_COVERAGE_AUDIT AS
(
    SELECT
        C.POLICY_CONTRACT_ID,
        C.POLICY_INCEPTION_DATE::DATE AS POLICY_INCEPTION_DATE,
        C.ANCHOR_TERM_START_DATE,
        C.RETENTION_13_1_IND,
        MAX(IFF(H.POLICY_CONTRACT_ID IS NULL, 0, 1))
            AS SAME_CONTRACT_COVERAGE_IND,
        MAX(IFF(
            H.POLICY_INCEPTION_DATE = C.POLICY_INCEPTION_DATE::DATE,
            1,
            0
        )) AS SAME_INCEPTION_COVERAGE_IND,
        COUNT(DISTINCT H.POLICY_INCEPTION_DATE)
            AS COVERING_INCEPTION_COUNT
    FROM CANDIDATE AS C
    LEFT JOIN FROZEN_INTERVALS AS H
        ON H.POLICY_CONTRACT_ID = C.POLICY_CONTRACT_ID
       AND H.EXPOSURE_START_DATE <= C.MONTH_13_EVALUATION_DATE
       AND H.EXPOSURE_END_DATE >= C.MONTH_13_EVALUATION_DATE
    GROUP BY
        C.POLICY_CONTRACT_ID,
        C.POLICY_INCEPTION_DATE::DATE,
        C.ANCHOR_TERM_START_DATE,
        C.RETENTION_13_1_IND
),
CONTINUITY_QA AS
(
    SELECT
        COALESCE(COUNT_IF(
            RETENTION_13_1_IND <> SAME_CONTRACT_COVERAGE_IND
        ), 0) AS TARGET_INTERVAL_RECOMPUTATION_FAILURE_COUNT,
        COALESCE(COUNT_IF(
            RETENTION_13_1_IND = 1
            AND SAME_INCEPTION_COVERAGE_IND = 0
        ), 0) AS CROSS_INCEPTION_RESCUE_COUNT,
        COALESCE(COUNT_IF(COVERING_INCEPTION_COUNT > 1), 0)
            AS MULTIPLE_INCEPTION_COVERAGE_COUNT
    FROM MONTH13_COVERAGE_AUDIT
),
OLD_NB AS
(
    SELECT
        POLICY_CONTRACT_ID,
        POLICY_INCEPTION_DATE::DATE AS POLICY_INCEPTION_DATE,
        EXPOSURE_ROW_ID,
        POLICY_TERM_START_DATE::DATE AS TERM_START_DATE,
        POLICY_TERM_END_DATE::DATE AS TERM_END_DATE,
        NB_MAX_EXPSR_END_EFF_DT::DATE AS NB_MAX_EXPOSURE_END_DATE,
        NB_MIN_EXPSR_EFF_DT::DATE AS NB_MIN_EXPOSURE_START_DATE,
        NB_POLICY_TERM_START_DATE::DATE AS NB_TERM_START_DATE,
        NB_POLICY_TERM_END_DATE::DATE AS NB_TERM_END_DATE,
        MONTH_1_EVALUATION_DATE::DATE AS MONTH_1_EVALUATION_DATE,
        MONTH_13_EVALUATION_DATE::DATE AS MONTH_13_EVALUATION_DATE,
        RETENTION_13_1_IND,
        1 AS OLD_PRESENT
    FROM {MAIN_TABLE_FQN}
    WHERE UPPER(TRIM(TO_VARCHAR(TENURE_YEARS_GRP))) = '0'
),
NEW_NB AS
(
    SELECT
        POLICY_CONTRACT_ID,
        POLICY_INCEPTION_DATE::DATE AS POLICY_INCEPTION_DATE,
        EXPOSURE_ROW_ID,
        ANCHOR_TERM_START_DATE AS TERM_START_DATE,
        ANCHOR_TERM_END_DATE AS TERM_END_DATE,
        NB_MAX_EXPSR_END_EFF_DT::DATE AS NB_MAX_EXPOSURE_END_DATE,
        NB_MIN_EXPSR_EFF_DT::DATE AS NB_MIN_EXPOSURE_START_DATE,
        NB_POLICY_TERM_START_DATE::DATE AS NB_TERM_START_DATE,
        NB_POLICY_TERM_END_DATE::DATE AS NB_TERM_END_DATE,
        MONTH_1_EVALUATION_DATE::DATE AS MONTH_1_EVALUATION_DATE,
        MONTH_13_EVALUATION_DATE::DATE AS MONTH_13_EVALUATION_DATE,
        RETENTION_13_1_IND,
        1 AS NEW_PRESENT
    FROM CANDIDATE
    WHERE ANCHOR_TENURE_YEARS_GRP = '0'
),
OLD_NB_DUPLICATES AS
(
    SELECT COUNT(*) AS KEY_COUNT
    FROM OLD_NB
    GROUP BY POLICY_CONTRACT_ID, TERM_START_DATE
    HAVING COUNT(*) > 1
),
NEW_NB_DUPLICATES AS
(
    SELECT COUNT(*) AS KEY_COUNT
    FROM NEW_NB
    GROUP BY POLICY_CONTRACT_ID, TERM_START_DATE
    HAVING COUNT(*) > 1
),
TENURE0_PARITY AS
(
    SELECT
        COALESCE(COUNT_IF(N.NEW_PRESENT IS NULL), 0)
            AS OLD_ONLY_ROWS,
        COALESCE(COUNT_IF(O.OLD_PRESENT IS NULL), 0)
            AS NEW_ONLY_ROWS,
        COALESCE(COUNT_IF(
            O.OLD_PRESENT = 1 AND N.NEW_PRESENT = 1
            AND NOT EQUAL_NULL(
                O.EXPOSURE_ROW_ID,
                N.EXPOSURE_ROW_ID
            )
        ), 0) AS TECHNICAL_INDEX_MISMATCH_COUNT,
        COALESCE(COUNT_IF(
            O.OLD_PRESENT = 1 AND N.NEW_PRESENT = 1
            AND NOT EQUAL_NULL(O.TERM_START_DATE, N.TERM_START_DATE)
        ), 0) AS TERM_START_MISMATCH_COUNT,
        COALESCE(COUNT_IF(
            O.OLD_PRESENT = 1 AND N.NEW_PRESENT = 1
            AND NOT EQUAL_NULL(O.TERM_END_DATE, N.TERM_END_DATE)
        ), 0) AS TERM_END_MISMATCH_COUNT,
        COALESCE(COUNT_IF(
            O.OLD_PRESENT = 1 AND N.NEW_PRESENT = 1
            AND NOT EQUAL_NULL(
                O.MONTH_1_EVALUATION_DATE,
                N.MONTH_1_EVALUATION_DATE
            )
        ), 0) AS MONTH1_MISMATCH_COUNT,
        COALESCE(COUNT_IF(
            O.OLD_PRESENT = 1 AND N.NEW_PRESENT = 1
            AND NOT EQUAL_NULL(
                O.MONTH_13_EVALUATION_DATE,
                N.MONTH_13_EVALUATION_DATE
            )
        ), 0) AS MONTH13_MISMATCH_COUNT,
        COALESCE(COUNT_IF(
            O.OLD_PRESENT = 1 AND N.NEW_PRESENT = 1
            AND O.RETENTION_13_1_IND = 1
            AND N.RETENTION_13_1_IND = 0
        ), 0) AS TARGET_REGRESSION_COUNT,
        COALESCE(COUNT_IF(
            O.OLD_PRESENT = 1 AND N.NEW_PRESENT = 1
            AND O.RETENTION_13_1_IND = 0
            AND N.RETENTION_13_1_IND = 1
        ), 0) AS TARGET_RESCUE_COUNT,
        COALESCE(COUNT_IF(
            O.OLD_PRESENT = 1 AND N.NEW_PRESENT = 1
            AND NOT EQUAL_NULL(
                O.NB_MAX_EXPOSURE_END_DATE,
                N.NB_MAX_EXPOSURE_END_DATE
            )
        ), 0) AS NB_MAX_EXPOSURE_END_MISMATCH_COUNT,
        COALESCE(COUNT_IF(
            O.OLD_PRESENT = 1 AND N.NEW_PRESENT = 1
            AND NOT EQUAL_NULL(
                O.NB_MIN_EXPOSURE_START_DATE,
                N.NB_MIN_EXPOSURE_START_DATE
            )
        ), 0) AS NB_MIN_EXPOSURE_START_MISMATCH_COUNT,
        COALESCE(COUNT_IF(
            O.OLD_PRESENT = 1 AND N.NEW_PRESENT = 1
            AND NOT EQUAL_NULL(O.NB_TERM_START_DATE, N.NB_TERM_START_DATE)
        ), 0) AS NB_TERM_START_MISMATCH_COUNT,
        COALESCE(COUNT_IF(
            O.OLD_PRESENT = 1 AND N.NEW_PRESENT = 1
            AND NOT EQUAL_NULL(O.NB_TERM_END_DATE, N.NB_TERM_END_DATE)
        ), 0) AS NB_TERM_END_MISMATCH_COUNT
    FROM OLD_NB AS O
    FULL OUTER JOIN NEW_NB AS N
        ON O.POLICY_CONTRACT_ID = N.POLICY_CONTRACT_ID
       AND O.TERM_START_DATE = N.TERM_START_DATE
)
SELECT
    COUNT(*) AS ROW_COUNT,
    COALESCE(SUM(RETENTION_13_1_IND), 0) AS RETAINED_COUNT,
    HASH_AGG(*) AS CONTENT_HASH,
    COUNT(DISTINCT ANCHOR_TENURE_YEARS_GRP)
        AS DISTINCT_TENURE_GROUP_COUNT,
    COALESCE(COUNT_IF(ANCHOR_TENURE_YEARS_GRP = '0'), 0)
        AS TENURE_GROUP_0_ROWS,
    COALESCE(COUNT_IF(ANCHOR_TENURE_YEARS_GRP = '1'), 0)
        AS TENURE_GROUP_1_ROWS,
    COALESCE(COUNT_IF(ANCHOR_TENURE_YEARS_GRP = '2'), 0)
        AS TENURE_GROUP_2_ROWS,
    COALESCE(COUNT_IF(ANCHOR_TENURE_YEARS_GRP = '3'), 0)
        AS TENURE_GROUP_3_ROWS,
    COALESCE(COUNT_IF(ANCHOR_TENURE_YEARS_GRP = '4+'), 0)
        AS TENURE_GROUP_4_PLUS_ROWS,
    COALESCE(COUNT_IF(
        POLICY_CONTRACT_ID IS NULL
        OR TRIM(TO_VARCHAR(POLICY_CONTRACT_ID)) = ''
    ), 0) AS NULL_POLICY_KEY_COUNT,
    COALESCE(COUNT_IF(POLICY_INCEPTION_DATE IS NULL), 0)
        AS NULL_SELECTED_INCEPTION_COUNT,
    COALESCE(COUNT_IF(
        EXPOSURE_ROW_ID IS NULL
        OR TRIM(TO_VARCHAR(EXPOSURE_ROW_ID)) = ''
    ), 0) AS NULL_TECHNICAL_INDEX_COUNT,
    COALESCE(COUNT_IF(ANCHOR_TERM_START_DATE IS NULL), 0)
        AS NULL_ANCHOR_START_COUNT,
    COALESCE(COUNT_IF(ANCHOR_TENURE_YEARS_GRP IS NULL), 0)
        AS NULL_ANCHOR_TENURE_COUNT,
    COALESCE(COUNT_IF(ANCHOR_TERM_END_DATE IS NULL), 0)
        AS NULL_ANCHOR_TERM_END_COUNT,
    COALESCE(COUNT_IF(MONTH_1_EVALUATION_DATE IS NULL), 0)
        AS NULL_MONTH1_DATE_COUNT,
    COALESCE(COUNT_IF(MONTH_13_EVALUATION_DATE IS NULL), 0)
        AS NULL_MONTH13_DATE_COUNT,
    COALESCE(COUNT_IF(ANCHOR_TENURE_VALUE_COUNT <> 1), 0)
        AS TENURE_AMBIGUITY_COUNT,
    COALESCE(COUNT_IF(ANCHOR_TERM_END_VALUE_COUNT <> 1), 0)
        AS TERM_END_AMBIGUITY_COUNT,
    COALESCE(COUNT_IF(
        UPPER(TRIM(TO_VARCHAR(TENURE_YEARS_GRP))) <>
            ANCHOR_TENURE_YEARS_GRP
    ), 0) AS REPRESENTATIVE_TENURE_FAILURE_COUNT,
    COALESCE(COUNT_IF(
        POLICY_TERM_START_DATE::DATE <> ANCHOR_TERM_START_DATE
        OR POLICY_TERM_END_DATE::DATE <> ANCHOR_TERM_END_DATE
    ), 0) AS REPRESENTATIVE_TERM_FAILURE_COUNT,
    COALESCE(COUNT_IF(
        MONTH_1_EVALUATION_DATE <>
        DATEADD(DAY, 30, ANCHOR_TERM_START_DATE)
    ), 0) AS MONTH1_DATE_FAILURE_COUNT,
    COALESCE(COUNT_IF(
        MONTH_13_EVALUATION_DATE <>
        DATEADD(DAY, 30, ANCHOR_TERM_END_DATE)
    ), 0) AS MONTH13_DATE_FAILURE_COUNT,
    COALESCE(COUNT_IF(
        EXPOSURE_START_DATE::DATE > MONTH_1_EVALUATION_DATE
        OR COALESCE(
            EXPOSURE_END_DATE::DATE,
            DATE '{TRUSTED_DATA_AS_OF_DATE}'
        ) < MONTH_1_EVALUATION_DATE
    ), 0) AS FEATURE_AT_MONTH1_FAILURE_COUNT,
    COALESCE(COUNT_IF(ACTIVE_AT_MONTH_1_IND <> 1), 0)
        AS MONTH1_ELIGIBILITY_FAILURE_COUNT,
    COALESCE(COUNT_IF(
        ACTIVE_AT_MONTH_13_IND NOT IN (0, 1)
        OR RETENTION_13_1_IND <> ACTIVE_AT_MONTH_13_IND
    ), 0) AS TARGET_FAILURE_COUNT,
    (SELECT TARGET_INTERVAL_RECOMPUTATION_FAILURE_COUNT FROM CONTINUITY_QA)
        AS TARGET_INTERVAL_RECOMPUTATION_FAILURE_COUNT,
    (SELECT CROSS_INCEPTION_RESCUE_COUNT FROM CONTINUITY_QA)
        AS CROSS_INCEPTION_RESCUE_COUNT,
    (SELECT MULTIPLE_INCEPTION_COVERAGE_COUNT FROM CONTINUITY_QA)
        AS MULTIPLE_INCEPTION_COVERAGE_COUNT,
    COALESCE(COUNT_IF(
        DATEDIFF(DAY, ANCHOR_TERM_START_DATE, ANCHOR_TERM_END_DATE)
            NOT BETWEEN {MIN_ANNUAL_TERM_DAYS} AND {MAX_ANNUAL_TERM_DAYS}
    ), 0) AS FIXED_HORIZON_FAILURE_COUNT,
    COALESCE(COUNT_IF(
        MONTH_13_EVALUATION_DATE > DATE '{TRUSTED_DATA_AS_OF_DATE}'
    ), 0) AS MATURITY_FAILURE_COUNT,
    COALESCE(COUNT_IF(
        ANCHOR_TENURE_YEARS_GRP NOT IN ('0', '1', '2', '3', '4+')
    ), 0) AS INVALID_TENURE_GROUP_COUNT,
    COALESCE(COUNT_IF(
        TENURE_YEARS_INT IS NOT NULL
        AND MOD(TENURE_YEARS_INT, 1) = 0
        AND TENURE_YEARS_INT >= 0
        AND (
            (
            TENURE_YEARS_INT BETWEEN 0 AND 3
            AND TENURE_YEARS_GRP <>
                TO_VARCHAR(TENURE_YEARS_INT::INTEGER)
            )
            OR (
            TENURE_YEARS_INT >= 4
            AND TENURE_YEARS_GRP <> '4+'
            )
        )
    ), 0) AS SPECIAL_TENURE_CROSSWALK_FAILURE_COUNT,
    COALESCE(COUNT_IF(
        DED_FINAL IS NULL
        OR DED_FINAL NOT IN
           ('000500', '001000', '001500', '002500',
            '005000', '010000', '10000+', 'NA')
    ), 0) AS INVALID_DED_FINAL_COUNT,
    (SELECT COUNT(*) FROM DUPLICATES) AS DUPLICATE_KEY_GROUPS,
    (SELECT COUNT(*) FROM TECHNICAL_INDEX_DUPLICATES)
        AS DUPLICATE_TECHNICAL_INDEX_GROUPS,
    (SELECT COUNT(*) FROM OLD_NB_DUPLICATES)
        AS TENURE0_OLD_DUPLICATE_GROUPS,
    (SELECT COUNT(*) FROM NEW_NB_DUPLICATES)
        AS TENURE0_NEW_DUPLICATE_GROUPS,
    (SELECT OLD_ONLY_ROWS FROM TENURE0_PARITY)
        AS TENURE0_OLD_ONLY_ROWS,
    (SELECT NEW_ONLY_ROWS FROM TENURE0_PARITY)
        AS TENURE0_NEW_ONLY_ROWS,
    (SELECT TECHNICAL_INDEX_MISMATCH_COUNT FROM TENURE0_PARITY)
        AS TENURE0_TECHNICAL_INDEX_MISMATCH_COUNT,
    (SELECT TERM_START_MISMATCH_COUNT FROM TENURE0_PARITY)
        AS TENURE0_TERM_START_MISMATCH_COUNT,
    (SELECT TERM_END_MISMATCH_COUNT FROM TENURE0_PARITY)
        AS TENURE0_TERM_END_MISMATCH_COUNT,
    (SELECT MONTH1_MISMATCH_COUNT FROM TENURE0_PARITY)
        AS TENURE0_MONTH1_MISMATCH_COUNT,
    (SELECT MONTH13_MISMATCH_COUNT FROM TENURE0_PARITY)
        AS TENURE0_MONTH13_MISMATCH_COUNT,
    (SELECT TARGET_REGRESSION_COUNT FROM TENURE0_PARITY)
        AS TENURE0_TARGET_REGRESSION_COUNT,
    (SELECT TARGET_RESCUE_COUNT FROM TENURE0_PARITY)
        AS TENURE0_TARGET_RESCUE_COUNT,
    (SELECT NB_MAX_EXPOSURE_END_MISMATCH_COUNT FROM TENURE0_PARITY)
        AS TENURE0_NB_MAX_EXPOSURE_END_MISMATCH_COUNT,
    (SELECT NB_MIN_EXPOSURE_START_MISMATCH_COUNT FROM TENURE0_PARITY)
        AS TENURE0_NB_MIN_EXPOSURE_START_MISMATCH_COUNT,
    (SELECT NB_TERM_START_MISMATCH_COUNT FROM TENURE0_PARITY)
        AS TENURE0_NB_TERM_START_MISMATCH_COUNT,
    (SELECT NB_TERM_END_MISMATCH_COUNT FROM TENURE0_PARITY)
        AS TENURE0_NB_TERM_END_MISMATCH_COUNT
FROM CANDIDATE
""".strip()




def build_preflight_qa_sql(build_sql: str | None = None) -> str:
    return _qa_select(candidate_relation(build_sql or load_build_sql()))




def build_source_anchor_qa_sql() -> str:
    """Audit every source anchor group, including rows later deemed ineligible.


    Term-source checks use the earliest exposure record for each contract and
    term start. They flag tied earliest records and unparseable deductibles.
    """


    return f"""
WITH SOURCE_TENURE_INPUTS AS
(
    SELECT
        S.*,
        UPPER(TRIM(TO_VARCHAR(S.TENURE_YEARS_GRP)))
            AS __RAW_TENURE_GROUP_VALUE,
        S.TENURE_YEARS_INT AS __RAW_TENURE_YEARS_INT
    FROM {SOURCE_TABLE_FQN} AS S
),
SOURCE_ROWS AS
(
    SELECT
        S.*,
        S.__RAW_TENURE_GROUP_VALUE AS __CANONICAL_TENURE_GROUP
    FROM SOURCE_TENURE_INPUTS AS S
),
SOURCE_TENURE_QUALITY AS
(
    SELECT
        COALESCE(COUNT_IF(
            __RAW_TENURE_GROUP_VALUE IS NULL
            OR __RAW_TENURE_GROUP_VALUE NOT IN ('0', '1', '2', '3', '4+')
        ), 0) AS INVALID_OR_NULL_SOURCE_TENURE_GROUP_COUNT,
        COALESCE(COUNT_IF(
            TENURE_YEARS_INT IS NOT NULL
            AND
            (
                __RAW_TENURE_YEARS_INT IS NULL
                OR MOD(__RAW_TENURE_YEARS_INT, 1) <> 0
            )
        ), 0) AS INVALID_NON_NULL_SOURCE_TENURE_INT_COUNT
    FROM SOURCE_ROWS
),
ANCHOR_GROUPS AS
(
    SELECT
        POLICY_CONTRACT_ID,
        POLICY_TERM_START_DATE::DATE AS ANCHOR_TERM_START_DATE,
        MIN(__CANONICAL_TENURE_GROUP)
            AS ANCHOR_TENURE_YEARS_GRP,
        COUNT(DISTINCT __CANONICAL_TENURE_GROUP)
            + IFF(
                COALESCE(COUNT_IF(__CANONICAL_TENURE_GROUP IS NULL), 0) > 0,
                1,
                0
            )
            AS TENURE_VALUE_COUNT,
        MAX(POLICY_TERM_END_DATE)::DATE AS ANCHOR_TERM_END_DATE,
        COUNT(DISTINCT POLICY_TERM_END_DATE::DATE)
            + IFF(
                COALESCE(COUNT_IF(POLICY_TERM_END_DATE IS NULL), 0) > 0,
                1,
                0
            )
            AS TERM_END_VALUE_COUNT
    FROM SOURCE_ROWS
    GROUP BY
        POLICY_CONTRACT_ID,
        POLICY_TERM_START_DATE::DATE
),
SPECIAL_TENURE_CROSSWALK AS
(
    SELECT
        COALESCE(COUNT_IF(
            __RAW_TENURE_YEARS_INT IS NOT NULL
            AND MOD(__RAW_TENURE_YEARS_INT, 1) = 0
            AND __RAW_TENURE_YEARS_INT >= 0
            AND
            (
                (
                    __RAW_TENURE_YEARS_INT BETWEEN 0 AND 3
                    AND __CANONICAL_TENURE_GROUP <>
                        TO_VARCHAR(__RAW_TENURE_YEARS_INT::INTEGER)
                )
                OR
                (
                    __RAW_TENURE_YEARS_INT >= 4
                    AND __CANONICAL_TENURE_GROUP <> '4+'
                )
            )
        ), 0) AS FAILURE_COUNT
    FROM SOURCE_ROWS
),
TERM_SOURCE_ROW_RANKS AS
(
    SELECT
        POLICY_CONTRACT_ID,
        POLICY_TERM_START_DATE::DATE AS TERM_START_DATE,
        DENSE_RANK() OVER
        (
            PARTITION BY
                POLICY_CONTRACT_ID,
                POLICY_TERM_START_DATE::DATE
            ORDER BY EXPOSURE_START_DATE::DATE ASC NULLS LAST
        ) AS TERM_SOURCE_ORDER_RANK
    FROM SOURCE_ROWS
    WHERE POLICY_TERM_START_DATE IS NOT NULL
),
TERM_SOURCE_ROW_TIES AS
(
    SELECT
        POLICY_CONTRACT_ID,
        TERM_START_DATE
    FROM TERM_SOURCE_ROW_RANKS
    WHERE TERM_SOURCE_ORDER_RANK = 1
    GROUP BY 1, 2
    HAVING COUNT(*) > 1
),
TERM_SOURCE_ROWS AS
(
    SELECT
        S.POLICY_CONTRACT_ID,
        S.POLICY_TERM_START_DATE::DATE AS TERM_START_DATE,
        SPLIT_PART(S.DED_ALLPERIL_OFST_NOTREND, '_', 1) AS DED_RAW,
        ROW_NUMBER() OVER
        (
            PARTITION BY
                S.POLICY_CONTRACT_ID,
                S.POLICY_TERM_START_DATE::DATE
            ORDER BY
                S.POLICY_TERM_START_DATE::DATE,
                S.EXPOSURE_START_DATE::DATE ASC NULLS LAST,
                TO_VARCHAR(S.EXPOSURE_ROW_ID) ASC NULLS LAST
        ) AS TERM_SOURCE_ROW_NUMBER
    FROM SOURCE_ROWS AS S
    WHERE S.POLICY_TERM_START_DATE IS NOT NULL
),
TERM_DEDUCTIBLE_PARSE_FAILURES AS
(
    SELECT COUNT(*) AS FAILURE_COUNT
    FROM TERM_SOURCE_ROWS
    WHERE TERM_SOURCE_ROW_NUMBER = 1
      AND DED_RAW IS NOT NULL
      AND TRY_TO_NUMBER(REPLACE(DED_RAW, '%', '')) IS NULL
),
ACTIVE_FEATURE_ROWS AS
(
    SELECT
        S.POLICY_CONTRACT_ID,
        S.POLICY_INCEPTION_DATE::DATE AS POLICY_INCEPTION_DATE,
        S.POLICY_TERM_START_DATE::DATE AS ANCHOR_TERM_START_DATE,
        S.EXPOSURE_START_DATE::DATE AS EXPOSURE_START_DATE,
        S.EXPOSURE_END_DATE::DATE AS EXPOSURE_END_DATE,
        COUNT(*) OVER
        (
            PARTITION BY
                S.POLICY_CONTRACT_ID,
                S.POLICY_TERM_START_DATE::DATE,
                S.EXPOSURE_START_DATE::DATE,
                S.EXPOSURE_END_DATE::DATE
        ) AS TIED_ROW_COUNT,
        MAX(S.EXPOSURE_START_DATE::DATE) OVER
        (
            PARTITION BY
                S.POLICY_CONTRACT_ID,
                S.POLICY_TERM_START_DATE::DATE
        ) AS LATEST_ACTIVE_EFFECTIVE_DATE
    FROM SOURCE_ROWS AS S
    WHERE S.POLICY_TERM_START_DATE IS NOT NULL
      AND S.EXPOSURE_START_DATE::DATE <=
          DATEADD(DAY, 30, S.POLICY_TERM_START_DATE::DATE)
      AND COALESCE(
          S.EXPOSURE_END_DATE::DATE,
          DATE '{TRUSTED_DATA_AS_OF_DATE}'
      ) >= DATEADD(DAY, 30, S.POLICY_TERM_START_DATE::DATE)
),
FEATURE_TIES AS
(
    SELECT
        POLICY_CONTRACT_ID,
        ANCHOR_TERM_START_DATE
    FROM ACTIVE_FEATURE_ROWS
    WHERE EXPOSURE_START_DATE = LATEST_ACTIVE_EFFECTIVE_DATE
    GROUP BY 1, 2
    HAVING COUNT(*) > 1
)
SELECT
    COUNT(*) AS SOURCE_ANCHOR_GROUP_COUNT,
    COALESCE(COUNT_IF(
        POLICY_CONTRACT_ID IS NULL
        OR TRIM(TO_VARCHAR(POLICY_CONTRACT_ID)) = ''
    ), 0) AS NULL_POLICY_KEY_COUNT,
    COALESCE(COUNT_IF(ANCHOR_TERM_START_DATE IS NULL), 0)
        AS NULL_ANCHOR_START_COUNT,
    COALESCE(COUNT_IF(ANCHOR_TENURE_YEARS_GRP IS NULL), 0)
        AS NULL_ANCHOR_TENURE_COUNT,
    COALESCE(COUNT_IF(ANCHOR_TERM_END_DATE IS NULL), 0)
        AS NULL_ANCHOR_TERM_END_COUNT,
    (SELECT INVALID_OR_NULL_SOURCE_TENURE_GROUP_COUNT
       FROM SOURCE_TENURE_QUALITY)
        AS INVALID_OR_NULL_SOURCE_TENURE_GROUP_COUNT,
    (SELECT INVALID_NON_NULL_SOURCE_TENURE_INT_COUNT
       FROM SOURCE_TENURE_QUALITY)
        AS INVALID_NON_NULL_SOURCE_TENURE_INT_COUNT,
    COALESCE(COUNT_IF(TENURE_VALUE_COUNT <> 1), 0)
        AS TENURE_AMBIGUITY_COUNT,
    COALESCE(COUNT_IF(TERM_END_VALUE_COUNT <> 1), 0)
        AS TERM_END_AMBIGUITY_COUNT,
    COALESCE(COUNT_IF(
        ANCHOR_TENURE_YEARS_GRP NOT IN ('0', '1', '2', '3', '4+')
    ), 0) AS INVALID_TENURE_GROUP_COUNT,
    (SELECT FAILURE_COUNT FROM SPECIAL_TENURE_CROSSWALK)
        AS SPECIAL_TENURE_CROSSWALK_FAILURE_COUNT,
    COALESCE(COUNT_IF(
        DATEDIFF(DAY, ANCHOR_TERM_START_DATE, ANCHOR_TERM_END_DATE)
            NOT BETWEEN {MIN_ANNUAL_TERM_DAYS} AND {MAX_ANNUAL_TERM_DAYS}
    ), 0) AS FIXED_HORIZON_FAILURE_COUNT,
    (SELECT COUNT(*) FROM FEATURE_TIES) AS FEATURE_ROW_TIE_GROUPS,
    (SELECT COUNT(*) FROM TERM_SOURCE_ROW_TIES)
        AS TERM_SOURCE_ROW_TIE_GROUPS,
    (SELECT FAILURE_COUNT FROM TERM_DEDUCTIBLE_PARSE_FAILURES)
        AS TERM_DEDUCTIBLE_PARSE_FAILURE_COUNT
FROM ANCHOR_GROUPS
""".strip()




def build_postbuild_qa_sql() -> str:
    return _qa_select(f"SELECT * FROM {MAIN_TABLE_FQN}")




def build_current_table_count_sql() -> str:
    return f"SELECT COUNT(*) AS ROW_COUNT FROM {MAIN_TABLE_FQN}"




def build_table_metadata_sql() -> str:
    database, schema, table = MAIN_TABLE_FQN.split(".")
    return f"""
SELECT
    CURRENT_ROLE() AS CURRENT_ROLE,
    TABLE_OWNER,
    TABLE_TYPE,
    RETENTION_TIME,
    GET_DDL('TABLE', '{MAIN_TABLE_FQN}', TRUE) AS TABLE_DDL
FROM {database}.INFORMATION_SCHEMA.TABLES
WHERE TABLE_CATALOG = '{database}'
  AND TABLE_SCHEMA = '{schema}'
  AND TABLE_NAME = '{table}'
""".strip()




def build_object_dependencies_sql() -> str:
    """Inventory every account-usage object that depends on the live table.


    Access failures are deliberately not caught by the refresh lifecycle: an
    operator who cannot establish dependency safety cannot replace the table.
    """


    database, schema, table = MAIN_TABLE_FQN.split(".")
    return f"""
SELECT
    REFERENCING_DATABASE,
    REFERENCING_SCHEMA,
    REFERENCING_OBJECT_NAME,
    REFERENCING_OBJECT_DOMAIN,
    DEPENDENCY_TYPE
FROM SNOWFLAKE.ACCOUNT_USAGE.OBJECT_DEPENDENCIES
WHERE REFERENCED_DATABASE = '{database}'
  AND REFERENCED_SCHEMA = '{schema}'
  AND REFERENCED_OBJECT_NAME = '{table}'
  AND REFERENCED_OBJECT_DOMAIN = 'TABLE'
ORDER BY
    REFERENCING_DATABASE,
    REFERENCING_SCHEMA,
    REFERENCING_OBJECT_NAME,
    REFERENCING_OBJECT_DOMAIN,
    DEPENDENCY_TYPE
""".strip()




def build_policy_references_sql() -> str:
    """Ask Snowflake directly for every policy attached to the live table."""


    database, _, _ = MAIN_TABLE_FQN.split(".")
    return f"""
SELECT *
FROM TABLE(
    {database}.INFORMATION_SCHEMA.POLICY_REFERENCES(
        REF_ENTITY_NAME => '{MAIN_TABLE_FQN}',
        REF_ENTITY_DOMAIN => 'TABLE'
    )
)
""".strip()




def build_account_policy_references_sql() -> str:
    """Complement the live function with the account-wide policy inventory."""


    database, schema, table = MAIN_TABLE_FQN.split(".")
    return f"""
SELECT *
FROM SNOWFLAKE.ACCOUNT_USAGE.POLICY_REFERENCES
WHERE REF_DATABASE_NAME = '{database}'
  AND REF_SCHEMA_NAME = '{schema}'
  AND REF_ENTITY_NAME = '{table}'
  AND REF_ENTITY_DOMAIN = 'TABLE'
""".strip()




def build_data_metric_function_references_sql() -> str:
    """Ask Snowflake directly for every DMF attached to the live table."""


    database, _, _ = MAIN_TABLE_FQN.split(".")
    return f"""
SELECT *
FROM TABLE(
    {database}.INFORMATION_SCHEMA.DATA_METRIC_FUNCTION_REFERENCES(
        REF_ENTITY_NAME => '{MAIN_TABLE_FQN}',
        REF_ENTITY_DOMAIN => 'TABLE'
    )
)
""".strip()




def build_account_data_metric_function_references_sql() -> str:
    """Complement the live function with the account-wide DMF inventory."""


    database, schema, table = MAIN_TABLE_FQN.split(".")
    return f"""
SELECT *
FROM SNOWFLAKE.ACCOUNT_USAGE.DATA_METRIC_FUNCTION_REFERENCES
WHERE REF_DATABASE_NAME = '{database}'
  AND REF_SCHEMA_NAME = '{schema}'
  AND REF_ENTITY_NAME = '{table}'
  AND REF_ENTITY_DOMAIN = 'TABLE'
""".strip()




def build_show_streams_sql() -> str:
    # Account scope is deliberate: streams can live outside the source
    # table's schema. SHOW still returns only objects visible to the role, so
    # the operator must run preflight with account-wide stream visibility.
    return "SHOW STREAMS IN ACCOUNT"




def build_tenure_distribution_sql() -> str:
    return f"""
SELECT
    ANCHOR_TENURE_YEARS_GRP AS TENURE_GROUP,
    COUNT(*) AS ROW_COUNT,
    AVG(RETENTION_13_1_IND) AS RETENTION_RATE,
    MIN(ANCHOR_TERM_START_DATE) AS EARLIEST_TERM_START,
    MAX(ANCHOR_TERM_START_DATE) AS LATEST_TERM_START
FROM {MAIN_TABLE_FQN}
GROUP BY ANCHOR_TENURE_YEARS_GRP
ORDER BY CASE ANCHOR_TENURE_YEARS_GRP
    WHEN '0' THEN 0
    WHEN '1' THEN 1
    WHEN '2' THEN 2
    WHEN '3' THEN 3
    WHEN '4+' THEN 4
    ELSE 5
END
""".strip()




def _normalized(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} returned {type(frame)!r}, not DataFrame.")
    result = frame.copy()
    result.columns = [str(column).strip().upper() for column in result.columns]
    return result




def _one_int(frame: pd.DataFrame, column: str) -> int:
    result = _normalized(frame, column)
    if len(result) != 1 or column not in result.columns:
        raise ValueError(f"Expected one-row result containing {column}.")
    value = result.iloc[0][column]
    return 0 if pd.isna(value) else int(value)




_UNSUPPORTED_TABLE_DDL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("MASKING POLICY", r"\bMASKING\s+POLICY\b"),
    ("ROW ACCESS POLICY", r"\bROW\s+ACCESS\s+POLICY\b"),
    ("AGGREGATION POLICY", r"\bAGGREGATION\s+POLICY\b"),
    ("JOIN POLICY", r"\bJOIN\s+POLICY\b"),
    ("PROJECTION POLICY", r"\bPROJECTION\s+POLICY\b"),
    ("STORAGE LIFECYCLE POLICY", r"\bSTORAGE\s+LIFECYCLE\s+POLICY\b"),
    ("CHANGE_TRACKING=TRUE", r"\bCHANGE_TRACKING\s*=\s*TRUE\b"),
    ("CLUSTER BY", r"\bCLUSTER\s+BY\b"),
    ("SEARCH OPTIMIZATION", r"\bSEARCH\s+OPTIMIZATION\b"),
    (
        "ENABLE_SCHEMA_EVOLUTION=TRUE",
        r"\bENABLE_SCHEMA_EVOLUTION\s*=\s*TRUE\b",
    ),
    ("DEFAULT_DDL_COLLATION", r"\bDEFAULT_DDL_COLLATION\b"),
    (
        "MAX_DATA_EXTENSION_TIME_IN_DAYS",
        r"\bMAX_DATA_EXTENSION_TIME_IN_DAYS\b",
    ),
    ("WITH CONTACT", r"\bWITH\s+CONTACT\b"),
    ("COMMENT", r"\bCOMMENT\s*(?:=\s*)?''"),
    ("TRANSIENT TABLE", r"\bTRANSIENT\s+TABLE\b"),
    ("HYBRID TABLE", r"\bHYBRID\s+TABLE\b"),
    ("ICEBERG TABLE", r"\bICEBERG\s+TABLE\b"),
    ("NOT NULL", r"\bNOT\s+NULL\b"),
    ("DEFAULT", r"\bDEFAULT\b"),
    ("PRIMARY KEY", r"\bPRIMARY\s+KEY\b"),
    ("FOREIGN KEY", r"\bFOREIGN\s+KEY\b"),
    ("UNIQUE", r"\bUNIQUE\b"),
    ("CHECK", r"\bCHECK\s*\("),
    ("AUTOINCREMENT", r"\bAUTOINCREMENT\b"),
    ("IDENTITY", r"\bIDENTITY\b"),
    ("COLLATE", r"\bCOLLATE\b"),
    ("ROW_TIMESTAMP", r"\bROW_TIMESTAMP\b"),
    ("ERROR_LOGGING=TRUE", r"\bERROR_LOGGING\s*=\s*TRUE\b"),
    (
        "ICEBERG_DEFAULT_DDL_COLLATION",
        r"\bICEBERG_DEFAULT_DDL_COLLATION\b",
    ),
    ("DATA METRIC FUNCTION", r"\bDATA\s+METRIC\s+FUNCTION\b"),
    ("DATA_METRIC_SCHEDULE", r"\bDATA_METRIC_SCHEDULE\b"),
)




def _ddl_contract_text(table_ddl: str) -> str:
    """Return DDL suitable for whitespace-insensitive clause detection.


    Literal contents and quoted identifiers are masked so words inside a
    comment value or unusual quoted column name cannot impersonate a clause.
    The quote pair remains, allowing COMMENT with either ``=`` or whitespace
    syntax to be detected exactly.
    """


    masked = re.sub(r"'(?:''|[^'])*'", "''", table_ddl)
    masked = re.sub(r'"(?:""|[^"])*"', '""', masked)
    return masked.upper()




def validate_table_metadata(frame: pd.DataFrame) -> dict[str, Any]:
    result = _normalized(frame, "table metadata")
    required = {
        "CURRENT_ROLE",
        "TABLE_OWNER",
        "TABLE_TYPE",
        "RETENTION_TIME",
        "TABLE_DDL",
    }
    if len(result) != 1 or not required.issubset(result.columns):
        raise ValueError(
            "Expected one current-table metadata row containing "
            f"{sorted(required)}."
        )
    row = result.iloc[0]
    current_role = str(row["CURRENT_ROLE"]).strip().upper()
    table_owner = str(row["TABLE_OWNER"]).strip().upper()
    table_type = str(row["TABLE_TYPE"]).strip().upper()
    table_ddl = str(row["TABLE_DDL"]).strip()
    retention_time = int(row["RETENTION_TIME"])
    if not current_role or current_role != table_owner:
        raise ValueError(
            "Run the refresh under the current table-owning role so ownership "
            f"does not change: current_role={current_role!r}, "
            f"table_owner={table_owner!r}."
        )
    if table_type != "BASE TABLE":
        raise ValueError(
            "Same-name refresh supports only a permanent BASE TABLE; "
            f"observed TABLE_TYPE={table_type!r}."
        )
    if retention_time < 1:
        raise ValueError(
            "The current CY table has no Time Travel retention. Set retention "
            "to at least one day before same-name replacement."
        )
    ddl_contract_text = _ddl_contract_text(table_ddl)
    unsupported_governance = [
        label
        for label, pattern in _UNSUPPORTED_TABLE_DDL_PATTERNS
        if re.search(pattern, ddl_contract_text)
    ]
    if unsupported_governance:
        raise ValueError(
            "The current CY has governance/change-tracking metadata that this "
            "same-name CTAS does not safely preserve: "
            f"{unsupported_governance}."
        )
    return {
        "current_role": current_role,
        "table_owner": table_owner,
        "table_type": table_type,
        "retention_time_days": retention_time,
        "table_ddl_sha256": _sha256_text(table_ddl),
        "table_ddl": table_ddl,
    }




def validate_object_dependencies(frame: pd.DataFrame) -> dict[str, Any]:
    """Allow only ordinary name-bound views; block every other dependency."""


    result = _normalized(frame, "object dependency inventory")
    required = {
        "REFERENCING_DATABASE",
        "REFERENCING_SCHEMA",
        "REFERENCING_OBJECT_NAME",
        "REFERENCING_OBJECT_DOMAIN",
        "DEPENDENCY_TYPE",
    }
    if not required.issubset(result.columns):
        raise ValueError(
            "OBJECT_DEPENDENCIES returned an unfamiliar schema; replacement "
            f"safety cannot be established. Missing={sorted(required.difference(result.columns))}."
        )
    domain = result["REFERENCING_OBJECT_DOMAIN"].fillna("").astype(str).str.upper()
    dependency_type = result["DEPENDENCY_TYPE"].fillna("").astype(str).str.upper()
    safe = domain.eq("VIEW") & dependency_type.eq("BY_NAME")
    blocked = result.loc[~safe]
    if not blocked.empty:
        inventory = blocked[list(required)].to_dict("records")
        raise ValueError(
            "Same-name replacement has unsupported or identity-bound dependent "
            "Snowflake objects: " + json.dumps(inventory, default=str, sort_keys=True)
        )
    return {
        "inventory_scope": "SNOWFLAKE.ACCOUNT_USAGE",
        "dependency_count": int(len(result)),
        "safe_name_bound_view_count": int(safe.sum()),
        "blocked_dependency_count": 0,
    }




def _validate_no_governance_references(
    frame: pd.DataFrame,
    *,
    label: str,
    required_columns: set[str],
    inventory_columns: tuple[str, ...],
) -> dict[str, Any]:
    result = _normalized(frame, label)
    if not required_columns.issubset(result.columns):
        raise ValueError(
            f"{label} returned an unfamiliar schema; replacement safety "
            "cannot be established. Missing="
            f"{sorted(required_columns.difference(result.columns))}."
        )
    if not result.empty:
        available = [column for column in inventory_columns if column in result.columns]
        inventory = result[available].to_dict("records")
        raise ValueError(
            f"Same-name CTAS cannot safely preserve {label}: "
            + json.dumps(inventory, default=str, sort_keys=True)
        )
    return {"reference_count": 0}




def validate_no_policy_references(frame: pd.DataFrame) -> dict[str, Any]:
    return _validate_no_governance_references(
        frame,
        label="policy references",
        required_columns={
            "POLICY_NAME",
            "POLICY_KIND",
            "REF_ENTITY_NAME",
            "REF_ENTITY_DOMAIN",
        },
        inventory_columns=(
            "POLICY_DB",
            "POLICY_SCHEMA",
            "POLICY_NAME",
            "POLICY_KIND",
            "REF_ENTITY_NAME",
            "REF_ENTITY_DOMAIN",
            "REF_COLUMN_NAME",
        ),
    )




def validate_no_data_metric_function_references(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    return _validate_no_governance_references(
        frame,
        label="data metric function references",
        required_columns={
            "METRIC_NAME",
            "REF_ENTITY_NAME",
            "REF_ENTITY_DOMAIN",
        },
        inventory_columns=(
            "METRIC_DATABASE",
            "METRIC_SCHEMA",
            "METRIC_NAME",
            "REF_ENTITY_NAME",
            "REF_ENTITY_DOMAIN",
            "REF_COLUMN_NAME",
            "SCHEDULE",
        ),
    )




def validate_no_stream_dependencies(frame: pd.DataFrame) -> dict[str, Any]:
    result = _normalized(frame, "stream inventory")
    reference_columns = [
        column
        for column in ("TABLE_NAME", "SOURCE_NAME", "BASE_TABLES")
        if column in result.columns
    ]
    if not reference_columns:
        raise ValueError(
            "SHOW STREAMS returned an unfamiliar schema; dependency safety "
            f"cannot be established. Columns={result.columns.tolist()}."
        )
    target_fqn = MAIN_TABLE_FQN.upper()
    table_name = target_fqn.split(".")[-1]
    dependent = pd.Series(False, index=result.index)
    for column in reference_columns:
        values = (
            result[column]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.replace('"', "", regex=False)
        )
        dependent |= values.eq(table_name) | values.str.contains(
            re.escape(target_fqn), regex=True
        )
    count = int(dependent.sum())
    if count:
        names = (
            result.loc[dependent, "NAME"].astype(str).tolist()
            if "NAME" in result.columns
            else []
        )
        raise ValueError(
            "Replacing the CY would stale dependent Snowflake streams. "
            f"Visible dependent streams={count}; names={names}."
        )
    return {
        "inventory_scope": "ACCOUNT_VISIBLE_TO_CURRENT_ROLE",
        "visible_stream_count": int(len(result)),
        "dependent_stream_count": 0,
    }




def validate_population_qa(frame: pd.DataFrame, *, stage: str) -> dict[str, int]:
    result = _normalized(frame, stage)
    summary = {
        "ROW_COUNT": _one_int(result, "ROW_COUNT"),
        "RETAINED_COUNT": _one_int(result, "RETAINED_COUNT"),
        "CONTENT_HASH": _one_int(result, "CONTENT_HASH"),
        "DISTINCT_TENURE_GROUP_COUNT": _one_int(
            result, "DISTINCT_TENURE_GROUP_COUNT"
        ),
        "TENURE_GROUP_0_ROWS": _one_int(result, "TENURE_GROUP_0_ROWS"),
        "TENURE_GROUP_1_ROWS": _one_int(result, "TENURE_GROUP_1_ROWS"),
        "TENURE_GROUP_2_ROWS": _one_int(result, "TENURE_GROUP_2_ROWS"),
        "TENURE_GROUP_3_ROWS": _one_int(result, "TENURE_GROUP_3_ROWS"),
        "TENURE_GROUP_4_PLUS_ROWS": _one_int(
            result, "TENURE_GROUP_4_PLUS_ROWS"
        ),
        **{
            column: _one_int(result, column)
            for column in STABLE_INFORMATIONAL_QA_COLUMNS
        },
    }
    failures = {
        column: _one_int(result, column)
        for column in REQUIRED_ZERO_COLUMNS
        if _one_int(result, column)
    }
    if summary["ROW_COUNT"] <= 0:
        failures["ROW_COUNT"] = summary["ROW_COUNT"]
    if summary["DISTINCT_TENURE_GROUP_COUNT"] != 5:
        failures["INCOMPLETE_TENURE_GROUP_DOMAIN"] = summary[
            "DISTINCT_TENURE_GROUP_COUNT"
        ]
    group_counts = {
        group: summary[column]
        for group, column in {
            "0": "TENURE_GROUP_0_ROWS",
            "1": "TENURE_GROUP_1_ROWS",
            "2": "TENURE_GROUP_2_ROWS",
            "3": "TENURE_GROUP_3_ROWS",
            "4+": "TENURE_GROUP_4_PLUS_ROWS",
        }.items()
    }
    missing_groups = [
        group for group, count in group_counts.items() if count <= 0
    ]
    if missing_groups:
        failures["MISSING_TENURE_GROUP_COUNT"] = len(missing_groups)
    if sum(group_counts.values()) != summary["ROW_COUNT"]:
        failures["TENURE_GROUP_ROW_COUNT_MISMATCH"] = abs(
            summary["ROW_COUNT"] - sum(group_counts.values())
        )
    if failures:
        raise ValueError(
            f"{stage} all-tenure QA failed: "
            + json.dumps(failures, sort_keys=True)
        )
    return summary




def validate_source_anchor_qa(frame: pd.DataFrame) -> dict[str, int]:
    result = _normalized(frame, "source anchor QA")
    summary = {
        "SOURCE_ANCHOR_GROUP_COUNT": _one_int(result, "SOURCE_ANCHOR_GROUP_COUNT")
    }
    failures = {
        column: _one_int(result, column)
        for column in SOURCE_ZERO_COLUMNS
        if _one_int(result, column)
    }
    if summary["SOURCE_ANCHOR_GROUP_COUNT"] <= 0:
        failures["SOURCE_ANCHOR_GROUP_COUNT"] = 0
    if failures:
        raise ValueError(
            "source anchor QA failed: " + json.dumps(failures, sort_keys=True)
        )
    return summary




def _receipt_path(run_id: str) -> Path:
    return Path(PIPELINE_OUTPUT_ROOT) / run_id / "source_refresh" / "receipt.json"




def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)




def refresh_table(*, preflight_only: bool = False) -> dict[str, Any]:
    run_id = (
        os.environ.get("RETENTION_PIPELINE_RUN_ID", "").strip()
        or new_run_id(Path(__file__).stem)
    )
    runtime_setup = setup_params(run_id)
    runtime_setup["SAVE_ARTIFACTS"] = True


    from func.setup.setup import Setup


    setup = Setup(**runtime_setup, MEMORY_REDUCTION=False)
    receipt: dict[str, Any] = {
        "run_id": run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_table": SOURCE_TABLE_FQN,
        "target_table": MAIN_TABLE_FQN,
        "build_sql_path": str(Path(ALL_TENURE_BUILD_SQL_PATH).resolve()),
        "build_sql_file_sha256": _sha256_file(ALL_TENURE_BUILD_SQL_PATH),
        "replacement_mode": "same_table_create_or_replace_copy_grants_copy_tags",
        "continuity_key": "POLICY_CONTRACT_ID",
        "observation_key": ["POLICY_CONTRACT_ID", "ANCHOR_TERM_START_DATE"],
        "cross_contract_rewrite_linkage": "UNAVAILABLE_NOT_INFERRED",
        "status": "RUNNING",
    }
    receipt_path = _receipt_path(run_id)
    try:
        connection = setup.conns[SF_CONN]
        source_schema_frame = connection.query_data(
            build_source_schema_sql(),
            query_name="all_tenure_source_schema",
            verbose=10,
        )
        receipt["source_schema"] = validate_source_schema(source_schema_frame)
        receipt["source_schema_query_id"] = getattr(connection, "query_id", None)
        raw_columns = raw_projection_columns(source_schema_frame)


        current_schema_frame = connection.query_data(
            build_target_schema_sql(),
            query_name="all_tenure_current_cy_schema",
            verbose=10,
        )
        receipt["current_schema_query_id"] = getattr(
            connection, "query_id", None
        )


        output_columns = final_output_columns(raw_columns, current_schema_frame)
        build_template_sql = load_build_sql(
            raw_columns=raw_columns,
            output_columns=output_columns,
        )
        build_relation_sql = candidate_relation(build_template_sql)
        receipt["executable_relation_sha256"] = _sha256_text(
            build_relation_sql
        )


        connection.query_data(
            build_candidate_schema_probe_sql(build_template_sql),
            query_name="all_tenure_candidate_schema_probe",
            verbose=10,
        )
        candidate_probe_query_id = getattr(connection, "query_id", None)
        if not candidate_probe_query_id:
            raise ValueError(
                "Candidate LIMIT 0 probe returned no Snowflake query ID; "
                "DESCRIBE RESULT cannot safely target the compiled statement."
            )
        receipt["candidate_schema_probe_query_id"] = candidate_probe_query_id
        candidate_schema_frame = connection.query_data(
            build_describe_result_sql(candidate_probe_query_id),
            query_name="all_tenure_candidate_schema_description",
            verbose=10,
        )
        receipt["schema_preservation"] = validate_schema_preservation(
            current_schema_frame, candidate_schema_frame
        )
        receipt["candidate_schema_description_query_id"] = getattr(
            connection, "query_id", None
        )


        metadata_frame = connection.query_data(
            build_table_metadata_sql(),
            query_name="all_tenure_current_cy_metadata",
            verbose=10,
        )
        receipt["prior_table_metadata"] = validate_table_metadata(metadata_frame)
        receipt["metadata_query_id"] = getattr(connection, "query_id", None)
        replace_sql = build_replace_sql(
            build_template_sql,
            retention_time_days=receipt["prior_table_metadata"][
                "retention_time_days"
            ],
        )
        receipt["rendered_build_sql_sha256"] = _sha256_text(replace_sql)


        object_dependency_frame = connection.query_data(
            build_object_dependencies_sql(),
            query_name="all_tenure_object_dependency_audit",
            verbose=10,
        )
        receipt["object_dependency_audit"] = validate_object_dependencies(
            object_dependency_frame
        )
        receipt["object_dependency_audit_query_id"] = getattr(
            connection, "query_id", None
        )


        policy_reference_frame = connection.query_data(
            build_policy_references_sql(),
            query_name="all_tenure_policy_reference_audit",
            verbose=10,
        )
        receipt["policy_reference_audit"] = validate_no_policy_references(
            policy_reference_frame
        )
        receipt["policy_reference_audit_query_id"] = getattr(
            connection, "query_id", None
        )


        account_policy_reference_frame = connection.query_data(
            build_account_policy_references_sql(),
            query_name="all_tenure_account_policy_reference_audit",
            verbose=10,
        )
        receipt[
            "account_policy_reference_audit"
        ] = validate_no_policy_references(account_policy_reference_frame)
        receipt["account_policy_reference_audit_query_id"] = getattr(
            connection, "query_id", None
        )


        dmf_reference_frame = connection.query_data(
            build_data_metric_function_references_sql(),
            query_name="all_tenure_data_metric_function_reference_audit",
            verbose=10,
        )
        receipt[
            "data_metric_function_reference_audit"
        ] = validate_no_data_metric_function_references(dmf_reference_frame)
        receipt["data_metric_function_reference_audit_query_id"] = getattr(
            connection, "query_id", None
        )


        account_dmf_reference_frame = connection.query_data(
            build_account_data_metric_function_references_sql(),
            query_name="all_tenure_account_data_metric_function_reference_audit",
            verbose=10,
        )
        receipt[
            "account_data_metric_function_reference_audit"
        ] = validate_no_data_metric_function_references(
            account_dmf_reference_frame
        )
        receipt[
            "account_data_metric_function_reference_audit_query_id"
        ] = getattr(connection, "query_id", None)


        stream_frame = connection.query_data(
            build_show_streams_sql(),
            query_name="all_tenure_stream_dependency_audit",
            verbose=10,
        )
        receipt["stream_dependency_audit"] = validate_no_stream_dependencies(
            stream_frame
        )
        receipt["stream_audit_query_id"] = getattr(connection, "query_id", None)


        current_count_frame = connection.query_data(
            build_current_table_count_sql(),
            query_name="all_tenure_current_cy_count",
            verbose=10,
        )
        receipt["prior_row_count"] = _one_int(current_count_frame, "ROW_COUNT")


        source_anchor_frame = connection.query_data(
            build_source_anchor_qa_sql(),
            query_name="all_tenure_source_anchor_qa",
            verbose=10,
        )
        receipt["source_anchor_qa"] = validate_source_anchor_qa(
            source_anchor_frame
        )
        receipt["source_anchor_query_id"] = getattr(
            connection, "query_id", None
        )


        preflight_frame = connection.query_data(
            build_preflight_qa_sql(build_template_sql),
            query_name="all_tenure_read_only_preflight",
            verbose=10,
        )
        receipt["preflight"] = validate_population_qa(
            preflight_frame, stage="preflight"
        )
        normalized_preflight = _normalized(preflight_frame, "preflight diagnostics")
        receipt["legacy_tenure0_migration_audit"] = {
            column.lower(): _one_int(normalized_preflight, column)
            for column in INFORMATIONAL_QA_COLUMNS
            if column.startswith("TENURE0_")
        }
        # The V12 contract intentionally collapses inception-split copies of
        # the same adjusted-contract/term observation. A smaller candidate is
        # therefore not itself a failure; the governed key, domain, schema,
        # interval-target, and content checks above remain authoritative.
        receipt["candidate_row_count_delta_vs_prior"] = (
            receipt["preflight"]["ROW_COUNT"] - receipt["prior_row_count"]
        )
        receipt["preflight_query_id"] = getattr(connection, "query_id", None)


        if preflight_only:
            receipt["status"] = "PREFLIGHT_COMPLETE"
            receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
            receipt["replacement_attempted"] = False
            _write_receipt(receipt_path, receipt)
            return {**receipt, "receipt_path": str(receipt_path.resolve())}


        receipt["replacement_attempted"] = True
        _write_receipt(receipt_path, receipt)
        # Clear the previous statement ID. FIGConnect sets this from the DDL
        # cursor immediately after execute(), before any result fetch.
        connection.query_id = None
        connection.query_data(
            replace_sql,
            query_name="replace_existing_cy_with_all_tenure",
            verbose=10,
        )
        receipt["replace_query_id"] = getattr(connection, "query_id", None)
        receipt["replacement_completed"] = True
        _write_receipt(receipt_path, receipt)


        post_schema_frame = connection.query_data(
            build_target_schema_sql(),
            query_name="all_tenure_postbuild_cy_schema",
            verbose=10,
        )
        receipt["postbuild_schema"] = validate_schema_identity(
            candidate_schema_frame, post_schema_frame
        )
        receipt["postbuild_schema_query_id"] = getattr(
            connection, "query_id", None
        )


        post_metadata_frame = connection.query_data(
            build_table_metadata_sql(),
            query_name="all_tenure_postbuild_cy_metadata",
            verbose=10,
        )
        receipt["postbuild_table_metadata"] = validate_table_metadata(
            post_metadata_frame
        )
        if (
            receipt["postbuild_table_metadata"]["table_owner"]
            != receipt["prior_table_metadata"]["table_owner"]
            or receipt["postbuild_table_metadata"]["retention_time_days"]
            != receipt["prior_table_metadata"]["retention_time_days"]
        ):
            raise ValueError(
                "Postbuild owner or Time Travel retention differs from the "
                "preflight table metadata."
            )
        receipt["postbuild_metadata_query_id"] = getattr(
            connection, "query_id", None
        )


        postbuild_frame = connection.query_data(
            build_postbuild_qa_sql(),
            query_name="all_tenure_postbuild_qa",
            verbose=10,
        )
        receipt["postbuild"] = validate_population_qa(
            postbuild_frame, stage="postbuild"
        )
        if receipt["postbuild"] != receipt["preflight"]:
            raise ValueError(
                "Postbuild QA does not exactly match preflight: "
                + json.dumps(
                    {
                        "preflight": receipt["preflight"],
                        "postbuild": receipt["postbuild"],
                    },
                    sort_keys=True,
                )
            )
        receipt["postbuild_query_id"] = getattr(connection, "query_id", None)


        tenure_distribution = _normalized(
            connection.query_data(
                build_tenure_distribution_sql(),
                query_name="all_tenure_distribution",
                verbose=10,
            ),
            "tenure distribution",
        )
        receipt["tenure_distribution"] = tenure_distribution.to_dict("records")
        receipt["tenure_distribution_query_id"] = getattr(
            connection, "query_id", None
        )
        receipt["status"] = "COMPLETE"
        receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
        _write_receipt(receipt_path, receipt)
        return {**receipt, "receipt_path": str(receipt_path.resolve())}
    except Exception as exc:
        if receipt.get("replacement_attempted") and not receipt.get(
            "replace_query_id"
        ):
            current_query_id = getattr(
                locals().get("connection", None), "query_id", None
            )
            if current_query_id:
                receipt["replace_query_id"] = current_query_id
        if receipt.get("replacement_completed"):
            receipt["status"] = "FAILED_AFTER_REPLACEMENT"
        elif receipt.get("replacement_attempted"):
            receipt["status"] = "REPLACEMENT_ATTEMPTED_STATUS_UNKNOWN"
        else:
            receipt["status"] = "FAILED_BEFORE_REPLACEMENT"
        receipt["failed_utc"] = datetime.now(timezone.utc).isoformat()
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        _write_receipt(receipt_path, receipt)
        raise
    finally:
        setup.shutdown()




def main() -> None:
    if os.getenv("RETENTION_REFRESH_ALL_TENURE_TABLE", "").strip().upper() not in {
        "1",
        "TRUE",
        "Y",
        "YES",
    }:
        raise RuntimeError(
            "This same-table CREATE OR REPLACE step is opt-in. Run root "
            "a0_one_click.py with --refresh-source."
        )
    preflight_only = os.getenv(
        "RETENTION_REFRESH_PREFLIGHT_ONLY", ""
    ).strip().upper() in {"1", "TRUE", "Y", "YES"}
    summary = refresh_table(preflight_only=preflight_only)
    print(
        "ALL-TENURE PREFLIGHT COMPLETE; EXISTING CY TABLE UNCHANGED"
        if preflight_only
        else "EXISTING CY TABLE REFRESHED WITH ALL TENURES AND VERIFIED"
    )
    print(json.dumps(summary, indent=2, default=str))




if __name__ == "__main__":
    main()
