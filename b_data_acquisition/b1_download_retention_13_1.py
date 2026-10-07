# -*- coding: utf-8 -*-
"""Query and download one tenure-scoped 13/1 population from Snowflake.


The full ROC config is retained as the variable dictionary. Its ``cv_ind=1``
rows define the initial configured candidate set, while target/key/date rows
remain in the same CSV with ``cv_ind=0``. SQL projection and parsing types are
therefore derived from one config rather than a hidden Python type allowlist.


The physical CY table already contains one labeled row per mature policy term
for every tenure.  This step applies the validated anchor-tenure profile and
the preserved CA exclusion in Snowflake before download.  It does not
reconstruct, repair, or deduplicate the physical table.
"""


from __future__ import annotations


import argparse
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
    ACTIVE_AT_MONTH_1_COL,
    ACTIVE_AT_MONTH_13_COL,
    ACTIVE_PIPELINE_STATE_PATH,
    ANCHOR_TENURE_COL,
    ANCHOR_TERM_END_COL,
    ANCHOR_TERM_START_COL,
    DOWNLOAD_CHUNKSIZE,
    DOWNLOAD_METADATA_PATH,
    DOWNLOAD_PATH,
    EXCLUDED_PROPERTY_STATES,
    EXPOSURE_END_COL,
    EXPOSURE_START_COL,
    EXPOSURE_YEAR_COL,
    INCLUDE_EXPOSURE_YEAR_AS_PREDICTOR,
    LATEST_ANCHOR_TERM_START_DATE_EXCLUSIVE,
    COHORT_DATE_COL,
    INDEX_COL,
    MAIN_TABLE,
    MAIN_TABLE_FQN,
    MAX_ANNUAL_TERM_DAYS,
    MIN_ANNUAL_TERM_DAYS,
    POLICY_ID_COL,
    POLICY_INCEPTION_COL,
    POLICY_SAMPLE_BUCKET_COUNT,
    POLICY_SAMPLE_SALT,
    POPULATION_SCOPE_MODE,
    POPULATION_SCOPE_PAYLOAD,
    PROFILE_HASH,
    PROFILE_LABEL,
    PROFILE_PAYLOAD,
    PROFILE_SLUG,
    PROJECT_CONTRACT_VERSION,
    RAW_STATE_COLUMN,
    RECORD_KEY_COLS,
    REQUIRED_QUERY_COLUMNS,
    SF_CONN,
    SAMPLE_BASIS_POINTS,
    TARGET,
    TARGET_CHECK_COL,
    TEST_POLICY_ID,
    TENURE_COL,
    TENURE_GROUP_DOMAIN,
    TENURE_IS_MODEL_CANDIDATE,
    TENURE_PROFILE,
    TERM_END_COL,
    TERM_START_COL,
    TRUSTED_DATA_AS_OF_DATE,
    new_run_id,
    resolve_data_config_path,
    setup_params,
)




_COLUMN_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_TABLE_RE = re.compile(
    r"^[A-Z][A-Z0-9_$]*\.[A-Z][A-Z0-9_$]*\.[A-Z][A-Z0-9_$]*$"
)
_STATE_RE = re.compile(r"^[A-Z]{2}$")
_TRUE_VALUES = frozenset({"1", "TRUE", "T", "Y", "YES"})
_FALSE_VALUES = frozenset({"0", "FALSE", "F", "N", "NO", ""})


ROC_CONFIG_COLUMNS = (
    "Column",
    "dtype",
    "var_type",
    "source_tbl",
    "seg",
    "cv_ind",
    "uwtgbm_model_step",
    "iglm_model_step",
    "sglm_model_step",
    "uwtgbm_control",
    "iglm_control",
    "sglm_control",
    "fe_sel",
    "rebased_to",
    "fixed_level",
    "sm_ind",
    "sm_excld_lvl",
    "poly_sm_dgr",
    "sv_sm_inc",
    "sv_sm_ratio",
    "sv_sm_turning",
    "sv_sm_trend",
    "interaction_ind",
    "roc_header",
    "roc_tab",
    "roc_var_name",
    "filing_var_name",
    "filing_tab",
    "description",
)


STRUCTURAL_OUTPUT_TYPES = {
    ACTIVE_AT_MONTH_1_COL: "discrete",
    ACTIVE_AT_MONTH_13_COL: "discrete",
    ANCHOR_TERM_START_COL: "date",
    ANCHOR_TERM_END_COL: "date",
    TENURE_COL: "discrete",
    ANCHOR_TENURE_COL: "discrete",
}


def _normalize_flag(value: Any, *, field: str, column: str) -> bool:
    if isinstance(value, bool):
        return value
    text = "" if pd.isna(value) else str(value).strip().upper()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{field} for {column} must be 0/1 or Y/N; received {value!r}."
    )




def _safe_column(value: Any) -> str:
    column = str(value).strip().upper()
    if not _COLUMN_RE.fullmatch(column):
        raise ValueError(f"Unsafe or invalid Snowflake column name: {value!r}")
    return column




def _safe_table(value: Any) -> str:
    table = str(value).strip().upper()
    if not _TABLE_RE.fullmatch(table):
        raise ValueError(
            f"Expected DATABASE.SCHEMA.TABLE using safe identifiers; got {value!r}."
        )
    return table




def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()




def _unique_in_order(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _safe_column(value)
        if normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result




def _semantic_type(value: Any) -> str:
    normalized = str(value).strip().lower()
    mapping = {
        "date": "date",
        "continuous": "continuous",
        "measurement": "continuous",
        "discrete": "discrete",
        "key": "discrete",
        "drv_trans": "discrete",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported ROC config var_type: {value!r}")
    return mapping[normalized]




def load_download_contract(config_path: str | Path) -> dict[str, Any]:
    """Load the complete ROC config and derive this run's SQL projection."""


    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ROC data config not found: {path}")


    original = pd.read_csv(path, dtype=str, keep_default_na=False, skip_blank_lines=True)
    if original.columns.tolist() != list(ROC_CONFIG_COLUMNS):
        raise ValueError(
            "The active file must keep the original 29-column ROC config schema. "
            f"Expected {list(ROC_CONFIG_COLUMNS)}; found {original.columns.tolist()}."
        )
    frame = original.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    frame["raw_column"] = frame["column"]
    frame["normalized_column"] = frame["column"].map(_safe_column)
    frame["cv_enabled"] = [
        _normalize_flag(value, field="cv_ind", column=column)
        for value, column in zip(frame["cv_ind"], frame["raw_column"])
    ]


    source_values = set(frame["source_tbl"].astype(str).str.strip().str.upper())
    if source_values != {MAIN_TABLE}:
        raise ValueError(
            f"Every source_tbl value must be {MAIN_TABLE}; found {sorted(source_values)}."
        )


    candidates = frame.loc[frame["cv_enabled"]].copy()
    duplicate_candidates = sorted(
        candidates.loc[
            candidates["normalized_column"].duplicated(keep=False),
            "normalized_column",
        ].unique()
    )
    if duplicate_candidates:
        raise ValueError(
            "Enabled ROC config rows collapse to duplicate Snowflake columns: "
            f"{duplicate_candidates}"
        )


    configured_model_features = candidates["normalized_column"].tolist()
    forbidden_structural_candidates = sorted(
        set(configured_model_features).intersection(REQUIRED_QUERY_COLUMNS)
        .difference({RAW_STATE_COLUMN, EXPOSURE_YEAR_COL, TENURE_COL})
    )
    if forbidden_structural_candidates:
        raise ValueError(
            "Structural/target fields cannot be enabled as model candidates: "
            f"{forbidden_structural_candidates}."
        )
    model_features = list(configured_model_features)
    if TENURE_COL in model_features and not TENURE_IS_MODEL_CANDIDATE:
        model_features.remove(TENURE_COL)
    if INCLUDE_EXPOSURE_YEAR_AS_PREDICTOR:
        if EXPOSURE_YEAR_COL not in model_features:
            model_features.append(EXPOSURE_YEAR_COL)
    elif EXPOSURE_YEAR_COL in model_features:
        model_features.remove(EXPOSURE_YEAR_COL)
    projected_columns = _unique_in_order((*REQUIRED_QUERY_COLUMNS, *model_features))


    config_types: dict[str, str] = {}
    for column, rows in frame.loc[
        frame["normalized_column"].isin(projected_columns)
    ].groupby("normalized_column", sort=False):
        semantic_types = {
            _semantic_type(value) for value in rows["var_type"].tolist()
        }
        if len(semantic_types) != 1:
            raise ValueError(
                f"Config rows for {column} disagree on var_type: "
                f"{sorted(semantic_types)}"
            )
        config_types[column] = semantic_types.pop()
    config_types.update(STRUCTURAL_OUTPUT_TYPES)
    missing_types = sorted(set(projected_columns).difference(config_types))
    if missing_types:
        raise ValueError(
            "No config-derived parsing type is available for projected "
            f"columns: {missing_types}"
        )
    var_types = {column: config_types[column] for column in projected_columns}


    all_config_columns = frame["normalized_column"].tolist()
    return {
        "path": path,
        "frame": frame,
        "configured_model_features": configured_model_features,
        "model_features": model_features,
        "projected_columns": projected_columns,
        "var_types": var_types,
        "all_config_columns": all_config_columns,
    }




def build_scope_predicate() -> str:
    """Return tenure, state, and optional policy-level test scope."""


    tenure = TENURE_PROFILE.sql_predicate(f'"{ANCHOR_TENURE_COL}"')
    states: list[str] = []
    for value in EXCLUDED_PROPERTY_STATES:
        state = str(value).strip().upper()
        if not _STATE_RE.fullmatch(state):
            raise ValueError(f"Invalid two-letter excluded state: {value!r}")
        states.append(state)
    states = sorted(set(states))
    predicates = [tenure]
    if states:
        literals = ", ".join(f"'{state}'" for state in states)
        predicates.append(
            f'("{RAW_STATE_COLUMN}" IS NULL '
            f'OR "{RAW_STATE_COLUMN}" NOT IN ({literals}))'
        )
    if POPULATION_SCOPE_MODE == "POLICY":
        policy_id = str(TEST_POLICY_ID)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", policy_id):
            raise ValueError(f"Unsafe test policy identifier: {policy_id!r}")
        predicates.append(
            f'TO_VARCHAR("{POLICY_ID_COL}") = \'{policy_id}\''
        )
    elif POPULATION_SCOPE_MODE == "SAMPLE":
        threshold = int(SAMPLE_BASIS_POINTS)
        if threshold < 1 or threshold >= POLICY_SAMPLE_BUCKET_COUNT:
            raise ValueError(f"Invalid policy-sample threshold: {threshold}")
        predicates.append(
            "MOD(MOD(HASH("
            f"'{POLICY_SAMPLE_SALT}', TO_VARCHAR(\"{POLICY_ID_COL}\")"
            f"), {POLICY_SAMPLE_BUCKET_COUNT}) + {POLICY_SAMPLE_BUCKET_COUNT}, "
            f"{POLICY_SAMPLE_BUCKET_COUNT}) < {threshold}"
        )
    return "\n  AND ".join(f"({predicate})" for predicate in predicates)




def build_select_sql(projected_columns: Iterable[str]) -> str:
    """Build the executable Snowflake SELECT from config-derived columns."""


    columns = _unique_in_order(projected_columns)
    if not columns:
        raise ValueError("At least one projected column is required.")
    projection = ",\n    ".join(f'"{column}"' for column in columns)
    table = _safe_table(MAIN_TABLE_FQN)
    return (
        "SELECT\n"
        f"    {projection}\n"
        f"FROM {table}\n"
        f"WHERE {build_scope_predicate()}"
    )




def build_schema_sql() -> str:
    database, schema, table = _safe_table(MAIN_TABLE_FQN).split(".")
    return f"""
SELECT COLUMN_NAME, ORDINAL_POSITION, DATA_TYPE
FROM {database}.INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = '{schema}'
  AND TABLE_NAME = '{table}'
ORDER BY ORDINAL_POSITION
""".strip()




def _population_relation(select_sql: str) -> str:
    return select_sql.strip().rstrip(";")




def build_source_qa_sql(select_sql: str) -> str:
    """Audit the scoped CY rows before any local transformation."""


    relation = _population_relation(select_sql)
    tenure_predicate = TENURE_PROFILE.sql_predicate(
        f'"{ANCHOR_TENURE_COL}"'
    )
    return f"""
SELECT
    COUNT(*) AS ROW_COUNT,
    COALESCE(COUNT_IF({TARGET} IS NULL), 0) AS TARGET_NULL_COUNT,
    COALESCE(COUNT_IF(
        {TARGET} IS NOT NULL AND {TARGET} NOT IN (0, 1)
    ), 0) AS TARGET_INVALID_COUNT,
    COALESCE(COUNT_IF(NOT ({tenure_predicate})), 0)
        AS TENURE_SCOPE_FAILURE_COUNT,
    COALESCE(COUNT_IF(
        "{TENURE_COL}" IS NULL
        OR "{ANCHOR_TENURE_COL}" IS NULL
        OR "{TENURE_COL}" <> "{ANCHOR_TENURE_COL}"
    ), 0) AS REPRESENTATIVE_TENURE_FAILURE_COUNT,
    COALESCE(COUNT_IF(
        {COHORT_DATE_COL} IS NULL
        OR {ANCHOR_TERM_START_COL} IS NULL
        OR {COHORT_DATE_COL} <> DATEADD(DAY, 30, {ANCHOR_TERM_START_COL})
    ), 0) AS MONTH1_DATE_RULE_FAILURE_COUNT,
    COALESCE(COUNT_IF(
        {ACTIVE_AT_MONTH_1_COL} IS NULL
        OR {ACTIVE_AT_MONTH_1_COL} <> 1
    ), 0) AS MONTH1_ACTIVE_FAILURE_COUNT,
    COALESCE(COUNT_IF(
        {TERM_START_COL} IS NULL
        OR {ANCHOR_TERM_START_COL} IS NULL
        OR {TERM_START_COL}::DATE <> {ANCHOR_TERM_START_COL}
    ), 0) AS SELECTED_TERM_START_FAILURE_COUNT,
    COALESCE(COUNT_IF(
        {TERM_END_COL} IS NULL
        OR {ANCHOR_TERM_END_COL} IS NULL
        OR {TERM_END_COL}::DATE <> {ANCHOR_TERM_END_COL}
    ), 0) AS SELECTED_TERM_END_FAILURE_COUNT,
    COALESCE(COUNT_IF(
        {TARGET_CHECK_COL} IS NULL
        OR {ANCHOR_TERM_END_COL} IS NULL
        OR {TARGET_CHECK_COL} <> DATEADD(DAY, 30, {ANCHOR_TERM_END_COL})
    ), 0) AS MONTH13_DATE_RULE_FAILURE_COUNT,
    COALESCE(COUNT_IF(
        {ACTIVE_AT_MONTH_13_COL} IS NULL
        OR {ACTIVE_AT_MONTH_13_COL} NOT IN (0, 1)
        OR {TARGET} <> {ACTIVE_AT_MONTH_13_COL}
    ), 0) AS TARGET_RULE_FAILURE_COUNT,
    COALESCE(COUNT_IF(
        DATEDIFF(DAY, {ANCHOR_TERM_START_COL}, {ANCHOR_TERM_END_COL})
            NOT BETWEEN {MIN_ANNUAL_TERM_DAYS} AND {MAX_ANNUAL_TERM_DAYS}
    ), 0) AS FIXED_HORIZON_FAILURE_COUNT,
    COALESCE(COUNT_IF(
        {EXPOSURE_START_COL} IS NULL
        OR {EXPOSURE_START_COL}::DATE > {COHORT_DATE_COL}
        OR {EXPOSURE_END_COL} IS NULL
        OR {EXPOSURE_END_COL}::DATE < {COHORT_DATE_COL}
    ), 0) AS FEATURE_AT_MONTH1_FAILURE_COUNT,
    COALESCE(COUNT_IF(
        {TARGET_CHECK_COL} IS NULL
        OR {TARGET_CHECK_COL} > DATE '{TRUSTED_DATA_AS_OF_DATE}'
    ), 0) AS MATURITY_FAILURE_COUNT,
    COALESCE(COUNT_IF(
        {ANCHOR_TERM_START_COL} IS NULL
        OR {ANCHOR_TERM_START_COL} >=
           DATE '{LATEST_ANCHOR_TERM_START_DATE_EXCLUSIVE}'
    ), 0) AS LATEST_START_FAILURE_COUNT,
    COALESCE(COUNT_IF(
        {POLICY_ID_COL} IS NULL
        OR TRIM(TO_VARCHAR({POLICY_ID_COL})) = ''
    ), 0) AS POLICY_KEY_FAILURE_COUNT,
    COALESCE(COUNT_IF({POLICY_INCEPTION_COL} IS NULL), 0)
        AS INCEPTION_NULL_COUNT,
    COALESCE(COUNT_IF({INDEX_COL} IS NULL), 0)
        AS TECHNICAL_INDEX_NULL_COUNT,
    COALESCE(COUNT_IF({EXPOSURE_YEAR_COL} IS NULL), 0)
        AS EXPOSURE_YEAR_NULL_COUNT,
    COALESCE(COUNT_IF(
        {EXPOSURE_YEAR_COL} IS NOT NULL
        AND (
            MOD({EXPOSURE_YEAR_COL}, 1) <> 0
            OR {EXPOSURE_YEAR_COL} < 1900
            OR {EXPOSURE_YEAR_COL} >
               YEAR(DATE '{TRUSTED_DATA_AS_OF_DATE}')
        )
    ), 0) AS EXPOSURE_YEAR_INVALID_COUNT,
    COALESCE(COUNT_IF({RAW_STATE_COLUMN} IS NULL), 0)
        AS RAW_STATE_NULL_COUNT
FROM (
{relation}
) AS SCOPED_CY_RETENTION_POPULATION
""".strip()




def build_key_qa_sql(select_sql: str) -> str:
    relation = _population_relation(select_sql)
    record_key = ", ".join(RECORD_KEY_COLS)
    return f"""
WITH SCOPED_CY AS (
{relation}
),
RECORD_KEY_DUPLICATES AS (
    SELECT COUNT(*) AS KEY_COUNT
    FROM SCOPED_CY
    GROUP BY {record_key}
    HAVING COUNT(*) > 1
),
TECHNICAL_INDEX_DUPLICATES AS (
    SELECT COUNT(*) AS KEY_COUNT
    FROM SCOPED_CY
    GROUP BY {INDEX_COL}
    HAVING COUNT(*) > 1
)
SELECT
    (SELECT COUNT(*) FROM RECORD_KEY_DUPLICATES) AS DUPLICATE_KEY_GROUPS,
    (SELECT COALESCE(SUM(KEY_COUNT - 1), 0) FROM RECORD_KEY_DUPLICATES)
        AS DUPLICATE_EXCESS_ROWS,
    (SELECT COUNT(*) FROM TECHNICAL_INDEX_DUPLICATES)
        AS TECHNICAL_INDEX_DUPLICATE_GROUPS,
    (SELECT COALESCE(SUM(KEY_COUNT - 1), 0) FROM TECHNICAL_INDEX_DUPLICATES)
        AS TECHNICAL_INDEX_DUPLICATE_EXCESS_ROWS
""".strip()




def build_tenure_distribution_sql(select_sql: str) -> str:
    """Count the selected population by canonical anchor-tenure group."""


    relation = _population_relation(select_sql)
    group_order = "\n        ".join(
        f"WHEN '{group}' THEN {position}"
        for position, group in enumerate(TENURE_GROUP_DOMAIN)
    )
    return f"""
SELECT
    "{ANCHOR_TENURE_COL}" AS TENURE_GROUP,
    COUNT(*) AS ROW_COUNT
FROM (
{relation}
) AS SCOPED_CY_RETENTION_POPULATION
GROUP BY "{ANCHOR_TENURE_COL}"
ORDER BY CASE "{ANCHOR_TENURE_COL}"
        {group_order}
        ELSE {len(TENURE_GROUP_DOMAIN)}
    END
""".strip()




def _normalized_result(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} query returned {type(frame)!r}, not DataFrame.")
    result = frame.copy()
    result.columns = [str(column).strip().upper() for column in result.columns]
    return result




def _normalize_chunk(
    chunk: pd.DataFrame,
    projected_columns: list[str],
    var_types: dict[str, str],
) -> pd.DataFrame:
    result = chunk.copy()
    result.columns = [str(column).strip().upper() for column in result.columns]
    if result.columns.tolist() != projected_columns:
        raise ValueError(
            "Snowflake chunk columns differ from the SQL projection. "
            f"expected={projected_columns}, observed={result.columns.tolist()}"
        )


    for column in projected_columns:
        if column in {TARGET, ACTIVE_AT_MONTH_1_COL, ACTIVE_AT_MONTH_13_COL}:
            result[column] = pd.to_numeric(result[column], errors="raise").astype(
                "Int8"
            )
        elif var_types[column] == "date":
            original_nonmissing = result[column].notna() & result[column].astype(
                "string"
            ).str.strip().ne("")
            converted = pd.to_datetime(result[column], errors="coerce")
            invalid = original_nonmissing & converted.isna()
            if invalid.any():
                examples = result.loc[invalid, column].astype(str).head(5).tolist()
                raise ValueError(
                    f"Could not parse non-null values in date field {column}: {examples}"
                )
            result[column] = converted
        elif var_types[column] == "continuous":
            original_nonmissing = result[column].notna() & result[column].astype(
                "string"
            ).str.strip().ne("")
            converted = pd.to_numeric(result[column], errors="coerce")
            invalid = original_nonmissing & converted.isna()
            if invalid.any():
                examples = result.loc[invalid, column].astype(str).head(5).tolist()
                raise ValueError(
                    f"Could not parse non-null values in numeric field {column}: {examples}"
                )
            result[column] = converted.astype("Float64")
        else:
            result[column] = result[column].astype("string")
    return result




def _tenure_groups_from_series(series: pd.Series, column: str) -> pd.Series:
    groups = series.astype("string").str.strip()
    invalid_group = groups.isna() | ~groups.isin(TENURE_GROUP_DOMAIN)
    if invalid_group.any():
        examples = (
            series.loc[invalid_group]
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        raise ValueError(
            f"{column} must contain only canonical tenure groups "
            f"{TENURE_GROUP_DOMAIN}; examples: {examples}"
        )
    return groups




def _tenure_group_counts_from_chunk(chunk: pd.DataFrame) -> dict[str, int]:
    groups = _tenure_groups_from_series(
        chunk[ANCHOR_TENURE_COL], ANCHOR_TENURE_COL
    )
    raw_counts = groups.value_counts()
    return {
        group: int(raw_counts[group])
        for group in TENURE_GROUP_DOMAIN
        if group in raw_counts.index
    }




def _write_parquet_chunks(
    chunks: Iterable[pd.DataFrame],
    temp_path: Path,
    projected_columns: list[str],
    var_types: dict[str, str],
) -> tuple[int, dict[str, int]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to write the retention snapshot."
        ) from exc


    writer = None
    arrow_schema = None
    row_count = 0
    tenure_group_row_counts: dict[str, int] = {}
    try:
        for chunk_number, raw_chunk in enumerate(chunks, start=1):
            chunk = _normalize_chunk(raw_chunk, projected_columns, var_types)
            if chunk.empty:
                continue
            chunk_group_counts = _tenure_group_counts_from_chunk(chunk)
            for group, count in chunk_group_counts.items():
                tenure_group_row_counts[group] = (
                    tenure_group_row_counts.get(group, 0) + count
                )
            if writer is None:
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                arrow_schema = table.schema
                writer = pq.ParquetWriter(
                    temp_path,
                    arrow_schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            else:
                table = pa.Table.from_pandas(
                    chunk,
                    schema=arrow_schema,
                    preserve_index=False,
                    safe=False,
                )
            writer.write_table(table)
            row_count += int(len(chunk))
            print(
                f"Downloaded chunk {chunk_number:,}: {len(chunk):,} rows; "
                f"running total {row_count:,}."
            )
    finally:
        if writer is not None:
            writer.close()
    if row_count == 0:
        raise ValueError("Snowflake returned zero eligible rows; nothing was published.")
    return row_count, {
        group: tenure_group_row_counts[group]
        for group in TENURE_GROUP_DOMAIN
        if group in tenure_group_row_counts
    }




def _single_int(frame: pd.DataFrame, column: str) -> int:
    result = _normalized_result(frame, column)
    if len(result) != 1 or column not in result.columns:
        raise ValueError(f"Expected one-row QA result containing {column}.")
    value = result.iloc[0][column]
    return 0 if pd.isna(value) else int(value)




def validate_tenure_distribution(
    frame: pd.DataFrame,
    *,
    expected_rows: int,
) -> dict[str, int]:
    """Validate counts; one-policy smoke tests may observe a subset of groups."""


    result = _normalized_result(frame, "tenure distribution")
    required = {"TENURE_GROUP", "ROW_COUNT"}
    missing = sorted(required.difference(result.columns))
    if missing:
        raise ValueError(
            f"Tenure distribution query is missing columns: {missing}"
        )
    counts: dict[str, int] = {}
    for row in result.loc[:, ["TENURE_GROUP", "ROW_COUNT"]].itertuples(
        index=False
    ):
        if pd.isna(row.TENURE_GROUP) or pd.isna(row.ROW_COUNT):
            raise ValueError("Tenure-group distribution contains null values.")
        group = str(row.TENURE_GROUP).strip()
        if group not in TENURE_GROUP_DOMAIN:
            raise ValueError(
                "Tenure-group distribution contains an unknown group: "
                f"{row.TENURE_GROUP!r}; expected {TENURE_GROUP_DOMAIN}."
            )
        numeric_count = pd.to_numeric(
            pd.Series([row.ROW_COUNT]), errors="coerce"
        ).iloc[0]
        if pd.isna(numeric_count) or numeric_count % 1 != 0:
            raise ValueError(
                f"Tenure-group {group!r} has a non-integer row count."
            )
        count = int(numeric_count)
        if count <= 0:
            raise ValueError(
                "Tenure-group distribution must contain positive row counts."
            )
        if group in counts:
            raise ValueError(f"Tenure distribution repeats group {group!r}.")
        if not TENURE_PROFILE.includes_group(group):
            raise ValueError(
                f"Tenure distribution contains out-of-scope group {group!r}."
            )
        counts[group] = count
    if sum(counts.values()) != int(expected_rows):
        raise ValueError(
            "Tenure distribution row counts do not match source QA: "
            f"distribution={sum(counts.values()):,}, expected={expected_rows:,}."
        )
    missing_groups = [
        group for group in TENURE_PROFILE.tenure_groups if group not in counts
    ]
    if missing_groups and POPULATION_SCOPE_MODE != "POLICY":
        raise ValueError(
            "The all-tenure table has no eligible rows for requested tenure "
            f"group(s): {missing_groups}."
        )
    observed_groups = tuple(
        group for group in TENURE_GROUP_DOMAIN if group in counts
    )
    if (
        POPULATION_SCOPE_MODE != "POLICY"
        and TENURE_PROFILE.business_scope == "ALL_TENURES"
        and (
        observed_groups != TENURE_GROUP_DOMAIN
        )
    ):
        raise ValueError(
            "ALL requires exactly the five canonical tenure groups "
            f"{TENURE_GROUP_DOMAIN}; observed={observed_groups}."
        )
    return {group: counts[group] for group in observed_groups}




def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")




def validate_runtime_environment() -> dict[str, str]:
    """Import live dependencies without opening a database connection."""


    import importlib


    modules = {
        "numpy": "numpy",
        "pandas": "pandas",
        "pyarrow": "pyarrow",
        "PyYAML": "yaml",
        "Jinja2": "jinja2",
        "snowflake-connector-python": "snowflake.connector",
        "snowflake-sqlalchemy": "snowflake.sqlalchemy",
        "SQLAlchemy": "sqlalchemy",
        "scipy": "scipy",
        "scikit-learn": "sklearn",
        "xgboost": "xgboost",
    }
    versions: dict[str, str] = {}
    failures: list[str] = []
    for package, module_name in modules.items():
        try:
            module = importlib.import_module(module_name)
            versions[package] = str(getattr(module, "__version__", "installed"))
        except Exception as exc:  # pragma: no cover - dependency diagnostics
            failures.append(f"{package}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError(
            "Required retention packages are missing or cannot be imported:\n- "
            + "\n- ".join(failures)
        )
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "This project requires Python 3.10 or newer; "
            f"observed {sys.version.split()[0]}."
        )
    from func.setup.setup import Setup  # noqa: F401
    from func.utils import get_run_id  # noqa: F401


    versions["python"] = sys.version.split()[0]
    versions["contract"] = PROJECT_CONTRACT_VERSION
    return versions




def _read_active_pipeline_state() -> dict[str, Any]:
    path = Path(ACTIVE_PIPELINE_STATE_PATH)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "MALFORMED"}
    if not isinstance(payload, dict):
        return {"status": "MALFORMED"}
    if not str(payload.get("status", "")).strip():
        return {**payload, "status": "UNKNOWN"}
    return payload




def assert_snapshot_refresh_allowed(run_id: str) -> None:
    """Protect the immutable snapshot used by a running or paused model run."""


    state = _read_active_pipeline_state()
    status = str(state.get("status", "")).upper()
    active_run_id = str(state.get("run_id", "")).strip()
    active_profile_hash = str(state.get("profile_hash", "")).strip()
    if status == "PAUSED":
        raise RuntimeError(
            "The active Logistic run is PAUSED. Do not refresh its Parquet. "
            "Run a0_one_click.py to resume modeling against the same snapshot."
        )
    if (
        status == "RUNNING"
        and active_profile_hash
        and active_profile_hash != PROFILE_HASH
    ):
        raise RuntimeError(
            "The active run belongs to a different tenure profile; snapshot "
            "refresh is blocked."
        )
    if status == "RUNNING" and active_run_id and active_run_id != run_id:
        raise RuntimeError(
            f"Pipeline run {active_run_id} is already RUNNING; refusing to "
            f"overwrite its snapshot from run {run_id}."
        )
    if status == "RUNNING" and not active_run_id:
        raise RuntimeError(
            "Pipeline state is RUNNING but has no run_id; snapshot refresh is blocked."
        )
    if state and status not in {"RUNNING", "COMPLETE", "FAILED"}:
        raise RuntimeError(
            f"Pipeline state is {status or 'UNKNOWN'}; snapshot refresh is blocked "
            "until active_run.json is repaired or restored."
        )




def verify_published_artifacts(
    output_path: str | Path,
    metadata_path: str | Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Verify a Parquet/metadata pair without loading all policy rows."""


    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - live environment only
        raise RuntimeError("pyarrow is required to verify the Parquet output.") from exc


    data_path = Path(output_path).resolve()
    meta_path = Path(metadata_path).resolve()
    if not data_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            "A complete retention snapshot requires both files: "
            f"data={data_path.is_file()} ({data_path}); "
            f"metadata={meta_path.is_file()} ({meta_path})."
        )
    with meta_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    parquet = pq.ParquetFile(data_path)


    expected = {
        "contract_version": PROJECT_CONTRACT_VERSION,
        "source_table": MAIN_TABLE_FQN,
        "config_row_count": len(contract["frame"]),
        "column_count": len(contract["projected_columns"]),
        "model_feature_count": len(contract["model_features"]),
        "target": TARGET,
        "profile_hash": PROFILE_HASH,
        **PROFILE_PAYLOAD,
        "profile_slug": PROFILE_SLUG,
        "tenure_predicate": TENURE_PROFILE.sql_predicate(
            f'"{ANCHOR_TENURE_COL}"'
        ),
        "tenure_is_model_candidate": TENURE_IS_MODEL_CANDIDATE,
        "exposure_year_as_predictor": INCLUDE_EXPOSURE_YEAR_AS_PREDICTOR,
        "record_key_columns": list(RECORD_KEY_COLS),
        "trusted_data_as_of_date": TRUSTED_DATA_AS_OF_DATE,
        "latest_anchor_term_start_date_exclusive": (
            LATEST_ANCHOR_TERM_START_DATE_EXCLUSIVE
        ),
        "excluded_property_states": sorted(EXCLUDED_PROPERTY_STATES),
        **POPULATION_SCOPE_PAYLOAD,
    }
    mismatches = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Retention metadata contract mismatch: {mismatches}")
    if metadata.get("config_sha256") != _sha256(contract["path"]):
        raise ValueError("Published metadata was created from a different config.")
    if metadata.get("generated_sql") != build_select_sql(
        contract["projected_columns"]
    ):
        raise ValueError(
            "Published metadata contains a different tenure-scoped SQL query."
        )
    if int(parquet.metadata.num_rows) != int(metadata.get("row_count", -1)):
        raise ValueError("Parquet and metadata row counts differ.")
    if parquet.schema_arrow.names != contract["projected_columns"]:
        raise ValueError("Published Parquet columns differ from the SQL projection.")
    if metadata.get("projected_columns") != contract["projected_columns"]:
        raise ValueError("Metadata projected columns differ from the config-derived SQL.")
    if metadata.get("model_features") != contract["model_features"]:
        raise ValueError("Metadata model candidates differ from the ROC cv_ind rows.")
    raw_group_counts = metadata.get("tenure_group_row_counts")
    if not isinstance(raw_group_counts, dict) or not raw_group_counts:
        raise ValueError(
            "Published metadata has no tenure-group row-count distribution."
        )
    group_counts: dict[str, int] = {}
    try:
        for raw_group, raw_value in raw_group_counts.items():
            group = str(raw_group).strip()
            if group != raw_group or group not in TENURE_GROUP_DOMAIN:
                raise ValueError(
                    f"Noncanonical tenure-group row-count key: {raw_group!r}."
                )
            if group in group_counts:
                raise ValueError(
                    f"Duplicate normalized tenure-group key: {group}."
                )
            count = int(raw_value)
            if isinstance(raw_value, bool) or count != raw_value:
                raise ValueError(
                    f"Tenure group {group!r} has a non-integer row count."
                )
            group_counts[group] = count
    except (TypeError, ValueError) as exc:
        raise ValueError("Published tenure-group row counts are invalid.") from exc
    if any(value <= 0 for value in group_counts.values()):
        raise ValueError("Published tenure-group row counts must be positive.")
    if sum(group_counts.values()) != int(metadata.get("row_count", -1)):
        raise ValueError(
            "Published tenure-group row counts do not sum to row_count."
        )
    observed_groups = [
        group for group in TENURE_GROUP_DOMAIN if group in group_counts
    ]
    if list(group_counts) != observed_groups:
        raise ValueError(
            "Published tenure-group row counts are not in canonical order."
        )
    if metadata.get("observed_tenure_groups") != observed_groups:
        raise ValueError(
            "Published observed tenure groups do not match row counts."
        )
    out_of_scope = [
        group
        for group in observed_groups
        if not TENURE_PROFILE.includes_group(group)
    ]
    if out_of_scope:
        raise ValueError(
            "Published metadata contains out-of-scope tenure groups: "
            f"{out_of_scope}"
        )
    missing_selected = [
        group
        for group in TENURE_PROFILE.tenure_groups
        if group not in group_counts
    ]
    if missing_selected and POPULATION_SCOPE_MODE != "POLICY":
        raise ValueError(
            "Published metadata is missing explicitly selected tenure "
            f"groups: {missing_selected}"
        )
    if (
        POPULATION_SCOPE_MODE != "POLICY"
        and TENURE_PROFILE.business_scope == "ALL_TENURES"
        and tuple(observed_groups) != TENURE_GROUP_DOMAIN
    ):
        raise ValueError(
            "Published ALL snapshot must contain exactly the five canonical "
            f"tenure groups {TENURE_GROUP_DOMAIN}; observed={observed_groups}."
        )
    if metadata.get("target_null_count") != 0 or metadata.get(
        "target_invalid_count"
    ) != 0:
        raise ValueError("Published metadata reports a failed target QA check.")
    if metadata.get("population_contract_failure_count") != 0:
        raise ValueError("Published metadata reports failed CY population invariants.")
    if metadata.get("duplicate_key_groups") not in (0, None) or metadata.get(
        "duplicate_excess_rows"
    ) not in (0, None):
        raise ValueError("Published metadata reports duplicate observation keys.")
    if metadata.get("technical_index_duplicate_groups") not in (0, None) or metadata.get(
        "technical_index_duplicate_excess_rows"
    ) not in (0, None):
        raise ValueError("Published metadata reports duplicate technical row keys.")
    return metadata




def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="ROC config CSV; otherwise use the validated project resolver.",
    )
    parser.add_argument("--output", type=Path, default=Path(DOWNLOAD_PATH))
    parser.add_argument(
        "--metadata-output", type=Path, default=Path(DOWNLOAD_METADATA_PATH)
    )
    parser.add_argument("--chunksize", type=int, default=DOWNLOAD_CHUNKSIZE)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Verify and reuse an existing snapshot instead of querying CY.",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--environment-check", action="store_true")
    modes.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.chunksize < 1:
        parser.error("--chunksize must be positive.")
    return args




def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    contract = load_download_contract(resolve_data_config_path(args.config))
    sql = build_select_sql(contract["projected_columns"])
    summary = {
        "contract_version": PROJECT_CONTRACT_VERSION,
        "source_table": MAIN_TABLE_FQN,
        "config_row_count": len(contract["frame"]),
        "projected_column_count": len(contract["projected_columns"]),
        "model_feature_count": len(contract["model_features"]),
        "profile": PROFILE_LABEL,
        "profile_hash": PROFILE_HASH,
        "profile_slug": PROFILE_SLUG,
        "tenure_filter_mode": PROFILE_PAYLOAD["tenure_filter_mode"],
        "selected_tenure_groups": PROFILE_PAYLOAD["selected_tenure_groups"],
        "tenure_predicate": TENURE_PROFILE.sql_predicate(
            f'"{ANCHOR_TENURE_COL}"'
        ),
        "row_selection": (
            "physical all-tenure CY rows; the validated tenure filter and "
            "preserved CA exclusion and optional policy-level test scope "
            "execute in Snowflake before download"
        ),
        **POPULATION_SCOPE_PAYLOAD,
    }


    if args.validate_only:
        print(json.dumps(summary, indent=2))
        return
    if args.environment_check:
        summary["environment"] = validate_runtime_environment()
        print(json.dumps(summary, indent=2))
        return
    if args.verify_existing:
        metadata = verify_published_artifacts(
            args.output, args.metadata_output, contract
        )
        summary["row_count"] = metadata["row_count"]
        print("EXISTING RETENTION SNAPSHOT VERIFIED")
        print(json.dumps(summary, indent=2))
        return
    if args.dry_run:
        print(json.dumps({**summary, "sql": sql}, indent=2))
        return


    output_path = args.output.resolve()
    metadata_path = args.metadata_output.resolve()
    if args.reuse_existing:
        metadata = verify_published_artifacts(output_path, metadata_path, contract)
        summary["row_count"] = metadata["row_count"]
        print("EXISTING RETENTION SNAPSHOT VERIFIED; CY QUERY SKIPPED BY REQUEST")
        print(json.dumps(summary, indent=2))
        return


    for path in (output_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    temp_token = f"{os.getpid()}_{datetime.now():%Y%m%d%H%M%S%f}"
    temp_output = output_path.with_name(
        f".{output_path.stem}.{temp_token}.tmp{output_path.suffix}"
    )
    temp_metadata = metadata_path.with_name(
        f".{metadata_path.name}.{temp_token}.tmp"
    )


    from func.setup.setup import Setup


    run_id = (
        os.environ.get("RETENTION_PIPELINE_RUN_ID", "").strip()
        or new_run_id(Path(__file__).stem)
    )
    assert_snapshot_refresh_allowed(run_id)
    runtime_setup = setup_params(run_id)
    runtime_setup["SAVE_ARTIFACTS"] = True
    setup = Setup(**runtime_setup, MEMORY_REDUCTION=False)
    started_utc = datetime.now(timezone.utc).isoformat()
    query_id = None
    try:
        connection = setup.conns[SF_CONN]
        schema = _normalized_result(
            connection.query_data(build_schema_sql(), verbose=10), "schema"
        )
        if "COLUMN_NAME" not in schema.columns:
            raise ValueError("INFORMATION_SCHEMA result is missing COLUMN_NAME.")
        available = set(schema["COLUMN_NAME"].astype(str).str.strip().str.upper())
        missing_projected = sorted(
            set(contract["projected_columns"]).difference(available)
        )
        if missing_projected:
            raise ValueError(
                "Config-selected or required SQL columns are missing from CY: "
                f"{missing_projected}"
            )
        unavailable_disabled = sorted(
            set(contract["all_config_columns"]).difference(available)
            - set(contract["model_features"])
        )


        source_qa = _normalized_result(
            connection.query_data(build_source_qa_sql(sql), verbose=10),
            "source QA",
        )
        expected_rows = _single_int(source_qa, "ROW_COUNT")
        target_nulls = _single_int(source_qa, "TARGET_NULL_COUNT")
        target_invalid = _single_int(source_qa, "TARGET_INVALID_COUNT")
        if expected_rows <= 0:
            raise ValueError("The SQL population query returned zero eligible rows.")
        tenure_distribution = _normalized_result(
            connection.query_data(
                build_tenure_distribution_sql(sql), verbose=10
            ),
            "tenure distribution",
        )
        tenure_group_row_counts = validate_tenure_distribution(
            tenure_distribution,
            expected_rows=expected_rows,
        )
        population_qa_columns = (
            "TARGET_NULL_COUNT",
            "TARGET_INVALID_COUNT",
            "TENURE_SCOPE_FAILURE_COUNT",
            "REPRESENTATIVE_TENURE_FAILURE_COUNT",
            "MONTH1_DATE_RULE_FAILURE_COUNT",
            "MONTH1_ACTIVE_FAILURE_COUNT",
            "SELECTED_TERM_START_FAILURE_COUNT",
            "SELECTED_TERM_END_FAILURE_COUNT",
            "MONTH13_DATE_RULE_FAILURE_COUNT",
            "TARGET_RULE_FAILURE_COUNT",
            "FIXED_HORIZON_FAILURE_COUNT",
            "FEATURE_AT_MONTH1_FAILURE_COUNT",
            "MATURITY_FAILURE_COUNT",
            "LATEST_START_FAILURE_COUNT",
            "POLICY_KEY_FAILURE_COUNT",
            "INCEPTION_NULL_COUNT",
            "TECHNICAL_INDEX_NULL_COUNT",
            "EXPOSURE_YEAR_INVALID_COUNT",
        )
        population_qa = {
            column: _single_int(source_qa, column)
            for column in population_qa_columns
        }
        raw_state_nulls = _single_int(source_qa, "RAW_STATE_NULL_COUNT")
        exposure_year_nulls = _single_int(
            source_qa, "EXPOSURE_YEAR_NULL_COUNT"
        )
        failed_population_qa = {
            column: count for column, count in population_qa.items() if count
        }
        if failed_population_qa:
            raise ValueError(
                "The authoritative CY population failed pre-download QA: "
                + "; ".join(
                    f"{column}={count:,}"
                    for column, count in failed_population_qa.items()
                )
            )


        key_qa = _normalized_result(
            connection.query_data(build_key_qa_sql(sql), verbose=10), "key QA"
        )
        duplicate_groups = _single_int(key_qa, "DUPLICATE_KEY_GROUPS")
        duplicate_excess_rows = _single_int(key_qa, "DUPLICATE_EXCESS_ROWS")
        technical_index_duplicate_groups = _single_int(
            key_qa, "TECHNICAL_INDEX_DUPLICATE_GROUPS"
        )
        technical_index_duplicate_excess_rows = _single_int(
            key_qa, "TECHNICAL_INDEX_DUPLICATE_EXCESS_ROWS"
        )
        if (
            duplicate_groups
            or duplicate_excess_rows
            or technical_index_duplicate_groups
            or technical_index_duplicate_excess_rows
        ):
            raise ValueError(
                "SQL observation keys are not unique: "
                f"adjusted-contract/anchor-term groups={duplicate_groups:,}, "
                f"adjusted-contract/anchor-term excess={duplicate_excess_rows:,}, "
                f"{INDEX_COL} groups={technical_index_duplicate_groups:,}, "
                f"{INDEX_COL} excess={technical_index_duplicate_excess_rows:,}."
            )


        chunks = connection.query_data(
            sql,
            query_name="retention_13_1_cy_population_download",
            chunksize=args.chunksize,
            verbose=10,
        )
        query_id = getattr(connection, "query_id", None)
        downloaded_rows, downloaded_group_row_counts = _write_parquet_chunks(
            chunks,
            temp_output,
            contract["projected_columns"],
            contract["var_types"],
        )
        if downloaded_rows != expected_rows:
            raise ValueError(
                f"Downloaded {downloaded_rows:,} rows but SQL QA reported "
                f"{expected_rows:,}; prior published files were left untouched."
            )
        if downloaded_group_row_counts != tenure_group_row_counts:
            raise ValueError(
                "Downloaded tenure distribution differs from the pre-download "
                "Snowflake QA result, possibly because the shared source was "
                "replaced during extraction. Prior published files were left "
                "untouched. "
                f"expected={tenure_group_row_counts}, "
                f"downloaded={downloaded_group_row_counts}"
            )


        payload = {
            "contract_version": PROJECT_CONTRACT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "started_utc": started_utc,
            "run_id": run_id,
            "source_table": MAIN_TABLE_FQN,
            "sf_connection_name": SF_CONN,
            "snowflake_query_id": query_id,
            "config_path": str(contract["path"]),
            "config_sha256": _sha256(contract["path"]),
            "config_row_count": len(contract["frame"]),
            "generated_sql": sql,
            "row_count": downloaded_rows,
            "column_count": len(contract["projected_columns"]),
            "model_feature_count": len(contract["model_features"]),
            "projected_columns": contract["projected_columns"],
            "configured_model_features": contract[
                "configured_model_features"
            ],
            "model_features": contract["model_features"],
            "var_types": contract["var_types"],
            "target": TARGET,
            **PROFILE_PAYLOAD,
            "profile_slug": PROFILE_SLUG,
            "profile_hash": PROFILE_HASH,
            "profile": PROFILE_LABEL,
            "profile_label": PROFILE_LABEL,
            "tenure_predicate": TENURE_PROFILE.sql_predicate(
                f'"{ANCHOR_TENURE_COL}"'
            ),
            "tenure_group_row_counts": downloaded_group_row_counts,
            "observed_tenure_groups": [
                group
                for group in TENURE_GROUP_DOMAIN
                if group in downloaded_group_row_counts
            ],
            "tenure_is_model_candidate": TENURE_IS_MODEL_CANDIDATE,
            "exposure_year_as_predictor": (
                INCLUDE_EXPOSURE_YEAR_AS_PREDICTOR
            ),
            "record_key_columns": list(RECORD_KEY_COLS),
            "trusted_data_as_of_date": TRUSTED_DATA_AS_OF_DATE,
            "latest_anchor_term_start_date_exclusive": (
                LATEST_ANCHOR_TERM_START_DATE_EXCLUSIVE
            ),
            "excluded_property_states": sorted(EXCLUDED_PROPERTY_STATES),
            **POPULATION_SCOPE_PAYLOAD,
            "disabled_config_columns_missing_in_cy": unavailable_disabled,
            "duplicate_key_groups": duplicate_groups,
            "duplicate_excess_rows": duplicate_excess_rows,
            "technical_index_duplicate_groups": (
                technical_index_duplicate_groups
            ),
            "technical_index_duplicate_excess_rows": (
                technical_index_duplicate_excess_rows
            ),
            "target_null_count": target_nulls,
            "target_invalid_count": target_invalid,
            "population_qa": population_qa,
            "population_contract_failure_count": int(
                sum(population_qa.values())
            ),
            "raw_state_null_count": raw_state_nulls,
            "exposure_year_null_count": exposure_year_nulls,
            "chunksize": args.chunksize,
            "output_path": str(output_path),
        }
        _write_json(temp_metadata, payload)
        verify_published_artifacts(temp_output, temp_metadata, contract)


        # Publish only after the complete new pair has passed verification.
        os.replace(temp_output, output_path)
        os.replace(temp_metadata, metadata_path)
        verify_published_artifacts(output_path, metadata_path, contract)
        print("RETENTION CY QUERY AND DOWNLOAD VERIFIED")
        print(f"Data:     {output_path}")
        print(f"Metadata: {metadata_path}")
        print(
            f"Rows: {downloaded_rows:,}; columns: "
            f"{len(contract['projected_columns']):,}; config candidates: "
            f"{len(contract['model_features']):,}."
        )
    finally:
        setup.shutdown()
        for temp_path in (temp_output, temp_metadata):
            if temp_path.exists():
                temp_path.unlink()




if __name__ == "__main__":
    main()
