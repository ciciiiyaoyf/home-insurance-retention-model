"""Offline checks of the public warehouse contract and artificial fixture."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re

from b_data_acquisition import b0_refresh_all_tenure_table as refresh
from examples.render_snowflake_sql import render
from params import MAIN_TABLE_FQN, SOURCE_TABLE_FQN


ROOT = Path(__file__).resolve().parents[1]


def read_csv(relative):
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_source_ddl_covers_exact_preserved_download_and_build_contract():
    dictionary = read_csv("docs/schema/source_columns.csv")
    columns = [row["column"] for row in dictionary]
    assert len(columns) == 393
    assert len(columns) == len(set(columns))
    assert set(columns) == set(refresh.raw_projection_columns())
    assert refresh.BUILD_SOURCE_REQUIRED_COLUMNS.issubset(columns)
    ddl = render("source-ddl")
    assert f"CREATE TABLE IF NOT EXISTS {SOURCE_TABLE_FQN}" in ddl
    for row in dictionary:
        assert re.search(r'\b' + re.escape(row["column"]) + r'\s+' + re.escape(row["snowflake_type"]) + r'(?=\s|,)', ddl)
    model_columns = set(columns) | set(refresh.DERIVED_OUTPUT_COLUMNS)
    routes = json.loads((ROOT / "c_modeling/feature_variant_map.json").read_text())["routes"]
    for route in routes:
        for family in ("logistic", "xgboost"):
            definition = route[family]
            if definition.get("origin") == "derived":
                assert set(definition["source_columns"]).issubset(model_columns)
            else:
                assert definition["feature_name"] in model_columns


def test_reconstructed_sql_passes_preserved_refresh_contract_without_snowflake():
    sql = refresh.load_build_sql()
    candidate = refresh.candidate_relation(sql)
    assert f"FROM {SOURCE_TABLE_FQN}" in candidate
    assert "FROM ELIGIBLE_ANCHOR_TERMS" in candidate
    assert len(refresh.final_output_columns(refresh.raw_projection_columns())) == 416
    executable = refresh.build_replace_sql(sql, retention_time_days=1)
    assert "/*__" not in executable
    assert f"CREATE OR REPLACE TABLE {MAIN_TABLE_FQN}" in executable
    assert "CANDIDATE AS" in refresh.build_preflight_qa_sql(sql)
    initial = render("initial-build")
    assert f"CREATE TABLE IF NOT EXISTS {MAIN_TABLE_FQN}" in initial
    assert "CREATE OR REPLACE" not in initial
    assert candidate in initial
    qa = refresh.build_postbuild_qa_sql()
    for name in (*refresh.REQUIRED_ZERO_COLUMNS, *refresh.INFORMATIONAL_QA_COLUMNS):
        assert re.search(r'\bAS\s+' + re.escape(name) + r'\b', qa, flags=re.IGNORECASE)
