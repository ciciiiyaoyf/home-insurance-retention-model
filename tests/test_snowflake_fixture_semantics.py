"""Check rendered warehouse logic on artificial rows using a local SQL engine.

SQLGlot reads the actual Snowflake SQL and translates its relational expression
to DuckDB for this test only. Production ingestion still uses Snowflake. This
test does not establish live Snowflake privileges or execution compatibility.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import sqlglot
from sqlglot import exp
from sqlglot.errors import ErrorLevel

from b_data_acquisition.b0_refresh_all_tenure_table import candidate_relation, load_build_sql
from examples.render_snowflake_sql import render


ROOT = Path(__file__).resolve().parents[1]


def _local_sql(statement):
    """Omit documentation-only DDL comments; reject other unsupported syntax."""
    statement = statement.copy()
    for column in statement.find_all(exp.ColumnDef):
        column.set("constraints", [
            item for item in column.args.get("constraints", [])
            if not isinstance(item.args.get("kind"), exp.CommentColumnConstraint)
        ])
    for properties in statement.find_all(exp.Properties):
        properties.set("expressions", [
            item for item in properties.expressions
            if not isinstance(item, exp.SchemaCommentProperty)
        ])
    return statement.sql(dialect="duckdb", unsupported_level=ErrorLevel.RAISE)


def test_real_rendered_sql_matches_explicit_policy_term_expectations():
    expected = pd.read_csv(
        ROOT / "examples" / "expected_policy_terms.csv", dtype=str, keep_default_na=False
    )
    with duckdb.connect() as connection:
        connection.execute("ATTACH ':memory:' AS INSURANCE_ANALYTICS")
        connection.execute("CREATE SCHEMA INSURANCE_ANALYTICS.RETENTION")
        for kind in ("source-ddl", "fixture"):
            for statement in sqlglot.parse(render(kind), read="snowflake"):
                if statement is not None:
                    connection.execute(_local_sql(statement))
        relation = sqlglot.parse_one(candidate_relation(load_build_sql()), read="snowflake")
        actual = connection.execute(_local_sql(relation)).df()

    keys = ["POLICY_CONTRACT_ID", "ANCHOR_TERM_START_DATE"]
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    assert len(actual) == len(expected) == 4
    assert not actual.duplicated(keys).any()
    numbers = {
        "RETENTION_13_1_IND", "WRITTEN_PREM", "PREV_WRITTEN_PREM",
        "RATE_CHANGE", "RATE_CHANGE2", "DED_PROCESS",
    }
    dates = {
        "ANCHOR_TERM_START_DATE", "MONTH_1_EVALUATION_DATE", "MONTH_13_EVALUATION_DATE",
    }
    for name in expected.columns:
        if name in numbers:
            np.testing.assert_allclose(
                pd.to_numeric(actual[name]).to_numpy(dtype=float),
                pd.to_numeric(expected[name], errors="coerce").to_numpy(dtype=float),
                rtol=1e-8, atol=1e-8, equal_nan=True, err_msg=name,
            )
        elif name in dates:
            assert pd.to_datetime(actual[name]).dt.strftime("%Y-%m-%d").tolist() == expected[name].tolist()
        else:
            assert actual[name].astype(str).tolist() == expected[name].tolist()

    # These two views must intentionally differ for the multi-interval first term:
    # latest active Month-1 row, but earliest term-row premium/deductible features.
    first = actual.iloc[0]
    assert first["EXPOSURE_ROW_ID"] == "SYNTH_A_2021_02"
    assert first["WRITTEN_PREM"] == 1200.0
    assert first["DED_PROCESS"] == 1000.0
    assert actual["RETENTION_13_1_IND"].tolist() == [1, 1, 0, 0]
