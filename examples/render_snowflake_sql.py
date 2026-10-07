"""Render the existing Snowflake ingestion SQL for review, without connecting.

This utility does not replace the pipeline or fit a model. It exposes the
warehouse schema and a first-time table build using the same SQL relation as
the guarded refresh runner. Execution in Snowflake is a separate explicit step.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from params import MAIN_TABLE_FQN, SOURCE_TABLE_FQN  # noqa: E402
from b_data_acquisition.b0_refresh_all_tenure_table import (  # noqa: E402
    build_postbuild_qa_sql,
    build_source_anchor_qa_sql,
    candidate_relation,
    load_build_sql,
)


def render(kind: str) -> str:
    if kind == "initial-build":
        relation = candidate_relation(load_build_sql())
        return (
            "-- First-time table creation only; an existing table is left unchanged.\n"
            "-- Uses the same source projection and candidate relation as the refresh runner.\n"
            "-- Run source QA before executing; use guarded refresh for existing tables.\n"
            f"CREATE TABLE IF NOT EXISTS {MAIN_TABLE_FQN}\n"
            "DATA_RETENTION_TIME_IN_DAYS = 1\nAS\n"
            + relation + ";\n"
        )
    if kind == "source-qa":
        return build_source_anchor_qa_sql() + ";\n"
    if kind == "model-qa":
        return build_postbuild_qa_sql() + ";\n"
    relative_path = {
        "source-ddl": "b_data_acquisition/sql/create_policy_exposure_history.sql",
        "fixture": "examples/load_policy_exposure_history_sample.sql",
    }[kind]
    text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
    return text.replace(
        "INSURANCE_ANALYTICS.RETENTION.POLICY_EXPOSURE_HISTORY", SOURCE_TABLE_FQN
    ).replace(
        "INSURANCE_ANALYTICS.RETENTION.POLICY_RETENTION_13_1", MAIN_TABLE_FQN
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind", required=True,
        choices=("source-ddl", "fixture", "initial-build", "source-qa", "model-qa"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sql = render(args.kind)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(sql, encoding="utf-8")
    print(f"Rendered {args.kind}: {args.output.resolve()}")
    print("No Snowflake connection or SQL execution occurred.")


if __name__ == "__main__":
    main()
