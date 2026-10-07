# Data Acquisition

Prepare the tenure-scoped modeling population and input contracts for the Home Retention 13/1 pipeline.

[Back to the project README](../README.md) · [Snowflake schema and source examples](../docs/snowflake_schema.md)

## Inputs and data sources

| Input | Purpose |
| --- | --- |
| `INSURANCE_ANALYTICS.RETENTION.POLICY_RETENTION_13_1` | Shared all-tenure modeling table used for extraction. |
| `INSURANCE_ANALYTICS.RETENTION.POLICY_EXPOSURE_HISTORY` | Frozen source history underlying predictors and retention outcomes. |
| `../params.py` | Population rules, profile settings, paths, and connection configuration. |
| `sql/create_policy_exposure_history.sql` | Public source-table DDL and column types. |
| `sql/build_cy_ho_ret_13_1_all_tenure.sql` | Shared CY-table build SQL used by the refresh workflow. |
| `sql/qa_cy_ho_ret_13_1_all_tenure.sql` | Shared CY-table QA SQL. |

The public database and schema names are illustrative; they can be configured for your own Snowflake account. Source rows represent policy-exposure history. The [source data dictionary](../docs/schema/source_columns.csv) and [artificial rows](../examples/policy_exposure_history_sample.csv) show the expected structure without company data.

The modeling unit is a policy contract and anchor term. Predictors describe the Month-1 population; `RETENTION_13_1_IND` records retention at Month 13.

## Scripts and execution order

| Script | Responsibility |
| --- | --- |
| `b0_one_click.py` | Coordinates acquisition and propagates the run/profile context. |
| `b0_refresh_all_tenure_table.py` | Optional shared-table preflight, refresh, and postbuild validation. |
| `b1_download_retention_13_1.py` | Downloads the requested population from Snowflake. |
| `b2_prepare_model_inputs.py` | Produces the feature contract and initial candidate pool. |

Normal acquisition runs `b1` followed by `b2`. A requested source refresh runs before these steps.

Use `a_business_understanding/a0_one_click.py` to launch this module. The stage controller `b0_one_click.py` accepts no direct profile arguments.

## Population contract

| Rule | Current setting |
| --- | --- |
| Target | `RETENTION_13_1_IND`: retained `1`, churned `0`. |
| Record identity | `POLICY_CONTRACT_ID` and `ANCHOR_TERM_START_DATE`. |
| Cohort date | `MONTH_1_EVALUATION_DATE`. |
| Outcome-availability date | `MONTH_13_EVALUATION_DATE`. |
| Annual-term length | 360–370 days. |
| Latest anchor term start | Earlier than `2024-01-01`. |
| Trusted data-as-of date | `2024-12-31`. |
| Modeling state exclusion | California (`CA`). |
| Stored tenure groups | `0`, `1`, `2`, `3`, and top-coded `4+`. |

Month-13 outcomes must be observable under the data contract. California is excluded from the modeling extraction; the physical shared CY table remains all-state.

The tenure filter is applied in Snowflake before download. Selecting multiple tenure groups produces one pooled population.

## Commands

Run all commands from the repository root.

```powershell
# Print the extraction SQL without connecting to Snowflake.
python a_business_understanding/a0_one_click.py --tenures 2 --sql-dry-run

# Download and prepare model inputs without fitting models.
python a_business_understanding/a0_one_click.py --tenures 2 --acquisition-only

# Check a deterministic 1% policy-contract sample.
python a_business_understanding/a0_one_click.py --tenures 2 --sample-percent 1 --acquisition-only

# Check one policy contract; replace CONTRACT_ID.
python a_business_understanding/a0_one_click.py --tenures 2 --test-policy "CONTRACT_ID" --acquisition-only
```

`--sample-percent` accepts values from `0.01` inclusive to `100` exclusive, with at most two decimal places. Omit it for the full population. Sampling selects policy contracts and retains their selected terms. `--sample-percent` and `--test-policy` are mutually exclusive.

Sample and single-policy scopes support engineering checks. The current `c4` finalization stage requires a `FULL` population, so these examples stop after acquisition.

## Shared-table refresh

Refresh is explicit and affects the shared all-tenure CY table. The refresh controller expects an existing modeling table, so use the [initial-build procedure](../docs/snowflake_schema.md) when creating the public schema for the first time. The tiny source fixture is for inspection and SQL illustration; it is not sized to satisfy the full production-style refresh and modeling checks.

```powershell
# Run live preflight checks without replacing the table.
python a_business_understanding/a0_one_click.py --refresh-preflight-only

# Refresh and validate the table, then stop.
python a_business_understanding/a0_one_click.py --refresh-source-only

# Refresh and validate the table, then run the tenure-2 pipeline.
python a_business_understanding/a0_one_click.py --tenures 2 --refresh-source
```

The refresh workflow checks source/schema compatibility, candidate population QA, table metadata, and dependencies. A completed refresh also validates the resulting table against the preflight population checks.

Refresh flags are mutually exclusive and cannot be combined with sample/single-policy scope or `--sql-dry-run`.

## Outputs

Paths below are relative to `output/pipeline_runs/<RUN_ID>/`.

| Artifact | Contents |
| --- | --- |
| `input_snapshot/retention_13_1_population.parquet` | Frozen extracted population. |
| `input_snapshot/retention_13_1_population.metadata.json` | Snapshot metadata and population identity. |
| `model_inputs/feature_contract.csv` | Input feature contract consumed by modeling. |
| `model_inputs/candidate_pool.txt` | Initial candidate feature list. |
| `source_refresh/receipt.json` | Checks, query provenance, and refresh/preflight status, when requested. |
| `pipeline_status.json` | Run status maintained by orchestration. |

Family-specific routing and quality screening occur later in `c_modeling/c1_prepare_modeling_data.py`.

## Configuration and recovery

- Snowflake uses the project `SNOWFLAKE` connection, configured through environment variables or `SNOWFLAKE_CONNECTION_NAME`; see the root [setup instructions](../README.md#setup).
- `RETENTION_DATA_CONFIG_PATH` selects an alternative data configuration.
- `RETENTION_DOWNLOAD_CHUNKSIZE` defaults to `100000`.
- Modeling recovery reuses verified acquisition artifacts. Keep the snapshot, metadata, feature contract, and candidate pool together.
- Use the recovery commands in the [project README](../README.md); downstream recovery does not automatically refresh Snowflake data.
