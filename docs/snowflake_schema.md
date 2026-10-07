# Snowflake source schema and policy-term lineage

The repository retains the original project architecture: **Snowflake exposure history → Snowflake policy-term modeling table → Python download and validation → feature preparation, modeling, evaluation, and reports**. The files here expose the warehouse input contract without distributing private records or requiring the original warehouse. They do not introduce a different local-file model.

## Tables and field dictionaries

| Layer | Generic relation | Grain |
| --- | --- | --- |
| Source | `INSURANCE_ANALYTICS.RETENTION.POLICY_EXPOSURE_HISTORY` | One exposure segment/record, identified by `EXPOSURE_ROW_ID`. A contract-term may have several records. |
| Modeling | `INSURANCE_ANALYTICS.RETENTION.POLICY_RETENTION_13_1` | One eligible `POLICY_CONTRACT_ID` + `ANCHOR_TERM_START_DATE`. |
| Download | Run-specific Parquet and metadata written by `b1_download_retention_13_1.py` | The selected modeling-table population and configured predictors, retaining structural audit fields. |

The checked-in SQL shows generic default names. `SNOWFLAKE_DATABASE` and `SNOWFLAKE_SCHEMA` configure the Python connection and rendered relations. Table names remain fixed to the generic names above so the feature configuration's `source_tbl` stays consistent. Use the renderer below when choosing a different database or schema; editing only a connection setting does not alter a SQL file already copied into a worksheet.

- [Source DDL](../b_data_acquisition/sql/create_policy_exposure_history.sql) declares **393 source columns**, including disabled configuration fields so the input shape remains visible.
- [Source field dictionary](schema/source_columns.csv) records types, configuration roles, structural requirements, candidate flags, and intended prediction-time availability.
- [Derived model fields](schema/model_derived_columns.csv) documents **24 build outputs**. Fresh builds emit 23 of these because `ANCHOR_TENURE_YEARS_INT` is an optional compatibility field. The default complete model relation has **416 columns** before the downloader's narrower projection.
- [Schema manifest and generation notes](schema/README.md) explain type choices and how the dictionaries were reconciled with the configuration and code.

The source contract comes from the 404-row feature configuration, excluding 14 fields that SQL recomputes, and adding three required build inputs: `TENURE_YEARS_INT`, `EARNED_EXPOSURE_VALUE`, and `TOT_ERND_PREM_AMT`. IDs use `VARCHAR`, including an exposure identifier whose legacy configuration dtype is numeric. This permits artificial or genuine alphanumeric identifiers without losing leading zeros. No private table contents are included.

The model table is created by the existing build relation; it is not an independent replacement input. Later model-specific features such as `WRITTEN_PREM_BIN` and retention-based state groups are produced by the Python training workflow from the columns identified in `c_modeling/feature_variant_map.json`.

## Observation and feature lineage

1. **Group exposure history into contract-terms.** A term must have an unambiguous scheduled end and canonical tenure group (`0`, `1`, `2`, `3`, or `4+`). The group field is authoritative; integer tenure is retained for consistency checks.
2. **Apply the preserved observation window.** Annual terms span 360–370 days. Anchor starts precede January 1, 2024, and `term_end + 30 days` must be observable by December 31, 2024.
3. **Require activity at Month 1.** `MONTH_1_EVALUATION_DATE = term_start + 30 days`. Select the latest exposure record active on that date, breaking any remaining ordering tie by exposure identifier. Source QA rejects unresolved feature-row ties. Interval comparisons are inclusive. A null exposure end is treated as the trusted data cutoff.
4. **Build deductible and premium variables.** The refresh runner's contract selects the earliest exposure record in each contract-term for these specific calculations. It extracts the deductible token, converts a percent deductible using coverage A, and creates deductible bands. `WRITTEN_PREM` is the preserved ratio `TOT_ERND_PREM_AMT / EARNED_EXPOSURE_VALUE`, with zero for a nonpositive denominator. This is a calculated proxy despite its historical field name.
5. **Calculate the previous-term premium before maturity filtering.** `LAG` operates across all source terms for the contract. `RATE_CHANGE = (current − previous) / current × 100`; it is not the usual denominator-of-previous percentage change. The first term has null `RATE_CHANGE` and `RATE_CHANGE2 = -100`. Preserve those conventions when interpreting output.
6. **Observe activity at Month 13.** `MONTH_13_EVALUATION_DATE = term_end + 30 days`. `RETENTION_13_1_IND = 1` if any exposure interval on the same contract covers that date, including intervals associated with another inception record. Otherwise the label is `0`. The contract is the continuity key.
7. **Download the configured population.** The physical build preserves the source's state scope; downstream Python applies the preserved state and tenure selection rules. The default extraction excludes California. Configuration flags, structural fields, and model-family routing determine what becomes a predictor.

Exposure interval validity establishes which row is eligible, but does not establish when every value on that row became available. In particular, earned-value aggregates used by the premium proxy need a point-in-time review when connecting an authorized source. The dictionaries distinguish intended availability from verified upstream timing. This reconstruction preserves the source project's feature calculations without claiming that row selection alone proves absence of leakage.

## Small artificial exposure example

[The sample CSV](../examples/policy_exposure_history_sample.csv) contains **five wholly invented exposure records for two artificial contracts**, with twenty representative source fields. [The matching INSERT](../examples/load_policy_exposure_history_sample.sql) loads those exact records into the full nullable schema. Omitted predictors stay null. Synthetic category names intentionally do not reproduce original encodings or distributions.

| Contract | Anchor term start | Month-13 observation | Label | Reason |
| --- | --- | --- | --- | --- |
| `SYNTH_CONTRACT_A` | 2021-01-01 | 2022-01-31 | 1 | The next term covers the observation date. |
| `SYNTH_CONTRACT_A` | 2022-01-01 | 2023-01-31 | 1 | The following term covers the observation date. |
| `SYNTH_CONTRACT_A` | 2023-01-01 | 2024-01-31 | 0 | Its exposure ends in July 2023, with no later coverage. |
| `SYNTH_CONTRACT_B` | 2022-06-01 | 2023-07-01 | 0 | Coverage stops at the end of the first term. |

Contract A's first term has two source rows. The Month-1 snapshot selects `SYNTH_A_2021_02`, while the preserved term-feature contract takes its premium and deductible calculations from the earlier row. [Expected policy-term values](../examples/expected_policy_terms.csv) make this distinction explicit. Numeric comparisons should allow ordinary floating-point tolerance.

This example illustrates the actual warehouse shape and target construction. It is not a generated training dataset and does not support performance claims. It contains only tenure groups 0–2, so the guarded all-tenure refresh correctly refuses it as a production population requiring all five groups. Sparse predictors and four model rows are also insufficient for the full selection, cross-validation, and holdout pipeline.

## Review and run the warehouse example

Install the project's dependencies and configure the Snowflake environment variables as described in the root README. A database and schema must already exist with permissions appropriate to your workspace. The commands below **only render SQL files locally**; they do not connect to Snowflake or execute statements.

```bash
python examples/render_snowflake_sql.py --kind source-ddl --output output/schema_demo/01_source_ddl.sql
python examples/render_snowflake_sql.py --kind fixture --output output/schema_demo/02_fixture.sql
python examples/render_snowflake_sql.py --kind source-qa --output output/schema_demo/03_source_qa.sql
python examples/render_snowflake_sql.py --kind initial-build --output output/schema_demo/04_initial_model.sql
python examples/render_snowflake_sql.py --kind model-qa --output output/schema_demo/05_model_qa.sql
```

For a new, empty demonstration schema, review and execute these files in order in your own Snowflake worksheet:

1. The source DDL uses `CREATE TABLE IF NOT EXISTS`; it does not replace an existing source.
2. The fixture INSERT appends records. Execute it once in the empty demonstration source; rerunning it creates duplicate artificial records.
3. Source QA reports null structural values, tenure/term ambiguities, feature-row ties, and deductible parsing failures. These failure counts should be zero for the fixture.
4. The initial model SQL uses `CREATE TABLE IF NOT EXISTS` with the **same candidate relation** as the refresh runner. It leaves an existing model table unchanged. This explicit first-time route is needed because the guarded refresh inspects and preserves an existing target table.
5. Model QA returns the runner's exact diagnostic column contract. For this tiny fixture, expect four rows, two retained outcomes, and only three distinct tenure groups. Standalone old/new parity checks compare the current table to itself; use the guarded refresh's preflight to compare a proposed replacement with the existing table.

Once an adequately populated authorized warehouse source and initial model table exist, run the original Python workflow. Use the existing opt-in refresh route for replacements so schema preservation, tags/grants, retention settings, dependencies, source QA, and preflight/postbuild checks remain in force. The example renderer never performs a replacement.

## Reconciliation of recovered SQL and Python

The recovered SQL document was shorter than the retained refresh runner's required contract. It lacked all three rendering markers, several audit/compatibility outputs, the latest-start boundary, and the runner's explicit tenure-normalization text. It also computed deductible/premium from the Month-1 row and lagged only eligible terms, whereas the Python guard explicitly required earliest-term feature construction and lagging before maturity filtering.

The checked-in build template was reconstructed to satisfy those **existing Python requirements**, retaining the target interval rule, deductible conversion/bands, premium formula, and rate-change conventions. It adds the audit and compatibility fields consumed by the runner rather than weakening the runner's checks. Both standalone QA files were rendered from the runner's existing query generators. This reconciliation is a portability repair, not evidence that the recovered documents were a complete, internally consistent snapshot of the last company run.

Offline checks validate schema coverage, SQL rendering contracts, QA output contracts, and the sample's manually specified interval outcomes. No live Snowflake execution or original-data performance reproduction is claimed.
