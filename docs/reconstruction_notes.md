# Public repository reconstruction

This repository reconstructs the author's retention project from saved source
documents. It keeps the modular project and Snowflake-to-Python workflow. It
does not claim to restore the original Git commit history or include the
company's records, credentials, fitted models, or measured results.

## Preserved project

The latest coherent versions of the acquisition, preparation, Logistic
Regression, XGBoost, chronological evaluation, cross-tenure, reporting,
orchestration, and test modules are included. The model algorithms, numerical
defaults, selection rules, target polarity, and holdout separation were retained.
Structural field identifiers were renamed consistently across Python, SQL,
feature configuration, routing, and tests.

The current reporting scripts belong in `e_evaluate_and_report/`; their imports
and test modules identify that location. Older duplicate refresh/plotting
documents were superseded by the current versions. Two preparation documents
were identical, so only one canonical module is included.

## Public adaptations and restored dependencies

| Area | Change |
| --- | --- |
| Warehouse namespace | The default source is `INSURANCE_ANALYTICS.RETENTION.POLICY_EXPOSURE_HISTORY`; the model table is `INSURANCE_ANALYTICS.RETENTION.POLICY_RETENTION_13_1`. Database/schema are validated environment settings. |
| Connection utilities | The historical `FIGConnect` interface remains at its original import path. Company-specific connection inventory, hosts, keyring prompts, and unused institutional backends were replaced by lazy, user-configured Snowflake connections. Query batching, query IDs, logging, and run artifacts remain supported. |
| Contract identity | The public contract has a new version so renamed schemas cannot silently reuse an original run's metadata/profile identity. |
| Feature routing asset | The recovered JSON routing document lacked its final closing brace. That syntax defect was repaired without changing its routes. |
| Build SQL | The saved short SQL did not satisfy the preserved refresh runner's required projection markers, normalization, audit columns, and term-feature construction. The SQL template was reconciled to that runner. The schema guide documents the resulting source and derived fields. |
| QA SQL | Missing standalone QA files were generated from the existing refresh runner's source/post-build QA query functions. |
| Optional reselection helpers | `reselect_saved_xgboost.py` and `reselected_outer_support.py` were not recoverable as source documents. They were reconstructed against the preserved tuned-finalist artifacts, selection policy, profile, and state-integrity interfaces. They do not fit models, open final holdout data, or invent results. |
| Optional evaluator paths | Repository-root detection, policy import qualification, and the launcher source path were corrected to match the reconstructed layout. |
| Dependencies | Runtime and test requirements were made standalone; Conda uses the same dependency files. Unused Oracle, ODBC, and corporate authentication dependencies are no longer mandatory. |
| Existing test maintenance | One XGBoost test's AST helper harness and fixture were updated to match the current confirmation-queue API. Its assertions now verify that a per-pass budget does not truncate the required queue. |
| New public schema assets | Source DDL, typed dictionaries, SQL renderer, and a five-row artificial exposure fixture document the warehouse contract. These are examples, not original data or a replacement modeling project. |

The source schema has **393 columns**. The reconciled build produces **416
model-table columns**. The retained **404-row feature configuration** determines
the extraction contract; the default validation reports **124 projected columns**
and **107 model candidates** before later routing and training-only screening.
These are code/schema counts, not production data statistics.

## Validation performed

On Python 3.12, the complete repository test suite passed:

```text
104 passed
```

The tests cover the existing model/evaluation contracts, public connection
behavior without authentication, saved-artifact integrity, schema coverage,
and rendered SQL. The model/reporting modules imported successfully, seven
command-line entry points displayed help, and the main launcher's tenure-2 SQL
dry run completed without opening a warehouse connection.

The actual rendered Snowflake source/fixture/build relation was parsed with
SQLGlot and translated to DuckDB **inside a test only**. Its four policy-term
outputs matched the explicit example targets, dates, premiums, price changes,
and deductibles. Production ingestion still uses Snowflake. This test is useful
for relational logic and is not a live Snowflake execution test.

No live Snowflake query, full-population model training, or original-data
performance comparison was performed. The Conda definition retains Python
3.10.15; that separate interpreter was not exercised during this verification.
The small fixture cannot satisfy the full pipeline's population/history gates.
Use an appropriately populated instance of the documented schema for a full run.
