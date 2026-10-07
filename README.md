# Home Insurance Retention Modeling

This is the portfolio version of my homeowner-retention modeling project at Farmers Insurance. The business objective is to estimate whether a policy will remain active after its first observed year, helping assess retention risk and informing analysis of customer value, pricing, and future policy tenure. The Month-13 retention probability is a starting point for longer-term policy-life analysis, rather than a direct estimate of lifetime value or policy lifespan.

The repository preserves the project's modular Python and SQL workflow: **Snowflake extraction → feature preparation → Logistic Regression and XGBoost selection → chronological evaluation → cross-tenure comparisons and reporting**. It retains the source code's model-selection and recovery design while replacing company-specific infrastructure and structural data identifiers with a documented, generic Snowflake contract.

No company records, credentials, fitted models, or actual model results are included. The small artificial examples explain the schema; they are not substitutes for the population required to fit and evaluate the full pipeline.

For a quick code walkthrough, read the [repository guide](docs/repository_guide.md). The [Snowflake schema guide](docs/snowflake_schema.md) shows the source columns, example records, table-build logic, and initial setup. The [reconstruction notes](docs/reconstruction_notes.md) distinguish preserved code from restored dependencies and public-facing adaptations.

- **Target:** `RETENTION_13_1_IND` (`1` = retained at Month 13; `0` = churned), for the eligible Month-1 population.
- **Observation:** one policy contract and anchor term. Row counts are policy-term counts, not necessarily distinct-policy counts.
- **Model families:** regularized Logistic Regression and XGBoost.
- **Modeling table:** `INSURANCE_ANALYTICS.RETENTION.POLICY_RETENTION_13_1`.
- **Predictor lineage:** `INSURANCE_ANALYTICS.RETENTION.POLICY_EXPOSURE_HISTORY`.
- **Final holdout:** reserved during development; scored separately after model and tenure-grouping decisions are frozen.

## Repository structure

| Location | Responsibility |
| --- | --- |
| `a_business_understanding/a0_one_click.py` | Main launcher, profile selection, run state, and recovery. |
| [b_data_acquisition](b_data_acquisition/README.md) | Snowflake extraction, shared-table refresh, and model-input contracts. |
| [c_modeling](c_modeling/README.md) | Preparation, feature selection, tuning, outer evaluation, diagnostics, and provisional refit. |
| [d_cross_tenure](d_cross_tenure/README.md) | Frozen cross-tenure comparisons, native baseline lift, and pooled-model assessment. |
| [e_evaluate_and_report](e_evaluate_and_report/README.md) | Saved-model reports and explicit final-holdout scoring. |
| `docs/`, `examples/` | Public schema, data dictionary, illustrative source rows, and reconstruction notes. |
| `params.py` | Shared data, profile, path, and validation settings. |
| `func/` | Shared connection, setup, orchestration, and locking utilities. |
| `tests/` | Software tests. |
| `output/` (generated) | Run artifacts, cross-tenure results, and orchestration state; excluded from Git. |

## Execution flow

The main launcher runs these controllers in order:

1. `b_data_acquisition/b0_one_click.py`: optional source refresh, population download, and input-contract preparation.
2. `c_modeling/c0_one_click.py`: `c1` preparation, `c2` Logistic selection, `c3` XGBoost selection, and `c4` outer evaluation and finalization.

Cross-tenure analysis, additional reports, and final-test scoring are separate commands. Use the main launcher for acquisition and modeling; `b0` and `c0` do not accept profile options directly.

## Setup

Run commands from the repository root, where `params.py` and the module directories are located. The original project targets Python 3.10. Create an isolated Python environment and install the requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python a_business_understanding/a0_one_click.py --help
```

On macOS/Linux, activate the environment with `source .venv/bin/activate` instead.

### Configure Snowflake

The data path remains Snowflake-backed. Set connection values for your own account as described in [.env.example](.env.example). For example, in PowerShell with browser SSO:

```powershell
$env:SNOWFLAKE_ACCOUNT = "YOUR_ORG-YOUR_ACCOUNT"
$env:SNOWFLAKE_USER = "YOUR_USER"
$env:SNOWFLAKE_WAREHOUSE = "YOUR_WAREHOUSE"
$env:SNOWFLAKE_ROLE = "YOUR_ROLE"
$env:SNOWFLAKE_AUTHENTICATOR = "externalbrowser"
```

A named connection in Snowflake's standard `connections.toml` can also be selected with `SNOWFLAKE_CONNECTION_NAME`. `.env.example` is a reference; the application does not automatically load a `.env` file. Set values in the process environment or configure a named connection.

The default public namespace is `INSURANCE_ANALYTICS.RETENTION`. `SNOWFLAKE_DATABASE` and `SNOWFLAKE_SCHEMA` can point to your own namespace. `POLICY_EXPOSURE_HISTORY` supplies source history; `POLICY_RETENTION_13_1` is the derived modeling table. Follow the [schema setup instructions](docs/snowflake_schema.md) to create these objects and populate source records before a live pipeline run. An account, warehouse, and suitable object privileges must already be available.

The included source fixture contains only two artificial contracts and demonstrates the table relationship. Full modeling requires a suitably sized population, sufficient chronological coverage, and the tenure groups needed by the configured validation checks. This repository does not include the original population.

### Inspect extraction without warehouse access

After installing dependencies, print the generated extraction query:

```powershell
python a_business_understanding/a0_one_click.py --tenures 2 --sql-dry-run
```

`RETENTION_DATA_CONFIG_PATH` can select an alternative feature configuration. The default configuration is stored under `b_data_acquisition/config_vault/HOME/`.

## Run a pipeline

Run a full tenure-2 profile:

```powershell
python a_business_understanding/a0_one_click.py --tenures 2
```

Other business scopes:

```powershell
python a_business_understanding/a0_one_click.py --new-business
python a_business_understanding/a0_one_click.py --renewal
python a_business_understanding/a0_one_click.py --all
```

Each command creates a separate run. Choose the scope needed for the analysis.

### Tenure selectors

| Selector | Population |
| --- | --- |
| No selector | All stored tenure groups; current default is `ALL`. |
| `--new-business` | Group `0`. |
| `--renewal` | Groups `1`, `2`, `3`, and `4+`. |
| `--all` or `--tenures ALL` | All business scopes. |
| `--tenure 2` or `--tenures 2` | One tenure group. |
| `--tenures 1 3 "4+"` | One pooled profile containing the listed groups. |
| `--tenures 1-3` | One pooled profile containing groups `1` through `3`. |
| `--tenure-min 1 --tenure-max 3` | An inclusive range of stored groups. |

The business-scope flags are mutually exclusive. Tenure is top-coded: any whole-number selector of `4` or greater selects the complete `4+` group. For example, `--tenure 7` does not isolate seventh-year policies.

Selecting multiple groups fits one pooled profile. Run separate single-group profiles to obtain the native models needed for cross-tenure comparisons.

## Development and evaluation design

- Cohorts are ordered by `MONTH_1_EVALUATION_DATE`; outcome maturity is checked using `MONTH_13_EVALUATION_DATE`.
- Development uses **5 outer chronological folds**, each with a **3-month** validation window.
- Feature discovery uses **3 inner folds**, each with a **3-month** validation window, within the oldest outer training boundary.
- The final **6-month holdout** remains unscored during feature selection, model-family selection, and standard cross-tenure analysis.
- Logistic and XGBoost share the prepared population and fold definitions; their selected history policies may use different eligible training subsets.

Actual dates and row counts are recorded in each run's preparation artifacts. See the [modeling README](c_modeling/README.md) for selection methods, diagnostics, and final-test interpretation.

## Recovery and reruns

Replace `RUN_ID` with the existing run identifier and use its original explicit tenure or business-scope selector.

```powershell
# Resume compatible failed work.
python a_business_understanding/a0_one_click.py --tenures 2 --resume-run-id "RUN_ID"

# Rebuild preparation and downstream modeling for a failed run.
python a_business_understanding/a0_one_click.py --tenures 2 --resume-run-id "RUN_ID" --rerun-preparation

# Restart canonical XGBoost selection and dependent evaluation.
python a_business_understanding/a0_one_click.py --tenures 2 --resume-run-id "RUN_ID" --restart-modeling-from xgboost

# Create separate XGBoost reconfirmation results for a completed run.
python a_business_understanding/a0_one_click.py --tenures 2 --rerun-xgboost-run-id "RUN_ID"
```

These are alternative recovery operations:

| Operation | Behavior |
| --- | --- |
| Resume | Validates a failed run and resumes the latest compatible boundary, including reusable Logistic checkpoints. |
| Rerun preparation | Reuses verified acquisition inputs; archives preparation and dependent outputs; reruns `c1` through `c4`. |
| Restart from XGBoost | Reuses acquisition, preparation, and completed Logistic; archives XGBoost and dependent evaluation outputs; accepts complete or failed runs. |
| XGBoost reconfirmation | Writes `xgboost_reconfirmation_<timestamp>/` while preserving canonical XGBoost artifacts and pipeline status. |

After a safe Logistic pause, repeat the identical profile command. If a previous process has stopped but left a stale `RUNNING` marker, start a fresh run with:

```powershell
python a_business_understanding/a0_one_click.py --tenures 2 --recover-stale-run
```

Stale-run recovery does not resume checkpoints. Recovery paths validate profile, snapshot, preparation, code, and artifact compatibility.

## Outputs

Pipeline artifacts are stored under `output/pipeline_runs/<RUN_ID>/`.

| Location | Contents |
| --- | --- |
| `pipeline_status.json` | Overall status and recovery metadata. |
| `input_snapshot/`, `model_inputs/` | Frozen population, metadata, feature contract, and initial candidates. |
| `preparation/` | Fold definitions, candidate audits, and family-specific feature contracts. |
| `logistic/`, `xgboost/` | Selection results, model recipes, and diagnostics. |
| `outer_evaluation/` | Outer-fold predictions, metrics, model comparison, and fold bundles. |
| `deployment/` | Provisional pre-holdout refit and its manifest. |
| `reporting/`, `final_test/` | Outputs from separate reporting and final-test commands. |

Cross-tenure results are written under `output/cross_tenure/`. The provisional bundle is marked `PROVISIONAL_PREHOLDOUT_REFIT` with `deployment_eligible = false`; pipeline completion does not deploy a production model.

## Runtime and tests

Shared defaults include `SEED = 42`, `RETENTION_N_JOBS = 8`, and `RETENTION_XGB_DEVICE = cpu`. Model-stage settings are documented in the modeling module and recorded in run artifacts.

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

This command checks software behavior; it does not validate a live warehouse or fit the full model. `e_evaluate_and_report/run_test.py` separately evaluates a saved selected model on its final holdout.

## Operational notes

- Shared run state lives in `output/pipeline_state/active_run.json`; `operation.lock` protects state updates. Use separate repository working copies for simultaneous independent profile runs.
- Preserve each run's snapshots, recipes, fold bundles, manifests, and hashes together.
- Use stage status files and manifests to confirm completion before interpreting results.
- Pin run IDs for reproducible comparisons; automatic latest-run resolution can change after another run completes.
