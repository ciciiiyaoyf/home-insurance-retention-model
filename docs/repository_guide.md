# Repository Guide

This work sample shows how a policy-retention question becomes a reproducible modeling workflow. A reviewer can follow the complete path from Snowflake SQL to saved model recipes, temporal validation, and retention-risk reports.

## Business question and modeling unit

The model estimates whether an eligible homeowner policy will remain active at Month 13 using information available at Month 1. A retention probability supports analysis of which policies are more likely to leave and how retention differs by customer tenure. This annual prediction can support subsequent policy-life and customer-value analysis; the current target is not a full survival model.

The unit is **one policy contract and anchor term**, so one policy may contribute multiple annual observations. `RETENTION_13_1_IND = 1` means retained; `0` means not retained. Churn-oriented evaluation uses `1 - P_RETENTION`.

## Suggested reading order

| Step | Start here | What to inspect |
| --- | --- | --- |
| 1. Data contract | [Snowflake schema guide](snowflake_schema.md) and [source column dictionary](schema/source_columns.csv) | Source record structure, observation dates, key fields, and a small artificial example. |
| 2. SQL construction | [Build SQL](../b_data_acquisition/sql/build_cy_ho_ret_13_1_all_tenure.sql) | How source exposure intervals become eligible policy-term rows with Month-1 features and Month-13 outcomes. |
| 3. Acquisition | [Data-acquisition README](../b_data_acquisition/README.md) | Tenure filters, deterministic contract sampling, population checks, and frozen model inputs. |
| 4. Modeling | [Modeling README](../c_modeling/README.md) | Model-specific feature routing, chronological folds, selection, tuning, and family comparison. |
| 5. Transfer and pooling | [Cross-tenure README](../d_cross_tenure/README.md) | Whether a frozen model transfers to another tenure population and how pooled models are assessed. |
| 6. Reporting | [Reporting README](../e_evaluate_and_report/README.md) | Saved-model diagnostic reports and the separate final-holdout scoring step. |
| 7. Operations | [Root README](../README.md#recovery-and-reruns) | Run identities, artifact manifests, recovery, and repeatable comparisons. |

The main entry point is [`a_business_understanding/a0_one_click.py`](../a_business_understanding/a0_one_click.py). Shared settings live in [`params.py`](../params.py). The controller executes acquisition and modeling; cross-tenure analysis and final reporting are explicit later commands.

## Public Snowflake contract

| Object | Role |
| --- | --- |
| `INSURANCE_ANALYTICS.RETENTION.POLICY_EXPOSURE_HISTORY` | Source exposure history with structural identifiers, term dates, exposure intervals, and predictor fields. |
| `INSURANCE_ANALYTICS.RETENTION.POLICY_RETENTION_13_1` | Derived all-tenure modeling table with one eligible contract/anchor-term observation. |
| `input_snapshot/retention_13_1_population.parquet` | A run's locally saved extraction from Snowflake, created during execution. |

The schema and example rows are supplied so readers can understand the warehouse input without access to the original data. Database/schema names and connection settings are configurable. The pipeline still queries Snowflake; the CSV fixture explains the source shape and can be loaded into the example warehouse table.

The source fixture contains two artificial contracts. It is deliberately small enough to read and is not a training dataset. The repository contains neither actual company records nor company model-performance results. Full execution needs appropriately populated Snowflake tables satisfying the documented date, maturity, tenure, and validation contracts.

## Methodological details

- **Observation timing:** predictors are anchored to Month 1; training labels must already be observable at each training cutoff.
- **Temporal evaluation:** five outer chronological folds assess frozen recipes. Three inner folds support discovery within the oldest outer training boundary. A six-month holdout is scored separately.
- **Logistic Regression:** ridge regression with grouped floating-forward selection, eligible backward removals, history-window selection, and a seven-value regularization grid.
- **XGBoost:** Optuna TPE tuning, history selection, grouped gain ranking, forward-prefix feature-count search, refinement, and confirmation across seeds.
- **State grouping:** empirical Bayes stabilization followed by dynamic-programming grouping of states ordered by retention behavior. Geographic adjacency is not enforced.
- **Cross-tenure scoring:** a saved source-tenure model, including preprocessing and state mapping, scores matched target-tenure validation rows. Standard pairwise analysis does not refit that model on target data.
- **Pooled assessment:** the current pooled-versus-native script uses final-holdout rows and is therefore a final assessment after grouping choices are fixed. It does not supply an outer-fold pooling-selection workflow.

The current defaults and exact selection rules are documented in the module READMEs and defined in code. Run manifests record the effective choices. Historical project metrics are not used as outputs of this public version.

## Optional saved-selection analysis

The root-level helpers support an additional comparison without restarting acquisition or feature discovery:

| Component | Role |
| --- | --- |
| `reselect_saved_xgboost.py` | Re-evaluates saved, retuned finalist evidence under a supplied selection policy and exports a challenger recipe. |
| `evaluate_reselected_xgboost.py` | Fits that saved challenger recipe on the existing outer folds and compares aligned predictions with the baseline. `--check-only` validates inputs without fitting. |
| `xgboost_outer_comparison.py` | Produces post-hoc reports from aligned baseline/challenger predictions. |

These tools do not promote a challenger into the canonical selected model or open the final holdout. Some supporting helpers were reconstructed from the calling code's artifact contract; see the reconstruction notes before interpreting them as recovered original source.

## What is preserved and what changed

The repository preserves the recovered project modules, model-selection workflows, temporal design, cross-tenure analysis, reporting, and run-recovery structure. The public adaptation generalizes Snowflake object names and structural fields, uses account-provided connection settings, and restores dependencies needed by the recovered source.

The [reconstruction notes](reconstruction_notes.md) identify source-recovery repairs and reconstructed assets. These distinctions matter when reviewing provenance: a restored helper or public connector is an adaptation, while the modeling modules retain the recovered implementation.

Software checks and query generation can be run without the original warehouse. A live Snowflake build and full data-dependent training are separate verification steps; passing unit tests alone does not establish end-to-end warehouse execution or predictive performance.
