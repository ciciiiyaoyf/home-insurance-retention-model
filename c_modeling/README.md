# Modeling and Evaluation

Prepare chronological development folds, select Logistic Regression and XGBoost models, compare them on aligned outer validation windows, and save the selected model for explicit final-test evaluation.

[Back to the project README](../README.md) · [Saved-model reporting](../e_evaluate_and_report/README.md)

## Inputs

This module consumes the run's frozen acquisition outputs:

- `input_snapshot/retention_13_1_population.parquet` and its metadata.
- `model_inputs/feature_contract.csv`.
- `model_inputs/candidate_pool.txt`.
- Shared profile and validation settings from `params.py`.

The target is `RETENTION_13_1_IND`. Retention probability is converted to churn probability as `1 - P_RETENTION` for churn-oriented metrics and lift charts.

## Execution order

| Script | Responsibility |
| --- | --- |
| `c0_one_click.py` | Coordinates modeling, recovery, and completion checks. |
| `c1_prepare_modeling_data.py` | Validates data, defines folds, audits candidates, and writes family-specific feature contracts. |
| `c2_run_logistic_regression.py` | Runs Logistic history selection, grouped floating-forward selection, and regularization tuning. |
| `c3_run_xgboost.py` | Runs XGBoost tuning, feature-count selection, confirmation, and finalist retuning. |
| `c4_evaluate_and_finalize.py` | Evaluates frozen recipes on outer folds, selects a family, and saves fold models and a provisional final refit. |
| `../e_evaluate_and_report/make_reports.py` | Generates additional reports from saved models without opening the final holdout. |
| `../e_evaluate_and_report/run_test.py` | Scores the final holdout using the saved selected model and generates train/validation/test reports. |

The main pipeline runs `c1` through `c4`. Reporting and final-test commands are separate.

```powershell
# Run from the repository root.
python a_business_understanding/a0_one_click.py --tenures 2
```

Use the main launcher for profile selection and recovery. `c0_one_click.py` does not accept these options directly. Current `c4` requires the `FULL` population scope.

## Chronological validation

| Setting | Current default |
| --- | --- |
| Cohort date | `MONTH_1_EVALUATION_DATE` |
| Label-availability date | `MONTH_13_EVALUATION_DATE` |
| Outer folds | `5` |
| Outer validation horizon | `3` months |
| Inner folds | `3` |
| Inner validation horizon | `3` months |
| Minimum outer/inner training horizon | `12` months each |
| Final holdout horizon | `6` months |
| Additional label-availability buffer | `0` days |
| Inner discovery anchor | Outer fold `1` |

Training rows must precede the applicable validation window and have observable Month-13 outcomes by the training cutoff. A zero-day additional buffer still enforces label maturity.

Feature discovery is nested inside the oldest outer training boundary. `INNER_SELECTION_OUTER_FOLD` is fixed at `1`; the former environment override for moving discovery to a later fold is unsupported.

Inspect `preparation/outer_fold_definitions.csv`, `inner_fold_definitions.csv`, and `preparation_info.json` for the actual dates and population identities.

## Feature preparation

- Candidate quality screening uses the oldest inner training partition.
- Feature routing gives Logistic and XGBoost their appropriate representations, including configured binned/unbinned counterparts.
- Encoded columns from one raw feature are handled as a group during selection.
- Learned preprocessing and state mappings are fitted on training partitions and frozen when applied to validation/scoring rows.
- `COHORT_CALENDAR_YEAR` is a mandatory continuous calendar-year control.

`EXPOSURE_CALENDAR_YEAR` is a separate variable. It is QA-only by default and can compete as a discrete challenger predictor:

```powershell
python a_business_understanding/a0_one_click.py --tenures 2 --include-exposure-year-predictor
```

### State grouping

`state_grouping.py` stabilizes state-level retention estimates with empirical Bayes shrinkage, orders states by retention behavior, and forms groups using dynamic programming. The default retention-gap threshold between adjacent learned groups is `0.01`, or one percentage point.

This is retention-based grouping; geographic adjacency is not a constraint. A learned mapping remains fixed when its model scores another population or tenure group.

## Logistic Regression

The active `c2` workflow uses ridge Logistic Regression:

1. Compare recent-history candidates of **36, 48, and 60 months** using inner validation.
2. Hold the selected history and screening `C = 0.01` fixed during grouped sequential floating-forward selection.
3. Add useful feature groups and allow eligible backward removals, using churn AP skill and lift guardrails.
4. Check accepted structures on full development-fold rows.
5. Tune regularization for the selected structure and save the recommended recipe.

The default post-selection regularization grid contains **7 C values**, defined in `logistic_feature_selection.py`:

```text
0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0
```

`logistic_feature_selection.py` is the single source of truth for the default grid. `c2_run_logistic_regression.py` reads that definition for reuse checks; it does not maintain a separate default grid or override it with five values.

History screening and feature selection use the fixed screening value `C = 0.01`. The seven-value grid is used after the feature structure has been selected. An explicit `RETENTION_LR_C_GRID` environment override changes the effective tuning grid and must include the screening value.

Initial selection uses row caps of 100,000 training rows and 50,000 validation rows; these caps do not define the final outer-evaluation population.

## XGBoost

The active workflow combines:

1. Optuna TPE hyperparameter tuning and training-history selection.
2. Grouped gain ranking and cumulative forward-prefix feature search.
3. Local refinement of candidate feature counts.
4. Confirmation across multiple seeds and finalist retuning.
5. A final parsimony decision governed by practical AP-skill/recall loss limits and per-fold deterioration limits.

`RETENTION_XGB_OPTUNA_TRIALS` and `RETENTION_XGB_OPTUNA_BATCH_SIZE` both default to `8` in `c3`. These are tuning-stage budgets, not the total number of fits across ranking, confirmation, and finalist retuning.

Feature count is selected from the evaluation evidence; the pipeline does not guarantee a fixed number of selected predictors.

## Outer evaluation and final refit

`c4` evaluates both frozen model recipes on the same five outer validation windows. Family comparison uses churn AP skill and recall at 10% capacity with paired one-standard-error rules. Logistic receives the parsimony tie-break when both families remain eligible.

The final refit relearns model parameters and preprocessing on the selected eligible, label-mature pre-holdout training population. Features, history policy, tuning choices, seeds, and boosting rounds remain frozen.

The saved bundle is marked `PROVISIONAL_PREHOLDOUT_REFIT` with `deployment_eligible = false`. The final holdout is still unscored at pipeline completion.

## Development artifacts

Paths below are relative to `output/pipeline_runs/<RUN_ID>/`.

| Location | Key artifacts |
| --- | --- |
| `preparation/` | `preparation_info.json`, fold definitions, `candidate_audit.csv`, quality-screen reports, and `feature_routing_audit.csv`. |
| `preparation/` | `quality_screened_logistic_candidate_pool.txt`, `quality_screened_xgboost_candidate_pool.txt`, and family-specific feature-types JSON files. |
| `logistic/` | `run_status.json`, `run_info.json`, `recommended_features.txt`, `recommended_model.json`, and `history_selection_decision.json`. |
| `xgboost/` | `run_info.json`, `selected_features.txt`, `selected_features.json`, and tuning/confirmation artifacts. |
| `<family>/diagnostics/` | Diagnostics manifests, one-way summaries, calendar-year analyses, and retention trends; XGBoost also produces SHAP outputs. |
| `outer_evaluation/` | `outer_oof_predictions.parquet`, `outer_fold_metrics.csv`, `outer_metric_summary.csv`, `outer_tenure_metrics.csv`, and `history_application_audit.csv`. |
| `outer_evaluation/` | `model_comparison.csv`, `family_selection.json`, and `outer_fold_bundle_manifest.json`. |
| `outer_evaluation/fold_models/` | Saved outer-fold models used by cross-tenure evaluation. |
| `deployment/` | Selected provisional model and `final_model_manifest.json`. |
| `deployment/diagnostics/` | Descriptive final-refit retention trends. |

For Logistic-versus-XGBoost retention over the pre-holdout timeline, inspect `outer_evaluation/plots/model_retention_trend_full_history.png` and its CSV/manifest. Full-history descriptive plots may include in-sample periods; use OOF metrics for outer-validation performance.

The XGBoost SHAP scatter routine excludes the mandatory `COHORT_CALENDAR_YEAR` control. Diagnostics manifests provide the generated artifact inventory.

## Additional reports

```powershell
python e_evaluate_and_report/make_reports.py --run-id "RUN_ID"
python e_evaluate_and_report/make_reports.py --run-id "RUN_ID" --only history
python e_evaluate_and_report/make_reports.py --run-id "RUN_ID" --only curves
```

`--only` accepts `both` (default), `history`, or `curves`. Outputs are written to `reporting/saved_model_reports_<timestamp>/` within the run.

This reporter reuses saved models without fitting or tuning. It leaves the final holdout unscored, so it cannot produce an actual final-test curve even if a filename contains `train_validation_test`.

## Final-test evaluation

After model and tenure-grouping decisions are frozen:

```powershell
python e_evaluate_and_report/run_test.py --run-id "RUN_ID"
```

The command scores the saved selected family on its exact six-month final holdout. It performs no fitting, tuning, or deployment. Final-test rows are always scored in full.

Both reporting scripts default to `--train-sample-rows 100000` and `--batch-rows 100000`. Use `--train-sample-rows 0` to include all eligible training rows in comparison curves:

```powershell
python e_evaluate_and_report/run_test.py --run-id "RUN_ID" --train-sample-rows 0
```

| Artifact under `final_test/` | Contents |
| --- | --- |
| `train_validation_test_roc_pr.png` | Train, validation, and final-test ROC/PR curves. |
| `split_metrics.csv` | Split/fold metrics, plotted row counts, full training-population row count, and churn prevalence. |
| `curve_points.csv` | Comparison curve coordinates. |
| `test_metrics.csv` | Final frozen-model test metrics. |
| `train_predictions.parquet` | Training comparison predictions, sampled unless disabled. |
| `validation_predictions.parquet` | Saved outer-validation predictions for the selected family. |
| `test_predictions.parquet` | Full final-test predictions. |
| `plots/` | Test-only ROC/PR, lift/gains, calibration, and temporal-stability plots. |
| `test_status.json` | Model identity, scoring/reporting status, and holdout-exposure record. |

### Interpreting the three splits

- Training and test curves use the final frozen model. The training curve is descriptive in-sample performance.
- The validation curve is a macro-average of outer-fold curves from their respective fold models.
- Final-refit training may include older cohorts that previously served as outer validation. Each validation prediction remains out-of-fold for its own fold model.
- The final holdout remains excluded from training.
- Counts refer to policy-term rows. Distinct-contract counts require deduplication of the prediction records.
- Retention rate equals `1 - churn prevalence`. Sampled training prevalence and an unweighted fold-average validation prevalence are not necessarily full-population rates.

Repeating final-test reporting for the same verified frozen model reuses committed test predictions. Using test results for further model choices turns that holdout into reused development evidence.

## Recovery

Use the root launcher's resume, preparation-rebuild, canonical XGBoost-restart, or separate reconfirmation commands described in the [project README](../README.md).

Preserve snapshots, preparation contracts, selected recipes, model bundles, and manifests together. Changes to preparation or selection policy may require an explicit rebuild rather than checkpoint reuse.
