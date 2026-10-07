# Cross-Tenure Evaluation

Compare retention models across customer-tenure groups, generate native baseline lift reports, and evaluate pooled versus separate tenure models to support consolidation analysis.

[Back to the project README](../README.md)

## Purpose and direction

For each matched outer fold, a frozen source-tenure model scores the target tenure's validation rows. Its performance is compared with the target tenure's own saved predictions on those same rows.

**Source 2 to target 3 means the tenure-2 model scores tenure-3 data.**

The default path transfers the fitted model, preprocessing, feature handling, and learned state mapping. It does not retrain the source model on target data or rerun feature selection for the target population.

The default purpose is `TENURE_GROUPING_MODEL_STRUCTURE`, with the `outer-oof` evaluation window. OOF means out-of-fold: predictions for validation observations excluded from their corresponding model's training data.

## Scripts

| Script | Responsibility |
| --- | --- |
| `d0_cross_tenure_one_click.py` | Resolves the project/components and launches pair comparisons or native baseline reporting. |
| `cross_tenure_evaluation.py` | Validates saved artifacts, aligns validation populations, scores source models, and writes metrics and charts. |
| `pooled_vs_native_evaluation.py` | Compares pooled and tenure-specific models on matched rows; also reports pooled baseline lift. Run directly. Its current implementation uses the final holdout. |

The launcher recognizes `d_cross_tenure/`, the legacy `d_evaluation/` location, and placement at the repository root. `--project-root` overrides automatic root detection.

## Prerequisites

Run `c4_evaluate_and_finalize.py` through the main pipeline for each required tenure profile first. Standard comparisons require completed full-population runs with compatible outer windows and verified artifacts.

Relevant run artifacts include:

- `outer_evaluation/outer_oof_predictions.parquet`.
- `outer_evaluation/outer_fold_bundle_manifest.json`.
- `outer_evaluation/fold_models/`.
- The corresponding snapshot, preparation contracts, and model/run metadata.

Logistic/XGBoost selection results alone are insufficient if outer evaluation has not completed. Preserve the full run directories.

## Pairwise comparisons

Run commands from the repository root.

```powershell
# Tenure-1 models score tenure-2 validation data.
python d_cross_tenure/d0_cross_tenure_one_click.py --project-root . --source-tenure 1 --target-tenure 2

# Tenure-2 models score tenure-3 validation data.
python d_cross_tenure/d0_cross_tenure_one_click.py --project-root . --source-tenure 2 --target-tenure 3

# Pin the source and target runs for reproducibility.
python d_cross_tenure/d0_cross_tenure_one_click.py --project-root . --source-tenure 1 --target-tenure 2 --source-run "output/pipeline_runs/SOURCE_RUN_ID" --target-run "output/pipeline_runs/TARGET_RUN_ID"
```

Replace the run placeholders with actual completed runs. Without explicit run choices, eligible runs are resolved from completion/profile metadata; later training can change which run is selected.

`--target-run` is forwarded by the launcher to the evaluator. The documented commands use the default outer-OOF path and leave the final holdout sealed.

## Native baseline lift reports

Native baselines describe each tenure's own Logistic and XGBoost outer-validation predictions.

```powershell
# Report all four renewal-tenure baselines.
python d_cross_tenure/d0_cross_tenure_one_click.py --project-root . --baseline-tenures 1 2 3 "4+"

# Pin individual baseline runs.
python d_cross_tenure/d0_cross_tenure_one_click.py --project-root . --baseline-tenures 1 2 --baseline-run "1=RUN_ID_1" --baseline-run "2=RUN_ID_2"
```

Baseline reporting reads saved native outer-OOF predictions. It performs no fitting, new population scoring, or final-holdout access.

Use stored labels `0`, `1`, `2`, `3`, or `4+`. Do not combine `--baseline-tenures` with source/target options or `--materialize-source-bundle`.

## Pooled models and tenure consolidation

`pooled_vs_native_evaluation.py` is a retained component of this module. It supports the decision about whether tenure groups can share a model. Run it directly; the current `d0` launcher does not invoke it.

Evaluate a candidate set of tenure groups using each group's own model and a model trained on the combined population, for both Logistic Regression and XGBoost. The `--tenures` argument selects the candidate groups.

**Tenures 2 and 3 are used below only to illustrate the workflow. They are not a recommended grouping or an observed consolidation result.** Replace these tenure selections and their matching run IDs for the candidate set being assessed.

**Pooled 2-3 means training on tenures 2 and 3 together. Source 2 to target 3 means applying a tenure-2 model to tenure-3 data.**

| Comparison | Models evaluated | Evaluation population |
| --- | --- | --- |
| Tenure 2 | Native tenure-2 model versus pooled 2-3 model | Identical tenure-2 rows |
| Tenure 3 | Native tenure-3 model versus pooled 2-3 model | Identical tenure-3 rows |
| Combined 2-3 baseline | Pooled 2-3 model | Combined tenure-2 and tenure-3 population |

The combined baseline describes overall separation. The two within-tenure comparisons show whether pooling preserves performance for each group. A strong combined lift spread can conceal weaker separation within one tenure.

### Prepare the required model profiles

Prepare one completed full-population run for each individual tenure and one for the combined candidate set. For the illustrative set of tenures 2 and 3, the required profiles are `2`, `3`, and `2,3`:

```powershell
python a_business_understanding/a0_one_click.py --tenures 2
python a_business_understanding/a0_one_click.py --tenures 3
python a_business_understanding/a0_one_click.py --tenures 2 3
```

Preserve each run's snapshot, preparation metadata, and frozen Logistic/XGBoost bundles. Pooled and native runs must have compatible evaluation calendars and verifiable policy identities. The comparison script scores saved models; it does not train the pooled model.

### Development evidence and the current holdout limitation

Choose tenure groupings using matched outer-validation evidence. For the same illustrative set, obtain native outer-validation baselines with:

```powershell
python d_cross_tenure/d0_cross_tenure_one_click.py --project-root . --baseline-tenures 2 3 --baseline-run "2=RUN_ID_2" --baseline-run "3=RUN_ID_3"
```

The current `pooled_vs_native_evaluation.py` scores the **sealed final holdout** in both comparison and pooled-baseline modes. It has no outer-OOF mode. The commands below therefore provide a final assessment after grouping choices are fixed. A matched outer-fold pooled-versus-native comparison is still needed for development-stage consolidation selection.

Do not combine the native outer-OOF spreads above and a pooled final-holdout spread into one directly comparable three-model scorecard. Their evaluation periods, populations, and aggregation methods differ. The paired command below compares pooled and native models on identical holdout rows within each tenure.

### Compare the candidate pooled model with separate models

Replace the example tenure selections and all run placeholders with the matching completed runs. Supply one `--native-run TENURE=PATH` for each selected tenure. This command opens the final holdout:

```powershell
python d_cross_tenure/pooled_vs_native_evaluation.py --project-root . --tenures 2 3 --pooled-run "output/pipeline_runs/RUN_ID_23" --native-run "2=output/pipeline_runs/RUN_ID_2" --native-run "3=output/pipeline_runs/RUN_ID_3" --lift-bins "3,4,5,10,20"
```

The script reports each selected tenure separately for both model families, including lift-spread differences and paired policy-cluster bootstrap uncertainty. The default is 500 bootstrap replicates.

For the pooled model's overall lift spread across the combined candidate population, run the separate pooled-baseline mode. The example below also opens the final holdout:

```powershell
python d_cross_tenure/pooled_vs_native_evaluation.py --project-root . --tenures 2 3 --pooled-run "output/pipeline_runs/RUN_ID_23" --pooled-baseline-only --lift-bins "3,4,5,10,20"
```

`--pooled-baseline-only` reports the combined population and does not load the separate native models. It supplements the paired comparisons.

### Pooled comparison outputs

Paired comparisons are written under `output/pooled_vs_native/tenures_<groups>_<timestamp>/`, where `<groups>` reflects the selected tenure labels.

| Artifact | Purpose |
| --- | --- |
| `lift_spread_summary.csv` | Native and pooled lift spreads by tenure, model family, and bin count. |
| `lift_spread_comparison.csv` | Native/pooled values, pooled-minus-native differences, and bootstrap intervals. |
| `lift_by_equal_count_group.csv` | Values underlying the lift charts. |
| `model_metrics_by_tenure.csv` | Ranking, probability, and calibration metrics for each tenure/model. |
| `paired_metric_comparison.csv` | Paired differences and uncertainty for the performance metrics. |
