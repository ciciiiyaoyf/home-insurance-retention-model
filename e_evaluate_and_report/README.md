# Saved-Model Reporting

Generate reports from completed pipeline artifacts and score the final holdout in a separate, explicit step.

[Back to the project README](../README.md) · [Modeling and artifact details](../c_modeling/README.md)

## Components

| Script | Responsibility |
| --- | --- |
| `make_reports.py` | Reuses saved models for history and curve reports while leaving the final holdout unscored. |
| `run_test.py` | Scores the saved selected model on its exact final holdout and reports train, validation, and test performance. |
| `report_plots.py` | Shared plotting helpers used by both commands. |

A completed full-population run is required, including its snapshot, preparation metadata, saved outer-validation predictions, selected model, and manifests. Preserve the full run directory when moving artifacts.

## Development reports

Run from the repository root and replace `RUN_ID` with an existing completed run:

```powershell
python e_evaluate_and_report/make_reports.py --run-id "RUN_ID"
python e_evaluate_and_report/make_reports.py --run-id "RUN_ID" --only history
python e_evaluate_and_report/make_reports.py --run-id "RUN_ID" --only curves
```

`--only` accepts `both` (default), `history`, or `curves`. Results go under `output/pipeline_runs/<RUN_ID>/reporting/saved_model_reports_<timestamp>/`.

These commands do not fit or tune models. A development report does not contain actual final-test performance, including when an inherited filename contains `train_validation_test`.

## Final-holdout evaluation

After feature, model-family, and tenure-grouping choices are fixed:

```powershell
python e_evaluate_and_report/run_test.py --run-id "RUN_ID"
```

This command scores all final-holdout rows with the saved selected model and writes results under `output/pipeline_runs/<RUN_ID>/final_test/`. It performs no fitting or deployment. Outputs include split metrics, test predictions, ROC/PR curves, lift/gains, calibration, temporal-stability plots, and a holdout-exposure status record.

Both reporting commands use `--train-sample-rows 100000` and `--batch-rows 100000` by default. To use all eligible training rows for descriptive comparison curves:

```powershell
python e_evaluate_and_report/run_test.py --run-id "RUN_ID" --train-sample-rows 0
```

Training curves describe the final model's in-sample predictions. Validation curves summarize out-of-fold predictions from their respective fold models. Test curves use the final frozen model on the reserved holdout. Repeated reporting of that verified frozen model reuses committed test predictions; test results should not be treated as untouched evidence after they inform a new modeling choice.

All output directories are generated locally and excluded from Git. No original company predictions or performance results are included in this repository.
