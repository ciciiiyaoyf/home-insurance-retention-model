#!/usr/bin/env python
"""Plot tenure response from one saved pooled/all-tenure C4 run; never fit.


Reads the C4 receipt and recorded OOF predictions. The one-way plot uses every
OOF row. Partial dependence uses the SAME deterministic sample of validation
rows for both families (10,000 per fold by default; 0 means all). Each row is
scored by its own frozen outer-fold model. Only the requested tenure input is
varied; all other input values and all learned transforms stay fixed.


No new outcome columns are read from the raw snapshot. No final-test bundle or
final-test predictions are loaded. All outputs go to a NEW report directory.


Requirements: numpy, pandas, plotly, joblib, scipy, scikit-learn, xgboost, and a
Parquet reader (normally pyarrow) for Parquet inputs. Use the ORIGINAL model
scoring environment: bundle version/hash checks are not bypassed.


Method references:
https://scikit-learn.org/stable/modules/partial_dependence.html
https://plotly.com/python/interactive-html-export/
"""
from __future__ import annotations


import argparse
import hashlib
import html
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


import numpy as np
import pandas as pd


FAMILIES = ("logistic", "xgboost")
OUTER_CONTRACT = "retention_outer_evaluation"
BUNDLE_CONTRACT = "retention_frozen_source_model_bundle"




def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value




def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()




def normalize(value: Any) -> str:
    text = str(value).strip()
    return text.split(".")[0] if re.fullmatch(r"[+-]?\d+\.0+", text) else text




def tenure_order(values: Sequence[Any]) -> list[str]:
    def key(value: str) -> tuple[float, str]:
        match = re.match(r"^([0-9]+)", value)
        return (float(match[1]) if match else float("inf"), value)
    return sorted(set(map(normalize, values)), key=key)




def resolve_reference(reference: Any, owner: Path, root: Path, run: Path) -> Path:
    """Resolve recorded identities, including moving a project between servers."""
    raw = reference
    if isinstance(reference, Mapping):
        raw = next((reference.get(k) for k in
                    ("manifest_path", "path", "relative_path", "manifest_relative_path")
                    if reference.get(k)), None)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Missing artifact path in {owner}: {reference!r}")
    path = Path(raw.replace("\\", "/")).expanduser()
    candidates = [path] if path.is_absolute() else [owner.parent / path, root / path]
    parts = raw.replace("\\", "/").split("/")
    if run.name in parts:
        index = len(parts) - 1 - parts[::-1].index(run.name)
        candidates.append(run.joinpath(*parts[index + 1:]))
    # The frozen registry may be outside the run directory but inside output/.
    if "output" in parts:
        index = len(parts) - 1 - parts[::-1].index("output")
        candidates.append(root.joinpath(*parts[index:]))
    found = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    if len(found) != 1:
        raise FileNotFoundError(f"Artifact must resolve uniquely: {raw!r}; matches={found}")
    return found[0]




def verified_reference(reference: Any, owner: Path, root: Path, run: Path) -> Path:
    if not isinstance(reference, Mapping):
        raise ValueError(f"Expected a path-and-hash artifact reference in {owner}")
    path = resolve_reference(reference, owner, root, run)
    expected = str(reference.get("sha256", reference.get("manifest_sha256", ""))).lower()
    if not expected or sha256(path) != expected:
        raise RuntimeError(f"Recorded SHA-256 missing or mismatched: {path}")
    return path




def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"row_id": "string"}, low_memory=False)
    raise ValueError(f"Expected CSV or Parquet: {path}")




def canonical_identity(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in columns:
        values = frame[column]
        if values.isna().any():
            raise ValueError(f"Missing row identity in {column}")
        if "DATE" in column.upper() or column.upper().endswith("_DT") or pd.api.types.is_datetime64_any_dtype(values):
            parsed = pd.to_datetime(values, errors="raise")
            output[column] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        else:
            output[column] = values.astype("string")
    return output




def identity_keys(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    return pd.util.hash_pandas_object(canonical_identity(frame, columns), index=False).astype("uint64")




def validate_oof(frame: pd.DataFrame, manifest: dict[str, Any], ids: list[str]) -> pd.DataFrame:
    required = {"row_id", "fold", "cohort_date", "tenure_group", "model_family", "y_retention", "p_retention", *ids}
    missing = required.difference(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"OOF predictions empty or missing columns: {sorted(missing)}")
    data = frame.copy()
    if data[list(required)].isna().any().any():
        raise ValueError("OOF identity, tenure, outcome, and probability columns must be nonmissing.")
    data["row_id"] = data["row_id"].astype("string")
    data["model_family"] = data["model_family"].astype(str).str.lower().str.strip()
    data["tenure_group"] = data["tenure_group"].map(normalize)
    if set(data["model_family"]) != set(FAMILIES):
        raise ValueError("Expected both Logistic and XGBoost, and no other model families.")
    for column in ("y_retention", "p_retention", "fold"):
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column]).all():
            raise ValueError(f"Nonfinite OOF values: {column}")
    if not data["y_retention"].isin([0, 1]).all():
        raise ValueError("Retention outcomes must be binary, before integer conversion.")
    if not data["p_retention"].between(0, 1).all():
        raise ValueError("OOF retention probabilities must lie in [0, 1].")
    if not data["fold"].eq(np.floor(data["fold"])).all():
        raise ValueError("Fold IDs must be integers.")
    data["fold"] = data["fold"].astype(int)
    data["y_retention"] = data["y_retention"].astype(int)
    data["cohort_date"] = pd.to_datetime(data["cohort_date"], errors="raise")
    if data.duplicated(["model_family", "row_id"]).any():
        raise ValueError("Duplicate OOF model-family/row IDs; cannot count these as distinct rows.")
    folds = {int(f["fold"]): f for f in manifest["folds"]}
    if len(folds) != len(manifest["folds"]) or set(data["fold"]) != set(folds):
        raise ValueError("OOF fold IDs disagree with the C4 receipt.")
    holdout_start = pd.Timestamp(manifest["excluded_holdout"]["cohort_start"])
    for number, fold in folds.items():
        start, end = pd.Timestamp(fold["validation_start"]), pd.Timestamp(fold["validation_end"])
        if start > end or end >= holdout_start or fold.get("final_holdout_rows_scored") != 0:
            raise ValueError(f"Invalid or unsealed outer validation fold {number}")
        if not data.loc[data["fold"].eq(number), "cohort_date"].between(start, end).all():
            raise ValueError(f"OOF dates leave their recorded validation window in fold {number}")
    common = ["row_id", "fold", "cohort_date", "tenure_group", "y_retention"]
    left = data.loc[data.model_family.eq("logistic")].sort_values("row_id").reset_index(drop=True)
    right = data.loc[data.model_family.eq("xgboost")].sort_values("row_id").reset_index(drop=True)
    if not left[common].equals(right[common]) or not canonical_identity(left, ids).equals(canonical_identity(right, ids)):
        raise ValueError("Logistic and XGBoost are not evaluated on identical rows/outcomes/tenures.")
    if canonical_identity(left, ids).duplicated().any():
        raise ValueError("Duplicate governed row identities in outer OOF data.")
    return data




def summarize_one_way(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = [("All outer folds", oof), *[(f"Fold {int(k)}", v) for k, v in oof.groupby("fold", sort=True)]]
    for scope, part in scopes:
        for (family, tenure), group in part.groupby(["model_family", "tenure_group"], observed=True):
            n = len(group)
            rows.append({"scope": scope, "model_family": family, "tenure_group": tenure,
                         "records": n, "volume_share": n / int(part.model_family.eq(family).sum()),
                         "actual_retention": float(group.y_retention.mean()),
                         "predicted_retention": float(group.p_retention.mean()),
                         "actual_churn": float(1 - group.y_retention.mean()),
                         "predicted_churn": float(1 - group.p_retention.mean())})
    return pd.DataFrame(rows)




def choose_sample(oof: pd.DataFrame, ids: list[str], cap: int, seed: int) -> pd.DataFrame:
    base = oof.loc[oof.model_family.eq("logistic"), ["row_id", "fold", "cohort_date", "tenure_group", *ids]].copy()
    base = base.loc[:, ~base.columns.duplicated()]
    parts = []
    for number, group in base.groupby("fold", sort=True):
        rank = pd.util.hash_pandas_object(pd.DataFrame({"id": group.row_id, "seed": str(seed)}), index=False)
        picked = group.assign(_rank=rank).sort_values(["_rank", "row_id"])
        if cap:
            picked = picked.iloc[:cap]
        parts.append(picked.drop(columns="_rank").assign(full_fold_rows=len(group)))
    sample = pd.concat(parts, ignore_index=True)
    sample["_join_key"] = identity_keys(sample, ids)
    if sample._join_key.duplicated().any():
        raise ValueError("Nonunique sampled row identities or an identity-hash collision.")
    return sample




def checked_bundle(path: Path, family: str, fold: int, outer: dict[str, Any]) -> dict[str, Any]:
    manifest = load_json(path)
    meta = manifest.get("source_metadata", manifest)
    if manifest.get("contract") != BUNDLE_CONTRACT or manifest.get("model_family") != family:
        raise ValueError(f"Unexpected saved model contract/family: {path}")
    if meta.get("bundle_role", manifest.get("bundle_role")) != "OUTER_FOLD_EVALUATION":
        raise ValueError(f"Refusing non-outer-fold model: {path}")
    if int(meta.get("outer_fold", manifest.get("outer_fold", -1))) != fold:
        raise ValueError(f"Wrong outer fold in bundle: {path}")
    if str(meta.get("source_run_id", meta.get("run_id", ""))) != str(outer["run_id"]):
        raise ValueError(f"Bundle belongs to a different run: {path}")
    if str(meta.get("source_profile_hash", meta.get("profile_hash", ""))) != str(outer["profile_hash"]):
        raise ValueError(f"Bundle profile differs from the run: {path}")
    if manifest.get("target_column") != outer["target_column"]:
        raise ValueError(f"Bundle and outer receipt disagree on the outcome: {path}")
    outer_input = outer.get("source_data_identity", outer.get("input_snapshot", {}))
    bundle_input = meta.get("source_data_identity", manifest.get("source_data_identity", {}))
    if not bundle_input.get("sha256") or bundle_input.get("sha256") != outer_input.get("sha256"):
        raise ValueError(f"Bundle input snapshot differs from the C4 receipt: {path}")
    fold_definition = next(f for f in outer["folds"] if int(f["fold"]) == fold)
    excluded = manifest.get("excluded_holdout", meta.get("excluded_holdout", {}))
    if (not excluded.get("cohort_start") or not excluded.get("cohort_end")
            or pd.Timestamp(excluded["cohort_start"]) != pd.Timestamp(fold_definition["validation_start"])
            or pd.Timestamp(excluded["cohort_end"]) != pd.Timestamp(fold_definition["validation_end"])):
        raise ValueError(f"Bundle validation window differs from its C4 fold: {path}")
    if outer["target_column"] in manifest["features"]:
        raise ValueError("Outcome is illegally included as a model input.")
    inventory = manifest.get("artifact_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError(f"No bundle artifact inventory: {path}")
    for item in inventory:
        artifact = (path.parent / str(item["relative_path"]).replace("\\", "/")).resolve()
        if path.parent.resolve() not in artifact.parents or not artifact.is_file():
            raise ValueError(f"Invalid/missing bundle inventory path: {artifact}")
        if sha256(artifact) != str(item["sha256"]).lower():
            raise RuntimeError(f"Frozen bundle artifact hash mismatch: {artifact}")
    return manifest




class FrozenScorer:
    """Load each estimator and its packaged transforms once; predict only."""
    def __init__(self, path: Path, manifest: dict[str, Any]):
        import joblib
        candidates = []
        sources = manifest.get("runtime_sources", {})
        for item in (sources.values() if isinstance(sources, Mapping) else sources):
            raw = item.get("relative_path", item.get("path")) if isinstance(item, Mapping) else item
            p = (path.parent / str(raw).replace("\\", "/")).resolve()
            if p.suffix == ".py" and p.is_file():
                source = p.read_text(encoding="utf-8-sig")
                if "def score_logistic_bundle(" in source and "def score_xgboost_bundle(" in source:
                    candidates.append(p)
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) != 1:
            raise ValueError(f"Cannot identify exactly one packaged scoring runtime: {path}")
        runtime_path = candidates[0]
        inventory_paths = {(path.parent / x["relative_path"].replace("\\", "/")).resolve() for x in manifest["artifact_inventory"]}
        if runtime_path not in inventory_paths:
            raise ValueError("Packaged scoring runtime is absent from the verified inventory.")
        name = "_tenure_plot_runtime_" + sha256(runtime_path)[:20]
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, runtime_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load frozen runtime: {runtime_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        self.runtime = sys.modules[name]
        loaded, self.preprocessor = self.runtime.load_bundle(path.parent, verify_hashes=True)
        if loaded != manifest:
            raise RuntimeError("Loaded bundle differs from the manifest validated by the report.")
        self.manifest, self.family = loaded, loaded["model_family"]
        self.features = list(loaded["features"])
        self.models = []
        if self.family == "logistic":
            self.models.append(joblib.load(path.parent / loaded["model_file"]))
        else:
            import xgboost as xgb
            for relative in loaded["model_files"]:
                model = xgb.XGBClassifier()
                model.load_model(str(path.parent / relative))
                self.models.append(model)
        if not self.models:
            raise ValueError("No saved models found.")
        self.target = str(loaded["prediction_target"]).lower()
        if ("churn" in self.target) == ("retention" in self.target):
            raise ValueError("Prediction target must unambiguously identify retention or churn.")


    def raw_columns(self) -> set[str]:
        contracts = self.preprocessor.get("derived_feature_contracts", {})
        virtual = {d["derived_feature"]: d["source_feature"] for d in contracts.values()
                   if isinstance(d, Mapping) and "derived_feature" in d and "source_feature" in d}
        return {virtual.get(feature, feature) for feature in self.features}


    def retention(self, frame: pd.DataFrame, batch_size: int) -> np.ndarray:
        parts = []
        for start in range(0, len(frame), batch_size):
            matrix = self.runtime.transform_frame(self.preprocessor, frame.iloc[start:start + batch_size])
            probabilities = []
            for model in self.models:
                classes = list(model.classes_)
                if 1 not in classes:
                    raise ValueError("Saved estimator has no class 1.")
                values = np.asarray(model.predict_proba(matrix)[:, classes.index(1)], dtype=float)
                if values.shape != (matrix.shape[0],) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                    raise ValueError("Saved estimator returned invalid probabilities; refusing to clip errors away.")
                probabilities.append(values)
            p = np.clip(np.mean(probabilities, axis=0), 1e-7, 1 - 1e-7)
            parts.append(1 - p if "churn" in self.target else p)
        result = np.concatenate(parts)
        if len(result) != len(frame) or not np.isfinite(result).all():
            raise ValueError("Saved model produced invalid predictions.")
        return result


    def category_values(self, feature: str, levels: list[str]) -> dict[str, Any]:
        if feature not in self.features:
            if feature in self.raw_columns():
                raise ValueError("Tenure is the source of a derived input; use a dedicated joint-feature diagnostic.")
            return {v: v for v in levels}
        if self.preprocessor["feature_types"][feature] != "discrete":
            raise ValueError(f"{feature} is not discrete. This report expects the saved grouped tenure feature, not exact numeric tenure.")
        transformer = self.preprocessor["transformers"][feature]
        encoder = transformer.named_steps.get("onehot")
        if encoder is None or not hasattr(encoder, "categories_"):
            raise ValueError("Cannot verify the saved tenure category encoding.")
        actual = {normalize(v): v for v in encoder.categories_[0] if not pd.isna(v)}
        return {v: actual[v] for v in levels if v in actual}




def load_sample_features(path: Path, columns: list[str], sample: pd.DataFrame,
                         ids: list[str], batch_size: int, target: str) -> pd.DataFrame:
    """Stream feature-only batches and retain only sampled OOF identities."""
    if target in columns:
        raise ValueError("Refusing to read outcome values from the raw snapshot.")
    keys = set(sample._join_key)
    parts = []
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Streaming Parquet input requires pyarrow in the original scoring environment.") from exc
        parquet = pq.ParquetFile(path)
        missing = set(columns).difference(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"Snapshot lacks saved-model source fields: {sorted(missing)}")
        batches = (batch.to_pandas() for batch in parquet.iter_batches(batch_size=batch_size, columns=columns))
    elif path.suffix.lower() == ".csv":
        dtype = {column: "string" for column in ids if "DATE" not in column.upper() and not column.upper().endswith("_DT")}
        batches = pd.read_csv(path, usecols=columns, dtype=dtype, chunksize=batch_size, low_memory=False)
    else:
        raise ValueError(f"Unsupported snapshot type: {path}")
    for index, batch in enumerate(batches, 1):
        batch["_join_key"] = identity_keys(batch, ids)
        selected = batch.loc[batch._join_key.isin(keys)]
        if not selected.empty:
            parts.append(selected.copy())
        if index % 25 == 0:
            print(f"  Read {index} feature-only batches; retained {sum(len(p) for p in parts):,} rows", flush=True)
    if not parts:
        raise ValueError("No sampled OOF rows match the recorded snapshot.")
    found = pd.concat(parts, ignore_index=True)
    if found._join_key.duplicated().any() or set(found._join_key) != keys:
        raise ValueError("Snapshot does not match sampled OOF rows one-to-one.")
    found = found.set_index("_join_key").loc[sample._join_key].reset_index()
    if not canonical_identity(found, ids).reset_index(drop=True).equals(canonical_identity(sample, ids).reset_index(drop=True)):
        raise ValueError("Raw/OOF row identities disagree (or a hash collision occurred).")
    return found




def factor_tables(oof: pd.DataFrame, sample: pd.DataFrame, raw: pd.DataFrame,
                  scorers: dict[tuple[int, str], FrozenScorer], feature: str,
                  levels: list[str], reference: str, batch_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, checks = [], []
    for number, fold in sample.groupby("fold", sort=True):
        data = raw.loc[fold.index].reset_index(drop=True)
        if feature in data and not data[feature].map(normalize).eq(fold.tenure_group.to_numpy()).all():
            raise ValueError(f"Raw tenure feature and OOF tenure groups disagree in fold {number}.")
        for family in FAMILIES:
            scorer = scorers[(int(number), family)]
            original = scorer.retention(data, batch_size)
            saved = oof.loc[oof.model_family.eq(family)].set_index("row_id").loc[fold.row_id, "p_retention"].to_numpy()
            maximum_error = float(np.max(np.abs(original - saved)))
            if not np.allclose(original, saved, rtol=1e-5, atol=1e-5):
                raise RuntimeError(f"{family} fold {number}: frozen scores do not reproduce saved OOF probabilities (max error={maximum_error:g}). No factor report is published.")
            retained = feature in scorer.features
            actual_values = scorer.category_values(feature, levels)
            checks.append({"fold": int(number), "model_family": family, "feature": feature,
                           "feature_retained": retained, "max_oof_probability_error": maximum_error,
                           "sample_rows": len(data), "full_fold_rows": int(fold.full_fold_rows.iloc[0]),
                           "feature_count": len(scorer.features),
                           "other_tenure_named_inputs": ";".join(f for f in scorer.features if "TENURE" in f.upper() and f != feature)})
            print(f"  {family} fold {number}: {len(data):,} rows; {feature} retained={retained}", flush=True)
            for level in levels:
                if level not in actual_values:
                    mean, status = np.nan, "UNSEEN_TRAINING_CATEGORY"
                elif not retained:
                    mean, status = float(original.mean()), "FEATURE_NOT_SELECTED"
                else:
                    changed = data.copy()
                    # Change the RAW categorical field, not one dummy column.
                    changed[feature] = pd.Series([actual_values[level]] * len(changed), index=changed.index)
                    mean, status = float(scorer.retention(changed, batch_size).mean()), "ESTIMATED"
                rows.append({"scope": f"Fold {int(number)}", "fold": int(number), "model_family": family,
                             "tenure_group": level, "adjusted_retention": mean,
                             "sample_rows": len(data), "full_fold_rows": int(fold.full_fold_rows.iloc[0]),
                             "feature_retained": retained, "status": status})
    result = pd.DataFrame(rows)
    pooled = []
    for (family, level), group in result.groupby(["model_family", "tenure_group"], observed=True):
        complete = bool(np.isfinite(group.adjusted_retention).all())
        flags = set(group.feature_retained)
        pooled.append({"scope": "All outer folds", "fold": 0, "model_family": family, "tenure_group": level,
                       "adjusted_retention": float(np.average(group.adjusted_retention, weights=group.full_fold_rows)) if complete else np.nan,
                       "sample_rows": int(group.sample_rows.sum()), "full_fold_rows": int(group.full_fold_rows.sum()),
                       "feature_retained": bool(group.feature_retained.any()),
                       "status": "INCOMPLETE_CATEGORY_SUPPORT" if not complete else
                                 "FEATURE_NOT_SELECTED" if flags == {False} else "ESTIMATED"})
    result = pd.concat([pd.DataFrame(pooled), result], ignore_index=True)
    result["adjusted_churn"] = 1 - result.adjusted_retention
    result["reference_tenure"] = reference
    for _, group in result.groupby(["model_family", "scope"], observed=True):
        reference_row = group.loc[group.tenure_group.eq(reference), "adjusted_retention"]
        base = float(reference_row.iloc[0]) if len(reference_row) == 1 else np.nan
        result.loc[group.index, "retention_difference_pp"] = 100 * (group.adjusted_retention - base)
        result.loc[group.index, "retention_probability_factor"] = group.adjusted_retention / base if base > 0 else np.nan
    return result, pd.DataFrame(checks)




def scopes(frame: pd.DataFrame) -> list[str]:
    values = list(frame.scope.unique())
    return sorted(values, key=lambda x: (0, 0) if x == "All outer folds" else (1, int(x.split()[-1])))




def common_layout(figure: Any, title: str, note: str) -> None:
    figure.update_layout(title={"text": title}, height=750,
                         margin={"l": 80, "r": 50, "t": 165, "b": 185},
                         xaxis={"title": "Tenure group", "type": "category"},
                         yaxis={"title": "Retention probability", "tickformat": ".1%"},
                         legend={"orientation": "h", "y": -0.17, "yanchor": "top"},
                         annotations=[{"text": note, "x": 0, "y": -0.30, "xref": "paper", "yref": "paper",
                                       "showarrow": False, "xanchor": "left", "yanchor": "top", "align": "left", "font": {"size": 11}}])




def one_way_figure(summary: pd.DataFrame, levels: list[str], run_id: str) -> Any:
    import plotly.graph_objects as go
    figure, churn_arrays = go.Figure(), []
    scope_list = scopes(summary)
    for scope in scope_list:
        block = summary.loc[summary.scope.eq(scope)]
        for label, family, field in [("Actual", "logistic", "actual_retention"),
                                     ("Logistic predicted", "logistic", "predicted_retention"),
                                     ("XGBoost predicted", "xgboost", "predicted_retention")]:
            table = block.loc[block.model_family.eq(family)].set_index("tenure_group").reindex(levels)
            y = table[field].tolist()
            churn_arrays.append((1 - table[field]).tolist())
            figure.add_scatter(x=levels, y=y, mode="lines+markers", name=label,
                               visible=scope == scope_list[0], connectgaps=False,
                               customdata=table[["records", "volume_share"]].to_numpy(),
                               hovertemplate="Tenure %{x}<br>Probability %{y:.2%}<br>Policy terms %{customdata[0]:,.0f}<br>Volume share %{customdata[1]:.1%}<extra>%{fullData.name}</extra>")
    retention_arrays = [list(trace.y) for trace in figure.data]
    common_layout(figure, "Actual vs predicted by tenure<br><sup>" + html.escape(run_id) + "</sup>",
                  "All saved outer-OOF rows. Different tenure groups contain different policy mixes.<br>This is a subgroup/calibration plot, not an isolated tenure effect; 4+ means four or more.")
    figure.update_layout(updatemenus=[
        {"type": "dropdown", "x": 0, "y": 1.17, "xanchor": "left", "active": 0,
         "buttons": [{"label": scope, "method": "update", "args": [{"visible": [i // 3 == k for i in range(len(figure.data))]}]}
                     for k, scope in enumerate(scope_list)]},
        {"type": "buttons", "direction": "right", "x": 0.43, "y": 1.17,
         "buttons": [{"label": label, "method": "update", "args": [{"y": arrays}, {"yaxis.title.text": title}]}
                     for label, arrays, title in [("Retention", retention_arrays, "Retention probability"), ("Churn", churn_arrays, "Churn probability")]]}])
    return figure




def factor_figure(table: pd.DataFrame, family: str, feature: str, levels: list[str], reference: str, run_id: str) -> Any:
    import plotly.graph_objects as go
    part = table.loc[table.model_family.eq(family)]
    scope_list = scopes(part)
    figure, blocks = go.Figure(), []
    for scope in scope_list:
        block = part.loc[part.scope.eq(scope)].set_index("tenure_group").reindex(levels)
        blocks.append(block)
        figure.add_scatter(x=levels, y=block.adjusted_retention.tolist(), mode="lines+markers",
                           name=scope, visible=scope == scope_list[0], connectgaps=False,
                           customdata=block[["sample_rows", "status"]].to_numpy(),
                           hovertemplate="Tenure %{x}<br>Value %{y:.4f}<br>Reference rows %{customdata[0]:,.0f}<br>%{customdata[1]}<extra>%{fullData.name}</extra>")
    absent = not part.feature_retained.any()
    status = "NOT SELECTED: changing this field alone cannot change the model score." if absent else "Same reference rows at every tenure level; only the selected tenure field changes."
    common_layout(figure, f"{'XGBoost' if family == 'xgboost' else 'Logistic'} tenure factor: {html.escape(feature)}<br><sup>{html.escape(run_id)}</sup>",
                  status + "<br>Model sensitivity, not a causal aging effect. Correlated inputs can create unrealistic combinations.<br>Overall curve is weighted by full validation-fold size. Fold differences are not confidence intervals.")
    figure.update_layout(showlegend=False)
    modes = [("Retention", "adjusted_retention", "Adjusted retention probability", ".1%", None),
             ("Churn", "adjusted_churn", "Adjusted churn probability", ".1%", None),
             ("Difference (pp)", "retention_difference_pp", f"Retention difference vs tenure {reference} (percentage points)", ".2f", 0),
             ("Factor", "retention_probability_factor", f"Retention probability / tenure {reference} probability", ".3f", 1)]
    buttons = []
    for label, field, title, tick, baseline in modes:
        shapes = [] if baseline is None else [{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "yref": "y", "y0": baseline, "y1": baseline, "line": {"dash": "dash"}}]
        buttons.append({"label": label, "method": "update", "args": [{"y": [block[field].tolist() for block in blocks]},
                       {"yaxis.title.text": title, "yaxis.tickformat": tick, "yaxis.autorange": True, "shapes": shapes}]})
    figure.update_layout(updatemenus=[
        {"type": "dropdown", "x": 0, "y": 1.17, "xanchor": "left", "active": 0,
         "buttons": [{"label": scope, "method": "update", "args": [{"visible": [i == k for i in range(len(scope_list))]}]}
                     for k, scope in enumerate(scope_list)]},
        {"type": "buttons", "direction": "right", "x": 0.40, "y": 1.17, "buttons": buttons}])
    return figure




def publish_figures(out: Path, figures: dict[str, Any], metadata: dict[str, Any]) -> None:
    for name, figure in figures.items():
        figure.write_html(str(out / name), include_plotlyjs=True, full_html=True,
                          config={"responsive": True, "displaylogo": False})
    cards = "".join(f'<li><a href="{html.escape(name)}">{html.escape(name)}</a></li>' for name in figures)
    index = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tenure factor report</title><body style="font-family:Arial,sans-serif;max-width:960px;margin:45px auto;line-height:1.6;padding:0 20px">
<h1>Tenure factor report</h1><p><strong>Run:</strong> {html.escape(metadata['run_id'])}</p>
<p><strong>Feature:</strong> {html.escape(metadata['tenure_feature'])}. No fitting, tuning, or final-holdout scoring.</p>
<ul>{cards}</ul><h2>Read the two views differently</h2>
<p><strong>Actual vs predicted:</strong> averages over the policies that actually belong to each tenure group. The policies and other inputs differ between groups.</p>
<p><strong>Factor curves:</strong> the same reference policies are scored at every tenure value by their saved outer-fold models. Other inputs stay fixed. A field absent from a model produces a flat direct-response curve.</p>
<p>Factor means adjusted retention probability divided by the adjusted probability at reference tenure {html.escape(metadata['reference_tenure'])}; it is <strong>not an odds ratio, premium factor, or lift</strong>. Difference mode is measured in percentage points.</p>
<p>Changing tenure alone may create unrealistic combinations with correlated inputs. These plots explain model response, not what would causally happen as a policy ages. 4+ is a top-coded category, not an exact fourth year.</p>
<p>The one-way plot uses all OOF rows. The factor reference population uses at most {metadata['sample_per_fold']:,} rows per fold (0 means all), selected without outcomes. Overall partial dependence weights folds by their full OOF row counts, not by the capped sample counts. Fold curves are not confidence intervals.</p>
<p>See <a href="tenure_feature_audit.csv">feature audit</a>, <a href="tenure_factor_values.csv">factor values</a>, and <a href="tenure_one_way_values.csv">one-way values</a> for exact numbers and support flags.</p></body></html>'''
    (out / "index.html").write_text(index, encoding="utf-8")




def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", "--run", dest="run_id", required=True, help="One saved all-tenure/pooled run ID or run directory")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--outer-manifest", type=Path)
    parser.add_argument("--tenure-feature", default="TENURE_YEARS_GRP")
    parser.add_argument("--reference-tenure", help="Default: lowest tenure in the saved profile")
    parser.add_argument("--sample-per-fold", type=int, default=10000, help="PDP reference rows per fold; 0 uses all. Never samples the one-way plot")
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, help="New, nonexisting output directory; never overwrites a report")
    args = parser.parse_args(argv)
    if args.sample_per_fold < 0 or args.batch_size <= 0:
        parser.error("sample-per-fold must be >=0; batch-size must be >0")
    root = args.project_root or (Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "d_cross_tenure" else Path.cwd())
    root = root.expanduser().resolve()
    value = Path(args.run_id).expanduser()
    candidates = [value] if value.is_absolute() else [root / "output" / "pipeline_runs" / value, root / value]
    found = list(dict.fromkeys(p.resolve() for p in candidates if p.is_dir()))
    if len(found) != 1:
        raise FileNotFoundError(f"Run must resolve uniquely: {args.run_id}; matches={found}")
    run = found[0]
    manifest_path = args.outer_manifest or run / "outer_evaluation" / "outer_fold_bundle_manifest.json"
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Saved C4 outer manifest is required: {manifest_path}. This utility will not train or rebuild missing models.")
    outer = load_json(manifest_path)
    if outer.get("contract") != OUTER_CONTRACT or outer.get("contract_revision") != 1:
        raise ValueError("Unsupported C4 outer-evaluation receipt.")
    if str(outer.get("run_id")) != run.name or outer.get("final_holdout_rows_scored") != 0:
        raise ValueError("C4 receipt has the wrong run identity or does not preserve a sealed final holdout.")
    status_path = manifest_path.parent / "run_status.json"
    if not status_path.is_file() or load_json(status_path).get("status") != "COMPLETE":
        raise ValueError("C4 is not complete. Do not graph an in-progress/restarted model set.")
    if (manifest_path.parent / "run_status.json.incomplete").exists():
        raise ValueError("C4 has an incomplete marker. This report requires a completed saved evaluation.")
    profile = outer["profile"]
    levels = tenure_order(profile["selected_tenure_groups"])
    if len(levels) < 2:
        raise ValueError("A factor comparison requires one pooled/all-tenure run, not a singleton-tenure model.")
    reference = normalize(args.reference_tenure) if args.reference_tenure is not None else levels[0]
    if reference not in levels:
        raise ValueError(f"Reference tenure must be in {levels}")
    ids = list(outer["identity_columns"])
    oof_reference = outer.get("outer_oof_predictions", outer.get("oof_predictions"))
    oof_path = verified_reference(oof_reference, manifest_path, root, run)
    oof = validate_oof(read_table(oof_path), outer, ids)
    if oof_reference.get("rows") is not None and len(oof) != int(oof_reference["rows"]):
        raise ValueError("OOF record count differs from the C4 receipt.")
    if set(oof.tenure_group) != set(levels):
        raise ValueError("Observed OOF tenure groups do not cover the saved selected-tenure profile.")
    feature = args.tenure_feature.strip().upper()
    print(f"TENURE FACTOR REPORT\nRun: {run.name}\nFeature: {feature}\nFinal holdout rows to score: 0", flush=True)
    scorers, bundle_audit = {}, []
    for fold in outer["folds"]:
        number = int(fold["fold"])
        for family in FAMILIES:
            path = verified_reference(fold["families"][family], manifest_path, root, run)
            saved = checked_bundle(path, family, number, outer)
            scorers[(number, family)] = FrozenScorer(path, saved)
            bundle_audit.append({"fold": number, "family": family, "manifest": str(path), "sha256": sha256(path)})
    needed = set(ids)
    for scorer in scorers.values():
        needed.update(scorer.raw_columns())
    needed.add(feature)
    if outer["target_column"] in needed:
        raise ValueError("Target found in raw scoring column requirements.")
    snapshot_ref = outer.get("source_data_identity", outer.get("input_snapshot"))
    print("Verifying the recorded feature snapshot...", flush=True)
    snapshot = verified_reference(snapshot_ref, manifest_path, root, run)
    sample = choose_sample(oof, ids, args.sample_per_fold, args.seed)
    raw = load_sample_features(snapshot, sorted(needed), sample, ids, args.batch_size, outer["target_column"])
    factors, audit = factor_tables(oof, sample, raw, scorers, feature, levels, reference, args.batch_size)
    summary = summarize_one_way(oof)
    figures = {"tenure_actual_vs_predicted.html": one_way_figure(summary, levels, run.name)}
    for family in FAMILIES:
        figures[f"{family}_tenure_factor.html"] = factor_figure(factors, family, feature, levels, reference, run.name)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.output_dir or root / "output" / "tenure_factor" / stamp
    out = out if out.is_absolute() else root / out
    out.mkdir(parents=True, exist_ok=False)
    metadata = {"run_id": run.name, "created_utc": datetime.now(timezone.utc).isoformat(),
                "tenure_feature": feature, "tenure_levels": levels, "reference_tenure": reference,
                "outer_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
                "oof_predictions": {"path": str(oof_path), "sha256": sha256(oof_path)},
                "source_snapshot": {"path": str(snapshot), "sha256": str(snapshot_ref["sha256"])},
                "sample_per_fold": args.sample_per_fold, "actual_reference_rows": len(sample), "seed": args.seed,
                "one_way_uses_all_oof_rows": True, "tenure_grid_changes_only_requested_field": True,
                "overall_pdp_weighting": "full validation-fold row count, not capped sample count",
                "models_refit": 0, "final_holdout_rows_scored": 0, "new_raw_outcome_columns_read": [],
                "purpose": "MODEL_RESPONSE_DIAGNOSTIC_NOT_CAUSAL", "bundles": bundle_audit,
                "input_runtime_and_bundle_checks_bypassed": False,
                "methods": ["https://scikit-learn.org/stable/modules/partial_dependence.html",
                            "https://plotly.com/python/interactive-html-export/"]}
    summary.to_csv(out / "tenure_one_way_values.csv", index=False)
    factors.to_csv(out / "tenure_factor_values.csv", index=False)
    audit.to_csv(out / "tenure_feature_audit.csv", index=False)
    publish_figures(out, figures, metadata)
    metadata["output_artifacts"] = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in out.iterdir() if p.is_file()}
    (out / "report_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nReport complete: {out / 'index.html'}\nFinal holdout rows scored: 0", flush=True)
    print(audit[["fold", "model_family", "feature_retained", "max_oof_probability_error"]].to_string(index=False))
    return 0




if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nTENURE FACTOR REPORT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
