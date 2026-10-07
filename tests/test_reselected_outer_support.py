"""Contract tests use temporary metadata fixtures, never policyholder records."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

import reselect_saved_xgboost as exporter
import reselected_outer_support as support


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def record(path):
    return {"path": str(path), "sha256": exporter.sha256(path)}


def fixture_run(parent):
    run = Path(parent) / "TEST_RUN"
    directory = run / "xgboost"
    directory.mkdir(parents=True)
    prep = {"pipeline_run_id": run.name, "profile_hash": "test-profile",
            "profile": {"selected_tenure_groups": ["1"]},
            "input": r"C:\private\retention_13_1_population.parquet",
            "input_sha256": "a" * 64, "final_holdout_rows_scored": 0}
    prep_path = write_json(run / "preparation/preparation_info.json", prep)
    type_path = write_json(run / "preparation/xgboost_feature_types.json", {"A": "continuous", "B": "continuous"})
    write_json(directory / "xgb_resolved_model_feature_types.json", {"A": "continuous", "B": "continuous"})
    history = {"history_id": "RECENT_36M", "kind": "recent", "months": 36}
    groups = {"XGB_GAIN_FORWARD_N1": ["A"], "XGB_GAIN_FORWARD_N2": ["A", "B"]}
    configs = {}
    rows = []
    for group, features in groups.items():
        h = hashlib.sha256(("\n".join(features)+"\n").encode()).hexdigest()
        cfg = {"group_id": group, "raw_feature_count": len(features), "ordered_feature_sha256": h,
               "selected_candidate_id": f"{group}_CONFIG", "selected_history": history,
               "selected_model_spec": {"model_id": "SAVED_MODEL", "max_depth": 3}}
        configs[group] = cfg
        for fold in (1, 2, 3):
            rows.append({"group_id": group, "fold": fold, "raw_feature_count": len(features),
                         "churn_ap_skill": .1 if len(features) == 2 else .0995,
                         "churn_recall_10pct": .2, "validation_rows": 100,
                         "churn_prevalence_baseline": .1, "seed_count": 2, "seeds": "42;43",
                         "selected_candidate_id": cfg["selected_candidate_id"],
                         "selected_history_id": history["history_id"], "selected_model_id": "SAVED_MODEL",
                         "ordered_feature_sha256": h, "best_rounds": 10+fold})
    metrics = directory / exporter.METRICS_FILE
    pd.DataFrame(rows).to_csv(metrics, index=False)
    configs_path = write_json(directory / exporter.CONFIGURATIONS_FILE,
                             {"contract_version": 3, "profile_hash": "test-profile",
                              "finalist_group_ids": list(groups), "configurations": configs,
                              "outer_validation_rows_scored": 0, "final_holdout_rows_scored": 0})
    pd.DataFrame([{"group_id": group, "raw_feature_count": len(features), "features": ";".join(features)}
                  for group, features in groups.items()]).to_csv(directory / exporter.INVENTORY_FILE, index=False)
    artifacts = {exporter.METRICS_FILE: record(metrics), exporter.CONFIGURATIONS_FILE: record(configs_path)}
    chosen = configs["XGB_GAIN_FORWARD_N2"]
    selected = {"selected_group_id": chosen["group_id"], "selected_features": ["A", "B"],
                "selected_configuration_id": chosen["selected_candidate_id"],
                "selected_history": history, "selected_model_spec": chosen["selected_model_spec"],
                "recommended_boosting_rounds_from_inner_confirmation": 12,
                "seeds": [42, 43], "xgboost_version": "3.0.2", "outer_validation_rows_scored": 0,
                "final_holdout_rows_scored": 0, "finalist_tuning_policy": {"artifacts": artifacts}}
    write_json(directory / "selected_features.json", selected)
    write_json(directory / "run_info.json",
               {"profile_hash": "test-profile", "finalist_tuning_contract_version": 3,
                "input": {"sha256": "a"*64}, "preparation_info": record(prep_path),
                "feature_types_file": record(type_path), "finalist_artifacts": artifacts,
                "arguments": {"rare_min_frequency": .001}, "xgboost_version": "3.0.2",
                "finalist_group_ids": list(groups), "outer_validation_rows_scored": 0,
                "final_holdout_rows_scored": 0})
    write_json(run / "pipeline_status.json", {"run_id": run.name, "profile_hash": "test-profile", "status": "COMPLETE"})
    return run


class ReselectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = fixture_run(self.temporary.name)

    def test_inner_metrics_choose_parsimonious_frozen_recipe_without_population(self):
        before = {p: exporter.sha256(p) for p in self.run.rglob("*") if p.is_file()}
        recipe = exporter.export_reselection(self.run)
        loaded = support.load_reselection(self.run)
        self.assertEqual(loaded["recipe_path"], recipe)
        self.assertEqual(loaded["recipe"]["selected_features"], ["A"])
        self.assertEqual(loaded["recipe"]["recommended_boosting_rounds_from_inner_confirmation"], 12)
        self.assertEqual(loaded["recipe"]["final_holdout_rows_scored"], 0)
        from c_modeling import c4_evaluate_and_finalize as c4
        from evaluate_reselected_xgboost import make_recipe
        self.assertEqual(make_recipe(loaded["recipe"], c4).history_months, 36)
        self.assertFalse((self.run / "input_snapshot").exists())
        self.assertEqual(before, {p: exporter.sha256(p) for p in before})

    def test_policy_change_exports_separate_recipe_and_requires_unambiguous_choice(self):
        first = exporter.export_reselection(self.run)
        strict = dict(exporter.validated_policy(), mean_ap_skill_loss=0, max_fold_ap_skill_loss=0)
        second = exporter.export_reselection(self.run, policy=strict)
        self.assertNotEqual(first, second)
        self.assertEqual(support.load_reselection(self.run, second)["recipe"]["selected_features"], ["A", "B"])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            support.load_reselection(self.run)
        with self.assertRaises(FileExistsError):
            exporter.export_reselection(self.run)

    def test_rejects_mutated_saved_source_after_export(self):
        recipe = exporter.export_reselection(self.run)
        path = self.run / "xgboost" / exporter.METRICS_FILE
        path.write_text(path.read_text()+"\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            support.load_reselection(self.run, recipe)

    def test_rejects_tampered_recipe_even_when_receipt_hash_updated(self):
        path = exporter.export_reselection(self.run)
        payload = exporter.read_json(path)
        payload["recommended_boosting_rounds_from_inner_confirmation"] = 900
        write_json(path, payload)
        status_path = path.parent / "run_status.json"
        status = exporter.read_json(status_path)
        status["output_sha256"]["recipe.json"] = exporter.sha256(path)
        write_json(status_path, status)
        with self.assertRaisesRegex(ValueError, "Reproduced saved challenger"):
            support.load_reselection(self.run, path)

    def test_rejects_selection_that_scored_holdout_or_outer_rows(self):
        path = self.run / "xgboost/selected_features.json"
        original = exporter.read_json(path)
        for field in ("final_holdout_rows_scored", "outer_validation_rows_scored"):
            with self.subTest(field=field):
                write_json(path, {**original, field: 1})
                with self.assertRaisesRegex(ValueError, "exclusion"):
                    exporter.load_saved_inputs(self.run)
        write_json(path, original)

    def test_rejects_feature_inventory_changed_without_matching_hash(self):
        path = self.run / "xgboost" / exporter.INVENTORY_FILE
        text = path.read_text().replace("A;B", "B;A")
        path.write_text(text)
        with self.assertRaisesRegex(ValueError, "feature order"):
            exporter.build_recipe(exporter.load_saved_inputs(self.run))

    def test_rejects_export_outside_dedicated_directory(self):
        with self.assertRaisesRegex(ValueError, "Exports must"):
            exporter.export_reselection(self.run, output_dir=self.run / "xgboost")

    def test_declared_artifact_cannot_escape_root(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            exporter.checked_file(self.run, "../outside.json")


class ProfileActivationTests(unittest.TestCase):
    def test_reproduces_saved_profile_without_launching_pipeline(self):
        from a_business_understanding import a0_one_click
        from func.tenure_profile import profile_from_values
        import params
        root = Path(__file__).resolve().parents[1]
        try:
            with patch.dict(os.environ, dict(os.environ), clear=True):
                runtime, _ = a0_one_click._activate_profile(
                    profile_from_values(tenure_values=["2", "3"]),
                    include_exposure_year_predictor=True, test_policy=None, sample_percent=None)
                preparation = {"profile": runtime.PROFILE_PAYLOAD,
                               "profile_hash": runtime.PROFILE_HASH, "pipeline_run_id": "TEST_RUN"}
                saved_hash = runtime.PROFILE_HASH
                os.environ["RETENTION_SCORE_FINAL_HOLDOUT"] = "1"
                activated = support.activate_profile(root, preparation, "TEST_RUN")
                self.assertEqual(activated.PROFILE_HASH, saved_hash)
                self.assertEqual(activated.POPULATION_SCOPE_MODE, "FULL")
                self.assertNotIn("RETENTION_SCORE_FINAL_HOLDOUT", os.environ)
                self.assertEqual(os.environ["RETENTION_PIPELINE_RUN_ID"], "TEST_RUN")
        finally:
            importlib.reload(params)


class SharedStateTests(unittest.TestCase):
    def test_content_addressed_state_is_verified_and_original_bundle_untouched(self):
        from c_modeling import deployment_bundle as bundles, state_grouping
        state = SimpleNamespace(mapping={"TX": "TX", "AZ": "G01"}, categories=["TX", "G01"])
        contract = bundles.state_contract_from_artifact(state)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            relative = "preprocessing/state_grouping/fitted_state_grouper_deadbeef.json"
            path = write_json(root / relative, {"algorithm_id": state_grouping.ALGORITHM_ID})
            fitted = {"state_contract": {**contract, "artifact_files": [relative]}}
            manifest = {"contract": bundles.BUNDLE_CONTRACT,
                        "contract_revision": bundles.BUNDLE_CONTRACT_REVISION,
                        "artifact_inventory": [{"relative_path": relative, "sha256": exporter.sha256(path)}],
                        "source_metadata": {"shared_state_grouping": contract}}
            manifest["manifest_identity"] = bundles.sha256_json(manifest)
            before = path.read_bytes()
            def validated_loader(temporary):
                alias = Path(temporary) / "fitted_state_grouper.json"
                self.assertEqual(alias.read_bytes(), before)
                return state
            with patch.object(state_grouping.FittedStateGrouper, "load", side_effect=validated_loader):
                self.assertIs(support.load_shared_state(root, manifest, fitted), state)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse((path.parent / "fitted_state_grouper.json").exists())
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                support.load_shared_state(root, manifest, fitted)


if __name__ == "__main__":
    unittest.main()
