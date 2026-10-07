"""Recover completed C4 fits without Optuna or production data access."""


from __future__ import annotations


from dataclasses import replace
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock


import numpy as np
import pandas as pd


from c_modeling import c4_evaluate_and_finalize as c4
from c_modeling import deployment_bundle as bundle
from c_modeling import modeling_utils




class TinyState:
    mapping = {"TX": "TX", "OH": "G1"}
    categories = ["TX", "G1"]


    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "mapping.json").write_text(json.dumps(self.mapping), encoding="utf-8")




class TinyXGBArtifact:
    """Only the seed-file writer is stubbed; integrity validation is real."""
    def __init__(self, seed):
        self.seed = seed


    def fit(self, matrix, target, **kwargs):
        return self


    def save_model(self, filename):
        Path(filename).write_bytes(f"test-seed-{self.seed}".encode())


    def predict_proba(self, matrix):
        return np.tile([0.4, 0.6], (matrix.shape[0], 1))




class SavedFitRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name).resolve() / "RUN"
        self.run.mkdir()
        self.frame = pd.DataFrame({
            "ID": ["A", "B", "C", "D", "E", "F"],
            "COHORT": pd.to_datetime(["2021-01-01"] * 3 + ["2021-02-01"] * 3),
            "Y": [0, 0, 1, 0, 1, 1], "F": [1., 2., 3., 4., 5., 6.],
        })
        self.params = types.SimpleNamespace(
            PROFILE_HASH="test-profile", PROFILE_PAYLOAD={"selected_tenure_groups": [0, 1, 2, 3, "4+"]},
            SEED=42, XGB_DEVICE="cpu", N_JOBS=1,
        )
        self.utils = types.SimpleNamespace(
            WRITTEN_PREM_BIN_FEATURE="WRITTEN_PREM_BIN", WRITTEN_PREM_SOURCE_FEATURE="WRITTEN_PREM",
            STATE_GROUP_FEATURE="STATE_GRP", STATE_SOURCE_FEATURE="STATE",
            prefit_derived_feature_artifacts=lambda *a, **k: {},
        )
        self.state = TinyState()
        self.state_contract = bundle.state_contract_from_artifact(self.state)
        self.state_contract.update(derived_feature="STATE_GRP", source_feature="STATE")
        self.base_population = c4._population_contract(
            self.frame, indices=self.frame.index.to_numpy(), cohort_col="COHORT",
            identity_columns=["ID", "COHORT"], deployment_bundle=bundle, definition="test training rows",
        )
        self.sources = {"modeling_utils": Path(modeling_utils.__file__)}
        self.history = {"kind": "recent", "months": 36, "history_id": "RECENT_36M"}
        self.recipe = c4.FamilyRecipe(
            family="logistic", features=("F",), feature_types={"F": "continuous"},
            history=self.history, history_months=36, rare_min_frequency=0.001,
            model_recipe={"penalty": "l2", "C": 0.1, "solver": "lbfgs", "max_iter": 100,
                          "tol": 1e-4, "class_weight": None, "random_seed": 42,
                          "selected_history": self.history},
        )


    def metadata(self, *, family="logistic", fold=1, final=False, legacy=False):
        role = "FINAL_REFIT" if final else "OUTER_FOLD_EVALUATION"
        scope = {"bundle_role": role, "base_training_index_sha256": self.base_population["index_sha256"]}
        if final:
            scope["selected_model_family"] = family
            if not legacy:
                scope["fitted_model_family"] = family
        else:
            scope.update(outer_fold=fold, validation_index_sha256="validation-indices")
        return c4._source_metadata(
            params=self.params, run_id=self.run.name, run_root=self.run,
            source_data_identity={"sha256": "snapshot-hash", "bytes": 123}, source_tenure="ALL_SELECTED",
            excluded_holdout={"rows": 2, "population_sha256": "holdout-hash"},
            sealed_final_holdout={"rows": 2, "population_sha256": "final-holdout-hash", "rows_scored": 0},
            evaluation_training_population=self.base_population, cohort_col="COHORT",
            shared_state_contract=self.state_contract, pairing_scope=scope, bundle_role=role,
            outer_fold=None if final else fold,
            history_audit={"cutoff_month_inclusive": "2018-03", "anchor_month": "2021-02", "selected_rows": 6},
        )


    def make_bundle(self, *, family="logistic", fold=1, final=False, legacy=False):
        destination = (
            self.run / "deployment/final_model" / family if final
            else self.run / "outer_evaluation/fold_models" / f"fold_{fold:02d}" / family
        )
        recipe = self.recipe
        if family == "xgboost":
            recipe = replace(recipe, family=family, seeds=(42, 2026, 31415), boosting_rounds=8,
                             model_recipe={"max_depth": 2, "learning_rate": 0.1, "selected_history": self.history})
        arguments = dict(
            family=family, destination=destination, training_frame=self.frame, recipe=recipe,
            target_col="Y", cohort_col="COHORT", identity_columns=["ID", "COHORT"],
            modeling_utils=self.utils, deployment_bundle=bundle, shared_state_artifact=self.state,
            source_metadata=self.metadata(family=family, fold=fold, final=final, legacy=legacy),
            source_code_paths=self.sources, params=self.params,
        )
        if family == "logistic":
            c4._export_family_bundle(**arguments)
        else:
            # Exercise real preprocessing, publication, JSON and file hashing
            # without requiring the native XGBoost library in the test runner.
            export_kwargs = {key: arguments[key] for key in (
                "destination", "training_frame", "target_col", "cohort_col", "identity_columns",
                "modeling_utils", "shared_state_artifact", "source_metadata", "source_code_paths",
            )}
            bundle.export_xgboost_source_bundle(
                **export_kwargs, features=recipe.features, feature_types=recipe.feature_types,
                rare_min_frequency=recipe.rare_min_frequency, seeds=recipe.seeds,
                boosting_rounds=recipe.boosting_rounds, sample_weight=None,
                model_factory=lambda seed, rounds: TinyXGBArtifact(seed),
                model_recipe=recipe.model_recipe, xgboost_version="test-writer",
            )
        return arguments


    def archive(self, attempt="attempt1"):
        records = c4._archive_partial_c4_attempt(self.run, recovery_attempt_id=attempt)
        return c4._validated_c4_archive_records(self.run, records)


    def write_windows_runtime_references(self, destination):
        """Reproduce the Windows exporter's native runtime path separators.


        The real exporter normalizes inventory entries but writes runtime
        references using str(Path). Recreate that manifest on any test OS,
        leaving every model, preprocessor and runtime file unchanged.
        """
        path = c4._find_bundle_manifest(destination)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["runtime_sources"] = {
            label: relative.replace("/", "\\")
            for label, relative in manifest["runtime_sources"].items()
        }
        self.rewrite_fixture_manifest(path, manifest)


    def rewrite_fixture_manifest(self, path, manifest):
        manifest["manifest_identity"] = bundle.sha256_json({
            key: value for key, value in manifest.items() if key != "manifest_identity"
        })
        new_path = path.with_name(f"family_manifest_{manifest['manifest_identity'][:20]}.json")
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if new_path != path:
            path.rename(new_path)


    def test_windows_runtime_paths_reuse_both_families_without_changing_bundle_bytes(self):
        for family in ("logistic", "xgboost"):
            with self.subTest(family=family):
                arguments = self.make_bundle(family=family)
                destination = arguments["destination"]
                self.write_windows_runtime_references(destination)
                expected = {str(path.relative_to(destination)): c4._sha256(path)
                            for path in destination.rglob("*") if path.is_file()}
                before = (bundle.score_logistic_bundle(destination, self.frame)
                          if family == "logistic" else None)
                archives = self.archive(attempt=f"windows_{family}")
                with mock.patch.object(bundle, "validate_runtime_versions"), mock.patch.object(
                    bundle, "export_logistic_source_bundle", side_effect=AssertionError("must not fit")
                ), mock.patch.object(
                    bundle, "export_xgboost_source_bundle", side_effect=AssertionError("must not fit")
                ):
                    c4._export_family_bundle(**arguments, recovery_archives=archives)
                observed = {str(path.relative_to(destination)): c4._sha256(path)
                            for path in destination.rglob("*") if path.is_file()}
                self.assertEqual(observed, expected)
                if before is not None:
                    np.testing.assert_array_equal(
                        before, bundle.score_logistic_bundle(destination, self.frame)
                    )


    def test_windows_runtime_paths_do_not_hide_corrupted_artifacts(self):
        arguments = self.make_bundle()
        self.write_windows_runtime_references(arguments["destination"])
        archives = self.archive()
        source = Path(archives[0]["archive_directory"]) / "fold_models/fold_01/logistic"
        manifest = json.loads(c4._find_bundle_manifest(source).read_text(encoding="utf-8"))
        with mock.patch.object(bundle, "export_logistic_source_bundle", side_effect=AssertionError("must not fit")):
            for relative in (manifest["model_file"], manifest["preprocessor_file"],
                             *[value.replace("\\", "/") for value in manifest["runtime_sources"].values()]):
                with self.subTest(artifact=relative):
                    artifact = source / relative
                    original = artifact.read_bytes()
                    artifact.write_bytes(original + b"corrupt")
                    try:
                        with self.assertRaisesRegex(RuntimeError, "artifact hash mismatch"):
                            c4._export_family_bundle(**arguments, recovery_archives=archives)
                        self.assertFalse(arguments["destination"].exists())
                    finally:
                        artifact.write_bytes(original)


    def test_windows_runtime_lookup_still_requires_safe_paths_and_recorded_hashes(self):
        arguments = self.make_bundle()
        self.write_windows_runtime_references(arguments["destination"])
        archives = self.archive()
        source = Path(archives[0]["archive_directory"]) / "fold_models/fold_01/logistic"
        original = json.loads(c4._find_bundle_manifest(source).read_text(encoding="utf-8"))
        runtime_path = original["runtime_sources"]["modeling_utils"].replace("\\", "/")
        missing_entry = copy.deepcopy(original)
        missing_entry["artifact_inventory"] = [item for item in missing_entry["artifact_inventory"]
                                                if item["relative_path"] != runtime_path]
        missing_hash = copy.deepcopy(original)
        next(item for item in missing_hash["artifact_inventory"]
             if item["relative_path"] == runtime_path).pop("sha256")
        cases = [(missing_entry, "absent from the integrity inventory"),
                 (missing_hash, "no valid SHA-256")]
        for relative in ("runtime\\..\\modeling_utils.py", "C:\\runtime\\modeling_utils.py"):
            manifest = copy.deepcopy(original)
            manifest["runtime_sources"]["modeling_utils"] = relative
            cases.append((manifest, "unsafe relative file reference"))
        with mock.patch.object(bundle, "export_logistic_source_bundle", side_effect=AssertionError("must not fit")):
            for manifest, message in cases:
                with self.subTest(message=message):
                    self.rewrite_fixture_manifest(c4._find_bundle_manifest(source), manifest)
                    with self.assertRaisesRegex(RuntimeError, message):
                        c4._export_family_bundle(**arguments, recovery_archives=archives)
                    self.assertFalse(arguments["destination"].exists())


    def test_real_logistic_fit_is_reused_and_predictions_are_identical(self):
        arguments = self.make_bundle()
        before = bundle.score_logistic_bundle(arguments["destination"], self.frame)
        manifest_hash = c4._sha256(c4._find_bundle_manifest(arguments["destination"]))
        archives, actions = self.archive(), []
        with mock.patch.object(bundle, "export_logistic_source_bundle", side_effect=AssertionError("must not fit")):
            _, path = c4._export_family_bundle(**arguments, recovery_archives=archives, recovery_records=actions)
        after = bundle.score_logistic_bundle(arguments["destination"], self.frame)
        np.testing.assert_array_equal(before, after)
        self.assertEqual(c4._sha256(path), manifest_hash)
        self.assertEqual(actions[0]["action"], "REUSED_VERIFIED_BUNDLE")
        self.assertTrue(Path(actions[0]["source_bundle"]).is_dir())


    def test_all_five_fold_bundles_recover_without_calling_an_exporter(self):
        arguments = [self.make_bundle(fold=fold) for fold in c4.OUTER_FOLD_IDS]
        archives, actions = self.archive(), []
        with mock.patch.object(bundle, "export_logistic_source_bundle", side_effect=AssertionError("must not fit")):
            for entry in arguments:
                c4._export_family_bundle(**entry, recovery_archives=archives, recovery_records=actions)
        self.assertEqual([item["outer_fold"] for item in actions], list(c4.OUTER_FOLD_IDS))


    def test_legacy_final_refit_pairing_is_accepted_without_rewriting_model(self):
        arguments = self.make_bundle(final=True, legacy=True)
        original_hash = c4._sha256(c4._find_bundle_manifest(arguments["destination"]))
        arguments["source_metadata"] = self.metadata(final=True)
        archives = self.archive()
        with mock.patch.object(bundle, "export_logistic_source_bundle", side_effect=AssertionError("must not fit")):
            _, path = c4._export_family_bundle(**arguments, recovery_archives=archives)
        self.assertEqual(c4._sha256(path), original_hash)


    def test_changed_recipe_routing_rows_fold_snapshot_or_holdout_is_rejected(self):
        arguments = self.make_bundle()
        archives = self.archive()
        changes = [
            {"recipe": replace(self.recipe, model_recipe={**self.recipe.model_recipe, "C": 2.0})},
            {"recipe": replace(self.recipe, feature_types={"F": "discrete"})},
            {"recipe": replace(self.recipe, rare_min_frequency=0.01)},
            {"recipe": replace(self.recipe, history_months=48,
                               history={"kind": "recent", "months": 48, "history_id": "RECENT_48M"},
                               model_recipe={**self.recipe.model_recipe, "selected_history": {
                                   "kind": "recent", "months": 48, "history_id": "RECENT_48M"}})},
            {"training_frame": self.frame.assign(ID=["X", "B", "C", "D", "E", "F"])},
            {"source_metadata": self.metadata(fold=2)},
            {"source_metadata": {**self.metadata(), "profile_hash": "different-profile"}},
            {"source_metadata": {**self.metadata(), "source_data_identity": {"sha256": "different"}}},
            {"source_metadata": {**self.metadata(), "sealed_final_holdout": {"rows_scored": 1}}},
        ]
        with mock.patch.object(bundle, "export_logistic_source_bundle", side_effect=AssertionError("must not fit")):
            for change in changes:
                with self.subTest(change=list(change)):
                    with self.assertRaisesRegex(RuntimeError, "no replacement fit was started"):
                        c4._export_family_bundle(**{**arguments, **change}, recovery_archives=archives)
                    self.assertFalse(arguments["destination"].exists())


    def test_corrupted_model_and_changed_runtime_are_rejected(self):
        arguments = self.make_bundle()
        archives = self.archive()
        source = Path(archives[0]["archive_directory"]) / "fold_models/fold_01/logistic"
        manifest, _ = bundle.load_bundle(source)
        current_code = self.run / "different_utils.py"
        current_code.write_text("# different runtime", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "runtime source modeling_utils"):
            c4._export_family_bundle(**{**arguments, "source_code_paths": {"modeling_utils": current_code}}, recovery_archives=archives)
        (source / manifest["model_file"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(RuntimeError, "artifact hash mismatch"):
            c4._export_family_bundle(**arguments, recovery_archives=archives)


    def test_xgboost_seed_files_are_reused_and_seed_or_round_changes_rejected(self):
        arguments = self.make_bundle(family="xgboost")
        archives = self.archive()
        # Runtime-version compatibility itself is already enforced by load_bundle;
        # this fixture uses test seed files rather than native XGBoost estimators.
        with mock.patch.object(bundle, "validate_runtime_versions"), mock.patch.object(
            bundle, "export_xgboost_source_bundle", side_effect=AssertionError("must not fit")
        ):
            for recipe in (
                replace(arguments["recipe"], seeds=(42,)),
                replace(arguments["recipe"], boosting_rounds=9),
            ):
                with self.assertRaisesRegex(RuntimeError, "no replacement fit was started"):
                    c4._export_family_bundle(**{**arguments, "recipe": recipe}, recovery_archives=archives)
            manifest, _ = c4._export_family_bundle(**arguments, recovery_archives=archives)
        self.assertEqual(manifest["seeds"], [42, 2026, 31415])
        self.assertEqual(len(manifest["model_files"]), 3)


    def test_missing_comparison_bundle_is_the_only_new_fit(self):
        arguments = self.make_bundle(final=True)
        archives = self.archive()
        missing = {**arguments, "destination": self.run / "outer_evaluation/history_models/logistic"}
        missing["source_metadata"] = {**self.metadata(final=True), "bundle_role": "FINAL_REFIT_COMPARISON"}
        actions = []
        with mock.patch.object(bundle, "export_logistic_source_bundle", wraps=bundle.export_logistic_source_bundle) as fit:
            c4._export_family_bundle(**missing, recovery_archives=archives, recovery_records=actions)
        self.assertEqual(fit.call_count, 1)
        self.assertEqual(actions[0]["action"], "FITTED_NEW_BUNDLE")


    def test_archive_validation_rejects_another_run_and_altered_receipt(self):
        self.make_bundle()
        archives = self.archive()
        changed = copy.deepcopy(archives)
        changed[0]["run_id"] = "OTHER"
        with self.assertRaisesRegex(RuntimeError, "does not belong"):
            c4._validated_c4_archive_records(self.run, changed)
        changed = copy.deepcopy(archives)
        changed[0]["recovery_attempt_id"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "receipt differs"):
            c4._validated_c4_archive_records(self.run, changed)


    def test_second_resume_can_find_a_fit_in_an_earlier_validated_archive(self):
        first = self.make_bundle(fold=1)
        second = self.make_bundle(fold=2)
        original = self.archive("original")
        c4._export_family_bundle(**first, recovery_archives=original)
        newest = self.archive("retry")
        sources = c4._validated_c4_archive_records(self.run, [*newest, *original])
        with mock.patch.object(bundle, "export_logistic_source_bundle", side_effect=AssertionError("must not fit")):
            c4._export_family_bundle(**second, recovery_archives=sources)


    def test_copy_failure_preserves_archive_and_does_not_publish_partial_model(self):
        arguments = self.make_bundle()
        archives = self.archive()
        source = Path(archives[0]["archive_directory"]) / "fold_models/fold_01/logistic"
        original_hash = c4._sha256(c4._find_bundle_manifest(source))
        with mock.patch.object(c4.shutil, "copytree", side_effect=OSError("copy interrupted")):
            with self.assertRaisesRegex(OSError, "copy interrupted"):
                c4._export_family_bundle(**arguments, recovery_archives=archives)
        self.assertFalse(arguments["destination"].exists())
        self.assertEqual(c4._sha256(c4._find_bundle_manifest(source)), original_hash)
        self.assertEqual(list(arguments["destination"].parent.glob(".r_*")), [])


    def test_existing_destination_is_never_overwritten(self):
        arguments = self.make_bundle()
        archives = self.archive()
        destination = arguments["destination"]
        destination.mkdir(parents=True)
        (destination / "owner.txt").write_text("existing", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            c4._export_family_bundle(**arguments, recovery_archives=archives)
        self.assertEqual((destination / "owner.txt").read_text(encoding="utf-8"), "existing")




if __name__ == "__main__":
    unittest.main()
