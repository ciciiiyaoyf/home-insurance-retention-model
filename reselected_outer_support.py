"""Validate saved challenger exports and restore frozen outer-fold state.

Reconstructed against the retained selection and deployment-bundle contracts.
These helpers never query Snowflake, select using outer metrics, fit models, or
read final-test labels. A missing or incompatible artifact fails explicitly.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from reselect_saved_xgboost import (
    EXPORT_CONTRACT, build_recipe, checked_file, load_saved_inputs, read_json,
    require_equal, sha256, verify_sha,
)


def load_reselection(run_dir: Path, recipe_path: Path | None = None) -> dict[str, Any]:
    """Accept exactly one completed export and reproduce its inner-only recipe."""
    run = run_dir.expanduser().resolve(strict=True)
    if recipe_path is None:
        candidates = []
        for path in sorted((run / "xgboost_reselections").glob("*/recipe.json")):
            receipt = path.parent / "run_status.json"
            if receipt.is_file() and read_json(receipt).get("status") == "COMPLETE":
                candidates.append(path)
        if len(candidates) != 1:
            raise ValueError("Supply --recipe: exactly one completed saved reselection export is required.")
        path = candidates[0].resolve(strict=True)
    else:
        path = recipe_path.expanduser().resolve(strict=True)
    payload = read_json(path)
    receipt = read_json(path.parent / "run_status.json")
    require_equal(payload.get("contract"), EXPORT_CONTRACT, "Recipe contract")
    require_equal(receipt.get("contract"), EXPORT_CONTRACT, "Export receipt contract")
    require_equal(receipt.get("status"), "COMPLETE", "Export completion")
    require_equal(payload.get("source_run_id"), run.name, "Recipe run identity")
    require_equal(receipt.get("source_run_id"), run.name, "Receipt run identity")
    for body in (payload, receipt):
        require_equal(body.get("final_holdout_rows_scored"), 0, "Export final-test exclusion")
        require_equal(body.get("outer_validation_rows_scored"), 0, "Export outer exclusion")
    outputs = receipt.get("output_sha256", {})
    if set(outputs) != {"recipe.json", "selection_summary.csv"} or path.name != "recipe.json":
        raise ValueError("Completed export has an incompatible output inventory.")
    export_hashes = {}
    for relative, digest in outputs.items():
        file_path = checked_file(path.parent, relative)
        verify_sha(file_path, digest)
        export_hashes[file_path] = digest
    saved = load_saved_inputs(run)
    recorded_sources = payload.get("source_artifacts", {})
    actual_sources = {str(p.relative_to(run)): digest for p, digest in saved["source_hashes"].items()}
    require_equal(recorded_sources, actual_sources, "Frozen source artifact identities")
    # Recompute the decision and all recipe fields; no unrecorded manual changes
    # to feature order, history, rounds, model parameters, or policy are accepted.
    expected, _ = build_recipe(saved, payload["selection_policy"])
    require_equal(payload, expected, "Reproduced saved challenger recipe")
    return {"recipe": payload, "recipe_path": path, "saved": saved,
            "source_hashes": {**saved["source_hashes"], **export_hashes,
                              path.parent / "run_status.json": sha256(path.parent / "run_status.json")}}


def activate_profile(project_root: Path, preparation: Mapping[str, Any], run_id: str) -> Any:
    """Activate the exact saved FULL profile through the retained launcher API."""
    root = project_root.expanduser().resolve(strict=True)
    if not (root / "params.py").is_file():
        raise ValueError("Project root must contain params.py.")
    from a_business_understanding import a0_one_click
    from func.tenure_profile import profile_from_values

    groups = preparation["profile"]["selected_tenure_groups"]
    if not groups:
        raise ValueError("Preparation has no recorded tenure groups.")
    if "pipeline_run_id" in preparation:
        require_equal(preparation["pipeline_run_id"], run_id, "Preparation run identity")
    profile = profile_from_values(tenure_values=groups)
    for include_year in (False, True):
        params, _ = a0_one_click._activate_profile(
            profile, include_exposure_year_predictor=include_year,
            test_policy=None, sample_percent=None,
        )
        if str(params.PROFILE_HASH) == str(preparation["profile_hash"]):
            require_equal(Path(params.PROJECT_ROOT).resolve(), root, "Imported project root")
            require_equal(params.POPULATION_SCOPE_MODE, "FULL", "Population scope")
            os.environ["RETENTION_PIPELINE_RUN_ID"] = run_id
            params = importlib.reload(params)
            require_equal(params.PROFILE_HASH, preparation["profile_hash"], "Activated profile")
            return params
    raise ValueError("Current parameters cannot reproduce the saved FULL profile; restore its matching parameters.")


def load_shared_state(bundle_dir: Path, manifest: Mapping[str, Any], fitted: Mapping[str, Any]):
    """Load the original audited grouper from content-addressed bundle files.

    Bundles rename their JSON artifacts using hashes, whereas the grouper's
    existing loader expects a fixed filename. A temporary alias permits use of
    that loader's complete integrity checks without changing the saved bundle.
    """
    from c_modeling import deployment_bundle as bundles, state_grouping

    root = bundle_dir.expanduser().resolve(strict=True)
    require_equal(manifest.get("contract"), bundles.BUNDLE_CONTRACT, "Bundle contract")
    require_equal(manifest.get("contract_revision"), bundles.BUNDLE_CONTRACT_REVISION, "Bundle revision")
    identity = bundles.sha256_json({key: value for key, value in manifest.items() if key != "manifest_identity"})
    require_equal(manifest.get("manifest_identity"), identity, "Bundle manifest identity")
    state_contract = fitted["state_contract"]
    files = state_contract.get("artifact_files", [])
    if not files or len(files) != len(set(files)):
        raise ValueError("Shared state artifact inventory is missing or duplicated.")
    inventory = {item["relative_path"]: item["sha256"] for item in manifest["artifact_inventory"]}
    if len(inventory) != len(manifest["artifact_inventory"]):
        raise ValueError("Bundle inventory has duplicate artifact names.")
    candidates = []
    for relative in files:
        if relative not in inventory:
            raise ValueError("Shared state artifact is absent from bundle inventory.")
        path = checked_file(root, relative)
        verify_sha(path, inventory[relative])
        if path.suffix.lower() == ".json":
            payload = read_json(path)
            if payload.get("algorithm_id") == state_grouping.ALGORITHM_ID:
                candidates.append(path)
    if len(candidates) != 1:
        raise ValueError("Expected exactly one serialized shared state grouper in the bundle.")
    with tempfile.TemporaryDirectory(prefix="retention_state_read_") as temporary:
        alias = Path(temporary) / "fitted_state_grouper.json"
        alias.write_bytes(candidates[0].read_bytes())
        state = state_grouping.FittedStateGrouper.load(temporary)
    actual = bundles.state_contract_from_artifact(state)
    for key in ("mapping", "categories", "mapping_sha256", "scope"):
        require_equal(actual[key], state_contract.get(key), f"Preprocessor shared-state {key}")
    source_contract = manifest["source_metadata"]["shared_state_grouping"]
    for key in ("mapping", "categories", "mapping_sha256", "scope"):
        require_equal(actual[key], source_contract.get(key), f"Manifest shared-state {key}")
    return state
