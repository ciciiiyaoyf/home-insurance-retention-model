# -*- coding: utf-8 -*-
"""Prepare model-family candidate pools and chronological folds.


This stage does not select a Logistic Regression or XGBoost model. It prepares
one validated modeling snapshot, one set of folds, and family-specific feature
representations. Both families use the supervised state group while other
paired features can still route binned to Logistic and raw to XGBoost. They use
the same population, target, and fold definitions.
"""


from __future__ import annotations


import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


import numpy as np
import pandas as pd




PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from c_modeling import feature_routing, modeling_utils  # noqa: E402
from b_data_acquisition.b1_download_retention_13_1 import (  # noqa: E402
    load_download_contract,
    verify_published_artifacts,
)
from params import (  # noqa: E402
    DOMINANT_LEVEL_DROP_THRESHOLD,
    DOWNLOAD_PATH,
    DOWNLOAD_METADATA_PATH,
    FEATURE_CONTRACT_PATH,
    FINAL_HOLDOUT_MONTHS,
    HIGH_CARDINALITY_RATIO,
    INNER_SELECTION_OUTER_FOLD,
    INNER_VALIDATION_FOLDS,
    INNER_VALIDATION_MONTHS,
    LABEL_AVAILABILITY_BUFFER_DAYS,
    MAX_CATEGORY_LEVELS,
    MIN_INNER_TRAIN_MONTHS,
    MIN_OUTER_TRAIN_MONTHS,
    MISSING_DROP_THRESHOLD,
    OUTER_VALIDATION_MONTHS,
    PIPELINE_OUTPUT_ROOT,
    PROFILE_HASH,
    PROFILE_LABEL,
    PROFILE_PAYLOAD,
    PROFILE_SLUG,
    PROJECT_CONTRACT_VERSION,
    QUALITY_SCREEN_MAX_ROWS,
    ROLLING_VALIDATION_FOLDS,
    SEED,
    STRICT_DATASET_CONTRACT,
    resolve_data_config_path,
)




FEATURE_CONTRACT = Path(FEATURE_CONTRACT_PATH)
SNAPSHOT_METADATA = Path(DOWNLOAD_METADATA_PATH)
MODELING_UTILS_FILE = Path(modeling_utils.__file__).resolve()
FEATURE_ROUTING_FILE = Path(feature_routing.__file__).resolve()
FEATURE_VARIANT_MAP_FILE = Path(feature_routing.FEATURE_VARIANT_MAP_PATH).resolve()
STATE_GROUPING_FILE = PROJECT_ROOT / "c_modeling" / "state_grouping.py"
PREPARATION_SCRIPT_FILE = Path(__file__).resolve()
RUN_ID = os.environ.get("RETENTION_PIPELINE_RUN_ID", "manual").strip() or "manual"
OUTPUT_DIR = Path(PIPELINE_OUTPUT_ROOT) / RUN_ID / "preparation"


ELIGIBLE_POOL_FILES = {
    "logistic": "eligible_logistic_candidate_pool.txt",
    "xgboost": "eligible_xgboost_candidate_pool.txt",
}
QUALITY_SCREENED_POOL_FILES = {
    "logistic": "quality_screened_logistic_candidate_pool.txt",
    "xgboost": "quality_screened_xgboost_candidate_pool.txt",
}
FEATURE_TYPES_FILES = {
    "logistic": "logistic_feature_types.json",
    "xgboost": "xgboost_feature_types.json",
}
ROUTING_AUDIT_FILENAME = "feature_routing_audit.csv"




def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()




def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)




def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str))




def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)




def _quality_screen_routed_candidates(
    routed: feature_routing.RoutingResult,
    screened_source_candidates: list[str],
    quality_report: pd.DataFrame,
) -> tuple[list[str], dict[str, str], list[dict[str, str]]]:
    """Propagate the source-only quality screen through one routed pool.


    Virtual features cannot be materialized before their fold-local fit. Their
    quality eligibility therefore follows their declared raw dependencies.
    Physical variants follow their own source quality-screen decision.
    """


    screened = {str(value).strip().upper() for value in screened_source_candidates}
    if "feature" not in quality_report or "reason" not in quality_report:
        raise ValueError("Quality report must contain feature and reason columns.")
    reason_by_feature = {
        str(row.feature).strip().upper(): str(row.reason)
        for row in quality_report.loc[:, ["feature", "reason"]].itertuples(
            index=False
        )
    }


    kept: list[str] = []
    kept_types: dict[str, str] = {}
    audit_rows: list[dict[str, str]] = []
    for feature in routed.candidates:
        dependencies = feature_routing.derived_source_requirements(feature)
        basis = dependencies or (feature,)
        failed = [source for source in basis if source not in screened]
        if failed:
            decision = "DROP"
            reasons = [
                f"{source}: {reason_by_feature.get(source, 'not quality-screened')}"
                for source in failed
            ]
            reason = "; ".join(reasons)
        else:
            decision = "KEEP"
            reason = "selected representation and all source dependencies passed"
            kept.append(feature)
            kept_types[feature] = str(routed.feature_types[feature])
        audit_rows.append(
            {
                "model_family": routed.model_family,
                "selected_feature": feature,
                "quality_screen_basis": "|".join(basis),
                "quality_screen_decision": decision,
                "quality_screen_reason": reason,
            }
        )


    if not kept:
        raise ValueError(
            f"Training-only quality screening removed every {routed.model_family} "
            "candidate."
        )
    feature_routing.validate_one_variant_per_concept(kept, routed.model_family)
    return kept, kept_types, audit_rows




def _family_quality_screen_basis(
    routed: feature_routing.RoutingResult,
    source_feature_types: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    """Return physical columns/types that can be screened before fold fitting."""


    candidates: list[str] = []
    types: dict[str, str] = {}
    for selected in routed.candidates:
        dependencies = feature_routing.derived_source_requirements(selected)
        if dependencies:
            for source in dependencies:
                if source not in source_feature_types:
                    raise KeyError(
                        f"No source type is available for derived {selected}: {source}."
                    )
                if source not in types:
                    candidates.append(source)
                    types[source] = source_feature_types[source]
            continue
        candidates.append(selected)
        types[selected] = str(routed.feature_types[selected])
    if len(candidates) != len(set(candidates)):
        raise ValueError("Family quality-screen basis contains duplicates.")
    return candidates, types




def _oldest_complete_train(
    frame: pd.DataFrame,
    outer_fold: dict[str, Any],
    cohort_col: str,
) -> pd.DataFrame:
    """Use only complete cohort months from the oldest outer training fold."""


    train_idx = np.asarray(outer_fold["train_idx"])
    month = pd.to_datetime(frame[cohort_col], errors="raise").dt.to_period("M")
    in_train = pd.Series(frame.index.isin(train_idx), index=frame.index)
    complete = in_train.groupby(month).all()
    output = frame.loc[month.isin(complete[complete].index)].copy()
    if output.empty or np.setdiff1d(output.index.to_numpy(), train_idx).size:
        raise AssertionError("Invalid oldest-complete-month selection population.")
    return output




def _inner_selection_outer_fold(
    outer_folds: list[dict[str, Any]],
) -> dict[str, Any]:
    """Resolve the governed oldest outer fold by identity, not list position."""


    if type(INNER_SELECTION_OUTER_FOLD) is not int or INNER_SELECTION_OUTER_FOLD != 1:
        raise RuntimeError(
            "INNER_SELECTION_OUTER_FOLD must remain the governed integer value 1."
        )
    matches = [
        fold
        for fold in outer_folds
        if not isinstance(fold.get("fold"), (bool, np.bool_))
        and isinstance(fold.get("fold"), (int, np.integer))
        and int(fold["fold"]) == INNER_SELECTION_OUTER_FOLD
    ]
    if len(matches) != 1:
        raise ValueError(
            "Outer folds must contain exactly one fold numbered 1 for inner "
            "feature discovery."
        )
    return matches[0]




def _split_settings() -> dict[str, Any]:
    """Return the complete, serializable chronological-split fingerprint."""


    return {
        "project_contract_version": PROJECT_CONTRACT_VERSION,
        "profile": PROFILE_PAYLOAD,
        "profile_hash": PROFILE_HASH,
        "seed": SEED,
        "rolling_validation_folds": ROLLING_VALIDATION_FOLDS,
        "outer_validation_months": OUTER_VALIDATION_MONTHS,
        "final_holdout_months": FINAL_HOLDOUT_MONTHS,
        "min_outer_train_months": MIN_OUTER_TRAIN_MONTHS,
        "inner_validation_folds": INNER_VALIDATION_FOLDS,
        "inner_validation_months": INNER_VALIDATION_MONTHS,
        "min_inner_train_months": MIN_INNER_TRAIN_MONTHS,
        "inner_selection_outer_fold": INNER_SELECTION_OUTER_FOLD,
        "label_availability_buffer_days": LABEL_AVAILABILITY_BUFFER_DAYS,
        "strict_dataset_contract": STRICT_DATASET_CONTRACT,
    }




def _can_reuse(info_path: Path) -> bool:
    if not info_path.is_file():
        return False
    try:
        payload = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    required_outputs = [
        OUTPUT_DIR / "quality_screened_candidate_pool.txt",
        OUTPUT_DIR / "outer_fold_definitions.csv",
        OUTPUT_DIR / "inner_fold_definitions.csv",
        OUTPUT_DIR / "feature_types.json",
        OUTPUT_DIR / "model_family_quality_screen.csv",
        OUTPUT_DIR / ROUTING_AUDIT_FILENAME,
        *(
            OUTPUT_DIR / filename
            for filename in ELIGIBLE_POOL_FILES.values()
        ),
        *(
            OUTPUT_DIR / filename
            for filename in QUALITY_SCREENED_POOL_FILES.values()
        ),
        *(
            OUTPUT_DIR / filename
            for filename in FEATURE_TYPES_FILES.values()
        ),
    ]
    fingerprinted_inputs = [
        Path(DOWNLOAD_PATH),
        SNAPSHOT_METADATA,
        FEATURE_CONTRACT,
        MODELING_UTILS_FILE,
        FEATURE_ROUTING_FILE,
        FEATURE_VARIANT_MAP_FILE,
        STATE_GROUPING_FILE,
        PREPARATION_SCRIPT_FILE,
    ]
    return (
        all(path.is_file() for path in required_outputs)
        and all(path.is_file() for path in fingerprinted_inputs)
        and payload.get("input_sha256") == _sha256(Path(DOWNLOAD_PATH))
        and payload.get("input_metadata_sha256") == _sha256(SNAPSHOT_METADATA)
        and payload.get("feature_contract_sha256") == _sha256(FEATURE_CONTRACT)
        and payload.get("modeling_utils_sha256")
        == _sha256(MODELING_UTILS_FILE)
        and payload.get("feature_routing_sha256")
        == _sha256(FEATURE_ROUTING_FILE)
        and payload.get("feature_variant_map")
        == str(FEATURE_VARIANT_MAP_FILE)
        and payload.get("feature_variant_map_sha256")
        == _sha256(FEATURE_VARIANT_MAP_FILE)
        and payload.get("state_grouping_sha256")
        == _sha256(STATE_GROUPING_FILE)
        and payload.get("preparation_script_sha256")
        == _sha256(PREPARATION_SCRIPT_FILE)
        and payload.get("split_settings") == _split_settings()
    )




def main() -> None:
    data_path = Path(DOWNLOAD_PATH).resolve()
    required = [
        data_path,
        SNAPSHOT_METADATA,
        FEATURE_CONTRACT,
        MODELING_UTILS_FILE,
        FEATURE_ROUTING_FILE,
        FEATURE_VARIANT_MAP_FILE,
        STATE_GROUPING_FILE,
        PREPARATION_SCRIPT_FILE,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required modeling-preparation input(s) missing:\n"
            + "\n".join(f"- {path}" for path in missing)
        )


    download_contract = load_download_contract(resolve_data_config_path())
    verify_published_artifacts(
        data_path,
        SNAPSHOT_METADATA,
        download_contract,
    )


    info_path = OUTPUT_DIR / "preparation_info.json"
    if _can_reuse(info_path):
        print(f"MODELING PREPARATION VERIFIED; REUSING {OUTPUT_DIR}")
        return


    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Incomplete or mismatched preparation outputs already exist: {OUTPUT_DIR}"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    raw = modeling_utils.read_table(data_path)
    manifest = pd.read_csv(FEATURE_CONTRACT, dtype=str, keep_default_na=False)
    prepared, outer_folds, final_test_idx, split_summary, config = (
        modeling_utils.load_project_population_and_folds(
            raw,
            rolling_validation_folds=ROLLING_VALIDATION_FOLDS,
            validation_months=OUTER_VALIDATION_MONTHS,
            test_months=FINAL_HOLDOUT_MONTHS,
            min_train_months=MIN_OUTER_TRAIN_MONTHS,
            strict_dataset_contract=STRICT_DATASET_CONTRACT,
        )
    )
    del raw


    audit, candidates, feature_types = modeling_utils.build_candidate_audit(
        manifest, prepared.columns
    )
    if not candidates:
        raise ValueError("The config-derived candidate pool is empty.")
    routed = {
        family: feature_routing.route_candidates(
            candidates, feature_types, family
        )
        for family in feature_routing.MODEL_FAMILIES
    }


    for fold in outer_folds:
        if np.intersect1d(final_test_idx, np.asarray(fold["train_idx"])).size:
            raise AssertionError("Final holdout entered an outer training fold.")
        if np.intersect1d(final_test_idx, np.asarray(fold["validation_idx"])).size:
            raise AssertionError("Final holdout entered an outer validation fold.")


    discovery = _oldest_complete_train(
        prepared,
        _inner_selection_outer_fold(outer_folds),
        config.cohort_date_col,
    )
    inner_folds, inner_holdout = modeling_utils.build_expanding_folds(
        discovery,
        cohort_col=config.cohort_date_col,
        target_check_col=config.target_check_col,
        target_col=config.target_col,
        validation_folds=INNER_VALIDATION_FOLDS,
        validation_months=INNER_VALIDATION_MONTHS,
        min_train_months=MIN_INNER_TRAIN_MONTHS,
        holdout_months=0,
        label_buffer_days=config.label_availability_buffer_days,
    )
    if len(inner_holdout):
        raise AssertionError("Modeling preparation unexpectedly created an inner holdout.")
    inner_fold_frame = modeling_utils.fold_definitions_frame(
        discovery,
        inner_folds,
        cohort_col=config.cohort_date_col,
        target_check_col=config.target_check_col,
        target_col=config.target_col,
    )


    # Learn feature-quality eligibility from the oldest inner training
    # partition only. Later inner-validation months must not influence the
    # missingness, cardinality, or constant-feature decisions.
    quality_population = discovery.loc[
        np.asarray(inner_folds[0]["train_idx"])
    ].copy()
    quality_training = modeling_utils.proportional_month_target_sample(
        quality_population,
        max_rows=QUALITY_SCREEN_MAX_ROWS,
        cohort_col=config.cohort_date_col,
        target_col=config.target_col,
        random_seed=SEED,
    )
    screened, quality_report = modeling_utils.quality_screen_candidates(
        quality_training,
        candidates,
        feature_types,
        missing_drop_threshold=MISSING_DROP_THRESHOLD,
        dominant_level_drop_threshold=DOMINANT_LEVEL_DROP_THRESHOLD,
        high_cardinality_ratio=HIGH_CARDINALITY_RATIO,
        max_category_levels=MAX_CATEGORY_LEVELS,
    )
    if not screened:
        raise ValueError("Generic quality screening removed every candidate.")


    screened_types = {feature: feature_types[feature] for feature in screened}
    routed_screened: dict[str, list[str]] = {}
    routed_screened_types: dict[str, dict[str, str]] = {}
    quality_routing_rows: list[dict[str, str]] = []
    family_quality_reports: list[pd.DataFrame] = []
    for family, result in routed.items():
        basis_candidates, basis_types = _family_quality_screen_basis(
            result, feature_types
        )
        family_screened_basis, family_quality_report = (
            modeling_utils.quality_screen_candidates(
                quality_training,
                basis_candidates,
                basis_types,
                missing_drop_threshold=MISSING_DROP_THRESHOLD,
                dominant_level_drop_threshold=DOMINANT_LEVEL_DROP_THRESHOLD,
                high_cardinality_ratio=HIGH_CARDINALITY_RATIO,
                max_category_levels=MAX_CATEGORY_LEVELS,
            )
        )
        family_quality_report.insert(0, "model_family", family)
        family_quality_reports.append(family_quality_report)
        family_screened, family_types, family_quality_rows = (
            _quality_screen_routed_candidates(
                result,
                family_screened_basis,
                family_quality_report,
            )
        )
        routed_screened[family] = family_screened
        routed_screened_types[family] = family_types
        quality_routing_rows.extend(family_quality_rows)


    routing_audit = pd.DataFrame(
        [dict(row) for result in routed.values() for row in result.audit_rows]
    )
    quality_routing_audit = pd.DataFrame(quality_routing_rows)
    routing_audit = routing_audit.merge(
        quality_routing_audit,
        on=["model_family", "selected_feature"],
        how="left",
        validate="one_to_one",
    )
    _atomic_csv(OUTPUT_DIR / "candidate_audit.csv", audit)
    _atomic_csv(OUTPUT_DIR / "quality_screen.csv", quality_report)
    _atomic_csv(
        OUTPUT_DIR / "model_family_quality_screen.csv",
        pd.concat(family_quality_reports, ignore_index=True),
    )
    _atomic_csv(OUTPUT_DIR / ROUTING_AUDIT_FILENAME, routing_audit)
    outer_fold_frame = modeling_utils.fold_definitions_frame(
        prepared,
        outer_folds,
        cohort_col=config.cohort_date_col,
        target_check_col=config.target_check_col,
        target_col=config.target_col,
    )
    _atomic_csv(OUTPUT_DIR / "outer_fold_definitions.csv", outer_fold_frame)
    _atomic_csv(OUTPUT_DIR / "inner_fold_definitions.csv", inner_fold_frame)
    _atomic_text(
        OUTPUT_DIR / "eligible_candidate_pool.txt",
        "\n".join(candidates) + "\n",
    )
    _atomic_text(
        OUTPUT_DIR / "quality_screened_candidate_pool.txt",
        "\n".join(screened) + "\n",
    )
    _atomic_json(OUTPUT_DIR / "feature_types.json", screened_types)
    for family, result in routed.items():
        _atomic_text(
            OUTPUT_DIR / ELIGIBLE_POOL_FILES[family],
            "\n".join(result.candidates) + "\n",
        )
        _atomic_text(
            OUTPUT_DIR / QUALITY_SCREENED_POOL_FILES[family],
            "\n".join(routed_screened[family]) + "\n",
        )
        _atomic_json(
            OUTPUT_DIR / FEATURE_TYPES_FILES[family],
            routed_screened_types[family],
        )


    final_dates = pd.to_datetime(
        prepared.loc[np.asarray(final_test_idx), config.cohort_date_col]
    )
    _atomic_json(
        info_path,
        {
            "pipeline_run_id": RUN_ID,
            "profile": PROFILE_PAYLOAD,
            "profile_hash": PROFILE_HASH,
            "profile_slug": PROFILE_SLUG,
            "profile_label": PROFILE_LABEL,
            "input": str(data_path),
            "input_sha256": _sha256(data_path),
            "input_metadata": str(SNAPSHOT_METADATA.resolve()),
            "input_metadata_sha256": _sha256(SNAPSHOT_METADATA),
            "feature_contract": str(FEATURE_CONTRACT),
            "feature_contract_sha256": _sha256(FEATURE_CONTRACT),
            "modeling_utils": str(MODELING_UTILS_FILE),
            "modeling_utils_sha256": _sha256(MODELING_UTILS_FILE),
            "feature_routing": str(FEATURE_ROUTING_FILE),
            "feature_routing_sha256": _sha256(FEATURE_ROUTING_FILE),
            "feature_variant_map": str(FEATURE_VARIANT_MAP_FILE),
            "feature_variant_map_sha256": _sha256(FEATURE_VARIANT_MAP_FILE),
            "state_grouping": str(STATE_GROUPING_FILE),
            "state_grouping_sha256": _sha256(STATE_GROUPING_FILE),
            "preparation_script": str(PREPARATION_SCRIPT_FILE),
            "preparation_script_sha256": _sha256(PREPARATION_SCRIPT_FILE),
            "eligible_before_quality_screen": len(candidates),
            "eligible_after_quality_screen": len(screened),
            "model_family_candidate_counts": {
                family: {
                    "eligible": len(routed[family].candidates),
                    "eligible_after_quality_screen": len(
                        routed_screened[family]
                    ),
                }
                for family in feature_routing.MODEL_FAMILIES
            },
            "model_family_artifacts": {
                family: {
                    "eligible_pool": str(
                        (OUTPUT_DIR / ELIGIBLE_POOL_FILES[family]).resolve()
                    ),
                    "quality_screened_pool": str(
                        (
                            OUTPUT_DIR
                            / QUALITY_SCREENED_POOL_FILES[family]
                        ).resolve()
                    ),
                    "feature_types": str(
                        (OUTPUT_DIR / FEATURE_TYPES_FILES[family]).resolve()
                    ),
                }
                for family in feature_routing.MODEL_FAMILIES
            },
            "feature_routing_audit": str(
                (OUTPUT_DIR / ROUTING_AUDIT_FILENAME).resolve()
            ),
            "quality_screen_source_rows": int(len(quality_population)),
            "quality_screen_sample_rows": int(len(quality_training)),
            "quality_screen_scope": "oldest_inner_training_partition_only",
            "outer_fold_count": len(outer_folds),
            "inner_fold_count": len(inner_folds),
            "final_holdout_rows_identified": int(len(final_test_idx)),
            "final_holdout_start": str(final_dates.min().date()),
            "final_holdout_end": str(final_dates.max().date()),
            "final_holdout_rows_scored": 0,
            "split_settings": _split_settings(),
            "split_summary": split_summary,
        },
    )
    print("MODELING DATA PREPARATION COMPLETE")
    print(f"Eligible before quality screen: {len(candidates):,}")
    print(f"Eligible after quality screen:  {len(screened):,}")
    for family in feature_routing.MODEL_FAMILIES:
        print(
            f"{family.title()} routed pool: "
            f"{len(routed[family].candidates):,} eligible; "
            f"{len(routed_screened[family]):,} after quality screen"
        )
    print(f"Final holdout rows scored:       0 of {len(final_test_idx):,}")
    print(f"Outputs: {OUTPUT_DIR}")




if __name__ == "__main__":
    main()
