"""Regenerate the public source contract without importing modeling modules.

Run from any directory with Python 3.10+: python docs/schema/generate_schema.py
Only declared metadata is read; no warehouse or policy records are accessed.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONFIG = ROOT / "b_data_acquisition/config_vault/HOME/b0_home_data_config_V1.0_C2.csv"
BUILD = ROOT / "b_data_acquisition/b0_refresh_all_tenure_table.py"
PARAMS = ROOT / "params.py"
DDL = ROOT / "b_data_acquisition/sql/create_policy_exposure_history.sql"
SOURCE_TABLE = "INSURANCE_ANALYTICS.RETENTION.POLICY_EXPOSURE_HISTORY"


def declared_constants(path: Path) -> dict:
    """Read simple literal declarations, never execute the application code."""
    known: dict = {}

    def resolve(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return known[node.id]
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            values = []
            for item in node.elts:
                if isinstance(item, ast.Starred):
                    values.extend(resolve(item.value))
                else:
                    values.append(resolve(item))
            return {ast.Tuple: tuple, ast.List: list, ast.Set: set}[type(node)](values)
        if isinstance(node, ast.Dict):
            return {resolve(k): resolve(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            allowed = {"frozenset": frozenset, "tuple": tuple}
            if node.func.id in allowed and len(node.args) == 1 and not node.keywords:
                return allowed[node.func.id](resolve(node.args[0]))
        raise ValueError("Not a supported literal declaration")

    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = resolve(node.value)
        except (ValueError, KeyError, TypeError):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                known[target.id] = value
    return known


EXTRA_INPUTS = {
    "TENURE_YEARS_INT": (
        "float64", "continuous", "Legacy numeric tenure compatibility value; categorical tenure remains authoritative."
    ),
    "EARNED_EXPOSURE_VALUE": (
        "float64", "measurement", "Exposure fraction used as the denominator of the derived premium proxy."
    ),
    "TOT_ERND_PREM_AMT": (
        "float64", "measurement", "Earned premium amount used as the numerator of the derived premium proxy."
    ),
}

DESCRIPTION_OVERRIDES = {
    "POLICYHOLDER_ID": "Opaque policyholder identifier; only artificial identifiers belong in the demo.",
    "EXPOSURE_ROW_ID": "Unique opaque exposure-row identifier; may contain letters and digits.",
    "MPD_HME_ROC_EXP_SK": "Legacy opaque exposure mapping key retained for schema completeness.",
    "POLICY_CONTRACT_ID": "Opaque contract identifier linking exposure intervals and successive terms.",
    "POLICY_ID": "Opaque policy record identifier.",
    "PNUM_IND": "Legacy policy identifier field retained as an opaque value.",
    "RAND_INTEGER": "Legacy randomization key retained as an opaque value; not a feature.",
    "POLICY_SOURCE_ID": "Opaque policy source-system identifier.",
    "HOUSEHOLD_ID": "Opaque household identifier.",
    "ADDRESS_ID": "Opaque address identifier; no actual address is supplied.",
    "POLICY_TERM_END_DATE": "Scheduled policy-term end date, distinct from actual coverage termination.",
    "POLICY_TERM_START_DATE": "Scheduled policy-term start date used to define contract-term anchors.",
    "EXPOSURE_END_DATE": "Actual exposure interval end date used to reconstruct activity and future outcomes.",
    "EXPOSURE_START_DATE": "Exposure interval start date used to reconstruct coverage activity.",
    "ZIP_CD": "Postal code represented as text to preserve leading zeros; not an approved feature.",
    "CLUE_CCA_NH": "Configured prior-insurance history category.",
    "EARLY_SHOP_IND_COMB_V2": "Early-shopping category combining a discount indicator with quote-history timing.",
    "FDR_HOME_HOME_IND": "Multiple-home indicator based on the source household grouping.",
    "MDM_HOME_HOME_IND2": "Multiple-home indicator based on all linked household members.",
}


def generic_description(name: str, row: dict) -> str:
    description = DESCRIPTION_OVERRIDES.get(name, row["description"] or f"Configured {row['var_type']} field {name}.")
    description = re.sub(r"\bNearmap\b", "Property imagery", description, flags=re.IGNORECASE)
    description = re.sub(r"\bISO\b", "Territory reference", description)
    return description


def public_type(name: str, dtype: str, var_type: str) -> tuple[str, str]:
    if var_type == "key" or name.endswith("_ID") or name == "ZIP_CD":
        return "VARCHAR", "Opaque identifier or postal code; preserve letters and leading zeros."
    if var_type == "date" or dtype.startswith("datetime") or name.endswith("_DATE"):
        return "DATE", "Calendar date; timestamps are not required by this contract."
    if dtype == "string":
        return "VARCHAR", "Categorical text, including numeric-looking labels and bands."
    if name == "EXPOSURE_CALENDAR_YEAR":
        return "NUMBER(38,0)", "Whole calendar year."
    if name in {"COVA_PER_SQFT", "BLDG_ORDN_OR_LAW_CVG_AMT", "NM_PRIM_BLD_2D_CLP"}:
        return "FLOAT", "Amount, ratio, or physical measurement; allow fractional values despite legacy integer dtype."
    if name.startswith(("CREDIT_CARD_CNT_", "CREDITDEBIT_CARD_CNT_", "DEBIT_CARD_CNT_")):
        return "NUMBER(38,0)", "Whole payment-event count; declared wider than the original memory dtype."
    if var_type in {"continuous", "measurement"} or dtype.startswith("float"):
        return "FLOAT", "Numeric measurement; allow arithmetic and fractional values without integer truncation."
    if dtype.startswith("int"):
        return "NUMBER(38,0)", "Integer count, flag, or coded category; no narrow integer-width limit."
    raise ValueError(f"Unhandled source type for {name}: {dtype}/{var_type}")


def source_availability(name: str, row: dict, candidate: bool, structural: bool) -> tuple[bool, str, str]:
    if row["var_type"] == "measurement" or name == "EXPOSURE_END_DATE":
        return False, "not_established", "Retrospective measurement or interval end may include future information; used for reconstruction only until timing is verified."
    if candidate:
        return True, "conditional", "Intended Month-1 input only if its source value and every upstream calculation existed by Month 1; configuration alone does not establish availability."
    if structural and name not in {"RAND_INTEGER", "MPD_HME_ROC_EXP_SK"}:
        return True, "conditional_structural", "Intended identifier, calendar, or eligibility input; verify its as-of version at Month 1. Structural use does not permit use as a predictor."
    return False, "not_approved", "Retained for schema completeness; disabled or otherwise not approved for prediction, with no source timing guarantee."


DERIVED_DETAILS = {
    "ANCHOR_TERM_START_DATE": ("structural", True, "Scheduled start of the selected contract-term anchor."),
    "ANCHOR_TERM_END_DATE": ("structural", True, "Scheduled end of the selected contract-term anchor; not the actual cancellation date."),
    "ANCHOR_TENURE_YEARS_GRP": ("structural", True, "Authoritative tenure group for the anchor term."),
    "ANCHOR_TENURE_YEARS_INT": ("compatibility", False, "Passive numeric tenure compatibility output; group selection does not read it."),
    "ANCHOR_TENURE_VALUE_COUNT": ("diagnostic", False, "Count of distinct tenure values across anchor-term source history; retrospective consistency check."),
    "ANCHOR_TERM_END_VALUE_COUNT": ("diagnostic", False, "Count of distinct scheduled term ends across anchor-term history; retrospective consistency check."),
    "ANCHOR_MIN_EXPSR_EFF_DT": ("diagnostic", False, "Minimum exposure start observed for the anchor term; retrospective history diagnostic."),
    "POLICY_MAX_EXPSR_END_EFF_DT": ("outcome_support", False, "Maximum observed exposure end for the contract; includes follow-up beyond prediction."),
    "DED_RAW": ("feature", True, "Raw deductible token taken from the retained Month-1 source row."),
    "DED_PROCESS": ("feature", True, "Numeric deductible amount; percentage deductibles use the same Month-1 dwelling coverage amount."),
    "DED_FINAL": ("feature", True, "Deductible band derived from the Month-1 numeric deductible."),
    "WRITTEN_PREM": ("feature_requires_timing_review", False, "Premium proxy computed as earned premium divided by earned exposure on the selected row, with the implemented nonpositive-exposure rule."),
    "PREV_WRITTEN_PREM": ("feature_requires_timing_review", False, "Previous eligible term's premium proxy; inherits the upstream earned-measurement timing requirement."),
    "RATE_CHANGE": ("feature_requires_timing_review", False, "100 * (current minus previous premium proxy) / current premium proxy; NULL for missing previous or zero current premium."),
    "RATE_CHANGE2": ("feature_requires_timing_review", False, "Same premium change calculation; -100 for missing previous premium, otherwise NULL for zero current premium."),
    "MONTH_1_EVALUATION_DATE": ("structural", True, "Prediction and eligibility date: scheduled anchor-term start plus 30 days."),
    "MONTH_13_EVALUATION_DATE": ("outcome_schedule", False, "Label evaluation date: scheduled anchor-term end plus 30 days. Reserved for outcome maturity and evaluation, not prediction."),
    "NB_MAX_EXPSR_END_EFF_DT": ("legacy_outcome_support", False, "Compatibility alias for the maximum tenure-zero exposure end; may contain future follow-up."),
    "NB_MIN_EXPSR_EFF_DT": ("legacy_diagnostic", False, "Compatibility alias for the minimum tenure-zero exposure start; retained for historical compatibility only."),
    "NB_POLICY_TERM_END_DATE": ("legacy_diagnostic", False, "Compatibility alias for tenure-zero scheduled policy-term end; retained for historical compatibility only."),
    "NB_POLICY_TERM_START_DATE": ("legacy_diagnostic", False, "Compatibility alias for tenure-zero scheduled policy-term start; retained for historical compatibility only."),
    "ACTIVE_AT_MONTH_1_IND": ("eligibility", True, "Month-1 survival indicator; cohort eligibility condition, not a predictor."),
    "ACTIVE_AT_MONTH_13_IND": ("outcome", False, "Whether same-contract exposure covers the Month-13 date; future outcome."),
    "RETENTION_13_1_IND": ("target", False, "Binary retention target: 1 active at Month 13, 0 otherwise, conditional on Month-1 eligibility."),
}


FIELDS = [
    "column", "snowflake_type", "original_config_dtype", "var_type", "role",
    "in_original_config", "config_cv_ind", "feature_candidate_enabled",
    "required_for_build", "required_for_download", "available_at_prediction",
    "availability_status", "requires_upstream_timing_verification", "description",
    "availability_note", "type_note",
]


def write_csv(path: Path, records: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({key: str(value).lower() if isinstance(value, bool) else value for key, value in record.items()})


def main() -> None:
    build = declared_constants(BUILD)
    params = declared_constants(PARAMS)
    with CONFIG.open(encoding="utf-8-sig", newline="") as handle:
        config_rows = list(csv.DictReader(handle))
    config = {row["Column"]: row for row in config_rows}
    if len(config) != len(config_rows):
        raise ValueError("The configuration contains duplicate columns")
    derived_set = set(build["DERIVED_BUILD_COLUMNS"])
    required_build = set(build["BUILD_SOURCE_REQUIRED_COLUMNS"])
    required_download = set(params["REQUIRED_STRUCTURAL_COLUMNS"])
    output_types = build["DERIVED_OUTPUT_TYPES"]
    raw_names = [name for name in config if name not in derived_set]
    raw_names += sorted((required_build | required_download) - set(raw_names) - derived_set)
    if set(output_types) != set(DERIVED_DETAILS):
        raise ValueError("Derived output declarations changed; review descriptions and availability")

    source = []
    for name in raw_names:
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            raise ValueError(f"Unsafe column name: {name}")
        row = config.get(name)
        if row is None:
            dtype, var_type, description = EXTRA_INPUTS[name]
            row = {"dtype": dtype, "var_type": var_type, "description": description, "cv_ind": ""}
        structural = name in required_build | required_download or row["var_type"] == "key" or name.endswith("_ID")
        candidate = row["cv_ind"] == "1"
        role = "structural" if structural else "candidate" if candidate else "disabled"
        dtype, type_note = public_type(name, row["dtype"], row["var_type"])
        available, status, note = source_availability(name, row, candidate, structural)
        source.append({
            "column": name, "snowflake_type": dtype,
            "original_config_dtype": config.get(name, {}).get("dtype", ""),
            "var_type": row["var_type"], "role": role, "in_original_config": name in config,
            "config_cv_ind": row["cv_ind"], "feature_candidate_enabled": candidate,
            "required_for_build": name in required_build, "required_for_download": name in required_download,
            "available_at_prediction": available, "availability_status": status,
            "requires_upstream_timing_verification": available or row["var_type"] == "measurement",
            "description": generic_description(name, row),
            "availability_note": note, "type_note": type_note,
        })

    derived = []
    for name, dtype in output_types.items():
        purpose, available, description = DERIVED_DETAILS[name]
        row = config.get(name, {})
        premium = purpose == "feature_requires_timing_review"
        derived.append({
            "column": name, "snowflake_type": dtype, "original_config_dtype": row.get("dtype", ""),
            "var_type": row.get("var_type", "date" if dtype == "DATE" else "discrete"),
            "role": "derived", "in_original_config": name in config,
            "config_cv_ind": row.get("cv_ind", ""), "feature_candidate_enabled": row.get("cv_ind") == "1",
            "required_for_build": False, "required_for_download": name in required_download or name == params["TARGET"],
            "available_at_prediction": available,
            "availability_status": "conditional" if available else "not_established" if premium else "not_approved",
            "requires_upstream_timing_verification": available or premium, "description": description,
            "availability_note": (
                "Intended Month-1 output only after all contributing values and their as-of versions are verified; no original availability is asserted."
                if available else
                "Earned values can accumulate after Month 1. Verify source snapshots or replace with quoted/written premium known at Month 1 before treating this output as prediction-available."
                if premium else
                "Reserved for labels, retrospective diagnostics, or compatibility; do not use as a prediction feature."
            ),
            "type_note": "Declared DERIVED_OUTPUT_TYPES contract; NUMBER here describes a type family and does not prescribe integer rounding.",
            "derived_purpose": purpose,
        })

    assert required_build <= set(raw_names)
    assert not set(raw_names) & derived_set
    assert not set(raw_names) & set(output_types)
    assert set(config) <= set(raw_names) | set(output_types)
    assert required_download <= set(raw_names) | set(output_types)

    write_csv(HERE / "source_columns.csv", source, FIELDS)
    write_csv(HERE / "model_derived_columns.csv", derived, FIELDS + ["derived_purpose"])
    ddl = [
        "-- Generalized public source contract; no confidential records or observed distributions.",
        "-- Grain: one exposure interval within a policy contract and scheduled term.",
        "-- Provision the database and schema separately; use your own authorized connection.",
        "-- IF NOT EXISTS preserves existing tables and does not migrate an existing schema.",
        "-- Prediction timing and feature approval are documented in docs/schema/README.md.",
        "-- All fields are nullable so sparse synthetic fixtures can leave unused fields empty.",
        f"CREATE TABLE IF NOT EXISTS {SOURCE_TABLE}", "(",
    ]
    for position, row in enumerate(source):
        comment = row["description"].replace("'", "''").replace("\n", " ").replace("\r", " ")
        comma = "," if position < len(source) - 1 else ""
        ddl.append(f"    {row['column']} {row['snowflake_type']} COMMENT '{comment}'{comma}")
    ddl += [")", "COMMENT = 'Generalized exposure-history schema for a retention modeling demonstration; populate only with authorized data.';", ""]
    DDL.write_text("\n".join(ddl), encoding="utf-8")
    artifacts = [HERE / "source_columns.csv", HERE / "model_derived_columns.csv", DDL]
    manifest = {
        "schema_version": "1.0", "source_table": SOURCE_TABLE,
        "config_rows": len(config_rows), "source_columns": len(source),
        "config_columns_in_source": len(set(config) & set(raw_names)),
        "config_columns_recomputed_by_build": sorted(set(config) & derived_set),
        "source_columns_added_for_build": sorted(set(raw_names) - set(config)),
        "derived_output_columns": len(derived),
        "derived_default_output_columns": len(derived) - len(build["OPTIONAL_LEGACY_OUTPUT_COLUMNS"]),
        "optional_legacy_output_columns": sorted(build["OPTIONAL_LEGACY_OUTPUT_COLUMNS"]),
        "fresh_build_model_columns": len(source) + len(derived) - len(build["OPTIONAL_LEGACY_OUTPUT_COLUMNS"]),
        "compatible_model_columns_with_optional_output": len(source) + len(derived),
        "source_role_counts": dict(sorted(Counter(row["role"] for row in source).items())),
        "source_type_counts": dict(sorted(Counter(row["snowflake_type"] for row in source).items())),
        "all_config_columns_covered": True, "source_derived_overlap": [],
        "confidential_records_included": False, "real_distributions_reconstructed": False,
        "availability_claim": "Intended conditional contract only; no original warehouse point-in-time availability has been verified.",
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts},
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("config_rows", "source_columns", "derived_output_columns", "fresh_build_model_columns", "source_role_counts")}, indent=2))


if __name__ == "__main__":
    main()
