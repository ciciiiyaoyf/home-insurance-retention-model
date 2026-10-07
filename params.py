# -*- coding: utf-8 -*-
"""Central runtime parameters for reusable Home 13/1 retention models."""


from __future__ import annotations


from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import os
from pathlib import Path
import re


from func.tenure_profile import (
    PROFILE_HASH_ENV,
    TENURE_DIMENSION,
    TENURE_GROUP_DOMAIN,
    profile_from_environment,
)




def _environment_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().upper()
    if normalized in {"1", "TRUE", "T", "Y", "YES"}:
        return True
    if normalized in {"0", "FALSE", "F", "N", "NO", ""}:
        return False
    raise ValueError(f"{name} must be a Boolean 0/1 or Y/N value; got {value!r}.")




# ---------------------------------------------------------------------------
# Project identity and runtime profile
# ---------------------------------------------------------------------------


PROJECT_ROOT = Path(__file__).resolve().parent
PATH = str(PROJECT_ROOT)
PROJECT_PATH = PROJECT_ROOT
PROJECT = "Home_Retention_13_1"
PROJECT_NAME = PROJECT
PROJECT_VERSION = 12
PROJECT_CONTRACT_VERSION = (
    "HOME_RETENTION_13_1_PUBLIC_V1_MONTH1_GROUPED_TENURE"
)


LOB = os.getenv("LOB", "HOME").strip().upper()
COV = os.getenv("COV", "RET131").strip().upper()
MODEL_TYPE = os.getenv("MODEL_TYPE", "all").strip().lower()
MODEL_VERSION = os.getenv("MODEL_VERSION", "").strip()


EXPECTED_LOB = "HOME"
EXPECTED_COV = "RET131"
ALLOWED_MODEL_TYPES = [
    "all",
    "logistic",
    "xgboost",
    "sglm",
    "iglm",
    "uwtgbm",
]
MODEL_FAMILIES = ("logistic", "xgboost")
if LOB != EXPECTED_LOB:
    raise ValueError(f"This project requires LOB={EXPECTED_LOB}; observed {LOB!r}.")
if COV != EXPECTED_COV:
    raise ValueError(f"This project requires COV={EXPECTED_COV}; observed {COV!r}.")
if MODEL_TYPE not in ALLOWED_MODEL_TYPES:
    raise ValueError(f"MODEL_TYPE={MODEL_TYPE!r} is not in {ALLOWED_MODEL_TYPES}.")


TENURE_PROFILE = profile_from_environment()
PROFILE_PAYLOAD = TENURE_PROFILE.payload
BUSINESS_SCOPE = TENURE_PROFILE.business_scope
TENURE_GROUPS = TENURE_PROFILE.tenure_groups
TENURE_FILTER_MODE = TENURE_PROFILE.filter_mode


# Optional engineering-test scope. Sampling is deterministic at the adjusted
# policy-contract level, so every selected policy keeps all selected terms.
# These controls affect extraction only; they never rebuild the shared CY.
TEST_POLICY_ID_ENV = "RETENTION_TEST_POLICY_ID"
SAMPLE_PERCENT_ENV = "RETENTION_SAMPLE_PERCENT"
POLICY_SAMPLE_SALT = "RET131_POLICY_SAMPLE_V1"
POLICY_SAMPLE_BUCKET_COUNT = 10_000
_POLICY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")




def _population_scope_from_environment() -> dict[str, object]:
    policy_id = os.getenv(TEST_POLICY_ID_ENV, "").strip()
    sample_text = os.getenv(SAMPLE_PERCENT_ENV, "").strip()
    if policy_id and sample_text:
        raise ValueError(
            f"{TEST_POLICY_ID_ENV} and {SAMPLE_PERCENT_ENV} are mutually exclusive."
        )
    if policy_id:
        if not _POLICY_ID_RE.fullmatch(policy_id):
            raise ValueError(
                f"{TEST_POLICY_ID_ENV} must contain 1-128 letters, digits, '.', "
                "'_', or '-' and must start with a letter or digit."
            )
        return {
            "population_scope_mode": "POLICY",
            "test_policy_id": policy_id,
            "sample_basis_points": None,
            "sample_percent": None,
            "policy_sample_salt": None,
        }
    if sample_text:
        try:
            percent = Decimal(sample_text)
        except InvalidOperation as exc:
            raise ValueError(
                f"{SAMPLE_PERCENT_ENV} must be a number such as 2."
            ) from exc
        if not percent.is_finite() or percent < Decimal("0.01") or percent >= 100:
            raise ValueError(
                f"{SAMPLE_PERCENT_ENV} must be at least 0.01 and less than 100."
            )
        quantized = percent.quantize(Decimal("0.01"))
        if quantized != percent:
            raise ValueError(
                f"{SAMPLE_PERCENT_ENV} supports at most two decimal places."
            )
        basis_points = int(quantized * 100)
        return {
            "population_scope_mode": "SAMPLE",
            "test_policy_id": None,
            "sample_basis_points": basis_points,
            "sample_percent": format(quantized, "f"),
            "policy_sample_salt": POLICY_SAMPLE_SALT,
        }
    return {
        "population_scope_mode": "FULL",
        "test_policy_id": None,
        "sample_basis_points": None,
        "sample_percent": None,
        "policy_sample_salt": None,
    }




POPULATION_SCOPE_PAYLOAD = _population_scope_from_environment()
POPULATION_SCOPE_MODE = str(POPULATION_SCOPE_PAYLOAD["population_scope_mode"])
TEST_POLICY_ID = POPULATION_SCOPE_PAYLOAD["test_policy_id"]
SAMPLE_BASIS_POINTS = POPULATION_SCOPE_PAYLOAD["sample_basis_points"]
SAMPLE_PERCENT = POPULATION_SCOPE_PAYLOAD["sample_percent"]


if POPULATION_SCOPE_MODE == "POLICY":
    scope_token = hashlib.sha256(str(TEST_POLICY_ID).encode("utf-8")).hexdigest()[:10]
    POPULATION_SCOPE_SLUG = f"test-policy-{scope_token}"
    POPULATION_SCOPE_LABEL = f"test policy {TEST_POLICY_ID}"
elif POPULATION_SCOPE_MODE == "SAMPLE":
    POPULATION_SCOPE_SLUG = f"sample-{SAMPLE_BASIS_POINTS}bp"
    POPULATION_SCOPE_LABEL = f"deterministic {SAMPLE_PERCENT}% policy sample"
else:
    POPULATION_SCOPE_SLUG = "full-population"
    POPULATION_SCOPE_LABEL = "full selected population"


PROFILE_SLUG = (
    TENURE_PROFILE.slug
    if POPULATION_SCOPE_MODE == "FULL"
    else f"{TENURE_PROFILE.slug}_{POPULATION_SCOPE_SLUG}"
)
PROFILE_LABEL = (
    TENURE_PROFILE.label
    if POPULATION_SCOPE_MODE == "FULL"
    else f"{TENURE_PROFILE.label}; {POPULATION_SCOPE_LABEL}"
)


# The canonical tenure group is both the governed population selector and a
# categorical model candidate. Exact one-group runs legitimately make it
# constant; the training-only quality screen removes it in those runs. The
# exposure calendar year remains structural unless explicitly enabled.
INCLUDE_EXPOSURE_YEAR_AS_PREDICTOR = _environment_bool(
    "RETENTION_INCLUDE_EXPOSURE_YEAR_AS_PREDICTOR", False
)
TENURE_IS_MODEL_CANDIDATE = True
PROFILE_HASH = TENURE_PROFILE.fingerprint(
    PROJECT_CONTRACT_VERSION,
    exposure_year_as_predictor=INCLUDE_EXPOSURE_YEAR_AS_PREDICTOR,
    tenure_as_predictor=TENURE_IS_MODEL_CANDIDATE,
    population_scope=POPULATION_SCOPE_PAYLOAD,
)
_SUPPLIED_PROFILE_HASH = os.getenv(PROFILE_HASH_ENV, "").strip()
if _SUPPLIED_PROFILE_HASH and _SUPPLIED_PROFILE_HASH != PROFILE_HASH:
    raise ValueError(
        f"{PROFILE_HASH_ENV} does not match the selected population/model profile."
    )


REFRESH_ALL_TENURE_TABLE = _environment_bool(
    "RETENTION_REFRESH_ALL_TENURE_TABLE", False
)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S%f")
SEED = 42
check_config = True




# ---------------------------------------------------------------------------
# ROC-style paths and full variable config
# ---------------------------------------------------------------------------


OUTPUT_FOLDER = str(Path("output") / "pipeline_runs")
DATA_PATH = PROJECT_ROOT / "data"
BASE_CONFIG_PATH = PROJECT_ROOT / "b_data_acquisition" / "config_vault" / LOB
CONFIG_PATH = BASE_CONFIG_PATH / MODEL_VERSION if MODEL_VERSION else BASE_CONFIG_PATH
FACTOR_PATH = PROJECT_ROOT / "b_data_acquisition" / "factor_vault" / LOB


DATA_CONFIG_ENV_VAR = "RETENTION_DATA_CONFIG_PATH"




def resolve_data_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve one explicit ROC config without guessing between candidates."""


    supplied = config_path
    if supplied is None:
        supplied = os.getenv(DATA_CONFIG_ENV_VAR, "").strip() or None
    if supplied is not None:
        expanded = Path(os.path.expandvars(str(supplied))).expanduser()
        attempted = (
            [expanded.resolve()]
            if expanded.is_absolute()
            else [
                (Path.cwd() / expanded).resolve(),
                (PROJECT_ROOT / expanded).resolve(),
            ]
        )
        for path in dict.fromkeys(attempted):
            if path.is_file():
                return path
        rendered = ", ".join(str(path) for path in dict.fromkeys(attempted))
        raise FileNotFoundError(f"ROC data config not found. Checked: {rendered}")


    search_roots = [CONFIG_PATH]
    if CONFIG_PATH != BASE_CONFIG_PATH:
        search_roots.append(BASE_CONFIG_PATH)
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        if not root.is_dir():
            continue
        for candidate in sorted(root.rglob("*.csv")):
            resolved = candidate.resolve()
            if resolved not in seen:
                candidates.append(resolved)
                seen.add(resolved)
        if candidates:
            break
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No ROC config CSV was found beneath {CONFIG_PATH.resolve()}."
        )
    rendered = "\n- ".join(str(path) for path in candidates)
    raise RuntimeError(
        "Multiple ROC config CSV files were found. Supply --config or set "
        f"{DATA_CONFIG_ENV_VAR}. Candidates:\n- {rendered}"
    )




# ---------------------------------------------------------------------------
# Snowflake source, physical all-tenure table, and observation contract
# ---------------------------------------------------------------------------


SF_CONN = "SNOWFLAKE"
SP_CONN = "SNOWPARK"
def _warehouse_identifier(environment_name: str, default: str) -> str:
    value = os.getenv(environment_name, default).strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9_$]*", value):
        raise ValueError(f"{environment_name} must be an unquoted SQL identifier.")
    return value


DATABASE = _warehouse_identifier("SNOWFLAKE_DATABASE", "INSURANCE_ANALYTICS")
SCHEMA = _warehouse_identifier("SNOWFLAKE_SCHEMA", "RETENTION")
SOURCE_TABLE = "POLICY_EXPOSURE_HISTORY"
SOURCE_TABLE_FQN = f"{DATABASE}.{SCHEMA}.{SOURCE_TABLE}"
# This all-tenure physical object replaces the former new-business-only CY.
MAIN_TABLE = "POLICY_RETENTION_13_1"
MAIN_TABLE_FQN = f"{DATABASE}.{SCHEMA}.{MAIN_TABLE}"
MAPPING_TABLE = f"{MAIN_TABLE}_C3_2_MAPPING"


ALL_TENURE_BUILD_SQL_PATH = (
    PROJECT_ROOT
    / "b_data_acquisition"
    / "sql"
    / "build_cy_ho_ret_13_1_all_tenure.sql"
)
ALL_TENURE_QA_SQL_PATH = (
    PROJECT_ROOT
    / "b_data_acquisition"
    / "sql"
    / "qa_cy_ho_ret_13_1_all_tenure.sql"
)


TARGET = "RETENTION_13_1_IND"
RETAINED_LABEL = 1
PRED = "P_RETENTION"
CHURN_PRED = "P_CHURN"
SAMPLE_WEIGHT = None
MEASUREMENTS: list[str] = []
THRESHOLD_VAR = None
P_PP_TABLE = None
SAMPLE_VAR = None


INDEX_COL = "EXPOSURE_ROW_ID"
POLICY_ID_COL = "POLICY_CONTRACT_ID"
POLICY_INCEPTION_COL = "POLICY_INCEPTION_DATE"
TERM_START_COL = "POLICY_TERM_START_DATE"
TERM_END_COL = "POLICY_TERM_END_DATE"
EXPOSURE_START_COL = "EXPOSURE_START_DATE"
EXPOSURE_END_COL = "EXPOSURE_END_DATE"
TENURE_COL = "TENURE_YEARS_GRP"
EXPOSURE_YEAR_COL = "EXPOSURE_CALENDAR_YEAR"


ANCHOR_TERM_START_COL = "ANCHOR_TERM_START_DATE"
ANCHOR_TERM_END_COL = "ANCHOR_TERM_END_DATE"
ANCHOR_TENURE_COL = "ANCHOR_TENURE_YEARS_GRP"
ACTIVE_AT_MONTH_1_COL = "ACTIVE_AT_MONTH_1_IND"
ACTIVE_AT_MONTH_13_COL = "ACTIVE_AT_MONTH_13_IND"


RECORD_KEY_COLS = (
    POLICY_ID_COL,
    ANCHOR_TERM_START_COL,
)
COHORT_DATE_COL = "MONTH_1_EVALUATION_DATE"
TARGET_CHECK_COL = "MONTH_13_EVALUATION_DATE"
RAW_STATE_COLUMN = "PROPERTY_STATE"
GROUPED_STATE_PREFIX = "STATE_GRP_"
# Preserve the August production population rule; the tenure change must not
# silently introduce a new state scope.  The physical CY remains all-state so
# this extraction rule can be governed independently.
EXCLUDED_PROPERTY_STATES = ("CA",)


TRAIN_TEST_SPLIT_VAR = COHORT_DATE_COL
TRAIN_FILTER = None


REQUIRED_STRUCTURAL_COLUMNS = (
    INDEX_COL,
    TARGET_CHECK_COL,
    COHORT_DATE_COL,
    ACTIVE_AT_MONTH_1_COL,
    ACTIVE_AT_MONTH_13_COL,
    ANCHOR_TERM_START_COL,
    ANCHOR_TERM_END_COL,
    ANCHOR_TENURE_COL,
    POLICY_ID_COL,
    POLICY_INCEPTION_COL,
    "POLICY_REWRITE_IND",
    TERM_START_COL,
    TERM_END_COL,
    EXPOSURE_START_COL,
    EXPOSURE_END_COL,
    RAW_STATE_COLUMN,
    TENURE_COL,
    EXPOSURE_YEAR_COL,
)
REQUIRED_QUERY_COLUMNS = (TARGET, *REQUIRED_STRUCTURAL_COLUMNS)


TRUSTED_DATA_AS_OF_DATE = "2024-12-31"
LATEST_ANCHOR_TERM_START_DATE_EXCLUSIVE = "2024-01-01"
MIN_ANNUAL_TERM_DAYS = 360
MAX_ANNUAL_TERM_DAYS = 370




# ---------------------------------------------------------------------------
# Run-scoped data products and state
# ---------------------------------------------------------------------------


PIPELINE_OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_FOLDER
PIPELINE_RUN_ID = os.getenv("RETENTION_PIPELINE_RUN_ID", "").strip()
if PIPELINE_RUN_ID:
    RUN_INPUT_ROOT = PIPELINE_OUTPUT_ROOT / PIPELINE_RUN_ID
else:
    RUN_INPUT_ROOT = DATA_PATH / "manual" / PROFILE_SLUG


SNAPSHOT_DIR = RUN_INPUT_ROOT / "input_snapshot"
MODEL_INPUT_DIR = RUN_INPUT_ROOT / "model_inputs"
DOWNLOAD_PATH = SNAPSHOT_DIR / "retention_13_1_population.parquet"
DOWNLOAD_METADATA_PATH = SNAPSHOT_DIR / "retention_13_1_population.metadata.json"
FEATURE_CONTRACT_PATH = MODEL_INPUT_DIR / "feature_contract.csv"
CANDIDATE_POOL_PATH = MODEL_INPUT_DIR / "candidate_pool.txt"
DOWNLOAD_CHUNKSIZE = int(os.getenv("RETENTION_DOWNLOAD_CHUNKSIZE", "100000"))


ACTIVE_PIPELINE_STATE_RELATIVE_PATH = Path("output") / "pipeline_state" / "active_run.json"
ACTIVE_PIPELINE_STATE_PATH = PROJECT_ROOT / ACTIVE_PIPELINE_STATE_RELATIVE_PATH
PIPELINE_OPERATION_LOCK_RELATIVE_PATH = Path("output") / "pipeline_state" / "operation.lock"
PIPELINE_OPERATION_LOCK_PATH = PROJECT_ROOT / PIPELINE_OPERATION_LOCK_RELATIVE_PATH




# ---------------------------------------------------------------------------
# Shared chronological development design
# ---------------------------------------------------------------------------


ROLLING_VALIDATION_FOLDS = 5
OUTER_VALIDATION_MONTHS = 3
FINAL_HOLDOUT_MONTHS = 6
MIN_OUTER_TRAIN_MONTHS = 12
INNER_VALIDATION_FOLDS = 3
INNER_VALIDATION_MONTHS = 3
MIN_INNER_TRAIN_MONTHS = 12
# Governance invariant: candidate discovery is nested inside the oldest outer
# training boundary.  This is deliberately not environment-configurable.
INNER_SELECTION_OUTER_FOLD = 1
LABEL_AVAILABILITY_BUFFER_DAYS = 0
STRICT_DATASET_CONTRACT = True


QUALITY_SCREEN_MAX_ROWS = 100_000
MISSING_DROP_THRESHOLD = 0.95
DOMINANT_LEVEL_DROP_THRESHOLD = 0.995
HIGH_CARDINALITY_RATIO = 0.50
MAX_CATEGORY_LEVELS = 500


XGB_DEVICE = os.getenv("RETENTION_XGB_DEVICE", "cpu")
N_JOBS = int(os.getenv("RETENTION_N_JOBS", "8"))




def _readable_tenure_run_token() -> str:
    """Return a short folder-safe tenure label without changing identity."""


    groups = tuple(str(value) for value in TENURE_GROUPS)
    rendered = tuple(value.replace("+", "PLUS") for value in groups)
    if groups == tuple(TENURE_GROUP_DOMAIN):
        return "ALLTEN"
    if len(rendered) == 1:
        return f"TEN{rendered[0]}"
    positions = [TENURE_GROUP_DOMAIN.index(value) for value in groups]
    if positions == list(range(positions[0], positions[-1] + 1)):
        return f"TEN{rendered[0]}-{rendered[-1]}"
    return "TEN" + "-".join(rendered)




def _readable_population_run_token() -> str:
    """Return FULL/SAMPLE/POLICY scope in business-readable form."""


    if POPULATION_SCOPE_MODE == "FULL":
        return "FULL"
    if POPULATION_SCOPE_MODE == "SAMPLE":
        percent = str(SAMPLE_PERCENT).replace(".", "p")
        return f"S{percent}"
    return POPULATION_SCOPE_SLUG.upper().replace("_", "-")




def _readable_timestamp_token() -> str:
    match = re.fullmatch(r"(\d{8})_(\d{6})(\d{6})", TIMESTAMP)
    if match:
        return "-".join(match.groups())
    return TIMESTAMP.replace("_", "-")




def new_run_id(step_name: str = "a0_one_click") -> str:
    """Create a unique output folder name that states tenure and test scope."""


    raw_step = str(step_name).strip().lower()
    step_aliases = {
        "a0_one_click": "P",
        "b0_one_click": "A",
        "c0_one_click": "M",
        "b1_download_retention_13_1": "D",
        "b0_refresh_all_tenure_table": "R",
    }
    safe_step = step_aliases.get(
        raw_step,
        re.sub(r"[^A-Za-z0-9]+", "", raw_step).upper()[:8],
    )
    return "_".join(
        (
            "RET131",
            _readable_tenure_run_token(),
            _readable_population_run_token(),
            safe_step or "RUN",
            _readable_timestamp_token(),
        )
    )




def profile_env() -> dict[str, str]:
    environment = TENURE_PROFILE.environment(PROFILE_HASH)
    environment["RETENTION_INCLUDE_EXPOSURE_YEAR_AS_PREDICTOR"] = (
        "1" if INCLUDE_EXPOSURE_YEAR_AS_PREDICTOR else "0"
    )
    if TEST_POLICY_ID is not None:
        environment[TEST_POLICY_ID_ENV] = str(TEST_POLICY_ID)
    if SAMPLE_PERCENT is not None:
        environment[SAMPLE_PERCENT_ENV] = str(SAMPLE_PERCENT)
    return environment




def profile_state_payload() -> dict[str, object]:
    return {
        **PROFILE_PAYLOAD,
        "profile_slug": PROFILE_SLUG,
        "profile_hash": PROFILE_HASH,
        "profile_label": PROFILE_LABEL,
        "tenure_is_model_candidate": TENURE_IS_MODEL_CANDIDATE,
        "exposure_year_as_predictor": INCLUDE_EXPOSURE_YEAR_AS_PREDICTOR,
        **POPULATION_SCOPE_PAYLOAD,
    }




def setup_params(run_id: str) -> dict[str, object]:
    return {
        "PROJECT_NAME": PROJECT_NAME,
        "PROJECT_DIR": PATH,
        "RUN_ID": str(run_id),
        "RUN_FOLDER": OUTPUT_FOLDER,
        "LOGGING": True,
        "LOGGING_MODE": "a",
        "SAVE_ARTIFACTS": False,
    }




ls_params = [
    name
    for name in globals()
    if name.isupper()
    or name
    in {
        "check_config",
        "new_run_id",
        "profile_env",
        "profile_state_payload",
        "resolve_data_config_path",
        "setup_params",
    }
]
