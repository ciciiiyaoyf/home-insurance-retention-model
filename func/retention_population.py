# -*- coding: utf-8 -*-
"""Profile-driven population validation and chronological split utilities."""


from __future__ import annotations


from dataclasses import dataclass
from typing import Any


import numpy as np
import pandas as pd


from func.tenure_profile import (
    ALL_TENURES,
    MIXED_TENURES,
    NEW_BUSINESS,
    RENEWAL,
)


from params import (
    ACTIVE_AT_MONTH_1_COL,
    ACTIVE_AT_MONTH_13_COL,
    ANCHOR_TENURE_COL,
    ANCHOR_TERM_END_COL,
    ANCHOR_TERM_START_COL,
    BUSINESS_SCOPE,
    COHORT_DATE_COL,
    EXPOSURE_END_COL,
    EXPOSURE_START_COL,
    EXPOSURE_YEAR_COL,
    LABEL_AVAILABILITY_BUFFER_DAYS,
    LATEST_ANCHOR_TERM_START_DATE_EXCLUSIVE,
    MAX_ANNUAL_TERM_DAYS,
    MIN_ANNUAL_TERM_DAYS,
    POLICY_ID_COL,
    POLICY_INCEPTION_COL,
    PROFILE_HASH,
    PROFILE_SLUG,
    RETAINED_LABEL,
    STRICT_DATASET_CONTRACT,
    TARGET,
    TARGET_CHECK_COL,
    TENURE_COL,
    TENURE_GROUP_DOMAIN,
    TENURE_GROUPS,
    TERM_END_COL,
    TERM_START_COL,
    TRUSTED_DATA_AS_OF_DATE,
)




@dataclass(frozen=True)
class ModelingDataConfig:
    """Target, term-level key, tenure scope, maturity, and split contract."""


    target_col: str = TARGET
    policy_id_col: str = POLICY_ID_COL
    policy_inception_col: str = POLICY_INCEPTION_COL
    anchor_term_start_col: str = ANCHOR_TERM_START_COL
    anchor_term_end_col: str = ANCHOR_TERM_END_COL
    anchor_tenure_col: str = ANCHOR_TENURE_COL
    active_at_month1_col: str = ACTIVE_AT_MONTH_1_COL
    active_at_month13_col: str = ACTIVE_AT_MONTH_13_COL
    cohort_date_col: str = COHORT_DATE_COL
    target_check_col: str = TARGET_CHECK_COL
    term_start_col: str = TERM_START_COL
    term_end_col: str = TERM_END_COL
    exposure_start_col: str = EXPOSURE_START_COL
    exposure_end_col: str = EXPOSURE_END_COL
    exposure_year_col: str = EXPOSURE_YEAR_COL
    tenure_col: str = TENURE_COL
    business_scope: str = BUSINESS_SCOPE
    tenure_group_domain: tuple[str, ...] = TENURE_GROUP_DOMAIN
    tenure_groups: tuple[str, ...] = TENURE_GROUPS
    profile_slug: str = PROFILE_SLUG
    profile_hash: str = PROFILE_HASH
    trusted_data_as_of_date: str = TRUSTED_DATA_AS_OF_DATE
    latest_anchor_term_start_date_exclusive: str = (
        LATEST_ANCHOR_TERM_START_DATE_EXCLUSIVE
    )
    retained_label: int = RETAINED_LABEL
    rolling_validation_folds: int = 5
    validation_months: int = 3
    test_months: int = 6
    min_train_months: int = 12
    label_availability_buffer_days: int = LABEL_AVAILABILITY_BUFFER_DAYS
    strict_dataset_contract: bool = STRICT_DATASET_CONTRACT


    def __post_init__(self) -> None:
        if self.retained_label != 1:
            raise ValueError("RETENTION_13_1_IND is locked to 1=retained.")
        domain = tuple(str(group).strip() for group in self.tenure_group_domain)
        if domain != TENURE_GROUP_DOMAIN:
            raise ValueError(
                "tenure_group_domain must equal the canonical ordered domain "
                f"{TENURE_GROUP_DOMAIN}; received {domain}."
            )
        groups = tuple(str(group).strip() for group in self.tenure_groups)
        if not groups:
            raise ValueError("tenure_groups cannot be empty.")
        if len(groups) != len(set(groups)):
            raise ValueError("tenure_groups cannot contain duplicates.")
        unknown = [group for group in groups if group not in domain]
        if unknown:
            raise ValueError(
                f"tenure_groups contains unknown group(s) {unknown}; "
                f"expected a subset of {domain}."
            )
        canonical_groups = tuple(group for group in domain if group in groups)
        if groups != canonical_groups:
            raise ValueError(
                "tenure_groups must be supplied in canonical domain order; "
                f"received {groups}."
            )
        object.__setattr__(self, "tenure_group_domain", domain)
        object.__setattr__(self, "tenure_groups", groups)
        if self.business_scope == NEW_BUSINESS:
            if groups != ("0",):
                raise ValueError("NEW_BUSINESS must be exactly tenure group '0'.")
        elif self.business_scope == RENEWAL:
            if "0" in groups:
                raise ValueError("RENEWAL cannot contain tenure group '0'.")
        elif self.business_scope == ALL_TENURES:
            if groups != domain:
                raise ValueError(
                    "ALL_TENURES must contain exactly the five canonical tenure "
                    f"groups {domain}."
                )
        elif self.business_scope == MIXED_TENURES:
            if "0" not in groups or len(groups) < 2 or groups == domain:
                raise ValueError(
                    "MIXED_TENURES must contain group '0' and at least one "
                    "renewal group, without selecting the complete ALL domain."
                )
        else:
            raise ValueError(f"Unknown business_scope={self.business_scope!r}.")
        positive = {
            "rolling_validation_folds": self.rolling_validation_folds,
            "validation_months": self.validation_months,
            "test_months": self.test_months,
            "min_train_months": self.min_train_months,
        }
        invalid = [name for name, value in positive.items() if int(value) < 1]
        if invalid:
            raise ValueError(f"Temporal split values must be positive: {invalid}")
        if self.label_availability_buffer_days < 0:
            raise ValueError("label_availability_buffer_days cannot be negative.")


    @property
    def record_key_cols(self) -> tuple[str, str]:
        return (
            self.policy_id_col,
            self.anchor_term_start_col,
        )


    def tenure_in_scope(self, tenure_group: pd.Series) -> pd.Series:
        return tenure_group.isin(self.tenure_groups)




def _date_series(frame: pd.DataFrame, column: str) -> pd.Series:
    raw = frame[column]
    converted = pd.to_datetime(raw, errors="coerce")
    invalid = raw.notna() & raw.astype("string").str.strip().ne("") & converted.isna()
    if invalid.any():
        examples = raw.loc[invalid].astype(str).drop_duplicates().head(5).tolist()
        raise ValueError(f"{column} contains invalid dates; examples: {examples}")
    return converted.dt.normalize()




def _integer_series(
    frame: pd.DataFrame,
    column: str,
    *,
    allow_null: bool = False,
) -> pd.Series:
    raw = frame[column]
    numeric = pd.to_numeric(raw, errors="coerce")
    invalid = numeric.lt(0) | (numeric.notna() & numeric.mod(1).ne(0))
    if not allow_null:
        invalid |= numeric.isna()
    if invalid.any():
        examples = raw.loc[invalid].astype(str).drop_duplicates().head(5).tolist()
        raise ValueError(
            f"{column} must contain nonnegative integers; examples: {examples}"
        )
    return numeric.astype("Int64" if allow_null else "int64")




def _tenure_group_series(
    frame: pd.DataFrame,
    column: str,
    domain: tuple[str, ...],
) -> pd.Series:
    raw = frame[column]
    groups = raw.astype("string").str.strip()
    invalid = groups.isna() | ~groups.isin(domain)
    if invalid.any():
        examples = raw.loc[invalid].astype(str).drop_duplicates().head(5).tolist()
        raise ValueError(
            f"{column} must contain only canonical tenure groups {domain}; "
            f"examples: {examples}"
        )
    return groups




def _coerce_binary(series: pd.Series, column: str, *, require_both: bool) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{column} contains {int(series.isna().sum()):,} nulls.")
    numeric = pd.to_numeric(series, errors="raise")
    invalid = ~numeric.isin([0, 1])
    if invalid.any():
        examples = numeric.loc[invalid].drop_duplicates().head(10).tolist()
        raise ValueError(f"{column} must be 0/1; examples: {examples}")
    result = numeric.astype(int)
    if require_both and set(result.unique()) != {0, 1}:
        raise ValueError(f"{column} must contain both 0 and 1.")
    return result




def _canonicalize_keys(
    frame: pd.DataFrame, config: ModelingDataConfig
) -> pd.DataFrame:
    policy_raw = frame[config.policy_id_col]
    if policy_raw.isna().any():
        raise ValueError(f"{config.policy_id_col} contains nulls.")
    policy = policy_raw.astype("string").str.strip()
    if (policy.isna() | policy.eq("")).any():
        raise ValueError(f"{config.policy_id_col} contains blank identifiers.")
    frame[config.policy_id_col] = policy
    for column in (
        config.policy_inception_col,
        config.anchor_term_start_col,
    ):
        frame[column] = _date_series(frame, column)
        if frame[column].isna().any():
            raise ValueError(f"{column} contains nulls.")
    return frame




def prepare_population(
    frame: pd.DataFrame,
    config: ModelingDataConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate one already-scoped physical-table extract without repairing it."""


    required = {
        config.target_col,
        *config.record_key_cols,
        config.anchor_term_end_col,
        config.anchor_tenure_col,
        config.active_at_month1_col,
        config.active_at_month13_col,
        config.cohort_date_col,
        config.target_check_col,
        config.term_start_col,
        config.term_end_col,
        config.exposure_start_col,
        config.exposure_end_col,
        config.exposure_year_col,
        config.tenure_col,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Required population columns are missing: {missing}")
    if frame.empty:
        raise ValueError("The profile extract contains zero rows.")


    work = frame.copy(deep=False)
    audit: dict[str, Any] = {
        "input_rows": int(len(work)),
        "profile_slug": config.profile_slug,
        "profile_hash": config.profile_hash,
        "business_scope": config.business_scope,
        "tenure_group_domain": list(config.tenure_group_domain),
        "selected_tenure_groups": list(config.tenure_groups),
    }
    work[config.target_col] = _coerce_binary(
        work[config.target_col], config.target_col, require_both=True
    )
    work[config.active_at_month1_col] = _coerce_binary(
        work[config.active_at_month1_col],
        config.active_at_month1_col,
        require_both=False,
    )
    work[config.active_at_month13_col] = _coerce_binary(
        work[config.active_at_month13_col],
        config.active_at_month13_col,
        require_both=False,
    )
    anchor_tenure_group = _tenure_group_series(
        work,
        config.anchor_tenure_col,
        config.tenure_group_domain,
    )
    representative_tenure_group = _tenure_group_series(
        work,
        config.tenure_col,
        config.tenure_group_domain,
    )
    exposure_year = _integer_series(
        work, config.exposure_year_col, allow_null=True
    )
    work[config.anchor_tenure_col] = anchor_tenure_group
    work[config.tenure_col] = representative_tenure_group
    work[config.exposure_year_col] = exposure_year


    out_of_scope = ~config.tenure_in_scope(anchor_tenure_group)
    audit["rows_outside_requested_tenure_group_scope"] = int(
        out_of_scope.sum()
    )
    if out_of_scope.any():
        examples = [
            group
            for group in config.tenure_group_domain
            if group in set(anchor_tenure_group.loc[out_of_scope])
        ]
        raise ValueError(
            f"Extract contains {int(out_of_scope.sum()):,} rows outside profile "
            f"{config.profile_slug}; observed tenure groups include {examples}."
        )
    raw_group_counts = anchor_tenure_group.value_counts()
    group_row_counts = {
        group: int(raw_group_counts[group])
        for group in config.tenure_group_domain
        if group in raw_group_counts.index
    }
    observed_groups = list(group_row_counts)
    audit["tenure_group_row_counts"] = group_row_counts
    audit["observed_tenure_groups"] = observed_groups
    missing_requested = [
        group for group in config.tenure_groups if group not in group_row_counts
    ]
    audit["missing_requested_tenure_groups"] = missing_requested
    if missing_requested:
        raise ValueError(
            "The extracted population has no rows for selected tenure "
            f"group(s): {missing_requested}."
        )
    if (
        config.business_scope == ALL_TENURES
        and tuple(observed_groups) != config.tenure_group_domain
    ):
        raise ValueError(
            "The ALL_TENURES profile requires exactly the five canonical "
            f"groups {config.tenure_group_domain}; observed {observed_groups}."
        )


    cohort = _date_series(work, config.cohort_date_col)
    target_check = _date_series(work, config.target_check_col)
    anchor_start = _date_series(work, config.anchor_term_start_col)
    anchor_end = _date_series(work, config.anchor_term_end_col)
    representative_start = _date_series(work, config.term_start_col)
    representative_end = _date_series(work, config.term_end_col)
    feature_start = _date_series(work, config.exposure_start_col)
    feature_end = _date_series(work, config.exposure_end_col)
    work[config.cohort_date_col] = cohort
    work[config.target_check_col] = target_check
    work[config.anchor_term_end_col] = anchor_end


    expected_target = work[config.active_at_month13_col]
    term_days = (anchor_end - anchor_start).dt.days
    checks = {
        "Month-1 date rule": (
            cohort.isna()
            | anchor_start.isna()
            | cohort.ne(anchor_start + pd.Timedelta(days=30))
        ),
        "Month-13 date rule": (
            target_check.isna()
            | anchor_end.isna()
            | target_check.ne(anchor_end + pd.Timedelta(days=30))
        ),
        "representative term-start rule": (
            representative_start.isna() | representative_start.ne(anchor_start)
        ),
        "representative term-end rule": (
            representative_end.isna() | representative_end.ne(anchor_end)
        ),
        "representative tenure-group rule": representative_tenure_group.ne(
            anchor_tenure_group
        ),
        "Month-1 active rule": work[config.active_at_month1_col].ne(1),
        "target rule": work[config.target_col].ne(expected_target),
        "fixed annual horizon rule": (
            term_days.isna()
            | term_days.lt(MIN_ANNUAL_TERM_DAYS)
            | term_days.gt(MAX_ANNUAL_TERM_DAYS)
        ),
        "feature active at Month-1 rule": (
            feature_start.isna()
            | feature_start.gt(cohort)
            | feature_end.isna()
            | feature_end.lt(cohort)
        ),
        "feature exposure chronology rule": (
            feature_start.notna() & feature_end.notna() & feature_end.lt(feature_start)
        ),
        "checkpoint chronology rule": (
            cohort.notna() & target_check.notna() & target_check.le(cohort)
        ),
        "anchor term chronology rule": (
            anchor_start.notna() & anchor_end.notna() & anchor_end.lt(anchor_start)
        ),
        "target maturity rule": (
            target_check.isna()
            | target_check.gt(pd.Timestamp(config.trusted_data_as_of_date))
        ),
        "latest anchor-start rule": (
            anchor_start.isna()
            | anchor_start.ge(
                pd.Timestamp(config.latest_anchor_term_start_date_exclusive)
            )
        ),
    }
    for name, mask in checks.items():
        audit[name] = int(mask.sum())
    failed = {name: int(mask.sum()) for name, mask in checks.items() if mask.any()}
    if failed and config.strict_dataset_contract:
        raise ValueError(
            "The profile extract violates its population contract: "
            + "; ".join(f"{name}={count:,}" for name, count in failed.items())
        )


    # Keep explicit diagnostics in addition to the strict point-in-time gate.
    audit["feature_rows_starting_after_month1"] = int(
        (feature_start.isna() | feature_start.gt(cohort)).sum()
    )
    audit["representative_exposure_timing_is_exclusion"] = True
    audit["exposure_year_null_rows"] = int(exposure_year.isna().sum())


    work = _canonicalize_keys(work, config)
    duplicate_mask = work.duplicated(list(config.record_key_cols), keep=False)
    audit["duplicate_key_rows"] = int(duplicate_mask.sum())
    if duplicate_mask.any():
        examples = (
            work.loc[duplicate_mask, list(config.record_key_cols)]
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "Population must contain one row per adjusted policy contract/anchor "
            "term. "
            f"Found {int(duplicate_mask.sum()):,} duplicate rows; examples: {examples}"
        )


    audit["eligible_model_rows"] = int(len(work))
    audit["retention_rate"] = float(work[config.target_col].mean())
    audit["target_definition"] = (
        "1=covered at anchor term end + 30 days; 0=not covered, conditional "
        "on survival through anchor term start + 30 days"
    )
    audit["rewrite_semantics"] = (
        "continuity may cross inception records within the same adjusted policy "
        "contract; a different contract remains unlinked unless a trusted "
        "predecessor/successor key is added"
    )
    return work, audit




def prepare_nb_population(
    frame: pd.DataFrame,
    config: ModelingDataConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Backward-compatible function name; validation is profile-driven."""


    return prepare_population(frame, config)




def _split_summary(
    frame: pd.DataFrame,
    indices: np.ndarray,
    config: ModelingDataConfig,
) -> dict[str, Any]:
    dates = pd.to_datetime(frame.loc[indices, config.cohort_date_col])
    target = frame.loc[indices, config.target_col]
    return {
        "rows": int(len(indices)),
        "cohort_start": str(dates.min().date()),
        "cohort_end": str(dates.max().date()),
        "cohort_months": int(dates.dt.to_period("M").nunique()),
        "retention_rate": float(target.mean()),
    }




def rolling_time_splits(
    frame: pd.DataFrame,
    config: ModelingDataConfig,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    """Build expanding OOT folds with Month-13 label-availability embargoes."""


    cohort = pd.to_datetime(
        frame[config.cohort_date_col], errors="raise"
    ).dt.to_period("M")
    months = pd.Index(sorted(cohort.unique()))
    required_months = (
        config.min_train_months
        + config.rolling_validation_folds * config.validation_months
        + config.test_months
    )
    if len(months) < required_months:
        raise ValueError(
            f"Need at least {required_months} cohort months; found {len(months)}. "
            "Do not replace the chronological design with a random split."
        )


    final_months = months[-config.test_months :]
    development_months = months[: -config.test_months]
    final_idx = frame.index[cohort.isin(final_months)].to_numpy()
    target_check = pd.to_datetime(frame[config.target_check_col], errors="raise")
    target = frame[config.target_col]


    folds: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    validation_span = config.rolling_validation_folds * config.validation_months
    first_validation = len(development_months) - validation_span


    for offset in range(config.rolling_validation_folds):
        start = first_validation + offset * config.validation_months
        stop = start + config.validation_months
        validation_values = development_months[start:stop]
        validation_start = validation_values[0].to_timestamp(how="start")
        label_cutoff = validation_start - pd.Timedelta(
            days=config.label_availability_buffer_days
        )
        train_mask = cohort.isin(development_months[:start]) & target_check.lt(
            label_cutoff
        )
        validation_mask = cohort.isin(validation_values)
        train_idx = frame.index[train_mask].to_numpy()
        validation_idx = frame.index[validation_mask].to_numpy()


        for name, indices in (("train", train_idx), ("validation", validation_idx)):
            if not len(indices):
                raise ValueError(f"Fold {offset + 1} {name} is empty.")
            if target.loc[indices].nunique() != 2:
                raise ValueError(f"Fold {offset + 1} {name} lacks both classes.")
        train_months = pd.to_datetime(
            frame.loc[train_idx, config.cohort_date_col]
        ).dt.to_period("M").nunique()
        if train_months < config.min_train_months:
            raise ValueError(
                f"Fold {offset + 1} has {train_months} label-available training "
                f"months; need {config.min_train_months}."
            )
        if np.intersect1d(train_idx, validation_idx).size:
            raise AssertionError("A row entered both training and validation.")


        fold = {
            "fold": offset + 1,
            "train_idx": train_idx,
            "validation_idx": validation_idx,
            "validation_months": [str(value) for value in validation_values],
            "label_cutoff": label_cutoff,
        }
        folds.append(fold)
        fold_summaries.append(
            {
                "fold": offset + 1,
                "label_cutoff_exclusive": str(label_cutoff.date()),
                "train": _split_summary(frame, train_idx, config),
                "validation": _split_summary(frame, validation_idx, config),
            }
        )


    final_start = final_months[0].to_timestamp(how="start")
    final_cutoff = final_start - pd.Timedelta(
        days=config.label_availability_buffer_days
    )
    evaluation_train_idx = frame.index[
        cohort.isin(development_months) & target_check.lt(final_cutoff)
    ].to_numpy()
    for name, indices in (
        ("evaluation_train", evaluation_train_idx),
        ("final_holdout", final_idx),
    ):
        if not len(indices) or target.loc[indices].nunique() != 2:
            raise ValueError(f"{name} is empty or lacks both classes.")
    if np.intersect1d(evaluation_train_idx, final_idx).size:
        raise AssertionError("Final holdout overlaps evaluation training rows.")


    summary = {
        "split_rule": (
            "expanding Month-1 cohorts; training Month-13 labels must be "
            "available strictly before each prediction block"
        ),
        "rolling_validation_folds": fold_summaries,
        "evaluation_train_label_cutoff_exclusive": str(final_cutoff.date()),
        "evaluation_train": _split_summary(frame, evaluation_train_idx, config),
        "final_holdout": _split_summary(frame, final_idx, config),
        "final_holdout_scored": False,
        "exposure_year_used_for_split": False,
    }
    return folds, evaluation_train_idx, final_idx, summary
