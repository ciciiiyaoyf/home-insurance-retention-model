# -*- coding: utf-8 -*-
"""Validated new-business and renewal tenure-group scopes.


The physical Snowflake table contains every eligible anchor term. A profile
controls only the final Snowflake extraction filter and the identity of the
local snapshot/model run; it never removes history used to construct targets.


``TENURE_YEARS_GRP`` is a finite, top-coded categorical dimension. In
particular, ``4+`` is one group containing every actual tenure of four years
or more; it is not an exact fourth-year value.
"""


from __future__ import annotations


from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
import os
import re
from typing import Mapping




BUSINESS_SCOPE_ENV = "RETENTION_BUSINESS_SCOPE"
TENURE_GROUPS_ENV = "RETENTION_TENURE_GROUPS"
TENURE_FILTER_MODE_ENV = "RETENTION_TENURE_FILTER_MODE"
PROFILE_SLUG_ENV = "RETENTION_PROFILE_SLUG"
PROFILE_HASH_ENV = "RETENTION_PROFILE_HASH"
PROFILE_PAYLOAD_ENV = "RETENTION_PROFILE_PAYLOAD_JSON"


# Stale values from the removed integer-tenure contract are rejected by the
# environment parser and cleared by the root launcher.
LEGACY_TENURE_ENV_VARS = (
    "RETENTION_TENURE_MIN",
    "RETENTION_TENURE_MAX",
    "RETENTION_TENURE_VALUES",
)


TENURE_DIMENSION = "TENURE_YEARS_GRP"
TENURE_GROUP_DOMAIN = ("0", "1", "2", "3", "4+")
TENURE_GROUP_INDEX = {
    group: position for position, group in enumerate(TENURE_GROUP_DOMAIN)
}
NEW_BUSINESS_GROUPS = ("0",)
RENEWAL_GROUPS = TENURE_GROUP_DOMAIN[1:]


NEW_BUSINESS = "NEW_BUSINESS"
RENEWAL = "RENEWAL"
ALL_TENURES = "ALL_TENURES"
MIXED_TENURES = "MIXED_TENURES"
ALLOWED_BUSINESS_SCOPES = (
    NEW_BUSINESS,
    RENEWAL,
    ALL_TENURES,
    MIXED_TENURES,
)


EXACT_FILTER = "EXACT"
VALUES_FILTER = "VALUES"
RANGE_FILTER = "RANGE"
OPEN_ENDED_FILTER = "OPEN_ENDED"
ALL_FILTER = "ALL"
ALLOWED_FILTER_MODES = (
    EXACT_FILTER,
    VALUES_FILTER,
    RANGE_FILTER,
    OPEN_ENDED_FILTER,
    ALL_FILTER,
)




def _normalize_scope(value: object) -> str:
    normalized = str(value).strip().upper().replace("-", "_")
    aliases = {
        "NB": NEW_BUSINESS,
        "NEW": NEW_BUSINESS,
        "NEW_BUSINESS": NEW_BUSINESS,
        "RN": RENEWAL,
        "RENEWAL": RENEWAL,
        "ALL": ALL_TENURES,
        "ALL_TENURES": ALL_TENURES,
        "MIXED": MIXED_TENURES,
        "MIXED_TENURES": MIXED_TENURES,
    }
    if normalized not in aliases:
        raise ValueError(
            f"business_scope must be one of {ALLOWED_BUSINESS_SCOPES}; "
            f"received {value!r}."
        )
    return aliases[normalized]




def _normalize_group(value: object, *, field: str) -> str:
    """Validate an already-canonical stored tenure group."""


    if isinstance(value, bool):
        raise ValueError(f"{field} must be one of {TENURE_GROUP_DOMAIN}, not boolean.")
    group = str(value).strip()
    if group not in TENURE_GROUP_INDEX:
        raise ValueError(
            f"{field} must be one of {TENURE_GROUP_DOMAIN}; received {value!r}."
        )
    return group




def _selector_integer(value: object) -> int | None:
    """Return a nonnegative CLI integer, without accepting signs or decimals."""


    if isinstance(value, bool):
        return None
    token = str(value).strip()
    if re.fullmatch(r"[0-9]+", token) is None:
        return None
    return int(token)




def _normalize_selector_group(value: object, *, field: str) -> str:
    """Canonicalize a public selector while keeping stored profiles strict."""


    if isinstance(value, bool):
        raise ValueError(
            f"{field} must be a nonnegative integer or one of "
            f"{TENURE_GROUP_DOMAIN}, not boolean."
        )
    token = str(value).strip()
    if token == "4+":
        return token
    actual_tenure = _selector_integer(token)
    if actual_tenure is None:
        raise ValueError(
            f"{field} must be a nonnegative whole number or '4+'; "
            f"received {value!r}."
        )
    if actual_tenure >= 4:
        return "4+"
    return str(actual_tenure)




def _selector_groups_between(lower: object, upper: object) -> tuple[str, ...]:
    lower_integer = _selector_integer(lower)
    upper_integer = _selector_integer(upper)
    if (
        lower_integer is not None
        and upper_integer is not None
        and upper_integer < lower_integer
    ):
        raise ValueError(
            f"Tenure range end ({upper_integer}) cannot be below start "
            f"({lower_integer})."
        )


    start = _normalize_selector_group(lower, field="tenure range start")
    end = _normalize_selector_group(upper, field="tenure range end")
    start_index = TENURE_GROUP_INDEX[start]
    end_index = TENURE_GROUP_INDEX[end]
    if end_index < start_index:
        raise ValueError(
            f"Tenure range end ({end}) cannot be below start ({start})."
        )
    return TENURE_GROUP_DOMAIN[start_index : end_index + 1]




def _normalize_canonical_groups(value: object) -> tuple[str, ...]:
    """Validate stored/environment group values in canonical domain order."""


    if value is None:
        raise ValueError("tenure groups cannot be empty.")
    if isinstance(value, str):
        raw_values: list[object] = [value]
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        raw_values = list(value)
    else:
        raw_values = [value]
    if not raw_values:
        raise ValueError("tenure groups cannot be empty.")


    parsed: set[str] = set()
    for raw_value in raw_values:
        pieces = str(raw_value).split(",") if isinstance(raw_value, str) else [raw_value]
        if any(not str(piece).strip() for piece in pieces):
            raise ValueError(
                "--tenures contains an empty value; use groups such as "
                "--tenures 1 3 4+ or --tenures 1,3,4+."
            )
        for piece in pieces:
            token = str(piece).strip()
            if token.upper() == "ALL":
                raise ValueError("ALL must be the only --tenures value.")
            parsed.add(_normalize_group(token, field="tenure group"))
    if not parsed:
        raise ValueError("tenure groups cannot be empty.")
    return tuple(group for group in TENURE_GROUP_DOMAIN if group in parsed)




def _normalize_selector_values(value: object) -> tuple[str, ...]:
    """Parse public CLI values, including exact-tenure aliases and ranges."""


    if value is None:
        raise ValueError("tenure groups cannot be empty.")
    if isinstance(value, str):
        raw_values: list[object] = [value]
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        raw_values = list(value)
    else:
        raw_values = [value]
    if not raw_values:
        raise ValueError("tenure groups cannot be empty.")


    parsed: set[str] = set()
    for raw_value in raw_values:
        pieces = str(raw_value).split(",") if isinstance(raw_value, str) else [raw_value]
        if any(not str(piece).strip() for piece in pieces):
            raise ValueError(
                "--tenures contains an empty value; use values such as "
                "--tenures 1 7 or --tenures 1,7."
            )
        for piece in pieces:
            token = str(piece).strip()
            if token.upper() == "ALL":
                raise ValueError("ALL must be the only --tenures value.")
            if "-" in token:
                bounds = token.split("-")
                if len(bounds) != 2 or not all(bound.strip() for bound in bounds):
                    raise ValueError(
                        f"Invalid tenure range {token!r}; use a form such as 1-7."
                    )
                parsed.update(_selector_groups_between(bounds[0], bounds[1]))
            else:
                parsed.add(
                    _normalize_selector_group(token, field="tenure selector")
                )
    if not parsed:
        raise ValueError("tenure groups cannot be empty.")
    return tuple(group for group in TENURE_GROUP_DOMAIN if group in parsed)




def _is_consecutive(groups: tuple[str, ...]) -> bool:
    indexes = [TENURE_GROUP_INDEX[group] for group in groups]
    return indexes == list(range(indexes[0], indexes[-1] + 1))




def _scope_for_groups(groups: tuple[str, ...]) -> str:
    if groups == TENURE_GROUP_DOMAIN:
        return ALL_TENURES
    if groups == NEW_BUSINESS_GROUPS:
        return NEW_BUSINESS
    if "0" in groups:
        return MIXED_TENURES
    return RENEWAL




def _slug_group(group: str) -> str:
    return group.replace("+", "plus")




def _sql_literal(group: str) -> str:
    return "'" + group.replace("'", "''") + "'"




@dataclass(frozen=True)
class TenureProfile:
    """One immutable population selected from the finite tenure-group domain."""


    business_scope: str
    tenure_groups: tuple[str, ...]


    def __post_init__(self) -> None:
        scope = _normalize_scope(self.business_scope)
        groups = _normalize_canonical_groups(self.tenure_groups)
        inferred_scope = _scope_for_groups(groups)
        if scope != inferred_scope:
            raise ValueError(
                f"Selected tenure groups belong to {inferred_scope}, not {scope}."
            )
        object.__setattr__(self, "business_scope", scope)
        object.__setattr__(self, "tenure_groups", groups)


    @property
    def filter_mode(self) -> str:
        if self.business_scope == ALL_TENURES:
            return ALL_FILTER
        if len(self.tenure_groups) == 1:
            return EXACT_FILTER
        if self.tenure_groups == RENEWAL_GROUPS:
            return OPEN_ENDED_FILTER
        if _is_consecutive(self.tenure_groups):
            return RANGE_FILTER
        return VALUES_FILTER


    @property
    def slug(self) -> str:
        if self.business_scope == ALL_TENURES:
            return "all_tenures"
        if self.business_scope == NEW_BUSINESS:
            return "nb_g0"
        prefix = "mix" if self.business_scope == MIXED_TENURES else "rn"
        if self.filter_mode == OPEN_ENDED_FILTER:
            return "rn_g1plus"
        if self.filter_mode == RANGE_FILTER:
            return (
                f"{prefix}_g{_slug_group(self.tenure_groups[0])}to"
                f"{_slug_group(self.tenure_groups[-1])}"
            )
        if self.filter_mode == EXACT_FILTER:
            return f"{prefix}_g{_slug_group(self.tenure_groups[0])}"
        joined = "_".join(_slug_group(group) for group in self.tenure_groups)
        return f"{prefix}_gv{joined}"


    @property
    def label(self) -> str:
        if self.business_scope == ALL_TENURES:
            return "All tenure groups (0, 1, 2, 3, 4+)"
        if self.business_scope == NEW_BUSINESS:
            return "New business (tenure group 0)"
        rendered = ", ".join(self.tenure_groups)
        if self.business_scope == MIXED_TENURES:
            return f"Mixed new business/renewal (tenure groups {rendered})"
        if self.tenure_groups == RENEWAL_GROUPS:
            return "Renewal (tenure groups 1, 2, 3, 4+)"
        return f"Renewal (tenure groups {rendered})"


    @property
    def contains_multiple_tenures(self) -> bool:
        """Compatibility name; the values are categorical tenure groups."""


        return len(self.tenure_groups) > 1


    @property
    def contains_multiple_groups(self) -> bool:
        return len(self.tenure_groups) > 1


    @property
    def payload(self) -> dict[str, object]:
        return {
            "business_scope": self.business_scope,
            "tenure_dimension": TENURE_DIMENSION,
            "tenure_group_domain": list(TENURE_GROUP_DOMAIN),
            "selected_tenure_groups": list(self.tenure_groups),
            "tenure_filter_mode": self.filter_mode,
            "profile_slug": self.slug,
        }


    def fingerprint(
        self,
        contract_version: str,
        *,
        exposure_year_as_predictor: bool,
        tenure_as_predictor: bool = False,
        population_scope: Mapping[str, object] | None = None,
    ) -> str:
        payload = {
            "contract_version": str(contract_version),
            **self.payload,
            "exposure_year_as_predictor": bool(exposure_year_as_predictor),
            "tenure_as_predictor": bool(tenure_as_predictor),
            "population_scope": dict(population_scope or {}),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()


    def sql_predicate(self, column: str) -> str:
        literals = ", ".join(_sql_literal(group) for group in self.tenure_groups)
        if len(self.tenure_groups) == 1:
            return f"{column} = {literals}"
        return f"{column} IN ({literals})"


    def includes_group(self, value: object) -> bool:
        group = _normalize_group(value, field="tenure group")
        return group in self.tenure_groups


    def includes_tenure(self, value: object) -> bool:
        """Compatibility name for callers migrating to ``includes_group``."""


        return self.includes_group(value)


    def environment(self, profile_hash: str) -> dict[str, str]:
        return {
            BUSINESS_SCOPE_ENV: self.business_scope,
            TENURE_GROUPS_ENV: ",".join(self.tenure_groups),
            TENURE_FILTER_MODE_ENV: self.filter_mode,
            PROFILE_SLUG_ENV: self.slug,
            PROFILE_HASH_ENV: str(profile_hash),
            PROFILE_PAYLOAD_ENV: json.dumps(
                self.payload, sort_keys=True, separators=(",", ":")
            ),
        }




def profile_from_values(
    *,
    business_scope: str | None = None,
    tenure: object | None = None,
    tenure_values: object | None = None,
    tenure_min: object | None = None,
    tenure_max: object | None = None,
) -> TenureProfile:
    """Resolve CLI-style values into one unambiguous group profile."""


    selector_count = sum(
        (
            tenure is not None,
            tenure_values is not None,
            tenure_min is not None or tenure_max is not None,
        )
    )
    if selector_count > 1:
        raise ValueError(
            "--tenures cannot be combined with --tenure or "
            "--tenure-min/--tenure-max."
        )
    normalized_scope = (
        None if business_scope is None else _normalize_scope(business_scope)
    )


    if tenure is not None:
        groups = (_normalize_selector_group(tenure, field="tenure selector"),)
    elif tenure_values is not None:
        raw_items = (
            [tenure_values]
            if isinstance(tenure_values, str)
            else list(tenure_values)
            if isinstance(tenure_values, Iterable)
            and not isinstance(tenure_values, (bytes, bytearray))
            else [tenure_values]
        )
        tokens = [
            piece.strip()
            for item in raw_items
            for piece in str(item).split(",")
        ]
        if any(token.upper() == "ALL" for token in tokens):
            if len(tokens) != 1 or tokens[0].upper() != "ALL":
                raise ValueError("ALL must be the only --tenures value.")
            groups = TENURE_GROUP_DOMAIN
        else:
            groups = _normalize_selector_values(tenure_values)
    elif tenure_min is not None or tenure_max is not None:
        if tenure_min is None:
            raise ValueError("--tenure-max requires --tenure-min.")
        minimum = _normalize_selector_group(
            tenure_min, field="tenure selector minimum"
        )
        groups = (
            TENURE_GROUP_DOMAIN[TENURE_GROUP_INDEX[minimum] :]
            if tenure_max is None
            else _selector_groups_between(tenure_min, tenure_max)
        )
    elif normalized_scope == RENEWAL:
        groups = RENEWAL_GROUPS
    elif normalized_scope == ALL_TENURES:
        groups = TENURE_GROUP_DOMAIN
    else:
        groups = NEW_BUSINESS_GROUPS


    inferred_scope = _scope_for_groups(groups)
    if normalized_scope is not None and normalized_scope != inferred_scope:
        raise ValueError(
            f"Selected tenure groups belong to {inferred_scope}, not "
            f"{normalized_scope}."
        )
    return TenureProfile(inferred_scope, groups)




def profile_from_environment(
    environment: Mapping[str, str] | None = None,
) -> TenureProfile:
    values = os.environ if environment is None else environment
    legacy = [
        name for name in LEGACY_TENURE_ENV_VARS if str(values.get(name, "")).strip()
    ]
    if legacy:
        raise ValueError(
            "Legacy integer-tenure environment variables are not supported by "
            f"the grouped-tenure contract: {legacy}."
        )


    scope = str(values.get(BUSINESS_SCOPE_ENV, "")).strip()
    groups = str(values.get(TENURE_GROUPS_ENV, "")).strip()
    supplied_mode = str(values.get(TENURE_FILTER_MODE_ENV, "")).strip().upper()
    supplied_slug = str(values.get(PROFILE_SLUG_ENV, "")).strip()
    supplied_payload = str(values.get(PROFILE_PAYLOAD_ENV, "")).strip()
    supplied_hash = str(values.get(PROFILE_HASH_ENV, "")).strip()
    if not any((scope, groups, supplied_mode, supplied_slug, supplied_payload, supplied_hash)):
        return TenureProfile(NEW_BUSINESS, NEW_BUSINESS_GROUPS)
    if not scope or not groups:
        raise ValueError(
            "The tenure-group environment is incomplete. Both "
            f"{BUSINESS_SCOPE_ENV} and {TENURE_GROUPS_ENV} are required."
        )


    profile = TenureProfile(_normalize_scope(scope), tuple(groups.split(",")))
    if supplied_mode:
        if supplied_mode not in ALLOWED_FILTER_MODES:
            raise ValueError(
                f"{TENURE_FILTER_MODE_ENV} must be one of "
                f"{ALLOWED_FILTER_MODES}; got {supplied_mode!r}."
            )
        if supplied_mode != profile.filter_mode:
            raise ValueError(
                f"{TENURE_FILTER_MODE_ENV}={supplied_mode!r} does not match "
                f"{profile.filter_mode!r}."
            )
    if supplied_slug and supplied_slug != profile.slug:
        raise ValueError(
            f"{PROFILE_SLUG_ENV}={supplied_slug!r} does not match {profile.slug!r}."
        )
    if supplied_payload:
        try:
            payload = json.loads(supplied_payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{PROFILE_PAYLOAD_ENV} is not valid JSON.") from exc
        if payload != profile.payload:
            raise ValueError(
                f"{PROFILE_PAYLOAD_ENV} does not match the tenure-group profile."
            )
    return profile




def validate_environment_fingerprint(
    profile: TenureProfile,
    contract_version: str,
    environment: Mapping[str, str] | None = None,
    *,
    exposure_year_as_predictor: bool = False,
    tenure_as_predictor: bool = False,
    population_scope: Mapping[str, object] | None = None,
) -> str:
    """Validate an optional inherited hash and return the canonical value."""


    values = os.environ if environment is None else environment
    expected = profile.fingerprint(
        contract_version,
        exposure_year_as_predictor=exposure_year_as_predictor,
        tenure_as_predictor=tenure_as_predictor,
        population_scope=population_scope,
    )
    supplied = str(values.get(PROFILE_HASH_ENV, "")).strip()
    if supplied and supplied != expected:
        raise ValueError(
            f"{PROFILE_HASH_ENV} does not match the selected tenure-group profile."
        )
    return expected
