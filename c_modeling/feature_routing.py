"""Model-family routing for alternate representations of one business feature.


The source feature contract remains the union of columns that may be used by
either model family.  This module turns that union into a family-specific pool
before quality screening or feature selection, so raw and binned versions of
the same concept never compete in one model.


Physical pairs are discovered from the enabled candidate names: an enabled
``X_BIN`` or ``X_GRP`` is paired only when the same pool also contains enabled
``X`` (or the conventional raw ``X_NUM`` form).  The suffix is the contract;
legacy ``var_type`` values are not used to decide which representation is raw.
The external mapping is reserved for virtual transforms such as premium bins
and retention-behavior state groups that do not physically exist in the ROC
config.
"""


from __future__ import annotations


from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence




MODEL_FAMILIES = ("logistic", "xgboost")
SUPPORTED_VAR_TYPES = frozenset({"continuous", "discrete"})
SUPPORTED_ORIGINS = frozenset({"source", "derived"})
PHYSICAL_ALTERNATE_SUFFIXES = ("_BIN", "_GRP")




def _column_name(value: Any) -> str:
    name = str(value).strip().upper()
    if not name:
        raise ValueError("Feature names cannot be empty.")
    return name




def _family_name(value: Any) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "lr": "logistic",
        "logistic_regression": "logistic",
        "xgb": "xgboost",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in MODEL_FAMILIES:
        raise ValueError(
            f"model_family must be one of {MODEL_FAMILIES}; got {value!r}."
        )
    return normalized




def _var_type(value: Any, *, feature: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in SUPPORTED_VAR_TYPES:
        raise ValueError(
            f"Unsupported var_type {value!r} for {feature}; expected "
            f"one of {sorted(SUPPORTED_VAR_TYPES)}."
        )
    return normalized




@dataclass(frozen=True)
class FeatureVariant:
    """One model-specific representation of a logical business feature."""


    feature_name: str
    var_type: str
    origin: str = "source"
    source_columns: tuple[str, ...] = ()
    transform: str = ""


    def normalized(self) -> "FeatureVariant":
        feature = _column_name(self.feature_name)
        origin = str(self.origin).strip().lower()
        if origin not in SUPPORTED_ORIGINS:
            raise ValueError(
                f"Unsupported origin {self.origin!r} for {feature}; expected "
                f"one of {sorted(SUPPORTED_ORIGINS)}."
            )
        sources = tuple(_column_name(value) for value in self.source_columns)
        if len(sources) != len(set(sources)):
            raise ValueError(f"Duplicate source columns configured for {feature}.")
        if origin == "derived" and not sources:
            raise ValueError(
                f"Derived feature {feature} must declare at least one source column."
            )
        if origin == "source" and not sources:
            sources = (feature,)
        if origin == "source" and feature not in sources:
            raise ValueError(
                f"Source feature {feature} must include itself in source_columns."
            )
        transform = str(self.transform).strip()
        if origin == "derived" and not transform:
            raise ValueError(f"Derived feature {feature} must declare a transform.")
        if origin == "source" and transform:
            raise ValueError(
                f"Source feature {feature} cannot declare transform={transform!r}."
            )
        return FeatureVariant(
            feature_name=feature,
            var_type=_var_type(self.var_type, feature=feature),
            origin=origin,
            source_columns=sources,
            transform=transform,
        )




@dataclass(frozen=True)
class FeatureRoute:
    """The Logistic and XGBoost variants of one logical feature."""


    logical_feature: str
    logistic: FeatureVariant
    xgboost: FeatureVariant


    def normalized(self) -> "FeatureRoute":
        logical = _column_name(self.logical_feature)
        logistic = self.logistic.normalized()
        xgboost = self.xgboost.normalized()
        if (
            logistic.feature_name == xgboost.feature_name
            and logistic != xgboost
        ):
            raise ValueError(
                f"Route {logical} must use different Logistic and XGBoost "
                "feature names unless both families use the exact same "
                "derived-feature specification."
            )
        return FeatureRoute(logical, logistic, xgboost)


    def variant(self, model_family: str) -> FeatureVariant:
        family = _family_name(model_family)
        return self.logistic if family == "logistic" else self.xgboost


    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (self.logistic.feature_name, self.xgboost.feature_name)
            )
        )


    @property
    def activation_names(self) -> tuple[str, ...]:
        """Candidate names that make this route relevant to a family pool."""


        return tuple(
            dict.fromkeys(
                (
                    *self.feature_names,
                    *self.logistic.source_columns,
                    *self.xgboost.source_columns,
                )
            )
        )




@dataclass(frozen=True)
class RoutingResult:
    """One deterministic model-family routing result."""


    model_family: str
    candidates: tuple[str, ...]
    feature_types: Mapping[str, str]
    audit_rows: tuple[Mapping[str, str], ...]




def _normalize_route_contract(
    routes: Sequence[FeatureRoute],
) -> tuple[FeatureRoute, ...]:
    normalized = tuple(route.normalized() for route in routes)
    if not normalized:
        raise ValueError("The feature-variant mapping contains no routes.")
    concepts: set[str] = set()
    feature_owner: dict[str, str] = {}
    for route in normalized:
        if route.logical_feature in concepts:
            raise ValueError(
                f"Duplicate logical feature route: {route.logical_feature}."
            )
        concepts.add(route.logical_feature)
        for feature in route.activation_names:
            prior = feature_owner.get(feature)
            if prior is not None and prior != route.logical_feature:
                raise ValueError(
                    f"Feature {feature} belongs to both {prior} and "
                    f"{route.logical_feature}."
                )
            feature_owner[feature] = route.logical_feature
    return normalized




FEATURE_VARIANT_MAP_ENV = "RETENTION_FEATURE_VARIANT_MAP"
DEFAULT_FEATURE_VARIANT_MAP_PATH = Path(__file__).with_name(
    "feature_variant_map.json"
)




def resolve_feature_variant_map_path(path: str | Path | None = None) -> Path:
    """Resolve the external route contract without a machine-specific path."""


    configured = path
    if configured is None:
        configured = os.environ.get(FEATURE_VARIANT_MAP_ENV, "").strip()
    candidate = Path(configured) if configured else DEFAULT_FEATURE_VARIANT_MAP_PATH
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()




def _variant_from_payload(value: Any, *, label: str) -> FeatureVariant:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    allowed = {
        "feature_name",
        "var_type",
        "origin",
        "source_columns",
        "transform",
    }
    unknown = sorted(set(value).difference(allowed))
    missing = sorted({"feature_name", "var_type"}.difference(value))
    if unknown or missing:
        raise ValueError(
            f"Invalid keys for {label}: missing={missing}; unknown={unknown}."
        )
    sources = value.get("source_columns", [])
    if not isinstance(sources, list):
        raise ValueError(f"{label}.source_columns must be a JSON list.")
    return FeatureVariant(
        feature_name=value["feature_name"],
        var_type=value["var_type"],
        origin=value.get("origin", "source"),
        source_columns=tuple(sources),
        transform=value.get("transform", ""),
    )




def load_feature_variant_routes(
    path: str | Path | None = None,
) -> tuple[FeatureRoute, ...]:
    """Load and fail-closed validate the data-driven routing contract."""


    contract_path = resolve_feature_variant_map_path(path)
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Feature-variant mapping not found: {contract_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid feature-variant JSON at {contract_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {
        "contract_version",
        "routes",
    }:
        raise ValueError(
            "Feature-variant mapping must contain exactly contract_version "
            "and routes."
        )
    contract_version = payload["contract_version"]
    if (
        isinstance(contract_version, bool)
        or not isinstance(contract_version, int)
        or contract_version < 1
    ):
        raise ValueError(
            "Feature-variant contract_version must be a positive integer; "
            f"got {contract_version!r}."
        )
    raw_routes = payload["routes"]
    if not isinstance(raw_routes, list):
        raise ValueError("Feature-variant routes must be a JSON list.")
    routes: list[FeatureRoute] = []
    for index, raw_route in enumerate(raw_routes, start=1):
        label = f"routes[{index}]"
        if not isinstance(raw_route, dict) or set(raw_route) != {
            "logical_feature",
            "logistic",
            "xgboost",
        }:
            raise ValueError(
                f"{label} must contain exactly logical_feature, logistic, "
                "and xgboost."
            )
        routes.append(
            FeatureRoute(
                logical_feature=raw_route["logical_feature"],
                logistic=_variant_from_payload(
                    raw_route["logistic"], label=f"{label}.logistic"
                ),
                xgboost=_variant_from_payload(
                    raw_route["xgboost"], label=f"{label}.xgboost"
                ),
            )
        )
    return _normalize_route_contract(routes)




FEATURE_VARIANT_MAP_PATH = resolve_feature_variant_map_path()
FEATURE_VARIANT_ROUTES = load_feature_variant_routes(FEATURE_VARIANT_MAP_PATH)




def validate_routes(
    routes: Sequence[FeatureRoute] = FEATURE_VARIANT_ROUTES,
) -> tuple[FeatureRoute, ...]:
    """Normalize a route contract and reject ambiguous ownership."""


    return _normalize_route_contract(routes)




def infer_physical_feature_routes(
    candidates: Iterable[Any],
) -> tuple[FeatureRoute, ...]:
    """Infer enabled raw/alternate pairs from the config-derived pool.


    This does not inspect ``var_type`` and does not route a lone raw or lone
    grouped feature.  Exact unsuffixed names take priority; ``X_NUM`` is the
    only supported raw-name variant when the alternate is named ``X_BIN`` or
    ``X_GRP``.  Ambiguous multiple alternates fail closed.
    """


    pool = [_column_name(value) for value in candidates]
    if len(pool) != len(set(pool)):
        raise ValueError("Cannot infer physical routes from duplicate candidates.")
    pool_set = set(pool)
    alternate_by_raw: dict[str, str] = {}
    for alternate in pool:
        matching_suffix = next(
            (
                suffix
                for suffix in PHYSICAL_ALTERNATE_SUFFIXES
                if alternate.endswith(suffix)
            ),
            None,
        )
        if matching_suffix is None:
            continue
        stem = alternate[: -len(matching_suffix)]
        raw_matches = [
            raw
            for raw in (stem, f"{stem}_NUM")
            if raw in pool_set and raw != alternate
        ]
        if len(raw_matches) > 1:
            raise ValueError(
                f"Ambiguous raw counterparts for {alternate}: {raw_matches}."
            )
        if not raw_matches:
            continue
        raw = raw_matches[0]
        prior = alternate_by_raw.get(raw)
        if prior is not None and prior != alternate:
            raise ValueError(
                f"Raw feature {raw} has multiple enabled alternate "
                f"representations: {prior}, {alternate}."
            )
        alternate_by_raw[raw] = alternate


    inferred = tuple(
        FeatureRoute(
            logical_feature=raw,
            logistic=FeatureVariant(
                feature_name=alternate,
                var_type="discrete",
            ),
            xgboost=FeatureVariant(
                feature_name=raw,
                var_type="continuous",
            ),
        )
        for raw, alternate in alternate_by_raw.items()
    )
    return _normalize_route_contract(inferred) if inferred else ()




def _normalized_inputs(
    candidates: Iterable[Any], feature_types: Mapping[Any, Any]
) -> tuple[list[str], dict[str, str]]:
    candidate_list = [_column_name(value) for value in candidates]
    if not candidate_list:
        raise ValueError("The input candidate pool is empty.")
    if len(candidate_list) != len(set(candidate_list)):
        duplicates = sorted(
            {value for value in candidate_list if candidate_list.count(value) > 1}
        )
        raise ValueError(f"The input candidate pool contains duplicates: {duplicates}.")


    normalized_types: dict[str, str] = {}
    for raw_feature, raw_type in feature_types.items():
        feature = _column_name(raw_feature)
        if feature in normalized_types:
            raise ValueError(f"Duplicate normalized feature type key: {feature}.")
        normalized_types[feature] = _var_type(raw_type, feature=feature)
    missing_types = [value for value in candidate_list if value not in normalized_types]
    if missing_types:
        raise ValueError(
            "No input feature type was supplied for candidates: "
            + ", ".join(missing_types)
        )
    return candidate_list, normalized_types




def _active_routes(
    candidates: Sequence[str], routes: Sequence[FeatureRoute]
) -> tuple[FeatureRoute, ...]:
    candidate_set = set(candidates)
    return tuple(
        route
        for route in routes
        if candidate_set.intersection(route.activation_names)
    )




def validate_one_variant_per_concept(
    candidates: Iterable[Any],
    model_family: str | None = None,
    *,
    routes: Sequence[FeatureRoute] = FEATURE_VARIANT_ROUTES,
) -> None:
    """Require at most one routed representation per logical feature.


    When ``model_family`` is supplied, the one representation must also be the
    variant declared for that family.
    """


    pool = [_column_name(value) for value in candidates]
    if len(pool) != len(set(pool)):
        raise ValueError("The routed pool contains duplicate feature names.")
    family = _family_name(model_family) if model_family is not None else None
    pool_set = set(pool)
    for route in validate_routes(routes):
        selected = [name for name in route.feature_names if name in pool_set]
        if len(selected) > 1:
            raise ValueError(
                f"Logical feature {route.logical_feature} has multiple variants: "
                f"{selected}."
            )
        if family is not None and selected:
            expected = route.variant(family).feature_name
            if selected != [expected]:
                raise ValueError(
                    f"{family} requires {expected} for {route.logical_feature}; "
                    f"found {selected[0]}."
                )




def route_candidates(
    candidates: Iterable[Any],
    feature_types: Mapping[Any, Any],
    model_family: str,
    *,
    routes: Sequence[FeatureRoute] = FEATURE_VARIANT_ROUTES,
) -> RoutingResult:
    """Route a union candidate pool and its types to one model family.


    Unlisted candidates retain their original order and type.  A declared
    route activates only when at least one of its variants is present in the
    union pool.  Once active, missing physical variants or missing raw inputs
    for a derived variant fail closed instead of silently selecting a fallback.
    """


    family = _family_name(model_family)
    pool, input_types = _normalized_inputs(candidates, feature_types)
    declared_routes = validate_routes(routes)
    inferred_routes = infer_physical_feature_routes(pool)
    normalized_routes = validate_routes((*declared_routes, *inferred_routes))
    active = _active_routes(pool, normalized_routes)
    active_by_feature = {
        feature: route for route in active for feature in route.activation_names
    }
    selected_by_concept: dict[str, FeatureVariant] = {}
    pool_set = set(pool)
    for route in active:
        selected = route.variant(family)
        if selected.origin == "source" and selected.feature_name not in pool_set:
            raise ValueError(
                f"Cannot route {route.logical_feature} to missing physical "
                f"feature {selected.feature_name} for {family}."
            )
        missing_sources = [
            source for source in selected.source_columns if source not in pool_set
        ]
        if missing_sources:
            raise ValueError(
                f"Cannot route {route.logical_feature} to "
                f"{selected.feature_name} for {family}; required source "
                f"candidate(s) are missing: {missing_sources}."
            )
        selected_by_concept[route.logical_feature] = selected


    output: list[str] = []
    output_types: dict[str, str] = {}
    audit: list[dict[str, str]] = []
    emitted_concepts: set[str] = set()
    for feature in pool:
        route = active_by_feature.get(feature)
        if route is None:
            output.append(feature)
            output_types[feature] = input_types[feature]
            audit.append(
                {
                    "model_family": family,
                    "logical_feature": feature,
                    "selected_feature": feature,
                    "selected_var_type": input_types[feature],
                    "selected_origin": "source",
                    "transform": "",
                    "required_source_columns": feature,
                    "input_variants_present": feature,
                    "removed_variants": "",
                    "decision": "PRESERVE_UNLISTED",
                }
            )
            continue
        if route.logical_feature in emitted_concepts:
            continue
        emitted_concepts.add(route.logical_feature)
        selected = selected_by_concept[route.logical_feature]
        output.append(selected.feature_name)
        output_types[selected.feature_name] = selected.var_type
        present = [name for name in route.activation_names if name in pool_set]
        removed = [name for name in present if name != selected.feature_name]
        audit.append(
            {
                "model_family": family,
                "logical_feature": route.logical_feature,
                "selected_feature": selected.feature_name,
                "selected_var_type": selected.var_type,
                "selected_origin": selected.origin.upper(),
                "transform": selected.transform,
                "required_source_columns": "|".join(selected.source_columns),
                "input_variants_present": "|".join(present),
                "removed_variants": "|".join(removed),
                "decision": (
                    "ROUTE_DERIVED"
                    if selected.origin == "derived"
                    else "ROUTE_SOURCE_VARIANT"
                ),
            }
        )


    validate_one_variant_per_concept(
        output, family, routes=normalized_routes
    )
    return RoutingResult(
        model_family=family,
        candidates=tuple(output),
        feature_types=dict(output_types),
        audit_rows=tuple(audit),
    )




def route_candidate_list_and_types(
    candidates: Iterable[Any],
    feature_types: Mapping[Any, Any],
    model_family: str,
    *,
    routes: Sequence[FeatureRoute] = FEATURE_VARIANT_ROUTES,
) -> tuple[list[str], dict[str, str]]:
    """Convenience wrapper returning mutable list/dict integration objects."""


    result = route_candidates(
        candidates, feature_types, model_family, routes=routes
    )
    return list(result.candidates), dict(result.feature_types)




def routing_audit_rows(
    candidates: Iterable[Any],
    feature_types: Mapping[Any, Any],
    model_family: str,
    *,
    routes: Sequence[FeatureRoute] = FEATURE_VARIANT_ROUTES,
) -> list[dict[str, str]]:
    """Return CSV-ready routing audit dictionaries."""


    result = route_candidates(
        candidates, feature_types, model_family, routes=routes
    )
    return [dict(row) for row in result.audit_rows]




def derived_source_requirements(
    feature_name: Any,
    *,
    routes: Sequence[FeatureRoute] = FEATURE_VARIANT_ROUTES,
) -> tuple[str, ...]:
    """Return raw source columns for a declared virtual feature, else empty."""


    feature = _column_name(feature_name)
    for route in validate_routes(routes):
        for variant in (route.logistic, route.xgboost):
            if variant.feature_name == feature and variant.origin == "derived":
                return variant.source_columns
    return ()




def is_derived_feature(
    feature_name: Any,
    *,
    routes: Sequence[FeatureRoute] = FEATURE_VARIANT_ROUTES,
) -> bool:
    """Whether ``feature_name`` is a virtual output declared by this contract."""


    return bool(derived_source_requirements(feature_name, routes=routes))




def derived_requirements_for_candidates(
    candidates: Iterable[Any],
    *,
    routes: Sequence[FeatureRoute] = FEATURE_VARIANT_ROUTES,
) -> dict[str, tuple[str, ...]]:
    """Return only the virtual features present in a routed candidate pool."""


    requirements: dict[str, tuple[str, ...]] = {}
    for value in candidates:
        feature = _column_name(value)
        sources = derived_source_requirements(feature, routes=routes)
        if sources:
            requirements[feature] = sources
    return requirements




AUDIT_COLUMNS = (
    "model_family",
    "logical_feature",
    "selected_feature",
    "selected_var_type",
    "selected_origin",
    "transform",
    "required_source_columns",
    "input_variants_present",
    "removed_variants",
    "decision",
)




__all__ = [
    "AUDIT_COLUMNS",
    "DEFAULT_FEATURE_VARIANT_MAP_PATH",
    "FEATURE_VARIANT_MAP_ENV",
    "FEATURE_VARIANT_MAP_PATH",
    "FEATURE_VARIANT_ROUTES",
    "MODEL_FAMILIES",
    "PHYSICAL_ALTERNATE_SUFFIXES",
    "FeatureRoute",
    "FeatureVariant",
    "RoutingResult",
    "derived_requirements_for_candidates",
    "derived_source_requirements",
    "infer_physical_feature_routes",
    "is_derived_feature",
    "load_feature_variant_routes",
    "resolve_feature_variant_map_path",
    "route_candidate_list_and_types",
    "route_candidates",
    "routing_audit_rows",
    "validate_one_variant_per_concept",
    "validate_routes",
]
