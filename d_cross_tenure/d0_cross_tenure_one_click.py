#!/usr/bin/env python
"""One-click launcher for cross-tenure model-structure evidence.




The default, governed path is score-only and outer-OOF: it consumes c4 fold
bundles and target-native outer-OOF predictions, scores identical target
validation windows, and leaves the final holdout sealed. The former frozen
final-bundle path remains available only as an explicitly acknowledged
non-selection diagnostic. This launcher never trains on target data.


Use --baseline-tenures 1 2 3 4+ to report each tenure's own saved c4
outer-OOF predictions with the existing Logistic and XGBoost HTML lift writer.
Baseline reporting never fits, scores a new population, or opens the final
holdout. This is the outer-OOF baseline, not the legacy final-holdout baseline.
"""




from __future__ import annotations




import argparse
import ast
import hashlib
import importlib.util
import inspect
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterator, Sequence








LAUNCHER_VERSION = "BASELINES_HTML_20260911_v2"
EVALUATOR_ROLE = "frozen_source_evaluator"
BUILDER_ROLE = "frozen_source_model_builder"
TENURE_GROUPING_PURPOSE = "TENURE_GROUPING_MODEL_STRUCTURE"
NON_SELECTION_DIAGNOSTIC_PURPOSE = "NON_SELECTION_DIAGNOSTIC"








@dataclass(frozen=True)
class ComponentPaths:
    """Resolved code and data roots for one launcher invocation."""




    project_root: Path
    evaluator: Path
    builder: Path | None








def _nonempty_tenure(value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise argparse.ArgumentTypeError("tenure cannot be empty")
    return normalized








def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score target-tenure outer-validation rows with frozen source-tenure "
            "outer-fold Logistic and XGBoost bundles."
        ),
        epilog=(
            "Run c4 for the source and target first. Any additional artifact-path "
            "options are forwarded to the cross-tenure evaluator."
        ),
    )
    parser.add_argument("--version", action="version", version=LAUNCHER_VERSION)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--source-tenure", type=_nonempty_tenure)
    parser.add_argument("--target-tenure", type=_nonempty_tenure)
    parser.add_argument(
        "--baseline-tenures", nargs="+", type=_nonempty_tenure,
        metavar="TENURE",
        help=("Report native baseline HTML lifts for one or more tenures, e.g. "
              "--baseline-tenures 1 2 3 4+. Reads saved c4 outer-OOF predictions; "
              "no training, new scoring, or final-holdout access."),
    )
    parser.add_argument(
        "--baseline-run", action="append", default=[], metavar="TENURE=RUN_ID",
        help=("Optionally pin a baseline tenure to a completed run. Repeat for "
              "multiple tenures; otherwise use the evaluator's metadata-based "
              "latest COMPLETE/FULL run resolution."),
    )
    parser.add_argument(
        "--source-run",
        help=(
            "Optional source run path or run identifier. Resolution otherwise "
            "uses run metadata, not a versioned folder-name pattern."
        ),
    )
    parser.add_argument(
        "--materialize-source-bundle",
        action="store_true",
        help=(
            "Legacy diagnostic only: explicitly fit/save a final source model on "
            "source rows. Not permitted for tenure-grouping selection."
        ),
    )
    parser.add_argument(
        "--purpose",
        choices=(TENURE_GROUPING_PURPOSE, NON_SELECTION_DIAGNOSTIC_PURPOSE),
        default=TENURE_GROUPING_PURPOSE,
        help=(
            "Default TENURE_GROUPING_MODEL_STRUCTURE mode uses c4 outer-OOF "
            "evidence only. The "
            "legacy mode is never eligible to select a grouping."
        ),
    )
    parser.add_argument(
        "--evaluation-window",
        choices=("outer-oof", "source-holdout", "all-target", "final-test"),
        default=None,
    )
    parser.add_argument(
        "--confirm-final-holdout-diagnostic",
        action="store_true",
        help=(
            "Explicitly acknowledge final-holdout exposure in a legacy, "
            "non-selection diagnostic."
        ),
    )
    return parser








def parse_launcher_args(
    argv: Sequence[str] | None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = build_parser()
    args, evaluator_only_args = parser.parse_known_args(argv)
    if args.baseline_tenures is not None:
        if args.source_tenure or args.target_tenure or args.source_run:
            parser.error(
                "Use --baseline-tenures alone, without source/target options; "
                "pin a baseline with --baseline-run TENURE=RUN_ID."
            )
        if args.materialize_source_bundle:
            parser.error("Native outer-OOF baselines reuse saved c4 predictions; "
                         "--materialize-source-bundle is not permitted.")
        if (args.purpose != TENURE_GROUPING_PURPOSE
                or args.evaluation_window not in (None, "outer-oof")
                or args.confirm_final_holdout_diagnostic):
            parser.error("Native baseline reporting uses outer-oof only and "
                         "never opens the final holdout.")
        tenures = [str(value).strip() for value in args.baseline_tenures]
        if len(tenures) != len(set(tenures)):
            parser.error("--baseline-tenures must not contain duplicates.")
        if any(value not in {"0", "1", "2", "3", "4+"} for value in tenures):
            parser.error("Baseline tenures must be stored groups: 0, 1, 2, 3, 4+.")
        args.baseline_tenures = tenures
        args.evaluation_window = "outer-oof"
        return args, evaluator_only_args
    if args.baseline_run:
        parser.error("--baseline-run requires --baseline-tenures.")
    if not args.source_tenure or not args.target_tenure:
        parser.error("Supply --source-tenure and --target-tenure, or use "
                     "--baseline-tenures 1 2 3 4+.")
    if args.source_tenure == args.target_tenure:
        parser.error("--source-tenure and --target-tenure must be different")
    if args.purpose == TENURE_GROUPING_PURPOSE:
        if args.materialize_source_bundle:
            parser.error(
                "--materialize-source-bundle creates a final-refit bundle and is "
                "not valid for TENURE_GROUPING_MODEL_STRUCTURE. Run c4 to create "
                "outer-fold bundles."
            )
        if args.evaluation_window not in (None, "outer-oof"):
            parser.error(
                "Tenure-grouping permits only --evaluation-window outer-oof."
            )
        if args.confirm_final_holdout_diagnostic:
            parser.error(
                "Final-holdout acknowledgement is incompatible with "
                "--purpose TENURE_GROUPING_MODEL_STRUCTURE."
            )
        args.evaluation_window = "outer-oof"
    else:
        if args.evaluation_window in (None, "final-test"):
            args.evaluation_window = "source-holdout"
        if args.evaluation_window == "outer-oof":
            parser.error(
                "Use --purpose TENURE_GROUPING_MODEL_STRUCTURE for the governed "
                "outer-OOF path."
            )
        if not args.confirm_final_holdout_diagnostic:
            parser.error(
                "Legacy diagnostics can expose final-holdout outcomes; add "
                "--confirm-final-holdout-diagnostic."
            )
    return args, evaluator_only_args








def _derived_project_root(launcher_path: Path) -> Path:
    launcher_dir = launcher_path.resolve().parent
    if launcher_dir.name.casefold() in {"d_cross_tenure", "d_evaluation"}:
        return launcher_dir.parent
    return launcher_dir








def _component_search_directories(
    launcher_path: Path, project_root: Path
) -> tuple[Path, ...]:
    launcher_dir = launcher_path.resolve().parent
    candidates = [launcher_dir, project_root]
    if project_root.is_dir():
        candidates.extend(
            path for path in project_root.iterdir() if path.is_dir()
        )
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)








def _declared_component_role(path: Path) -> str | None:
    """Read a component role without importing or executing candidate code."""




    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError):
        return None
    for node in tree.body:
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "COMPONENT_ROLE"
            for target in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "COMPONENT_ROLE"
        ):
            value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value.strip()
    return None








def _find_component(
    launcher_path: Path, project_root: Path, expected_role: str
) -> Path:
    searched_directories: list[Path] = []
    matches: list[Path] = []
    launcher_resolved = launcher_path.resolve()
    for directory in _component_search_directories(launcher_path, project_root):
        if not directory.is_dir():
            continue
        searched_directories.append(directory)
        for candidate in directory.glob("*.py"):
            if candidate.resolve() == launcher_resolved:
                continue
            if _declared_component_role(candidate) == expected_role:
                matches.append(candidate.resolve())
    matches = list(dict.fromkeys(matches))
    if not matches:
        rendered = "\n  - ".join(str(path) for path in searched_directories)
        raise FileNotFoundError(
            f"Required component role {expected_role!r} was not found. "
            f"Searched Python files in:\n  - {rendered}"
        )
    if len(matches) > 1:
        rendered = "\n  - ".join(str(path) for path in matches)
        raise RuntimeError(
            f"Ambiguous component role {expected_role!r}; keep one current copy "
            f"in the active project layout. Found:\n  - {rendered}"
        )
    return matches[0]








def resolve_components(
    *,
    launcher_path: Path,
    project_root: Path | None,
    build_bundle: bool,
) -> ComponentPaths:
    root = (
        project_root.expanduser().resolve()
        if project_root is not None
        else _derived_project_root(launcher_path)
    )
    if not root.is_dir():
        raise NotADirectoryError(f"Project root does not exist: {root}")
    evaluator = _find_component(launcher_path, root, EVALUATOR_ROLE)
    builder = (
        _find_component(launcher_path, root, BUILDER_ROLE)
        if build_bundle
        else None
    )
    return ComponentPaths(
        project_root=root,
        evaluator=evaluator,
        builder=builder,
    )








def _load_component(path: Path, role: str) -> ModuleType:
    component_directory = str(path.resolve().parent)
    if component_directory not in sys.path:
        # Keep sibling support modules importable for the component's full run,
        # including lazy imports inside its main() implementation.
        sys.path.insert(0, component_directory)
    identity = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    module_name = f"_cross_tenure_{role}_{identity}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {role} component from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module








@contextmanager
def _temporary_argv(program: Path, argv: Sequence[str]) -> Iterator[None]:
    original = sys.argv
    sys.argv = [str(program), *map(str, argv)]
    try:
        yield
    finally:
        sys.argv = original








def invoke_component(path: Path, role: str, argv: Sequence[str]) -> int:
    """Invoke either ``main(argv)`` or a legacy zero-argument ``main()``."""




    module = _load_component(path, role)
    entrypoint = getattr(module, "main", None)
    if not callable(entrypoint):
        raise AttributeError(f"{role} component has no callable main(): {path}")




    signature = inspect.signature(entrypoint)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    accepts_varargs = any(
        parameter.kind == parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    if positional or accepts_varargs:
        result = entrypoint(list(map(str, argv)))
    else:
        with _temporary_argv(path, argv):
            result = entrypoint()
    return 0 if result is None else int(result)








def _builder_arguments(args: argparse.Namespace, project_root: Path) -> list[str]:
    """Build a source-only contract; target tenure is intentionally unavailable."""




    forwarded = [
        "--project-root",
        str(project_root),
        "--source-tenure",
        args.source_tenure,
    ]
    if args.source_run:
        forwarded.extend(("--source-run", str(args.source_run)))
    return forwarded








def _evaluator_arguments(
    args: argparse.Namespace,
    project_root: Path,
    evaluator_only_args: Sequence[str],
) -> list[str]:
    forwarded = [
        "--project-root",
        str(project_root),
        "--source-tenure",
        args.source_tenure,
        "--target-tenure",
        args.target_tenure,
        "--purpose",
        args.purpose,
        "--evaluation-window",
        args.evaluation_window,
    ]
    if args.confirm_final_holdout_diagnostic:
        forwarded.append("--confirm-final-holdout-diagnostic")
    if args.source_run:
        forwarded.extend(("--source-run", str(args.source_run)))
    forwarded.extend(map(str, evaluator_only_args))
    return forwarded










def _baseline_options(args: argparse.Namespace, extra: Sequence[str], evaluator: ModuleType):
    """Parse baseline-only artifact options without invoking cross-tenure scoring."""
    parser = argparse.ArgumentParser(prog="native-baseline report")
    parser.add_argument("--lift-bins", type=evaluator._parse_lift_bins,
                        default=evaluator.DEFAULT_LIFT_BINS)
    parser.add_argument("--output-dir", type=Path)
    options = parser.parse_args(list(extra))
    pinned: dict[str, Path] = {}
    for specification in args.baseline_run:
        tenure, separator, run = str(specification).partition("=")
        tenure, run = tenure.strip(), run.strip()
        if not separator or not run or tenure not in args.baseline_tenures:
            parser.error("--baseline-run must be TENURE=RUN_ID for a requested tenure.")
        if tenure in pinned:
            parser.error(f"Duplicate --baseline-run for tenure {tenure!r}.")
        pinned[tenure] = Path(run)
    return options, pinned




def _native_baseline_tables(native, record, bins, evaluator):
    """Reuse the exact evaluator's lift/spread formulas, ranking within each fold."""
    np, pd = evaluator.np, evaluator.pd
    families = tuple(evaluator.MODEL_FAMILIES)
    if set(native["model_family"]) != set(families):
        raise ValueError("Native baseline requires both Logistic and XGBoost predictions.")
    if native.duplicated(["model_family", "row_id"]).any():
        raise ValueError("A native baseline row is repeated within/across outer folds.")
    columns = list(dict.fromkeys([
        "row_id", "fold", "cohort_date", "y_retention", *record.identity_columns,
    ]))
    reference = None
    for family in families:
        part = native.loc[native["model_family"].eq(family), columns]
        part = part.sort_values("row_id", kind="mergesort").reset_index(drop=True)
        if reference is not None and not reference.equals(part):
            raise ValueError("Native Logistic and XGBoost must use identical rows, "
                             "folds, cohort dates, and outcomes.")
        reference = part
    holdout = record.payload.get("excluded_holdout", {})
    if not isinstance(holdout, dict) or not holdout.get("cohort_start"):
        raise ValueError("Native baseline needs the c4 final-holdout boundary.")
    holdout_start = pd.Timestamp(holdout["cohort_start"])
    if pd.isna(holdout_start) or native["cohort_date"].isna().any():
        raise ValueError("Native baseline cohort/holdout dates cannot be missing.")
    if native["cohort_date"].ge(holdout_start).any():
        raise ValueError("Native OOF data reaches the final holdout; reporting refused.")
    folds = {item.fold: item for item in record.folds}
    lift_parts, spread_rows, metric_rows = [], [], []
    for (family, fold_number), part in native.groupby(["model_family", "fold"], sort=True):
        fold = folds[int(fold_number)]
        if fold.validation_end >= holdout_start or not part["cohort_date"].between(
            fold.validation_start, fold.validation_end, inclusive="both"
        ).all():
            raise ValueError(f"Native {family} fold {fold_number} violates its saved window.")
        if len(part) < max(bins):
            raise ValueError(f"Native {family} fold {fold_number} has fewer rows "
                             f"than the requested {max(bins)} lift groups.")
        metadata = {"model_family": family, "fold": int(fold_number),
                    "model_role": "TARGET_NATIVE_OUTER_OOF"}
        metric_rows.append({**metadata, **evaluator._outer_metrics(
            part["y_retention"].to_numpy(), part["p_retention"].to_numpy(),
            row_id=part["row_id"],
        )})
        for count in bins:
            table = evaluator.build_lift_table(
                1 - part["y_retention"].to_numpy(dtype=int),
                1.0 - part["p_retention"].to_numpy(dtype=float),
                n_bins=int(count), row_id=part["row_id"],
            )
            table = table.assign(**metadata, n_bins=int(count))
            lift_parts.append(table)
            spread_rows.append({**metadata, "n_bins": int(count),
                                **evaluator.summarize_lift_spread(table)})
    lift_frame = pd.concat(lift_parts, ignore_index=True)
    chart_tables = {}
    summary_rows = []
    for family in families:
        family_tables = {}
        for count in bins:
            selected = lift_frame.loc[lift_frame["model_family"].eq(family)
                                      & lift_frame["n_bins"].eq(count)]
            rows = []
            for rank, group in selected.groupby("bin_rank", sort=True):
                values = pd.to_numeric(group["lift"], errors="raise").to_numpy(float)
                weights = pd.to_numeric(group["records"], errors="raise").to_numpy(float)
                valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
                rows.append({
                    "bin_rank": int(rank),
                    "records": int(group["records"].sum()),
                    "lift": (float(np.average(values[valid], weights=weights[valid]))
                             if valid.any() else np.nan),
                    "contributing_folds": int(valid.sum()),
                })
            table = pd.DataFrame(rows).sort_values("bin_rank", kind="mergesort")
            family_tables[int(count)] = table
            summary_rows.append({
                "model_family": family, "n_bins": int(count),
                "summary_basis": "SPREAD_OF_RECORD_WEIGHTED_WITHIN_BIN_FOLD_LIFTS",
                **evaluator.summarize_lift_spread(table),
            })
        chart_tables[family] = family_tables
    return (lift_frame, pd.DataFrame(spread_rows), pd.DataFrame(metric_rows),
            chart_tables, pd.DataFrame(summary_rows))




def _run_native_baselines(args, extra, components: ComponentPaths) -> int:
    """Write all requested native-baseline plots from existing, verified c4 OOF."""
    from datetime import datetime, timezone
    from html import escape


    evaluator = _load_component(components.evaluator, EVALUATOR_ROLE)
    required = (
        "resolve_outer_evaluation_manifest", "load_outer_evaluation_record",
        "_load_target_native_oof", "build_lift_table", "summarize_lift_spread",
        "plot_lift_like_graph1", "_outer_metrics", "_parse_lift_bins",
    )
    missing = [name for name in required if not callable(getattr(evaluator, name, None))]
    if missing:
        raise RuntimeError(f"Current outer-OOF evaluator lacks baseline helpers: {missing}")
    options, pinned = _baseline_options(args, extra, evaluator)
    # Check the actual chart writer before any large OOF file is read. A missing
    # or incompatible Plotly installation must not silently drop the requested HTML.
    try:
        import plotly.graph_objects  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("HTML baseline plots require Plotly in the active environment.") from exc


    resolved = {}
    for tenure in args.baseline_tenures:
        resolved[tenure] = evaluator.resolve_outer_evaluation_manifest(
            components.project_root, tenure, explicit_manifest=None,
            explicit_run=pinned.get(tenure),
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if options.output_dir is None:
        output = (components.project_root / "output" / "cross_tenure"
                  / f"native_baselines_{stamp}")
    else:
        output = options.output_dir.expanduser()
        if not output.is_absolute():
            output = components.project_root / output
        output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    print("\nHOME RETENTION 13/1 - NATIVE BASELINE HTML LIFTS", flush=True)
    print(f"Launcher: {Path(__file__).resolve()} [{LAUNCHER_VERSION}]", flush=True)
    print("Population: each tenure's own saved c4 outer-validation predictions", flush=True)
    print("No training, no new model scoring, no final-holdout access", flush=True)
    print(f"Output: {output}", flush=True)
    all_spreads, all_status, receipts, index_rows = [], [], [], []
    for tenure in args.baseline_tenures:
        print(f"\nTenure {tenure}: validating {resolved[tenure]}", flush=True)
        record = evaluator.load_outer_evaluation_record(
            resolved[tenure], project_root=components.project_root, expected_tenure=tenure,
        )
        native, audit = evaluator._load_target_native_oof(
            record, target_tenure=tenure, explicit_path=None,
            project_root=components.project_root,
        )
        lift, spreads, metrics, charts, summary = _native_baseline_tables(
            native, record, options.lift_bins, evaluator,
        )
        run_id = str(record.payload.get("run_id", record.payload.get("source_run_id", "")))
        directory = output / ("tenure_" + tenure.replace("+", "plus"))
        directory.mkdir(exist_ok=False)
        for table, filename in (
            (lift, "lift_by_outer_fold_equal_count_group.csv"),
            (spreads, "lift_spread_by_outer_fold.csv"),
            (metrics, "native_outer_oof_metrics_by_fold.csv"),
            (summary, "lift_spread_summary.csv"),
        ):
            table.insert(0, "run_id", run_id)
            table.insert(0, "tenure", tenure)
            table.to_csv(directory / filename, index=False)
        chart_table_parts = []
        chart_links = []
        for family in evaluator.MODEL_FAMILIES:
            chart_path = directory / "charts" / f"{family}_lift_chart.html"
            try:
                evaluator.plot_lift_like_graph1(
                    charts[family],
                    title=(f"Tenure {tenure} | {family.title()} | Native Outer-OOF Lift"
                           "<br><sup>Record-weighted fold lifts; ranked within each fold</sup>"),
                    save_html_path=chart_path,
                )
                if not chart_path.is_file() or chart_path.stat().st_size == 0:
                    raise RuntimeError("HTML writer did not create a nonempty file.")
            except Exception as exc:
                all_status.append({"tenure": tenure, "model_family": family,
                                   "status": "FAILED", "path": str(chart_path),
                                   "detail": str(exc)})
                evaluator.pd.DataFrame(all_status).to_csv(
                    output / "chart_generation_status.csv", index=False,
                )
                raise RuntimeError(f"Baseline HTML chart failed: {chart_path}. "
                                   "The completed CSV outputs have been retained.") from exc
            all_status.append({"tenure": tenure, "model_family": family,
                               "status": "WRITTEN", "path": str(chart_path), "detail": ""})
            relative = chart_path.relative_to(output).as_posix()
            chart_links.append(f'<a href="{escape(relative, quote=True)}">{escape(family)}</a>')
            for count, table in charts[family].items():
                chart_table_parts.append(table.assign(
                    tenure=tenure, run_id=run_id, model_family=family, n_bins=count,
                    aggregation="RECORD_WEIGHTED_WITHIN_BIN_ACROSS_OUTER_FOLDS",
                ))
        evaluator.pd.concat(chart_table_parts, ignore_index=True).to_csv(
            directory / "lift_chart_values.csv", index=False,
        )
        all_spreads.append(summary)
        receipts.append({"tenure": tenure, "run_id": run_id,
                         "outer_manifest": str(record.path),
                         "outer_manifest_sha256": evaluator.sha256_file(record.path),
                         "native_outer_oof": audit,
                         "output_directory": str(directory)})
        index_rows.append(f"<tr><td>{escape(tenure)}</td><td>{escape(run_id)}</td>"
                          f"<td>{' &nbsp; '.join(chart_links)}</td></tr>")
        print(f"Tenure {tenure}: Logistic and XGBoost HTML charts written.", flush=True)
        del native
    evaluator.pd.concat(all_spreads, ignore_index=True).to_csv(
        output / "native_baseline_lift_spread_summary.csv", index=False,
    )
    evaluator.pd.DataFrame(all_status).to_csv(output / "chart_generation_status.csv", index=False)
    evaluator._write_json(output / "native_baseline_manifest.json", {
        "contract": "native_baseline_outer_oof_html_v1", "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_tenures": args.baseline_tenures, "lift_bins": list(options.lift_bins),
        "evaluation_window": "outer-oof", "source_equals_target": True,
        "models_refitted": False, "new_model_scoring": False, "final_holdout_rows_scored": 0,
        "lift_definition": "within-fold group churn rate divided by within-fold churn rate",
        "chart_aggregation": "record-weighted within-bin mean of outer-fold lifts",
        "spread_summary_basis": "spread of aggregated chart values; per-fold spreads also saved",
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evaluator_sha256": evaluator.sha256_file(components.evaluator),
        "runs": receipts,
    })
    (output / "index.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>Native baseline lifts</title>'
        '</head><body><h1>Native baseline lifts</h1>'
        '<p>Each tenure uses its own saved outer-validation predictions. No final-test rows. '
        'Lift groups are ranked within each fold. Charts average fold lifts using each '
        'group\'s record count as its weight.</p>'
        '<table border="1" cellpadding="8"><tr><th>Tenure</th><th>Run</th><th>HTML plots</th></tr>'
        + ''.join(index_rows) + '</table></body></html>', encoding="utf-8",
    )
    print(f"\nNATIVE BASELINES COMPLETE: {output / 'index.html'}", flush=True)
    return 0




def run(argv: Sequence[str] | None, *, launcher_path: Path) -> int:
    args, evaluator_only_args = parse_launcher_args(argv)
    components = resolve_components(
        launcher_path=launcher_path,
        project_root=args.project_root,
        build_bundle=bool(args.materialize_source_bundle),
    )




    if args.baseline_tenures is not None:
        return _run_native_baselines(args, evaluator_only_args, components)


    print("\nHOME RETENTION 13/1 - CROSS-TENURE EVALUATION")
    print(f"\nProject root:  {components.project_root}")
    print(f"Source tenure: {args.source_tenure}")
    print(f"Target tenure: {args.target_tenure}")
    print(f"Purpose:       {args.purpose}")
    print(f"Conda env:     {os.environ.get('CONDA_DEFAULT_ENV', '<not reported>')}")




    if args.materialize_source_bundle:
        if components.builder is None:  # defensive; resolve_components enforces it
            raise AssertionError("Source-model builder path was not resolved")
        print(
            "\nBUILD SOURCE BUNDLE: fitting source-tenure rows only. "
            "Target-tenure rows are not loaded, fit, or trained in this step."
        )
        code = invoke_component(
            components.builder,
            "frozen_source_model_builder",
            _builder_arguments(args, components.project_root),
        )
        if code != 0:
            print(f"\nSource-model build failed with exit code {code}.")
            return code
    else:
        if args.purpose == TENURE_GROUPING_PURPOSE:
            print(
                "\nOUTER-OOF SCORE-ONLY MODE: loading c4 frozen fold bundles; "
                "inner metrics and final-holdout rows are unavailable to this "
                "decision path."
            )
        else:
            print(
                "\nNON-SELECTION DIAGNOSTIC: loading an existing frozen final "
                "source bundle; no model fitting or target training is permitted."
            )




    return invoke_component(
        components.evaluator,
        "frozen_source_evaluator",
        _evaluator_arguments(args, components.project_root, evaluator_only_args),
    )








def main(argv: Sequence[str] | None = None) -> int:
    return run(argv, launcher_path=Path(__file__))








if __name__ == "__main__":
    raise SystemExit(main())
