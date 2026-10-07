# -*- coding: utf-8 -*-
"""Helpers for explicit one-click pipeline orchestration.


The project deliberately keeps orchestration simple: callers provide an
ordered list of Python files, and each file runs in its own process using the
same interpreter that launched the one-click script.  This module does not
search a directory, expand wildcards, or infer which project steps to run.
"""


from __future__ import annotations


from collections.abc import Mapping, Sequence
import glob
import os
from pathlib import Path
import subprocess
import sys
import time




def _phase_directory(path: str | os.PathLike[str]) -> Path:
    """Return a validated, resolved phase directory."""


    phase_dir = Path(path).resolve()
    if not phase_dir.exists():
        raise FileNotFoundError(f"Phase directory does not exist: {phase_dir}")
    if not phase_dir.is_dir():
        raise NotADirectoryError(f"Phase path is not a directory: {phase_dir}")
    return phase_dir




def _resolve_script(
    script: str | os.PathLike[str],
    phase_dir: Path,
) -> Path:
    """Resolve one explicit Python script strictly beneath ``phase_dir``."""


    raw_name = os.fspath(script)
    if not raw_name.strip():
        raise ValueError("Script names cannot be empty.")
    if glob.has_magic(raw_name):
        raise ValueError(
            f"Wildcard script expressions are not allowed: {raw_name!r}. "
            "List each Python file explicitly."
        )


    relative_path = Path(raw_name)
    if relative_path.is_absolute():
        raise ValueError(
            f"Scripts must be relative to the phase directory: {raw_name!r}."
        )
    if relative_path.suffix.lower() != ".py":
        raise ValueError(f"Only .py scripts can be orchestrated: {raw_name!r}.")


    resolved = (phase_dir / relative_path).resolve()
    try:
        resolved.relative_to(phase_dir)
    except ValueError as exc:
        raise ValueError(
            f"Script resolves outside the phase directory: {raw_name!r}."
        ) from exc


    if resolved == phase_dir:
        raise ValueError(f"Script must be a file beneath {phase_dir}: {raw_name!r}.")
    if not resolved.is_file():
        raise FileNotFoundError(f"Required one-click script not found: {resolved}")
    return resolved




def run_scripts(
    scripts: Sequence[str | os.PathLike[str]],
    path: str | os.PathLike[str],
    *,
    env: Mapping[str, str] | None = None,
) -> list[subprocess.CompletedProcess]:
    """Run an explicit sequence of Python scripts and stop on first failure.


    Parameters
    ----------
    scripts:
        Ordered, explicit script names relative to ``path``. Wildcards and
        non-Python files are rejected.
    path:
        Directory containing the phase scripts.
    env:
        Optional environment entries to overlay on a copy of the current
        process environment. Existing variables are preserved by default.


    Returns
    -------
    list[subprocess.CompletedProcess]
        Completed process objects, in execution order.


    Raises
    ------
    subprocess.CalledProcessError
        The first time a child script exits unsuccessfully. Later scripts are
        not started.
    """


    if isinstance(scripts, (str, bytes, os.PathLike)):
        raise TypeError("scripts must be an ordered sequence of explicit filenames.")


    phase_dir = _phase_directory(path)
    script_names = list(scripts)
    if not script_names:
        raise ValueError("At least one script must be listed for one-click execution.")


    # Resolve the complete explicit list before starting any work. This avoids
    # a partial run caused by a misspelled or missing later step.
    resolved_scripts = [
        _resolve_script(script, phase_dir) for script in script_names
    ]


    child_env = os.environ.copy()
    if env is not None:
        child_env.update({str(key): str(value) for key, value in env.items()})


    total = len(resolved_scripts)
    print(
        f"ONE-CLICK START: {total} script(s) from {phase_dir}",
        flush=True,
    )
    completed: list[subprocess.CompletedProcess] = []


    for index, script_path in enumerate(resolved_scripts, start=1):
        label = f"[{index}/{total}] {script_path.name}"
        print(f"{label} - START", flush=True)
        started = time.perf_counter()
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(phase_dir),
                env=child_env,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            elapsed = time.perf_counter() - started
            print(
                f"{label} - FAILED after {elapsed:.1f}s "
                f"(exit code {exc.returncode})",
                flush=True,
            )
            raise


        elapsed = time.perf_counter() - started
        completed.append(result)
        print(f"{label} - COMPLETE in {elapsed:.1f}s", flush=True)


    print(f"ONE-CLICK COMPLETE: {total}/{total} scripts succeeded.", flush=True)
    return completed
