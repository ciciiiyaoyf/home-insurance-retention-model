# -*- coding: utf-8 -*-
"""Small cross-platform lock shared by pipeline startup and cleanup."""


from __future__ import annotations


from contextlib import contextmanager
import os
from pathlib import Path
from typing import Iterator




class OperationLockBusy(RuntimeError):
    """Raised when another project operation owns the coordination lock."""




def _lock(handle) -> None:
    handle.seek(0)
    if os.name == "nt":  # pragma: no cover - exercised on company Windows
        import msvcrt


        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise OperationLockBusy(
                "Another pipeline startup or cleanup operation is in progress."
            ) from exc
        return


    import fcntl


    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise OperationLockBusy(
            "Another pipeline startup or cleanup operation is in progress."
        ) from exc




def _unlock(handle) -> None:
    handle.seek(0)
    if os.name == "nt":  # pragma: no cover - exercised on company Windows
        import msvcrt


        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return


    import fcntl


    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)




@contextmanager
def exclusive_operation_lock(path: Path) -> Iterator[None]:
    """Hold the project operation lock; the OS releases it after a crash."""


    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        _lock(handle)
        try:
            yield
        finally:
            _unlock(handle)
