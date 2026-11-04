"""Serialize changes to an extract folder across processes."""
from __future__ import annotations

import fcntl
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def folder_lock(folder: Path, timeout: float):
    """Hold the folder's persistent lock file exclusively, or time out."""
    deadline = time.monotonic() + timeout
    with (Path(folder) / ".folder.lock").open("a") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"folder busy: {folder}") from None
                time.sleep(min(0.05, remaining))
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
