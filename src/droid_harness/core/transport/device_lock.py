"""One process per phone: a machine-wide lock file per device serial (issue #18).

Every scrcpy session takes the lock for its device before forwarding the port, so a second
``droid-harness`` process (the autopilot service, the Studio, a terminal) fails fast with
who holds the phone instead of stealing the port and both sessions breaking.

* Machine-wide: the lock lives in ``$XDG_RUNTIME_DIR/droid-harness`` (or
  ``/tmp/droid-harness-<uid>``), not in the checkout, so two checkouts share it.
  ``DROID_HARNESS_LOCK_DIR`` overrides the folder.
* Re-entrant within a process: a run that reconnects (stop, then start) or holds the lock
  across reconnects keeps it; the file unlocks when the last holder releases.
* The kernel drops an ``flock`` when the process dies, so a crash never leaves the phone
  locked. On systems without ``fcntl`` the lock is a no-op.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]


class DeviceBusy(RuntimeError):
    """Another process is using this phone."""


def lock_dir() -> Path:
    env = os.environ.get("DROID_HARNESS_LOCK_DIR")
    if env:
        return Path(env)
    run = os.environ.get("XDG_RUNTIME_DIR")
    if run and Path(run).is_dir():
        return Path(run) / "droid-harness"
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return Path("/tmp") / f"droid-harness-{uid}"


def _key(serial: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", serial) or "default"


_held: dict[str, list[Any]] = {}  # key -> [fd, count]
_registry = threading.Lock()


def holder(serial: str) -> dict | None:
    """Who holds the phone right now (pid, command, since), or None if nobody does."""
    if fcntl is None:
        return None
    path = lock_dir() / f"{_key(serial)}.lock"
    if not path.exists():
        return None
    with _registry:
        if _key(serial) in _held:
            return _read(path) or {"pid": os.getpid()}
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return _read(path) or {"pid": None}
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text() or "null")
    except (OSError, ValueError):
        return None


def describe(info: dict | None) -> str:
    if not info:
        return "another process"
    since = info.get("since")
    when = f", since {time.strftime('%H:%M', time.localtime(since))}" if since else ""
    return f"PID {info.get('pid', '?')} ({info.get('command', 'droid-harness')}{when})"


class DeviceLock:
    def __init__(self, serial: str) -> None:
        self.serial = serial
        self.key = _key(serial)
        self.held = False

    def acquire(self) -> DeviceLock:
        if self.held or fcntl is None:
            self.held = True
            return self
        with _registry:
            entry = _held.get(self.key)
            if entry is not None:
                entry[1] += 1
                self.held = True
                return self
            d = lock_dir()
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{self.key}.lock"
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                raise DeviceBusy(f"the phone {self.serial} is in use by {describe(_read(path))}. Stop that "
                                 f"first (or wait for it to finish).") from None
            info = {"pid": os.getpid(), "command": " ".join(["droid-harness", *sys.argv[1:3]])[:120],
                    "since": time.time(), "serial": self.serial}
            os.ftruncate(fd, 0)
            os.write(fd, json.dumps(info).encode())
            _held[self.key] = [fd, 1]
            self.held = True
        return self

    def release(self) -> None:
        if not self.held:
            return
        self.held = False
        if fcntl is None:
            return
        with _registry:
            entry = _held.get(self.key)
            if entry is None:
                return
            entry[1] -= 1
            if entry[1] > 0:
                return
            fd = entry[0]
            del _held[self.key]
            try:
                os.ftruncate(fd, 0)
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def __enter__(self) -> DeviceLock:
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
