"""Thin wrapper over the ``adb`` CLI. Nothing here is on the hot path."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence


class AdbError(RuntimeError):
    pass


class Adb:
    def __init__(self, serial: str, adb_path: str | None = None, timeout_s: float = 20.0) -> None:
        self.serial = serial
        self.adb_path = adb_path or shutil.which("adb") or "adb"
        self.timeout_s = timeout_s

    def _cmd(self, *args: str) -> list[str]:
        return [self.adb_path, "-s", self.serial, *args]

    def run(self, *args: str, check: bool = True, timeout_s: float | None = None) -> str:
        try:
            proc = subprocess.run(self._cmd(*args), capture_output=True, text=True,
                                  timeout=timeout_s or self.timeout_s)
        except subprocess.TimeoutExpired as e:
            raise AdbError(f"adb {' '.join(args)} timed out") from e
        if check and proc.returncode != 0:
            raise AdbError(f"adb {' '.join(args)} failed: {proc.stderr.strip()}")
        return proc.stdout

    def popen(self, *args: str) -> subprocess.Popen:
        return subprocess.Popen(self._cmd(*args), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def shell(self, command: str | Sequence[str], **kw) -> str:
        if isinstance(command, str):
            return self.run("shell", command, **kw)
        return self.run("shell", *command, **kw)

    def push(self, local: str, remote: str) -> None:
        self.run("push", local, remote)

    def forward(self, local_port: int, remote: str) -> None:
        self.run("forward", f"tcp:{local_port}", remote)

    def forward_remove(self, local_port: int) -> None:
        self.run("forward", "--remove", f"tcp:{local_port}", check=False)

    def ping(self, timeout_s: float = 3.0) -> bool:
        try:
            self.run("shell", "true", timeout_s=timeout_s)
            return True
        except AdbError:
            return False

    def package_version(self, package: str) -> str | None:
        out = self.shell(f"dumpsys package {package}", check=False)
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("versionName="):
                return line.split("=", 1)[1]
        return None

    def force_stop(self, package: str) -> None:
        self.shell(f"am force-stop {package}", check=False)

    def launch(self, package: str) -> None:
        self.shell(f"monkey -p {package} -c android.intent.category.LAUNCHER 1", check=False)

    def keyevent(self, keycode: int) -> None:
        """Slow path (spawns a process) — menus/recovery only, never in battle."""
        self.shell(f"input keyevent {keycode}", check=False)
