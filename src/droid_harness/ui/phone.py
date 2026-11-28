"""Manual phone input from the Studio: taps, swipes and keys over ``adb``, behind a lock.

The preview shows the scrcpy stream, which is scaled (``max_size``) and maybe cropped, so a
click on it is mapped to the phone's real screen first. ``adb shell input`` is used instead of
the bot's scrcpy control socket so manual input works with or without a job running and never
touches the bot's executor. The lock starts closed every time the app starts.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

KEYS = {"back": 4, "home": 3, "app_switch": 187}


class PhoneLocked(RuntimeError):
    pass


class PhoneError(RuntimeError):
    pass


def parse_wm_size(text: str) -> tuple[int, int] | None:
    """``adb shell wm size`` output; an override size wins over the physical one."""
    found = {}
    for kind, w, h in re.findall(r"(Physical|Override) size:\s*(\d+)x(\d+)", text):
        found[kind] = (int(w), int(h))
    return found.get("Override") or found.get("Physical")


def parse_crop(crop: str | None) -> tuple[int, int, int, int] | None:
    """scrcpy ``crop`` is ``width:height:x:y`` in device pixels."""
    if not crop:
        return None
    try:
        w, h, x, y = (int(v) for v in str(crop).split(":"))
    except ValueError:
        return None
    return w, h, x, y


def frame_to_device(x: float, y: float, frame_w: float, frame_h: float, display: tuple[int, int],
                    crop: tuple[int, int, int, int] | None = None) -> tuple[int, int]:
    """Map a point on the streamed frame to device pixels."""
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError("frame size must be positive")
    dw, dh = display
    if (frame_w > frame_h) != (dw > dh):  # the phone is rotated relative to its natural size
        dw, dh = dh, dw
    if crop:
        cw, ch, cx, cy = crop
    else:
        cw, ch, cx, cy = dw, dh, 0, 0
    px = cx + (min(max(x, 0.0), frame_w) / frame_w) * cw
    py = cy + (min(max(y, 0.0), frame_h) / frame_h) * ch
    return int(round(min(px, dw - 1))), int(round(min(py, dh - 1)))


class PhoneControl:
    def __init__(self, settings, root: Path = Path("."), *,
                 runner: Callable[[list[str], float], subprocess.CompletedProcess] | None = None) -> None:
        self.settings, self.root = settings, Path(root)
        self._run = runner or _run
        self.locked = True
        self._lock = threading.Lock()
        self._display: tuple[str, float, tuple[int, int]] | None = None
        self._reach: tuple[str, float, bool] = ("", 0.0, False)
        self._probing = False
        self.last_input: dict[str, Any] | None = None

    # --- configuration -------------------------------------------------------------------
    def profile(self):
        from ..core.config import load_profile

        p = self.root / "configs/device_profiles" / f"{self.settings['device']}.yaml"
        try:
            return load_profile(p)
        except Exception as e:  # noqa: BLE001 - reported to the page
            raise PhoneError(f"can't read the device profile {p}: {e}") from e

    def serial(self) -> str:
        return str(self.settings.get("serial") or "") or self.profile().serial

    def adb_cmd(self, *args: str, serial: str | None = None) -> list[str]:
        adb = str(self.settings.get("adb_path") or "") or shutil.which("adb") or "adb"
        s = serial if serial is not None else self.serial()
        return [adb, "-s", s, *args] if s else [adb, *args]

    def _adb(self, *args: str, timeout: float = 8.0, binary: bool = False) -> bytes | str:
        try:
            cp = self._run(self.adb_cmd(*args), timeout)
        except FileNotFoundError as e:
            raise PhoneError("adb isn't installed or isn't on PATH (set its location in Settings)") from e
        except subprocess.TimeoutExpired as e:
            raise PhoneError(f"adb {' '.join(args[:2])} timed out: is the phone connected?") from e
        if cp.returncode != 0:
            err = cp.stderr.decode(errors="replace") if isinstance(cp.stderr, bytes) else str(cp.stderr or "")
            raise PhoneError(err.strip() or f"adb {' '.join(args[:2])} failed")
        out = cp.stdout or b""
        return out if binary else out.decode(errors="replace")

    # --- state ---------------------------------------------------------------------------
    def set_locked(self, locked: bool) -> bool:
        self.locked = bool(locked)
        return self.locked

    def display_size(self) -> tuple[int, int]:
        serial = self.serial()
        if self._display and self._display[0] == serial and time.monotonic() - self._display[1] < 120:
            return self._display[2]
        size = parse_wm_size(str(self._adb("shell", "wm", "size")))
        if size is None:
            raise PhoneError("couldn't read the phone's screen size (adb shell wm size)")
        self._display = (serial, time.monotonic(), size)
        return size

    def status(self) -> dict[str, Any]:
        """Lock state plus a cached reachability probe that refreshes in the background."""
        try:
            serial = self.serial()
            err = None
        except PhoneError as e:
            serial, err = "", str(e)
        s, t, ok = self._reach
        if serial and (s != serial or time.monotonic() - t > 5.0) and not self._probing:
            self._probing = True
            threading.Thread(target=self._probe, args=(serial,), daemon=True).start()
        return {"serial": serial, "reachable": ok if s == serial else None, "locked": self.locked,
                "error": err, "last_input": self.last_input}

    def _probe(self, serial: str) -> None:
        try:
            cp = self._run(self.adb_cmd("shell", "true", serial=serial), 4.0)
            ok = cp.returncode == 0
        except Exception:  # noqa: BLE001
            ok = False
        self._reach = (serial, time.monotonic(), ok)
        self._probing = False

    def devices(self) -> list[dict[str, str]]:
        """``adb devices -l`` as a list, for picking a serial in Settings."""
        adb = str(self.settings.get("adb_path") or "") or shutil.which("adb") or "adb"
        try:
            cp = self._run([adb, "devices", "-l"], 5.0)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []
        out = []
        for line in (cp.stdout or b"").decode(errors="replace").splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2:
                model = next((p.split(":", 1)[1] for p in parts[2:] if p.startswith("model:")), "")
                out.append({"serial": parts[0], "state": parts[1], "model": model})
        return out

    # --- input ---------------------------------------------------------------------------
    def _check(self) -> None:
        if self.locked:
            raise PhoneLocked("taps are locked: unlock them in the Studio first")

    def _map(self, x: float, y: float, fw: float, fh: float) -> tuple[int, int]:
        crop = None
        try:
            crop = parse_crop(self.profile().crop)
        except PhoneError:
            pass
        return frame_to_device(x, y, fw, fh, self.display_size(), crop)

    def tap(self, x: float, y: float, frame_w: float, frame_h: float) -> dict[str, Any]:
        self._check()
        with self._lock:
            dx, dy = self._map(x, y, frame_w, frame_h)
            self._adb("shell", "input", "tap", str(dx), str(dy))
        self.last_input = {"kind": "tap", "device_xy": [dx, dy], "t": time.time()}
        return self.last_input

    def swipe(self, x1: float, y1: float, x2: float, y2: float, frame_w: float, frame_h: float,
              duration_ms: int = 300) -> dict[str, Any]:
        self._check()
        ms = int(min(max(duration_ms, 40), 3000))
        with self._lock:
            a = self._map(x1, y1, frame_w, frame_h)
            b = self._map(x2, y2, frame_w, frame_h)
            self._adb("shell", "input", "swipe", str(a[0]), str(a[1]), str(b[0]), str(b[1]), str(ms),
                      timeout=ms / 1000 + 8)
        self.last_input = {"kind": "swipe", "device_xy": [*a, *b], "ms": ms, "t": time.time()}
        return self.last_input

    def key(self, name: str) -> dict[str, Any]:
        self._check()
        if name not in KEYS:
            raise ValueError(f"unknown key {name!r}; one of {', '.join(KEYS)}")
        with self._lock:
            self._adb("shell", "input", "keyevent", str(KEYS[name]))
        self.last_input = {"kind": "key", "key": name, "t": time.time()}
        return self.last_input

    def screencap(self) -> bytes:
        """A full-resolution PNG straight from the phone (used when no stream is running)."""
        png = self._adb("exec-out", "screencap", "-p", timeout=15.0, binary=True)
        if not isinstance(png, bytes) or not png.startswith(b"\x89PNG"):
            raise PhoneError("the phone didn't return a PNG screenshot")
        return png


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=timeout)
