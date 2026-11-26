"""Settle detection: wait for the UI to stop changing instead of sleeping.

scrcpy sends frames only when the screen changes, so "no new frame for ``quiet_ms``" is
itself the settled signal. Blinking cursors and spinners keep frames coming, so a run of
near-identical frames also counts. When an accessibility tree is available, it must be
stable too.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import cv2
import numpy as np


def frame_diff(a: np.ndarray, b: np.ndarray, size: tuple[int, int] = (64, 128)) -> float:
    """Mean absolute difference of downscaled grayscale frames, in [0, 1]."""
    if a.shape != b.shape:
        return 1.0
    ga = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA)
    gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA)
    return float(np.abs(ga.astype(np.int16) - gb.astype(np.int16)).mean() / 255.0)


def wait_settled(client, tree_hash: Callable[[], str | None] | None = None, *, quiet_ms: float = 250,
                 eps: float = 0.002, n: int = 5, timeout_s: float = 5.0) -> bool:
    """True once settled; False on timeout (log it and continue cautiously)."""
    t_end = time.monotonic() + timeout_s
    prev = client.slot.peek()
    low_diff = 0
    last_hash = tree_hash() if tree_hash else None
    while time.monotonic() < t_end:
        f = client.slot.wait_newer(prev.seq if prev else 0, timeout=quiet_ms / 1000)
        if f is None:
            quiet = client.ms_since_last_packet() >= quiet_ms
        else:
            low_diff = low_diff + 1 if prev is not None and frame_diff(prev.img, f.img) < eps else 0
            prev = f
            quiet = False
        if quiet or low_diff >= n:
            if tree_hash is None:
                return True
            h = tree_hash()
            if h == last_hash:
                return True
            last_hash = h
    return False
