"""Loopback latency test (plan §5.4).

With *Show taps* enabled on the phone, inject a finger-down at a quiet spot and watch the
decoded stream for the touch indicator. RTT = input path + render + encode + transport
+ decode: the full I/O loop, measured entirely in software.
"""

from __future__ import annotations

import time

import numpy as np

from ..transport.scrcpy_protocol import MotionAction
from .latency import percentiles


def _roi_mean(img: np.ndarray, x: int, y: int, r: int) -> np.ndarray:
    h, w = img.shape[:2]
    return img[max(0, y - r):min(h, y + r), max(0, x - r):min(w, x + r)].astype(np.float32).mean(axis=(0, 1))


def loopback_rtt(client, spot: tuple[int, int], *, samples: int = 50, radius: int = 8,
                 threshold: float = 40.0, timeout_s: float = 1.0, settle_s: float = 0.15,
                 pointer_id: int = 9) -> dict:
    """Return {'samples_ms': [...], **percentiles, 'timeouts': n}. ``spot`` is in frame pixels."""
    x, y = spot
    results: list[float] = []
    timeouts = 0
    for _ in range(samples):
        base_f = client.slot.wait_newer(0, timeout=1.0) or client.slot.peek()
        if base_f is None:
            raise RuntimeError("no frames from device")
        baseline = _roi_mean(base_f.img, x, y, radius)
        seq = base_f.seq
        t_send = client.touch(MotionAction.DOWN, x, y, pointer_id=pointer_id)
        seen = None
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            f = client.slot.wait_newer(seq, timeout=0.05)
            if f is None:
                continue
            seq = f.seq
            if np.abs(_roi_mean(f.img, x, y, radius) - baseline).max() > threshold:
                seen = f.t_decoded_ns
                break
        client.touch(MotionAction.UP, x, y, pointer_id=pointer_id)
        if seen is None:
            timeouts += 1
        else:
            results.append((seen - t_send) / 1e6)
        time.sleep(settle_s)  # let the indicator disappear
    return {"samples_ms": results, "timeouts": timeouts, **percentiles(results)}
