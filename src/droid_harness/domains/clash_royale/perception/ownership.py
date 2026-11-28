"""Unit ownership and health from the HP-bar strip above each detection box.

Blue bar = yours, red bar = theirs; the colored fraction of the bar width is ``hp_pct``.
When no bar is readable, return None and let the tracker fall back to the spawn-side prior.
"""

from __future__ import annotations

import cv2
import numpy as np

# HSV ranges (OpenCV hue 0..180). Calibrate on your phone; these are starting points.
BLUE = ((95, 120, 120), (125, 255, 255))
RED_LO = ((0, 120, 120), (10, 255, 255))
RED_HI = ((163, 120, 120), (180, 255, 255))


def bar_strip(img: np.ndarray, box: tuple[float, float, float, float], height_px: int = 8,
              gap_px: int = 2) -> np.ndarray:
    x1, y1, x2, _ = (int(round(v)) for v in box)
    h, w = img.shape[:2]
    top = max(0, y1 - gap_px - height_px)
    bottom = max(0, y1 - gap_px)
    return img[top:bottom, max(0, x1):min(w, x2)]


def ownership_and_hp(img: np.ndarray, box, min_fill: float = 0.05) -> tuple[str | None, float | None]:
    strip = bar_strip(img, box)
    if strip.size == 0:
        return None, None
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array(BLUE[0]), np.array(BLUE[1]))
    red = cv2.inRange(hsv, np.array(RED_LO[0]), np.array(RED_LO[1])) | \
        cv2.inRange(hsv, np.array(RED_HI[0]), np.array(RED_HI[1]))
    cols_blue = (blue > 0).mean(axis=0) > 0.4
    cols_red = (red > 0).mean(axis=0) > 0.4
    fb, fr = cols_blue.mean(), cols_red.mean()
    if max(fb, fr) < min_fill:
        return None, None
    side = "me" if fb >= fr else "opp"
    cols = cols_blue if side == "me" else cols_red
    filled = np.flatnonzero(cols)
    hp = (filled.max() + 1) / cols.size if filled.size else 0.0
    return side, float(min(1.0, hp))


def strong_bar_side(img: np.ndarray, box, min_fill: float = 0.18) -> tuple[str | None, float | None]:
    """Strict HP-bar check used to catch high-confidence blue/red team overrides."""
    return ownership_and_hp(img, box, min_fill=min_fill)
