"""HUD readers that run every frame (< 1 ms total)."""

from __future__ import annotations

import cv2
import numpy as np

from .grid import Roi


def read_elixir(img: np.ndarray, roi: Roi, hsv_lo, hsv_hi, segments: int = 10) -> float | None:
    """Filled fraction of the elixir bar × 10. Returns None if the bar isn't visible."""
    crop = roi.crop(img)
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8))
    cols = mask.mean(axis=0) > 127  # per-column "filled"
    if cols.size == 0:
        return None
    # The bar fills left to right: take the contiguous filled run from the left edge,
    # tolerant of a few noisy columns.
    filled = np.flatnonzero(cols)
    if filled.size == 0:
        return 0.0
    return float(segments * min(1.0, (filled.max() + 1) / cols.size))


def mean_saturation(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    return float(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1].mean())


def is_greyed(crop: np.ndarray, threshold: float = 60.0) -> bool:
    """Unaffordable cards are desaturated: a free affordability signal to cross-check elixir."""
    return mean_saturation(crop) < threshold
