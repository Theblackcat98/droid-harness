"""Screen identification by difference hashes of fixed regions."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .grid import Roi


def dhash(img: np.ndarray, size: int = 8) -> int:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass
class ScreenClassifier:
    regions: list[Roi]
    references: dict[str, list[list[int]]] = field(default_factory=dict)
    max_distance: int = 10  # summed over regions

    def signature(self, img: np.ndarray) -> list[int]:
        return [dhash(r.crop(img)) for r in self.regions]

    def register(self, name: str, img: np.ndarray) -> None:
        self.references.setdefault(name, []).append(self.signature(img))

    def classify(self, img: np.ndarray) -> tuple[str, int]:
        sig = self.signature(img)
        best, dist = "UNKNOWN", 10**9
        for name, refs in self.references.items():
            for ref in refs:
                d = sum(hamming(a, b) for a, b in zip(sig, ref, strict=False))
                if d < dist:
                    best, dist = name, d
        return (best, dist) if dist <= self.max_distance else ("UNKNOWN", dist)


# --- result-screen verification ---------------------------------------------
# The result layout is identical for a win and a loss (top = opponent banner,
# bottom = your banner).  A dhash of the HUD/top area cannot tell them apart, and a
# single misclassified battle frame must never end a match.  These helpers look for
# the two result banners and count the large gold crowns in each crown row instead.

_CROWN_TOP = (0.18, 0.32)
_CROWN_BOTTOM = (0.40, 0.55)


def _colour_fraction(img: np.ndarray, x0: float, y0: float, x1: float, y1: float,
                     lo: tuple[int, int, int], hi: tuple[int, int, int]) -> float:
    h, w = img.shape[:2]
    crop = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    if crop.size == 0:
        return 0.0
    mask = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV),
                       np.array(lo, np.uint8), np.array(hi, np.uint8))
    return float((mask > 0).mean())


def _large_gold_components(img: np.ndarray, y0: float, y1: float, min_area: int | None = None) -> int:
    h = img.shape[0]
    crop = img[int(y0 * h):int(y1 * h), :]
    if crop.size == 0:
        return 0
    if min_area is None:
        # Relative area keeps this working at 360p, 464p or 1080p captures.
        min_area = max(500, int(0.002 * img.shape[0] * img.shape[1]))
    mask = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV),
                       np.array([12, 120, 120], np.uint8), np.array([38, 255, 255], np.uint8))
    _n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    return sum(1 for s in stats[1:] if s[4] >= min_area)


def read_result(img: np.ndarray) -> str | None:
    """Return ``win``/``loss``/``draw`` for a result screen, else ``None``.

    Requires both result banners (pink top, blue bottom) so ordinary battle frames
    cannot be mistaken for a result.  The player with more gold crowns wins; equal
    crowns is a draw (the game's lowest-tower tiebreak is not modeled here).
    """
    top_pink = _colour_fraction(img, 0.08, 0.24, 0.92, 0.36, (160, 80, 80), (180, 255, 255))
    top_red = _colour_fraction(img, 0.08, 0.24, 0.92, 0.36, (0, 120, 80), (10, 255, 255))
    bottom_blue = _colour_fraction(img, 0.08, 0.44, 0.92, 0.56, (95, 80, 60), (130, 255, 255))
    if max(top_pink, top_red) < 0.30 or bottom_blue < 0.20:
        return None
    top = min(3, _large_gold_components(img, *_CROWN_TOP))
    bottom = min(3, _large_gold_components(img, *_CROWN_BOTTOM))
    if bottom > top:
        return "win"
    if top > bottom:
        return "loss"
    return "draw"
