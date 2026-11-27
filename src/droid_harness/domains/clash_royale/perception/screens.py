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
    # Arena-independent battle check. The dhash refs are captured in one arena with one
    # opponent name in the top HUD, so a ladder match (other arena theme, real player
    # nameplate) hashes far from every ref and came back UNKNOWN — the bot then sat in a
    # live match and never played. The elixir bar is present in *every* battle whatever
    # the arena, and is absent from home/menus/loading/results.
    battle_roi: Roi | None = None
    battle_label: str = "IN_BATTLE"
    battle_hsv_lo: tuple[int, int, int] = (140, 60, 80)
    battle_hsv_hi: tuple[int, int, int] = (175, 255, 255)
    battle_min_frac: float = 0.12

    def signature(self, img: np.ndarray) -> list[int]:
        return [dhash(r.crop(img)) for r in self.regions]

    def battle_evidence(self, img: np.ndarray) -> float:
        """Fraction of magenta (elixir) pixels in the elixir-bar region."""
        if self.battle_roi is None:
            return 0.0
        crop = self.battle_roi.crop(img)
        if crop.size == 0:
            return 0.0
        mask = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV),
                           np.array(self.battle_hsv_lo, np.uint8), np.array(self.battle_hsv_hi, np.uint8))
        return float((mask > 0).mean())

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
        if dist <= self.max_distance:
            return best, dist
        # No reference matched: fall back to the arena-independent battle HUD check.
        # Only reached for UNKNOWN, so it can upgrade a miss but never overrides a ref.
        if self.battle_evidence(img) >= self.battle_min_frac:
            return self.battle_label, dist
        return "UNKNOWN", dist


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


def read_result_detail(img: np.ndarray) -> tuple[str | None, int, int]:
    """Return ``(outcome, opp_crowns, my_crowns)`` for a result screen.

    ``outcome`` is ``win``/``loss``/``draw`` from *your* perspective (bottom banner
    is yours), or ``None`` when the result banners are absent.  Crown counts are the
    capped connected-component counts in each crown row; callers that need a stable
    final result should require the full triple to repeat before trusting it, since
    the crown-reveal animation passes through intermediate counts.
    """
    tp_l = _colour_fraction(img, 0.08, 0.24, 0.45, 0.36, (160, 80, 80), (180, 255, 255))
    tp_r = _colour_fraction(img, 0.55, 0.24, 0.92, 0.36, (160, 80, 80), (180, 255, 255))
    tr_l = _colour_fraction(img, 0.08, 0.24, 0.45, 0.36, (0, 120, 80), (10, 255, 255))
    tr_r = _colour_fraction(img, 0.55, 0.24, 0.92, 0.36, (0, 120, 80), (10, 255, 255))
    bb_l = _colour_fraction(img, 0.08, 0.44, 0.45, 0.56, (95, 80, 60), (130, 255, 255))
    bb_r = _colour_fraction(img, 0.55, 0.44, 0.92, 0.56, (95, 80, 60), (130, 255, 255))
    if max(tp_l, tr_l) < 0.20 or max(tp_r, tr_r) < 0.20 or bb_l < 0.15 or bb_r < 0.15:
        return None, 0, 0
    top = min(3, _large_gold_components(img, *_CROWN_TOP))
    bottom = min(3, _large_gold_components(img, *_CROWN_BOTTOM))
    if bottom > top:
        return "win", top, bottom
    if top > bottom:
        return "loss", top, bottom
    return "draw", top, bottom


def read_result(img: np.ndarray) -> str | None:
    """Return ``win``/``loss``/``draw`` for a result screen, else ``None``.

    Requires both result banners (pink top, blue bottom) so ordinary battle frames
    cannot be mistaken for a result.  The player with more gold crowns wins; equal
    crowns is a draw (the game's lowest-tower tiebreak is not modeled here).
    """
    outcome, _, _ = read_result_detail(img)
    return outcome
