"""Tower state from fixed ROIs. Towers never move, so this needs no detector.

A destroyed tower is found by template-matching each tower ROI (the same ROIs the HP
digits use) against per-phone ``alive`` and ``rubble`` references captured with
``droid-harness tower-template``. A tower counts as destroyed once ``rubble`` wins
``confirm`` reads in a row, and stays destroyed for the rest of the match.

Files: ``<dir>/<tower>__<state>__<i>.png``, e.g. ``opp_left__rubble__0.png``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .grid import Roi

STATES = ("alive", "rubble")


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation of two same-size gray images (resizes ``a`` to ``b``)."""
    a = cv2.resize(_gray(a), (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    b = b.astype(np.float32)
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-6 else 0.0


class TowerStateReader:
    def __init__(self, templates: dict[str, dict[str, list[np.ndarray]]], *, min_score: float = 0.6,
                 margin: float = 0.05, confirm: int = 2) -> None:
        self.templates = templates  # tower -> state -> gray templates
        self.min_score, self.margin, self.confirm = min_score, margin, confirm
        self.reset()

    @classmethod
    def load(cls, directory: str | Path, **kw) -> TowerStateReader:
        t: dict[str, dict[str, list[np.ndarray]]] = {}
        for p in sorted(Path(directory).glob("*__*__*.png")):
            tower, state, _ = p.stem.split("__")
            if state not in STATES:
                continue
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                t.setdefault(tower, {}).setdefault(state, []).append(img)
        return cls(t, **kw)

    def reset(self) -> None:
        self.destroyed: set[str] = set()
        self._streak: dict[str, int] = {}

    def classify(self, crop: np.ndarray, tower: str) -> tuple[str | None, float]:
        """Best state for one ROI crop, or (None, score) if nothing is confident."""
        refs = self.templates.get(tower, {})
        scores = {st: max((ncc(crop, r) for r in refs.get(st, [])), default=-1.0) for st in STATES}
        best = max(scores, key=scores.get)
        other = min(scores, key=scores.get)
        if scores[best] < self.min_score:
            return None, scores[best]
        if refs.get(other) and scores[best] - scores[other] < self.margin:
            return None, scores[best]
        return best, scores[best]

    def read(self, img: np.ndarray, rois: dict[str, Roi]) -> set[str]:
        """Towers newly confirmed destroyed in this frame."""
        new = set()
        for name, roi in rois.items():
            if name in self.destroyed or name not in self.templates:
                continue
            state, _ = self.classify(roi.crop(img), name)
            if state == "rubble":
                self._streak[name] = self._streak.get(name, 0) + 1
                if self._streak[name] >= self.confirm:
                    self.destroyed.add(name)
                    new.add(name)
            else:
                self._streak[name] = 0
        return new
