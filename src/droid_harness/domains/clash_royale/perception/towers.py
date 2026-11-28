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


def _aspect_ok(a_shape: tuple[int, ...], b_shape: tuple[int, ...], max_diff: float) -> bool:
    ah, aw = a_shape[:2]
    bh, bw = b_shape[:2]
    if ah <= 0 or aw <= 0 or bh <= 0 or bw <= 0:
        return False
    ra, rb = aw / ah, bw / bh
    return abs(ra - rb) / max(ra, rb) <= max_diff


def has_tower_hp_bar(img: np.ndarray, roi: Roi, side: str, offset: tuple[float, float] = (0.0, 0.0)) -> bool | None:
    """Detect whether a tower's HP bar (colored fill or dark empty track) is present.

    Returns ``None`` on grayscale or synthetic neutral (zero-saturation) images so tests
    without colored bars are not misclassified as destroyed.
    """
    if img.ndim < 3 or img.size == 0:
        return None
    h, w = img.shape[:2]
    x0 = max(0, int(round((roi.x0 + offset[0] - 0.01) * w)))
    x1 = min(w, int(round((roi.x1 + offset[0] + 0.02) * w)))
    y0 = max(0, int(round((roi.y0 + offset[1] - 0.01) * h)))
    y1 = min(h, int(round((roi.y1 + offset[1] + 0.025) * h)))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    if int(S.max()) == 0:
        return None
    if side == "opp":
        color_mask = ((H <= 10) | (H >= 165)) & (S >= 115) & (V >= 120)
    else:
        color_mask = (H >= 92) & (H <= 122) & (S >= 115) & (V >= 120)
    color_ok = float(color_mask.mean()) >= 0.008
    dark_ok = float(((V <= 75).mean(axis=1)).max()) >= 0.55
    return bool(color_ok or dark_ok)


class TowerStateReader:
    def __init__(self, templates: dict[str, dict[str, list[np.ndarray]]], *, min_score: float = 0.6,
                 margin: float = 0.05, confirm: int = 2) -> None:
        self.templates = templates  # tower -> state -> gray templates
        self.min_score, self.margin, self.confirm = min_score, margin, confirm
        self.reset()

    @classmethod
    def load(cls, directory: str | Path, *, rois: dict[str, Roi] | None = None,
             frame_shape: tuple[int, int, int] = (1024, 464, 3), max_aspect_diff: float = 0.12,
             **kw) -> TowerStateReader:
        t: dict[str, dict[str, list[np.ndarray]]] = {}
        files: dict[str, list[tuple[Path, tuple[int, int]]]] = {}
        for p in sorted(Path(directory).glob("*__*__*.png")):
            tower, state, _ = p.stem.split("__")
            if state not in STATES:
                continue
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                t.setdefault(tower, {}).setdefault(state, []).append(img)
                files.setdefault(tower, []).append((p, img.shape[:2]))
        for tower, entries in files.items():
            ref_shape = entries[0][1]
            if rois is not None and tower in rois:
                x0, y0, x1, y1 = rois[tower].to_px(frame_shape)
                ref_shape = (max(1, y1 - y0), max(1, x1 - x0))
            for p, shape in entries:
                if not _aspect_ok(shape, ref_shape, max_aspect_diff):
                    raise ValueError(
                        f"{p.name}: template shape {shape[1]}x{shape[0]} disagrees with "
                        f"expected {ref_shape[1]}x{ref_shape[0]} for {tower}"
                    )
        return cls(t, **kw)

    def reset(self) -> None:
        self.destroyed: set[str] = set()
        self._streak: dict[str, int] = {}

    def classify(self, crop: np.ndarray, tower: str) -> tuple[str | None, float]:
        """Best state for one ROI crop, or (None, score) if nothing is confident."""
        refs = self.templates.get(tower, {})
        scores = {
            st: max((ncc(crop, r) for r in refs.get(st, []) if _aspect_ok(crop.shape, r.shape, 0.50)), default=-1.0)
            for st in STATES
        }
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
