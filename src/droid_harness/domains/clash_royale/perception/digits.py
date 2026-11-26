"""Digit reading for the clock and tower HP: segment glyphs, match against 0-9 templates."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

GLYPH = (14, 20)  # (w, h)


def binarize(crop_bgr: np.ndarray, white_text: bool = True) -> np.ndarray:
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if not white_text:
        b = 255 - b
    return b


def segment(binary: np.ndarray, min_h_frac: float = 0.45) -> list[np.ndarray]:
    """Connected components tall enough to be digits, left to right."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, 4)
    h = binary.shape[0]
    boxes = [stats[i] for i in range(1, n) if stats[i][3] >= h * min_h_frac]
    boxes.sort(key=lambda s: s[0])
    out = []
    for x, y, w, hh, _ in boxes:
        glyph = binary[y:y + hh, x:x + w]
        out.append(cv2.resize(glyph, GLYPH, interpolation=cv2.INTER_AREA))
    return out


@dataclass
class DigitReader:
    templates: dict[str, np.ndarray] = field(default_factory=dict)  # "0".."9" -> GLYPH-sized binary
    min_score: float = 0.6

    def classify(self, glyph: np.ndarray) -> tuple[str | None, float]:
        best, score = None, -1.0
        g = glyph.astype(np.float32)
        for d, t in self.templates.items():
            s = float(cv2.matchTemplate(g, t.astype(np.float32), cv2.TM_CCOEFF_NORMED).max())
            if s > score:
                best, score = d, s
        return (best, score) if score >= self.min_score else (None, score)

    def read(self, crop_bgr: np.ndarray, white_text: bool = True) -> int | None:
        glyphs = segment(binarize(crop_bgr, white_text))
        if not glyphs:
            return None
        chars = []
        for g in glyphs:
            d, _ = self.classify(g)
            if d is None:
                return None
            chars.append(d)
        return int("".join(chars))

    @classmethod
    def from_font(cls, font=cv2.FONT_HERSHEY_SIMPLEX, scale: float = 1.0, thickness: int = 2) -> DigitReader:
        """Bootstrap templates from an OpenCV font (tests; replace with crops from your phone)."""
        r = cls()
        text = " ".join("0123456789")
        (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
        img = np.full((th + base + 8, tw + 8, 3), 20, np.uint8)
        cv2.putText(img, text, (4, th + 4), font, scale, (255, 255, 255), thickness)
        glyphs = segment(binarize(img), min_h_frac=0.3)  # same processing ``read`` applies
        if len(glyphs) != 10:
            raise ValueError(f"font produced {len(glyphs)} glyphs; adjust scale/thickness")
        r.templates = dict(zip("0123456789", glyphs, strict=True))
        return r
