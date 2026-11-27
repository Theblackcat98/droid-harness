"""Digit reading for the clock and tower HP: segment glyphs, match against 0-9 templates."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

GLYPH = (14, 20)  # (w, h)


def binarize(crop_bgr: np.ndarray, white_text: bool = True, *, allow_red: bool = False) -> np.ndarray:
    if crop_bgr.size == 0:
        return np.zeros((0, 0), np.uint8)
    if crop_bgr.ndim == 2:
        _, b = cv2.threshold(crop_bgr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return b if white_text else 255 - b
    hsv0 = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    # Synthetic scenes (CRScene) render pure neutral grayscale (S == 0) with no outline or HP bar.
    if int(hsv0[:, :, 1].max()) == 0 or not white_text:
        _, b = cv2.threshold(hsv0[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return b if white_text else 255 - b
    # Real HUD crops: 3x bicubic upscale separates 1-px anti-aliased strokes between adjacent digits
    # while rejecting yellow-gold unactivated King crowns, colored HP bars, and sunlit arena sand.
    up = cv2.resize(crop_bgr, (crop_bgr.shape[1] * 3, crop_bgr.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(up, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    fg = (v_ch >= 195) & (s_ch <= 95)
    if allow_red:
        fg = fg | ((v_ch >= 180) & ((h_ch <= 10) | (h_ch >= 170)) & (s_ch >= 130))
    dark = v_ch <= 155
    if int(dark.sum()) < max(12, int(0.03 * up.shape[0] * up.shape[1])):
        return np.zeros(up.shape[:2], np.uint8)
    dark_dil = cv2.dilate(dark.astype(np.uint8) * 255, np.ones((5, 5), np.uint8), iterations=1) > 0
    mask = fg.astype(np.uint8) * 255
    H, W = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 4)
    out = np.zeros_like(mask)
    for i in range(1, n):
        _, _, ww, hh, _ = stats[i]
        if hh < H * 0.38 or ww > W * 0.72 or ww < 4:
            continue
        comp = labels == i
        if np.any(comp & dark_dil):
            out[comp] = 255
    return out


def segment(binary: np.ndarray, min_h_frac: float = 0.38) -> list[np.ndarray]:
    """Connected components tall enough to be digits, left to right (splitting wide touching pairs)."""
    if binary.size == 0:
        return []
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, 4)
    h = binary.shape[0]
    boxes = [stats[i] for i in range(1, n) if stats[i][3] >= h * min_h_frac]
    boxes.sort(key=lambda s: s[0])
    out = []
    for x, y, w, hh, _ in boxes:
        sub = binary[y:y + hh, x:x + w]
        parts = [sub]
        while any(p.shape[1] > p.shape[0] * 1.32 for p in parts):
            nxt = []
            for p in parts:
                pw, ph = p.shape[1], p.shape[0]
                if pw > ph * 1.32:
                    lo, hi = max(2, int(pw * 0.28)), min(pw - 2, int(pw * 0.72))
                    cut = lo + int(np.argmin(p[:, lo:hi].sum(axis=0)))
                    nxt.extend([p[:, :cut], p[:, cut:]])
                else:
                    nxt.append(p)
            if len(nxt) == len(parts):
                break
            parts = nxt
        for p in parts:
            ys, xs = np.nonzero(p)
            if len(xs) == 0 or (xs.max() - xs.min() + 1) < 2:
                continue
            glyph = p[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            out.append(cv2.resize(glyph, GLYPH, interpolation=cv2.INTER_AREA))
    return out


@dataclass
class DigitReader:
    templates: dict[str, np.ndarray] = field(default_factory=dict)  # "0".."9" -> GLYPH-sized binary
    min_score: float = 0.58

    def classify(self, glyph: np.ndarray) -> tuple[str | None, float]:
        best, score = None, -1.0
        g = glyph.astype(np.float32)
        for d, t in self.templates.items():
            s = float(cv2.matchTemplate(g, t.astype(np.float32), cv2.TM_CCOEFF_NORMED).max())
            if s > score:
                best, score = d, s
        return (best, score) if score >= self.min_score else (None, score)

    def read(self, crop_bgr: np.ndarray, white_text: bool = True, *, allow_red: bool = False) -> int | None:
        glyphs = segment(binarize(crop_bgr, white_text, allow_red=allow_red))
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
