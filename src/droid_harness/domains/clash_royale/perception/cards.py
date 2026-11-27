"""Hand-card identification that survives greyed-out (unaffordable) and selected states.

Templates are lightly blurred luma (grayscale) images captured on *your* phone, several
per card (affordable, greyed, selected). Desaturation keeps luma and darkening scales it;
masked normalized cross-correlation (mean-subtracted, unit-variance) is invariant to that
kind of brightness/contrast change and tolerant of video compression. A mask hides the
elixir-cost badge and a margin check guards against the runner-up. It's a closed set:
only your 8 deck cards. Still capture greyed templates on the device — the game's
grey-out isn't guaranteed to be a pure luma scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

TEMPLATE_SIZE = (48, 60)  # (w, h) every crop/template is resized to this


def to_lightness(crop_bgr: np.ndarray, size: tuple[int, int] = TEMPLATE_SIZE) -> np.ndarray:
    resized = cv2.resize(crop_bgr, size, interpolation=cv2.INTER_AREA)
    luma = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
    return cv2.GaussianBlur(luma, (3, 3), 0)


def local_znorm(luma: np.ndarray) -> np.ndarray:
    """Local zero-mean, unit-variance luma normalization.

    Cancels the brightness/contrast shift of Clash Royale's radial cooldown wipe
    so greyed-out cards in mid-cooldown still match clean templates.
    """
    f = luma.astype(np.float32)
    mu = cv2.GaussianBlur(f, (11, 11), 3.0)
    sq = cv2.GaussianBlur(f * f, (11, 11), 3.0)
    sd = np.sqrt(np.maximum(0.0, sq - mu * mu)) + 6.0
    return (f - mu) / sd


def masked_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Pearson correlation of two same-size images over mask > 0 (range -1..1)."""
    m = mask > 0
    x = a[m].astype(np.float32)
    y = b[m].astype(np.float32)
    x -= x.mean()
    y -= y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / denom) if denom > 1e-6 else 0.0


def default_mask(size: tuple[int, int] = TEMPLATE_SIZE, badge_frac: float = 0.28) -> np.ndarray:
    """255 everywhere except the elixir-cost badge (bottom-center in CR, bottom-left in sim)
    and outer 2-px border (selected white glow / card frame)."""
    w, h = size
    m = np.full((h, w), 255, np.uint8)
    m[:2, :] = 0
    m[-2:, :] = 0
    m[:, :2] = 0
    m[:, -2:] = 0
    m[int(h * (1 - badge_frac)):, int(w * 0.28):int(w * 0.72)] = 0
    m[int(h * (1 - badge_frac)):, :int(w * 0.35)] = 0
    return m


@dataclass
class MatchResult:
    card: str | None
    score: float
    margin: float
    accepted: bool


@dataclass
class CardMatcher:
    templates: dict[str, list[np.ndarray]] = field(default_factory=dict)
    mask: np.ndarray = field(default_factory=default_mask)
    min_score: float = 0.40
    min_margin: float = 0.08
    _znorm_cache: dict[str, list[np.ndarray]] = field(default_factory=dict, init=False, repr=False)

    def add(self, card: str, crop_bgr: np.ndarray) -> None:
        L = to_lightness(crop_bgr)
        self.templates.setdefault(card, []).append(L)
        self._znorm_cache.setdefault(card, []).append(local_znorm(L))

    def _znorm_for(self, card: str) -> list[np.ndarray]:
        tpls = self.templates[card]
        cached = self._znorm_cache.get(card)
        if cached is None or len(cached) != len(tpls):
            cached = [local_znorm(t) for t in tpls]
            self._znorm_cache[card] = cached
        return cached

    def match(self, crop_bgr: np.ndarray) -> MatchResult:
        if crop_bgr.size == 0 or not self.templates:
            return MatchResult(None, 0.0, 0.0, False)
        z = local_znorm(to_lightness(crop_bgr))
        per_card = {}
        for card, ztpls in ((c, self._znorm_for(c)) for c in self.templates):
            per_card[card] = max(masked_ncc(z, zt, self.mask) for zt in ztpls)
        ranked = sorted(per_card.items(), key=lambda kv: kv[1], reverse=True)
        best, s1 = ranked[0]
        s2 = ranked[1][1] if len(ranked) > 1 else 0.0
        ok = s1 >= self.min_score and (s1 - s2) >= self.min_margin
        return MatchResult(best, s1, s1 - s2, ok)

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for card, tpls in self.templates.items():
            for i, t in enumerate(tpls):
                cv2.imwrite(str(d / f"{card}__{i}.png"), t)

    @classmethod
    def load(cls, directory: str | Path, **kw) -> CardMatcher:
        m = cls(**kw)
        for p in sorted(Path(directory).glob("*.png")):
            card = p.stem.split("__")[0]
            m.templates.setdefault(card, []).append(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
        return m
