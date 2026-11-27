"""Hand-template harvesting for cards the matcher doesn't know yet (opt-in: ``--harvest-templates DIR``).

When you put a new card in the deck, its hand slot can't be read until someone captures
templates. But the card's identity is already known one step earlier: the card in the
"next" slot is the one that refills the slot you just played from. So when a play empties
slot *i* while the next slot shows card *X*, and slot *i* then stays unreadable after the
refill, the crop there is *X*. Those crops are saved (per card and affordable/greyed state)
for ``droid-harness promote-templates`` to turn into templates after a quick look.

Passive: it never changes what perception reports.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .cards import TEMPLATE_SIZE, to_lightness


class TemplateHarvester:
    def __init__(self, out_dir: str | Path, *, max_per_state: int = 12, refill_ms: int = 1700,
                 window_ms: int = 6000, min_gap_ms: int = 700) -> None:
        self.out = Path(out_dir)
        self.max_per_state = max_per_state
        self.refill_ms, self.window_ms, self.min_gap_ms = refill_ms, window_ms, min_gap_ms
        self.saved = 0
        self._expect: dict[int, tuple[str, int]] = {}
        self._last_save: dict[int, int] = {}

    def reset(self) -> None:
        self._expect.clear()
        self._last_save.clear()

    def expect(self, slot: int, card: str | None, t_ms: int) -> None:
        """A play emptied ``slot`` while the next slot showed ``card``."""
        if card:
            self._expect[slot] = (card, t_ms)

    def observe(self, slot: int, crop: np.ndarray, read_card: str | None, t_ms: int, greyed: bool) -> Path | None:
        exp = self._expect.get(slot)
        if exp is None:
            return None
        card, t0 = exp
        if read_card is not None or t_ms - t0 > self.window_ms:
            self._expect.pop(slot, None)  # the slot read fine (or the moment passed): nothing to learn
            return None
        if t_ms - t0 < self.refill_ms or t_ms - self._last_save.get(slot, -10**9) < self.min_gap_ms:
            return None
        state = "greyed" if greyed else "affordable"
        d = self.out / card
        existing = list(d.glob(f"{state}_*.png")) if d.exists() else []
        if len(existing) >= self.max_per_state or crop.size == 0:
            return None
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{state}_{int(time.time())}_{t_ms}_{slot}_{self.saved}.png"
        cv2.imwrite(str(p), crop)
        self._last_save[slot] = t_ms
        self.saved += 1
        return p


def _vec(img: np.ndarray) -> np.ndarray:
    L = to_lightness(img) if img.ndim == 3 else cv2.resize(img, TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
    v = L.astype(np.float32).ravel()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def promote(harvest_dir: str | Path, templates_dir: str | Path, *, min_crops: int = 3,
            min_agreement: float = 0.6, dry_run: bool = False) -> list[dict]:
    """For each card and state with enough crops, write the medoid crop as a new template.

    Skips a card/state whose crops disagree (mean correlation to the medoid below
    ``min_agreement``): that usually means the expectation was wrong (a card was played
    again before the refill) and the crops need a look.
    """
    harvest_dir, templates_dir = Path(harvest_dir), Path(templates_dir)
    report = []
    for card_dir in sorted(p for p in harvest_dir.iterdir() if p.is_dir()) if harvest_dir.exists() else []:
        card = card_dir.name
        for state in ("affordable", "greyed"):
            files = sorted(card_dir.glob(f"{state}_*.png"))
            rec = {"card": card, "state": state, "crops": len(files), "written": None}
            if len(files) < min_crops:
                rec["skip"] = f"fewer than {min_crops} crops"
                report.append(rec)
                continue
            imgs = [cv2.imread(str(f)) for f in files]
            vecs = np.stack([_vec(i) for i in imgs])
            sim = vecs @ vecs.T
            m = int(np.argmax(sim.sum(axis=1)))
            agreement = float(np.delete(sim[m], m).mean())
            rec["agreement"] = round(agreement, 3)
            if agreement < min_agreement:
                rec["skip"] = "crops disagree; check them by eye"
                report.append(rec)
                continue
            i = len(list(templates_dir.glob(f"{card}__*.png")))
            out = templates_dir / f"{card}__{i}.png"
            if not dry_run:
                templates_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), to_lightness(imgs[m]))
            rec["written"] = str(out)
            report.append(rec)
    return report
