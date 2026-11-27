"""How the detector treats cards it was never trained on (todo R6).

KataCR's hand-labeled real frames contain ~150 unit types; our detector names a few dozen and
maps the rest to ``other_ground_*`` / ``other_air_*``. This report takes every labeled unit
whose card has **no named class in the model**, runs the detector on the frame, and counts
what happened to it:

* ``catch_all``: detected as ``other_ground``/``other_air`` (what we want: a generic threat),
* ``wrong_name``: detected as a named class (the policy will treat it as that card: wrong
  cost, wrong counters),
* ``missed``: not detected at all (invisible to the bot).

Per card and overall, plus whether the side (yours/theirs) came out right. Run it after each
detector retrain; the leave-one-card-out variant is the same report on a model trained
without that card.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import cv2

from ..domains.clash_royale.perception.classes import is_effect, is_generic, split_side
from .evaluate import iou
from .katacr import load_alias, load_katacr_names


def unseen_report(detector, part2: str | Path, model_classes: list[str], *, alias_path: str | Path,
                  generic_path: str | Path | None = None, kata_yaml: str | Path | None = None,
                  iou_thr: float = 0.4, limit: int = 0) -> dict:
    part2 = Path(part2)
    kata_names = load_katacr_names(kata_yaml or part2 / "ClashRoyale_detection.yaml")
    named = load_alias(alias_path)  # KataCR name -> our base, named cards only
    generic = load_alias(alias_path, generic_path) if generic_path else named
    bases = {split_side(c)[0] for c in model_classes}
    per: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    frames = 0
    labels = sorted(t for t in part2.rglob("*.txt") if any(t.with_suffix(e).exists() for e in (".jpg", ".png")))
    for txt in labels:
        rows = []
        for ln in txt.read_text().splitlines():
            p = ln.split()
            if len(p) < 5:
                continue
            kata = kata_names.get(int(float(p[0])))
            if kata is None:
                continue
            mapped = generic.get(kata)
            if mapped is None or is_effect(mapped):
                continue  # towers, bars, spells: not units
            if named.get(kata) in bases:
                continue  # the model knows this card by name
            rows.append((kata, int(float(p[5])) if len(p) > 5 else None, tuple(map(float, p[1:5]))))
        if not rows:
            continue
        img_p = next(txt.with_suffix(e) for e in (".jpg", ".png") if txt.with_suffix(e).exists())
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        frames += 1
        h, w = img.shape[:2]
        dets = detector(img)
        for kata, bel, (cx, cy, bw, bh) in rows:
            box = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
            best, best_iou = None, 0.0
            for d in dets:
                v = iou(d.box, box)
                if v > best_iou:
                    best, best_iou = d, v
            if best is None or best_iou < iou_thr:
                outcome = "missed"
            else:
                base, side = split_side(best.cls)
                outcome = "catch_all" if is_generic(base) else "wrong_name"
                if bel is not None and side is not None:
                    per[kata]["side_ok" if side == ("me" if bel == 0 else "opp") else "side_wrong"] += 1
                if outcome == "wrong_name":
                    per[kata][f"as:{base}"] += 1
            per[kata][outcome] += 1
            totals[outcome] += 1
        if limit and frames >= limit:
            break
    n = sum(totals[k] for k in ("catch_all", "wrong_name", "missed"))
    return {
        "frames": frames, "units": n,
        "share": {k: round(totals[k] / n, 3) if n else 0.0 for k in ("catch_all", "wrong_name", "missed")},
        "per_card": {k: dict(v) for k, v in sorted(per.items(), key=lambda kv: -sum(
            kv[1][o] for o in ("catch_all", "wrong_name", "missed")))},
    }


def check_classes(class_names: list[str], cards) -> dict:
    """Which detector classes don't resolve to a card in the card DB (and so can't drive play events)."""
    unresolved, ok = [], []
    for c in class_names:
        base, _ = split_side(c)
        if is_generic(base) or base.startswith("tower_"):
            continue
        (ok if cards.card_for_class(base) is not None else unresolved).append(c)
    return {"classes": len(class_names), "resolved": len(ok), "unresolved": unresolved}
