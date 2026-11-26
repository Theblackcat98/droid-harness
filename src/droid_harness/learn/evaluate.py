"""Per-class detection metrics on a labeled split, using the detector exactly as the bot runs it.

Aggregate mAP hides what matters for play: missing an enemy Mini P.E.K.K.A loses a tower,
mislabeling a friendly Knight rarely matters. So the report is per class, and the
deployment gate is recall on *threat* classes (``*_opp``) at a fixed IoU, computed by
running the exported ONNX through ``OnnxDetector`` on held-out real frames.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2

from .datasets import _load, _split_dirs


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def read_labels(path: Path, names: list[str], w: int, h: int) -> list[tuple[str, tuple]]:
    out = []
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
        out.append((names[c], ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


def match(dets, gts, iou_thr: float = 0.5) -> dict[str, dict[str, int]]:
    """Greedy per-class matching, highest confidence first. Returns class -> tp/fp/fn."""
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for cls in {c for c, _ in gts} | {d.cls for d in dets}:
        g = [b for c, b in gts if c == cls]
        used = [False] * len(g)
        for d in sorted((d for d in dets if d.cls == cls), key=lambda d: -d.conf):
            best, j_best = 0.0, -1
            for j, b in enumerate(g):
                if not used[j]:
                    v = iou(d.box, b)
                    if v > best:
                        best, j_best = v, j
            if best >= iou_thr:
                used[j_best] = True
                stats[cls]["tp"] += 1
            else:
                stats[cls]["fp"] += 1
        stats[cls]["fn"] += used.count(False)
    return stats


def evaluate(detector, data_yaml: str | Path, split: str = "val", iou_thr: float = 0.5,
             limit: int | None = None) -> dict:
    names = _load(Path(data_yaml))["names"]
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    n = 0
    for d in _split_dirs(Path(data_yaml), split):
        for img_p in sorted(p for p in d.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")):
            if limit is not None and n >= limit:
                break
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            lbl = Path(str(img_p).replace(f"/images/{split}/", f"/labels/{split}/")).with_suffix(".txt")
            gts = read_labels(lbl, names, img.shape[1], img.shape[0])
            for cls, st in match(detector(img), gts, iou_thr).items():
                for k, v in st.items():
                    totals[cls][k] += v
            n += 1
    per_class = {}
    for cls in names:
        t = totals.get(cls, {"tp": 0, "fp": 0, "fn": 0})
        gt = t["tp"] + t["fn"]
        pred = t["tp"] + t["fp"]
        per_class[cls] = {**t, "support": gt, "recall": t["tp"] / gt if gt else None,
                          "precision": t["tp"] / pred if pred else None}
    return {"images": n, "iou": iou_thr, "split": split, "per_class": per_class}


def threat_gate(report: dict, min_recall: float, min_support: int = 20,
                is_threat=lambda c: c.endswith("_opp")) -> tuple[bool, list[str]]:
    """Every threat class with enough examples must reach ``min_recall``. Classes with too
    few examples are listed as failing too: an unmeasured class isn't a passing one."""
    failing = []
    for cls, m in report["per_class"].items():
        if not is_threat(cls):
            continue
        if m["support"] < min_support:
            failing.append(f"{cls}: only {m['support']} labeled examples")
        elif m["recall"] < min_recall:
            failing.append(f"{cls}: recall {m['recall']:.2f} < {min_recall:.2f}")
    return not failing, failing
