"""Tower-HP read audit over recorded video (``droid-harness hp-audit``).

For each match with ``stream.h264``: decode ~1 frame per second, read every tower's HP label
at its calibrated region and with the offset search, and report per-tower coverage, the
offsets that worked, and sample crops. Run it on the machine that keeps the videos (veritas)
after a calibration change, or when ``summary.json`` shows ``tower_hp_diff: null``.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from .perception.digits import DigitReader, locate_number
from .perception.grid import Layout, Roi
from .replay import iter_frames


def audit_match(match_dir: Path, layout: Layout, digits: DigitReader, *, every_ms: int = 1000,
                crops_out: Path | None = None, max_crops: int = 3) -> dict:
    meta = json.loads((match_dir / "meta.json").read_text()) if (match_dir / "meta.json").exists() else {}
    start = meta.get("start_pts_us")
    rois = layout.tower_hp or layout.towers
    fixed, searched, frames = Counter(), Counter(), 0
    offsets: dict[str, Counter] = defaultdict(Counter)
    saved: Counter = Counter()
    last = None
    for f in iter_frames(match_dir):
        if start is not None and f.pts_us < start:
            continue
        t_ms = (f.pts_us - (start or f.pts_us)) / 1000
        if last is not None and t_ms - last < every_ms:
            continue
        last = t_ms
        frames += 1
        h, w = f.img.shape[:2]
        for name, roi in rois.items():
            if digits.read(roi.crop(f.img)) is not None:
                fixed[name] += 1
            v, off, _ = locate_number(f.img, roi, digits)
            if v is None:
                continue
            searched[name] += 1
            offsets[name][(round(off[0] * w), round(off[1] * h))] += 1
            if crops_out is not None and saved[name] < max_crops:
                r = Roi(roi.x0 + off[0], roi.y0 + off[1], roi.x1 + off[0], roi.y1 + off[1])
                crops_out.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(crops_out / f"{match_dir.name}_{name}_{int(t_ms)}_{v}.png"), r.crop(f.img))
                saved[name] += 1
    return {
        "match": match_dir.name, "frames": frames,
        "coverage_fixed": {k: round(fixed[k] / frames, 3) if frames else 0.0 for k in rois},
        "coverage_searched": {k: round(searched[k] / frames, 3) if frames else 0.0 for k in rois},
        "offsets_px": {k: [[list(o), n] for o, n in c.most_common(3)] for k, c in offsets.items()},
    }


def suggest_rois(reports: list[dict], layout: Layout, size: tuple[int, int] = (464, 1024)) -> dict[str, list[float]]:
    """Calibrated regions moved by each tower's most common working offset (for rois/<phone>.yaml)."""
    w, h = size
    votes: dict[str, Counter] = defaultdict(Counter)
    for r in reports:
        for name, offs in r["offsets_px"].items():
            for (dx, dy), n in offs:
                votes[name][(dx, dy)] += n
    out = {}
    for name, roi in (layout.tower_hp or layout.towers).items():
        if not votes.get(name):
            continue
        (dx, dy), _ = votes[name].most_common(1)[0]
        out[name] = [round(roi.x0 + dx / w, 4), round(roi.y0 + dy / h, 4),
                     round(roi.x1 + dx / w, 4), round(roi.y1 + dy / h, 4)]
    return out
