"""Dataset plumbing: combine sources into one training config, and export your own frames.

The detector trains on several sources (synthetic, KataCR real, your recordings) and is
judged on *real* frames it never saw. ``combine`` writes one Ultralytics ``data.yaml``
whose train and val are lists of image dirs from those sources, and refuses to build one
where an image appears on both sides. ``export_frames`` turns recorded matches into
arena crops for labeling, split by match (never by frame), optionally pre-labeled by the
current detector (the auto-label loop: fix labels, retrain, repeat).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import yaml

from .katacr import assign_splits


def _load(data_yaml: Path) -> dict:
    d = yaml.safe_load(Path(data_yaml).read_text())
    names = d["names"]
    d["names"] = list(names) if isinstance(names, list) else [names[k] for k in sorted(names, key=int)]
    return d


def _split_dirs(data_yaml: Path, split: str) -> list[Path]:
    d = _load(data_yaml)
    root = Path(d.get("path") or Path(data_yaml).parent)
    entries = d.get(split) or []
    entries = entries if isinstance(entries, list) else [entries]
    return [(root / e).resolve() if not Path(e).is_absolute() else Path(e) for e in entries]


def _images(dirs: list[Path]) -> set[Path]:
    out = set()
    for d in dirs:
        if d.exists():
            out |= {p.resolve() for p in d.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    return out


def combine(out: str | Path, train: list[tuple[str | Path, str]], val: list[tuple[str | Path, str]]) -> Path:
    """``train``/``val``: (data.yaml, split) pairs, e.g. [("data/synth/data.yaml", "train"),
    ("data/katacr-real/data.yaml", "train")] and [("data/katacr-real/data.yaml", "val")]."""
    sources = [Path(y) for y, _ in train + val]
    names = _load(sources[0])["names"]
    for y in sources[1:]:
        if _load(y)["names"] != names:
            raise ValueError(f"{y}: class list differs from {sources[0]}")
    train_dirs = [d for y, sp in train for d in _split_dirs(Path(y), sp)]
    val_dirs = [d for y, sp in val for d in _split_dirs(Path(y), sp)]
    overlap = _images(train_dirs) & _images(val_dirs)
    if overlap:
        raise ValueError(f"{len(overlap)} images are in both train and val, e.g. {sorted(overlap)[0]}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump({"train": [str(d) for d in train_dirs], "val": [str(d) for d in val_dirs],
                                   "names": dict(enumerate(names))}, sort_keys=False))
    return out


def export_frames(match_dirs: list[Path], out_dir: str | Path, layout, names: list[str], *,
                  every_ms: int = 1000, val_frac: float = 0.2, detector=None, seed: int = 0) -> tuple[Path, dict]:
    """Arena crops every ``every_ms`` of battle from recorded matches, split by match.

    With ``detector``, each crop gets its predictions as YOLO labels to correct by hand
    (CVAT / Label Studio import YOLO); without, labels are empty files.
    """
    from ..domains.clash_royale.replay import iter_frames

    out_dir = Path(out_dir)
    split_of = assign_splits(sorted(d.name for d in match_dirs), val_frac, seed)
    counts = {"train": 0, "val": 0}
    for d in match_dirs:
        try:
            meta = json.loads((d / "meta.json").read_text())
        except (OSError, ValueError):
            meta = {}
        start = meta.get("start_pts_us")
        split = split_of[d.name]
        next_ms = 0
        for f in iter_frames(d):
            if start is not None and f.pts_us < start:
                continue
            t_ms = (f.pts_us - (start or 0)) // 1000
            if t_ms < next_ms:
                continue
            next_ms = t_ms + every_ms
            x0, y0, x1, y1 = layout.arena.to_px(f.img.shape)
            crop = f.img[y0:y1, x0:x1]
            stem = f"{d.name}__{t_ms:07d}"
            img_p = out_dir / "images" / split / f"{stem}.jpg"
            lbl_p = out_dir / "labels" / split / f"{stem}.txt"
            img_p.parent.mkdir(parents=True, exist_ok=True)
            lbl_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img_p), crop)
            rows = []
            if detector is not None:
                h, w = crop.shape[:2]
                for det in detector(crop):
                    if det.cls not in names:
                        continue
                    bx1, by1, bx2, by2 = det.box
                    rows.append(f"{names.index(det.cls)} {(bx1 + bx2) / 2 / w:.6f} {(by1 + by2) / 2 / h:.6f} "
                                f"{(bx2 - bx1) / w:.6f} {(by2 - by1) / h:.6f}")
            lbl_p.write_text("".join(r + "\n" for r in rows))
            counts[split] += 1
    (out_dir / "groups.json").write_text(json.dumps(split_of, indent=1))
    data = out_dir / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val",
                                    "names": dict(enumerate(names))}, sort_keys=False))
    return data, counts

