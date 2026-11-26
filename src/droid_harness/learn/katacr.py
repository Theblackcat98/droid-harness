"""KataCR's hand-labeled real frames -> our classes, in Ultralytics layout.

KataCR (MIT) ships ~7,400 hand-labeled real arena frames under ``images/part2`` (its own
validation set). That is a held-out *real* recall number for our detector before a single
frame of ours is labeled.

Their label rows have 12 columns: ``cls cx cy w h bel`` + 6 state flags, where ``bel`` is
the side (0 = bottom/yours, 1 = top/theirs). Ultralytics wants 5 columns, so rows are
mapped through ``models/katacr_alias.json`` to our names (side-aware when our class list
is), and everything we don't detect is dropped: those units become background, which is
what a closed-set detector should learn.

Frames are 568x896 arena crops, the same framing our pipeline feeds the detector.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path

import yaml

from ..domains.clash_royale.perception.classes import is_effect


def load_katacr_names(yaml_path: str | Path) -> dict[int, str]:
    """``ClashRoyale_detection.yaml`` in part2: ``names: {0: king-tower, ...}``."""
    d = yaml.safe_load(Path(yaml_path).read_text())
    names = d["names"]
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(k): str(v) for k, v in names.items()}


def load_alias(path: str | Path, generic: str | Path | None = None) -> dict[str, str]:
    """KataCR name -> our base class. ``generic`` (models/katacr_generic.json) adds
    evolutions of named cards and the ``other_ground``/``other_air`` catch-alls."""
    alias = dict(json.loads(Path(path).read_text()))
    if generic is not None and Path(generic).exists():
        g = json.loads(Path(generic).read_text())
        for kata, ours in g.get("named", {}).items():
            alias.setdefault(kata, ours)
        for base in ("other_ground", "other_air"):
            for kata in g.get(base, []):
                alias.setdefault(kata, base)
    return alias


def our_class(kata_name: str, bel: int | None, alias: dict[str, str], names: list[str]) -> str | None:
    base = alias.get(kata_name)
    if base is None:
        return None
    if is_effect(base):
        return base if base in names else None
    if bel is not None:
        sided = f"{base}_{'me' if bel == 0 else 'opp'}"
        if sided in names:
            return sided
    return base if base in names else None


def convert_line(line: str, kata_names: dict[int, str], alias: dict[str, str], names: list[str]) -> str | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    kata = kata_names.get(int(float(parts[0])))
    bel = int(float(parts[5])) if len(parts) > 5 else None
    cls = our_class(kata, bel, alias, names) if kata else None
    if cls is None:
        return None
    return " ".join([str(names.index(cls)), *parts[1:5]])


def convert(part2: str | Path, out_dir: str | Path, names: list[str], alias: dict[str, str], *,
            kata_yaml: str | Path | None = None, val_frac: float = 0.2, empty_keep: float = 0.1,
            seed: int = 0, link: bool = True) -> tuple[Path, Counter]:
    """Convert every ``*.txt`` label with a sibling image under ``part2`` into train/val.

    The split is by *source sequence* (``<video>/<episode>``), never by frame: adjacent
    frames of one episode are near-duplicates, so a random frame split would make val
    look easier than it is. Frames with none of our classes are kept with probability
    ``empty_keep`` as negatives. Images are symlinked (``link``) or copied.
    Returns (data.yaml, counts); ``groups.json`` records which sequence went where.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be in (0, 1): train and val must both exist and be disjoint")
    part2, out_dir = Path(part2), Path(out_dir)
    kata_names = load_katacr_names(kata_yaml or part2 / "ClashRoyale_detection.yaml")
    rng = random.Random(seed)
    labels = sorted(t for t in part2.rglob("*.txt")
                    if any(t.with_suffix(e).exists() for e in (".jpg", ".png")))
    groups = sorted({_group(t, part2) for t in labels})
    split_of = assign_splits(groups, val_frac, seed)
    counts: Counter = Counter()
    for txt in labels:
        img = next(txt.with_suffix(e) for e in (".jpg", ".png") if txt.with_suffix(e).exists())
        rows = [r for r in (convert_line(ln, kata_names, alias, names) for ln in txt.read_text().splitlines())
                if r is not None]
        if not rows and rng.random() >= empty_keep:
            counts["frames_dropped_empty"] += 1
            continue
        split = split_of[_group(txt, part2)]
        img_dir, lbl_dir = out_dir / "images" / split, out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        stem = "_".join(txt.relative_to(part2).with_suffix("").parts)
        dst = img_dir / f"{stem}{img.suffix}"
        if not dst.exists():
            if link:
                os.symlink(img.resolve(), dst)
            else:
                import shutil
                shutil.copy(img, dst)
        (lbl_dir / f"{stem}.txt").write_text("".join(r + "\n" for r in rows))
        counts[f"frames_{split}"] += 1
        counts.update(f"{split}:{names[int(r.split()[0])]}" for r in rows)
    (out_dir / "groups.json").write_text(json.dumps(split_of, indent=1))
    data = out_dir / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(out_dir.resolve()), "train": "images/train",
                                    "val": "images/val", "names": dict(enumerate(names))}, sort_keys=False))
    return data, counts


def _group(label: Path, root: Path) -> str:
    parts = label.relative_to(root).parts[:-1]
    return "/".join(parts[:2]) if parts else "."


def assign_splits(groups: list[str], val_frac: float, seed: int = 0) -> dict[str, str]:
    """Deterministic group -> split. With 2+ groups, both splits get at least one."""
    import hashlib

    def h(g: str) -> float:
        return int(hashlib.sha1(f"{seed}:{g}".encode()).hexdigest(), 16) / 16 ** 40

    out = {g: "val" if h(g) < val_frac else "train" for g in groups}
    if len(groups) >= 2:
        for want in ("val", "train"):
            if want not in out.values():
                pick = (min if want == "val" else max)(groups, key=h)
                out[pick] = want
    return out
