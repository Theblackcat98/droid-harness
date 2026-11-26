"""Synthetic arena-detection data (plan §6.3, KataCR-style), in the inference geometry.

Inference sees the *arena crop* letterboxed to the model input (``detector.letterbox``).
Synthetic images are built the same way, so the model never trains on a geometry it
won't see:

* backgrounds are arena crops (``tools/extract_backgrounds.py`` writes them), letterboxed
  to ``imgsz`` with the same function and padding as inference;
* sprites are pasted inside the image area only (never over padding), scaled to *this*
  phone's tile size: KataCR cut its sprites from 568-px-wide arena crops, so a sprite is
  scaled by ``our_arena_width / 568`` times the letterbox ratio, with a small jitter;
* sprites are drawn back to front by foot y, as the game draws them;
* HP bars (real KataCR bar sprites when given, else drawn) go on units only, colored by
  side, on top of everything, and are never labeled;
* a sprite is labeled only if at least ``min_visible`` of its pixels survive occlusion
  and clipping: slivers teach the model to fire on noise.

Labels use the tight alpha bounding box of the pasted sprite.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..domains.clash_royale.perception.classes import is_effect, split_side
from ..domains.clash_royale.perception.detector import letterbox

KATACR_ARENA_WIDTH_PX = 568.0  # KataCR frames and sprites come from 568x896 arena crops
_KATACR_FILE = re.compile(r"_(?P<bel>[01])_\d+\.png$")


@dataclass
class Sprite:
    img: np.ndarray  # BGRA cut-out
    cls: str  # our class name: side-aware unit (knight_opp) or effect (fireball_effect)


@dataclass
class PlacedBox:
    cls: str
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in output pixels

    def yolo(self, names: list[str], w: int, h: int) -> str:
        x1, y1, x2, y2 = self.box
        return (f"{names.index(self.cls)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")


@dataclass
class SynthConfig:
    imgsz: tuple[int, int] = (640, 384)  # (h, w): the static export shape
    src_arena_width_px: float = KATACR_ARENA_WIDTH_PX
    scale_jitter: float = 0.1
    min_units: int = 2
    max_units: int = 10
    hp_bar_prob: float = 0.7
    min_visible: float = 0.4
    seed: int = 0


def _tight_box(alpha: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(alpha > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bgra(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 4:
        return img
    rgb = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])


def _paste(canvas: np.ndarray, owner: np.ndarray | None, sprite: np.ndarray, x: int, y: int,
           area: tuple[int, int, int, int], idx: int = 0) -> None:
    """Alpha-paste ``sprite`` with its top-left at (x, y), clipped to ``area``; marks ``owner``."""
    ax0, ay0, ax1, ay1 = area
    h, w = sprite.shape[:2]
    x0, y0, x1, y1 = max(ax0, x), max(ay0, y), min(ax1, x + w), min(ay1, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    src = sprite[y0 - y:y1 - y, x0 - x:x1 - x]
    a = src[:, :, 3:4].astype(np.float32) / 255.0
    dst = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = (a * src[:, :, :3] + (1 - a) * dst).astype(np.uint8)
    if owner is not None:
        owner[y0:y1, x0:x1][src[:, :, 3] > 127] = idx


def draw_hp_bar(img: np.ndarray, box: tuple[float, float, float, float], frac: float, side: str | None,
                bars: dict[str, list[np.ndarray]] | None = None, rng: random.Random | None = None) -> None:
    """HP bar above ``box``: a real bar sprite for that side when available, else a drawn one."""
    x1, y1, x2, _ = (int(round(v)) for v in box)
    bw = max(8, x2 - x1)
    side = side or "opp"
    cands = (bars or {}).get(side) or []
    if cands:
        bar = (rng or random).choice(cands)
        bh = max(3, int(round(bar.shape[0] * bw / max(1, bar.shape[1]))))
        bar = cv2.resize(_bgra(bar), (bw, bh), interpolation=cv2.INTER_AREA)
        _paste(img, None, bar, x1, max(0, y1 - bh - 2), (0, 0, img.shape[1], img.shape[0]))
        return
    by = max(0, y1 - 8)
    color = (255, 0, 0) if side == "me" else (0, 0, 255)  # BGR: blue = yours, red = theirs
    cv2.rectangle(img, (x1, by), (x1 + bw, by + 4), (20, 20, 20), -1)
    cv2.rectangle(img, (x1, by), (x1 + int(bw * frac), by + 4), color, -1)


def compose(background: np.ndarray, sprites: list[Sprite], cfg: SynthConfig, rng: random.Random,
            bars: dict[str, list[np.ndarray]] | None = None,
            feet: list[tuple[float, float]] | None = None) -> tuple[np.ndarray, list[PlacedBox]]:
    """One synthetic image: ``background`` is an arena crop; ``sprites`` are the ones to place.

    ``feet`` optionally pins each sprite's bottom-center in output pixels (tests).
    """
    canvas, r, (left, top) = letterbox(background, cfg.imgsz)
    bh, bw = background.shape[:2]
    area = (left, top, left + int(round(bw * r)), top + int(round(bh * r)))
    base_scale = (bw / cfg.src_arena_width_px) * r

    placed = []  # (foot_y, sprite_img, x, y, cls)
    for j, sp in enumerate(sprites):
        s = base_scale * rng.uniform(1 - cfg.scale_jitter, 1 + cfg.scale_jitter)
        img = _bgra(sp.img)
        nw, nh = max(2, int(round(img.shape[1] * s))), max(2, int(round(img.shape[0] * s)))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        if feet is not None:
            fx, fy = feet[j]
        else:
            fx = rng.uniform(area[0], area[2])
            fy = rng.uniform(area[1] + nh * 0.5, area[3])  # keep most of the sprite inside vertically
        placed.append((fy, img, int(round(fx - nw / 2)), int(round(fy - nh)), sp.cls))
    placed.sort(key=lambda p: p[0])  # farther (higher on screen) first; the game draws lower on top

    owner = np.full(canvas.shape[:2], -1, np.int32)
    for i, (_, img, x, y, _) in enumerate(placed):
        _paste(canvas, owner, img, x, y, area, i)

    boxes: list[PlacedBox] = []
    bar_jobs = []
    for i, (_, img, x, y, cls) in enumerate(placed):
        alpha = img[:, :, 3]
        total = int((alpha > 127).sum())
        visible = int((owner == i).sum())
        if total == 0 or visible / total < cfg.min_visible:
            continue
        full = _tight_box(alpha > 127)
        fx0, fy0, fx1, fy1 = full
        # Label the sprite's full (clipped) extent, as KataCR does, not just its visible part.
        x0b = max(area[0], x + fx0)
        y0b = max(area[1], y + fy0)
        x1b = min(area[2], x + fx1)
        y1b = min(area[3], y + fy1)
        if x1b - x0b < 2 or y1b - y0b < 2:
            continue
        box = (float(x0b), float(y0b), float(x1b), float(y1b))
        boxes.append(PlacedBox(cls, box))
        if not is_effect(split_side(cls)[0]) and rng.random() < cfg.hp_bar_prob:
            bar_jobs.append((box, split_side(cls)[1]))
    for box, side in bar_jobs:  # UI layer: bars sit on top of every sprite
        draw_hp_bar(canvas, box, rng.uniform(0.2, 1.0), side, bars, rng)
    return canvas, boxes


def write_yolo(image_path: Path, label_path: Path, img: np.ndarray, boxes: list[PlacedBox],
               names: list[str]) -> Path:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), img)
    h, w = img.shape[:2]
    label_path.write_text("".join(b.yolo(names, w, h) + "\n" for b in boxes))
    return label_path


def write_data_yaml(out_dir: Path, names: list[str], train: str = "images/train",
                    val: str = "images/val") -> Path:
    p = out_dir / "data.yaml"
    p.write_text(yaml.safe_dump({"path": str(out_dir.resolve()), "train": train, "val": val,
                                 "names": dict(enumerate(names))}, sort_keys=False))
    return p


def generate_dataset(backgrounds: list[np.ndarray], sprites: list[Sprite], out_dir: Path, n: int,
                     names: list[str], cfg: SynthConfig | None = None,
                     bars: dict[str, list[np.ndarray]] | None = None, val_frac: float = 0.1) -> Path:
    """Write ``n`` synthetic image/label pairs in Ultralytics layout. Returns ``data.yaml``.

    ``names`` is required: labels index into it, and it becomes the model's embedded names.
    """
    cfg = cfg or SynthConfig()
    unknown = sorted({s.cls for s in sprites} - set(names))
    if unknown:
        raise ValueError(f"sprite classes not in the class list: {unknown}")
    if not backgrounds or not sprites:
        raise ValueError("need at least one background and one sprite")
    rng = random.Random(cfg.seed)
    pool = list(sprites)  # never mutate the caller's list
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    for i in range(n):
        split = "val" if rng.random() < val_frac else "train"
        k = rng.randint(cfg.min_units, cfg.max_units)
        chosen = [rng.choice(pool) for _ in range(k)]
        img, boxes = compose(rng.choice(backgrounds), chosen, cfg, rng, bars)
        counts.update(b.cls for b in boxes)
        stem = f"synth_{i:05d}"
        write_yolo(out_dir / "images" / split / f"{stem}.jpg", out_dir / "labels" / split / f"{stem}.txt",
                   img, boxes, names)
    (out_dir / "label_counts.yaml").write_text(yaml.safe_dump(dict(sorted(counts.items()))))
    return write_data_yaml(out_dir, names)


# --- asset loading -------------------------------------------------------------------
def sprite_side(path: Path) -> str | None:
    """KataCR file names carry the side: ``knight_0_0000059.png`` = yours, ``_1_`` = theirs."""
    m = _KATACR_FILE.search(path.name)
    if not m:
        return None
    return "me" if m["bel"] == "0" else "opp"


def class_for(base: str, side: str | None, names: list[str]) -> str | None:
    """Our class name for a sprite of ``base`` seen on ``side``, or None if the list lacks it."""
    if is_effect(base):
        return base if base in names else None
    if side is not None and f"{base}_{side}" in names:
        return f"{base}_{side}"
    return base if base in names else None


def load_sprites(segment_dir: str | Path, names: list[str]) -> tuple[list[Sprite], Counter]:
    """``<segment_dir>/<our_base_class>/*.png`` (``tools/fetch_katacr.py`` layout)."""
    sprites, counts = [], Counter()
    for d in sorted(p for p in Path(segment_dir).iterdir() if p.is_dir()):
        for f in sorted(d.glob("*.png")):
            cls = class_for(d.name, sprite_side(f), names)
            if cls is None:
                counts[f"skipped:{d.name}"] += 1
                continue
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            sprites.append(Sprite(img, cls))
            counts[cls] += 1
    return sprites, counts


def load_bars(bar_dir: str | Path | None) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {"me": [], "opp": []}
    if bar_dir is None or not Path(bar_dir).exists():
        return out
    for f in sorted(Path(bar_dir).glob("*.png")):
        side = sprite_side(f)
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if side and img is not None:
            out[side].append(img)
    return out


def load_backgrounds(directory: str | Path) -> list[np.ndarray]:
    imgs = []
    for f in sorted(Path(directory).iterdir()):
        if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
            img = cv2.imread(str(f))
            if img is not None:
                imgs.append(img)
    return imgs
