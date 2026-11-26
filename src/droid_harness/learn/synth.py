"""Synthetic arena-detection data generator (plan §6.3, KataCR-style).

Composites sprite cut-outs onto empty-arena backgrounds with random position,
scale, overlap, HP bars and spell effects, and writes YOLO labels. Sprites and
backgrounds are supplied later (phone captures / KataCR); this module owns the
compositing math so it is unit-testable without assets.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PlacedBox:
    cls_id: int
    cx: float  # normalized 0..1 in output image
    cy: float
    w: float
    h: float


@dataclass
class SynthConfig:
    out_size: tuple[int, int] = (640, 640)  # (w, h)
    min_scale: float = 0.5
    max_scale: float = 1.4
    max_units: int = 8
    hp_bar_prob: float = 0.7
    seed: int = 0


def _paste(bg: np.ndarray, sprite: np.ndarray, x: int, y: int) -> tuple[int, int, int, int]:
    """Alpha-paste sprite onto bg at top-left (x, y); returns clamped box (x1,y1,x2,y2)."""
    h, w = sprite.shape[:2]
    H, W = bg.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return (x1, y1, x1, y1)
    src = sprite[y1 - y:y2 - y, x1 - x:x2 - x]
    if src.shape[2] == 4:
        alpha = (src[:, :, 3:4].astype(np.float32)) / 255.0
        bg[y1:y2, x1:x2, :3] = (alpha * src[:, :, :3] + (1 - alpha) * bg[y1:y2, x1:x2, :3]).astype(np.uint8)
    else:
        bg[y1:y2, x1:x2, :3] = src[:, :, :3]
    return (x1, y1, x2, y2)


def draw_hp_bar(img: np.ndarray, box: tuple[int, int, int, int], frac: float, friendly: bool) -> None:
    x1, y1, x2, _ = box
    bw = max(8, x2 - x1)
    bx, by = x1, max(0, y1 - 8)
    color = (255, 0, 0) if friendly else (0, 0, 255)
    cv2.rectangle(img, (bx, by), (bx + bw, by + 4), (20, 20, 20), -1)
    cv2.rectangle(img, (bx, by), (bx + int(bw * frac), by + 4), color, -1)


def compose(background: np.ndarray, sprites: list[tuple[np.ndarray, int]],
            cfg: SynthConfig, rng: random.Random) -> tuple[np.ndarray, list[PlacedBox]]:
    """Composite up to ``cfg.max_units`` sprites onto background. Returns (image, boxes)."""
    W, H = cfg.out_size
    img = cv2.resize(background, (W, H), interpolation=cv2.INTER_AREA)
    boxes: list[PlacedBox] = []
    for sprite, cls_id in sprites[:cfg.max_units]:
        scale = rng.uniform(cfg.min_scale, cfg.max_scale)
        nw, nh = max(4, int(sprite.shape[1] * scale)), max(4, int(sprite.shape[0] * scale))
        resized = cv2.resize(sprite, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = rng.randint(-nw // 3, W - 2 * nw // 3)
        y = rng.randint(-nh // 3, H - 2 * nh // 3)
        x1, y1, x2, y2 = _paste(img, resized, x, y)
        if x2 <= x1 or y2 <= y1:
            continue
        if rng.random() < cfg.hp_bar_prob:
            draw_hp_bar(img, (x1, y1, x2, y2), rng.uniform(0.2, 1.0), rng.random() < 0.5)
        boxes.append(PlacedBox(cls_id, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                               (x2 - x1) / W, (y2 - y1) / H))
    return img, boxes


def write_yolo(image_path: Path, img: np.ndarray, boxes: list[PlacedBox]) -> Path:
    cv2.imwrite(str(image_path), img)
    label = image_path.with_suffix(".txt")
    with label.open("w") as f:
        for b in boxes:
            f.write(f"{b.cls_id} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}\n")
    return label


def generate_dataset(backgrounds: list[np.ndarray], sprites: list[tuple[np.ndarray, int]],
                     out_dir: Path, n: int, cfg: SynthConfig | None = None) -> int:
    """Write ``n`` synthetic (image, label) pairs. Returns the number written."""
    cfg = cfg or SynthConfig()
    rng = random.Random(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classes.txt").write_text("\n".join(_default_names(max([c for _, c in sprites], default=0))) + "\n")
    for i in range(n):
        bg = rng.choice(backgrounds)
        rng.shuffle(sprites)
        img, boxes = compose(bg, list(sprites), cfg, rng)
        write_yolo(out_dir / f"synth_{i:05d}.jpg", img, boxes)
    return n


def _default_names(max_id: int) -> list[str]:
    try:
        names = [ln.strip() for ln in Path("models/classes.txt").read_text().splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
        if len(names) > max_id:
            return names
    except OSError:
        pass
    return [f"class_{i}" for i in range(max_id + 1)]
