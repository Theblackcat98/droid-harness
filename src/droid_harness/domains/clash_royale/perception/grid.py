"""Normalized screen regions and the pixel <-> tile homography."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class Roi:
    """Region in normalized [0, 1] coordinates, so one config works at any capture size."""

    x0: float
    y0: float
    x1: float
    y1: float

    def to_px(self, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        h, w = shape[:2]
        return (int(round(self.x0 * w)), int(round(self.y0 * h)), int(round(self.x1 * w)), int(round(self.y1 * h)))

    def crop(self, img: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.to_px(img.shape)
        return img[y0:y1, x0:x1]

    def center_px(self, shape: tuple[int, ...]) -> tuple[int, int]:
        x0, y0, x1, y1 = self.to_px(shape)
        return ((x0 + x1) // 2, (y0 + y1) // 2)


@dataclass
class Layout:
    """All HUD regions for one phone (``configs/domains/clash_royale/rois/<phone>.yaml``)."""

    hand: list[Roi]
    next_card: Roi
    elixir_bar: Roi
    clock: Roi
    towers: dict[str, Roi]
    arena: Roi
    buttons: dict[str, Roi]
    elixir_hsv_lo: tuple[int, int, int] = (140, 80, 120)
    elixir_hsv_hi: tuple[int, int, int] = (170, 255, 255)
    tower_hp: dict[str, Roi] | None = None
    clock_countdown: bool = False

    def __post_init__(self) -> None:
        if self.tower_hp is None:
            self.tower_hp = dict(self.towers)

    @classmethod
    def load(cls, path: str | Path) -> Layout:
        d = yaml.safe_load(Path(path).read_text())

        def r(v):
            return Roi(*v)

        towers = {k: r(v) for k, v in d["towers"].items()}
        raw_hp = d.get("tower_hp")
        tower_hp = {k: r(v) for k, v in raw_hp.items()} if raw_hp else dict(towers)
        return cls(hand=[r(v) for v in d["hand"]], next_card=r(d["next_card"]), elixir_bar=r(d["elixir_bar"]),
                   clock=r(d["clock"]), towers=towers, tower_hp=tower_hp, arena=r(d["arena"]),
                   buttons={k: r(v) for k, v in d.get("buttons", {}).items()},
                   elixir_hsv_lo=tuple(d.get("elixir_hsv_lo", (140, 80, 120))),
                   elixir_hsv_hi=tuple(d.get("elixir_hsv_hi", (170, 255, 255))),
                   clock_countdown=bool(d.get("clock_countdown", False)))


class TileMapper:
    """Homography between normalized screen coordinates and arena tiles.

    Fit from >= 4 landmarks (bridge ends, tower centers) clicked during calibration.
    Normalized coordinates keep it valid when the capture resolution changes.
    """

    def __init__(self, H: np.ndarray) -> None:
        self.H = np.asarray(H, float)
        self.Hinv = np.linalg.inv(self.H)

    @classmethod
    def fit(cls, screen_norm: list[tuple[float, float]], tiles: list[tuple[float, float]]) -> TileMapper:
        if len(screen_norm) < 4:
            raise ValueError("need at least 4 landmarks")
        H, _ = cv2.findHomography(np.float32(screen_norm), np.float32(tiles), 0)
        if H is None:
            raise ValueError("degenerate landmarks")
        return cls(H)

    @staticmethod
    def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, float).reshape(-1, 2)
        homog = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
        return homog[:, :2] / homog[:, 2:3]

    def to_tiles(self, pts_norm) -> np.ndarray:
        return self._apply(self.H, pts_norm)

    def to_screen(self, tiles) -> np.ndarray:
        return self._apply(self.Hinv, tiles)

    def tile_to_px(self, tile: tuple[float, float], shape: tuple[int, ...]) -> tuple[int, int]:
        h, w = shape[:2]
        x, y = self.to_screen([tile])[0]
        return (int(round(x * w)), int(round(y * h)))

    def px_to_tile(self, px: tuple[float, float], shape: tuple[int, ...]) -> tuple[float, float]:
        h, w = shape[:2]
        t = self.to_tiles([(px[0] / w, px[1] / h)])[0]
        return (float(t[0]), float(t[1]))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump({"homography": self.H.tolist()}))

    @classmethod
    def load(cls, path: str | Path) -> TileMapper:
        return cls(np.array(yaml.safe_load(Path(path).read_text())["homography"]))
