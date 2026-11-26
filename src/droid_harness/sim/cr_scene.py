"""A synthetic Clash Royale-like scene for closed-loop tests and demos.

It is *not* the game: cards are procedural icons, units are colored boxes with HP bars.
It exists so the whole loop — H.264 stream -> perception -> policy -> touches -> world
reacting — can run in CI without a phone. Real calibration replaces every visual here.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..core.transport.scrcpy_protocol import MotionAction
from ..domains.clash_royale.cards import CardDB
from ..domains.clash_royale.perception.detector import Detection
from ..domains.clash_royale.perception.grid import Layout, Roi, TileMapper

ARENA_TOP, ARENA_BOTTOM = 0.05, 0.80  # normalized screen y of tile y=32 and y=0


def sim_layout() -> Layout:
    hand = [Roi(0.20 + i * 0.195, 0.835, 0.20 + i * 0.195 + 0.17, 0.955) for i in range(4)]
    return Layout(
        hand=hand, next_card=Roi(0.03, 0.875, 0.15, 0.955), elixir_bar=Roi(0.20, 0.965, 0.98, 0.99),
        clock=Roi(0.80, 0.0, 0.99, 0.04),
        towers={"opp_left": Roi(0.10, 0.17, 0.30, 0.20), "opp_right": Roi(0.70, 0.17, 0.90, 0.20),
                "opp_king": Roi(0.40, 0.06, 0.60, 0.09), "me_left": Roi(0.10, 0.70, 0.30, 0.73),
                "me_right": Roi(0.70, 0.70, 0.90, 0.73), "me_king": Roi(0.40, 0.77, 0.60, 0.80)},
        arena=Roi(0.0, ARENA_TOP, 1.0, ARENA_BOTTOM),
        buttons={"ok": Roi(0.4, 0.85, 0.6, 0.9)},
        elixir_hsv_lo=(140, 80, 120), elixir_hsv_hi=(170, 255, 255))


def sim_tiles() -> TileMapper:
    def scr(tx, ty):
        return (tx / 18.0, ARENA_BOTTOM - (ty / 32.0) * (ARENA_BOTTOM - ARENA_TOP))
    tiles = [(0, 0), (18, 0), (18, 32), (0, 32), (9, 16)]
    return TileMapper.fit([scr(*t) for t in tiles], tiles)


def card_icon(key: str, size: tuple[int, int], greyed: bool = False) -> np.ndarray:
    """Deterministic, high-contrast procedural icon per card."""
    w, h = size
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = rng.integers(30, 90, 3)
    for _ in range(6):
        color = tuple(int(c) for c in rng.integers(60, 255, 3))
        x0, y0 = int(rng.integers(0, w - 6)), int(rng.integers(0, h - 6))
        x1, y1 = int(rng.integers(x0 + 5, w)), int(rng.integers(y0 + 5, h))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
        else:
            cv2.circle(img, ((x0 + x1) // 2, (y0 + y1) // 2), max(3, (x1 - x0) // 2), color, -1)
    cv2.rectangle(img, (0, int(h * 0.78)), (int(w * 0.35), h - 1), (200, 60, 200), -1)  # cost badge
    if greyed:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor((g * 0.7).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return img


def class_color(cls: str) -> tuple[int, int, int]:
    seed = int(hashlib.sha256(cls.encode()).hexdigest()[:6], 16)
    return (40 + seed % 180, 40 + (seed >> 8) % 180, 40 + (seed >> 16) % 180)


@dataclass
class SimUnit:
    cls: str
    side: str
    tile: tuple[float, float]
    vel: tuple[float, float] = (0.0, 0.0)
    hp: float = 1.0
    ready_at_s: float = 0.0  # deploy time


@dataclass
class CRScene:
    cards: CardDB
    deck: list[str]
    width: int = 360
    height: int = 640
    elixir: float = 5.0
    units: list[SimUnit] = field(default_factory=list)
    towers: dict[str, int] = field(default_factory=lambda: {"opp_left": 2534, "opp_right": 2534, "opp_king": 4008,
                                                             "me_left": 2534, "me_right": 2534, "me_king": 4008})
    t_s: float = 0.0
    draw_tower_hp: bool = True

    def __post_init__(self) -> None:
        self.layout = sim_layout()
        self.tiles = sim_tiles()
        self.queue = list(self.deck)
        self.hand = [self.queue.pop(0) for _ in range(4)]
        self.selected: int | None = None
        self.plays: list[tuple[float, str, tuple[float, float]]] = []

    # --- world ----------------------------------------------------------------
    def step(self, dt: float) -> None:
        from ..domains.clash_royale.elixir import generated_between
        self.elixir = min(10.0, self.elixir + generated_between(self.t_s, self.t_s + dt))
        self.t_s += dt
        for u in self.units:
            if self.t_s >= u.ready_at_s:
                u.tile = (u.tile[0] + u.vel[0] * dt, u.tile[1] + u.vel[1] * dt)

    def spawn_enemy(self, cls: str, tile, vel=(0.0, 0.0), hp: float = 1.0) -> None:
        self.units.append(SimUnit(cls, "opp", tuple(tile), tuple(vel), hp))

    def play_card(self, slot: int, tile) -> None:
        key = self.hand[slot]
        card = self.cards[key]
        if card.cost > self.elixir + 1e-6:
            return
        self.elixir -= card.cost
        self.plays.append((self.t_s, key, tuple(tile)))
        if card.kind != "spell":
            for cls in card.unit_classes or (key,):
                for _ in range(max(1, card.units)):
                    self.units.append(SimUnit(cls, "me", tuple(tile), (0.0, 0.0), 1.0,
                                              self.t_s + card.deploy_time_s))
        self.queue.append(key)
        self.hand[slot] = self.queue.pop(0)

    # --- input (from the fake scrcpy server) ---------------------------------------
    def on_touch(self, msg: dict) -> None:
        if msg["action"] != MotionAction.DOWN:
            return
        x = msg["x"] * self.width / max(msg["w"], 1)
        y = msg["y"] * self.height / max(msg["h"], 1)
        for i, roi in enumerate(self.layout.hand):
            x0, y0, x1, y1 = roi.to_px((self.height, self.width))
            if x0 <= x <= x1 and y0 <= y <= y1:
                self.selected = i
                return
        if self.selected is not None:
            tile = self.tiles.px_to_tile((x, y), (self.height, self.width))
            self.play_card(self.selected, tile)
            self.selected = None

    # --- rendering ------------------------------------------------------------------
    def unit_box(self, u: SimUnit) -> tuple[int, int, int, int]:
        cx, cy = self.tiles.tile_to_px(u.tile, (self.height, self.width))
        return (cx - 9, cy - 18, cx + 9, cy)  # foot at the tile

    def render(self) -> np.ndarray:
        img = np.zeros((self.height, self.width, 3), np.uint8)
        img[:] = (60, 110, 60)
        a = self.layout.arena.to_px(img.shape)
        cv2.rectangle(img, (a[0], a[1]), (a[2], a[3]), (70, 140, 70), -1)
        ry0 = self.tiles.tile_to_px((0, 17), img.shape)[1]
        ry1 = self.tiles.tile_to_px((0, 15), img.shape)[1]
        cv2.rectangle(img, (0, ry0), (self.width, ry1), (160, 110, 40), -1)
        for u in self.units:
            x1, y1, x2, y2 = self.unit_box(u)
            cv2.rectangle(img, (x1, y1), (x2, y2), class_color(u.cls), -1)
            bar = (255, 120, 0) if u.side == "me" else (0, 0, 230)
            cv2.rectangle(img, (x1, y1 - 8), (x2, y1 - 3), (30, 30, 30), -1)
            cv2.rectangle(img, (x1, y1 - 8), (x1 + int((x2 - x1) * u.hp), y1 - 3), bar, -1)
        if self.draw_tower_hp:
            for name, roi in self.layout.towers.items():
                x0, y0, x1, y1 = roi.to_px(img.shape)
                cv2.rectangle(img, (x0, y0), (x1, y1), (20, 20, 20), -1)
                cv2.putText(img, str(self.towers[name]), (x0 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)
        hb = self.layout.hand[0].to_px(img.shape)
        cv2.rectangle(img, (0, hb[1] - 6), (self.width, self.height), (40, 30, 30), -1)
        for i, roi in enumerate(self.layout.hand):
            x0, y0, x1, y1 = roi.to_px(img.shape)
            key = self.hand[i]
            img[y0:y1, x0:x1] = card_icon(key, (x1 - x0, y1 - y0), greyed=self.cards.cost(key) > self.elixir)
        x0, y0, x1, y1 = self.layout.next_card.to_px(img.shape)
        img[y0:y1, x0:x1] = card_icon(self.queue[0], (x1 - x0, y1 - y0))
        x0, y0, x1, y1 = self.layout.elixir_bar.to_px(img.shape)
        cv2.rectangle(img, (x0, y0), (x1, y1), (50, 30, 50), -1)
        fill = x0 + int((x1 - x0) * self.elixir / 10)
        cv2.rectangle(img, (x0, y0), (fill, y1), (200, 40, 200), -1)
        return img

    def detections(self, noise_px: float = 0.0, rng=None) -> list[Detection]:
        out = []
        for u in self.units:
            x1, y1, x2, y2 = self.unit_box(u)
            if noise_px and rng is not None:
                dx, dy = rng.normal(0, noise_px, 2)
                x1, x2, y1, y2 = x1 + dx, x2 + dx, y1 + dy, y2 + dy
            out.append(Detection(u.cls, 0.9, (float(x1), float(y1), float(x2), float(y2))))
        return out


class SceneDetector:
    """Detector double that reads the scene's ground truth (pixel boxes, optional noise).

    Follows the real-detector contract: boxes are in the coordinates of the image
    passed in. When the pipeline passes the arena crop, ground-truth full-frame
    boxes are mapped into crop coordinates (the pipeline offsets them back).
    """

    def __init__(self, scene: CRScene, noise_px: float = 0.0, seed: int = 0) -> None:
        self.scene, self.noise_px = scene, noise_px
        self.rng = np.random.default_rng(seed)

    def __call__(self, img: np.ndarray) -> list[Detection]:
        dets = self.scene.detections(self.noise_px, self.rng)
        if img.shape[:2] == (self.scene.height, self.scene.width):
            sx, sy = img.shape[1] / self.scene.width, img.shape[0] / self.scene.height
            return [Detection(d.cls, d.conf, (d.box[0] * sx, d.box[1] * sy, d.box[2] * sx, d.box[3] * sy))
                    for d in dets]
        ax0, ay0, ax1, ay1 = self.scene.layout.arena.to_px((self.scene.height, self.scene.width))
        sx, sy = img.shape[1] / max(1, ax1 - ax0), img.shape[0] / max(1, ay1 - ay0)
        out = []
        for d in dets:
            x1 = (d.box[0] - ax0) * sx
            y1 = (d.box[1] - ay0) * sy
            x2 = (d.box[2] - ax0) * sx
            y2 = (d.box[3] - ay0) * sy
            if x2 <= 0 or y2 <= 0 or x1 >= img.shape[1] or y1 >= img.shape[0]:
                continue  # outside the arena crop
            out.append(Detection(d.cls, d.conf, (x1, y1, x2, y2)))
        return out


def build_matcher_from_scene(scene: CRScene, matcher_cls=None):
    from ..domains.clash_royale.perception.cards import CardMatcher
    m = (matcher_cls or CardMatcher)()
    for key in scene.deck:
        roi = scene.layout.hand[0]
        x0, y0, x1, y1 = roi.to_px((scene.height, scene.width))
        for greyed in (False, True):
            m.add(key, card_icon(key, (x1 - x0, y1 - y0), greyed=greyed))
    return m


def dist(a, b) -> float:
    return math.dist(a, b)
