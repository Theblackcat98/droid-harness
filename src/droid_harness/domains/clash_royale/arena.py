"""Arena geometry in tile coordinates.

x: 0 (left) .. width (right). y: 0 (your back edge) .. height (opponent's back edge).
Defaults follow the commonly used 18 x 32 tile model; positions are approximate and
meant to be confirmed by calibration (``configs/domains/clash_royale/arena.yaml``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

Point = tuple[float, float]


@dataclass(frozen=True)
class Arena:
    width: float = 18.0
    height: float = 32.0
    river_y: tuple[float, float] = (15.0, 17.0)  # river band (my edge, their edge)
    bridges_x: tuple[float, float] = (3.5, 14.5)  # left, right lane bridge centers
    towers: dict[str, Point] = field(default_factory=lambda: {
        "me_left": (3.5, 6.5), "me_right": (14.5, 6.5), "me_king": (9.0, 3.0),
        "opp_left": (3.5, 25.5), "opp_right": (14.5, 25.5), "opp_king": (9.0, 29.0),
    })
    tower_radius: float = 1.5
    pocket_depth: float = 4.0  # tiles past the river unlocked when that lane's princess tower falls
    king_tower_spell_origin: Point = (9.0, 3.0)

    @classmethod
    def load(cls, path: str | Path) -> Arena:
        d = yaml.safe_load(Path(path).read_text()) or {}
        for k in ("river_y", "bridges_x", "king_tower_spell_origin"):
            if k in d:
                d[k] = tuple(d[k])
        if "towers" in d:
            d["towers"] = {k: tuple(v) for k, v in d["towers"].items()}
        return cls(**d)

    def lane_of(self, x: float) -> str:
        return "left" if x < self.width / 2 else "right"

    def on_my_side(self, y: float) -> bool:
        return y < self.river_y[0]

    def bridge(self, lane: str, side: str = "me") -> Point:
        x = self.bridges_x[0 if lane == "left" else 1]
        return (x, self.river_y[0] if side == "me" else self.river_y[1])

    def target_tower(self, attacker_side: str, lane: str, alive: dict[str, bool]) -> tuple[str, Point]:
        """Tower a ground unit in ``lane`` walks to: the lane's princess tower, else the king."""
        defender = "me" if attacker_side == "opp" else "opp"
        princess = f"{defender}_{lane}"
        name = princess if alive.get(princess, True) else f"{defender}_king"
        return name, self.towers[name]

    def deployable(self, tile: Point, kind: str, pocket_open: dict[str, bool]) -> bool:
        x, y = tile
        if not (0 <= x <= self.width and 0 <= y <= self.height):
            return False
        if kind == "spell":
            return True
        if y < self.river_y[0]:
            return True
        lane = self.lane_of(x)
        return pocket_open.get(lane, False) and self.river_y[1] <= y <= self.river_y[1] + self.pocket_depth

    def clamp_deployable(self, tile: Point, kind: str, pocket_open: dict[str, bool]) -> Point:
        x = min(max(tile[0], 0.5), self.width - 0.5)
        y = min(max(tile[1], 0.5), self.height - 0.5)
        if self.deployable((x, y), kind, pocket_open):
            return (x, y)
        lane = self.lane_of(x)
        if pocket_open.get(lane, False) and y > self.river_y[1]:
            return (x, min(y, self.river_y[1] + self.pocket_depth))
        return (x, self.river_y[0] - 0.5)
