"""Where will a unit be in t seconds? Lane-following motion model.

Ground units walk to their lane's bridge, cross, then head for the target tower; flying
units go straight. Prediction error grows with the horizon, so rules should prefer
placements that don't rely on long look-aheads (beyond ~1.5 s).
"""

from __future__ import annotations

import math

from .arena import Arena, Point
from .state import Unit


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def waypoints(u: Unit, arena: Arena, target: Point) -> list[Point]:
    """Remaining path for an enemy unit (moving toward y=0) or a friendly one (toward y=height)."""
    pts: list[Point] = [u.tile]
    if u.flying:
        return pts + [target]
    lane = arena.lane_of(u.tile[0])
    if u.side == "opp" and u.tile[1] > arena.river_y[0]:
        pts += [arena.bridge(lane, "opp"), arena.bridge(lane, "me")]
    elif u.side == "me" and u.tile[1] < arena.river_y[1]:
        pts += [arena.bridge(lane, "me"), arena.bridge(lane, "opp")]
    return pts + [target]


def _walk(pts: list[Point], distance: float) -> Point:
    for a, b in zip(pts, pts[1:], strict=False):
        seg = _dist(a, b)
        if distance <= seg:
            f = distance / seg if seg else 0.0
            return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
        distance -= seg
    return pts[-1]


def path_distance_to_tower(u: Unit, arena: Arena, tower: Point, stop_short: float | None = None) -> float:
    pts = waypoints(u, arena, tower)
    total = sum(_dist(a, b) for a, b in zip(pts, pts[1:], strict=False))
    return max(0.0, total - (arena.tower_radius if stop_short is None else stop_short))


def advance(u: Unit, t_s: float, arena: Arena, *, speed: float | None = None,
            target: Point | None = None, stop_short: float | None = None,
            alive: dict[str, bool] | None = None) -> Point:
    """Predicted tile after ``t_s`` seconds along the lane path.

    ``alive``: tower name -> alive. Pass the real state: once a princess tower falls, units
    in that lane walk to the king tower, not to the rubble.
    """
    if target is None:
        alive = alive or {}
        side = "opp" if u.side == "opp" else "me"
        _, target = arena.target_tower(side, arena.lane_of(u.tile[0]), alive)
    v = u.speed if speed is None else speed
    pts = waypoints(u, arena, target)
    total = sum(_dist(a, b) for a, b in zip(pts, pts[1:], strict=False))
    reach = max(0.0, total - (arena.tower_radius if stop_short is None else stop_short))
    return _walk(pts, min(v * max(t_s, 0.0), reach))
