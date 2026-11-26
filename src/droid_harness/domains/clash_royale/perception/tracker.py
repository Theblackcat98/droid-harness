"""Lightweight multi-object tracker in tile space, with spawn events.

Greedy per-class nearest-neighbour association with a velocity-aware gate. It's small
and dependency-free; swap in ByteTrack (``supervision``) behind the same interface if
crowded fights cause ID switches (watch ``id_switch_suspects``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class TileDetection:
    cls: str
    conf: float
    tile: tuple[float, float]
    side: str | None = None  # from the HP bar, if readable
    hp_pct: float | None = None


@dataclass
class Track:
    id: int
    cls: str
    tile: tuple[float, float]
    first_tile: tuple[float, float]
    first_ms: int
    last_ms: int
    vel: tuple[float, float] = (0.0, 0.0)
    hp_pct: float = 1.0
    hits: int = 1
    missed: int = 0
    side_votes: dict[str, int] = field(default_factory=dict)
    conf: float = 0.0
    spawn_reported: bool = False

    @property
    def side(self) -> str:
        if self.side_votes:
            return max(self.side_votes, key=self.side_votes.get)
        return "opp" if self.first_tile[1] > 16.0 else "me"  # spawn-side prior


@dataclass(frozen=True)
class SpawnEvent:
    track_id: int
    cls: str
    side: str
    tile: tuple[float, float]
    t_ms: int


class Tracker:
    def __init__(self, *, gate_tiles: float = 1.5, max_speed: float = 4.0, max_missed: int = 6,
                 min_hits: int = 2, vel_alpha: float = 0.5, river_mid_y: float = 16.0) -> None:
        self.gate_tiles, self.max_speed, self.max_missed = gate_tiles, max_speed, max_missed
        self.min_hits, self.vel_alpha, self.river_mid_y = min_hits, vel_alpha, river_mid_y
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self.id_switch_suspects = 0
        self._recently_lost: list[Track] = []

    def reset(self) -> None:
        self.__init__(gate_tiles=self.gate_tiles, max_speed=self.max_speed, max_missed=self.max_missed,
                      min_hits=self.min_hits, vel_alpha=self.vel_alpha, river_mid_y=self.river_mid_y)

    def _predict(self, t: Track, t_ms: int) -> tuple[float, float]:
        dt = (t_ms - t.last_ms) / 1000
        return (t.tile[0] + t.vel[0] * dt, t.tile[1] + t.vel[1] * dt)

    def update(self, dets: list[TileDetection], t_ms: int) -> tuple[list[Track], list[SpawnEvent]]:
        pairs = []
        for tid, tr in self.tracks.items():
            pred = self._predict(tr, t_ms)
            gate = self.gate_tiles + self.max_speed * (t_ms - tr.last_ms) / 1000 * 0.5
            for j, d in enumerate(dets):
                if d.cls != tr.cls:
                    continue
                dist = math.dist(pred, d.tile)
                if dist <= gate:
                    pairs.append((dist, tid, j))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, tid, j in pairs:
            if tid in used_t or j in used_d:
                continue
            used_t.add(tid)
            used_d.add(j)
            self._apply(self.tracks[tid], dets[j], t_ms)
        for tid, tr in list(self.tracks.items()):
            if tid not in used_t:
                tr.missed += 1
                if tr.missed > self.max_missed:
                    self._recently_lost.append(tr)
                    del self.tracks[tid]
        for j, d in enumerate(dets):
            if j not in used_d:
                self._new(d, t_ms)
        self._recently_lost = [t for t in self._recently_lost if t_ms - t.last_ms < 1000]

        events = []
        for tr in self.tracks.values():
            if tr.hits >= self.min_hits and not tr.spawn_reported:
                tr.spawn_reported = True
                events.append(SpawnEvent(tr.id, tr.cls, tr.side, tr.first_tile, tr.first_ms))
        confirmed = [t for t in self.tracks.values() if t.hits >= self.min_hits and t.missed == 0]
        return confirmed, events

    def _apply(self, tr: Track, d: TileDetection, t_ms: int) -> None:
        dt = (t_ms - tr.last_ms) / 1000
        if dt > 0:
            inst = ((d.tile[0] - tr.tile[0]) / dt, (d.tile[1] - tr.tile[1]) / dt)
            a = self.vel_alpha if tr.hits > 1 else 1.0
            tr.vel = (a * inst[0] + (1 - a) * tr.vel[0], a * inst[1] + (1 - a) * tr.vel[1])
        tr.tile, tr.last_ms, tr.conf = d.tile, t_ms, d.conf
        tr.hits += 1
        tr.missed = 0
        if d.hp_pct is not None:
            tr.hp_pct = d.hp_pct
        if d.side:
            tr.side_votes[d.side] = tr.side_votes.get(d.side, 0) + 1

    def _new(self, d: TileDetection, t_ms: int) -> None:
        for lost in self._recently_lost:
            if lost.cls == d.cls and math.dist(lost.tile, d.tile) < self.gate_tiles and t_ms - lost.last_ms < 300:
                self.id_switch_suspects += 1
                break
        tr = Track(self._next_id, d.cls, d.tile, d.tile, t_ms, t_ms, conf=d.conf,
                   hp_pct=d.hp_pct if d.hp_pct is not None else 1.0)
        if d.side:
            tr.side_votes[d.side] = 1
        self.tracks[tr.id] = tr
        self._next_id += 1


@dataclass(frozen=True)
class PlayEvent:
    side: str
    card: str
    tile: tuple[float, float]
    t_ms: int
    units: int


class SpawnGrouper:
    """Groups spawns of one multi-unit card (e.g. 3 skeletons) into a single play."""

    def __init__(self, card_for_class, window_ms: int = 600, radius_tiles: float = 3.0) -> None:
        self.card_for_class = card_for_class  # cls -> Card | None
        self.window_ms, self.radius = window_ms, radius_tiles
        self._open: list[dict] = []

    def add(self, ev: SpawnEvent) -> list[PlayEvent]:
        card = self.card_for_class(ev.cls)
        if card is None:
            return []  # e.g. death spawns, effects not tied to a play
        for g in self._open:
            if (g["card"].key == card.key and g["side"] == ev.side and ev.t_ms - g["t_ms"] <= self.window_ms
                    and math.dist(g["tile"], ev.tile) <= self.radius and g["units"] < card.units):
                g["units"] += 1
                return []
        self._open.append({"card": card, "side": ev.side, "tile": ev.tile, "t_ms": ev.t_ms, "units": 1})
        return [PlayEvent(ev.side, card.key, ev.tile, ev.t_ms, card.units)]  # emit on first unit (lowest latency)

    def expire(self, t_ms: int) -> None:
        self._open = [g for g in self._open if t_ms - g["t_ms"] <= self.window_ms]
