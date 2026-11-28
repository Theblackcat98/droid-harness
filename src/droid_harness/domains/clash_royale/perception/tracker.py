"""Lightweight multi-object tracker in tile space, with spawn events.

Greedy per-class nearest-neighbour association with a velocity-aware gate. It's small
and dependency-free; swap in ByteTrack (``supervision``) behind the same interface if
crowded fights cause ID switches (watch ``id_switch_suspects``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

_GENERIC_CLS = {"other_ground", "other_air"}
_AIR_CLS = {"minion", "baby_dragon", "other_air", "bat", "balloon", "lava_hound",
            "mega_minion", "inferno_dragon", "flying_machine", "electro_dragon",
            "skeleton_dragon", "phoenix", "phoenix_big", "phoenix_small", "skeleton_barrel", "lava_pup"}
NON_PLAY_SPAWNS = {"goblin_brawler", "golemite", "lava_pup", "elixir_golem_mid",
                   "elixir_golem_small", "hog", "phoenix_egg", "royal_guardian"}
_SWARM_UPGRADES = {
    "minions": ("minion_horde", 4, 6),
    "skeletons": ("skeleton_army", 6, 15),
    "musketeer": ("three_musketeers", 3, 3),
}


def _is_air_cls(cls: str) -> bool:
    return cls in _AIR_CLS or cls == "other_air"


@dataclass
class TileDetection:
    cls: str
    conf: float
    tile: tuple[float, float]
    side: str | None = None  # from a side-aware class or the HP bar, if readable
    hp_pct: float | None = None
    key: str | None = None  # association key; defaults to cls (side-aware models: cls + side)

    @property
    def match_key(self) -> str:
        return self.key or self.cls


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
    cls_votes: dict[str, int] = field(default_factory=dict)
    conf: float = 0.0
    peak_conf: float = 0.0
    spawn_reported: bool = False
    key: str = ""
    river_mid_y: float = 16.0
    hist: list[tuple[int, float, float]] = field(default_factory=list)  # (t_ms, x, y), last ~6
    last_box: tuple[float, float, float, float] | None = None

    @property
    def side(self) -> str:
        if self.side_votes:
            return max(self.side_votes, key=self.side_votes.get)
        return "opp" if self.first_tile[1] > self.river_mid_y else "me"  # spawn-side prior


@dataclass(frozen=True)
class SpawnEvent:
    track_id: int
    cls: str
    side: str
    tile: tuple[float, float]
    t_ms: int
    peak_conf: float = 1.0
    hits: int = 2
    speed: float = 0.0


class Tracker:
    def __init__(self, *, gate_tiles: float = 1.5, max_speed: float = 4.0, max_missed: int = 6,
                 max_missed_ms: int = 500, min_hits: int = 2, vel_alpha: float = 0.5,
                 river_mid_y: float = 16.0, coast: int = 3, conf_alpha: float = 0.35,
                 class_votes: int = 2, coast_ms: int = 200, pos_alpha: float = 0.40,
                 vel_window_ms: int = 350) -> None:
        self.gate_tiles, self.max_speed, self.max_missed = gate_tiles, max_speed, max_missed
        self.max_missed_ms = max_missed_ms
        self.min_hits, self.vel_alpha, self.river_mid_y = min_hits, vel_alpha, river_mid_y
        # ``coast`` / ``coast_ms``: keep publishing a track at its predicted position during short
        # detector dropouts so units don't blink off the map. ``class_votes``: a named class needs
        # this many agreeing frames to replace the catch-all a track was born with.
        self.coast = coast
        self.coast_ms = coast_ms
        self.conf_alpha = conf_alpha
        self.min_class_votes = class_votes
        self.pos_alpha = pos_alpha
        self.vel_window_ms = vel_window_ms
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self.id_switch_suspects = 0
        self._recently_lost: list[Track] = []
        self.last_match: dict[int, int] = {}

    def reset(self) -> None:
        self.__init__(gate_tiles=self.gate_tiles, max_speed=self.max_speed, max_missed=self.max_missed,
                      max_missed_ms=self.max_missed_ms, min_hits=self.min_hits, vel_alpha=self.vel_alpha,
                      river_mid_y=self.river_mid_y, coast=self.coast, conf_alpha=self.conf_alpha,
                      class_votes=self.min_class_votes, coast_ms=self.coast_ms, pos_alpha=self.pos_alpha,
                      vel_window_ms=self.vel_window_ms)

    def _predict(self, t: Track, t_ms: int) -> tuple[float, float]:
        dt = min(max(0.0, (t_ms - t.last_ms) / 1000.0), 0.5)
        return (t.tile[0] + t.vel[0] * dt, t.tile[1] + t.vel[1] * dt)

    @staticmethod
    def _compatible_pass2(tr: Track, d: TileDetection, dist: float) -> bool:
        if d.side is None or tr.side != d.side:
            return False
        if "effect" in tr.cls or "effect" in d.cls:
            return False
        if dist > 1.4:
            return False
        tr_gen = tr.cls in _GENERIC_CLS
        d_gen = d.cls in _GENERIC_CLS
        if tr_gen and d_gen:
            return True  # generic air/ground wobble stays on same track
        if tr_gen or d_gen:
            return _is_air_cls(tr.cls) == _is_air_cls(d.cls)
        return _is_air_cls(tr.cls) == _is_air_cls(d.cls) and dist <= 0.9

    def update(self, dets: list[TileDetection], t_ms: int) -> tuple[list[Track], list[SpawnEvent]]:
        pairs = []
        for tid, tr in self.tracks.items():
            pred = self._predict(tr, t_ms)
            gate = self.gate_tiles + self.max_speed * (t_ms - tr.last_ms) / 1000 * 0.5
            for j, d in enumerate(dets):
                if d.match_key != tr.key:
                    continue
                dist = math.dist(pred, d.tile)
                if dist <= gate:
                    pairs.append((dist, tid, j))
        self.last_match = {}
        pairs.sort()
        used_t, used_d = set(), set()
        for _, tid, j in pairs:
            if tid in used_t or j in used_d:
                continue
            used_t.add(tid)
            used_d.add(j)
            self.last_match[j] = tid
            self._apply(self.tracks[tid], dets[j], t_ms)

        # Pass 2: same-side spatial re-association across class/catch-all flicker
        p2_pairs = []
        for tid, tr in self.tracks.items():
            if tid in used_t:
                continue
            pred = self._predict(tr, t_ms)
            for j, d in enumerate(dets):
                if j in used_d:
                    continue
                dist = math.dist(pred, d.tile)
                if self._compatible_pass2(tr, d, dist):
                    p2_pairs.append((dist, tid, j))
        p2_pairs.sort()
        for _, tid, j in p2_pairs:
            if tid in used_t or j in used_d:
                continue
            used_t.add(tid)
            used_d.add(j)
            self.last_match[j] = tid
            self._apply(self.tracks[tid], dets[j], t_ms)

        for tid, tr in list(self.tracks.items()):
            if tid not in used_t:
                tr.missed += 1
                if tr.missed > self.max_missed and (t_ms - tr.last_ms) >= self.max_missed_ms:
                    self._recently_lost.append(tr)
                    del self.tracks[tid]
        for j, d in enumerate(dets):
            if j not in used_d:
                self.last_match[j] = self._new(d, t_ms)
        self._recently_lost = [t for t in self._recently_lost if t_ms - t.last_ms < 1000]

        events = []
        for tr in self.tracks.values():
            if not tr.spawn_reported and self._spawn_ready(tr):
                tr.spawn_reported = True
                events.append(SpawnEvent(
                    tr.id, tr.cls, tr.side, tr.first_tile, tr.first_ms,
                    peak_conf=tr.peak_conf, hits=tr.hits, speed=round(math.hypot(*tr.vel), 3),
                ))
        confirmed = []
        for t in self.tracks.values():
            if t.hits < self.min_hits:
                continue
            if t.missed == 0:
                confirmed.append(t)
            elif t.missed <= self.coast or (t_ms - t.last_ms) <= self.coast_ms:
                # Coast at the predicted position instead of blinking off the map; update last_ms on
                # the published copy only so pipeline._extrapolate does not add vel * dt a second time.
                confirmed.append(replace(t, tile=self._predict(t, t_ms), last_ms=t_ms, conf=t.conf))
        return confirmed, events

    def _spawn_ready(self, tr: Track) -> bool:
        if tr.hits < self.min_hits:
            return False
        if tr.peak_conf >= 0.85:
            return True
        if tr.cls in _GENERIC_CLS:
            return tr.hits >= 6 and tr.peak_conf >= 0.62
        return tr.peak_conf >= 0.62 or (tr.hits >= 5 and tr.peak_conf >= 0.48)

    def _fit_vel(self, tr: Track) -> tuple[float, float]:
        """Velocity from a least-squares fit over recent raw positions (tiles/s), clamped."""
        h = tr.hist
        if len(h) < 3 or (h[-1][0] - h[0][0]) < 80:
            return tr.vel
        n = len(h)
        t0 = h[0][0]
        ts = [(p[0] - t0) / 1000.0 for p in h]
        tm = sum(ts) / n
        dt2 = sum((t - tm) ** 2 for t in ts)
        if dt2 <= 1e-6:
            return tr.vel
        vx = sum((t - tm) * (p[1] - h[0][1]) for t, p in zip(ts, h, strict=True)) / dt2
        vy = sum((t - tm) * (p[2] - h[0][2]) for t, p in zip(ts, h, strict=True)) / dt2
        speed = math.hypot(vx, vy)
        if speed > self.max_speed:
            k = self.max_speed / speed
            vx, vy = vx * k, vy * k
        return (vx, vy)

    def _vote_class(self, tr: Track) -> str:
        """Sticky class: a named class replaces the catch-all after enough votes; never downgrades."""
        named = {c: n for c, n in tr.cls_votes.items() if c not in _GENERIC_CLS}
        if not named:
            return tr.cls
        best = max(named, key=lambda c: (named[c], c == tr.cls))
        if tr.cls in _GENERIC_CLS:
            return best if named[best] >= self.min_class_votes else tr.cls
        if best != tr.cls and named[best] >= max(self.min_class_votes, tr.cls_votes.get(tr.cls, 0) + 1):
            return best
        return tr.cls

    def _apply(self, tr: Track, d: TileDetection, t_ms: int) -> None:
        dt = (t_ms - tr.last_ms) / 1000.0
        if tr.hits == 1 or dt <= 0:
            new_tile = d.tile
        else:
            pred = self._predict(tr, t_ms)
            a = min(1.0, self.pos_alpha + (1.0 - self.pos_alpha) * min(1.0, dt / 0.25))
            new_tile = (a * d.tile[0] + (1.0 - a) * pred[0], a * d.tile[1] + (1.0 - a) * pred[1])
        if dt > 0:
            inst = ((d.tile[0] - tr.tile[0]) / dt, (d.tile[1] - tr.tile[1]) / dt)
            n = math.hypot(*inst)
            if n > self.max_speed:
                k = self.max_speed / n
                inst = (inst[0] * k, inst[1] * k)
            va = self.vel_alpha if tr.hits > 1 else 1.0
            tr.vel = (va * inst[0] + (1.0 - va) * tr.vel[0], va * inst[1] + (1.0 - va) * tr.vel[1])
        tr.tile, tr.last_ms = new_tile, t_ms
        tr.hist.append((t_ms, d.tile[0], d.tile[1]))
        tr.hist = [p for p in tr.hist if t_ms - p[0] <= self.vel_window_ms][-10:]
        tr.vel = self._fit_vel(tr)
        tr.conf = (1.0 - self.conf_alpha) * tr.conf + self.conf_alpha * d.conf
        tr.peak_conf = max(tr.peak_conf, d.conf)
        tr.hits += 1
        tr.missed = 0
        if d.hp_pct is not None:
            tr.hp_pct = d.hp_pct
        if d.side:
            tr.side_votes[d.side] = tr.side_votes.get(d.side, 0) + 1
        tr.cls_votes[d.cls] = tr.cls_votes.get(d.cls, 0) + 1
        new_cls = self._vote_class(tr)
        if new_cls != tr.cls:
            tr.cls = new_cls
            tr.key = f"{new_cls}|{tr.key.split('|', 1)[1]}" if "|" in tr.key else new_cls

    def _new(self, d: TileDetection, t_ms: int) -> int:
        for lost in self._recently_lost:
            if lost.key == d.match_key and math.dist(lost.tile, d.tile) < self.gate_tiles and t_ms - lost.last_ms < 300:
                self.id_switch_suspects += 1
                break
        tr = Track(self._next_id, d.cls, d.tile, d.tile, t_ms, t_ms, conf=d.conf, peak_conf=d.conf,
                   hp_pct=d.hp_pct if d.hp_pct is not None else 1.0, key=d.match_key,
                   river_mid_y=self.river_mid_y, cls_votes={d.cls: 1},
                   hist=[(t_ms, d.tile[0], d.tile[1])])
        if d.side:
            tr.side_votes[d.side] = 1
        self.tracks[tr.id] = tr
        self._next_id += 1
        return tr.id


@dataclass(frozen=True)
class PlayEvent:
    side: str
    card: str
    tile: tuple[float, float]
    t_ms: int
    units: int
    upgraded_from: str | None = None
    peak_conf: float = 1.0
    hits: int = 2
    speed: float = 0.0


class SpawnGrouper:
    """Groups spawns of one multi-unit card (e.g. 3 skeletons) into a single play."""

    def __init__(self, card_for_class, window_ms: int = 900, radius_tiles: float = 3.5) -> None:
        self.card_for_class = card_for_class  # cls -> Card | None
        self.window_ms, self.radius = window_ms, radius_tiles
        self._open: list[dict] = []

    def add(self, ev: SpawnEvent) -> list[PlayEvent]:
        if ev.cls in NON_PLAY_SPAWNS:
            return []
        card = self.card_for_class(ev.cls)
        if card is None:
            if ev.cls in _GENERIC_CLS:
                for g in self._open:
                    if (g["key"] == ev.cls and g["side"] == ev.side and ev.t_ms - g["t_ms"] <= self.window_ms
                            and math.dist(g["tile"], ev.tile) <= self.radius and g["units"] < 4):
                        g["units"] += 1
                        return []
                self._open.append({"key": ev.cls, "side": ev.side, "tile": ev.tile,
                                   "t_ms": ev.t_ms, "units": 1, "max_units": 4, "upgraded": False})
                return [PlayEvent(ev.side, ev.cls, ev.tile, ev.t_ms, 1,
                                  peak_conf=ev.peak_conf, hits=ev.hits, speed=ev.speed)]
            return []  # e.g. death spawns, effects not tied to a play
        up = _SWARM_UPGRADES.get(card.key)
        cap = up[2] if up else card.units
        for g in self._open:
            if (g["key"] == card.key and g["side"] == ev.side and ev.t_ms - g["t_ms"] <= self.window_ms
                    and math.dist(g["tile"], ev.tile) <= self.radius and g["units"] < g["max_units"]):
                g["units"] += 1
                if up and not g["upgraded"] and g["units"] >= up[1]:
                    g["upgraded"] = True
                    return [PlayEvent(ev.side, up[0], g["tile"], ev.t_ms, up[2], upgraded_from=card.key,
                                      peak_conf=ev.peak_conf, hits=ev.hits, speed=ev.speed)]
                return []
        self._open.append({"key": card.key, "side": ev.side, "tile": ev.tile,
                           "t_ms": ev.t_ms, "units": 1, "max_units": cap, "upgraded": False})
        return [PlayEvent(ev.side, card.key, ev.tile, ev.t_ms, card.units,
                          peak_conf=ev.peak_conf, hits=ev.hits, speed=ev.speed)]  # lowest latency

    def expire(self, t_ms: int) -> None:
        self._open = [g for g in self._open if t_ms - g["t_ms"] <= self.window_ms]
