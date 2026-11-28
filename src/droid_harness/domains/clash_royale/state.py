"""Canonical Clash Royale game state (what perception produces and policy consumes)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from .arena import Arena, Point
from .cards import SUBSPAWN_SPEED, CardDB

Side = Literal["me", "opp"]


@dataclass
class Unit:
    track_id: int
    kind: str  # detector class
    side: Side
    tile: Point
    vel: Point = (0.0, 0.0)  # tiles / s
    hp_pct: float = 1.0
    first_seen_ms: int = 0
    conf: float = 1.0
    flying: bool = False

    @property
    def speed(self) -> float:
        return math.hypot(*self.vel)


@dataclass
class Threat:
    unit: Unit
    lane: str
    tower: str
    tower_pos: Point
    distance: float  # path distance to the tower edge, tiles
    eta_to_tower_s: float

    @property
    def kind(self) -> str:
        return self.unit.kind


@dataclass
class GameState:
    t_ms: int
    host_ns: int = 0  # perf_counter_ns of the frame this state came from
    frame_size: tuple[int, int] = (0, 0)  # (w, h) of the video frame, for touch coordinates
    phase: str = "x1"
    my_elixir: float = 5.0
    opp_elixir: tuple[float, float] = (5.0, 5.0)
    hand: list[str | None] = field(default_factory=lambda: [None] * 4)
    hand_affordable: list[bool] = field(default_factory=lambda: [False] * 4)
    # Per slot: was the card identity read in *this* frame? A slot that failed to read keeps
    # its last known card (for play detection) but is stale: never act on it. Empty = all fresh.
    hand_fresh: list[bool] = field(default_factory=list)
    next_card: str | None = None
    units: list[Unit] = field(default_factory=list)
    towers: dict[str, int] = field(default_factory=dict)  # hp; 0 = destroyed
    crowns: tuple[int, int] = (0, 0)
    opp_deck_seen: list[str] = field(default_factory=list)
    opp_hand_possible: set[str] = field(default_factory=set)
    opp_last_four: list[str] = field(default_factory=list)
    pocket_open: dict[str, bool] = field(default_factory=lambda: {"left": False, "right": False})
    posture: str = "BALANCED"
    last_action_ms: int = -10_000
    warnings: list[str] = field(default_factory=list)
    # King Levels read from Princess Tower max HP (None until a full-HP tower is read)
    my_king_level: int | None = None
    opp_king_level: int | None = None

    # --- convenience ------------------------------------------------------
    def enemy_units(self) -> list[Unit]:
        return [u for u in self.units if u.side == "opp"]

    def my_units(self) -> list[Unit]:
        return [u for u in self.units if u.side == "me"]

    def tower_alive(self) -> dict[str, bool]:
        return {k: hp > 0 for k, hp in self.towers.items()}

    def slot_fresh(self, i: int) -> bool:
        return not self.hand_fresh or (0 <= i < len(self.hand_fresh) and self.hand_fresh[i])

    def affordable_hand(self, cards: CardDB) -> list[tuple[int, str]]:
        """Affordable cards whose slot was read this frame (stale identities are never played)."""
        out = []
        for i, c in enumerate(self.hand):
            if c is not None and c in cards and cards.cost(c) <= self.my_elixir + 1e-6 and self.slot_fresh(i):
                out.append((i, c))
        return out

    def slot_of(self, card: str) -> int | None:
        try:
            return self.hand.index(card)
        except ValueError:
            return None

    def opp_cards_until_back(self, card: str) -> int:
        last = self.opp_last_four[-4:]
        if card not in last:
            return 0
        idx = len(last) - 1 - last[::-1].index(card)
        return 4 - (len(last) - 1 - idx)

    def opp_can_play_now(self, card: str, cards: CardDB, *, margin: float = 0.5) -> bool:
        if card not in self.opp_deck_seen or card not in cards:
            return False
        if self.opp_cards_until_back(card) != 0:
            return False
        return self.opp_elixir[1] >= cards.cost(card) - margin

    def opp_min_cycle_elixir_to_return(self, card: str, cards: CardDB) -> float:
        """Minimum elixir opponent must spend on OTHER plays before ``card`` returns to hand."""
        n = self.opp_cards_until_back(card)
        if n <= 0:
            return 0.0
        possible_hand = set(self.opp_deck_seen) - set(self.opp_last_four[-4:])
        in_hand_costs = sorted(
            cards.cost(k) if k in cards else 3.0
            for k in possible_hand
            if k != card
        )
        unknown_slots = max(0, 8 - len(self.opp_deck_seen))
        costs = list(in_hand_costs) + [2.5] * unknown_slots
        costs.sort()
        if len(costs) < n:
            costs.extend([2.5] * (n - len(costs)))
        return float(sum(costs[:n]))

    def opp_elixir_advantage(self) -> float:
        return self.my_elixir - (self.opp_elixir[0] + self.opp_elixir[1]) / 2.0

    def threats(self, arena: Arena, cards: CardDB, *, max_tiles_to_tower: float = 99.0,
                speed_floor: float = 0.3) -> list[Threat]:
        from .path import path_distance_to_tower  # local import avoids a cycle

        alive = self.tower_alive()
        out = []
        for u in self.enemy_units():
            lane = arena.lane_of(u.tile[0])
            tower, pos = arena.target_tower("opp", lane, alive)
            dist = path_distance_to_tower(u, arena, pos)
            if dist > max_tiles_to_tower:
                continue
            card = cards.card_for_class(u.kind)
            base_speed = (card.speed_tiles_s if card else
                          SUBSPAWN_SPEED.get(u.kind, 1.0 if u.kind in ("other_ground", "other_air") else 0.0))
            speed = max(u.speed, base_speed, speed_floor)
            out.append(Threat(u, lane, tower, pos, dist, dist / speed))
        return sorted(out, key=lambda t: t.eta_to_tower_s)

    def digest(self) -> dict:
        """Compact, JSON-friendly snapshot for logs and replay diffs."""
        return {
            "t_ms": self.t_ms, "phase": self.phase, "my_elixir": round(self.my_elixir, 2),
            "opp_elixir": self.opp_elixir, "hand": self.hand, "hand_fresh": self.hand_fresh,
            "next": self.next_card,
            "units": [(u.track_id, u.kind, u.side, round(u.tile[0], 2), round(u.tile[1], 2),
                       round(u.vel[0], 2), round(u.vel[1], 2), round(u.hp_pct, 2)) for u in self.units],
            "towers": self.towers, "crowns": self.crowns, "opp_last_four": self.opp_last_four,
            "opp_deck_seen": self.opp_deck_seen, "pocket_open": self.pocket_open,
            "posture": self.posture, "last_action_ms": self.last_action_ms, "warnings": self.warnings,
            "my_king_level": self.my_king_level, "opp_king_level": self.opp_king_level,
        }

    @classmethod
    def from_digest(cls, d: dict) -> GameState:
        units = [Unit(track_id=t, kind=k, side=s, tile=(x, y), vel=(vx, vy), hp_pct=hp)
                 for t, k, s, x, y, vx, vy, hp in d.get("units", [])]
        return cls(t_ms=d["t_ms"], phase=d.get("phase", "x1"), my_elixir=d.get("my_elixir", 5.0),
                   opp_elixir=tuple(d.get("opp_elixir", (5.0, 5.0))), hand=list(d.get("hand", [None] * 4)),
                   hand_fresh=list(d.get("hand_fresh", [])), next_card=d.get("next"), units=units, towers=dict(d.get("towers", {})),
                   crowns=tuple(d.get("crowns", (0, 0))), opp_last_four=list(d.get("opp_last_four", [])),
                   opp_deck_seen=list(d.get("opp_deck_seen", [])), pocket_open=dict(d.get("pocket_open", {})),
                   posture=d.get("posture", "BALANCED"), last_action_ms=d.get("last_action_ms", -10_000),
                   warnings=list(d.get("warnings", [])),
                   my_king_level=d.get("my_king_level"), opp_king_level=d.get("opp_king_level"))
