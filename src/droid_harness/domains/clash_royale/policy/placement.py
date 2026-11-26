"""Placement: where and when a card should land, leading moving targets.

A card takes effect after action latency + its deploy time (+ projectile travel for
spells). Moving targets keep moving meanwhile, so aim where they'll be.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import path
from ..arena import Arena, Point
from ..cards import Card, CardDB
from ..state import GameState, Threat, Unit
from .strategy import PlacementParams


@dataclass
class Placement:
    arena: Arena
    cards: CardDB
    params: PlacementParams
    exec_latency_s: float = 0.04  # measured p50 from the loopback test, per phone

    # --- lead targeting -----------------------------------------------------
    def effect_delay_s(self, card: Card, s: GameState, t_exec_ms: int, target: Point | None = None) -> float:
        d = (t_exec_ms - s.t_ms) / 1000 + self.exec_latency_s
        if card.is_spell:
            if target is not None:
                d += card.travel_time_s(self.arena.king_tower_spell_origin, target)
        else:
            d += card.deploy_time_s
        return d

    def lead_target(self, card: Card, unit: Unit, s: GameState, t_exec_ms: int) -> Point:
        """Where ``unit`` will be when ``card`` takes effect (two fixed-point passes for spell travel)."""
        speed = self._speed(unit)
        alive = s.tower_alive()
        t = self.effect_delay_s(card, s, t_exec_ms)
        target = path.advance(unit, t, self.arena, speed=speed, alive=alive)
        if card.is_spell and card.projectile_speed_tiles_s:
            for _ in range(2):
                t_total = t + card.travel_time_s(self.arena.king_tower_spell_origin, target)
                target = path.advance(unit, t_total, self.arena, speed=speed, alive=alive)
        return target

    def _speed(self, unit: Unit) -> float:
        if unit.speed > 0.1:
            return unit.speed
        c = self.cards.card_for_class(unit.kind)
        return c.speed_tiles_s if c else 0.0

    # --- defensive placement --------------------------------------------------
    def defensive(self, card: Card, threat: Threat, s: GameState, t_exec_ms: int) -> tuple[Point, Point]:
        """Returns (tile to place, predicted threat tile at effect time)."""
        predicted = self.lead_target(card, threat.unit, s, t_exec_ms)
        if card.kind == "building":
            tile = tuple(self.params.building_pull[threat.lane])
        elif card.is_spell:
            tile = predicted
        elif card.range_tiles >= 2.5:
            # ranged troop: stand off behind the predicted point, toward our side
            tile = (predicted[0], predicted[1] - self.params.ranged_standoff_tiles)
        else:
            tile = predicted
        return self.snap(tile, card, s), predicted

    def timing(self, card: Card, threat: Threat, s: GameState) -> int:
        """When to place. Buildings vs building-targeting units: when the unit reaches the bridge."""
        if card.kind == "building" and threat.unit.tile[1] > self.arena.river_y[0]:
            speed = max(self._speed(threat.unit), 0.3)
            to_bridge = max(0.0, threat.unit.tile[1] - self.arena.river_y[1]) / speed
            wait = max(0.0, to_bridge - card.deploy_time_s - self.exec_latency_s)
            return s.t_ms + int(wait * 1000)
        return s.t_ms

    def snap(self, tile: Point, card: Card, s: GameState) -> Point:
        x, y = self.arena.clamp_deployable(tile, card.kind, s.pocket_open)
        return (round(x * 2) / 2, round(y * 2) / 2)  # half-tile grid
