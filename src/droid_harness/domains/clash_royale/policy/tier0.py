"""Tier 0 reflex invariants: deterministic overrides checked before any scoring.

Each still has its thresholds in the strategy file and goes through the same evaluation
funnel as everything else.
"""

from __future__ import annotations

from ..arena import Arena
from ..cards import CardDB
from ..state import GameState
from .candidate import Candidate
from .placement import Placement
from .strategy import Params


class Tier0:
    def __init__(self, cards: CardDB, arena: Arena, placement: Placement) -> None:
        self.cards, self.arena, self.placement = cards, arena, placement
        self.reset()

    def reset(self) -> None:
        self._leak_since_ms: int | None = None
        self._tower_max: dict[str, int] = {}

    def check(self, s: GameState, p: Params) -> Candidate | None:
        for k, hp in s.towers.items():
            self._tower_max[k] = max(self._tower_max.get(k, 0), hp)
        return self.lethal_spell(s, p) or self.emergency_defend(s, p) or self.leak_guard(s, p)

    # --- invariants -----------------------------------------------------------
    def spell_tower_damage(self, card_key: str, p: Params) -> int:
        if card_key in p.tier0.spell_tower_damage:
            return p.tier0.spell_tower_damage[card_key]
        c = self.cards[card_key]
        return int(c.damage * c.ct_damage_pct / 100)

    def lethal_spell(self, s: GameState, p: Params) -> Candidate | None:
        best = None
        for slot, key in s.affordable_hand(self.cards):
            if not self.cards[key].is_spell:
                continue
            dmg = self.spell_tower_damage(key, p)
            for tower in ("opp_king", "opp_left", "opp_right"):  # the king first: it ends the game
                hp = s.towers.get(tower, 0)
                if 0 < hp and dmg >= hp + p.tier0.lethal_margin_hp:
                    cand = Candidate("t0.lethal_spell", key, slot, self.arena.towers[tower], 10.0,
                                     reason=f"{key} {dmg} >= {tower} {hp}")
                    if best is None:
                        best = cand
        return best

    def emergency_defend(self, s: GameState, p: Params) -> Candidate | None:
        for th in s.threats(self.arena, self.cards):
            if th.eta_to_tower_s > p.tier0.emergency_eta_s:
                break  # sorted by ETA
            hp, hp_max = s.towers.get(th.tower, 0), self._tower_max.get(th.tower, 0)
            if not hp_max or hp / hp_max > p.tier0.emergency_hp_frac:
                continue
            options = [(self.cards.cost(k), slot, k) for slot, k in s.affordable_hand(self.cards)
                       if p.counters.get((k, th.kind), 0.0) > 0 and self.cards[k].can_hit(th.unit.flying)]
            if not options:
                continue
            _, slot, key = min(options)
            tile, predicted = self.placement.defensive(self.cards[key], th, s, s.t_ms)
            return Candidate("t0.emergency_defend", key, slot, tile, 9.0, predicted_tile=predicted,
                             reason=f"{th.kind} eta {th.eta_to_tower_s:.1f}s, {th.tower} at {hp}/{hp_max}")
        return None

    def leak_guard(self, s: GameState, p: Params) -> Candidate | None:
        if s.my_elixir < p.tier0.leak_threshold:
            self._leak_since_ms = None
            return None
        if self._leak_since_ms is None:
            self._leak_since_ms = s.t_ms
        if s.t_ms - self._leak_since_ms < p.tier0.leak_ms:
            return None
        hand = s.affordable_hand(self.cards)
        troops = [(slot, k) for slot, k in hand if not self.cards[k].is_spell]
        pref = [(slot, k) for k in p.tier0.leak_card_preference for slot, kk in troops if kk == k]
        if pref:
            slot, key = pref[0]
        else:
            others = [(self.cards.cost(k), slot, k) for slot, k in troops if k not in p.win_conditions]
            if others:
                _, slot, key = min(others)
            else:
                spells = [(self.cards.cost(k), slot, k) for slot, k in hand if self.cards[k].is_spell]
                if not spells:
                    return None
                _, slot, key = min(spells)
        if self.cards[key].is_spell:
            alive = {k: hp for k, hp in s.towers.items() if k.startswith("opp_") and hp > 0}
            tower = min(alive, key=alive.get) if alive else "opp_right"
            return Candidate("t0.leak_guard", key, slot, self.arena.towers[tower], 8.0,
                             reason=f"elixir {s.my_elixir:.1f} (spell on {tower})")
        lanes = {"left": 0, "right": 0}
        for u in s.enemy_units():
            lanes[self.arena.lane_of(u.tile[0])] += 1
        lane = min(lanes, key=lambda k: lanes[k])
        tile = self.placement.snap(tuple(p.tier0.leak_tiles[lane]), self.cards[key], s)
        return Candidate("t0.leak_guard", key, slot, tile, 8.0, reason=f"elixir {s.my_elixir:.1f}")
