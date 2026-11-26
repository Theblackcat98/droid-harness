"""Hard constraints: legality filters. Never learned, never editable by the coach."""

from __future__ import annotations

from dataclasses import dataclass

from ..arena import Arena
from ..cards import CardDB
from ..elixir import generated_between
from ..state import GameState
from .candidate import Candidate


@dataclass
class Constraints:
    arena: Arena
    cards: CardDB
    min_action_gap_ms: int = 350

    def why_not(self, s: GameState, c: Candidate) -> str | None:
        card = self.cards.get(c.card)
        if card is None:
            return "unknown_card"
        if c.slot < 0 or c.slot >= len(s.hand) or s.hand[c.slot] != c.card:
            return "not_in_hand"
        if not s.slot_fresh(c.slot):
            return "stale_slot"  # last read failed: the slot may hold a different card now
        exec_ms = c.execute_at_ms if c.execute_at_ms is not None else s.t_ms
        if exec_ms < s.t_ms:
            return "in_the_past"
        elixir_then = min(10.0, s.my_elixir + generated_between(s.t_ms / 1000, exec_ms / 1000))
        if card.cost > elixir_then + 1e-6:
            return "not_enough_elixir"
        if not self.arena.deployable(c.tile, card.kind, s.pocket_open):
            return "illegal_tile"
        if exec_ms - s.last_action_ms < self.min_action_gap_ms:
            return "too_soon"
        return None

    def ok(self, s: GameState, c: Candidate) -> bool:
        return self.why_not(s, c) is None
