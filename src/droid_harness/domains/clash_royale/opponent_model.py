"""Opponent card-cycle tracking.

Decks have 8 cards with 4 in hand. A played card goes to the back of a 4-card queue, so
it returns to hand only after four more plays: once the deck is known, the opponent's
hand is the deck minus the last 4 cards played, and their next card is the oldest of
those 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .elixir import OpponentElixir

DECK_SIZE = 8
HAND_SIZE = 4


@dataclass
class CycleTracker:
    deck_seen: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)

    def on_play(self, card: str) -> None:
        if card not in self.deck_seen and len(self.deck_seen) < DECK_SIZE:
            self.deck_seen.append(card)
        self.history.append(card)

    @property
    def last_four(self) -> list[str]:
        return self.history[-HAND_SIZE:]

    @property
    def deck_known(self) -> bool:
        return len(self.deck_seen) == DECK_SIZE

    def possible_hand(self) -> set[str]:
        """Known cards that may be in hand. Unknown cards are also possible until the deck is known."""
        return set(self.deck_seen) - set(self.last_four)

    def certain_hand(self) -> set[str] | None:
        if not self.deck_known or len(self.history) < HAND_SIZE:
            return None
        return set(self.deck_seen) - set(self.last_four)

    def next_card(self) -> str | None:
        if not self.deck_known or len(self.history) < HAND_SIZE:
            return None
        return self.last_four[0]

    def out_of_cycle(self, card: str) -> bool:
        """True if ``card`` was played in the last 4 plays (cannot be in hand)."""
        return card in self.last_four


@dataclass
class OpponentModel:
    elixir: OpponentElixir = field(default_factory=OpponentElixir)
    cycle: CycleTracker = field(default_factory=CycleTracker)

    def on_play(self, card: str | None, cost: float | tuple[float, float], t_s: float) -> None:
        self.elixir.on_play(cost, t_s)
        if card:
            self.cycle.on_play(card)

    def advance(self, t_s: float) -> None:
        self.elixir.advance(t_s)
