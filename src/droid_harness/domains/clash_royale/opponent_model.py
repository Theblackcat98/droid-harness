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

    def cards_until_back(self, card: str) -> int:
        """Number of other plays required before ``card`` re-enters hand (0 if ready now)."""
        last = self.last_four
        if card not in last:
            return 0
        idx = len(last) - 1 - last[::-1].index(card)
        return HAND_SIZE - (len(last) - 1 - idx)

    @property
    def unknown_slots(self) -> int:
        return max(0, DECK_SIZE - len(self.deck_seen))

    def replace_recent(self, old: str, new: str) -> bool:
        """Upgrade a recently logged card in place (e.g. minions -> minion_horde)."""
        replaced = False
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i] == old:
                self.history[i] = new
                replaced = True
                break
        if old in self.deck_seen:
            if old not in self.history and new not in self.deck_seen:
                self.deck_seen[self.deck_seen.index(old)] = new
            elif new not in self.deck_seen and len(self.deck_seen) < DECK_SIZE:
                self.deck_seen.append(new)
        elif new not in self.deck_seen and len(self.deck_seen) < DECK_SIZE:
            self.deck_seen.append(new)
        return replaced

    def min_cycle_elixir_to_return(self, card: str, cards) -> float:
        """Minimum elixir opponent must spend on OTHER plays to rotate ``card`` back to hand."""
        n = self.cards_until_back(card)
        if n <= 0:
            return 0.0
        in_hand_costs = sorted(
            cards.cost(k) if k in cards else 3.0
            for k in self.possible_hand()
            if k != card
        )
        costs = list(in_hand_costs) + [2.5] * self.unknown_slots
        costs.sort()
        if len(costs) < n:
            costs.extend([2.5] * (n - len(costs)))
        return float(sum(costs[:n]))


@dataclass
class OpponentModel:
    elixir: OpponentElixir = field(default_factory=OpponentElixir)
    cycle: CycleTracker = field(default_factory=CycleTracker)

    def on_play(self, card: str | None, cost: float | tuple[float, float], t_s: float) -> None:
        self.elixir.on_play(cost, t_s)
        if card:
            self.cycle.on_play(card)

    def upgrade_play(self, old_card: str, new_card: str, extra_cost: float, t_s: float) -> None:
        """Apply incremental elixir & replace card in cycle when a multi-unit swarm expands."""
        if extra_cost > 0:
            self.elixir.on_play(extra_cost, t_s)
        else:
            self.elixir.advance(t_s)
        self.cycle.replace_recent(old_card, new_card)

    def advance(self, t_s: float) -> None:
        self.elixir.advance(t_s)
