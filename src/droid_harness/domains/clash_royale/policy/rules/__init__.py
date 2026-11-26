"""Typed, unit-testable strategy rules. Each proposes scored candidates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from .....core.runtime.intent_bus import Intent
from ...arena import Arena
from ...cards import CardDB
from ...state import GameState
from ..candidate import Candidate
from ..placement import Placement
from ..strategy import Params


@dataclass
class RuleContext:
    cards: CardDB
    arena: Arena
    placement: Placement


class Rule(Protocol):
    """Rules may also define ``reset()`` (new match) and ``on_played(card, t_ms, params)``
    (perception confirmed our card left the hand); CRPolicy calls them when present."""

    id: str

    def propose(self, s: GameState, p: Params, ctx: RuleContext, intent: Intent | None) -> Iterable[Candidate]:
        ...


def default_rules() -> list[Rule]:
    from .blind import BlindPush
    from .cycle import CheapCycle, OpeningSafeFirstPlay
    from .defend import DefendAirThreat, DefendLaneThreat
    from .offense import CounterPush, WinConditionWhenOppLow
    from .spells import SpellTowerChip, SpellValue

    return [DefendLaneThreat(), DefendAirThreat(), WinConditionWhenOppLow(), CounterPush(), SpellValue(),
            SpellTowerChip(), OpeningSafeFirstPlay(), CheapCycle(), BlindPush()]
