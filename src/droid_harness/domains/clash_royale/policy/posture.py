"""Deterministic macro postures. They rescale rule-family weights via the strategy file."""

from __future__ import annotations

import math

from ....core.runtime.intent_bus import Intent
from ..cards import CardDB
from ..state import GameState
from .strategy import Params

POSTURES = ("OPENING", "BALANCED", "DEFEND_LEAD", "PRESS", "SPELL_CYCLE")


def spell_casts_to_finish(s: GameState, p: Params, cards: CardDB, tower_damage) -> int | None:
    """Fewest casts of your best deck spell that would destroy some enemy tower."""
    spells = [k for k in p.deck if k in cards and cards[k].is_spell]
    if not spells:
        return None
    best = max(tower_damage(k, p) for k in spells)
    if best <= 0:
        return None
    alive = [hp for k, hp in s.towers.items() if k.startswith("opp_") and hp > 0]
    if not alive:
        return None
    return math.ceil(min(alive) / best)


def classify(s: GameState, p: Params, cards: CardDB, tower_damage, intent: Intent | None = None) -> str:
    if intent is not None and intent.posture in p.advisor_may_set:
        return intent.posture  # allowlisted advisor override
    overtime = s.phase in ("ot_x2", "ot_x3")
    casts = spell_casts_to_finish(s, p, cards, tower_damage)
    if overtime and casts is not None and casts <= p.spell_cycle_max_casts:
        return "SPELL_CYCLE"
    if s.crowns[0] > s.crowns[1] and s.phase != "x1":
        return "DEFEND_LEAD"
    if s.t_ms < p.opening_ms and not s.enemy_units():
        return "OPENING"
    if s.opp_elixir[1] <= p.press_opp_elixir_max:
        return "PRESS"
    return "BALANCED"
