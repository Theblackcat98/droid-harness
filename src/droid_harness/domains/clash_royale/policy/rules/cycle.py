from __future__ import annotations

from ..candidate import Candidate
from . import RuleContext
from .offense import _weaker_lane


def _cheapest_cycle_card(s, p, ctx: RuleContext):
    opts = [(ctx.cards.cost(k), slot, k) for slot, k in s.affordable_hand(ctx.cards)
            if not ctx.cards[k].is_spell and k not in p.win_conditions]
    if p.save_counters_for_in_hand_threats and len(opts) > 1:
        urgent_tanks = ("giant", "prince", "hog_rider", "goblin_giant", "pekka")
        if any(s.opp_can_play_now(t, ctx.cards) for t in urgent_tanks):
            filtered = [o for o in opts if o[2] not in ("mini_pekka", "minions")]
            if filtered:
                opts = filtered
    return min(opts) if opts else None


class OpeningSafeFirstPlay:
    """Opening: don't reveal a win condition first; drop a cheap card at the back near full elixir."""

    id = "opening.safe_first_play"

    def propose(self, s, p, ctx: RuleContext, intent):
        if s.t_ms >= p.opening_ms or s.enemy_units() or s.my_elixir < 9.0:
            return
        pick = _cheapest_cycle_card(s, p, ctx)
        if pick is None:
            return
        _, slot, key = pick
        tile = ctx.placement.snap(tuple(p.placement.back_tiles["left"]), ctx.cards[key], s)
        yield Candidate(self.id, key, slot, tile, score=0.5, reason="opening, near full elixir")


class CheapCycle:
    """Quiet board, high elixir, no win condition in hand: cycle the cheapest card."""

    id = "cycle.cheap_cycle"

    def propose(self, s, p, ctx: RuleContext, intent):
        if s.my_elixir < 8.0 or s.enemy_units():
            return
        if any(k in p.win_conditions for k in s.hand if k):
            return
        pick = _cheapest_cycle_card(s, p, ctx)
        if pick is None:
            return
        _, slot, key = pick
        lane = _weaker_lane(s, ctx)
        tile = ctx.placement.snap(tuple(p.placement.back_tiles[lane]), ctx.cards[key], s)
        yield Candidate(self.id, key, slot, tile, score=0.2, reason="cycle to win condition")
