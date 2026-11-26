from __future__ import annotations

from ..candidate import Candidate
from . import RuleContext


def _weaker_lane(s, ctx: RuleContext) -> str:
    """Attack the lane with the lower enemy princess HP; ties go to the lane with fewer enemies."""
    hp = {lane: s.towers.get(f"opp_{lane}", 0) or 10**9 for lane in ("left", "right")}
    crowd = {"left": 0, "right": 0}
    for u in s.enemy_units():
        crowd[ctx.arena.lane_of(u.tile[0])] += 1
    return min(("left", "right"), key=lambda lane: (hp[lane], crowd[lane]))


def unanswered_threat(s, p, ctx: RuleContext) -> bool:
    """An enemy push is coming and nothing of ours is in that lane to meet it."""
    for th in s.threats(ctx.arena, ctx.cards, max_tiles_to_tower=p.defend_trigger_tiles):
        if th.eta_to_tower_s > p.attack_blocked_by_threat_eta_s:
            continue
        defenders = [u for u in s.my_units() if ctx.arena.lane_of(u.tile[0]) == th.lane]
        if not defenders:
            return True
    return False


class WinConditionWhenOppLow:
    """Play a win condition at the bridge when you have elixir and they likely don't —
    unless an unanswered push is on its way (defend first)."""

    id = "offense.win_condition_when_opp_low"

    def propose(self, s, p, ctx: RuleContext, intent):
        if s.my_elixir < p.min_elixir_to_attack or unanswered_threat(s, p, ctx):
            return
        opp_hi = s.opp_elixir[1]
        aggression = intent.aggression if intent is not None else 0.5
        for slot, key in s.affordable_hand(ctx.cards):
            if key not in p.win_conditions:
                continue
            lane = intent.lane if intent is not None and intent.lane else _weaker_lane(s, ctx)
            tile = ctx.placement.snap(tuple(p.placement.bridge_tiles[lane]), ctx.cards[key], s)
            score = (1.0 - opp_hi / 10) * (0.5 + aggression) - p.elixir_weight * ctx.cards.cost(key) / 20
            yield Candidate(self.id, key, slot, tile, score=score,
                            reason=f"opp elixir ≤ {opp_hi:.1f}, lane {lane}")


class CounterPush:
    """Support surviving defenders that are heading back over the bridge."""

    id = "offense.counter_push"

    def propose(self, s, p, ctx: RuleContext, intent):
        groups: dict[str, float] = {"left": 0.0, "right": 0.0}
        lead: dict[str, tuple[float, float]] = {}
        for u in s.my_units():
            if u.vel[1] <= 0.2:  # not advancing
                continue
            lane = ctx.arena.lane_of(u.tile[0])
            groups[lane] += ctx.cards.unit_value(u.kind) * u.hp_pct
            if lane not in lead or u.tile[1] > lead[lane][1]:
                lead[lane] = u.tile
        for lane, value in groups.items():
            if value < 3.0 or lane not in lead:
                continue
            enemies = sum(1 for u in s.enemy_units() if ctx.arena.lane_of(u.tile[0]) == lane)
            for slot, key in s.affordable_hand(ctx.cards):
                card = ctx.cards[key]
                if card.is_spell or card.kind == "building":
                    continue
                tile = ctx.placement.snap((lead[lane][0], lead[lane][1] - 2.0), card, s)
                score = p.counter_push_bonus * value / 10 - 0.1 * enemies - p.elixir_weight * card.cost / 20
                yield Candidate(self.id, key, slot, tile, score=score,
                                reason=f"support {value:.1f} elixir push {lane}")
