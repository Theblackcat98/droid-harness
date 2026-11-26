from __future__ import annotations

from ..candidate import Candidate
from . import RuleContext


class DefendLaneThreat:
    """Ground threats approaching a tower: best counter by effectiveness × urgency − cost."""

    id = "defend.lane_threat"
    air = False

    def propose(self, s, p, ctx: RuleContext, intent):
        for th in s.threats(ctx.arena, ctx.cards, max_tiles_to_tower=p.defend_trigger_tiles):
            if th.unit.flying != self.air:
                continue
            urgency = max(0.0, 1.0 - th.eta_to_tower_s / p.defend_horizon_s)
            if urgency <= 0:
                continue
            for slot, key in s.affordable_hand(ctx.cards):
                card = ctx.cards[key]
                if not card.can_hit(th.unit.flying):
                    continue
                eff = p.counters.get((key, th.kind), 0.0)
                if eff <= 0:
                    continue
                t_exec = ctx.placement.timing(card, th, s)
                tile, predicted = ctx.placement.defensive(card, th, s, t_exec)
                yield Candidate(
                    self.id, key, slot, tile,
                    score=eff * urgency - p.elixir_weight * card.cost / 10,
                    execute_at_ms=t_exec, predicted_tile=predicted,
                    preselect=t_exec - s.t_ms > 200,
                    reason=f"{key} vs {th.kind} eta={th.eta_to_tower_s:.1f}s eff={eff:.2f}")


class DefendAirThreat(DefendLaneThreat):
    id = "defend.air_threat"
    air = True
