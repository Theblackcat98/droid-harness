from __future__ import annotations

from ..candidate import Candidate
from . import RuleContext

SERIOUS_SOLO_THREATS = {
    "giant", "hog_rider", "prince", "mini_pekka", "pekka", "balloon", "valkyrie",
    "battle_ram", "ram_rider", "goblin_brawler", "goblin_barrel", "golem", "royal_giant",
}


def _tower_hp_frac(s, tower: str) -> float:
    hp = s.towers.get(tower)
    if hp is None:
        return 1.0  # undamaged/unread tower
    if hp <= 0:
        return 0.0
    max_map = getattr(s, "tower_max_hp", None) or {}
    ref = max_map.get(tower) or (2400 if tower.endswith("_king") else 1400)
    return min(1.0, hp / max(ref, hp, 1))


class DefendLaneThreat:
    """Ground threats approaching a tower: best counter by effectiveness × urgency − cost."""

    id = "defend.lane_threat"
    air = False

    def propose(self, s, p, ctx: RuleContext, intent):
        lane_val: dict[str, float] = {"left": 0.0, "right": 0.0}
        for u in s.enemy_units():
            lane_val[ctx.arena.lane_of(u.tile[0])] += ctx.cards.unit_value(u.kind) * u.hp_pct

        for th in s.threats(ctx.arena, ctx.cards, max_tiles_to_tower=p.defend_trigger_tiles):
            if th.unit.flying != self.air:
                continue
            if (p.ignore_chip_max_threat_value > 0
                    and s.my_elixir < p.tier0.leak_threshold - 0.3
                    and th.kind not in SERIOUS_SOLO_THREATS
                    and lane_val[th.lane] <= p.ignore_chip_max_threat_value
                    and _tower_hp_frac(s, th.tower) >= p.ignore_chip_tower_hp_frac):
                continue  # Tower HP as a resource: absorb minor solo chip to bank push elixir
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
