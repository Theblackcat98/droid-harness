from __future__ import annotations

import math

from ..candidate import Candidate
from ..counter_prior import effectiveness
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


def _spell_defend_ok(card, th, predicted, s, p, ctx: RuleContext) -> bool:
    """Block Arrows/Fireball in defend.lane_threat on low-conf phantoms or lone weak/non-killable units."""
    if th.unit.conf < p.spell_min_conf:
        return False
    rad = max(card.radius_tiles, 2.0)
    hit_units = [
        u for u in s.enemy_units()
        if u.conf >= p.spell_min_conf
        and math.dist(ctx.placement.lead_target(card, u, s, s.t_ms), predicted) <= rad
    ]
    if not hit_units:
        return False
    total_val = 0.0
    kills = 0
    for u in hit_units:
        uc = ctx.cards.card_for_class(u.kind)
        max_hp = uc.hitpoints if (uc and uc.hitpoints > 0) else 500
        rem_hp = max(1.0, max_hp * u.hp_pct)
        dmg = card.damage
        dealt_frac = min(rem_hp, dmg) / max(max_hp, 1.0) if dmg > 0 else 0.5 * u.hp_pct
        if dmg > 0 and dmg >= rem_hp:
            kills += 1
        total_val += ctx.cards.unit_value(u.kind) * dealt_frac
    if len(hit_units) == 1:
        # Single unit: only spell-defend if it kills a troop worth at least the spell cost.
        return kills >= 1 and total_val >= max(2.5, card.cost - 0.2)
    return kills >= 1 and total_val >= max(2.0, card.cost - 0.6)


class DefendLaneThreat:
    """Ground threats approaching a tower: best counter by effectiveness × urgency − cost."""

    id = "defend.lane_threat"
    air = False

    def propose(self, s, p, ctx: RuleContext, intent):
        lane_val: dict[str, float] = {"left": 0.0, "right": 0.0}
        for u in s.enemy_units():
            uc = ctx.cards.card_for_class(u.kind)
            if uc is not None and uc.kind == "building":
                continue
            lane_val[ctx.arena.lane_of(u.tile[0])] += ctx.cards.unit_value(u.kind) * u.hp_pct

        wc_in_hand = [k for k in s.hand if k and k in p.win_conditions and k in ctx.cards]
        wc_cost = min((ctx.cards.cost(k) for k in wc_in_hand), default=0)

        for th in s.threats(ctx.arena, ctx.cards, max_tiles_to_tower=p.defend_trigger_tiles):
            if th.unit.flying != self.air:
                continue
            hp_frac = _tower_hp_frac(s, th.tower)
            if (p.ignore_chip_max_threat_value > 0
                    and s.my_elixir < p.tier0.leak_threshold - 0.3
                    and th.kind not in SERIOUS_SOLO_THREATS
                    and lane_val[th.lane] <= p.ignore_chip_max_threat_value
                    and hp_frac >= p.ignore_chip_tower_hp_frac):
                continue  # Tower HP as a resource: absorb minor solo chip to bank push elixir
            # Break 3-elixir starvation: when a win condition (Giant) is sitting in hand and we
            # can't afford it yet, defer defending low-value non-urgent probes against healthy towers.
            if (p.bank_for_wc_max_threat > 0
                    and wc_cost > 0
                    and s.my_elixir < wc_cost + 0.5
                    and th.eta_to_tower_s > 3.5
                    and th.kind not in SERIOUS_SOLO_THREATS
                    and lane_val[th.lane] <= p.bank_for_wc_max_threat
                    and hp_frac >= 0.55):
                continue
            urgency = max(0.0, 1.0 - th.eta_to_tower_s / p.defend_horizon_s)
            if urgency <= 0:
                continue
            # Weight defense priority by total lane threat value so a Giant + Mini P.E.K.K.A push
            # in the opposite lane always outscores a lone Goblin/Archer probe.
            threat_weight = 0.75 + 0.12 * min(lane_val[th.lane], 10.0)
            for slot, key in s.affordable_hand(ctx.cards):
                card = ctx.cards[key]
                if not card.can_hit(th.unit.flying):
                    continue
                eff = effectiveness(p, key, th.kind, ctx.cards, flying=th.unit.flying)
                if eff <= 0:
                    continue
                t_exec = ctx.placement.timing(card, th, s)
                tile, predicted = ctx.placement.defensive(card, th, s, t_exec)
                if card.is_spell and not _spell_defend_ok(card, th, predicted, s, p, ctx):
                    continue
                if not card.is_spell and card.kind != "building":
                    tile = ctx.placement.avoid_clump(tile, card, s, radius=p.avoid_clump_radius)
                yield Candidate(
                    self.id, key, slot, tile,
                    score=eff * urgency * threat_weight - p.elixir_weight * card.cost / 10,
                    execute_at_ms=t_exec, predicted_tile=predicted,
                    preselect=t_exec - s.t_ms > 200,
                    reason=f"{key} vs {th.kind} eta={th.eta_to_tower_s:.1f}s eff={eff:.2f}")


class DefendAirThreat(DefendLaneThreat):
    id = "defend.air_threat"
    air = True
