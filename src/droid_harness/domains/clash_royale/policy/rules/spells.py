from __future__ import annotations

import math

from ..candidate import Candidate
from . import RuleContext


class SpellValue:
    """Cast a spell where it hits at least ``spell_value_threshold`` elixir of units (HP-weighted)."""

    id = "spell.value"

    def propose(self, s, p, ctx: RuleContext, intent):
        enemies = s.enemy_units()
        if not enemies:
            return
        for slot, key in s.affordable_hand(ctx.cards):
            card = ctx.cards[key]
            if not card.is_spell or card.radius_tiles <= 0:
                continue
            best = None
            for anchor in enemies:  # candidate centers: each enemy's predicted position
                center = ctx.placement.lead_target(card, anchor, s, s.t_ms)
                hit, value = [], 0.0
                for u in enemies:
                    pu = ctx.placement.lead_target(card, u, s, s.t_ms)
                    if math.dist(pu, center) <= card.radius_tiles:
                        hit.append(u.kind)
                        value += ctx.cards.unit_value(u.kind) * u.hp_pct
                if best is None or value > best[0]:
                    best = (value, center, hit)
            value, center, hit = best
            if value < p.spell_value_threshold:
                continue
            yield Candidate(self.id, key, slot, ctx.placement.snap(center, card, s),
                            score=(value - card.cost) / max(card.cost, 1), predicted_tile=center,
                            reason=f"{key} hits {value:.1f} elixir: {','.join(hit)}")


class SpellTowerChip:
    """Spell the weakest enemy tower. Mostly dormant; the SPELL_CYCLE posture multiplies it up."""

    id = "spell.tower_chip"

    def propose(self, s, p, ctx: RuleContext, intent):
        if s.posture != "SPELL_CYCLE" and s.my_elixir < 9.5:
            return
        alive = {k: hp for k, hp in s.towers.items() if k.startswith("opp_") and hp > 0}
        if not alive:
            return
        tower = min(alive, key=alive.get)
        for slot, key in s.affordable_hand(ctx.cards):
            card = ctx.cards[key]
            if card.is_spell:
                yield Candidate(self.id, key, slot, ctx.arena.towers[tower], score=0.3,
                                reason=f"chip {tower} ({alive[tower]} hp)")
