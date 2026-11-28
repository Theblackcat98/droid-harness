from __future__ import annotations

import math

from ..candidate import Candidate
from ..levels import Levels
from . import RuleContext


class SpellValue:
    """Cast a spell where it hits at least ``spell_value_threshold`` elixir of units (HP-weighted)."""

    id = "spell.value"

    def propose(self, s, p, ctx: RuleContext, intent):
        enemies = [u for u in s.enemy_units() if u.conf >= p.spell_min_conf]
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
                kills, troop_hits, building_hits = 0, 0, 0
                for u in enemies:
                    pu = ctx.placement.lead_target(card, u, s, s.t_ms)
                    if math.dist(pu, center) <= card.radius_tiles:
                        hit.append(u.kind)
                        u_card = ctx.cards.card_for_class(u.kind)
                        is_bldg = u_card is not None and u_card.kind == "building"
                        if is_bldg:
                            building_hits += 1
                        else:
                            troop_hits += 1
                        if p.kill_aware_spells:
                            max_hp = u_card.hitpoints if (u_card and u_card.hitpoints > 0) else 500
                            dmg = card.damage
                            if p.level_aware:  # ~10 %/level on both sides
                                if u_card is not None:
                                    max_hp *= Levels.theirs(u_card.key, s.opp_king_level)
                                dmg *= ctx.levels.mine(key, s.my_king_level)
                            rem_hp = max(1.0, max_hp * u.hp_pct)
                            # Value is bounded by remaining HP fraction (never value a 7%-HP tank at 100%).
                            dealt_frac = min(rem_hp, dmg) / max(max_hp, 1.0) if dmg > 0 else 0.5 * u.hp_pct
                            is_kill = dmg >= rem_hp and dmg > 0
                            if is_kill:
                                kills += 1
                            lethal_mult = 1.25 if is_kill else 0.75
                            value += ctx.cards.unit_value(u.kind) * dealt_frac * lethal_mult
                        else:
                            value += ctx.cards.unit_value(u.kind) * u.hp_pct
                hits_tower = any(
                    k.startswith("opp_") and hp > 0
                    and math.dist(ctx.arena.towers[k], center) <= card.radius_tiles
                    for k, hp in s.towers.items()
                )
                if p.kill_aware_spells:
                    if building_hits > 0 and troop_hits == 0 and not hits_tower:
                        value *= 0.75  # don't burn a 4e spell on a lone Tombstone / Goblin Cage
                    if kills == 0 and not hits_tower:
                        value *= 0.65  # don't waste Arrows/Fireball on non-killable medium troops
                    if hits_tower:
                        value += 0.8 if card.cost >= 4 else 0.4
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
        if s.posture != "SPELL_CYCLE":
            # Don't throw low-damage Arrows onto a tower at 9.5 elixir when we have troops to play.
            has_troop = any(not ctx.cards[k].is_spell for _, k in s.affordable_hand(ctx.cards))
            if has_troop:
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
