"""A default "how well does this card answer that threat" from card attributes (opt-in: ``counter_prior``).

Counter CSVs are hand-written per deck, and a missing row used to mean "never defend with
this card". With ``params.counter_prior: true``, a missing row falls back to this prior; an
explicit CSV row (including an explicit 0) always wins. ``Card.can_hit`` still gates both.

The prior is deliberately coarse (the card DB has HP, range, unit count, targets and cost,
not DPS or splash): buildings pull building-targeters, cheap swarms and single high-cost
melee troops answer tanks, ranged troops answer most things a little, spells only answer
what they kill.
"""

from __future__ import annotations

from ..cards import Card, CardDB

TANK_HP = 1500
SWARM_HP = 260


def prior_effectiveness(card: Card, threat_kind: str, cards: CardDB, *, flying: bool) -> float:
    if not card.can_hit(flying):
        return 0.0
    threat = cards.card_for_class(threat_kind)
    unknown = threat is None or threat_kind.startswith("other_")
    t_hp = (threat.hitpoints if threat and threat.hitpoints else 0) if not unknown else 0
    t_swarm = (not unknown) and (threat.units >= 3 or 0 < t_hp <= SWARM_HP)
    t_tank = (not unknown) and (t_hp >= TANK_HP)
    t_building_only = (not unknown) and threat.buildings_only

    if card.is_spell:
        if unknown or card.radius_tiles <= 0:
            return 0.0
        kills = card.damage > 0 and t_hp > 0 and card.damage >= t_hp
        return 0.8 if (kills and t_swarm) else 0.45 if kills else 0.1
    if card.kind == "building":
        if t_building_only:
            return 0.8
        return 0.55 if not flying else (0.5 if card.targets_air else 0.0)
    ranged = card.range_tiles >= 4.0
    swarm = card.units >= 3
    heavy_single = card.units <= 1 and card.cost >= 4
    if unknown:
        return 0.5 if ranged else 0.45
    if t_tank or t_building_only:
        return 0.65 if heavy_single else 0.6 if swarm else 0.45
    if t_swarm:
        return 0.3 if swarm else 0.5 if ranged else 0.4
    return 0.55 if ranged else 0.5 if heavy_single else 0.45


def effectiveness(p, key: str, threat_kind: str, cards: CardDB, *, flying: bool) -> float:
    """The CSV row if there is one, else the prior when the strategy opts in, else 0."""
    row = p.counters.get((key, threat_kind))
    if row is not None:
        return row
    if getattr(p, "counter_prior", False) and key in cards:
        return prior_effectiveness(cards[key], threat_kind, cards, flying=flying) * p.counter_prior_scale
    return 0.0
