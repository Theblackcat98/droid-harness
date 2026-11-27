"""Card database.

The real database is generated locally by ``tools/build_card_db.py`` (see README) and is
not committed. Field units: seconds, tiles, tiles/second. Values that depend on your card
levels (spell damage to towers) should be overridden in the strategy file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Card:
    key: str
    name: str
    cost: int
    kind: str  # troop | spell | building
    deploy_time_s: float = 1.0
    units: int = 1
    speed_tiles_s: float = 0.0
    range_tiles: float = 0.0
    hitpoints: int = 0
    targets_air: bool = False
    targets_ground: bool = True
    buildings_only: bool = False
    flying: bool = False
    damage: int = 0  # spells: base damage at level 1
    ct_damage_pct: float = 100.0  # spells: % of damage dealt to crown towers
    projectile_speed_tiles_s: float | None = None  # spells: None = instant
    radius_tiles: float = 0.0
    unit_classes: tuple[str, ...] = field(default_factory=tuple)  # detector classes this card spawns

    @property
    def is_spell(self) -> bool:
        return self.kind == "spell"

    def can_hit(self, flying: bool) -> bool:
        """Can this card damage a unit that is (or isn't) flying? Spells hit both.

        A counter-table row never overrides this: Giant can't defend, Mini P.E.K.K.A can't
        touch Minions.
        """
        if self.is_spell:
            return True
        if self.buildings_only:
            return False
        return self.targets_air if flying else self.targets_ground

    @property
    def unit_value(self) -> float:
        """Elixir value per spawned unit, used for board value."""
        return self.cost / max(self.units, 1)

    def travel_time_s(self, src: tuple[float, float], dst: tuple[float, float]) -> float:
        if not self.projectile_speed_tiles_s:
            return 0.0
        d = ((src[0] - dst[0]) ** 2 + (src[1] - dst[1]) ** 2) ** 0.5
        return d / self.projectile_speed_tiles_s


class CardDB:
    def __init__(self, cards: dict[str, Card], meta: dict | None = None) -> None:
        self.cards = cards
        self.meta = meta or {}
        self._by_class: dict[str, str] = {}
        for c in cards.values():
            for cls in c.unit_classes or (c.key,):
                self._by_class.setdefault(cls, c.key)

    @classmethod
    def load(cls, path: str | Path) -> CardDB:
        raw = json.loads(Path(path).read_text())
        cards = {}
        for key, d in raw["cards"].items():
            d = dict(d)
            d["unit_classes"] = tuple(d.get("unit_classes") or (key,))
            cards[key] = Card(key=key, **{k: v for k, v in d.items() if k != "key"})
        return cls(cards, raw.get("meta"))

    def __getitem__(self, key: str) -> Card:
        return self.cards[key]

    def __contains__(self, key: str) -> bool:
        return key in self.cards

    def get(self, key: str) -> Card | None:
        return self.cards.get(key)

    def cost(self, key: str) -> int:
        return self.cards[key].cost

    def card_for_class(self, detector_class: str) -> Card | None:
        k = self._by_class.get(detector_class)
        return self.cards.get(k) if k else None

    def unit_value(self, detector_class: str) -> float:
        c = self.card_for_class(detector_class)
        if c is not None:
            return c.unit_value
        return 2.0 if detector_class in ("other_ground", "other_air") else 0.0
