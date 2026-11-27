"""Card and King levels as numbers the policy can use (opt-in: ``params.level_aware``).

The card DB holds each card's stats at its starting level. In-game stats grow ~10 % per level,
so a stat at level L is ``base × 1.1^(L − start)``, where ``start`` is 1 for commons, 3 rares,
6 epics, 9 legendaries and 11 champions (the unified 1–15 scale the game shows). Approximate,
by design: it's for "does this Fireball kill that Musketeer", not for exact damage.

Your levels come from ``configs/accounts/<account>.yaml`` (``Levels``). The opponent's are
estimated from their King Level, which perception reads off their Princess Tower HP; a card
is assumed to sit at their King Level (never below its starting level).
"""

from __future__ import annotations

from dataclasses import dataclass, field

LEVEL_GROWTH = 1.10

_RARE = """giant musketeer mini_pekka fireball valkyrie hog_rider tombstone goblin_hut goblin_cage battle_ram
wizard ice_golem mega_minion inferno_tower rocket earthquake furnace bomb_tower dart_goblin heal_spirit
elixir_golem battle_healer zappies flying_machine three_musketeers royal_hogs""".split()
_EPIC = """baby_dragon prince dark_prince witch pekka balloon skeleton_army golem x_bow poison freeze mirror
rage clone tornado lightning goblin_barrel barbarian_barrel hunter executioner cannon_cart giant_skeleton
bowler electro_dragon wall_breakers goblin_drill electro_giant""".split()
_LEGENDARY = """the_log princess miner ice_wizard sparky lava_hound inferno_dragon electro_wizard night_witch
bandit royal_ghost magic_archer ram_rider mega_knight graveyard fisherman lumberjack mother_witch phoenix""".split()
_CHAMPION = "archer_queen golden_knight skeleton_king mighty_miner monk little_prince goblinstein".split()
START_LEVEL: dict[str, int] = {**{k: 3 for k in _RARE}, **{k: 6 for k in _EPIC},
                               **{k: 9 for k in _LEGENDARY}, **{k: 11 for k in _CHAMPION}}


def start_level(card: str) -> int:
    return START_LEVEL.get(card, 1)  # commons and anything unlisted


def level_mult(card: str, level: float | None) -> float:
    if not level:
        return 1.0
    return LEVEL_GROWTH ** (max(level, start_level(card)) - start_level(card))


def king_level_from_princess_hp(hp: int | None) -> int | None:
    """Princess Tower max HP -> King Level (1..15), or None if it doesn't match the table."""
    if not hp:
        return None
    table = (1400, 1512, 1624, 1750, 1890, 2030, 2184, 2352, 2534, 2786, 3052, 3346, 3668, 4032, 4424)
    i = min(range(len(table)), key=lambda j: abs(table[j] - hp))
    return i + 1 if abs(table[i] - hp) <= 40 else None


@dataclass
class Levels:
    """Your levels (from the account file). Missing card levels fall back to your King Level."""

    king_level: int | None = None
    cards: dict[str, int] = field(default_factory=dict)

    def mine(self, card: str, fallback_king: int | None = None) -> float:
        lvl = self.cards.get(card) or self.king_level or fallback_king
        return level_mult(card, lvl)

    @staticmethod
    def theirs(card: str, opp_king_level: int | None) -> float:
        return level_mult(card, opp_king_level)


def level_gap(s) -> int:
    """Opponent King Level minus yours (0..4), from the levels perception read; 0 if unknown."""
    mine, theirs = getattr(s, "my_king_level", None), getattr(s, "opp_king_level", None)
    if not mine or not theirs:
        return 0
    return max(0, min(4, theirs - mine))
