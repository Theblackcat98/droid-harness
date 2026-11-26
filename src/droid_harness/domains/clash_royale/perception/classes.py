"""Detector class naming: one place that knows how class strings are built.

* Units are side-aware: ``<unit>_me`` / ``<unit>_opp``. Your units face up and the
  opponent's face down, so side is visible in the sprite itself (KataCR labels its data
  the same way). ``<unit>`` is the card DB's ``unit_classes`` name.
* Spell effects end in ``_effect`` and carry no side; the pipeline attributes them from
  your own plays.
* Tower states (``tower_*``) are tolerated from older models but routed to tower state,
  never into the unit tracker.
* Plain names (no suffix) still work; their side falls back to the HP bar / spawn prior.
* ``other_ground`` / ``other_air`` are catch-alls for units outside the named set: an
  unknown enemy card becomes a generic threat instead of background.
"""

from __future__ import annotations

from pathlib import Path

SIDE_SUFFIXES = {"_me": "me", "_opp": "opp"}
TOWER_CLASSES = frozenset({"tower_princess", "tower_king", "tower_rubble"})
# Catch-all classes for units outside the named set: base -> flying.
GENERIC_CLASSES = {"other_ground": False, "other_air": True}


def load_class_names(path: str | Path) -> list[str]:
    """One class per line; blank lines and ``#`` comments are ignored. Duplicates are an error."""
    names = []
    for ln in Path(path).read_text().splitlines():
        ln = ln.split("#", 1)[0].strip()
        if ln:
            names.append(ln)
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"{path}: duplicate classes {dupes}")
    return names


def split_side(cls: str) -> tuple[str, str | None]:
    """``knight_opp`` -> (``knight``, ``opp``); ``knight`` -> (``knight``, None)."""
    for suffix, side in SIDE_SUFFIXES.items():
        if cls.endswith(suffix) and len(cls) > len(suffix):
            return cls[: -len(suffix)], side
    return cls, None


def is_effect(base: str) -> bool:
    return base.endswith("_effect")


def is_tower(base: str) -> bool:
    return base in TOWER_CLASSES


def is_generic(base: str) -> bool:
    return base in GENERIC_CLASSES


def flying_of(base: str, card) -> bool:
    """Flying flag for a detected unit: from its card, else from a catch-all class."""
    if card is not None:
        return bool(card.flying)
    return GENERIC_CLASSES.get(base, False)


def base_classes(names: list[str]) -> set[str]:
    return {split_side(n)[0] for n in names}


def side_aware(names: list[str]) -> bool:
    return any(split_side(n)[1] is not None for n in names)


def missing_unit_classes(names: list[str], unit_classes: set[str]) -> list[str]:
    """Card-DB unit classes the detector can't produce (for a load-time warning).

    With side-aware names, a unit class needs both ``_me`` and ``_opp``.
    """
    have = set(names)
    out = []
    for u in sorted(unit_classes):
        ok = u in have if is_effect(u) else (u in have or {f"{u}_me", f"{u}_opp"} <= have)
        if not ok:
            out.append(u)
    return out
