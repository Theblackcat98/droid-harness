"""Account levels (``configs/accounts/<account>.yaml``): what upgrades you've made.

Recorded on every match (``meta.json`` → ``account_levels``) so experiments can refuse to span
an upgrade, and fed to the policy's level-aware math when a strategy opts in
(``params.level_aware``). ``doctor`` compares them with what perception measured (your
Princess Tower HP gives your King Level) and with each strategy's ``tier0.spell_tower_damage``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .policy.levels import Levels, level_mult


@dataclass
class AccountLevels:
    account: str
    king_level: int | None = None
    cards: dict[str, int | None] = field(default_factory=dict)
    updated: str | None = None
    path: str | None = None
    # Tower damage measured in game (lethal margin 0), authoritative over the card DB estimate.
    spell_tower_damage: dict[str, int] = field(default_factory=dict)

    @property
    def known_cards(self) -> dict[str, int]:
        return {k: int(v) for k, v in self.cards.items() if v}

    @property
    def levels_id(self) -> str:
        blob = json.dumps({"king": self.king_level, "cards": self.known_cards}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:10]

    def to_meta(self) -> dict:
        return {"account": self.account, "king_level": self.king_level, "cards": self.known_cards,
                "levels_id": self.levels_id, "updated": self.updated}

    def policy_levels(self) -> Levels:
        return Levels(king_level=self.king_level, cards=self.known_cards)

    @classmethod
    def load(cls, path: str | Path) -> AccountLevels:
        d = yaml.safe_load(Path(path).read_text()) or {}
        return cls(account=str(d.get("account") or Path(path).stem), king_level=d.get("king_level"),
                   cards=dict(d.get("cards") or {}), updated=str(d.get("updated")) if d.get("updated") else None,
                   path=str(path),
                   spell_tower_damage={k: int(v) for k, v in (d.get("spell_tower_damage") or {}).items()
                                       if isinstance(v, (int, float))})


def load_account(account: str | None, root: Path = Path(".")) -> AccountLevels | None:
    if not account:
        return None
    p = root / "configs/accounts" / f"{account}.yaml"
    return AccountLevels.load(p) if p.exists() else None


def derived_spell_tower_damage(cards, acct: AccountLevels, deck: list[str]) -> dict[str, int]:
    """Crown-tower damage of each spell in ``deck`` at your recorded levels (~10 %/level)."""
    out = {}
    for key in deck:
        c = cards.get(key) if cards is not None else None
        if c is None or not c.is_spell or c.damage <= 0:
            continue
        lvl = acct.known_cards.get(key) or acct.king_level
        out[key] = int(c.damage * level_mult(key, lvl) * c.ct_damage_pct / 100)
    return out


def check_strategy(params, cards, acct: AccountLevels | None) -> dict:
    """What ``doctor`` reports: missing card levels and tower-damage values far from the levels.

    A value measured in game (``spell_tower_damage`` in the account file, lethal margin 0) is
    authoritative: the card DB's per-level damage is coarse (flat below level 4), so an
    estimate must never override a measurement.
    """
    if acct is None:
        return {"account_file": "missing (configs/accounts/<account>.yaml)", "problems": []}
    problems = []
    missing = [k for k in params.deck if not acct.known_cards.get(k)]
    derived = derived_spell_tower_damage(cards, acct, list(params.deck))
    measured = {k: int(v) for k, v in acct.spell_tower_damage.items() if k in derived}
    configured = dict(params.tier0.spell_tower_damage)
    for k, v in {**derived, **measured}.items():
        have = configured.get(k)
        if have is None or abs(have - v) <= max(5, 0.15 * v):
            continue
        src = "your in-game measurement gives" if k in measured else "your levels give"
        problems.append(f"tier0.spell_tower_damage[{k}] = {have}, but {src} ~{v}")
    return {"account_file": acct.path, "king_level": acct.king_level, "levels_id": acct.levels_id,
            "card_levels_missing": missing, "derived_spell_tower_damage": derived,
            "measured_spell_tower_damage": measured, "problems": problems}
