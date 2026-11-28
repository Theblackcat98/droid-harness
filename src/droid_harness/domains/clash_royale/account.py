"""Account levels (``configs/accounts/<account>.yaml``): what upgrades you've made.

Recorded on every match (``meta.json`` → ``account_levels``) so experiments can refuse to span
an upgrade, and fed to the policy's level-aware math when a strategy opts in
(``params.level_aware``). ``doctor`` compares them with what perception measured (your
Princess Tower HP gives your King Level) and with each strategy's ``tier0.spell_tower_damage``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .policy.levels import Levels, level_mult

log = logging.getLogger(__name__)


@dataclass
class AccountLevels:
    account: str
    king_level: int | None = None
    cards: dict[str, int | None] = field(default_factory=dict)
    updated: str | None = None
    path: str | None = None
    # Tower damage measured in game (lethal margin 0), authoritative over the card DB estimate.
    spell_tower_damage: dict[str, int] = field(default_factory=dict)
    _king_seen: list[int] = field(default_factory=list, repr=False, compare=False)  # unconfirmed readings

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
        return Levels(king_level=self.king_level, cards=self.known_cards,
                      tower_damage=dict(self.spell_tower_damage))

    def record_upgrade(self, card_or_tower: str) -> bool:
        """Bump a card level (or King Level when Tower Princess upgrades) in memory."""
        if card_or_tower == "tower_princess":
            self.king_level = (self.king_level or 1) + 1
            self.updated = dt.date.today().isoformat()
            return True
        if card_or_tower in self.cards:
            level = self.cards[card_or_tower]
            if level is None:
                # Unknown before, so unknown after: a guess (it used to store 2) can sit below the
                # card's starting level and skew level-aware scaling (issue #13).
                log.warning("upgraded %s, but its level isn't recorded in %s: set it there so "
                            "level-aware strategies scale it", card_or_tower, self.path or "the account file")
                return False
            self.cards[card_or_tower] = int(level) + 1
            if card_or_tower in self.spell_tower_damage:
                # The measurement was at the old level; carry it up one level (~10 %, the same
                # factor level_mult uses) until it's re-measured in game.
                old = self.spell_tower_damage[card_or_tower]
                self.spell_tower_damage[card_or_tower] = int(round(old * level_mult(card_or_tower, 2)
                                                                   / level_mult(card_or_tower, 1)))
                log.info("%s tower damage %d -> %d after the upgrade (estimated: re-measure in game)",
                         card_or_tower, old, self.spell_tower_damage[card_or_tower])
            self.updated = dt.date.today().isoformat()
            return True
        return False

    def sync_king_level(self, observed_king_level: int | None, *, confirm: int = 2) -> bool:
        """Raise the stored King Level once ``confirm`` matches in a row observed the same higher level.

        One misread Princess Tower used to raise it for good (issue #12). A match that observes
        the stored level (or lower) resets the streak; one with no reading leaves it alone."""
        if observed_king_level is None:
            return False
        if observed_king_level <= (self.king_level or 0):
            self._king_seen.clear()
            return False
        self._king_seen.append(int(observed_king_level))
        recent = self._king_seen[-confirm:]
        if len(recent) < confirm or len(set(recent)) != 1:
            return False
        self.king_level = recent[0]
        self._king_seen.clear()
        self.updated = dt.date.today().isoformat()
        return True

    def save(self, path: str | Path | None = None) -> Path | None:
        """Persist updated levels back to ``configs/accounts/<account>.yaml``."""
        target = Path(path or self.path) if (path or self.path) else None
        if target is None:
            return None
        doc: dict = {
            "account": self.account,
            "king_level": self.king_level,
            "updated": self.updated or dt.date.today().isoformat(),
            "cards": dict(self.cards),
        }
        if self.spell_tower_damage:
            doc["spell_tower_damage"] = dict(self.spell_tower_damage)
        target.parent.mkdir(parents=True, exist_ok=True)
        old_text = target.read_text() if target.exists() else ""
        target.write_text(update_yaml_text(old_text, doc))
        self.path = str(target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> AccountLevels:
        d = yaml.safe_load(Path(path).read_text()) or {}
        return cls(account=str(d.get("account") or Path(path).stem), king_level=d.get("king_level"),
                   cards=dict(d.get("cards") or {}), updated=str(d.get("updated")) if d.get("updated") else None,
                   path=str(path),
                   spell_tower_damage={k: int(v) for k, v in (d.get("spell_tower_damage") or {}).items()
                                       if isinstance(v, (int, float))})


_TOP = re.compile(r"^(?P<key>[A-Za-z_][\w-]*):(?P<sp>[ \t]*)(?P<val>.*?)(?P<cmt>[ \t]+#.*)?$")
_CHILD = re.compile(r"^(?P<ind>[ \t]+)(?P<key>[A-Za-z_][\w-]*):(?P<sp>[ \t]*)(?P<val>.*?)(?P<cmt>[ \t]+#.*)?$")


def _cmt(m: re.Match) -> str:
    """The line's trailing comment, including ``key:   # note`` where the value is empty."""
    if m["cmt"]:
        return m["cmt"]
    return f"{m['sp'] or ' '}{m['val']}" if m["val"].startswith("#") else ""


def _scalar(v) -> str:
    return yaml.safe_dump(v, default_flow_style=True, width=10_000).splitlines()[0]


def update_yaml_text(old_text: str, doc: dict) -> str:
    """``doc`` as YAML, keeping ``old_text``'s comments, order and spacing (issue #14).

    Changes only the values: top-level scalars and one level of mappings (``cards``,
    ``spell_tower_damage``) are updated in place, new keys are added at the end of their
    section, and keys no longer in ``doc`` are removed. If the result doesn't load back to
    exactly ``doc`` (a hand-edited layout this updater doesn't understand), it falls back to a
    plain dump under the file's leading comment block, so the data is never wrong.
    """
    def plain() -> str:
        head = []
        for ln in old_text.splitlines():
            if ln.strip() and not ln.lstrip().startswith("#"):
                break
            head.append(ln)
        while head and not head[-1].strip():
            head.pop()
        return ("\n".join(head) + "\n" if head else "") + yaml.safe_dump(doc, sort_keys=False)

    if not old_text.strip():
        return yaml.safe_dump(doc, sort_keys=False)
    lines = old_text.splitlines()
    out: list[str] = []
    seen_top: set[str] = set()
    section: str | None = None  # the mapping we're inside
    seen_child: set[str] = set()
    indent = "  "

    def close_section() -> None:
        sub = doc.get(section) if section else None
        if isinstance(sub, dict):
            # insert missing children after the section's last child, before trailing comments
            at = len(out)
            while at > 0 and (not out[at - 1].strip() or out[at - 1].lstrip().startswith("#")) \
                    and not out[at - 1].startswith(indent + "#"):
                at -= 1
            new = [f"{indent}{k}: {_scalar(v)}" for k, v in sub.items() if k not in seen_child]
            out[at:at] = new

    for ln in lines:
        m_child = _CHILD.match(ln)
        if section is not None and m_child and not ln.lstrip().startswith("#"):
            k = m_child["key"]
            sub = doc.get(section)
            indent = m_child["ind"]
            if isinstance(sub, dict) and k in sub:
                seen_child.add(k)
                sp = " " if m_child["val"].startswith("#") else (m_child["sp"] or " ")
                out.append(f"{m_child['ind']}{k}:{sp}{_scalar(sub[k])}{_cmt(m_child)}")
            continue  # not in doc any more: drop the line
        m_top = _TOP.match(ln)
        if m_top and not ln.startswith((" ", "\t", "#")):
            if section is not None:
                close_section()
            k = m_top["key"]
            seen_top.add(k)
            section, seen_child = None, set()
            if k not in doc:
                continue
            v = doc[k]
            if isinstance(v, dict):
                section = k
                out.append(f"{k}:{_cmt(m_top)}")
            else:
                sp = " " if m_top["val"].startswith("#") else (m_top["sp"] or " ")
                out.append(f"{k}:{sp}{_scalar(v)}{_cmt(m_top)}")
            continue
        out.append(ln)  # comments, blank lines, anything else
    if section is not None:
        close_section()
    for k, v in doc.items():
        if k not in seen_top:
            out.append(yaml.safe_dump({k: v}, sort_keys=False).rstrip("\n"))
    text = "\n".join(out) + "\n"
    try:
        if yaml.safe_load(text) == yaml.safe_load(yaml.safe_dump(doc, sort_keys=False)):
            return text
    except yaml.YAMLError:
        pass
    log.warning("account file layout not understood; rewriting it (leading comments kept)")
    return plain()


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
        if getattr(params, "level_aware", False) and k in measured:
            continue  # level-aware strategies use the measurement itself (issue #15)
        have = configured.get(k)
        if have is None or abs(have - v) <= max(5, 0.15 * v):
            continue
        src = "your in-game measurement gives" if k in measured else "your levels give"
        problems.append(f"tier0.spell_tower_damage[{k}] = {have}, but {src} ~{v}")
    return {"account_file": acct.path, "king_level": acct.king_level, "levels_id": acct.levels_id,
            "card_levels_missing": missing, "derived_spell_tower_damage": derived,
            "measured_spell_tower_damage": measured, "problems": problems}
