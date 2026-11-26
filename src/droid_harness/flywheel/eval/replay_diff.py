"""Offline replay diff: run a candidate policy over logged states and compare decisions.

Gate 2 of the promotion funnel: catches crashes on real states and changes far larger
than the hypothesis claimed, before a single match is played.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ...domains.clash_royale.state import GameState


@dataclass
class DiffSummary:
    states: int = 0
    changed: int = 0
    errors: int = 0
    by_rule: Counter = field(default_factory=Counter)  # champion rule -> changed count
    examples: list[dict] = field(default_factory=list)

    @property
    def changed_pct(self) -> float:
        return 100.0 * self.changed / self.states if self.states else 0.0


def load_states(match_dirs: Iterable[str | Path]) -> list[GameState]:
    out = []
    for d in match_dirs:
        p = Path(d) / "decisions.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("state"):
                out.append(GameState.from_digest(rec["state"]))
    return out


def _key(decision) -> tuple | None:
    c = decision.chosen
    if c is None:
        return None
    return (c.card, round(c.tile[0] * 2) / 2, round(c.tile[1] * 2) / 2)


def replay_diff(champion, candidate, states: list[GameState], max_examples: int = 20) -> DiffSummary:
    """Each state is evaluated independently (fresh policy state), in time order."""
    out = DiffSummary()
    for s in states:
        out.states += 1
        for pol in (champion, candidate):
            pol.reset_match()  # independent evaluation: no in-flight action, no action-gap carry-over
        try:
            a = champion.decide(GameState.from_digest(s.digest()))
            b = candidate.decide(GameState.from_digest(s.digest()))
        except Exception as e:  # a crash on a real state fails the gate
            out.errors += 1
            if len(out.examples) < max_examples:
                out.examples.append({"t_ms": s.t_ms, "error": repr(e)})
            continue
        if _key(a) != _key(b):
            out.changed += 1
            out.by_rule[a.chosen.rule_id if a.chosen else "none"] += 1
            if len(out.examples) < max_examples:
                out.examples.append({"t_ms": s.t_ms, "champion": _key(a), "candidate": _key(b),
                                     "champion_rule": a.chosen.rule_id if a.chosen else None,
                                     "candidate_rule": b.chosen.rule_id if b.chosen else None})
    return out


def gate(summary: DiffSummary, expected_change_pct: float, tolerance: float = 3.0) -> tuple[bool, str]:
    if summary.errors:
        return False, f"{summary.errors} states crashed the candidate"
    if summary.changed_pct > expected_change_pct * tolerance + 1.0:
        return False, (f"changed {summary.changed_pct:.1f}% of decisions; proposal claimed "
                       f"{expected_change_pct:.1f}%")
    return True, f"changed {summary.changed_pct:.1f}% of decisions"
