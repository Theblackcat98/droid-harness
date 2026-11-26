"""Outcome-blind inclusion rule for win-rate tests (decided before any results exist).

Count every completed match. Exclude only infrastructure aborts (disconnect, server crash,
app restart mid-battle) — and compare abort rates between arms, because an arm that aborts
more is failing, not getting a free pass. Never exclude by triage tags or data quality:
those can correlate with the outcome. Matches from a different game version are never
mixed into one test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .stats import two_proportion_z


@dataclass
class ArmTally:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    aborts: int = 0
    excluded_other_version: int = 0
    match_ids: list[str] = field(default_factory=list)

    @property
    def completed(self) -> int:
        return self.wins + self.losses + self.draws


def tally(matches: list[dict], arm: str, game_version: str) -> ArmTally:
    t = ArmTally()
    for m in matches:
        if m.get("arm") != arm:
            continue
        if m.get("game_version") != game_version:
            t.excluded_other_version += 1
            continue
        if m.get("status") == "infra_abort":
            t.aborts += 1
            continue
        r = m.get("result")
        if r == "win":
            t.wins += 1
        elif r == "loss":
            t.losses += 1
        elif r == "draw":
            t.draws += 1
        else:
            continue
        t.match_ids.append(m.get("match_id", ""))
    return t


def abort_imbalance(a: ArmTally, b: ArmTally, alpha: float = 0.05) -> tuple[bool, float]:
    """True if abort rates differ significantly between arms (investigate before trusting the test)."""
    _, p = two_proportion_z(a.aborts, a.aborts + a.completed, b.aborts, b.aborts + b.completed)
    return p < alpha, p
