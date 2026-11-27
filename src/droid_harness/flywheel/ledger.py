"""Strategy Ledger: every experiment, whatever its result, so ideas aren't retried blindly."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Experiment:
    experiment_id: str
    title: str
    hypothesis: str
    proposer: str  # human | coach | optuna
    champion_id: str
    candidate_id: str
    test: str  # sprt_h2h | sprt_paired
    p0: float
    p1: float
    game_version: str
    balance_epoch: str
    status: str = "running"  # running | accepted | rejected | inconclusive | void
    matches: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    llr: float = 0.0
    decided_at: str = ""
    notes: str = ""
    tags: list[str] = field(default_factory=list)


class Ledger:
    """JSONL store + generated markdown (``ledger/STRATEGY_LEDGER.md``)."""

    def __init__(self, path: str | Path = "ledger/experiments.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def all(self) -> list[Experiment]:
        if not self.path.exists():
            return []
        latest: dict[str, Experiment] = {}
        for line in self.path.read_text().splitlines():
            if line.strip():
                e = Experiment(**json.loads(line))
                latest[e.experiment_id] = e  # last write wins
        return list(latest.values())

    def upsert(self, e: Experiment) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(e)) + "\n")

    def next_id(self) -> str:
        nums = [int(m.group(1)) for e in self.all() if (m := re.match(r"EXP-(\d+)", e.experiment_id))]
        return f"EXP-{(max(nums) + 1) if nums else 1:04d}"

    def similar_failures(self, hypothesis: str, game_version: str, threshold: float = 0.5) -> list[Experiment]:
        """Rejected experiments whose hypothesis overlaps this one (token Jaccard), same game version."""
        toks = _tokens(hypothesis)
        out = []
        for e in self.all():
            if e.status != "rejected" or e.game_version != game_version:
                continue
            other = _tokens(e.hypothesis)
            j = len(toks & other) / max(1, len(toks | other))
            if j >= threshold:
                out.append(e)
        return out

    def render_markdown(self, out: str | Path = "ledger/STRATEGY_LEDGER.md", last: int | None = None) -> str:
        exps = sorted(self.all(), key=lambda e: e.experiment_id, reverse=True)
        if last:
            exps = exps[:last]
        parts = ["# Strategy Ledger", "", "Generated from `ledger/experiments.jsonl`. Newest first.", ""]
        for e in exps:
            parts += [f"## {e.experiment_id} · {e.status.upper()} · {e.decided_at or 'running'} · game "
                      f"{e.game_version} · epoch {e.balance_epoch}",
                      f"**{e.title}** — proposed by {e.proposer}",
                      f"Hypothesis: {e.hypothesis}",
                      f"Candidate `{e.candidate_id}` vs champion `{e.champion_id}` · {e.test} [{e.p0:.2f}, {e.p1:.2f}]"
                      f" · {e.matches or (e.wins + e.losses + e.draws)} matches · candidate "
                      f"{e.wins}–{e.losses}–{e.draws} · LLR {e.llr:+.2f}"]
            if e.notes:
                parts.append(f"Note: {e.notes}")
            parts.append("")
        text = "\n".join(parts)
        Path(out).write_text(text)
        return text


def _tokens(s: str) -> set[str]:
    stop = {"the", "a", "an", "to", "of", "and", "in", "on", "for", "vs", "with", "is", "are", "by", "at"}
    return {t for t in re.findall(r"[a-z0-9_.]+", s.lower()) if t not in stop and len(t) > 1}
