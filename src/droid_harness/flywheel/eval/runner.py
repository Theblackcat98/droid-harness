"""Unattended champion/candidate experiment loop (overnight-plan Phase 2 / M5).

No LLM here: the runner alternates two strategy files per match, tags every match
with ``experiment_id``/``arm``, checks the game version, and stops on an SPRT
decision. Results go to the ledger and the warehouse ``experiments`` table.

Pairing note: Training Camp matches are unpaired (each arm plays the bot on its
own), so ``sprt_h2h`` runs a Bernoulli SPRT on the *candidate* arm's decisive
games. ``sprt_paired`` is for two-phone self-play discordant pairs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from .inclusion import ArmTally, tally
from .sprt import SprtResult, sprt


@dataclass
class ExperimentConfig:
    experiment_id: str
    title: str = ""
    hypothesis: str = ""
    proposer: str = "human"
    champion_id: str = ""
    candidate_id: str = ""
    test: str = "sprt_h2h"  # sprt_h2h | sprt_paired
    p0: float = 0.50
    p1: float = 0.58
    alpha: float = 0.05
    beta: float = 0.05
    game_version: str = ""
    balance_epoch: str = ""
    max_games: int = 400


class ExperimentRunner:
    """Pure bookkeeping: no phone, no I/O, so tests can drive it directly."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.matches: list[dict] = []
        self._next = "champion"  # strict alternation balances time-of-night effects
        self.decision: SprtResult | None = None
        self.void_reason: str | None = None

    # --- match loop ---------------------------------------------------------
    def next_arm(self) -> str:
        arm = self._next
        self._next = "candidate" if arm == "champion" else "champion"
        return arm

    def record(self, summary: dict) -> SprtResult:
        """Record one match summary; returns the current SPRT state."""
        if self.void_reason is None and summary.get("game_version") != self.cfg.game_version:
            self.void_reason = (
                f"game version changed: {summary.get('game_version')} != {self.cfg.game_version}")
        self.matches.append(summary)
        self.decision = self.evaluate()
        return self.decision

    def evaluate(self) -> SprtResult:
        cand = tally(self.matches, "candidate", self.cfg.game_version)
        if self.cfg.test == "sprt_paired":
            # Paired mode needs aligned pairs; without pair ids fall back to the
            # discordant-pair approximation on totals (documented limitation).
            champ = tally(self.matches, "champion", self.cfg.game_version)
            cw = max(0, cand.wins - champ.wins)
            lw = max(0, cand.losses - champ.losses)
            from .sprt import paired_sprt
            return paired_sprt(cw, lw, p1=self.cfg.p1, alpha=self.cfg.alpha, beta=self.cfg.beta)
        return sprt(cand.wins, cand.losses, self.cfg.p0, self.cfg.p1,
                    self.cfg.alpha, self.cfg.beta)

    @property
    def status(self) -> str:
        if self.void_reason is not None:
            return "void"
        if self.decision is None:
            return "running"
        return {"accept_H1": "accepted", "accept_H0": "rejected"}.get(self.decision.decision, "running")

    @property
    def done(self) -> bool:
        if self.void_reason is not None:
            return True
        if self.decision is not None and self.decision.decision != "continue":
            return True
        return len(self.matches) >= self.cfg.max_games

    def tallies(self) -> dict[str, ArmTally]:
        return {arm: tally(self.matches, arm, self.cfg.game_version)
                for arm in ("champion", "candidate")}

    # --- persistence ----------------------------------------------------------
    def to_ledger_kwargs(self) -> dict:
        t = self.tallies()["candidate"]
        return {"experiment_id": self.cfg.experiment_id, "title": self.cfg.title,
                "hypothesis": self.cfg.hypothesis, "proposer": self.cfg.proposer,
                "champion_id": self.cfg.champion_id, "candidate_id": self.cfg.candidate_id,
                "test": self.cfg.test, "p0": self.cfg.p0, "p1": self.cfg.p1,
                "game_version": self.cfg.game_version, "balance_epoch": self.cfg.balance_epoch,
                "status": self.status, "wins": t.wins, "losses": t.losses, "draws": t.draws,
                "llr": self.decision.llr if self.decision else 0.0,
                "decided_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) if self.done else "",
                "notes": self.void_reason or ""}

    def write_ledger(self, store: str = "ledger/experiments.jsonl"):
        from ..ledger import Experiment, Ledger

        Ledger(store).upsert(Experiment(**self.to_ledger_kwargs()))
        return store

    def write_warehouse(self, con) -> None:
        t = self.tallies()
        aborts = json.dumps({arm: t[arm].aborts for arm in t})
        con.execute(
            """INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT (experiment_id) DO UPDATE SET status=excluded.status, games=excluded.games,
               wins=excluded.wins, losses=excluded.losses, draws=excluded.draws,
               aborts_by_arm=excluded.aborts_by_arm, llr=excluded.llr, decided_at=excluded.decided_at,
               notes=excluded.notes""",
            [self.cfg.experiment_id, self.cfg.title, self.cfg.hypothesis, self.cfg.proposer,
             self.cfg.champion_id, self.cfg.candidate_id, "training_camp", self.cfg.test,
             self.cfg.game_version, self.cfg.balance_epoch, self.cfg.p0, self.cfg.p1,
             self.cfg.alpha, self.cfg.beta, self.status, len(self.matches),
             t["candidate"].wins + t["champion"].wins,
             t["candidate"].losses + t["champion"].losses,
             t["candidate"].draws + t["champion"].draws, aborts,
             self.decision.llr if self.decision else 0.0,
             time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) if self.done else None,
             self.void_reason or ""])
