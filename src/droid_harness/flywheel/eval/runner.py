"""Unattended champion/candidate experiment loop (overnight-plan Phase 2 / M5).

No LLM here: the runner assigns strategy files to matches, tags every match with
``experiment_id``/``arm``/``pair_id``, checks the game version, and stops on an SPRT
decision. Results go to the ledger and the warehouse ``experiments`` table.

Tests (``ExperimentConfig.test``):

* ``pairs`` (default; Training Camp, where each arm plays the bot on its own): matches
  are assigned in randomized blocks of two, one per arm, in random order. Each block is
  a matched pair (same night, same phone, same bot). Only *discordant* pairs carry
  information: candidate scored more than champion (win > draw > loss) or less. The SPRT
  asks whether the candidate wins discordant pairs more often than 50%, so a candidate
  is accepted only for beating the champion, never for beating a fixed rate.
* ``h2h`` (two-phone self-play, candidate plays the champion): every game is a direct
  comparison, so it's a Bernoulli SPRT on the candidate's score vs 50%. Refuses games
  that aren't head-to-head.
* ``fixed``: candidate vs a fixed rate ``p0``. Only for absolute questions ("does the
  blind strategy beat the trainer more than half the time?"). It ignores the champion,
  so it must never be used for promotion.

Draws count as half a win (``h2h``/``fixed``) or as the middle score (``pairs``); they
are never dropped. Infra aborts never count, but an aborted slot is replayed so its
pair can complete, and abort rates are compared between arms
(``inclusion.abort_imbalance``): an arm that aborts more is failing, not excused.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass

from .inclusion import ArmTally, abort_imbalance, tally
from .sprt import SprtResult, sprt

LEGACY_TESTS = {"sprt_h2h": "fixed", "sprt_paired": "h2h"}  # old names in older ledgers/CLIs
HEAD_TO_HEAD_KINDS = ("selfplay", "friendly")
SCORE = {"win": 1.0, "draw": 0.5, "loss": 0.0}


@dataclass
class ExperimentConfig:
    experiment_id: str
    title: str = ""
    hypothesis: str = ""
    proposer: str = "human"
    champion_id: str = ""
    candidate_id: str = ""
    test: str = "pairs"  # pairs | h2h | fixed
    p0: float = 0.50
    p1: float = 0.58
    alpha: float = 0.05
    beta: float = 0.05
    game_version: str = ""
    balance_epoch: str = ""
    max_games: int = 400
    seed: int = 0
    max_slot_retries: int = 3  # replays of one aborted slot before its pair is dropped

    def __post_init__(self) -> None:
        self.test = LEGACY_TESTS.get(self.test, self.test)
        if self.test not in ("pairs", "h2h", "fixed"):
            raise ValueError(f"unknown test {self.test!r}")


class ExperimentRunner:
    """Pure bookkeeping: no phone, no I/O, so tests can drive it directly."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.matches: list[dict] = []
        self.decision: SprtResult | None = None
        self.void_reason: str | None = None
        self._rng = random.Random(cfg.seed)
        self._queue: list[tuple[str, str]] = []  # (arm, pair_id) still to play
        self._pairs: dict[str, dict[str, str]] = {}  # pair_id -> arm -> result
        self._retries: dict[tuple[str, str], int] = {}
        self.dropped_pairs: list[str] = []
        self._n_blocks = 0
        self.last_pair_id: str | None = None

    # --- assignment ----------------------------------------------------------
    def next_assignment(self) -> tuple[str, str]:
        """(arm, pair_id) for the next match. Replays aborted slots before new blocks."""
        if not self._queue:
            pid = f"{self.cfg.experiment_id}-p{self._n_blocks:04d}"
            self._n_blocks += 1
            arms = ["champion", "candidate"]
            self._rng.shuffle(arms)  # random order inside the block: no first-mover bias
            self._queue = [(a, pid) for a in arms]
            self._pairs[pid] = {}
        arm, pid = self._queue[0]
        self.last_pair_id = pid
        return arm, pid

    def next_arm(self) -> str:
        return self.next_assignment()[0]

    def record(self, summary: dict) -> SprtResult:
        """Record one match summary; returns the current SPRT state."""
        if self.void_reason is None and summary.get("game_version") != self.cfg.game_version:
            self.void_reason = (
                f"game version changed: {summary.get('game_version')} != {self.cfg.game_version}")
        if (self.void_reason is None and self.cfg.test == "h2h"
                and summary.get("opponent_kind") not in HEAD_TO_HEAD_KINDS):
            self.void_reason = (f"h2h needs head-to-head games, got opponent_kind="
                                f"{summary.get('opponent_kind')!r}; use test=pairs for Training Camp")
        self.matches.append(summary)
        self._settle_slot(summary)
        self.decision = self.evaluate()
        return self.decision

    def _settle_slot(self, summary: dict) -> None:
        arm = summary.get("arm")
        pid = summary.get("pair_id") or self.last_pair_id
        slot = (arm, pid)
        if slot not in self._queue:
            return
        if summary.get("status") == "infra_abort" or summary.get("result") not in SCORE:
            n = self._retries.get(slot, 0) + 1
            self._retries[slot] = n
            if n > self.cfg.max_slot_retries:  # this pair can't complete: drop it whole
                self._queue = [q for q in self._queue if q[1] != pid]
                self.dropped_pairs.append(pid)
            return  # otherwise the slot stays at the head of the queue and is replayed
        self._queue.remove(slot)
        self._pairs.setdefault(pid, {})[arm] = summary["result"]

    # --- evaluation ------------------------------------------------------------
    def pair_counts(self) -> tuple[int, int, int]:
        """(candidate better, champion better, tied) over complete pairs."""
        better = worse = tied = 0
        for pid, res in self._pairs.items():
            if pid in self.dropped_pairs or len(res) < 2:
                continue
            c, h = SCORE[res["candidate"]], SCORE[res["champion"]]
            if c > h:
                better += 1
            elif h > c:
                worse += 1
            else:
                tied += 1
        return better, worse, tied

    def evaluate(self) -> SprtResult:
        cfg = self.cfg
        if cfg.test == "pairs":
            better, worse, _ = self.pair_counts()
            return sprt(better, worse, 0.5, cfg.p1, cfg.alpha, cfg.beta)
        cand = tally(self.matches, "candidate", cfg.game_version)
        w, lo = cand.wins + 0.5 * cand.draws, cand.losses + 0.5 * cand.draws
        p0 = 0.5 if cfg.test == "h2h" else cfg.p0
        return sprt(w, lo, p0, cfg.p1, cfg.alpha, cfg.beta)

    def abort_warning(self) -> str | None:
        t = self.tallies()
        a, b = t["champion"], t["candidate"]
        if a.aborts + b.aborts == 0:
            return None
        bad, pval = abort_imbalance(a, b)
        if bad:
            return (f"abort rates differ between arms (champion {a.aborts}/{a.aborts + a.completed}, "
                    f"candidate {b.aborts}/{b.aborts + b.completed}, p={pval:.3f}): investigate before trusting")
        return None

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
                "notes": self.notes()}

    def notes(self) -> str:
        parts = []
        if self.void_reason:
            parts.append(self.void_reason)
        if self.cfg.test == "pairs":
            b, w, t = self.pair_counts()
            parts.append(f"pairs: candidate better {b}, champion better {w}, tied {t}, "
                         f"dropped {len(self.dropped_pairs)}")
        warn = self.abort_warning()
        if warn:
            parts.append(warn)
        return "; ".join(parts)

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
             self.notes()])
