"""Unattended champion/candidate experiment loop (README §12; todo.md M5).

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
* ``dense`` (todo R8): same randomized pairs, but the outcome is the continuous paired
  difference of ``dense_match_score`` (result, crowns, tower-HP difference when known, speed),
  so W-W and L-L pairs count too. An always-valid mixture SPRT (mSPRT) on the mean
  difference stops as soon as it is clearly positive (accept) or negative (reject); the
  mixture scale ``dense_tau`` is the difference you care about. Needs ``min_dense_pairs``.
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
import math
import random
import time
from dataclasses import dataclass

from .inclusion import ArmTally, abort_imbalance, tally
from .sprt import SprtResult, sprt

LEGACY_TESTS = {"sprt_h2h": "fixed", "sprt_paired": "h2h"}  # old names in older ledgers/CLIs
HEAD_TO_HEAD_KINDS = ("selfplay", "friendly")
SCORE = {"win": 1.0, "draw": 0.5, "loss": 0.0}


def dense_match_score(summary: dict) -> float:
    """Continuous match dominance score so W-W and L-L pairs still measure margin."""
    res = str(summary.get("result") or "")
    base = 3.0 if res == "win" else (1.5 if res == "draw" else 0.0)
    my_c = float(summary.get("my_crowns") or 0.0)
    opp_c = float(summary.get("opp_crowns") or 0.0)
    # Unknown HP difference (a tower never read) contributes nothing rather than a biased guess.
    hp_diff = summary.get("tower_hp_diff")
    hp_diff = float(hp_diff) if isinstance(hp_diff, (int, float)) else 0.0
    dur = float(summary.get("duration_s") or 180.0)
    speed_bonus = max(0.0, (180.0 - dur) / 600.0) if res == "win" else 0.0
    return base + 0.5 * (my_c - opp_c) + max(-0.8, min(0.8, hp_diff / 3000.0)) + speed_bonus


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
    futility_min_pairs: int = 0  # >0 enables Stage-A dense-margin futility stop when UCB95 < 0
    account_levels_id: str = ""  # set -> a match played at other card levels voids the test
    dense_tau: float = 0.5  # dense test: mixture scale (a paired margin difference worth detecting)
    min_dense_pairs: int = 6  # dense test: pairs before the variance estimate is trusted

    def __post_init__(self) -> None:
        self.test = LEGACY_TESTS.get(self.test, self.test)
        if self.test not in ("pairs", "dense", "h2h", "fixed"):
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
        self._pair_summaries: dict[str, dict[str, dict]] = {}  # pair_id -> arm -> summary
        self._retries: dict[tuple[str, str], int] = {}
        self.dropped_pairs: list[str] = []
        self.futility_reason: str | None = None
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
        seen_levels = (summary.get("account_levels") or {}).get("levels_id")
        if (self.void_reason is None and self.cfg.account_levels_id and seen_levels
                and seen_levels != self.cfg.account_levels_id):
            self.void_reason = (f"account levels changed mid-experiment ({seen_levels} != "
                                f"{self.cfg.account_levels_id}): an upgrade changes both arms' strength")
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
        self._pair_summaries.setdefault(pid, {})[arm] = summary

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

    def dense_pair_stats(self) -> dict[str, float]:
        """Paired dense-dominance stats across ALL completed pairs (including W-W and L-L)."""
        diffs = []
        for pid, sums in self._pair_summaries.items():
            if pid in self.dropped_pairs or "candidate" not in sums or "champion" not in sums:
                continue
            diffs.append(dense_match_score(sums["candidate"]) - dense_match_score(sums["champion"]))
        n = len(diffs)
        if n == 0:
            return {"pairs": 0, "mean": 0.0, "se": 0.0, "lcb95": 0.0, "ucb95": 0.0}
        mean = sum(diffs) / n
        if n < 2:
            return {"pairs": n, "mean": round(mean, 4), "se": 0.0, "lcb95": round(mean, 4), "ucb95": round(mean, 4)}
        var = sum((x - mean) ** 2 for x in diffs) / (n - 1)
        se = math.sqrt(var / n)
        return {"pairs": n, "mean": round(mean, 4), "se": round(se, 4),
                "lcb95": round(mean - 1.96 * se, 4), "ucb95": round(mean + 1.96 * se, 4)}

    def msprt(self) -> tuple[float, float, int]:
        """(log Λ, mean paired difference, pairs) for the dense mixture SPRT.

        Λ_n = sqrt(σ²/(σ²+nτ²)) · exp(n²τ²x̄² / (2σ²(σ²+nτ²))) with H0: mean difference = 0, a
        N(0, τ²) mixture over the alternative and the plug-in variance; reject H0 when
        Λ ≥ 1/α (Johari et al., always-valid inference).
        """
        ds = self.dense_pair_stats()
        n, mean = int(ds["pairs"]), float(ds["mean"])
        if n < max(2, self.cfg.min_dense_pairs):
            return 0.0, mean, n
        var = max(1e-6, (ds["se"] ** 2) * n)
        tau2 = self.cfg.dense_tau ** 2
        log_l = 0.5 * math.log(var / (var + n * tau2)) + (n * n * tau2 * mean * mean) / (2 * var * (var + n * tau2))
        return log_l, mean, n

    def evaluate(self) -> SprtResult:
        cfg = self.cfg
        if cfg.test == "dense":
            better, worse, _ = self.pair_counts()
            log_l, mean, _ = self.msprt()
            bound = math.log(1 / cfg.alpha)
            signed = log_l if mean >= 0 else -log_l
            if log_l >= bound:
                return SprtResult("accept_H1" if mean > 0 else "accept_H0", signed, -bound, bound, better, worse)
            return SprtResult("continue", signed, -bound, bound, better, worse)
        if cfg.test == "pairs":
            better, worse, _ = self.pair_counts()
            res = sprt(better, worse, 0.5, cfg.p1, cfg.alpha, cfg.beta)
            if res.decision == "continue" and cfg.futility_min_pairs > 0:
                ds = self.dense_pair_stats()
                if ds["pairs"] >= cfg.futility_min_pairs and ds["ucb95"] < 0.0:
                    self.futility_reason = (f"stage-a dense-margin futility at {int(ds['pairs'])} pairs "
                                            f"(mean={ds['mean']:+.3f}, ucb95={ds['ucb95']:+.3f} < 0)")
                    return SprtResult("accept_H0", res.llr, res.lower, res.upper, res.wins, res.losses)
            return res
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
        if self.decision.decision == "continue":
            # Ran out of scheduled games without crossing an SPRT boundary.
            return "inconclusive" if self.done else "running"
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
                "status": self.status, "matches": len(self.matches),
                "wins": t.wins, "losses": t.losses, "draws": t.draws,
                "llr": self.decision.llr if self.decision else 0.0,
                "decided_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) if self.done else "",
                "notes": self.notes()}

    def notes(self) -> str:
        parts = []
        if self.void_reason:
            parts.append(self.void_reason)
        if self.futility_reason:
            parts.append(self.futility_reason)
        if self.cfg.test in ("pairs", "dense"):
            b, w, t = self.pair_counts()
            ds = self.dense_pair_stats()
            parts.append(f"pairs: candidate better {b}, champion better {w}, tied {t}, "
                         f"dropped {len(self.dropped_pairs)}, dense_margin={ds['mean']:+.3f}")
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
