"""The Clash Royale policy: Tier 0 → posture → utility-scored rules (→ optional ranker)."""

from __future__ import annotations

from dataclasses import replace

from ....core.runtime.intent_bus import Intent
from ..arena import Arena
from ..cards import CardDB
from ..state import GameState
from . import posture as posture_mod
from .candidate import Candidate, Decision
from .constraints import Constraints
from .placement import Placement
from .rules import Rule, RuleContext, default_rules
from .strategy import Params
from .tier0 import Tier0


class CRPolicy:
    def __init__(self, cards: CardDB, arena: Arena, params: Params, *, exec_latency_s: float = 0.04,
                 rules: list[Rule] | None = None, ranker=None, pending_timeout_ms: int = 700,
                 retry_after_ms: int | None = 400) -> None:
        self.cards, self.arena, self.p = cards, arena, params
        self.placement = Placement(arena, cards, params.placement, exec_latency_s)
        self.ctx = RuleContext(cards, arena, self.placement)
        self.tier0 = Tier0(cards, arena, self.placement)
        self.constraints = Constraints(arena, cards, params.min_action_gap_ms)
        self.rules = rules if rules is not None else default_rules()
        self.ranker = ranker
        self.pending_timeout_ms = pending_timeout_ms
        self.retry_after_ms = retry_after_ms  # re-tap once if the card hasn't left the hand (None = off)
        self.retries = 0
        self.reset_match()

    def reset_match(self) -> None:
        self.tier0.reset()
        self.last_action_ms = -10_000
        self.pending: Candidate | None = None
        self.pending_since_ms = 0
        self.pending_retried = False
        for r in self.rules:
            if hasattr(r, "reset"):
                r.reset()

    def confirm_played(self, card: str, t_ms: int) -> None:
        """Perception saw our card leave the hand."""
        if self.pending is not None and self.pending.card == card:
            self.pending = None
        for r in self.rules:
            if hasattr(r, "on_played"):
                r.on_played(card, t_ms, self.p)

    def total(self, c: Candidate, posture: str) -> float:
        mult = self.p.posture_multipliers.get(posture, {}).get(c.family, 1.0)
        return c.score * self.p.weight(c.rule_id) * mult

    def decide(self, s: GameState, intent: Intent | None = None) -> Decision:
        s.last_action_ms = max(s.last_action_ms, self.last_action_ms)
        s.posture = posture_mod.classify(s, self.p, self.cards, self.tier0.spell_tower_damage, intent)
        intent_rec = intent.__dict__ if intent is not None else None
        decision = Decision(s.t_ms, s.host_ns, s.posture, None, state_digest=s.digest(), intent=intent_rec)

        # One action in flight at a time; it clears when perception confirms the play.
        if self.pending is not None:
            retry = self._retry(s)
            if retry is not None:
                decision.candidates, decision.totals = [retry], {0: retry.score}
                decision.chosen = retry
                self.last_action_ms = s.t_ms
                self.pending, self.pending_since_ms = retry, s.t_ms
                return decision
            if s.t_ms - self.pending_since_ms < self.pending_timeout_ms:
                return decision
        self.pending = None

        t0 = self.tier0.check(s, self.p)
        if t0 is not None and self.constraints.ok(s, t0):
            decision.candidates = [t0]
            decision.totals = {0: t0.score}
            return self._choose(decision, t0, s)

        cands = [c for r in self.rules for c in r.propose(s, self.p, self.ctx, intent)]
        cands = [c for c in cands if self.constraints.ok(s, c)]
        if self.ranker is not None:
            cands = self.ranker.rescore(s, cands)
        decision.candidates = cands
        if not cands:
            return decision
        totals = {i: self.total(c, s.posture) for i, c in enumerate(cands)}
        decision.totals = totals
        i_best = max(totals, key=totals.get)
        if totals[i_best] < self.p.act_threshold:
            return decision
        return self._choose(decision, cands[i_best], s)

    def _retry(self, s: GameState) -> Candidate | None:
        """Re-tap once when a placed card is still sitting in its slot (missed tap).

        Only when the current frame still shows that card in that slot and it's
        affordable, so a late hand read can't make us tap whatever slid into the slot.
        """
        c = self.pending
        if (self.retry_after_ms is None or self.pending_retried
                or s.t_ms - self.pending_since_ms < self.retry_after_ms):
            return None
        if not (0 <= c.slot < len(s.hand) and s.hand[c.slot] == c.card and c.card in self.cards):
            return None
        if self.cards.cost(c.card) > s.my_elixir + 1e-6:
            return None
        self.pending_retried = True
        self.retries += 1
        return replace(c, execute_at_ms=None, preselect=False, reason=f"retry: {c.reason}")

    def _choose(self, decision: Decision, c: Candidate, s: GameState) -> Decision:
        decision.chosen = c
        exec_ms = c.execute_at_ms if c.execute_at_ms is not None else s.t_ms
        self.last_action_ms = exec_ms
        self.pending, self.pending_since_ms = c, exec_ms
        self.pending_retried = False
        return decision
