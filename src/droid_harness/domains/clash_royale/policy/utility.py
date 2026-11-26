"""The Clash Royale policy: Tier 0 → posture → utility-scored rules (→ optional ranker)."""

from __future__ import annotations

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
                 rules: list[Rule] | None = None, ranker=None, pending_timeout_ms: int = 700) -> None:
        self.cards, self.arena, self.p = cards, arena, params
        self.placement = Placement(arena, cards, params.placement, exec_latency_s)
        self.ctx = RuleContext(cards, arena, self.placement)
        self.tier0 = Tier0(cards, arena, self.placement)
        self.constraints = Constraints(arena, cards, params.min_action_gap_ms)
        self.rules = rules if rules is not None else default_rules()
        self.ranker = ranker
        self.pending_timeout_ms = pending_timeout_ms
        self.reset_match()

    def reset_match(self) -> None:
        self.tier0.reset()
        self.last_action_ms = -10_000
        self.pending: Candidate | None = None
        self.pending_since_ms = 0

    def confirm_played(self, card: str, t_ms: int) -> None:
        """Perception saw our card leave the hand."""
        if self.pending is not None and self.pending.card == card:
            self.pending = None

    def total(self, c: Candidate, posture: str) -> float:
        mult = self.p.posture_multipliers.get(posture, {}).get(c.family, 1.0)
        return c.score * self.p.weight(c.rule_id) * mult

    def decide(self, s: GameState, intent: Intent | None = None) -> Decision:
        s.last_action_ms = max(s.last_action_ms, self.last_action_ms)
        s.posture = posture_mod.classify(s, self.p, self.cards, self.tier0.spell_tower_damage, intent)
        intent_rec = intent.__dict__ if intent is not None else None
        decision = Decision(s.t_ms, s.host_ns, s.posture, None, state_digest=s.digest(), intent=intent_rec)

        # One action in flight at a time; it clears when perception confirms the play.
        if self.pending is not None and s.t_ms - self.pending_since_ms < self.pending_timeout_ms:
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

    def _choose(self, decision: Decision, c: Candidate, s: GameState) -> Decision:
        decision.chosen = c
        exec_ms = c.execute_at_ms if c.execute_at_ms is not None else s.t_ms
        self.last_action_ms = exec_ms
        self.pending, self.pending_since_ms = c, exec_ms
        return decision
