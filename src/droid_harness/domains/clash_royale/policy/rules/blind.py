"""No-vision offense: the bootstrap controller that won Training Camp 3–1, as a rule.

Without a detector the opponent-elixir interval never narrows (no plays are seen), so
``WinConditionWhenOppLow`` scores below ``act_threshold`` and the policy never attacks.
This rule plays on elixir, hand and match phase alone:

1. **Tank** (``blind_tanks``, default the win conditions) at the push lane's bridge once
   elixir reaches ``blind_push_elixir``.
2. **Support** troops behind the tank for ``blind_support_ms`` after the tank is confirmed
   (melee at the bridge, ranged ``blind_support_back_tiles`` behind it).
3. **Spells** on the lane's princess tower (the king once that falls): once per push when
   the tank should be at the tower, or any time elixir is about to leak.
4. **Pressure**: otherwise, the heaviest affordable troop at the bridge before elixir
   leaks (held at a higher threshold while a tank is in hand, so the push isn't starved).

Lanes alternate every ``blind_switch_every`` tank pushes (0 = never). Inactive unless the
strategy sets ``blind: true``; with a detector, keep ``blind: false`` and let the
vision rules decide.
"""

from __future__ import annotations

from ..candidate import Candidate
from . import RuleContext


class BlindPush:
    id = "offense.blind_push"

    def __init__(self) -> None:
        self.reset()

    # --- hooks called by CRPolicy -------------------------------------------------
    def reset(self) -> None:
        self.tank_ms: int | None = None
        self.pushes = 0
        self.spelled_this_push = False

    def on_played(self, card: str, t_ms: int, p) -> None:
        if not getattr(p, "blind", False):
            return
        if card in self._tanks(p):
            self.tank_ms = t_ms
            self.pushes += 1
            self.spelled_this_push = False
        elif self.tank_ms is not None and self._in_spell_window(p, t_ms - self.tank_ms):
            self.spelled_this_push = True  # one push-timed spell per push (any card counts as spent)

    # --- helpers ----------------------------------------------------------------------
    @staticmethod
    def _tanks(p) -> set[str]:
        return set(p.blind_tanks or p.win_conditions)

    @staticmethod
    def _in_spell_window(p, since_tank_ms: int) -> bool:
        return p.blind_spell_after_ms <= since_tank_ms <= p.blind_spell_after_ms + p.blind_spell_window_ms

    def lane(self, p) -> str:
        first = p.blind_lane
        other = "left" if first == "right" else "right"
        if p.blind_switch_every <= 0:
            return first
        return first if (self.pushes // p.blind_switch_every) % 2 == 0 else other

    @staticmethod
    def _tower(s, lane: str) -> str:
        princess = f"opp_{lane}"
        return "opp_king" if s.towers.get(princess) == 0 else princess

    # --- proposals ---------------------------------------------------------------------
    def propose(self, s, p, ctx: RuleContext, intent):
        if not p.blind:
            return
        x2 = s.phase != "x1"
        push_at = p.blind_push_elixir_x2 if x2 else p.blind_push_elixir
        pressure_at = p.blind_min_elixir_x2 if x2 else p.blind_min_elixir
        tanks = self._tanks(p)
        lane = self.lane(p)
        bx, by = p.placement.bridge_tiles[lane]
        since_tank = s.t_ms - self.tank_ms if self.tank_ms is not None else None
        supporting = since_tank is not None and 0 <= since_tank <= p.blind_support_ms
        spell_due = (since_tank is not None and not self.spelled_this_push
                     and self._in_spell_window(p, since_tank))
        tank_in_hand = any(k in tanks for k in s.hand if k)
        if tank_in_hand:
            pressure_at = max(pressure_at, p.blind_hold_for_tank_elixir)

        for slot, key in s.affordable_hand(ctx.cards):
            card = ctx.cards[key]
            if card.is_spell:
                if spell_due or s.my_elixir >= p.blind_spell_elixir:
                    tower = self._tower(s, lane)
                    yield Candidate(self.id, key, slot, ctx.arena.towers[tower],
                                    score=0.5 if spell_due else 0.3,
                                    reason=f"blind spell on {tower} ({'push' if spell_due else 'leak'})")
                continue
            if card.kind == "building":
                continue
            if key in tanks:
                if s.my_elixir >= push_at:
                    tile = ctx.placement.snap((bx, by), card, s)
                    yield Candidate(self.id, key, slot, tile, score=0.6,
                                    reason=f"blind tank push {lane} at {s.my_elixir:.1f} elixir")
                continue
            back = p.blind_support_back_tiles if card.range_tiles >= 2.5 else 0.0
            tile = ctx.placement.snap((bx, by - back), card, s)
            if supporting:
                yield Candidate(self.id, key, slot, tile, score=0.55 + 0.01 * card.cost,
                                reason=f"blind support {lane} (+{since_tank / 1000:.1f}s)")
            elif s.my_elixir >= pressure_at:
                yield Candidate(self.id, key, slot, tile, score=0.35 + 0.01 * card.cost,
                                reason=f"blind pressure {lane} at {s.my_elixir:.1f} elixir")
