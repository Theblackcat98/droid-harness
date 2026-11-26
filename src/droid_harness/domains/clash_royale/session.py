"""One phone's Clash Royale session: screens -> match lifecycle -> perception/policy/actions.

Implements the core ``PerceptionProvider`` / ``PolicyEngine`` / ``ActionMapper`` trio so the
generic ``BrainLoop`` can drive menus and battles alike.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from ...core.actuate.gestures import TouchEvent
from ...core.runtime.intent_bus import Intent
from ...core.telemetry.latency import LatencyStats
from ...core.transport.frame_slot import Frame
from .actions import CRActionMapper
from .menus import BATTLE_SCREENS, RESULT_SCREENS, MenuNavigator
from .perception.pipeline import CRPerception
from .perception.screens import ScreenClassifier
from .policy.candidate import Decision
from .policy.utility import CRPolicy
from .state import GameState


@dataclass
class Observation:
    screen: str
    frame: Frame
    state: GameState | None = None


@dataclass
class MenuDecision:
    screen: str
    events: list[TouchEvent] = field(default_factory=list)

    @property
    def chosen(self):
        return self.events or None


@dataclass
class MatchMeta:
    match_id: str
    device: str
    account: str
    strategy_id: str
    game_version: str | None
    balance_epoch: str | None
    opponent_kind: str
    experiment_id: str | None = None
    arm: str | None = None
    opponent_id: str | None = None
    start_pts_us: int | None = None  # device timestamp of the first battle frame (match t = 0)
    started_unix: float = field(default_factory=time.time)


class CRSession:
    def __init__(self, *, screens: ScreenClassifier, menus: MenuNavigator, perception: CRPerception,
                 policy: CRPolicy, mapper: CRActionMapper, recorder=None,
                 meta_factory: Callable[[], MatchMeta], on_match_end: Callable[[dict], None] | None = None,
                 screen_every: int = 3) -> None:
        self.screens, self.menus = screens, menus
        self.perception, self.policy, self.mapper = perception, policy, mapper
        self.recorder = recorder
        self.meta_factory = meta_factory
        self.on_match_end = on_match_end
        self.screen_every = screen_every
        self.screen = "UNKNOWN"
        self.screen_since = time.monotonic()
        self.match: MatchMeta | None = None
        self.last_state: GameState | None = None
        self.match_latency = LatencyStats()
        self._n = 0

    # --- PerceptionProvider ---------------------------------------------------
    def process(self, frame: Frame) -> Observation:
        if self._n % self.screen_every == 0 or self.screen not in BATTLE_SCREENS:
            name, _ = self.screens.classify(frame.img)
            # Inside a battle, transient overlays (emotes, effects) shouldn't end the match.
            if not (self.screen in BATTLE_SCREENS and name == "UNKNOWN"):
                self._set_screen(name, frame)
        self._n += 1
        if self.screen in BATTLE_SCREENS:
            state = self.perception.process(frame)
            self.last_state = state
            return Observation(self.screen, frame, state)
        return Observation(self.screen, frame)

    def _set_screen(self, name: str, frame: Frame) -> None:
        if name == self.screen:
            return
        prev, self.screen, self.screen_since = self.screen, name, time.monotonic()
        if name in BATTLE_SCREENS and prev not in BATTLE_SCREENS:
            self._start_match(frame)
        elif name in RESULT_SCREENS and self.match is not None:
            self._end_match(RESULT_SCREENS[name], status="completed")

    def screen_age_s(self) -> tuple[str, float]:
        return self.screen, time.monotonic() - self.screen_since

    def _start_match(self, frame: Frame) -> None:
        self.match = self.meta_factory()
        self.match.match_id = self.match.match_id or uuid.uuid4().hex[:12]
        self.perception.start_match(frame)
        self.match.start_pts_us = frame.pts_us
        self.policy.reset_match()
        self.match_latency = LatencyStats()
        if self.recorder is not None:
            self.recorder.start_match(self.match.match_id, self.match.__dict__)

    def _end_match(self, result: str, status: str) -> dict:
        s = self.last_state
        summary = {
            **self.match.__dict__, "status": status, "result": result,
            "my_crowns": s.crowns[0] if s else None, "opp_crowns": s.crowns[1] if s else None,
            "duration_s": (s.t_ms / 1000) if s else None,
            "overtime": bool(s and s.phase.startswith("ot")),
            "elixir_leaked": round(self.perception.elixir.leaked, 2),
            "opp_deck_seen": list(self.perception.opp.cycle.deck_seen),
            "perception_warnings": sum(self.perception.warnings.values()),
            "warning_counts": dict(self.perception.warnings),
            "loop_p50_ms": self.match_latency.summary("total").get("p50"),
            "loop_p99_ms": self.match_latency.summary("total").get("p99"),
            "events": self.perception.events,
        }
        if self.recorder is not None:
            self.recorder.end_match(summary)
        if self.on_match_end:
            self.on_match_end(summary)
        self.match = None
        return summary

    def abort_match(self, reason: str) -> dict | None:
        """Infrastructure failure mid-battle (disconnect, crash): recorded as infra_abort."""
        if self.match is None:
            return None
        return self._end_match(f"aborted:{reason}", status="infra_abort")

    # --- PolicyEngine -----------------------------------------------------------
    def decide(self, obs: Observation, intent: Intent | None):
        if obs.state is not None:
            return self.policy.decide(obs.state, intent)
        return MenuDecision(obs.screen, self.menus.events_for(obs.screen, obs.frame.img.shape,
                                                               time.perf_counter_ns()))

    # --- ActionMapper -----------------------------------------------------------
    def to_events(self, decision, obs: Observation, now_ns: int) -> list[TouchEvent]:
        if isinstance(decision, MenuDecision):
            return decision.events
        assert isinstance(decision, Decision) and obs.state is not None
        return self.mapper.to_events(decision, obs.state, now_ns)
