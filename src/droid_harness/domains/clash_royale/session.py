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
from .perception.screens import ScreenClassifier, read_result_detail
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
    pair_id: str | None = None  # experiment block this match belongs to (runner pairs arms)
    start_pts_us: int | None = None  # device timestamp of the first battle frame (match t = 0)
    started_unix: float = field(default_factory=time.time)


class CRSession:
    def __init__(self, *, screens: ScreenClassifier, menus: MenuNavigator, perception: CRPerception,
                 policy: CRPolicy, mapper: CRActionMapper, recorder=None,
                 meta_factory: Callable[[], MatchMeta], on_match_end: Callable[[dict], None] | None = None,
                 screen_every: int = 3, result_stable_frames: int = 3,
                 result_settle_s: float = 1.2, unknown_battle_timeout_s: float = 25.0,
                 left_battle_timeout_s: float = 5.0) -> None:
        self.screens, self.menus = screens, menus
        self.perception, self.policy, self.mapper = perception, policy, mapper
        self.recorder = recorder
        self.meta_factory = meta_factory
        self.on_match_end = on_match_end
        self.screen_every = screen_every
        self.result_stable_frames = result_stable_frames
        self.result_settle_s = result_settle_s
        self.unknown_battle_timeout_s = unknown_battle_timeout_s
        self.left_battle_timeout_s = left_battle_timeout_s
        self.screen = "UNKNOWN"
        self.screen_since = time.monotonic()
        self.match: MatchMeta | None = None
        self.last_state: GameState | None = None
        self.match_latency = LatencyStats()
        self._n = 0
        self._candidate = "UNKNOWN"
        self._candidate_n = 0
        self._post_result = False
        self._unknown_since: float | None = None
        # Result settling: the crown-reveal animation passes through intermediate
        # counts, so a result only ends the match after the full
        # (outcome, opp_crowns, my_crowns) triple repeats.
        self._pending_key: tuple[str, int, int] | None = None
        self._pending_n = 0
        self._pending_since = 0.0
        self._result_crowns: tuple[int, int] | None = None  # (my, opp)

    # --- PerceptionProvider ---------------------------------------------------
    def process(self, frame: Frame) -> Observation:
        if self._n % self.screen_every == 0 or self.screen not in BATTLE_SCREENS:
            name, _ = self.screens.classify(frame.img)
            # The top-only generic signature cannot tell a battle frame from a result
            # banner (the opponent name/HP are in the same place), and on ladder the
            # result screen hashes UNKNOWN (different player name, no elixir bar).
            # In or after a match, always verify the result banners/crowns directly.
            if self.screen in RESULT_SCREENS or self.match is not None:
                actual, opp_c, my_c = read_result_detail(frame.img)
                if actual is not None:
                    self._unknown_since = None
                    key = (actual, opp_c, my_c)
                    now = time.monotonic()
                    if key != self._pending_key:
                        self._pending_key, self._pending_n, self._pending_since = key, 1, now
                    else:
                        self._pending_n += 1
                    time_ok = (now - self._pending_since) >= self.result_settle_s
                    frames_ok = self._pending_n >= self.result_stable_frames
                    # Time-based settle, with a frame-count fallback (some harnesses
                    # feed frames faster than the wall clock advances in tests).
                    if (frames_ok and time_ok) or self._pending_n >= 20:
                        self._result_crowns = (my_c, opp_c)
                        name = f"RESULT_{actual.upper()}"
                    else:
                        name = self.screen  # still settling: don't leave the battle yet
                else:
                    self._pending_key, self._pending_n = None, 0
                    if self.match is not None and name in RESULT_SCREENS:
                        # Mid-match the classifier's RESULT label is untrusted on its
                        # own (the top HUD dhash confuses win/loss banners and fires
                        # during the victory animation). Stay in battle until the
                        # banners/crowns verify and settle.
                        name = self.screen
                    elif self.screen in RESULT_SCREENS and (name in BATTLE_SCREENS or name in RESULT_SCREENS):
                        name = self.screen  # victory animation / still-post-match: never re-enter battle
            elif name == "UNKNOWN":
                actual, _, _ = read_result_detail(frame.img)
                if actual is not None:
                    name = f"RESULT_{actual.upper()}"
            if self._post_result and name in BATTLE_SCREENS:
                name = self.screen  # wait for HOME/MENU/TRAINING_CONFIRM before a new match
            if name in {"HOME", "MENU", "TRAINING_CONFIRM"}:
                self._post_result = False
                if self.screen not in BATTLE_SCREENS:
                    self._unknown_since = None
            # Inside a battle, transient overlays or intro flashes shouldn't leave the battle
            # (or orphan the match directory): stay in battle whenever elixir is visible or
            # until the non-battle screen persists past timeout.
            if self.screen in BATTLE_SCREENS and name not in BATTLE_SCREENS and name not in RESULT_SCREENS:
                if self.screens.battle_evidence(frame.img) >= self.screens.battle_min_frac:
                    self._unknown_since = None
                else:
                    now = time.monotonic()
                    self._unknown_since = self._unknown_since or now
                    timeout = self.unknown_battle_timeout_s if name == "UNKNOWN" else self.left_battle_timeout_s
                    if now - self._unknown_since >= timeout:
                        reason = "battle_lost_screen" if name == "UNKNOWN" else f"left_battle:{name}"
                        self.abort_match(reason)
                        self._set_screen(name, frame)
                        self._unknown_since = None
            else:
                if name in BATTLE_SCREENS:
                    self._unknown_since = None
                if name == self._candidate:
                    self._candidate_n += 1
                else:
                    self._candidate, self._candidate_n = name, 1
                if self._candidate_n >= 1:
                    self._set_screen(name, frame)
        self._n += 1
        if self.screen in BATTLE_SCREENS:
            state = self.perception.process(frame)
            self.last_state = state
            if any(state.hand_fresh):
                self._unknown_since = None
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
        elif prev in BATTLE_SCREENS and name not in BATTLE_SCREENS and name not in RESULT_SCREENS and self.match is not None:
            self.abort_match(f"left_battle:{name}")

    def screen_age_s(self) -> tuple[str, float]:
        return self.screen, time.monotonic() - self.screen_since

    def _start_match(self, frame: Frame) -> None:
        self.match = self.meta_factory()
        self.match.match_id = self.match.match_id or uuid.uuid4().hex[:12]
        self.perception.start_match(frame)
        self.match.start_pts_us = frame.pts_us
        self.policy.reset_match()
        self.match_latency = LatencyStats()
        self._pending_key, self._pending_n = None, 0
        self._result_crowns = None
        self._unknown_since = None
        if self.recorder is not None:
            self.recorder.start_match(self.match.match_id, self.match.__dict__)

    def _end_match(self, result: str, status: str) -> dict:
        s = self.last_state
        if self._result_crowns is not None and status == "completed":
            my_crowns, opp_crowns = self._result_crowns
        else:
            my_crowns = s.crowns[0] if s else None
            opp_crowns = s.crowns[1] if s else None
        summary = {
            **self.match.__dict__, "status": status, "result": result,
            "my_crowns": my_crowns, "opp_crowns": opp_crowns,
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
        self._post_result = True
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
