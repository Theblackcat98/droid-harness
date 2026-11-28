"""One phone's Clash Royale session: screens -> match lifecycle -> perception/policy/actions.

Implements the core ``PerceptionProvider`` / ``PolicyEngine`` / ``ActionMapper`` trio so the
generic ``BrainLoop`` can drive menus and battles alike.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

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
from .recovery import HomeRecovery
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
    account_levels: dict | None = None  # configs/accounts/<account>.yaml at match start (levels_id inside)
    strategy_name: str | None = None
    start_pts_us: int | None = None  # device timestamp of the first battle frame (match t = 0)
    started_unix: float = field(default_factory=time.time)


class CRSession:
    def __init__(self, *, screens: ScreenClassifier, menus: MenuNavigator, perception: CRPerception,
                 policy: CRPolicy, mapper: CRActionMapper, recorder=None,
                 meta_factory: Callable[[], MatchMeta], on_match_end: Callable[[dict], None] | None = None,
                 screen_every: int = 3, result_stable_frames: int = 3,
                 result_settle_s: float = 1.2, unknown_battle_timeout_s: float = 25.0,
                 left_battle_timeout_s: float = 5.0, battle_no_progress_s: float = 30.0,
                 recovery: HomeRecovery | None = None,
                 send_key: Callable[[int], None] | None = None) -> None:
        self.screens, self.menus = screens, menus
        self.perception, self.policy, self.mapper = perception, policy, mapper
        self.recorder = recorder
        self.meta_factory = meta_factory
        self.on_match_end = on_match_end
        # Runs with (match_dir, summary) once the recorder has written the match (summary.json
        # on disk, files closed): video retention needs the finished files (issue #8).
        self.after_written: Callable[[Path, dict], None] | None = None
        # The brain thread ends matches on result screens while the runner thread aborts them
        # (app died, stuck screen, reconnect). Whoever takes the match under this lock ends it;
        # the other finds no match and does nothing (issue #10).
        self._match_lock = threading.Lock()
        self.screen_every = screen_every
        self.result_stable_frames = result_stable_frames
        self.result_settle_s = result_settle_s
        self.unknown_battle_timeout_s = unknown_battle_timeout_s
        self.left_battle_timeout_s = left_battle_timeout_s
        # A post-match chest/reward animation is purple, so the elixir-bar fallback
        # reads it as a full bar and the session would stay "in battle" forever. If
        # the clock, hand and elixir stop changing, the battle is over (or frozen).
        self.battle_no_progress_s = battle_no_progress_s
        self._progress_clock: float | None = None
        self._progress_elixir: float | None = None
        self._progress_since = time.monotonic()
        self._ever_had_battle_hud = False
        # Stuck-screen fallback: when a menu/result screen refuses to change, walk it
        # back to HOME with calibrated taps, visual button taps, BACK, then a restart.
        self.recovery = recovery or HomeRecovery(menus.buttons)
        self.send_key = send_key
        self.suspend_menu = False  # batch-end settle drives the UI directly
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
            # Block false IN_BATTLE entry on chest / Lucky Drop / menu screens whose top 18%
            # dhash resembles IN_BATTLE but have no magenta elixir bar at the bottom.
            if (name in BATTLE_SCREENS and self.screen not in BATTLE_SCREENS
                    and getattr(self.screens, "battle_roi", None) is not None
                    and self.screens.battle_evidence(frame.img) < self.screens.battle_min_frac):
                name = "UNKNOWN"
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
        self.recovery.observe(self.screen, time.monotonic())
        self._n += 1
        if self.suspend_menu:
            return Observation(self.screen, frame)
        if self.screen in BATTLE_SCREENS:
            state = self.perception.process(frame)
            self.last_state = state
            if any(state.hand_fresh):
                self._unknown_since = None
            now = time.monotonic()
            clock_t = getattr(state, "match_t_s", None)
            elixir = round(state.my_elixir, 1)
            if any(state.hand_fresh) or clock_t is not None or bool(state.towers):
                self._ever_had_battle_hud = True
            advanced = any(state.hand_fresh)
            if clock_t is not None and (self._progress_clock is None
                                        or abs(clock_t - self._progress_clock) >= 0.5):
                advanced = True
            if self._ever_had_battle_hud and (
                self._progress_elixir is None or abs(elixir - self._progress_elixir) >= 0.1
            ):
                advanced = True
            no_progress_limit = self.battle_no_progress_s if self._ever_had_battle_hud else min(
                self.battle_no_progress_s, 15.0)
            if advanced:
                self._progress_clock = clock_t if clock_t is not None else self._progress_clock
                self._progress_elixir = elixir
                self._progress_since = now
            elif now - self._progress_since >= no_progress_limit:
                # Clock, hand and elixir are frozen (or battle HUD never appeared on a chest/Lucky
                # Drop screen): record the abort and let menu recovery take over.
                self.abort_match("battle_no_progress")
                self._set_screen("UNKNOWN", frame)
                self._unknown_since = None
                return Observation(self.screen, frame)
            return Observation(self.screen, frame, state)
        return Observation(self.screen, frame)

    def _set_screen(self, name: str, frame: Frame) -> None:
        if name == self.screen:
            return
        if self.suspend_menu:
            # Batch-end settle: track the name but never start/end a match on a
            # transient frame (a HOME animation can hash as IN_BATTLE).
            self.screen, self.screen_since = name, time.monotonic()
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
        meta = self.meta_factory()
        meta.match_id = meta.match_id or uuid.uuid4().hex[:12]
        meta.start_pts_us = frame.pts_us
        with self._match_lock:
            self.match = meta
        self.perception.start_match(frame)
        self.policy.reset_match()
        self.match_latency = LatencyStats()
        self._pending_key, self._pending_n = None, 0
        self._result_crowns = None
        self._unknown_since = None
        self._progress_clock = None
        self._progress_elixir = None
        self._progress_since = time.monotonic()
        self._ever_had_battle_hud = False
        self.recovery.reset()
        if self.recorder is not None:
            self.recorder.start_match(self.match.match_id, self.match.__dict__)

    def _end_match(self, result: str, status: str) -> dict | None:
        with self._match_lock:
            m = self.match
            if m is None:
                return None  # another thread ended or aborted it first
            self.match = None
        s = self.last_state
        if self._result_crowns is not None and status == "completed":
            my_crowns, opp_crowns = self._result_crowns
        else:
            my_crowns = s.crowns[0] if s else None
            opp_crowns = s.crowns[1] if s else None
        # Only a like-for-like sum means anything: if any tower on either side was never read, the
        # difference is unknown (it used to sum what happened to be read, which flipped its sign
        # in losses where the opponent's labels were missed).
        tower_hp_diff = None
        if s and s.towers:
            names = [f"{side}_{pos}" for side in ("me", "opp") for pos in ("left", "right", "king")]
            if all(n in s.towers for n in names):
                tower_hp_diff = int(sum(s.towers[n] for n in names[:3]) - sum(s.towers[n] for n in names[3:]))
        summary = {
            **m.__dict__, "status": status, "result": result,
            "my_crowns": my_crowns, "opp_crowns": opp_crowns,
            "tower_hp_diff": tower_hp_diff,
            "my_king_level_seen": getattr(s, "my_king_level", None) if s else None,
            "opp_king_level_seen": getattr(s, "opp_king_level", None) if s else None,
            "tower_hp_reads": dict(getattr(self.perception, "tower_read_counts", {}) or {}),
            "tower_read_cycles": getattr(self.perception, "_tower_reads", 0),
            "hp_offsets": {k: [round(v[0], 4), round(v[1], 4)]
                           for k, v in (getattr(self.perception, "hp_offsets", {}) or {}).items()},
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
            if self.after_written is not None:
                hook = self.after_written
                self.recorder.end_match(summary, after=lambda d: hook(d, summary))
            else:
                self.recorder.end_match(summary)
        self._post_result = True
        if self.on_match_end:
            self.on_match_end(summary)
        return summary

    def abort_match(self, reason: str) -> dict | None:
        """Infrastructure failure mid-battle (disconnect, crash): recorded as infra_abort.

        Safe to call from any thread: it does nothing if the match already ended."""
        return self._end_match(f"aborted:{reason}", status="infra_abort")

    # --- PolicyEngine -----------------------------------------------------------
    def decide(self, obs: Observation, intent: Intent | None):
        if obs.state is not None:
            decision = self.policy.decide(obs.state, intent)
            if decision.chosen is not None and hasattr(self.perception, "record_my_action"):
                self.perception.record_my_action(decision.chosen.card, decision.chosen.tile, obs.state.t_ms)
            return decision
        if self.suspend_menu:
            return MenuDecision(obs.screen)
        now = time.monotonic()
        action = self.recovery.next_action(self.screen, obs.frame.img, now)
        if action is not None:
            if action.kind == "tap":
                return MenuDecision(obs.screen, action.events(time.perf_counter_ns()))
            if action.kind == "key" and self.send_key is not None:
                self.send_key(action.keycode)
                return MenuDecision(obs.screen)
        if self.recovery.is_stuck(self.screen, now):
            # Recovery owns this screen; the normal flow is what failed to leave it.
            return MenuDecision(obs.screen)
        return MenuDecision(obs.screen, self.menus.events_for(obs.screen, obs.frame.img.shape,
                                                               time.perf_counter_ns()))

    def take_restart_request(self) -> bool:
        """True once recovery has exhausted its taps and BACK and wants a fresh app."""
        return self.recovery.take_restart_request()

    # --- ActionMapper -----------------------------------------------------------
    def to_events(self, decision, obs: Observation, now_ns: int) -> list[TouchEvent]:
        if isinstance(decision, MenuDecision):
            return decision.events
        assert isinstance(decision, Decision) and obs.state is not None
        return self.mapper.to_events(decision, obs.state, now_ns)
