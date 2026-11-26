"""T2 brain loop: latest frame -> perception -> policy -> executor, with stage timing.

Domain-agnostic: everything game-specific comes in through the interfaces.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from ..actuate.executor import Executor
from ..interfaces import ActionMapper, DecisionLogger, PerceptionProvider, PolicyEngine
from ..telemetry.latency import LatencyStats, StageClock
from ..transport.frame_slot import LatestFrame
from .intent_bus import IntentBus

log = logging.getLogger(__name__)


class BrainLoop:
    def __init__(self, slot: LatestFrame, perception: PerceptionProvider, policy: PolicyEngine,
                 mapper: ActionMapper, executor: Executor, *, intents: IntentBus | None = None,
                 logger: DecisionLogger | None = None, active: Callable[[Any], bool] | None = None,
                 frame_log_every: int = 1, decision_log_every: int = 5,
                 on_timing: Callable[[dict], None] | None = None) -> None:
        self.slot = slot
        self.perception = perception
        self.policy = policy
        self.mapper = mapper
        self.executor = executor
        self.intents = intents or IntentBus()
        self.logger = logger
        self.active = active or (lambda obs: True)  # e.g. only act while IN_BATTLE
        self.frame_log_every = frame_log_every
        self.decision_log_every = decision_log_every  # also always logged when an action is chosen
        self.on_timing = on_timing
        self.stats = LatencyStats()
        self._running = False
        self._thread: threading.Thread | None = None
        self.frames = 0
        self.decisions = 0
        self.errors = 0

    def step(self, frame) -> Any:
        clock = StageClock(t0_ns=frame.t_recv_ns)
        clock.marks["decode"] = frame.t_decoded_ns
        obs = self.perception.process(frame)
        clock.mark("perceive")
        decision = None
        if self.active(obs):
            decision = self.policy.decide(obs, self.intents.latest())
            clock.mark("decide")
            if decision is not None and getattr(decision, "chosen", None) is not None:
                events = self.mapper.to_events(decision, obs, time.perf_counter_ns())
                self.executor.schedule(events)
                clock.mark("schedule")
                self.decisions += 1
        durations = clock.durations_ms()
        self.stats.add_many(durations)
        if self.on_timing is not None:
            self.on_timing(durations)
        self.frames += 1
        if self.logger is not None:
            if self.frames % self.frame_log_every == 0:
                self.logger.log("frames", {"seq": frame.seq, "pts_us": frame.pts_us, **durations})
            if decision is not None and hasattr(decision, "to_record"):
                acted = getattr(decision, "chosen", None) is not None
                if acted or self.frames % self.decision_log_every == 0:
                    self.logger.log("decisions", {"seq": self.frames, **decision.to_record(), "lat": durations})
        return decision

    def _loop(self) -> None:
        last = 0
        while self._running:
            f = self.slot.wait_newer(last, timeout=0.25)
            if f is None:
                continue  # static screen or idle — not an error (see liveness)
            last = f.seq
            try:
                self.step(f)
            except Exception:  # keep the loop alive; count and log
                self.errors += 1
                log.exception("brain step failed")

    def start(self) -> BrainLoop:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="T2-brain", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
