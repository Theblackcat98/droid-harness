"""T3 executor: a time-ordered queue of touch events sent at their ``at_ns``.

Deciding fast and placing at the right moment are separate concerns: the policy hands
over events with absolute times; this thread sends each one as close to its time as it
can (sleep until ~1 ms before, then spin) and reports what was actually sent.
"""

from __future__ import annotations

import heapq
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from .gestures import TouchEvent

TouchSink = Callable[[int, int, int, int], int]  # (action, x, y, pointer_id) -> t_sent_ns


@dataclass
class SentEvent:
    event: TouchEvent
    t_sent_ns: int

    @property
    def lateness_ms(self) -> float:
        return (self.t_sent_ns - self.event.at_ns) / 1e6


class Executor:
    def __init__(self, sink: TouchSink, *, clock: Callable[[], int] = time.perf_counter_ns,
                 on_sent: Callable[[SentEvent], None] | None = None, spin_ns: int = 1_000_000) -> None:
        self.sink = sink
        self.clock = clock
        self.on_sent = on_sent
        self.spin_ns = spin_ns
        self._heap: list[TouchEvent] = []
        self._cv = threading.Condition()
        self._cancelled: set[str] = set()
        self._running = False
        self._thread: threading.Thread | None = None
        self.sent: list[SentEvent] = []
        self.keep_history = 1000

    # --- scheduling -------------------------------------------------------
    def schedule(self, events: list[TouchEvent]) -> None:
        with self._cv:
            for e in events:
                heapq.heappush(self._heap, e)
            self._cv.notify()

    def cancel(self, action_id: str) -> int:
        """Drop not-yet-sent events of an action. Never cancels half a gesture already started."""
        with self._cv:
            started = any(s.event.action_id == action_id for s in self.sent[-50:])
            if started:
                return 0
            before = len(self._heap)
            self._heap = [e for e in self._heap if e.action_id != action_id]
            heapq.heapify(self._heap)
            return before - len(self._heap)

    def pending(self) -> int:
        with self._cv:
            return len(self._heap)

    # --- execution --------------------------------------------------------
    def run_due(self, now_ns: int | None = None) -> list[SentEvent]:
        """Send every event due at ``now_ns`` (used by the thread and by tests)."""
        now_ns = self.clock() if now_ns is None else now_ns
        due: list[TouchEvent] = []
        with self._cv:
            while self._heap and self._heap[0].at_ns <= now_ns:
                due.append(heapq.heappop(self._heap))
        out = []
        for e in due:
            t = self.sink(e.action, e.x, e.y, e.pointer_id)
            s = SentEvent(e, t)
            out.append(s)
            self.sent.append(s)
            if self.on_sent:
                self.on_sent(s)
        if len(self.sent) > self.keep_history:
            del self.sent[: len(self.sent) - self.keep_history]
        return out

    def _loop(self) -> None:
        while self._running:
            with self._cv:
                if not self._heap:
                    self._cv.wait(0.05)
                    continue
                wait_ns = self._heap[0].at_ns - self.clock() - self.spin_ns
                if wait_ns > 0:
                    self._cv.wait(wait_ns / 1e9)
                    continue
            while self._heap and self._heap[0].at_ns > self.clock():
                pass  # final spin for sub-ms precision
            self.run_due()

    def start(self) -> Executor:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="T3-executor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        with self._cv:
            self._cv.notify()
        if self._thread:
            self._thread.join(timeout=1)
