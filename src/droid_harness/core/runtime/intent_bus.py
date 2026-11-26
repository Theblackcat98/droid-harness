"""Latest-intent mailbox between the optional L2 advisor and the L1 brain.

The brain never waits on it: ``latest()`` returns None when nothing fresh is there.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class Intent:
    aggression: float = 0.5
    lane: str | None = None  # "left" | "right" | None
    hold: list[str] = field(default_factory=list)
    posture: str | None = None
    note: str = ""
    t_created: float = field(default_factory=time.monotonic)

    def age_s(self) -> float:
        return time.monotonic() - self.t_created


class IntentBus:
    def __init__(self, max_age_s: float = 6.0) -> None:
        self.max_age_s = max_age_s
        self._intent: Intent | None = None
        self._lock = threading.Lock()

    def publish(self, intent: Intent) -> None:
        with self._lock:
            self._intent = intent

    def latest(self) -> Intent | None:
        with self._lock:
            i = self._intent
        if i is None or i.age_s() > self.max_age_s:
            return None
        return i
