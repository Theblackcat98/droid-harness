"""Domain plugin interfaces. The core harness only talks to these."""

from __future__ import annotations

from typing import Any, Protocol

from .actuate.gestures import TouchEvent
from .runtime.intent_bus import Intent
from .transport.frame_slot import Frame


class PerceptionProvider(Protocol):
    def process(self, frame: Frame) -> Any:
        """Pixels -> a domain observation (e.g. GameState). Must be fast (< ~10 ms)."""


class PolicyEngine(Protocol):
    def decide(self, observation: Any, intent: Intent | None) -> Any:
        """Observation -> a domain decision record (with ``.chosen`` or None)."""


class ActionMapper(Protocol):
    def to_events(self, decision: Any, observation: Any, now_ns: int) -> list[TouchEvent]:
        """Decision -> timed touch events for the executor."""


class DecisionLogger(Protocol):
    def log(self, stream: str, record: dict) -> None: ...
