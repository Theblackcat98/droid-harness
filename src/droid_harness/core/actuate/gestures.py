"""Gesture builders: turn intents (tap, tap-tap card play, drag) into timed touch events."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from ..transport.scrcpy_protocol import MotionAction

_seq = itertools.count()
MS = 1_000_000  # ns per ms


@dataclass(order=True)
class TouchEvent:
    at_ns: int
    seq: int = field(default_factory=lambda: next(_seq))
    action: int = field(default=MotionAction.DOWN, compare=False)
    x: int = field(default=0, compare=False)
    y: int = field(default=0, compare=False)
    pointer_id: int = field(default=0, compare=False)
    action_id: str | None = field(default=None, compare=False)
    tag: str = field(default="", compare=False)


def tap(x: int, y: int, at_ns: int, *, hold_ms: float = 20, pointer_id: int = 0,
        action_id: str | None = None, tag: str = "tap") -> list[TouchEvent]:
    return [
        TouchEvent(at_ns, action=MotionAction.DOWN, x=x, y=y, pointer_id=pointer_id, action_id=action_id, tag=f"{tag}:down"),
        TouchEvent(at_ns + int(hold_ms * MS), action=MotionAction.UP, x=x, y=y, pointer_id=pointer_id,
                   action_id=action_id, tag=f"{tag}:up"),
    ]


def tap_tap(card_xy: tuple[int, int], tile_xy: tuple[int, int], at_ns: int, *, hold_ms: float = 20,
            gap_ms: float = 20, action_id: str | None = None, preselected: bool = False) -> list[TouchEvent]:
    """Select a card, then tap the arena tile. With ``preselected`` only the tile tap is sent."""
    if preselected:
        return tap(*tile_xy, at_ns, hold_ms=hold_ms, action_id=action_id, tag="place")
    select = tap(*card_xy, at_ns, hold_ms=hold_ms, action_id=action_id, tag="select")
    place_at = select[-1].at_ns + int(gap_ms * MS)
    return select + tap(*tile_xy, place_at, hold_ms=hold_ms, action_id=action_id, tag="place")


def preselect(card_xy: tuple[int, int], at_ns: int, *, hold_ms: float = 20,
              action_id: str | None = None) -> list[TouchEvent]:
    return tap(*card_xy, at_ns, hold_ms=hold_ms, action_id=action_id, tag="select")


def drag(p0: tuple[int, int], p1: tuple[int, int], at_ns: int, *, duration_ms: float = 120, steps: int = 8,
         pointer_id: int = 0, action_id: str | None = None) -> list[TouchEvent]:
    events = [TouchEvent(at_ns, action=MotionAction.DOWN, x=p0[0], y=p0[1], pointer_id=pointer_id,
                         action_id=action_id, tag="drag:down")]
    for k in range(1, steps + 1):
        f = k / steps
        x = round(p0[0] + (p1[0] - p0[0]) * f)
        y = round(p0[1] + (p1[1] - p0[1]) * f)
        events.append(TouchEvent(at_ns + int(duration_ms * f * MS), action=MotionAction.MOVE, x=x, y=y,
                                 pointer_id=pointer_id, action_id=action_id, tag="drag:move"))
    events.append(TouchEvent(events[-1].at_ns + 1, action=MotionAction.UP, x=p1[0], y=p1[1],
                             pointer_id=pointer_id, action_id=action_id, tag="drag:up"))
    return events
