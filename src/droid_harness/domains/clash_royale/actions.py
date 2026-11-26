"""Decision -> timed touch events (tap-tap by default, with pre-selection)."""

from __future__ import annotations

from ...core.actuate import gestures
from ...core.actuate.gestures import MS, TouchEvent
from .perception.grid import Layout, TileMapper
from .policy.candidate import Decision
from .state import GameState


class CRActionMapper:
    def __init__(self, layout: Layout, tiles: TileMapper, *, hold_ms: float = 20, gap_ms: float = 20,
                 preselect_min_lead_ms: float = 150) -> None:
        self.layout, self.tiles = layout, tiles
        self.hold_ms, self.gap_ms = hold_ms, gap_ms
        self.preselect_min_lead_ms = preselect_min_lead_ms

    def card_px(self, slot: int, s: GameState) -> tuple[int, int]:
        w, h = s.frame_size
        return self.layout.hand[slot].center_px((h, w))

    def tile_px(self, tile, s: GameState) -> tuple[int, int]:
        w, h = s.frame_size
        x, y = self.tiles.tile_to_px(tile, (h, w))
        return (min(max(x, 0), w - 1), min(max(y, 0), h - 1))

    def to_events(self, decision: Decision, s: GameState, now_ns: int) -> list[TouchEvent]:
        c = decision.chosen
        if c is None:
            return []
        exec_ms = c.execute_at_ms if c.execute_at_ms is not None else s.t_ms
        place_ns = max(now_ns, s.host_ns + int((exec_ms - s.t_ms) * MS))
        action_id = f"{s.t_ms}:{c.rule_id}:{c.card}"
        card_xy, tile_xy = self.card_px(c.slot, s), self.tile_px(c.tile, s)
        lead_ms = (place_ns - now_ns) / MS
        if c.preselect and lead_ms >= self.preselect_min_lead_ms:
            return (gestures.preselect(card_xy, now_ns, hold_ms=self.hold_ms, action_id=action_id)
                    + gestures.tap(*tile_xy, place_ns, hold_ms=self.hold_ms, action_id=action_id, tag="place"))
        select_ns = max(now_ns, place_ns - int((self.hold_ms + self.gap_ms) * MS))
        return gestures.tap_tap(card_xy, tile_xy, select_ns, hold_ms=self.hold_ms, gap_ms=self.gap_ms,
                                action_id=action_id)
