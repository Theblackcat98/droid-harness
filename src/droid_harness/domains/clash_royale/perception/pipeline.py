"""Pixels -> GameState for one battle (HUD every frame, detector at a cadence)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import numpy as np

from ....core.transport.frame_slot import Frame
from ..arena import Arena
from ..cards import CardDB
from ..elixir import ElixirModel, phase_at
from ..opponent_model import OpponentModel
from ..state import GameState, Unit
from .cards import CardMatcher
from .detector import Detector
from .digits import DigitReader
from .grid import Layout, TileMapper
from .hud import is_greyed, read_elixir
from .ownership import ownership_and_hp
from .tracker import PlayEvent, SpawnGrouper, TileDetection, Tracker

TOWER_SIDE = {"me_left": "me", "me_right": "me", "me_king": "me",
              "opp_left": "opp", "opp_right": "opp", "opp_king": "opp"}


class CRPerception:
    def __init__(self, *, layout: Layout, tiles: TileMapper, cards: CardDB, arena: Arena, matcher: CardMatcher,
                 digits: DigitReader | None = None, detector: Detector | None = None, detect_every: int = 1,
                 tower_every_ms: int = 500, on_my_play: Callable[[str, int], None] | None = None,
                 on_event: Callable[[dict], None] | None = None) -> None:
        self.layout, self.tiles, self.cards, self.arena = layout, tiles, cards, arena
        self.matcher, self.digits, self.detector = matcher, digits, detector
        self.detect_every, self.tower_every_ms = detect_every, tower_every_ms
        self.on_my_play, self.on_event = on_my_play, on_event
        self.start_pts_us: int | None = None
        self.reset()

    # --- match lifecycle ----------------------------------------------------
    def reset(self) -> None:
        self.elixir = ElixirModel()
        self.opp = OpponentModel()
        self.tracker = Tracker(river_mid_y=sum(self.arena.river_y) / 2)
        self.grouper = SpawnGrouper(self.cards.card_for_class)
        self.hand: list[str | None] = [None] * 4
        self.next_card: str | None = None
        self.towers: dict[str, int] = {}
        self._tower_read_ms = -10**9
        self.frame_i = 0
        self.warnings: Counter = Counter()
        self.units: list[Unit] = []
        self.events: list[dict] = []
        self._started = False

    def start_match(self, frame: Frame) -> None:
        self.reset()
        self.start_pts_us = frame.pts_us
        self._started = True

    # --- per frame ------------------------------------------------------------
    def process(self, frame: Frame) -> GameState:
        if not self._started:
            self.start_match(frame)
        img = frame.img
        t_ms = int((frame.pts_us - self.start_pts_us) / 1000)
        t_s = t_ms / 1000
        warns: list[str] = []

        hud = read_elixir(img, self.layout.elixir_bar, self.layout.elixir_hsv_lo, self.layout.elixir_hsv_hi)
        if self.frame_i == 0 and hud is not None:
            self.elixir.reset(hud, t_s)
            self.opp.elixir.low = self.opp.elixir.high = hud
        crops = self._read_hand(img, t_ms, warns)  # detects our plays (spend) before fusing the HUD
        self.elixir.observe(hud, t_s)
        self._check_affordability(crops, warns)
        if self.elixir.discrepancy_s >= 1.0:
            warns.append("elixir_discrepancy")
        self.opp.advance(t_s)

        if self.detector is not None and self.frame_i % self.detect_every == 0:
            self._detect(img, t_ms)
        if t_ms - self._tower_read_ms >= self.tower_every_ms and self.digits is not None:
            self._read_towers(img, warns)
            self._tower_read_ms = t_ms

        affordable = [c is not None and c in self.cards and self.cards.cost(c) <= self.elixir.value + 1e-6
                      for c in self.hand]
        crowns = (sum(1 for k, hp in self.towers.items() if k.startswith("opp_") and hp == 0),
                  sum(1 for k, hp in self.towers.items() if k.startswith("me_") and hp == 0))
        if self.towers.get("opp_king") == 0:
            crowns = (3, crowns[1])
        if self.towers.get("me_king") == 0:
            crowns = (crowns[0], 3)
        self.frame_i += 1
        for w in warns:
            self.warnings[w] += 1
        return GameState(
            t_ms=t_ms, host_ns=frame.t_decoded_ns, frame_size=(img.shape[1], img.shape[0]),
            phase=phase_at(t_s), my_elixir=self.elixir.value, opp_elixir=self.opp.elixir.interval(),
            hand=list(self.hand), hand_affordable=affordable, next_card=self.next_card, units=list(self.units),
            towers=dict(self.towers), crowns=crowns, opp_deck_seen=list(self.opp.cycle.deck_seen),
            opp_hand_possible=self.opp.cycle.possible_hand(), opp_last_four=self.opp.cycle.last_four,
            pocket_open={"left": self.towers.get("opp_left") == 0, "right": self.towers.get("opp_right") == 0},
            warnings=warns)

    # --- pieces ---------------------------------------------------------------
    def _read_hand(self, img: np.ndarray, t_ms: int, warns: list[str]) -> list:
        new = list(self.hand)
        crops = []
        for i, roi in enumerate(self.layout.hand):
            crop = roi.crop(img)
            r = self.matcher.match(crop)
            if not r.accepted:
                warns.append("hand_unreadable")
                continue
            new[i] = r.card
            crops.append((r.card, crop))
        nr = self.matcher.match(self.layout.next_card.crop(img))
        if nr.accepted:
            self.next_card = nr.card
        for i, (old, cur) in enumerate(zip(self.hand, new, strict=False)):
            if old is not None and cur is not None and old != cur:
                self._my_play(old, i, t_ms)
        self.hand = new
        return crops

    def _check_affordability(self, crops: list, warns: list[str]) -> None:
        """Greyed-out cards are a free check on the elixir reading."""
        for card, crop in crops:
            if card not in self.cards:
                continue
            cost, greyed = self.cards.cost(card), is_greyed(crop)
            if (greyed and self.elixir.value >= cost + 0.6) or (not greyed and self.elixir.value < cost - 0.6):
                warns.append("affordability_mismatch")

    def _my_play(self, card: str, slot: int, t_ms: int) -> None:
        if card in self.cards:
            self.elixir.spend(self.cards.cost(card))
        self.events.append({"kind": "my_play", "card": card, "slot": slot, "t_ms": t_ms})
        if self.on_event:
            self.on_event(self.events[-1])
        if self.on_my_play:
            self.on_my_play(card, t_ms)

    def _detect(self, img: np.ndarray, t_ms: int) -> None:
        dets = []
        for d in self.detector(img):
            side, hp = ownership_and_hp(img, d.box)
            tile = self.tiles.px_to_tile(d.foot, img.shape)
            dets.append(TileDetection(d.cls, d.conf, tile, side, hp))
        tracks, spawns = self.tracker.update(dets, t_ms)
        self.grouper.expire(t_ms)
        for ev in spawns:
            for play in self.grouper.add(ev):
                self._on_play(play)
        units = []
        for tr in tracks:
            card = self.cards.card_for_class(tr.cls)
            units.append(Unit(tr.id, tr.cls, tr.side, tr.tile, tr.vel, tr.hp_pct, tr.first_ms, tr.conf,
                              flying=bool(card and card.flying)))
        self.units = units

    def _on_play(self, play: PlayEvent) -> None:
        rec = {"kind": f"{play.side}_spawn_play", "card": play.card, "tile": play.tile, "t_ms": play.t_ms}
        self.events.append(rec)
        if self.on_event:
            self.on_event(rec)
        if play.side == "opp":
            self.opp.on_play(play.card, self.cards.cost(play.card), play.t_ms / 1000)

    def _read_towers(self, img: np.ndarray, warns: list[str]) -> None:
        for name, roi in self.layout.towers.items():
            hp = self.digits.read(roi.crop(img))
            if hp is None:
                continue
            prev = self.towers.get(name)
            if prev is not None and hp > prev + 5:
                warns.append("tower_hp_increase")
                continue
            self.towers[name] = hp
