"""Pixels -> GameState for one battle (HUD every frame, detector at a cadence).

Detector output is routed by class (see ``classes.py``):

* side-aware units (``knight_opp``) -> tracker, side from the class, HP from the bar;
* plain units (``knight``) -> tracker, side from the HP bar or the spawn-side prior;
* spell effects (``fireball_effect``) -> tracker for play events only (never ``units``),
  side from your own recent plays of that spell (they land on the *other* half);
* tower states (``tower_rubble``) -> tower state, never the tracker.

Between detections, units are extrapolated along their tracked velocity, so a detector
running every Nth frame (or on its own thread) still gives the policy current positions.
"""

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
from .classes import flying_of, is_effect, is_tower, split_side
from .detector import AsyncDetector, Detector
from .digits import DigitReader, binarize, segment
from .grid import Layout, TileMapper
from .hud import is_greyed, read_elixir
from .ownership import ownership_and_hp
from .towers import TowerStateReader
from .tracker import PlayEvent, SpawnGrouper, TileDetection, Track, Tracker

TOWER_SIDE = {"me_left": "me", "me_right": "me", "me_king": "me",
              "opp_left": "opp", "opp_right": "opp", "opp_king": "opp"}


class CRPerception:
    def __init__(self, *, layout: Layout, tiles: TileMapper, cards: CardDB, arena: Arena, matcher: CardMatcher,
                 digits: DigitReader | None = None, detector: Detector | None = None, detect_every: int = 1,
                 tower_every_ms: int = 500, on_my_play: Callable[[str, int], None] | None = None,
                 on_event: Callable[[dict], None] | None = None,
                 flying_foot_frac: float = 0.15, async_detect: bool = False,
                 tower_states: TowerStateReader | None = None, my_spell_window_ms: int = 3500,
                 extrapolate_max_ms: int = 500) -> None:
        self.layout, self.tiles, self.cards, self.arena = layout, tiles, cards, arena
        self.matcher, self.digits, self.detector = matcher, digits, detector
        self.detect_every, self.tower_every_ms = max(1, detect_every), tower_every_ms
        self.on_my_play, self.on_event = on_my_play, on_event
        self.flying_foot_frac = flying_foot_frac  # air units: box bottom is not the ground point
        self.tower_states = tower_states
        self.my_spell_window_ms = my_spell_window_ms  # my spell play -> its effect on screen
        self.extrapolate_max_ms = extrapolate_max_ms
        self.async_detector: AsyncDetector | None = None
        if detector is not None and async_detect:
            self.async_detector = detector if isinstance(detector, AsyncDetector) else AsyncDetector(detector)
        self.start_pts_us: int | None = None
        self._generation = 0
        self.reset()

    # --- match lifecycle ----------------------------------------------------
    def reset(self) -> None:
        self.elixir = ElixirModel()
        self.opp = OpponentModel()
        self.tracker = Tracker(river_mid_y=sum(self.arena.river_y) / 2)
        self._tracks: list[Track] = []
        self._my_spells: list[tuple[str, int]] = []  # (card key, t_ms) of my recent spell plays
        self._destroyed: set[str] = set()
        self._generation = getattr(self, "_generation", 0) + 1  # drops async results from a past match
        if self.tower_states is not None:
            self.tower_states.reset()
        self.grouper = SpawnGrouper(self.cards.card_for_class)
        self.hand: list[str | None] = [None] * 4
        self.hand_fresh: list[bool] = []
        self.next_card: str | None = None
        self.towers: dict[str, int] = {}
        self._tower_read_ms = -10**9
        self._clock_read_ms = -10**9
        self._clock_match_t_s: float | None = None
        self.clock_activity: float | None = None
        self.frame_i = 0
        self.warnings: Counter = Counter()
        self.units: list[Unit] = []
        self.events: list[dict] = []
        self._started = False
        self._last_t_ms = 0

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
        self._last_t_ms = t_ms
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

        if self.detector is not None:
            self._run_detector(img, t_ms)
            self.units = self._extrapolate(t_ms)
        if t_ms - self._tower_read_ms >= self.tower_every_ms and (self.digits is not None
                                                                   or self.tower_states is not None):
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
        match_t_s, clock_drift_s = self._read_clock(img, t_ms, warns)
        self.frame_i += 1
        for w in warns:
            self.warnings[w] += 1
        state = GameState(
            t_ms=t_ms, host_ns=frame.t_decoded_ns, frame_size=(img.shape[1], img.shape[0]),
            phase=phase_at(t_s), my_elixir=self.elixir.value, opp_elixir=self.opp.elixir.interval(),
            hand=list(self.hand), hand_affordable=affordable, hand_fresh=list(self.hand_fresh),
            next_card=self.next_card, units=list(self.units),
            towers=dict(self.towers), crowns=crowns, opp_deck_seen=list(self.opp.cycle.deck_seen),
            opp_hand_possible=self.opp.cycle.possible_hand(), opp_last_four=self.opp.cycle.last_four,
            pocket_open={"left": self.towers.get("opp_left") == 0, "right": self.towers.get("opp_right") == 0},
            warnings=warns)
        # GameState is the canonical state class; clock fields are attached here (rather
        # than changing its schema) because only perception produces them.
        state.match_t_s = match_t_s
        state.clock_drift_s = clock_drift_s
        state.clock_activity = self.clock_activity
        return state

    # --- pieces ---------------------------------------------------------------
    def _read_hand(self, img: np.ndarray, t_ms: int, warns: list[str]) -> list:
        # Unreadable slots keep their last known card so a play is still detected when the
        # next read succeeds, but they're marked stale and the policy won't act on them.
        new = list(self.hand)
        self.hand_fresh = [False] * len(self.layout.hand)
        crops = []
        for i, roi in enumerate(self.layout.hand):
            crop = roi.crop(img)
            r = self.matcher.match(crop)
            if not r.accepted:
                warns.append("hand_unreadable")
                continue
            new[i] = r.card
            self.hand_fresh[i] = True
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
            if self.cards[card].is_spell:
                self._my_spells = [(k, t) for k, t in self._my_spells if t_ms - t <= self.my_spell_window_ms]
                self._my_spells.append((card, t_ms))
        self.events.append({"kind": "my_play", "card": card, "slot": slot, "t_ms": t_ms})
        if self.on_event:
            self.on_event(self.events[-1])
        if self.on_my_play:
            self.on_my_play(card, t_ms)

    # --- detection ----------------------------------------------------------------
    def _arena_crop(self, img: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        # Arena crop: cuts latency and removes HUD false hits. Boxes are offset back to
        # full-image pixels before ownership sampling and tile mapping.
        x0, y0, x1, y1 = self.layout.arena.to_px(img.shape)
        return img[y0:y1, x0:x1], (x0, y0)

    def _run_detector(self, img: np.ndarray, t_ms: int) -> None:
        if self.async_detector is None:
            if self.frame_i % self.detect_every == 0:
                self._detect(img, t_ms)
            return
        r = self.async_detector.poll()
        if r is not None:
            (gen, t_det, frame_img, origin), dets, _ms = r
            if gen == self._generation:
                self._ingest(dets, frame_img, origin, t_det)
        if self.frame_i % self.detect_every == 0 and self.async_detector.idle:
            crop, origin = self._arena_crop(img)
            if crop.size:
                self.async_detector.submit(crop.copy(), (self._generation, t_ms, img, origin))

    def _detect(self, img: np.ndarray, t_ms: int) -> None:
        """Synchronous detection on this frame."""
        crop, origin = self._arena_crop(img)
        if crop.size == 0:
            return
        self._ingest(self.detector(crop), img, origin, t_ms)
        self.units = self._extrapolate(t_ms)

    def _ingest(self, raw_dets, img: np.ndarray, origin: tuple[int, int], t_ms: int) -> None:
        x0, y0 = origin
        dets = []
        for d in raw_dets:
            base, cls_side = split_side(d.cls)
            bx1, by1, bx2, by2 = (d.box[0] + x0, d.box[1] + y0, d.box[2] + x0, d.box[3] + y0)
            fx, fy = (bx1 + bx2) / 2, by2
            if is_tower(base):
                self._tower_detection(base, self.tiles.px_to_tile((fx, (by1 + by2) / 2), img.shape))
                continue
            card0 = self.cards.card_for_class(base)
            if flying_of(base, card0):
                fy -= self.flying_foot_frac * (by2 - by1)
            tile = self.tiles.px_to_tile((fx, fy), img.shape)
            if is_effect(base):
                side, hp = self._effect_side(base, t_ms), None
            else:
                bar_side, hp = ownership_and_hp(img, (bx1, by1, bx2, by2))
                side = cls_side or bar_side  # a side-aware class beats the HP-bar color
            key = f"{base}|{cls_side}" if cls_side else base  # never associate my knight with theirs
            dets.append(TileDetection(base, d.conf, tile, side, hp, key=key))
        tracks, spawns = self.tracker.update(dets, t_ms)
        self.grouper.expire(t_ms)
        for ev in spawns:
            for play in self.grouper.add(ev):
                self._on_play(play)
        self._tracks = tracks

    def _effect_side(self, base: str, t_ms: int) -> str:
        """A spell effect is mine if I played that spell recently, else the opponent's.

        Location is useless here: spells land on the *other* player's half.
        """
        card = self.cards.card_for_class(base)
        if card is not None:
            for key, t in self._my_spells:
                if key == card.key and 0 <= t_ms - t <= self.my_spell_window_ms:
                    return "me"
        return "opp"

    def _tower_detection(self, base: str, tile: tuple[float, float]) -> None:
        if base != "tower_rubble":
            return
        name, pos = min(self.arena.towers.items(), key=lambda kv: (kv[1][0] - tile[0]) ** 2 + (kv[1][1] - tile[1]) ** 2)
        if (pos[0] - tile[0]) ** 2 + (pos[1] - tile[1]) ** 2 <= 3.0 ** 2:
            self._mark_destroyed(name)

    def _mark_destroyed(self, name: str) -> None:
        if name in self._destroyed:
            return
        self._destroyed.add(name)
        self.towers[name] = 0
        rec = {"kind": "tower_destroyed", "tower": name, "t_ms": self._last_t_ms}
        self.events.append(rec)
        if self.on_event:
            self.on_event(rec)

    def _extrapolate(self, t_ms: int) -> list[Unit]:
        """Tracks -> units at ``t_ms``: positions advanced along velocity since the last detection."""
        units = []
        for tr in self._tracks:
            if is_effect(tr.cls):
                continue  # spell effects make play events, not units
            dt = min(max(0, t_ms - tr.last_ms), self.extrapolate_max_ms) / 1000
            x = min(max(tr.tile[0] + tr.vel[0] * dt, 0.0), self.arena.width)
            y = min(max(tr.tile[1] + tr.vel[1] * dt, 0.0), self.arena.height)
            card = self.cards.card_for_class(tr.cls)
            units.append(Unit(tr.id, tr.cls, tr.side, (x, y), tr.vel, tr.hp_pct, tr.first_ms, tr.conf,
                              flying=flying_of(tr.cls, card)))
        return units

    def _on_play(self, play: PlayEvent) -> None:
        rec = {"kind": f"{play.side}_spawn_play", "card": play.card, "tile": play.tile, "t_ms": play.t_ms}
        self.events.append(rec)
        if self.on_event:
            self.on_event(rec)
        if play.side == "opp":
            self.opp.on_play(play.card, self.cards.cost(play.card), play.t_ms / 1000)

    def _read_towers(self, img: np.ndarray, warns: list[str]) -> None:
        if self.tower_states is not None:
            for name in self.tower_states.read(img, self.layout.towers):
                self._mark_destroyed(name)
        if self.digits is None:
            return
        for name, roi in self.layout.towers.items():
            if name in self._destroyed:
                continue  # destruction is permanent; don't let a digit misread revive it
            hp = self.digits.read(roi.crop(img))
            if hp is None:
                continue
            prev = self.towers.get(name)
            if prev is not None and hp > prev + 5:
                warns.append("tower_hp_increase")
                continue
            self.towers[name] = hp

    # --- match clock ------------------------------------------------------------
    def _read_clock(self, img: np.ndarray, t_ms: int, warns: list[str]) -> tuple[float | None, float | None]:
        """Read the clock ROI at most once per second, then compare it to frame time.

        With digit templates available the ``M:SS`` glyphs are parsed into seconds;
        without them the crop mean is still refreshed as an activity signal (the HUD
        clock is never blank while a battle is live).  Drift is ``frame_time - clock``;
        more than a second means the captured stream and the game clock disagree.
        """
        if t_ms - self._clock_read_ms >= 1000:
            crop = self.layout.clock.crop(img)
            self._clock_read_ms = t_ms
            self.clock_activity = float(crop.mean()) if crop.size else None
            self._clock_match_t_s = self._parse_clock(crop)
        match_t_s = self._clock_match_t_s
        drift = None if match_t_s is None else t_ms / 1000.0 - match_t_s
        if drift is not None and drift > 1.0:
            warns.append("clock_drift")
        return match_t_s, drift

    def _parse_clock(self, crop: np.ndarray) -> float | None:
        """Parse ``M:SS`` from a clock crop; ``None`` when digits are unavailable/unreadable."""
        if self.digits is None or not getattr(self.digits, "templates", None) or crop.size == 0:
            return None
        # Clock digits can be small on a downscaled capture; 0.3 matches the template
        # bootstrap in ``DigitReader.from_font`` and still drops the tiny colon dots.
        glyphs = segment(binarize(crop), min_h_frac=0.3)
        if not glyphs:
            return None
        chars = []
        for glyph in glyphs:
            digit, _ = self.digits.classify(glyph)
            if digit is None:
                return None
            chars.append(digit)
        try:
            # The colon is not a tall component, so it never reaches ``segment``.
            if len(chars) == 1:
                minutes, seconds = 0, int(chars[0])
            else:
                minutes, seconds = int("".join(chars[:-2]) or "0"), int("".join(chars[-2:]))
        except ValueError:
            return None
        if seconds >= 60:
            return None
        return float(minutes * 60 + seconds)
