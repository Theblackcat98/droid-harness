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

import math
from collections import Counter
from collections.abc import Callable

import numpy as np

from ....core.transport.frame_slot import Frame
from ..arena import Arena
from ..cards import CardDB
from ..elixir import ElixirModel, phase_at
from ..opponent_model import OpponentModel
from ..policy.levels import king_level_from_princess_hp
from ..state import GameState, Unit
from .cards import CardMatcher
from .classes import flying_of, is_effect, is_generic, is_tower, split_side
from .detector import AsyncDetector, Detector
from .digits import DigitReader, binarize, locate_number, segment
from .grid import Layout, Roi, TileMapper
from .hud import is_greyed, read_elixir
from .ownership import ownership_and_hp, strong_bar_side
from .towers import TowerStateReader
from .tracker import PlayEvent, SpawnGrouper, TileDetection, Track, Tracker

TOWER_SIDE = {"me_left": "me", "me_right": "me", "me_king": "me",
              "opp_left": "opp", "opp_right": "opp", "opp_king": "opp"}

# Standard Clash Royale Princess Tower max HP -> King Tower max HP by level (Lv 1..15).
PRINCESS_TO_KING_HP = (
    (1400, 2400), (1512, 2568), (1624, 2736), (1750, 2904), (1890, 3096),
    (2030, 3312), (2184, 3528), (2352, 3768), (2534, 4008), (2786, 4392),
    (3052, 4824), (3346, 5304), (3668, 5832), (4032, 6408), (4424, 7032),
)


def infer_king_max_hp(princess_max_hp: int) -> int:
    best_p, best_k = min(PRINCESS_TO_KING_HP, key=lambda pk: abs(pk[0] - princess_max_hp))
    if abs(best_p - princess_max_hp) <= 80:
        return best_k
    return int(round(princess_max_hp * 1.66))


class CRPerception:
    def __init__(self, *, layout: Layout, tiles: TileMapper, cards: CardDB, arena: Arena, matcher: CardMatcher,
                 digits: DigitReader | None = None, detector: Detector | None = None, detect_every: int = 1,
                 tower_every_ms: int = 500, on_my_play: Callable[[str, int], None] | None = None,
                 on_event: Callable[[dict], None] | None = None,
                 flying_foot_frac: float = 0.15, async_detect: bool = False,
                 tower_states: TowerStateReader | None = None, my_spell_window_ms: int = 3500,
                 extrapolate_max_ms: int = 500, harvest=None) -> None:
        self.layout, self.tiles, self.cards, self.arena = layout, tiles, cards, arena
        self.matcher, self.digits, self.detector = matcher, digits, detector
        self.detect_every, self.tower_every_ms = max(1, detect_every), tower_every_ms
        self.on_my_play, self.on_event = on_my_play, on_event
        self.flying_foot_frac = flying_foot_frac  # air units: box bottom is not the ground point
        self.tower_states = tower_states
        self.harvest = harvest  # TemplateHarvester (opt-in): crops of cards the matcher can't read yet
        self.my_spell_window_ms = my_spell_window_ms  # my spell play -> its effect on screen
        self.extrapolate_max_ms = extrapolate_max_ms
        self.async_detector: AsyncDetector | None = None
        if detector is not None and async_detect:
            self.async_detector = detector if isinstance(detector, AsyncDetector) else AsyncDetector(detector)
        self.start_pts_us: int | None = None
        self._generation = 0
        self.hp_offsets: dict[str, tuple[float, float]] = {}  # learned HP-label offsets, kept across matches
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
        if getattr(self, "harvest", None) is not None:
            self.harvest.reset()
        self.grouper = SpawnGrouper(self.cards.card_for_class)
        self.hand: list[str | None] = [None] * 4
        self.hand_fresh: list[bool] = []
        self._slot_spent: list[bool] = [False] * len(self.layout.hand)
        self._slot_spent_ms: list[int] = [0] * len(self.layout.hand)
        self.next_card: str | None = None
        self.towers: dict[str, int] = {}
        self.tower_max_hp: dict[str, int] = {}
        self.tower_read_counts: Counter = Counter()  # successful HP reads per tower (coverage)
        self._hp_fails: Counter = Counter()  # consecutive failed reads per tower
        self._hp_search_ms: dict[str, int] = {}
        self._hp_search_misses: Counter = Counter()
        self._hp_rises: dict[str, list[int]] = {}
        self.last_detections: list[dict] = []
        self._tower_read_ms = -10**9
        self._tower_reads = 0  # tower read cycles this match
        self._clock_read_ms = -10**9
        self._clock_match_t_s: float | None = None
        self.clock_activity: float | None = None
        self.frame_i = 0
        self.warnings: Counter = Counter()
        self.units: list[Unit] = []
        self.events: list[dict] = []
        self._opp_last_play_ms: dict[str, int] = {}
        self._opp_recent_plays: list[tuple[int, tuple[float, float]]] = []
        self._pre_reported_opp: list[tuple[float, float]] = []
        self._last_hud: float | None = None
        self._battle_hud_seen: bool = False
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
        self._last_hud = hud
        if hud is not None and hud >= 1.0:
            self._battle_hud_seen = True
        if self.frame_i == 0 and hud is not None and hud >= 1.0:
            self.elixir.reset(hud, t_s)
            self.opp.elixir.low = self.opp.elixir.high = hud
        elif (t_s <= 4.0 and hud is not None and hud >= 1.0
              and not self._opp_recent_plays and not any(e["kind"] == "my_play" for e in self.events)
              and hud > self.elixir.value + 1.0):
            self.elixir.reset(hud, t_s)
            self.opp.elixir.low = max(self.opp.elixir.low, hud)
            self.opp.elixir.high = max(self.opp.elixir.high, hud)
        self.elixir.advance(t_s)
        crops = self._read_hand(img, t_ms, warns, hud=hud)  # detects our plays (spend) before fusing the HUD
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
        effective_t_s = match_t_s if (self.layout.clock_countdown and match_t_s is not None and match_t_s >= 0) else t_s
        self.frame_i += 1
        for w in warns:
            self.warnings[w] += 1
        state = GameState(
            t_ms=t_ms, host_ns=frame.t_decoded_ns, frame_size=(img.shape[1], img.shape[0]),
            phase=phase_at(effective_t_s), my_elixir=self.elixir.value, opp_elixir=self.opp.elixir.interval(),
            hand=list(self.hand), hand_affordable=affordable, hand_fresh=list(self.hand_fresh),
            next_card=self.next_card, units=list(self.units),
            towers=dict(self.towers), crowns=crowns, opp_deck_seen=list(self.opp.cycle.deck_seen),
            opp_hand_possible=self.opp.cycle.possible_hand(), opp_last_four=self.opp.cycle.last_four,
            pocket_open={"left": self.towers.get("opp_left") == 0, "right": self.towers.get("opp_right") == 0},
            warnings=warns)
        # GameState is the canonical state class; clock & max-HP fields are attached here
        # (rather than changing its schema) because only perception produces them.
        state.match_t_s = match_t_s
        state.clock_drift_s = clock_drift_s
        state.clock_activity = self.clock_activity
        state.tower_max_hp = dict(self.tower_max_hp)
        state.my_king_level = king_level_from_princess_hp(
            max(self.tower_max_hp.get("me_left", 0), self.tower_max_hp.get("me_right", 0)))
        state.opp_king_level = king_level_from_princess_hp(
            max(self.tower_max_hp.get("opp_left", 0), self.tower_max_hp.get("opp_right", 0)))
        return state

    # --- pieces ---------------------------------------------------------------
    def _read_hand(self, img: np.ndarray, t_ms: int, warns: list[str], *,
                   hud: float | None = None) -> list:
        # Unreadable slots keep their last known card so a play is still detected when the
        # next read succeeds, but they're marked stale and the policy won't act on them.
        new = list(self.hand)
        self.hand_fresh = [False] * len(self.layout.hand)
        crops = []
        unreadable: list[int] = []
        for i, roi in enumerate(self.layout.hand):
            crop = roi.crop(img)
            r = self.matcher.match(crop)
            if self.harvest is not None:
                self.harvest.observe(i, crop, r.card if r.accepted else None, t_ms, is_greyed(crop))
            if not r.accepted:
                unreadable.append(i)
                continue
            new[i] = r.card
            self.hand_fresh[i] = True
            crops.append((r.card, crop))
        nr = self.matcher.match(self.layout.next_card.crop(img))
        if nr.accepted:
            self.next_card = nr.card
        for i, (old, cur) in enumerate(zip(self.hand, new, strict=False)):
            if not self.hand_fresh[i]:
                continue
            if old is not None and cur is not None and old != cur:
                if not self._slot_spent[i]:
                    self._my_play(old, i, t_ms, hud=hud)
                self._slot_spent[i] = False
            elif old == cur:
                self._slot_spent[i] = False
        # Early spend detection while the played card's slot is empty (~1.5s before refill).
        while hud is not None:
            best: tuple[float, int, str] | None = None
            for i in unreadable:
                if self._slot_spent[i]:
                    continue
                old = self.hand[i]
                if old is None or old not in self.cards:
                    continue
                cost = self.cards.cost(old)
                if self.elixir.value - hud >= max(0.8, cost - 1.0):
                    err = abs(max(0.0, self.elixir.value - cost) - hud)
                    if err <= 1.0 and (best is None or err < best[0]):
                        best = (err, i, old)
            if best is None:
                break
            _, slot_i, old_card = best
            if self.harvest is not None:
                self.harvest.expect(slot_i, self.next_card, t_ms)  # the next card refills this slot
            self._my_play(old_card, slot_i, t_ms, hud=hud)
            self._slot_spent[slot_i] = True
            self._slot_spent_ms[slot_i] = t_ms
        for i in unreadable:
            if not (self._slot_spent[i] and t_ms - self._slot_spent_ms[i] <= 2500):
                warns.append("hand_unreadable")
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

    def _my_play(self, card: str, slot: int, t_ms: int, *, hud: float | None = None) -> None:
        if card in self.cards:
            cost = self.cards.cost(card)
            if hud is None or abs(max(0.0, self.elixir.value - cost) - hud) < abs(self.elixir.value - hud):
                self.elixir.spend(cost)
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
        vis_dets: list[dict] = []
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
                if cls_side and d.conf < 0.75:
                    strong_side, strong_hp = strong_bar_side(img, (bx1, by1, bx2, by2), min_fill=0.18)
                    if strong_side is not None and strong_side != cls_side:
                        side = strong_side
                        if strong_hp is not None:
                            hp = strong_hp
                if is_generic(base) and d.conf < 0.65 and side is not None:
                    cx_tile = self.tiles.px_to_tile((fx, (by1 + by2) / 2), img.shape)
                    if any(
                        TOWER_SIDE.get(tk) == side
                        and self.towers.get(tk, 1) > 0
                        and math.hypot(cx_tile[0] - tpos[0], cx_tile[1] - tpos[1]) <= 1.9
                        for tk, tpos in self.arena.towers.items()
                    ):
                        continue
            key = f"{base}|{side}" if side else base  # never associate my knight with theirs
            dets.append(TileDetection(base, d.conf, tile, side, hp, key=key))
            vis_dets.append({"cls": base, "side": side, "conf": round(float(d.conf), 3),
                             "box": (round(bx1, 1), round(by1, 1), round(bx2, 1), round(by2, 1)),
                             "tile": (round(tile[0], 2), round(tile[1], 2)), "hp": hp})
        self.last_detections = vis_dets
        self._pre_reported_opp = [
            tr.tile for tr in self.tracker.tracks.values() if tr.spawn_reported and tr.side == "opp"
        ]
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

    def _is_legal_opp_spawn(self, play: PlayEvent) -> bool:
        if is_effect(play.card) or play.card in ("miner", "goblin_barrel", "goblin_drill"):
            return True
        c = self.cards.get(play.card)
        if c is not None and c.is_spell:
            return True
        river_mid = sum(self.arena.river_y) / 2
        if play.tile[1] >= river_mid - 1.2:
            return True
        lane = self.arena.lane_of(play.tile[0])
        return self.towers.get(f"me_{lane}") == 0 and play.tile[1] >= 10.5

    def _is_building_spawn(self, play: PlayEvent) -> bool:
        if play.card == "skeletons":
            sources = ("tombstone", "witch")
        elif play.card == "spear_goblins":
            sources = ("goblin_hut",)
        else:
            return False
        return any(
            tr.side == "opp" and tr.cls in sources and math.dist(play.tile, tr.tile) <= 2.8
            for tr in self.tracker.tracks.values()
        )

    def _accept_opp_play(self, play: PlayEvent) -> bool:
        if self.frame_i > 0 and not self._battle_hud_seen and play.t_ms <= 3500:
            return False
        if not self._is_legal_opp_spawn(play) or self._is_building_spawn(play):
            return False
        t_ms = play.t_ms
        t_s = t_ms / 1000.0
        self.opp.advance(t_s)
        c = self.cards.get(play.card)
        if play.upgraded_from:
            prev_t = self._opp_last_play_ms.get(play.upgraded_from)
            return prev_t is not None and (t_ms - prev_t) <= 2000
        if c is not None and c.kind == "building":
            if play.peak_conf < 0.85 and (play.speed > 0.5 or play.hits < 5):
                return False
        min_cost = float(c.cost) if c is not None else 2.0
        if self.opp.elixir.high + 1.5 < min_cost and not (play.peak_conf >= 0.72 and play.hits >= 5):
            return False
        if (c is not None and len(self.opp.cycle.deck_seen) >= 8
                and play.card not in self.opp.cycle.deck_seen and play.card != "mirror"):
            if not (play.peak_conf >= 0.82 and play.hits >= 6):
                return False
        if is_generic(play.card):
            self._opp_recent_plays = [(t, p) for t, p in self._opp_recent_plays if t_ms - t <= 3000]
            if any((t_ms - t) <= 1500 and math.dist(play.tile, pt) <= 4.0 for t, pt in self._opp_recent_plays):
                return False
            if any(math.dist(play.tile, pt) <= 2.5 for pt in self._pre_reported_opp):
                return False
            return True
        prev_t = self._opp_last_play_ms.get(play.card)
        if prev_t is not None:
            dt_s = (t_ms - prev_t) / 1000.0
            if dt_s < 5.0:
                return False
            min_cycle_s = 12.0 if phase_at(t_s) == "x1" else 9.0
            if play.card in self.opp.cycle.last_four and dt_s < min_cycle_s:
                return False
        return True

    def _on_play(self, play: PlayEvent) -> None:
        if play.side == "opp":
            if not self._accept_opp_play(play):
                return
            t_s = play.t_ms / 1000.0
            if play.upgraded_from and play.card in self.cards:
                old_c = self.cards.cost(play.upgraded_from) if play.upgraded_from in self.cards else 0
                extra = max(0, self.cards.cost(play.card) - old_c)
                self.opp.upgrade_play(play.upgraded_from, play.card, extra, t_s)
                self._opp_last_play_ms.pop(play.upgraded_from, None)
                self._opp_last_play_ms[play.card] = play.t_ms
            elif play.card in self.cards:
                self.opp.on_play(play.card, self.cards.cost(play.card), t_s)
                self._opp_last_play_ms[play.card] = play.t_ms
            elif is_generic(play.card):
                self.opp.on_play(None, (2.0, 5.0), t_s)
            self._opp_recent_plays.append((play.t_ms, play.tile))
        rec = {"kind": f"{play.side}_spawn_play", "card": play.card, "tile": play.tile, "t_ms": play.t_ms}
        if play.upgraded_from:
            rec["upgraded_from"] = play.upgraded_from
        self.events.append(rec)
        if self.on_event:
            self.on_event(rec)

    def _read_towers(self, img: np.ndarray, warns: list[str]) -> None:
        if self.tower_states is not None:
            for name in self.tower_states.read(img, self.layout.towers):
                self._mark_destroyed(name)
        if self.digits is None:
            return
        hp_rois = self.layout.tower_hp or self.layout.towers
        self._tower_reads += 1
        searched = False
        for name, roi in hp_rois.items():
            if name in self._destroyed:
                continue  # destruction is permanent; don't let a digit misread revive it
            off = self.hp_offsets.get(name)
            use = roi if off is None else Roi(roi.x0 + off[0], roi.y0 + off[1], roi.x1 + off[0], roi.y1 + off[1])
            hp = self.digits.read(use.crop(img))
            if hp is None or hp <= 0 or hp > 9999:
                self._hp_fails[name] += 1
                # The calibrated region can sit a few pixels off the label (labels move between
                # arenas/modes; the opponent's were ~10 px low on phone-a). After two misses, search
                # a band around it once per 2 s and lock the offset that reads cleanly.
                # Back off when a label simply isn't there (an unactivated King Tower): 2 s, 4 s, ... 20 s.
                wait_ms = min(20_000, 2000 * 2 ** self._hp_search_misses[name])
                if (not searched and self._hp_fails[name] >= 2
                        and self._last_t_ms - self._hp_search_ms.get(name, -10**9) >= wait_ms):
                    searched = True  # at most one ~15 ms search per tower read cycle
                    self._hp_search_ms[name] = self._last_t_ms
                    mx = self.tower_max_hp.get(name)
                    v, offset, _ = locate_number(img, roi, self.digits,
                                                 plausible=lambda x, mx=mx: 100 <= x <= (mx + 5 if mx else 9999))
                    if v is not None:
                        self._hp_search_misses[name] = 0
                        self.hp_offsets[name] = offset
                        # Princess labels on one side share a row: give the twin the same vertical offset.
                        side, _, pos = name.partition("_")
                        twin = f"{side}_{'right' if pos == 'left' else 'left'}" if pos in ("left", "right") else None
                        if twin and twin not in self.hp_offsets:
                            self.hp_offsets[twin] = (0.0, offset[1])
                        hp = v
                    else:
                        self._hp_search_misses[name] += 1
                if hp is None or hp <= 0 or hp > 9999:
                    continue
            self._hp_fails[name] = 0
            prev = self.towers.get(name)
            if prev is None and self._last_t_ms <= 5000 and hp < 300:
                continue  # ignore 1-2 digit intro/crown artifacts on undamaged start towers
            if prev is not None and hp > prev + 5:
                # HP never rises. Either this read is wrong, or an earlier low read was: if the
                # same higher value repeats three times, the earlier one was the misread.
                rises = (self._hp_rises.get(name, []) + [hp])[-3:]
                self._hp_rises[name] = rises
                if len(rises) == 3 and max(rises) - min(rises) <= 10:
                    self._hp_rises[name] = []
                    self.events.append({"kind": "tower_hp_corrected", "tower": name, "from": prev, "to": hp,
                                        "t_ms": self._last_t_ms})
                else:
                    warns.append("tower_hp_increase")
                    continue
            else:
                self._hp_rises.pop(name, None)
            self.towers[name] = hp
            self.tower_read_counts[name] += 1
            self.tower_max_hp[name] = max(self.tower_max_hp.get(name, 0), hp)
        # Seed unactivated King Tower baseline HP from observed Princess Tower max HP so
        # engagements and emergency_defend know King HP before its bar appears on screen.
        for side in ("me", "opp"):
            king = f"{side}_king"
            if king in self._destroyed or king in self.towers:
                continue
            p_max = max(self.tower_max_hp.get(f"{side}_left", 0), self.tower_max_hp.get(f"{side}_right", 0))
            if p_max >= 1000:
                inferred = infer_king_max_hp(p_max)
                self.towers[king] = inferred
                self.tower_max_hp[king] = inferred

    # --- match clock ------------------------------------------------------------
    def _read_clock(self, img: np.ndarray, t_ms: int, warns: list[str]) -> tuple[float | None, float | None]:
        """Read the clock ROI at most once per second, then compare it to frame time.

        With digit templates available the ``M:SS`` glyphs are parsed into seconds;
        when ``layout.clock_countdown`` is true (real Clash Royale HUD), remaining time
        is converted to elapsed match seconds (``3:00 -> 0:00`` regulation, ``2:00 -> 0:00``
        overtime). Without templates, the crop mean is refreshed as an activity signal.
        """
        if t_ms - self._clock_read_ms >= 1000:
            crop = self.layout.clock.crop(img)
            self._clock_read_ms = t_ms
            self.clock_activity = float(crop.mean()) if crop.size else None
            raw_s = self._parse_clock(crop, allow_red=self.layout.clock_countdown)
            if raw_s is None:
                self._clock_match_t_s = None
            elif self.layout.clock_countdown:
                t_s = t_ms / 1000.0
                # Regulation counts down from 180s; overtime counts down from 120s.
                self._clock_match_t_s = (180.0 - raw_s) if (t_s < 175.0 and raw_s <= 180.0) else (300.0 - raw_s)
            else:
                self._clock_match_t_s = raw_s
        match_t_s = self._clock_match_t_s
        drift = None if match_t_s is None else t_ms / 1000.0 - match_t_s
        tol = 5.0 if self.layout.clock_countdown else 1.0  # allow 2-3s battle intro banner offset on real HUD
        if drift is not None and (abs(drift) > tol if self.layout.clock_countdown else drift > tol):
            warns.append("clock_drift")
        return match_t_s, drift

    def _parse_clock(self, crop: np.ndarray, *, allow_red: bool = False) -> float | None:
        """Parse ``M:SS`` from a clock crop; ``None`` when digits are unavailable/unreadable."""
        if self.digits is None or not getattr(self.digits, "templates", None) or crop.size == 0:
            return None
        # Clock digits can be small on a downscaled capture; 0.3 matches the template
        # bootstrap in ``DigitReader.from_font`` and still drops the tiny colon dots.
        glyphs = segment(binarize(crop, allow_red=allow_red), min_h_frac=0.3)
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
