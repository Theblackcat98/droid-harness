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
from .towers import TowerStateReader, has_tower_hp_bar
from .tracker import PlayEvent, SpawnGrouper, TileDetection, Track, Tracker

TOWER_SIDE = {"me_left": "me", "me_right": "me", "me_king": "me",
              "opp_left": "opp", "opp_right": "opp", "opp_king": "opp"}

# Valid sub-spawns produced by cards in our deck (for team='me' deck filtering).
_DECK_SUBSPAWNS: dict[str, tuple[str, ...]] = {
    "tombstone": ("skeleton",),
    "witch": ("skeleton",),
    "skeleton_barrel": ("skeleton",),
    "goblin_cage": ("goblin_brawler",),
    "goblin_hut": ("spear_goblin",),
    "goblin_giant": ("spear_goblin",),
    "goblin_drill": ("goblin",),
    "barbarian_hut": ("barbarian",),
    "battle_ram": ("barbarian",),
    "night_witch": ("bat",),
    "golem": ("golemite",),
    "lava_hound": ("lava_pup",),
    "elixir_golem": ("elixir_golem_mid", "elixir_golem_small"),
    "phoenix": ("phoenix_egg", "phoenix_small"),
}

# Common 214-class YOLO misclassifications -> preferred deck unit classes if present in our deck.
_VISUAL_CONFUSION_FALLBACK: dict[str, tuple[str, ...]] = {
    "pekka": ("mini_pekka",),
    "royal_giant": ("giant",),
    "electro_giant": ("giant",),
    "goblin_giant": ("giant",),
    "golem": ("giant",),
    "giant_skeleton": ("giant",),
    "mega_minion": ("minion",),
    "bat": ("minion",),
    "phoenix": ("minion",),
    "phoenix_big": ("minion",),
    "phoenix_small": ("minion",),
    "lava_pup": ("minion",),
    "minion_horde": ("minion",),
    "mother_witch": ("archer", "musketeer"),
    "firecracker": ("archer", "musketeer"),
    "princess": ("archer", "musketeer"),
    "wizard": ("archer", "musketeer"),
    "witch": ("archer", "musketeer"),
    "night_witch": ("archer", "musketeer"),
    "prince": ("knight", "mini_pekka"),
    "golden_knight": ("knight",),
    "little_prince": ("archer", "knight"),
    "valkyrie": ("knight",),
    "barbarian": ("knight",),
}

_BACKFIELD_INVADERS = frozenset({"miner", "goblin", "goblin_barrel", "goblin_drill", "skeleton"})


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = area_a + area_b - inter
    return inter / den if den > 1e-6 else 0.0


def _is_subseq(short: str, long: str) -> bool:
    it = iter(long)
    return all(ch in it for ch in short)


def _is_suspicious_hp_drop(prev: int, hp: int) -> bool:
    """Detect likely single-frame OCR leading/middle digit drops or rubble glitches."""
    if hp >= prev:
        return False
    sp, sh = str(prev), str(hp)
    if sp.endswith(sh):
        return True
    if len(sh) <= len(sp) - 2:
        return True
    if prev >= 350 and hp < 120:
        return True
    if prev - hp >= 300 and _is_subseq(sh, sp):
        return True
    return prev >= 1000 and hp < 1000 and abs(hp - (prev % 1000)) <= 30

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
                 extrapolate_max_ms: int = 500, harvest=None,
                 my_deck: list[str] | tuple[str, ...] | None = None,
                 min_unit_conf: float = 0.42) -> None:
        self.layout, self.tiles, self.cards, self.arena = layout, tiles, cards, arena
        self.matcher, self.digits, self.detector = matcher, digits, detector
        self.detect_every, self.tower_every_ms = max(1, detect_every), tower_every_ms
        self.on_my_play, self.on_event = on_my_play, on_event
        self.flying_foot_frac = flying_foot_frac  # air units: box bottom is not the ground point
        self.tower_states = tower_states
        self.harvest = harvest  # TemplateHarvester (opt-in): crops of cards the matcher can't read yet
        self.my_spell_window_ms = my_spell_window_ms  # my spell play -> its effect on screen
        self.extrapolate_max_ms = extrapolate_max_ms
        self.min_unit_conf = min_unit_conf
        self._my_deck_cards: set[str] = set()
        self._my_deck_unit_classes: set[str] = set()
        self.set_my_deck(my_deck)
        self.async_detector: AsyncDetector | None = None
        if detector is not None and async_detect:
            self.async_detector = detector if isinstance(detector, AsyncDetector) else AsyncDetector(detector)
        self.start_pts_us: int | None = None
        self._generation = 0
        self.hp_offsets: dict[str, tuple[float, float]] = {}  # learned HP-label offsets, kept across matches
        self.reset()

    def set_my_deck(self, deck: list[str] | tuple[str, ...] | None) -> None:
        """Configure our known 8-card deck to filter impossible team='me' hallucinations."""
        if not deck:
            self._my_deck_cards = set()
            self._my_deck_unit_classes = set()
            return
        self._my_deck_cards = set(deck)
        cls_set: set[str] = set()
        for k in deck:
            c = self.cards.get(k)
            if c is not None:
                cls_set.update(c.unit_classes)
            cls_set.update(_DECK_SUBSPAWNS.get(k, ()))
        self._my_deck_unit_classes = cls_set

    def record_my_action(self, card: str, tile: tuple[float, float], t_ms: int) -> None:
        """Record a troop/building placement chosen by our policy for team-flip suppression."""
        c = self.cards.get(card)
        if c is not None and c.is_spell:
            return
        self._my_recent_actions = [a for a in self._my_recent_actions if t_ms - a[2] <= 6000]
        self._my_recent_actions.append((card, (float(tile[0]), float(tile[1])), t_ms))

    # --- match lifecycle ----------------------------------------------------
    def reset(self) -> None:
        self.elixir = ElixirModel()
        self.opp = OpponentModel()
        self.tracker = Tracker(river_mid_y=sum(self.arena.river_y) / 2)
        self._tracks: list[Track] = []
        self._my_spells: list[tuple[str, int]] = []  # (card key, t_ms) of my recent spell plays
        self._my_recent_actions: list[tuple[str, tuple[float, float], int]] = []  # (card, tile, t_ms)
        self._my_recent_hand_plays: list[tuple[str, int]] = []  # (card, t_ms) from hand slot transitions
        self._me_last_spawn_ms: dict[str, int] = {}
        self._last_my_play_ms: int = -10**9
        self._destroyed: set[str] = set()
        self._generation = getattr(self, "_generation", 0) + 1  # drops async results from a past match
        if self.tower_states is not None:
            self.tower_states.reset()
        if getattr(self, "harvest", None) is not None:
            self.harvest.reset()
        self.grouper = SpawnGrouper(self.cards.card_for_class)
        self.hand: list[str | None] = [None] * 4
        self.hand_fresh: list[bool] = []
        n_slots = len(self.layout.hand)
        self._slot_spent: list[bool] = [False] * n_slots
        self._slot_spent_ms: list[int] = [0] * n_slots
        self._slot_last_play_ms: list[int] = [-10**9] * n_slots
        self._slot_unreadable_streak: list[int] = [0] * n_slots
        self._slot_prev_fresh: list[bool] = [False] * n_slots
        self._slot_pending_card: list[tuple[str | None, int]] = [(None, 0)] * n_slots
        self._afford_mismatch_streak: int = 0
        self.next_card: str | None = None
        self.towers: dict[str, int] = {}
        self.tower_max_hp: dict[str, int] = {}
        self.tower_read_counts: Counter = Counter()  # successful HP reads per tower (coverage)
        self._hp_fails: Counter = Counter()  # consecutive failed reads per tower
        self._hp_search_ms: dict[str, int] = {}
        self._hp_search_misses: Counter = Counter()
        self._hp_rises: dict[str, list[int]] = {}
        self._hp_drops: dict[str, list[int]] = {}
        self._hp_init_low: dict[str, list[int]] = {}
        self._tower_bar_seen: set[str] = set()
        self._tower_bar_absent: Counter = Counter()
        self.last_detections: list[dict] = []
        self._tower_read_ms = -10**9
        self._tower_reads = 0  # tower read cycles this match
        self._clock_read_ms = -10**9
        self._clock_match_t_s: float | None = None
        self._clock_in_ot: bool = False
        self._clock_last_raw_s: float | None = None
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
                self._slot_unreadable_streak[i] += 1
                self._slot_pending_card[i] = (None, 0)
                continue
            self._slot_unreadable_streak[i] = 0
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
                in_cooldown = (t_ms - self._slot_last_play_ms[i]) < 1200
                # Single-frame card-shimmer flicker guard: if the slot was continuously fresh
                # (never went empty) and elixir did not drop, require 3 agreeing frames before
                # updating identity and do not log a phantom play.
                no_drop = (
                    self._slot_prev_fresh[i]
                    and not self._slot_spent[i]
                    and hud is not None
                    and old in self.cards
                    and self.cards.cost(old) >= 2
                    and (self.elixir.value - hud) < 0.5
                )
                if in_cooldown:
                    self._slot_spent[i] = False
                    self._slot_pending_card[i] = (None, 0)
                elif no_drop:
                    p_card, p_cnt = self._slot_pending_card[i]
                    p_cnt = p_cnt + 1 if p_card == cur else 1
                    self._slot_pending_card[i] = (cur, p_cnt)
                    if p_cnt < 3:
                        new[i] = old  # hold previous card until flicker settles
                    else:
                        self._slot_pending_card[i] = (None, 0)
                        self._slot_spent[i] = False
                else:
                    self._slot_pending_card[i] = (None, 0)
                    if not self._slot_spent[i]:
                        self._my_play(old, i, t_ms, hud=hud)
                    self._slot_spent[i] = False
            elif old == cur:
                self._slot_spent[i] = False
                self._slot_pending_card[i] = (None, 0)
        # Early spend detection while the played card's slot is empty (~1.5s before refill).
        while hud is not None:
            best: tuple[float, int, str] | None = None
            for i in unreadable:
                if self._slot_spent[i] or (t_ms - self._slot_last_play_ms[i]) < 1200:
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
        if self._battle_hud_seen or t_ms > 4000:
            for i in unreadable:
                in_refill = (
                    (self._slot_spent[i] and t_ms - self._slot_spent_ms[i] <= 2500)
                    or (t_ms - self._slot_last_play_ms[i] <= 2500)
                )
                if not in_refill and self._slot_unreadable_streak[i] >= 4:
                    warns.append("hand_unreadable")
        self._slot_prev_fresh = list(self.hand_fresh)
        self.hand = new
        return crops

    def _check_affordability(self, crops: list, warns: list[str]) -> None:
        """Greyed-out cards are a free check on the elixir reading."""
        if (not self._battle_hud_seen and self._last_t_ms <= 4000) or (self._last_t_ms - self._last_my_play_ms) < 800:
            self._afford_mismatch_streak = 0
            return
        mismatched = False
        for card, crop in crops:
            if card not in self.cards:
                continue
            cost, greyed = self.cards.cost(card), is_greyed(crop)
            if (greyed and self.elixir.value >= cost + 0.8) or (not greyed and self.elixir.value < cost - 0.8):
                mismatched = True
                break
        if mismatched:
            self._afford_mismatch_streak += 1
            if self._afford_mismatch_streak >= 4:
                warns.append("affordability_mismatch")
        else:
            self._afford_mismatch_streak = 0

    def _my_play(self, card: str, slot: int, t_ms: int, *, hud: float | None = None) -> None:
        if 0 <= slot < len(self._slot_last_play_ms):
            self._slot_last_play_ms[slot] = t_ms
        self._last_my_play_ms = t_ms
        if card in self.cards:
            cost = self.cards.cost(card)
            if hud is None or abs(max(0.0, self.elixir.value - cost) - hud) < abs(self.elixir.value - hud):
                self.elixir.spend(cost)
            if self.cards[card].is_spell:
                self._my_spells = [(k, t) for k, t in self._my_spells if t_ms - t <= self.my_spell_window_ms]
                self._my_spells.append((card, t_ms))
            else:
                self._my_recent_hand_plays = [(k, t) for k, t in self._my_recent_hand_plays if t_ms - t <= 25_000]
                self._my_recent_hand_plays.append((card, t_ms))
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

    def _match_recent_own_play(self, base: str, flying: bool, tile: tuple[float, float],
                               t_ms: int, est_opp: list[tuple[tuple[float, float], int]]) -> str | None:
        """Return our played unit class if an ``opp`` detection matches our just-placed unit."""
        if tile[1] >= 13.0:
            return None

        def has_prior_opp(before_ms: int | None) -> bool:
            return any(
                math.dist(ot, tile) <= 2.8 and (before_ms is None or ofirst < before_ms)
                for ot, ofirst in est_opp
            )

        def matches_card(card_key: str) -> str | None:
            c = self.cards.get(card_key)
            if c is None or c.is_spell:
                return None
            uclasses = tuple(c.unit_classes) + _DECK_SUBSPAWNS.get(card_key, ())
            if not uclasses:
                return None
            if base in uclasses:
                return base
            if is_generic(base) and bool(c.flying) == flying:
                return uclasses[0]
            for cand in _VISUAL_CONFUSION_FALLBACK.get(base, ()):
                if cand in uclasses:
                    return cand
            if base == "musketeer" and "archer" in uclasses:
                return "archer"
            return None

        for card_key, act_tile, act_t in reversed(self._my_recent_actions):
            dt = t_ms - act_t
            if dt < 0 or dt > 4500:
                continue
            dx = abs(tile[0] - act_tile[0])
            dy = tile[1] - act_tile[1]
            if dx <= 2.8 and -1.5 <= dy <= 2.0 + 2.5 * (dt / 1000.0) and not has_prior_opp(act_t):
                mapped = matches_card(card_key)
                if mapped is not None:
                    return mapped

        lane = self.arena.lane_of(tile[0])
        if (tile[1] < 8.0 and self.towers.get(f"me_{lane}", 1) > 0
                and base not in _BACKFIELD_INVADERS and not has_prior_opp(None)):
            for card_key, play_t in reversed(self._my_recent_hand_plays):
                if 0 <= t_ms - play_t <= 5000:
                    mapped = matches_card(card_key)
                    if mapped is not None:
                        return mapped
        return None

    def _remap_my_class(self, base: str, flying: bool, tile: tuple[float, float],
                        conf: float, t_ms: int) -> str | None:
        """Filter/remap impossible non-deck classes and archer->musketeer confusions for team='me'."""
        if is_effect(base) or is_generic(base) or not self._my_deck_unit_classes:
            return base
        # Fix me:archers misclassified as me:musketeer when we played archers recently and not musketeer.
        if base == "musketeer" and "archer" in self._my_deck_unit_classes:
            arch_recent = (
                any(k == "archers" and 0 <= t_ms - t <= 8000 for k, _, t in self._my_recent_actions)
                or any(k == "archers" and 0 <= t_ms - t <= 8000 for k, t in self._my_recent_hand_plays)
            )
            musk_recent = (
                any(k == "musketeer" and 0 <= t_ms - t <= 25_000 for k, _, t in self._my_recent_actions)
                or any(k == "musketeer" and 0 <= t_ms - t <= 25_000 for k, t in self._my_recent_hand_plays)
            )
            if arch_recent and not musk_recent:
                return "archer"
        if base in self._my_deck_unit_classes:
            return base
        # Non-deck class on our team: remap to a recently played unit of the same air/ground type if nearby.
        for card_key, act_tile, act_t in reversed(self._my_recent_actions):
            dt = t_ms - act_t
            if 0 <= dt <= 5000 and math.dist(tile, act_tile) <= 3.5 + 2.0 * (dt / 1000.0):
                c = self.cards.get(card_key)
                if c is not None and not c.is_spell and bool(c.flying) == flying and c.unit_classes:
                    return c.unit_classes[0]
        for cand in _VISUAL_CONFUSION_FALLBACK.get(base, ()):
            if cand in self._my_deck_unit_classes:
                return cand
        if conf < 0.65:
            return None  # drop low-conf non-deck hallucination
        return "other_air" if flying else "other_ground"

    def _ingest(self, raw_dets, img: np.ndarray, origin: tuple[int, int], t_ms: int) -> None:
        x0, y0 = origin
        est_opp = [
            (tr.tile, tr.first_ms)
            for tr in self.tracker.tracks.values()
            if tr.side == "opp" and tr.hits >= 2
        ]
        parsed: list[dict] = []
        for d in raw_dets:
            base, cls_side = split_side(d.cls)
            bx1, by1, bx2, by2 = (d.box[0] + x0, d.box[1] + y0, d.box[2] + x0, d.box[3] + y0)
            fx, fy = (bx1 + bx2) / 2, by2
            if is_tower(base):
                self._tower_detection(base, self.tiles.px_to_tile((fx, (by1 + by2) / 2), img.shape))
                continue
            card0 = self.cards.card_for_class(base)
            flying = flying_of(base, card0)
            if flying:
                fy -= self.flying_foot_frac * (by2 - by1)
            tile = self.tiles.px_to_tile((fx, fy), img.shape)
            strong_side: str | None = None
            if is_effect(base):
                side, hp = self._effect_side(base, t_ms), None
            else:
                bar_side, hp = ownership_and_hp(img, (bx1, by1, bx2, by2))
                side = cls_side or bar_side  # a side-aware class beats weak HP-bar color
                if cls_side:
                    strong_side, strong_hp = strong_bar_side(img, (bx1, by1, bx2, by2), min_fill=0.18)
                    if strong_side is not None and strong_side != cls_side and (d.conf < 0.92 or is_generic(base)):
                        side = strong_side
                        if strong_hp is not None:
                            hp = strong_hp
                # Suppress own-unit team flips (team='opp' on our just-played spawn or backfield).
                if side == "opp" and strong_side != "opp":
                    flipped_cls = self._match_recent_own_play(base, flying, tile, t_ms, est_opp)
                    if flipped_cls is not None:
                        side = "me"
                        base = flipped_cls
                        card0 = self.cards.card_for_class(base)
                        flying = flying_of(base, card0)
                    elif (is_generic(base) and tile[1] < 8.0
                          and self.towers.get(f"me_{self.arena.lane_of(tile[0])}", 1) > 0
                          and not any(math.dist(ot, tile) <= 3.5 for ot, _ in est_opp)):
                        continue  # phantom generic opp unit on our empty backfield
                # Filter/remap non-deck hallucinations on team='me'.
                if side == "me":
                    remapped = self._remap_my_class(base, flying, tile, float(d.conf), t_ms)
                    if remapped is None:
                        continue
                    base = remapped
                    card0 = self.cards.card_for_class(base)
                    flying = flying_of(base, card0)
                if is_generic(base) and d.conf < 0.65 and side is not None:
                    cx_tile = self.tiles.px_to_tile((fx, (by1 + by2) / 2), img.shape)
                    if any(
                        TOWER_SIDE.get(tk) == side
                        and self.towers.get(tk, 1) > 0
                        and math.hypot(cx_tile[0] - tpos[0], cx_tile[1] - tpos[1]) <= 1.9
                        for tk, tpos in self.arena.towers.items()
                    ):
                        continue
            parsed.append({
                "base": base, "side": side, "conf": float(d.conf), "box": (bx1, by1, bx2, by2),
                "tile": tile, "hp": hp, "flying": flying, "strong_side": strong_side,
                "is_effect": is_effect(base),
            })

        # Cross-class and me/opp same-box deduplication (prevents triple giant/royal_giant/opp_giant).
        parsed.sort(key=lambda item: (
            not item["is_effect"],
            item["strong_side"] == item["side"],
            not is_generic(item["base"]),
            item["conf"],
        ), reverse=True)
        kept: list[dict] = []
        for item in parsed:
            if item["is_effect"]:
                kept.append(item)
                continue
            dup = False
            for idx, prev in enumerate(kept):
                if prev["is_effect"]:
                    continue
                same_spot = (
                    _box_iou(item["box"], prev["box"]) >= 0.55
                    or (math.dist(item["tile"], prev["tile"]) <= 0.65 and item["flying"] == prev["flying"])
                )
                if not same_spot:
                    continue
                if item["side"] == prev["side"]:
                    dup = True
                    break
                # Opposite-side duplicate on the exact same physical unit: prefer 'me' on our half
                # unless 'opp' has a strong red HP bar or a pre-existing opp track.
                has_opp = any(math.dist(ot, item["tile"]) <= 2.5 for ot, _ in est_opp)
                if (prev["side"] == "me" and item["side"] == "opp"
                        and item["strong_side"] != "opp" and (item["tile"][1] < 11.5 or not has_opp)):
                    dup = True
                    break
                if (prev["side"] == "opp" and item["side"] == "me"
                        and prev["strong_side"] != "opp" and item["tile"][1] < 11.5 and not has_opp):
                    kept[idx] = item
                    dup = True
                    break
            if not dup:
                kept.append(item)

        dets = []
        vis_dets: list[dict] = []
        for item in kept:
            base, side, conf = item["base"], item["side"], item["conf"]
            bx1, by1, bx2, by2 = item["box"]
            tile, hp = item["tile"], item["hp"]
            key = f"{base}|{side}" if side else base  # never associate my knight with theirs
            dets.append(TileDetection(base, conf, tile, side, hp, key=key))
            vis_dets.append({"cls": base, "side": side, "conf": round(conf, 3),
                             "box": (round(bx1, 1), round(by1, 1), round(bx2, 1), round(by2, 1)),
                             "tile": (round(tile[0], 2), round(tile[1], 2)), "hp": hp})
        self._pre_reported_opp = [
            tr.tile for tr in self.tracker.tracks.values() if tr.spawn_reported and tr.side == "opp"
        ]
        tracks, spawns = self.tracker.update(dets, t_ms)
        for j, vd in enumerate(vis_dets):
            tid = self.tracker.last_match.get(j)
            tr = self.tracker.tracks.get(tid) if tid is not None else None
            if tr is not None and tr.hits >= self.tracker.min_hits:
                vd["cls"] = tr.cls
                vd["conf"] = round(float(tr.conf), 3)
                vd["track_id"] = tr.id
                vd["coasted"] = False
                tr.last_box = vd["box"]
        matched_tids = set(self.tracker.last_match.values())
        for t in tracks:
            if t.id in matched_tids:
                continue
            base_tr = self.tracker.tracks.get(t.id)
            if base_tr is not None and base_tr.last_box is not None:
                p_now = self.tiles.tile_to_px(t.tile, img.shape)
                p_old = self.tiles.tile_to_px(base_tr.tile, img.shape)
                dx, dy = p_now[0] - p_old[0], p_now[1] - p_old[1]
                bx1, by1, bx2, by2 = base_tr.last_box
                vis_dets.append({
                    "cls": t.cls, "side": t.side, "conf": round(float(t.conf), 3),
                    "box": (round(bx1 + dx, 1), round(by1 + dy, 1), round(bx2 + dx, 1), round(by2 + dy, 1)),
                    "tile": (round(t.tile[0], 2), round(t.tile[1], 2)), "hp": t.hp_pct,
                    "track_id": t.id, "coasted": True,
                })
        self.last_detections = vis_dets
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
        self._hp_fails[name] = 0
        self._tower_bar_absent[name] = 0
        self._hp_rises.pop(name, None)
        self._hp_drops.pop(name, None)
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
            if tr.conf < self.min_unit_conf and not (tr.hits >= 5 and tr.peak_conf >= 0.48):
                continue  # suppress low-confidence phantom blips from state.units
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
        _SPAWN_SOURCES = {
            "skeletons": ("tombstone", "witch", "graveyard_effect", "skeleton_barrel", "skeleton_king"),
            "spear_goblins": ("goblin_hut", "goblin_giant"),
            "goblins": ("goblin_drill", "goblin_barrel_effect"),
            "fire_spirit": ("furnace",),
            "barbarians": ("barbarian_hut", "battle_ram"),
            "bats": ("night_witch",),
        }
        sources = _SPAWN_SOURCES.get(play.card)
        if not sources:
            return False
        return any(
            tr.side == "opp" and tr.cls in sources and math.dist(play.tile, tr.tile) <= 3.0
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
        elif play.side == "me" and self._my_deck_cards:
            if play.card not in self._my_deck_cards:
                return  # never log impossible non-deck me_spawn_play hallucinations
            prev_me = self._me_last_spawn_ms.get(play.card)
            if prev_me is not None and (play.t_ms - prev_me) < 2500:
                return
            self._me_last_spawn_ms[play.card] = play.t_ms
        rec = {"kind": f"{play.side}_spawn_play", "card": play.card, "tile": play.tile, "t_ms": play.t_ms}
        if play.upgraded_from:
            rec["upgraded_from"] = play.upgraded_from
        self.events.append(rec)
        if self.on_event:
            self.on_event(rec)

    def _read_towers(self, img: np.ndarray, warns: list[str]) -> None:
        if self.frame_i > 0 and not self._battle_hud_seen and self._last_t_ms <= 4000:
            return  # intro/matchmaking frames before the battle HUD appears
        if self.tower_states is not None:
            for name in self.tower_states.read(img, self.layout.towers):
                self._mark_destroyed(name)
        if self.digits is None:
            return
        hp_rois = self.layout.tower_hp or self.layout.towers
        self._tower_reads += 1
        searched = False
        hud_active = self._last_hud is not None and self._last_hud >= 0.05
        for name, roi in hp_rois.items():
            if name in self._destroyed:
                continue  # destruction is permanent; don't let a digit misread revive it
            off = self.hp_offsets.get(name)
            use = roi if off is None else Roi(roi.x0 + off[0], roi.y0 + off[1], roi.x1 + off[0], roi.y1 + off[1])
            hp = self.digits.read(use.crop(img))
            prev = self.towers.get(name)
            # Reject impossible King HP spikes above the Princess-inferred King max HP (e.g. 6710 on Lv2).
            if hp is not None and name.endswith("_king"):
                side = name.partition("_")[0]
                p_max = max(self.tower_max_hp.get(f"{side}_left", 0), self.tower_max_hp.get(f"{side}_right", 0))
                if p_max >= 1000 and hp > infer_king_max_hp(p_max) + 200:
                    hp = None
            # Check HP bar presence unconditionally for Princess towers so rubble digits (2/3/74/248)
            # never prevent locking a destroyed tower to 0.
            bar: bool | None = None
            if not name.endswith("_king") and hud_active:
                bar = has_tower_hp_bar(img, roi, TOWER_SIDE[name], off or (0.0, 0.0))
                if bar is True:
                    self._tower_bar_seen.add(name)
                    self._tower_bar_absent[name] = 0
                elif bar is False and name in self._tower_bar_seen:
                    self._tower_bar_absent[name] += 1
                    if self._tower_bar_absent[name] >= 3:
                        self._mark_destroyed(name)
                        continue
                    if hp is not None and hp < 300:
                        hp = None  # rubble noise without an HP bar
            # Reject clipped 2-3 digit fragments on undamaged start towers (e.g. 512/750/890/400)
            if hp is not None and prev is None and self._last_t_ms <= 15000:
                side, _, pos = name.partition("_")
                twin = f"{side}_{'right' if pos == 'left' else 'left'}" if pos in ("left", "right") else None
                twin_max = self.tower_max_hp.get(twin, 0) if twin else 0
                if hp < 1000 or (twin_max >= 1000 and hp < twin_max - 300):
                    hp = None
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
                    min_hp = 1000 if (prev is None and self._last_t_ms <= 15000) else 100
                    v, offset, _ = locate_number(img, roi, self.digits,
                                                 plausible=lambda x, mx=mx, lo=min_hp: lo <= x <= (mx + 5 if mx else 9999))
                    if v is not None:
                        self._hp_search_misses[name] = 0
                        self.hp_offsets[name] = offset
                        off = offset
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
            if bar is not False:
                self._tower_bar_seen.add(name)
                self._tower_bar_absent[name] = 0
            if prev is None and hp < 600:
                if self._last_t_ms <= 5000 and hp < 300:
                    continue  # ignore 1-2 digit intro/crown artifacts on undamaged start towers
                inits = (self._hp_init_low.get(name, []) + [hp])[-2:]
                self._hp_init_low[name] = inits
                if not (len(inits) == 2 and abs(inits[1] - inits[0]) <= 80):
                    continue  # require confirmation before seeding an unread tower with low/clipped HP
                self._hp_init_low.pop(name, None)
            else:
                self._hp_init_low.pop(name, None)
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
            if prev is not None and hp < prev:
                if _is_suspicious_hp_drop(prev, hp):
                    drops = (self._hp_drops.get(name, []) + [hp])[-2:]
                    self._hp_drops[name] = drops
                    if not (len(drops) == 2 and abs(drops[1] - drops[0]) <= 80):
                        continue
                    self._hp_drops.pop(name, None)
                else:
                    self._hp_drops.pop(name, None)
            else:
                self._hp_drops.pop(name, None)
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
                just_entered_ot = False
                if not self._clock_in_ot and (
                    (self._clock_last_raw_s is not None and self._clock_last_raw_s <= 35.0
                     and raw_s >= 90.0 and t_s >= 150.0)
                    or (t_s >= 182.0 and raw_s <= 125.0)
                ):
                    self._clock_in_ot = True
                    just_entered_ot = True
                cand = (300.0 - raw_s) if self._clock_in_ot else (180.0 - raw_s)
                if (self._clock_match_t_s is not None and not just_entered_ot
                        and abs(cand - self._clock_match_t_s) > 10.0):
                    pass  # reject single-frame clock OCR glitch (e.g. 1:50 -> 0:15)
                else:
                    self._clock_match_t_s = cand
                    self._clock_last_raw_s = raw_s
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
