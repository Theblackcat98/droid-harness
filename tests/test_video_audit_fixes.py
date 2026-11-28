"""Regression tests for all issues from the 31-match ladder-vision-v4 video + log audit."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from droid_harness.core.transport.frame_slot import Frame
from droid_harness.domains.clash_royale.arena import Arena
from droid_harness.domains.clash_royale.cards import Card, CardDB
from droid_harness.domains.clash_royale.menus import MenuNavigator
from droid_harness.domains.clash_royale.perception.cards import CardMatcher
from droid_harness.domains.clash_royale.perception.detector import Detection
from droid_harness.domains.clash_royale.perception.grid import Roi
from droid_harness.domains.clash_royale.perception.pipeline import CRPerception, _is_suspicious_hp_drop
from droid_harness.domains.clash_royale.perception.screens import ScreenClassifier
from droid_harness.domains.clash_royale.policy.strategy import load_strategy
from droid_harness.domains.clash_royale.policy.utility import CRPolicy
from droid_harness.domains.clash_royale.session import CRSession, MatchMeta
from droid_harness.domains.clash_royale.state import GameState, Unit
from droid_harness.sim.cr_scene import CRScene, build_matcher_from_scene, sim_layout, sim_tiles

ROOT = Path(__file__).resolve().parents[1]
STARTER_DECK = ["knight", "archers", "minions", "arrows", "fireball", "giant", "musketeer", "mini_pekka"]
TOWERS = {
    "me_left": 1512, "me_right": 1512, "me_king": 2568,
    "opp_left": 1750, "opp_right": 1750, "opp_king": 2904,
}


@pytest.fixture
def starter_cards():
    base = CardDB.load(ROOT / "tests/fixtures/cards_starter.json")
    extra = {
        "goblin_cage": Card("goblin_cage", "Goblin Cage", 4, "building", range_tiles=0.0, hitpoints=616),
        "tombstone": Card("tombstone", "Tombstone", 3, "building", range_tiles=0.0, hitpoints=422),
        "bomber": Card("bomber", "Bomber", 2, "troop", speed_tiles_s=1.0, range_tiles=4.5, hitpoints=272),
        "hog_rider": Card("hog_rider", "Hog Rider", 4, "troop", speed_tiles_s=2.0, range_tiles=0.8,
                          hitpoints=1408, buildings_only=True),
        "skeletons": Card("skeletons", "Skeletons", 1, "troop", units=3, speed_tiles_s=1.5, range_tiles=0.5,
                          hitpoints=67, unit_classes=("skeleton",)),
        "goblins": Card("goblins", "Goblins", 2, "troop", units=4, speed_tiles_s=2.0, range_tiles=0.5,
                        hitpoints=167, unit_classes=("goblin",)),
    }
    return CardDB({**extra, **base.cards})


class _ScriptedDet:
    def __init__(self):
        self.items: list[Detection] = []

    def __call__(self, _img):
        return list(self.items)


class _SeqDigits:
    def __init__(self, values):
        self.values = list(values)

    def read(self, _crop):
        return self.values.pop(0) if self.values else None

    def read_scored(self, _crop):
        return None, 0.0


# --- 1.1 Own-unit team flips & same-box duplicates ---------------------------

def test_own_unit_team_flip_and_triple_duplicate_suppressed(starter_cards):
    arena = Arena()
    scene = CRScene(starter_cards, STARTER_DECK, elixir=8.0, draw_tower_hp=False)
    det = _ScriptedDet()
    perc = CRPerception(
        layout=scene.layout, tiles=scene.tiles, cards=starter_cards, arena=arena,
        matcher=build_matcher_from_scene(scene), detector=det, my_deck=STARTER_DECK,
    )
    img = scene.render()
    shape = img.shape
    x0, y0, _, _ = scene.layout.arena.to_px(shape)

    def box_at(cls, tile, conf=0.85):
        px, py = scene.tiles.tile_to_px(tile, shape)
        return Detection(cls, conf, (px - 10 - x0, py - 20 - y0, px + 10 - x0, py - y0))

    # Bot plays Giant at (4.5, 1.5); YOLO detects it as opp giant + triple boxes
    perc.record_my_action("giant", (4.5, 1.5), 1500)
    det.items = [
        box_at("giant_me", (4.6, 3.6), conf=0.88),
        box_at("royal_giant_me", (4.6, 3.6), conf=0.72),
        box_at("giant_opp", (4.6, 3.6), conf=0.80),
    ]
    perc._detect(img, 2000)
    perc._detect(img, 2100)
    assert len(perc.units) == 1
    assert perc.units[0].side == "me" and perc.units[0].kind == "giant"

    # Solo flipped own archer on backfield (no me box at all) flips back to me
    perc.record_my_action("archers", (13.9, 3.0), 3000)
    det.items = [box_at("archer_opp", (13.95, 3.3), conf=0.85)]
    perc._detect(img, 3200)
    perc._detect(img, 3300)
    assert all(u.side == "me" for u in perc.units)
    assert any(u.kind == "archer" and u.side == "me" for u in perc.units)

    # Phantom generic other_air_opp on empty backfield with no enemies is dropped
    det.items = [box_at("other_air_opp", (14.5, 4.5), conf=1.0)]
    perc._detect(img, 9000)
    perc._detect(img, 9100)
    assert not [u for u in perc.units if u.side == "opp"]


def test_real_preexisting_enemy_not_flipped_when_we_drop_defender(starter_cards):
    arena = Arena()
    scene = CRScene(starter_cards, STARTER_DECK, elixir=8.0, draw_tower_hp=False)
    det = _ScriptedDet()
    perc = CRPerception(
        layout=scene.layout, tiles=scene.tiles, cards=starter_cards, arena=arena,
        matcher=build_matcher_from_scene(scene), detector=det, my_deck=STARTER_DECK,
    )
    img = scene.render()
    shape = img.shape
    x0, y0, _, _ = scene.layout.arena.to_px(shape)

    def box_at(cls, tile, conf=0.88):
        px, py = scene.tiles.tile_to_px(tile, shape)
        return Detection(cls, conf, (px - 10 - x0, py - 20 - y0, px + 10 - x0, py - y0))

    # Enemy knight already established walking on our side before we drop our knight
    det.items = [box_at("knight_opp", (4.0, 9.5))]
    perc._detect(img, 1000)
    perc._detect(img, 1100)
    assert [(u.kind, u.side) for u in perc.units] == [("knight", "opp")]

    perc.record_my_action("knight", (4.5, 9.0), 1500)
    det.items = [box_at("knight_opp", (4.0, 9.2)), box_at("knight_me", (5.5, 9.0))]
    perc._detect(img, 1700)
    perc._detect(img, 1800)
    sides = sorted(u.side for u in perc.units if u.kind == "knight")
    assert sides == ["me", "opp"]


# --- 1.2 Low-confidence suppression & 214-class me deck filtering ------------

def test_low_conf_units_suppressed_and_me_deck_filtered(starter_cards):
    arena = Arena()
    scene = CRScene(starter_cards, STARTER_DECK, elixir=8.0, draw_tower_hp=False)
    det = _ScriptedDet()
    perc = CRPerception(
        layout=scene.layout, tiles=scene.tiles, cards=starter_cards, arena=arena,
        matcher=build_matcher_from_scene(scene), detector=det, my_deck=STARTER_DECK,
    )
    img = scene.render()
    shape = img.shape
    x0, y0, _, _ = scene.layout.arena.to_px(shape)

    def box_at(cls, tile, conf):
        px, py = scene.tiles.tile_to_px(tile, shape)
        return Detection(cls, conf, (px - 10 - x0, py - 20 - y0, px + 10 - x0, py - y0))

    # 1. Very low-confidence detection is excluded from state.units
    det.items = [box_at("hog_rider_opp", (4.3, 18.0), conf=0.25)]
    perc._detect(img, 1000)
    perc._detect(img, 1100)
    assert perc.units == []

    # 2. Non-deck me hallucinations (mother_witch, pekka, lava_pup) remap/drop cleanly
    perc.record_my_action("archers", (13.5, 2.0), 2000)
    det.items = [
        box_at("mother_witch_me", (13.5, 2.5), conf=0.80),
        box_at("musketeer_me", (13.2, 4.0), conf=0.80),  # archers misclassified as musketeer
        box_at("pekka_me", (4.5, 9.0), conf=0.80),
        box_at("elixir_golem_small_me", (9.0, 6.0), conf=0.50),  # low-conf non-deck -> dropped
    ]
    perc._detect(img, 2500)
    perc._detect(img, 2600)
    kinds = sorted(u.kind for u in perc.units if u.side == "me")
    assert "mother_witch" not in kinds
    assert "pekka" not in kinds
    assert "elixir_golem_small" not in kinds
    assert "musketeer" not in kinds  # remapped to archer
    assert "archer" in kinds and "mini_pekka" in kinds
    me_spawns = [e["card"] for e in perc.events if e["kind"] == "me_spawn_play"]
    assert all(c in STARTER_DECK for c in me_spawns)


# --- 1.3 Tower HP OCR leading-digit truncation & rubble lock -----------------

def test_tower_hp_truncation_guards_and_rubble_lock_with_digits(starter_cards):
    for prev, bad in [(1512, 512), (1890, 890), (1358, 38), (1045, 4), (1015, 15),
                      (2347, 47), (961, 101), (650, 13), (2568, 26)]:
        assert _is_suspicious_hp_drop(prev, bad), (prev, bad)
    assert not _is_suspicious_hp_drop(1512, 1350)
    assert not _is_suspicious_hp_drop(1400, 444)

    # Single-frame truncations never poison towers
    lay = sim_layout()
    lay = replace(lay, towers={"opp_left": Roi(0.15, 0.15, 0.27, 0.18)},
                  tower_hp={"opp_left": Roi(0.15, 0.15, 0.27, 0.18)})
    p = CRPerception(
        layout=lay, tiles=sim_tiles(), cards=starter_cards, arena=Arena(),
        matcher=CardMatcher(), digits=_SeqDigits([1358, 38, 1358, 15, 1358, 4]),
    )
    img = np.zeros((1024, 464, 3), np.uint8)
    p._last_t_ms = 20_000
    seen = []
    for _ in range(6):
        p._read_towers(img, [])
        seen.append(p.towers.get("opp_left"))
    assert seen == [1358] * 6

    # Rubble producing garbage digits ("2", "74", "3") with NO HP bar still locks to 0!
    color_alive = np.full((1024, 464, 3), (120, 160, 180), np.uint8)
    # Draw red opp HP bar in opp_left ROI
    x0, y0, x1, y1 = lay.tower_hp["opp_left"].to_px(color_alive.shape)
    color_alive[y0:y1, x0:x1] = (20, 20, 230)
    color_rubble = np.full((1024, 464, 3), (120, 160, 180), np.uint8)  # sand, no red/dark bar

    p2 = CRPerception(
        layout=lay, tiles=sim_tiles(), cards=starter_cards, arena=Arena(),
        matcher=CardMatcher(), digits=_SeqDigits([1750, 2, 74, 3, 248]),
    )
    p2._last_t_ms = 30_000
    p2._last_hud = 5.0
    p2._battle_hud_seen = True
    p2._read_towers(color_alive, [])
    assert p2.towers["opp_left"] == 1750
    for _ in range(3):
        p2._read_towers(color_rubble, [])
    assert p2.towers["opp_left"] == 0
    # Cannot revive even if a bogus read appears later
    p2._read_towers(color_rubble, [])
    assert p2.towers["opp_left"] == 0

    # Impossible King HP spike above Princess-inferred max (e.g. 6710 on Lv2 2568 King) is rejected
    lay_k = replace(lay, towers={"me_king": Roi(0.4, 0.7, 0.6, 0.76)},
                    tower_hp={"me_king": Roi(0.4, 0.7, 0.6, 0.76)})
    pk = CRPerception(
        layout=lay_k, tiles=sim_tiles(), cards=starter_cards, arena=Arena(),
        matcher=CardMatcher(), digits=_SeqDigits([6710, 6710, 6710, 2400]),
    )
    pk.tower_max_hp["me_left"] = 1512
    pk.towers["me_king"] = 2568
    pk.tower_max_hp["me_king"] = 2568
    pk._last_t_ms = 30_000
    for _ in range(4):
        pk._read_towers(img, [])
    assert pk.towers["me_king"] == 2400


# --- 1.4 & 3.1 Hand unreadable debounce & my_play double-log fix -------------

def test_my_play_slot_cooldown_and_flicker_debounce(starter_cards):
    arena = Arena()
    scene = CRScene(starter_cards, STARTER_DECK, elixir=7.0, draw_tower_hp=False)
    plays = []
    perc = CRPerception(
        layout=scene.layout, tiles=scene.tiles, cards=starter_cards, arena=arena,
        matcher=build_matcher_from_scene(scene), on_my_play=lambda c, t: plays.append((c, t)),
    )
    s0 = perc.process(Frame(1, scene.render(), 0, 0, 0))
    orig_card = s0.hand[1]

    # Play card in slot 1 at t=1000ms -> 1 my_play logged
    scene.play_card(1, (9.0, 10.0))
    perc.process(Frame(2, scene.render(), 1_000_000, 0, 0))
    assert len(plays) == 1 and plays[0][0] == orig_card

    # 33ms later during slide-in, slot 1 temporarily matches a different card -> NO second my_play!
    scene.hand[1] = "giant" if scene.hand[1] != "giant" else "knight"
    perc.process(Frame(3, scene.render(), 1_033_000, 0, 0))
    assert len(plays) == 1

    # Later (past cooldown), a 1-frame flicker with NO elixir drop does NOT fire my_play
    scene.step(3.0)
    prev_slot2 = scene.hand[2]
    flicker_card = "giant" if prev_slot2 != "giant" else "fireball"
    scene.hand[2] = flicker_card
    perc.process(Frame(4, scene.render(), 4_000_000, 0, 0))
    scene.hand[2] = prev_slot2
    perc.process(Frame(5, scene.render(), 4_033_000, 0, 0))
    assert len(plays) == 1

    # Transient 2-frame unreadable slot does NOT spam hand_unreadable warning
    scene.step(1.0)
    blank = scene.render()
    x0, y0, x1, y1 = scene.layout.hand[0].to_px(blank.shape)
    blank[y0:y1, x0:x1] = 0
    s_u1 = perc.process(Frame(6, blank, 5_000_000, 0, 0))
    s_u2 = perc.process(Frame(7, blank, 5_033_000, 0, 0))
    assert s_u1.hand_fresh[0] is False and "hand_unreadable" not in s_u1.warnings
    assert "hand_unreadable" not in s_u2.warnings


# --- 2.1-2.5 Policy fixes ----------------------------------------------------

def test_opening_win_condition_counterpush_is_gated(starter_cards):
    """No Giant behind a safe first play before 0:30: two back-plays starve the defense."""
    arena = Arena()
    _, p, _ = load_strategy(ROOT / "configs/strategies/ladder-vision-v11.yaml")
    pol = CRPolicy(starter_cards, arena, p)

    s_open = GameState(
        t_ms=6200, my_elixir=6.15, opp_elixir=(5.0, 5.0),
        hand=["giant", "musketeer", "fireball", "knight"], towers=dict(TOWERS),
        units=[Unit(1, "knight", "me", (4.5, 1.5), vel=(0.0, 1.0), hp_pct=1.0)],
    )
    cands = [c for r in pol.rules for c in r.propose(s_open, p, pol.ctx, None)
             if c.rule_id == "offense.counter_push" and c.card == "giant"]
    assert cands == []

    # Past the opening, or with a banked elixir pool, the same counter-push is allowed again.
    s_banked = GameState(
        t_ms=6200, my_elixir=9.0, opp_elixir=(5.0, 5.0),
        hand=["giant", "musketeer", "fireball", "knight"], towers=dict(TOWERS),
        units=[Unit(1, "knight", "me", (4.5, 1.5), vel=(0.0, 1.0), hp_pct=1.0)],
    )
    cands_late = [c for r in pol.rules for c in r.propose(s_banked, p, pol.ctx, None)
                  if c.rule_id == "offense.counter_push" and c.card == "giant"]
    assert cands_late


def test_policy_opening_starvation_lane_switch_spells_and_placement(starter_cards):
    arena = Arena()
    _, p, _ = load_strategy(ROOT / "configs/strategies/ladder-vision-v4.yaml")
    pol = CRPolicy(starter_cards, arena, p)

    # 2.2 No opening Giant throw at t=1.2s with opp at 7.5 elixir
    s_open = GameState(
        t_ms=1200, my_elixir=7.6, opp_elixir=(7.5, 7.5),
        hand=["giant", "knight", "archers", "arrows"], towers=dict(TOWERS),
    )
    d_open = pol.decide(s_open)
    assert d_open.chosen is None or d_open.chosen.card != "giant"

    # 2.1 Break 3-elixir starvation: with Giant in hand at 3.5e, defer counterpush support & minor far probe
    s_bank_cp = GameState(
        t_ms=60_000, my_elixir=3.5, opp_elixir=(3.0, 3.5),
        hand=["giant", "archers", "knight", "arrows"], towers=dict(TOWERS),
        units=[Unit(1, "knight", "me", (14.5, 13.0), vel=(0.0, 1.0), hp_pct=1.0)],
    )
    cands_cp = [c for r in pol.rules for c in r.propose(s_bank_cp, p, pol.ctx, None)
                if c.rule_id == "offense.counter_push"]
    assert cands_cp == []

    s_bank_def = GameState(
        t_ms=60_000, my_elixir=3.5, opp_elixir=(4.0, 4.5),
        hand=["giant", "archers", "knight", "arrows"], towers=dict(TOWERS),
        units=[Unit(2, "archer", "opp", (14.5, 17.5), vel=(0.0, -1.0), hp_pct=1.0)],
    )
    cands_def = [c for r in pol.rules for c in r.propose(s_bank_def, p, pol.ctx, None)
                 if c.rule_id == "defend.lane_threat"]
    assert cands_def == []

    # 2.3 Lane switch away from fortified/dead lane + prioritize heavy opposite-lane push
    s_fort = GameState(
        t_ms=60_000, my_elixir=8.5, opp_elixir=(1.0, 2.0),
        hand=["giant", "knight", "archers", "arrows"],
        towers={**TOWERS, "opp_right": 1650, "opp_left": 1750},
        units=[
            Unit(10, "tombstone", "opp", (14.5, 22.0), vel=(0.0, 0.0)),
            Unit(11, "musketeer", "opp", (14.5, 24.0), vel=(0.0, 0.0)),
        ],
    )
    wc = [c for r in pol.rules for c in r.propose(s_fort, p, pol.ctx, None)
          if c.rule_id == "offense.win_condition_when_opp_low"][0]
    assert arena.lane_of(wc.tile[0]) == "left"  # switched away from fortified right lane

    # Opposite-lane Giant + Mini P.E.K.K.A outscores a lone right-lane goblin
    s_two_lanes = GameState(
        t_ms=60_000, my_elixir=5.0, opp_elixir=(1.0, 2.0),
        hand=["mini_pekka", "knight", "archers", "minions"], towers=dict(TOWERS),
        units=[
            Unit(20, "giant", "opp", (3.5, 13.0), vel=(0.0, -0.75)),
            Unit(21, "mini_pekka", "opp", (3.8, 13.5), vel=(0.0, -1.5)),
            Unit(22, "goblin", "opp", (14.5, 11.5), vel=(0.0, -2.0)),
        ],
    )
    d_two = pol.decide(s_two_lanes)
    assert d_two.chosen is not None and arena.lane_of(d_two.chosen.tile[0]) == "left"

    # 2.4 Spell gating: dying 7%-HP Hog, single skeleton, lone tombstone/goblin_cage, low-conf phantom
    s_dying_hog = GameState(
        t_ms=72_000, my_elixir=5.0, opp_elixir=(3.0, 4.0),
        hand=["arrows", "fireball", "knight", "archers"], towers=dict(TOWERS),
        units=[Unit(30, "hog_rider", "opp", (4.3, 9.0), vel=(0.0, -2.0), hp_pct=0.07)],
    )
    assert not [c for r in pol.rules for c in r.propose(s_dying_hog, p, pol.ctx, None)
                if c.rule_id == "spell.value"]

    s_one_skel = GameState(
        t_ms=60_000, my_elixir=5.0, opp_elixir=(3.0, 4.0),
        hand=["arrows", "giant", "fireball", "knight"], towers=dict(TOWERS),
        units=[Unit(31, "skeleton", "opp", (3.5, 10.0), vel=(0.0, -1.5))],
    )
    assert not [c for r in pol.rules for c in r.propose(s_one_skel, p, pol.ctx, None)
                if c.card == "arrows"]

    s_bldg = GameState(
        t_ms=60_000, my_elixir=6.0, opp_elixir=(3.0, 4.0),
        hand=["arrows", "fireball", "knight", "archers"], towers=dict(TOWERS),
        units=[Unit(32, "goblin_cage", "opp", (9.0, 20.0), vel=(0.0, 0.0))],
    )
    assert s_bldg.threats(arena, starter_cards) == []
    assert not [c for r in pol.rules for c in r.propose(s_bldg, p, pol.ctx, None)
                if c.rule_id == "spell.value"]

    # Non-killable pair (knight + mini_pekka) must NOT trigger Arrows
    s_nokill = GameState(
        t_ms=60_000, my_elixir=6.0, opp_elixir=(2.0, 3.0),
        hand=["arrows", "knight", "archers", "giant"], towers=dict(TOWERS),
        units=[
            Unit(33, "knight", "opp", (3.5, 16.0), vel=(0.0, -1.0)),
            Unit(34, "mini_pekka", "opp", (3.7, 16.2), vel=(0.0, -1.5)),
        ],
    )
    assert not [c for r in pol.rules for c in r.propose(s_nokill, p, pol.ctx, None)
                if c.rule_id == "spell.value" and c.card == "arrows"]

    # Low-conf emergency_defend phantom ignored
    pol2 = CRPolicy(starter_cards, arena, p)
    pol2.tier0._tower_max["me_right"] = 1512
    s_em_low = GameState(
        t_ms=130_000, my_elixir=5.0, opp_elixir=(3.0, 4.0),
        hand=["arrows", "knight", "archers", "giant"],
        towers={**TOWERS, "me_right": 200},
        units=[Unit(35, "giant", "opp", (14.5, 7.5), vel=(0.0, -0.75), conf=0.11)],
    )
    assert pol2.tier0.emergency_defend(s_em_low, p) is None

    # 2.5 Ranged placement never hides behind alive Princess Tower, and avoid_clump spreads bridge troops
    th_giant = GameState(
        t_ms=60_000, my_elixir=5.0, towers=dict(TOWERS),
        units=[Unit(40, "giant", "opp", (3.5, 8.0), vel=(0.0, -0.75))],
    ).threats(arena, starter_cards)[0]
    arch_tile, _ = pol.placement.defensive(
        starter_cards["archers"], th_giant,
        GameState(t_ms=60_000, my_elixir=5.0, towers=dict(TOWERS)), 60_000,
    )
    assert arch_tile[1] >= 7.0  # never tucked behind tower at y=5.0

    spread = pol.placement.avoid_clump(
        (14.5, 14.0), starter_cards["archers"],
        GameState(t_ms=60_000, units=[Unit(41, "musketeer", "me", (14.5, 14.0))]),
    )
    assert spread != (14.5, 14.0)


# --- 3.2 Chest / Lucky Drop battle-entry guard & no-HUD abort ----------------

def test_chest_screen_blocked_from_battle_entry_and_no_hud_aborts():
    bar = Roi(0.2, 0.962, 0.98, 0.988)
    top = Roi(0.0, 0.0, 1.0, 0.18)
    sc = ScreenClassifier([top], battle_roi=bar)

    # Full-screen purple Lucky Drop / chest glow must return 0.0 battle_evidence
    lucky_drop = np.full((240, 108, 3), (230, 60, 190), np.uint8)
    assert sc.battle_evidence(lucky_drop) == 0.0
    assert sc.classify(lucky_drop)[0] == "UNKNOWN"

    # Chest screen whose top 18% matches IN_BATTLE dhash but has NO elixir bar must NOT start a match
    chest_img = np.full((240, 108, 3), (40, 40, 40), np.uint8)
    sc.register("IN_BATTLE", chest_img)
    assert sc.classify(chest_img)[0] == "IN_BATTLE"  # dhash alone would say IN_BATTLE

    class _DriftingPerception:
        def __init__(self):
            self.elixir = SimpleNamespace(leaked=0.0)
            self.warnings = {}
            self.events = []
            self.opp = SimpleNamespace(cycle=SimpleNamespace(deck_seen=[]))
            self.e = 5.0

        def start_match(self, _frame):
            self.e = 5.0

        def process(self, _frame):
            self.e = min(10.0, self.e + 0.3)  # drifting elixir with zero readable hand cards
            return GameState(t_ms=0, my_elixir=self.e, opp_elixir=(5.0, 5.0),
                             hand=[None] * 4, hand_fresh=[False] * 4)

    class _DummyPolicy:
        def reset_match(self):
            pass

    ended = []
    sess = CRSession(
        screens=sc, menus=MenuNavigator({}), perception=_DriftingPerception(), policy=_DummyPolicy(),
        mapper=None, recorder=None,
        meta_factory=lambda: MatchMeta("", "phone-a", "acct", "sid", "9.x", "2026-11", "ladder"),
        on_match_end=ended.append, screen_every=1, battle_no_progress_s=0.05,
    )
    obs = sess.process(Frame(1, chest_img, 0, 0, 0))
    assert obs.screen == "UNKNOWN" and sess.match is None and ended == []

    # Even if battle is entered, drifting elixir without ever reading hand/clock/towers still aborts!
    class _AlwaysBattleScreens:
        battle_min_frac = 0.12

        def classify(self, _img):
            return "IN_BATTLE", 0

        def battle_evidence(self, _img):
            return 0.5

    sess2 = CRSession(
        screens=_AlwaysBattleScreens(), menus=MenuNavigator({}), perception=_DriftingPerception(),
        policy=_DummyPolicy(), mapper=None, recorder=None,
        meta_factory=lambda: MatchMeta("", "phone-a", "acct", "sid", "9.x", "2026-11", "ladder"),
        on_match_end=ended.append, screen_every=1, battle_no_progress_s=0.05,
    )
    assert sess2.process(Frame(1, chest_img, 0, 0, 0)).screen == "IN_BATTLE"
    time.sleep(0.06)
    obs2 = sess2.process(Frame(2, chest_img, 60_000, 0, 0))
    assert obs2.screen == "UNKNOWN" and len(ended) == 1 and ended[0]["result"] == "aborted:battle_no_progress"
