import numpy as np
import pytest

from droid_harness.core.transport.frame_slot import Frame
from droid_harness.domains.clash_royale.perception.cards import CardMatcher
from droid_harness.domains.clash_royale.perception.detector import OnnxDetector, letterbox
from droid_harness.domains.clash_royale.perception.digits import DigitReader
from droid_harness.domains.clash_royale.perception.hud import is_greyed, read_elixir
from droid_harness.domains.clash_royale.perception.ownership import ownership_and_hp
from droid_harness.domains.clash_royale.perception.pipeline import CRPerception
from droid_harness.domains.clash_royale.perception.screens import ScreenClassifier
from droid_harness.domains.clash_royale.perception.tracker import SpawnGrouper, TileDetection, Tracker
from droid_harness.sim.cr_scene import CRScene, SceneDetector, build_matcher_from_scene, card_icon

DECK = ["hog_rider", "musketeer", "ice_golem", "cannon", "fireball", "the_log", "ice_spirit", "skeletons"]


@pytest.fixture
def scene(cards):
    return CRScene(cards, DECK)


def frame(img, seq=1, pts_us=0):
    return Frame(seq, img, pts_us, 0, 0)


def test_tile_mapper_roundtrip(scene):
    for tile in [(3.5, 6.5), (14.5, 25.5), (9, 16)]:
        px = scene.tiles.tile_to_px(tile, (640, 360))
        back = scene.tiles.px_to_tile(px, (640, 360))
        assert back == pytest.approx(tile, abs=0.1)


def test_elixir_bar_reading(scene):
    for v in (0.0, 3.0, 7.5, 10.0):
        scene.elixir = v
        img = scene.render()
        lay = scene.layout
        assert read_elixir(img, lay.elixir_bar, lay.elixir_hsv_lo, lay.elixir_hsv_hi) == pytest.approx(v, abs=0.15)


def test_card_matcher_handles_greyed_cards(scene):
    m = CardMatcher()
    for k in DECK:  # templates only from the *affordable* state
        m.add(k, card_icon(k, (60, 77)))
    for k in DECK:
        r = m.match(card_icon(k, (61, 76), greyed=True))
        assert r.card == k, (k, r)
    assert is_greyed(card_icon("hog_rider", (60, 77), greyed=True))
    assert not is_greyed(card_icon("hog_rider", (60, 77)))


def test_digit_reader_on_rendered_tower_hp(scene):
    reader = DigitReader.from_font(scale=0.5, thickness=1)
    img = scene.render()
    for name, roi in scene.layout.towers.items():
        assert reader.read(roi.crop(img)) == scene.towers[name], name


def test_ownership_from_hp_bar(scene):
    scene.spawn_enemy("knight", (4, 20), hp=0.5)
    scene.units.append(type(scene.units[0])("knight", "me", (14, 10), hp=1.0))
    img = scene.render()
    (side_e, hp_e), (side_m, hp_m) = [ownership_and_hp(img, scene.unit_box(u)) for u in scene.units]
    assert side_e == "opp" and hp_e == pytest.approx(0.5, abs=0.12)
    assert side_m == "me" and hp_m == pytest.approx(1.0, abs=0.06)


def test_tracker_velocity_and_spawn_events():
    tr = Tracker(min_hits=2)
    events = []
    for k in range(6):
        tracks, ev = tr.update([TileDetection("hog_rider", 0.9, (3.5, 24 - 0.2 * k), side="opp")], k * 100)
        events += ev
    assert len(events) == 1 and events[0].side == "opp"
    assert tracks[0].vel[1] == pytest.approx(-2.0, abs=0.05)


def test_spawn_grouper_merges_multi_unit_cards(cards):
    g = SpawnGrouper(cards.card_for_class)
    from droid_harness.domains.clash_royale.perception.tracker import SpawnEvent
    plays = []
    for i in range(3):
        plays += g.add(SpawnEvent(i, "skeleton", "opp", (5 + 0.3 * i, 20), 1000 + 50 * i))
    assert len(plays) == 1 and plays[0].card == "skeletons"
    plays += g.add(SpawnEvent(9, "skeleton", "opp", (5, 20), 1000 + 2000))
    assert len(plays) == 2
    # Generic catch-all units group together without polluting deck_seen
    assert cards.unit_value("other_ground") == 2.0 and cards.unit_value("other_air") == 2.0
    g_plays = []
    for i in range(3):
        g_plays += g.add(SpawnEvent(20 + i, "other_ground", "opp", (4 + 0.2 * i, 22), 5000 + 50 * i))
    assert len(g_plays) == 1 and g_plays[0].card == "other_ground"


def test_pipeline_generic_spawn_updates_opp_elixir_without_deck_pollution(scene, cards, arena):
    from droid_harness.domains.clash_royale.perception.detector import Detection

    class GenericDet:
        def __init__(self):
            self.n = 0

        def __call__(self, img):
            self.n += 1
            h, w = img.shape[:2]
            return [Detection("other_ground_opp", 0.9, (w * 0.2, h * 0.3, w * 0.25, h * 0.4))]

    perc = CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards, arena=arena,
                        matcher=build_matcher_from_scene(scene), detector=GenericDet())
    s = perc.process(frame(scene.render(), 1, 0))
    s = perc.process(frame(scene.render(), 2, 100_000))
    assert s.opp_deck_seen == []  # never adds fake "other_ground" card to cycle tracker
    assert s.opp_elixir[0] < s.opp_elixir[1] and s.opp_elixir[1] < 4.0  # deducted (2.0, 5.0) interval


def test_screen_classifier(scene):
    from droid_harness.domains.clash_royale.perception.grid import Roi
    sc = ScreenClassifier([Roi(0, 0, 1, 0.5), Roi(0, 0.5, 1, 1)])
    battle = scene.render()
    menu = np.full_like(battle, 200)
    menu[100:200, 50:300] = 30
    sc.register("IN_BATTLE", battle)
    sc.register("HOME", menu)
    assert sc.classify(battle)[0] == "IN_BATTLE" and sc.classify(menu)[0] == "HOME"
    assert sc.classify(np.random.default_rng(0).integers(0, 255, battle.shape, np.uint8))[0] == "UNKNOWN"


def test_pipeline_reads_state_and_detects_plays(scene, cards, arena):
    plays = []
    perc = CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards, arena=arena,
                        matcher=build_matcher_from_scene(scene), digits=DigitReader.from_font(scale=0.5, thickness=1),
                        detector=SceneDetector(scene), on_my_play=lambda c, t: plays.append(c))
    s = perc.process(frame(scene.render(), 1, 0))
    assert s.hand == scene.hand and s.my_elixir == pytest.approx(5.0, abs=0.2)
    assert s.towers["opp_left"] == 2534
    scene.spawn_enemy("hog_rider", (3.5, 22.0), vel=(0, -2.0))
    for i in range(2, 8):
        scene.step(0.1)
        s = perc.process(frame(scene.render(), i, int((i - 1) * 100_000)))
    assert [u.kind for u in s.enemy_units()] == ["hog_rider"]
    assert s.enemy_units()[0].vel[1] == pytest.approx(-2.0, abs=0.3)
    assert s.opp_deck_seen == ["hog_rider"] and s.opp_elixir[1] < 5.0  # spent 4, regenerated a bit
    played = scene.hand[1]
    scene.play_card(1, (9, 10))
    s = perc.process(frame(scene.render(), 9, 800_000))
    assert plays == [played] and s.hand == scene.hand
    assert not s.warnings


def _const_onnx(path, out):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])
    outp = helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out.shape))
    const = helper.make_node("Constant", [], ["out"], value=numpy_helper.from_array(out.astype(np.float32)))
    g = helper.make_graph([const], "g", [inp], [outp])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, path)


def test_onnx_detector_decodes_end2end_and_raw_heads(tmp_path):
    # end-to-end head: (1, N, 6) in letterboxed 64x64 coordinates
    img = np.zeros((128, 64, 3), np.uint8)  # letterbox: scale 0.5, pad left 16
    e2e = np.array([[[16 + 5, 10, 16 + 15, 30, 0.9, 1], [0, 0, 1, 1, 0.1, 0]]])
    _const_onnx(tmp_path / "e2e.onnx", e2e)
    det = OnnxDetector(tmp_path / "e2e.onnx", ["a", "b"], imgsz=64, providers=["CPUExecutionProvider"])
    out = det(img)
    assert len(out) == 1 and out[0].cls == "b" and out[0].box == pytest.approx((10, 20, 30, 60))
    # raw head: (1, 4 + nc, anchors), xywh centers, duplicate boxes -> NMS keeps one
    raw = np.zeros((1, 6, 3))
    raw[0, :4, 0] = [26, 20, 10, 20]
    raw[0, :4, 1] = [26.5, 20, 10, 20]
    raw[0, 4, 0], raw[0, 4, 1], raw[0, 5, 2] = 0.8, 0.7, 0.1
    _const_onnx(tmp_path / "raw.onnx", raw)
    det2 = OnnxDetector(tmp_path / "raw.onnx", ["a", "b"], imgsz=64, providers=["CPUExecutionProvider"])
    out2 = det2(img)
    assert len(out2) == 1 and out2[0].cls == "a" and out2[0].conf == pytest.approx(0.8)
    canvas, r, pad = letterbox(img, 64)
    assert canvas.shape == (64, 64, 3) and r == 0.5 and pad == (16, 0)


def test_phone_a_result_reader():
    """The real result banners/crowns must beat the dhash ambiguity."""
    from pathlib import Path

    import cv2

    from droid_harness.domains.clash_royale.perception.screens import read_result, read_result_detail

    root = Path(__file__).resolve().parents[1] / "configs/domains/clash_royale/screens/phone-a"
    needed = ("RESULT_WIN__0.png", "RESULT_LOSS__0.png", "RESULT_LOSS__1.png", "IN_BATTLE__0.png")
    if not all((root / n).exists() for n in needed):
        pytest.skip("phone-a reference screens are local calibration (gitignored)")
    assert read_result(cv2.imread(str(root / "RESULT_WIN__0.png"))) == "win"
    assert read_result(cv2.imread(str(root / "RESULT_LOSS__0.png"))) == "loss"
    assert read_result(cv2.imread(str(root / "RESULT_LOSS__1.png"))) == "loss"
    assert read_result(cv2.imread(str(root / "IN_BATTLE__0.png"))) is None
    # Detail returns crowns from the correct sides: bottom is yours, top is theirs.
    outcome, opp, my = read_result_detail(cv2.imread(str(root / "RESULT_WIN__0.png")))
    assert outcome == "win" and my > opp
    outcome, opp, my = read_result_detail(cv2.imread(str(root / "RESULT_LOSS__0.png")))
    assert outcome == "loss" and opp > my
    assert read_result_detail(cv2.imread(str(root / "IN_BATTLE__0.png")))[0] is None


def test_empty_slot_early_spend_and_no_double_subtract(scene, cards, arena):
    """When a card is played on a real client, the slot goes empty (~1.5s) while HUD elixir
    drops by cost, then the new card slides in. Perception must deduct once immediately and
    never double-subtract when the refill appears (or if HUD already snapped)."""
    plays = []
    perc = CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards, arena=arena,
                        matcher=build_matcher_from_scene(scene), digits=DigitReader.from_font(scale=0.5, thickness=1),
                        detector=SceneDetector(scene), on_my_play=lambda c, t: plays.append((c, t)))
    scene.elixir = 6.0
    s0 = perc.process(frame(scene.render(), 1, 0))
    played = s0.hand[1]
    cost = cards.cost(played)

    # Phase 1: slot 1 goes empty/unreadable (solid dark fill) while HUD drops by cost.
    scene.play_card(1, (9, 10))  # drops scene.elixir by cost AND replaces scene.hand[1]
    empty_img = scene.render()
    x0, y0, x1, y1 = scene.layout.hand[1].to_px(empty_img.shape)
    empty_img[y0:y1, x0:x1] = (20, 20, 20)
    s1 = perc.process(frame(empty_img, 2, 200_000))
    assert plays == [(played, 200)]
    assert s1.hand_fresh[1] is False
    assert s1.my_elixir == pytest.approx(6.0 - cost + 200 / 2800, abs=0.25)
    assert "hand_unreadable" not in s1.warnings
    assert "elixir_discrepancy" not in s1.warnings

    # Stay empty for >1s, then refill: must NOT double-subtract when the new card slides in.
    s2 = perc.process(frame(empty_img, 3, 1_500_000))
    assert "hand_unreadable" not in s2.warnings
    refill_img = scene.render()
    s3 = perc.process(frame(refill_img, 4, 1_600_000))
    assert len(plays) == 1
    assert s3.hand_fresh[1] is True and s3.hand[1] == scene.hand[1]
    assert s3.my_elixir == pytest.approx(scene.elixir + 1600 / 2800 - 0.0, abs=0.35)

    # Case 2: if ElixirModel already snapped to HUD before refill is seen, _my_play does not
    # subtract a second time when old != cur is finally observed.
    plays2 = []
    perc2 = CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards, arena=arena,
                         matcher=build_matcher_from_scene(scene), on_my_play=lambda c, t: plays2.append(c))
    perc2.process(frame(scene.render(), 1, 0))
    old_card = perc2.hand[0]
    old_cost = cards.cost(old_card)
    before = perc2.elixir.value
    perc2.elixir.value = max(0.0, before - old_cost)  # simulate prior snap to HUD
    perc2._my_play(old_card, 0, 500, hud=perc2.elixir.value)
    assert perc2.elixir.value == pytest.approx(max(0.0, before - old_cost), abs=1e-6)
    assert plays2 == [old_card]


def test_intro_zero_elixir_and_tower_skin_suppression(scene, cards, arena):
    """Regressions from live ladder losses (128b426f043d / f4662249910c):
    1. Intro frame_i == 0 with hud == 0.0 must NOT zero opp_elixir; syncing to 7.0 at t=0.5s
       must not trigger elixir_discrepancy.
    2. Low-conf generic other_ground_opp with no HP bar on an alive tower footprint is ignored,
       while high-conf or specific-class units near a tower are kept.
    """
    from droid_harness.domains.clash_royale.perception.detector import Detection

    class TowerHallucDet:
        def __call__(self, img):
            # Project opponent left princess tower (3.5, 25.5) and a knight near it to crop px
            x0, y0, _, _ = scene.layout.arena.to_px(img.shape)
            tx, ty = scene.tiles.tile_to_px((4.2, 24.5), img.shape)
            kx, ky = scene.tiles.tile_to_px((5.0, 24.5), img.shape)
            return [
                # Low-conf generic tower skin decoration (no HP bar -> suppressed)
                Detection("other_ground_opp", 0.42, (tx - x0 - 10, ty - y0 - 15, tx - x0 + 10, ty - y0 + 15)),
                # High-conf real generic unit near tower (kept)
                Detection("other_ground_opp", 0.75, (tx - x0 - 10, ty - y0 - 15, tx - x0 + 10, ty - y0 + 15)),
                # Specific class near tower (kept even at moderate conf)
                Detection("knight_opp", 0.48, (kx - x0 - 10, ky - y0 - 15, kx - x0 + 10, ky - y0 + 15)),
            ]

    scene.elixir = 0.0
    perc0 = CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards, arena=arena,
                         matcher=build_matcher_from_scene(scene))
    s0 = perc0.process(frame(scene.render(), 1, 0))
    assert s0.opp_elixir[0] >= 5.0
    scene.elixir = 7.0
    s1 = perc0.process(frame(scene.render(), 2, 500_000))
    assert s1.opp_elixir[0] >= 7.0 and s1.my_elixir == pytest.approx(7.0, abs=0.2)
    assert "elixir_discrepancy" not in s1.warnings

    perc = CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards, arena=arena,
                        matcher=build_matcher_from_scene(scene), detector=TowerHallucDet())
    perc.process(frame(scene.render(), 1, 0))
    s2 = perc.process(frame(scene.render(), 2, 100_000))
    # Low-conf tower hallucination suppressed; 1 high-conf other_ground + 1 knight kept -> 2 units
    assert sorted(u.kind for u in s2.units) == ["knight", "other_ground"]


def test_card_db_canonical_by_class_and_goblin_brawler(cards):
    from droid_harness.domains.clash_royale.cards import Card, CardDB

    m = dict(cards.cards)
    # Add swarm/gang colliding variants deliberately before cheap variants in insertion order
    ext = CardDB({
        "minion_horde": Card("minion_horde", "Minion Horde", 5, "troop", units=6, flying=True, targets_air=True,
                             unit_classes=("minion",)),
        "skeleton_army": Card("skeleton_army", "Skeleton Army", 3, "troop", units=15, unit_classes=("skeleton",)),
        "goblin_gang": Card("goblin_gang", "Goblin Gang", 3, "troop", units=6,
                            unit_classes=("goblin", "spear_goblin")),
        "goblins": Card("goblins", "Goblins", 2, "troop", units=4, unit_classes=("goblin",)),
        "spear_goblins": Card("spear_goblins", "Spear Goblins", 2, "troop", units=3, targets_air=True,
                              unit_classes=("spear_goblin",)),
        **m,
    })
    assert ext.card_for_class("minion").key == "minions"
    assert ext.card_for_class("skeleton").key == "skeletons"
    assert ext.card_for_class("goblin").key == "goblins"
    assert ext.card_for_class("spear_goblin").key == "spear_goblins"
    assert ext.card_for_class("goblin_brawler") is None
    assert ext.unit_value("goblin_brawler") == 2.5


def test_tracker_anti_flicker_and_time_missed():
    tr = Tracker(min_hits=2, max_missed=6, max_missed_ms=500)
    ev_total = []
    # Two initial knight hits establish track
    for t in (0, 17):
        _, ev = tr.update([TileDetection("knight", 0.7, (4.0, 20.0), side="opp")], t)
        ev_total += ev
    assert len(ev_total) == 1
    tid0 = next(iter(tr.tracks))
    # 10 missed frames at 60 Hz (only 170ms < 500ms) -> track stays alive
    for k in range(1, 11):
        _, ev = tr.update([], 17 + k * 17)
        ev_total += ev
    assert tid0 in tr.tracks
    # Flickers to other_ground for one frame, then back to knight within 0.8 tiles -> same ID, 0 new spawns
    _, ev = tr.update([TileDetection("other_ground", 0.5, (4.3, 19.6), side="opp")], 220)
    ev_total += ev
    tracks, ev = tr.update([TileDetection("knight", 0.75, (4.4, 19.4), side="opp")], 240)
    ev_total += ev
    assert len(ev_total) == 1 and len(tracks) == 1 and tracks[0].id == tid0 and tracks[0].cls == "knight"


def test_spawn_grouper_swarm_upgrade_and_death_spawns(cards):
    from droid_harness.domains.clash_royale.perception.tracker import SpawnEvent

    g = SpawnGrouper(cards.card_for_class)
    assert g.add(SpawnEvent(1, "goblin_brawler", "opp", (5.0, 22.0), 1000)) == []
    plays = []
    for i in range(6):
        plays += g.add(SpawnEvent(10 + i, "minion", "opp", (14.0 + 0.2 * i, 22.0), 2000 + 30 * i))
    assert [(p.card, p.upgraded_from) for p in plays] == [("minions", None), ("minion_horde", "minions")]


def test_pipeline_legal_zone_building_spawns_and_cycle_dedup(scene, cards, arena):
    from droid_harness.domains.clash_royale.cards import Card, CardDB
    from droid_harness.domains.clash_royale.perception.detector import Detection

    ext = CardDB({
        **cards.cards,
        "tombstone": Card("tombstone", "Tombstone", 3, "building", units=1, unit_classes=("tombstone",)),
        "minion_horde": Card("minion_horde", "Minion Horde", 5, "troop", units=6, flying=True, targets_air=True,
                             unit_classes=("minion",)),
    })

    class FakeDet:
        def __init__(self):
            self.items = []

        def __call__(self, img):
            return list(self.items)

    det = FakeDet()
    perc = CRPerception(layout=scene.layout, tiles=scene.tiles, cards=ext, arena=arena,
                        matcher=build_matcher_from_scene(scene), detector=det)
    shape = scene.render().shape
    x0, y0, _, _ = scene.layout.arena.to_px(shape)

    def box_at(cls, tile):
        px, py = scene.tiles.tile_to_px(tile, shape)
        return Detection(cls, 0.9, (px - 8 - x0, py - 16 - y0, px + 8 - x0, py - y0))

    img = np.zeros(shape, np.uint8)
    # 1. Illegal enemy knight on OUR side (y=9.0): kept in units for defense, NEVER billed to opp_elixir
    det.items = [box_at("knight_opp", (4.0, 9.0))]
    perc._detect(img, 0)
    perc._detect(img, 100)
    assert [(u.kind, u.side) for u in perc.units] == [("knight", "opp")]
    assert [e for e in perc.events if e["kind"] == "opp_spawn_play"] == []

    # 2. Tombstone on their half (y=24.0) + skeleton free spawn nearby -> ONLY tombstone billed
    det.items = [box_at("tombstone_opp", (9.0, 24.0))]
    perc._detect(img, 5000)
    perc._detect(img, 5100)
    det.items = [box_at("tombstone_opp", (9.0, 24.0)), box_at("skeleton_opp", (9.0, 22.8))]
    perc._detect(img, 6000)
    perc._detect(img, 6100)
    assert [e["card"] for e in perc.events if e["kind"] == "opp_spawn_play"] == ["tombstone"]

    # 3. Same card re-spawning < 5s later (tracker flicker recovery) is deduplicated
    det.items = [box_at("tombstone_opp", (9.0, 24.0)), box_at("musketeer_opp", (14.0, 25.0))]
    perc._detect(img, 8000)
    perc._detect(img, 8100)
    det.items = [box_at("musketeer_opp", (14.0, 24.0))]  # different track jumps > 5 tiles 1.5s later
    perc._detect(img, 9600)
    perc._detect(img, 9700)
    opp_cards = [e["card"] for e in perc.events if e["kind"] == "opp_spawn_play"]
    assert opp_cards.count("musketeer") == 1


def test_cycle_tracker_cards_until_back_and_upgrade(cards):
    from droid_harness.domains.clash_royale.opponent_model import OpponentModel

    om = OpponentModel()
    om.elixir.low = om.elixir.high = 8.0
    om.on_play("minions", 3, 1.0)
    assert om.cycle.cards_until_back("minions") == 4
    om.upgrade_play("minions", "minion_horde", 2, 1.4)
    assert om.cycle.deck_seen == ["minion_horde"]
    assert om.cycle.cards_until_back("minion_horde") == 4
    assert om.elixir.high == pytest.approx(3.5, abs=0.01)
    for c in ("knight", "archers", "arrows"):
        om.on_play(c, 3, 2.0)
    assert om.cycle.cards_until_back("minion_horde") == 1
    om.on_play("fireball", 4, 3.0)
    assert om.cycle.cards_until_back("minion_horde") == 0


def test_spawn_evidence_invariants_tracker_and_pipeline(scene, cards, arena):
    from droid_harness.domains.clash_royale.perception.detector import Detection

    # 1. Low-conf 2-frame blip appears in confirmed units for instant defense, but doesn't fire SpawnEvent
    #    until peak_conf >= 0.62 or hits >= 5 at conf >= 0.48.
    tr = Tracker(min_hits=2)
    tracks, ev = tr.update([TileDetection("knight", 0.42, (4.0, 20.0), side="opp")], 0)
    assert ev == []
    tracks, ev = tr.update([TileDetection("knight", 0.50, (4.0, 19.9), side="opp")], 50)
    assert len(tracks) == 1 and ev == []  # unit tracked for defense without charging opp_elixir
    for k in range(2, 5):
        tracks, ev2 = tr.update([TileDetection("knight", 0.50, (4.0, 19.9 - 0.05 * k), side="opp")], 50 + k * 50)
        ev += ev2
    assert len(ev) == 1 and ev[0].hits == 5

    # 2. Intro zero-HUD suppression + battle-start hud=7.0 sync + physical elixir & 8-card deck gates
    class ScriptedDet:
        def __init__(self):
            self.items = []

        def __call__(self, img):
            return list(self.items)

    det = ScriptedDet()
    perc = CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards, arena=arena,
                        matcher=build_matcher_from_scene(scene), detector=det)
    shape = scene.render().shape
    x0, y0, _, _ = scene.layout.arena.to_px(shape)

    def box_at(cls, tile, conf=0.70):
        px, py = scene.tiles.tile_to_px(tile, shape)
        return Detection(cls, conf, (px - 8 - x0, py - 16 - y0, px + 8 - x0, py - y0))

    # Pre-battle intro (hud == 0.0): fake high-conf giant banner detection must NOT fire opp_spawn_play
    scene.elixir = 0.0
    det.items = [box_at("golem_opp", (14.5, 22.0), conf=0.90)]
    perc.process(frame(scene.render(), 1, 0))
    s_intro = perc.process(frame(scene.render(), 2, 400_000))
    assert not [e for e in perc.events if e["kind"] == "opp_spawn_play"]
    # Battle begins (hud == 7.0 at t=1.5s): opp_elixir syncs upward cleanly to 7.0
    scene.elixir = 7.0
    det.items = []
    s_start = perc.process(frame(scene.render(), 3, 1_500_000))
    assert s_start.opp_elixir[0] >= 7.0 and s_intro.opp_elixir[0] >= 5.0

    # Physical elixir gate: drop opp_elixir to 0.5 and feed moderate-conf (0.68, 2 hits) 8-cost golem -> rejected
    perc.opp.elixir.low = perc.opp.elixir.high = 0.5
    det.items = [box_at("golem_opp", (14.5, 22.0), conf=0.68)]
    perc._detect(np.zeros(shape, np.uint8), 5000)
    perc._detect(np.zeros(shape, np.uint8), 5100)
    assert not [e for e in perc.events if e["card"] == "golem"]

    # 8-card deck lock: fill 8 cards in cycle, then feed moderate-conf 9th card -> rejected
    perc.opp.elixir.low = perc.opp.elixir.high = 9.0
    perc.opp.cycle.deck_seen = ["knight", "musketeer", "arrows", "fireball",
                                "skeletons", "minions", "cannon", "baby_dragon"]
    det.items = [box_at("balloon_opp", (14.5, 22.0), conf=0.68)]
    perc._detect(np.zeros(shape, np.uint8), 10000)
    perc._detect(np.zeros(shape, np.uint8), 10100)
    assert not [e for e in perc.events if e["card"] == "balloon"]


def test_result_detail_ignores_arena_sand_outside_cushions_and_low_sat():
    import cv2

    from droid_harness.domains.clash_royale.perception.screens import read_result_detail

    h, w = 1024, 464
    img = np.zeros((h, w, 3), np.uint8)
    # Top pink opponent banner, bottom blue player banner.
    img[int(0.24 * h):int(0.36 * h), int(0.08 * w):int(0.92 * w)] = (147, 20, 220)
    img[int(0.44 * h):int(0.56 * h), int(0.08 * w):int(0.92 * w)] = (220, 120, 20)
    # Genuine crown on top-left cushion (x=0.27) in gold BGR (0, 215, 255)
    img[int(0.21 * h):int(0.29 * h), int(0.22 * w):int(0.32 * w)] = (0, 215, 255)
    # Bone Pit sunlit golden sand patches: side edges (x=0.05..0.14, x=0.86..0.95) & dull gold inside crown band
    sand_edge_bgr = cv2.cvtColor(np.uint8([[[17, 144, 123]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    sand_dull_bgr = cv2.cvtColor(np.uint8([[[15, 134, 136]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    img[int(0.20 * h):int(0.29 * h), int(0.86 * w):int(0.95 * w)] = sand_edge_bgr
    img[int(0.42 * h):int(0.51 * h), int(0.05 * w):int(0.14 * w)] = sand_edge_bgr
    img[int(0.19 * h):int(0.25 * h), int(0.66 * w):int(0.76 * w)] = sand_dull_bgr
    assert read_result_detail(img) == ("loss", 1, 0)


def test_tracker_vel_clamp_coasting_sticky_class_and_gui_smoothing(scene, cards, arena):
    import math

    from droid_harness.domains.clash_royale.perception.detector import Detection

    tr = Tracker(min_hits=2, max_speed=4.0, coast=2, coast_ms=200, conf_alpha=0.35, class_votes=2)
    # Noisy high-speed jumps must be clamped to max_speed (4.0 tiles/s)
    tr.update([TileDetection("other_ground", 0.40, (4.0, 22.0), side="opp", key="other_ground|opp")], 0)
    tracks, _ = tr.update([TileDetection("musketeer", 0.80, (5.0, 21.8), side="opp", key="musketeer|opp")], 50)
    assert len(tracks) == 1
    assert math.hypot(*tracks[0].vel) <= 4.0 + 1e-6
    # Second musketeer vote upgrades catch-all -> musketeer and updates key, never downgrades back
    tracks, _ = tr.update([TileDetection("musketeer", 0.90, (5.0, 21.6), side="opp", key="musketeer|opp")], 100)
    assert tracks[0].cls == "musketeer" and tracks[0].key == "musketeer|opp"
    # Confidence is EMA-smoothed, not raw 0.90 -> 0.40
    assert tracks[0].conf > 0.55
    tracks, _ = tr.update([TileDetection("other_ground", 0.40, (5.0, 21.5), side="opp", key="other_ground|opp")], 150)
    assert tracks[0].cls == "musketeer" and tracks[0].conf > 0.50

    # Coasting: missed frame keeps track published with last_ms=t_ms (no double-extrapolation)
    coasted, _ = tr.update([], 200)
    assert len(coasted) == 1 and coasted[0].last_ms == 200

    # Pipeline last_detections carries smoothed conf and stabilized cls
    class FlickerDet:
        def __init__(self):
            self.seq = 0

        def __call__(self, img):
            self.seq += 1
            h, w = img.shape[:2]
            cls = "other_ground_opp" if self.seq == 1 else "musketeer_opp"
            conf = 0.40 if self.seq == 1 else 0.90
            return [Detection(cls, conf, (w * 0.5, h * 0.45, w * 0.5 + 20, h * 0.45 + 30))]

    perc = CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards, arena=arena,
                        matcher=build_matcher_from_scene(scene), detector=FlickerDet())
    img = scene.render()
    perc.process(frame(img, 1, 0))
    perc.process(frame(img, 2, 50_000))
    perc.process(frame(img, 3, 100_000))
    assert perc.last_detections[0]["cls"] == "musketeer"
    assert perc.last_detections[0]["conf"] < 0.85  # EMA-smoothed, not raw 0.90


def test_ot_clock_transition_spawner_subspawn_and_coasted_boxes(scene, cards, arena):
    from droid_harness.domains.clash_royale.cards import SUBSPAWN_VALUE
    from droid_harness.domains.clash_royale.perception.detector import Detection
    from droid_harness.domains.clash_royale.perception.grid import Roi
    from droid_harness.domains.clash_royale.perception.tracker import PlayEvent, SpawnEvent
    from droid_harness.domains.clash_royale.state import GameState, Unit
    from droid_harness.ui.snapshot import snapshot_from_record

    # 1. Clock OT transition & single-frame glitch filter
    lay = scene.layout
    lay = type(lay)(
        hand=lay.hand, next_card=lay.next_card, elixir_bar=lay.elixir_bar, clock=lay.clock,
        towers=lay.towers, arena=lay.arena, buttons=lay.buttons, elixir_hsv_lo=lay.elixir_hsv_lo,
        elixir_hsv_hi=lay.elixir_hsv_hi, tower_hp=lay.tower_hp, clock_countdown=True,
    )
    p = CRPerception(layout=lay, tiles=scene.tiles, cards=cards, arena=arena,
                     matcher=build_matcher_from_scene(scene))
    seq = iter([10.0, 2.0, 120.0, 15.0, 118.0])
    p._parse_clock = lambda crop, allow_red=False: next(seq)
    img = scene.render()
    t1, _ = p._read_clock(img, 166_000, [])
    t2, _ = p._read_clock(img, 174_000, [])
    t3, _ = p._read_clock(img, 176_000, [])  # 2:00 OT banner arrives before 180s
    t4, _ = p._read_clock(img, 177_500, [])  # glitch 15.0s rejected
    t5, _ = p._read_clock(img, 179_000, [])
    assert (t1, t2, t3, t4, t5) == (170.0, 178.0, 180.0, 180.0, 182.0)

    # 2. Expanded spawner suppression & 3M swarm upgrade
    class StaticDet:
        def __init__(self):
            self.items = []

        def __call__(self, im):
            return list(self.items)

    det = StaticDet()
    p2 = CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards, arena=arena,
                      matcher=build_matcher_from_scene(scene), detector=det)
    shape = img.shape
    x0, y0, _, _ = scene.layout.arena.to_px(shape)
    px, py = scene.tiles.tile_to_px((9.0, 24.0), shape)
    det.items = [Detection("furnace_opp", 0.9, (px - 8 - x0, py - 16 - y0, px + 8 - x0, py - y0))]
    p2._detect(img, 0)
    p2._detect(img, 100)
    assert p2._is_building_spawn(PlayEvent("opp", "fire_spirit", (9.2, 23.0), 500, 1))
    assert not p2._is_building_spawn(PlayEvent("opp", "knight", (9.2, 23.0), 500, 1))

    g = SpawnGrouper(cards.card_for_class)
    plays = []
    for i in range(3):
        plays += g.add(SpawnEvent(30 + i, "musketeer", "opp", (9.0 + 0.2 * i, 28.0), 1000 + 30 * i))
    assert [(pl.card, pl.upgraded_from) for pl in plays] == [("musketeer", None), ("three_musketeers", "musketeer")]

    # 3. Sub-spawn value, threat speed, coasted boxes, and snapshot conf
    assert cards.unit_value("golemite") == SUBSPAWN_VALUE["golemite"]
    st = GameState(t_ms=1000, units=[Unit(1, "golemite", "opp", (3.5, 20.0), conf=0.77)])
    assert st.threats(arena, cards)[0].eta_to_tower_s < 50.0
    snap = snapshot_from_record({"t_ms": 1000, "state": st.digest()})
    assert "conf" in snap["units"][0]

    # Coasted detection box appears in last_detections when detector drops a frame
    det.items = []
    p2._detect(img, 200)
    assert any(d.get("coasted") is True for d in p2.last_detections)
    _ = Roi  # keep lint clean
