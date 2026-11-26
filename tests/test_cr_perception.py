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
