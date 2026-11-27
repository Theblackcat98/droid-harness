"""Detector runtime: class names, output layouts, NMS, async inference; pipeline routing."""

from pathlib import Path

import numpy as np
import pytest

from droid_harness.core.transport.frame_slot import Frame
from droid_harness.domains.clash_royale.cards import CardDB
from droid_harness.domains.clash_royale.perception.classes import (
    load_class_names,
    missing_unit_classes,
    split_side,
)
from droid_harness.domains.clash_royale.perception.detector import (
    AsyncDetector,
    Detection,
    OnnxDetector,
    TorchDetector,
    letterbox,
    load_detector,
    names_from_metadata,
    output_mode,
)
from droid_harness.domains.clash_royale.perception.pipeline import CRPerception
from droid_harness.domains.clash_royale.perception.towers import TowerStateReader
from droid_harness.domains.clash_royale.perception.tracker import TileDetection, Tracker
from droid_harness.sim.cr_scene import CRScene, build_matcher_from_scene

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures"
STARTER = ["knight", "archers", "minions", "arrows", "fireball", "giant", "musketeer", "mini_pekka"]


def _onnx(path, out, *, in_hw=(64, 64), names=None, end2end=None):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, *in_hw])
    outp = helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out.shape))
    const = helper.make_node("Constant", [], ["out"], value=numpy_helper.from_array(out.astype(np.float32)))
    m = helper.make_model(helper.make_graph([const], "g", [inp], [outp]),
                          opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    meta = {}
    if names is not None:
        meta["names"] = str(dict(enumerate(names)))
    if end2end is not None:
        meta["end2end"] = str(end2end)
    for k, v in meta.items():
        p = m.metadata_props.add()
        p.key, p.value = k, v
    onnx.save(m, path)
    return path


# --- class names ------------------------------------------------------------------
def test_class_file_strips_comments_and_rejects_duplicates(tmp_path):
    names = load_class_names(ROOT / "models/classes.txt")
    assert names[0] == "knight_me" and not any(n.startswith("#") or " " in n for n in names)
    assert len(names) == 18 and {"other_ground_opp", "other_air_opp"} <= set(names)
    (tmp_path / "c.txt").write_text("# header\na\n\nb  # trailing\na\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_class_names(tmp_path / "c.txt")


def test_split_side_and_missing_classes():
    assert split_side("knight_opp") == ("knight", "opp")
    assert split_side("mini_pekka_me") == ("mini_pekka", "me")
    assert split_side("fireball_effect") == ("fireball_effect", None)
    names = ["knight_me", "knight_opp", "minion_opp", "fireball_effect"]
    assert missing_unit_classes(names, {"knight", "minion", "fireball_effect", "arrows_effect"}) == \
        ["arrows_effect", "minion"]


def test_names_from_ultralytics_metadata():
    assert names_from_metadata({"names": "{0: 'a', 1: 'b', 10: 'k'}"})[-1] == "k"
    assert names_from_metadata({}) is None


def test_detector_refuses_mismatched_class_list(tmp_path):
    e2e = np.zeros((1, 5, 6))
    path = _onnx(tmp_path / "m.onnx", e2e, names=["knight_me", "knight_opp"])
    assert OnnxDetector(path, providers=["CPUExecutionProvider"]).class_names == ["knight_me", "knight_opp"]
    ok = tmp_path / "ok.txt"
    ok.write_text("# comment lines must not become classes\nknight_me\nknight_opp\n")
    assert OnnxDetector(path, ok, providers=["CPUExecutionProvider"]).class_names == ["knight_me", "knight_opp"]
    with pytest.raises(ValueError, match="disagrees"):
        OnnxDetector(path, ["knight_opp", "knight_me"], providers=["CPUExecutionProvider"])
    bare = _onnx(tmp_path / "bare.onnx", e2e)
    with pytest.raises(ValueError, match="no class names"):
        OnnxDetector(bare, None, providers=["CPUExecutionProvider"])


# --- output layouts and NMS ---------------------------------------------------------
def test_output_mode_is_pinned_from_shape():
    assert output_mode([1, 300, 6], 12) == "e2e"
    assert output_mode([1, 16, 5040], 12) == "raw"
    assert output_mode([1, 5040, 16], 12) == "raw_t"
    assert output_mode([1, 16, 6], 12) == "e2e"  # max_det == 4 + nc: the old heuristic sent this to raw
    assert output_mode([1, 16, 6], 12, end2end=False) == "raw"
    assert output_mode(["batch", "n", 6], 12) == "e2e"  # dynamic max_det, 6 columns
    assert output_mode(["batch", "c", "anchors"], 12) is None  # pinned on the first real output


def test_raw_head_with_nothing_above_conf_returns_empty(tmp_path):
    raw = np.zeros((1, 6, 50))  # 2 classes, all scores 0: an empty arena
    det = OnnxDetector(_onnx(tmp_path / "raw.onnx", raw, names=["a", "b"]), providers=["CPUExecutionProvider"])
    assert det.mode == "raw"
    assert det(np.zeros((64, 64, 3), np.uint8)) == []


def test_raw_head_nms_is_per_class(tmp_path):
    raw = np.zeros((1, 6, 3))
    raw[0, :4, 0] = [30, 30, 10, 20]  # class a
    raw[0, :4, 1] = [31, 30, 10, 20]  # class b, overlapping a (a fight)
    raw[0, :4, 2] = [31, 31, 10, 20]  # class a duplicate -> suppressed
    raw[0, 4, 0], raw[0, 5, 1], raw[0, 4, 2] = 0.9, 0.8, 0.6
    path = _onnx(tmp_path / "raw.onnx", raw, names=["a", "b"])
    img = np.zeros((64, 64, 3), np.uint8)
    out = OnnxDetector(path, providers=["CPUExecutionProvider"])(img)
    assert sorted(d.cls for d in out) == ["a", "b"]
    out_agnostic = OnnxDetector(path, agnostic_nms=True, providers=["CPUExecutionProvider"])(img)
    assert [d.cls for d in out_agnostic] == ["a"]


def test_rectangular_static_input_and_letterbox(tmp_path):
    # Arena crops are ~0.6 aspect: a 64x32 (h, w) input wastes no pixels on padding.
    e2e = np.array([[[2, 4, 12, 24, 0.9, 0]]])
    det = OnnxDetector(_onnx(tmp_path / "r.onnx", e2e, in_hw=(64, 32), names=["a"]), imgsz=640,
                       providers=["CPUExecutionProvider"])
    assert det.imgsz == (64, 32)  # the model's static shape wins
    img = np.zeros((128, 64, 3), np.uint8)
    canvas, r, pad = letterbox(img, (64, 32))
    assert canvas.shape == (64, 32, 3) and r == 0.5 and pad == (0, 0)
    (d,) = det(img)
    assert d.box == pytest.approx((4, 8, 24, 48))


def test_async_detector_returns_tagged_results():
    calls = []

    def slow(img):
        calls.append(img.shape)
        return [Detection("knight_opp", 0.9, (0, 0, 4, 4))]

    a = AsyncDetector(slow)
    try:
        assert a.idle and a.poll() is None
        a.submit(np.zeros((8, 8, 3), np.uint8), ("tag", 7))
        assert a.wait_idle(2.0)
        tag, dets, ms = a.poll()
        assert tag == ("tag", 7) and dets[0].cls == "knight_opp" and ms >= 0
        assert a.poll() is None  # consumed
    finally:
        a.close()


# --- pipeline routing -----------------------------------------------------------------
@pytest.fixture
def starter():
    return CardDB.load(FIX / "cards_starter.json")


class ScriptedDetector:
    """Returns pre-set detections given in *tile* coordinates (converted to crop pixels)."""

    def __init__(self, perc):
        self.perc, self.next = perc, []

    def __call__(self, crop):
        full_shape = self.perc._shape
        x0, y0, _, _ = self.perc.layout.arena.to_px(full_shape)
        out = []
        for cls, tile, half in self.next:
            px, py = self.perc.tiles.tile_to_px(tile, full_shape)
            out.append(Detection(cls, 0.9, (px - half - x0, py - 2 * half - y0, px + half - x0, py - y0)))
        return out


def _perc(starter, arena, **kw):
    scene = CRScene(starter, STARTER)
    perc = CRPerception(layout=scene.layout, tiles=scene.tiles, cards=starter, arena=arena,
                        matcher=build_matcher_from_scene(scene), **kw)
    perc._shape = scene.render().shape
    return scene, perc


def _feed(perc, det, img, t_ms, *dets):
    det.next = list(dets)
    perc._detect(img, t_ms)


def test_side_aware_class_beats_spawn_prior(starter, arena):
    scene, perc = _perc(starter, arena)
    det = ScriptedDetector(perc)
    perc.detector = det
    img = np.zeros(perc._shape, np.uint8)  # no HP bars anywhere
    # An enemy knight first seen on *my* half (re-acquired after a lost track): the old
    # prior called it "me" and the policy ignored it.
    for t in (0, 100):
        _feed(perc, det, img, t, ("knight_opp", (4.0, 9.0), 6), ("knight_me", (5.0, 9.0), 6))
    sides = sorted((u.kind, u.side) for u in perc.units)
    assert sides == [("knight", "me"), ("knight", "opp")]  # two tracks, never merged


def test_my_spell_effect_is_not_charged_to_the_opponent(starter, arena):
    scene, perc = _perc(starter, arena)
    det = ScriptedDetector(perc)
    perc.detector = det
    img = np.zeros(perc._shape, np.uint8)
    perc._my_play("fireball", 2, 1000)  # hand diff saw my fireball leave
    for t in (1600, 1700):  # it lands on *their* half
        _feed(perc, det, img, t, ("fireball_effect", (14.5, 25.0), 10))
    kinds = [e["kind"] for e in perc.events]
    assert "opp_spawn_play" not in kinds and perc.opp.cycle.deck_seen == []
    assert perc.units == []  # effects make play events, never units

    # Their fireball on my tower, with no recent fireball of mine: theirs.
    for t in (9000, 9100):
        _feed(perc, det, img, t, ("fireball_effect", (3.5, 7.0), 10))
    assert [e["card"] for e in perc.events if e["kind"] == "opp_spawn_play"] == ["fireball"]


def test_tower_classes_go_to_tower_state_not_tracker(starter, arena):
    scene, perc = _perc(starter, arena)
    det = ScriptedDetector(perc)
    perc.detector = det
    img = np.zeros(perc._shape, np.uint8)
    for t in (0, 100):
        _feed(perc, det, img, t, ("tower_rubble", arena.towers["opp_left"], 12),
              ("tower_princess", arena.towers["opp_right"], 12))
    assert perc.units == [] and perc.towers == {"opp_left": 0}
    assert any(e["kind"] == "tower_destroyed" and e["tower"] == "opp_left" for e in perc.events)


def test_units_are_extrapolated_between_detections(starter, arena):
    scene, perc = _perc(starter, arena, detect_every=3)
    det = ScriptedDetector(perc)
    perc.detector = det
    img = np.zeros(perc._shape, np.uint8)
    _feed(perc, det, img, 0, ("giant_opp", (14.5, 20.0), 8))
    _feed(perc, det, img, 400, ("giant_opp", (14.5, 19.6), 8))  # -1 tile/s
    (u,) = perc._extrapolate(700)
    assert u.tile[1] == pytest.approx(19.3, abs=0.1)
    (u,) = perc._extrapolate(5000)  # capped: never fly off on a lost track
    assert u.tile[1] == pytest.approx(19.6 - 0.5 * 1.0, abs=0.15)


def test_async_detection_in_process_loop(starter, arena):
    scene, perc = _perc(starter, arena, async_detect=False)
    det = ScriptedDetector(perc)
    det.next = [("musketeer_opp", (9.0, 22.0), 6)]
    scene2, perc2 = _perc(starter, arena, async_detect=True)
    perc2.detector = det
    perc2.async_detector = AsyncDetector(det)
    det.perc = perc2
    try:
        img = scene2.render()
        for i in range(12):
            perc2.process(Frame(i + 1, img, i * 50_000, 0, 0))
            perc2.async_detector.wait_idle(1.0)
        assert [(u.kind, u.side) for u in perc2.units] == [("musketeer", "opp")]
    finally:
        perc2.async_detector.close()


def test_tracker_river_comes_from_calibration():
    tr = Tracker(river_mid_y=15.75, min_hits=1)
    tracks, _ = tr.update([TileDetection("knight", 0.9, (4.0, 15.9))], 0)
    assert tracks[0].side == "opp"  # 15.9 is past a 15.75 river (hardcoded 16.0 said "me")


def test_tower_state_reader_confirms_rubble(starter, arena):
    rng = np.random.default_rng(0)
    alive = rng.integers(0, 255, (20, 40), dtype=np.uint8)
    rubble = rng.integers(0, 255, (20, 40), dtype=np.uint8)
    reader = TowerStateReader({"opp_left": {"alive": [alive], "rubble": [rubble]}}, confirm=2)
    scene, perc = _perc(starter, arena, tower_states=reader)
    roi = perc.layout.towers["opp_left"]
    img = np.zeros(perc._shape, np.uint8)
    x0, y0, x1, y1 = roi.to_px(img.shape)
    import cv2
    img[y0:y1, x0:x1] = cv2.cvtColor(cv2.resize(rubble, (x1 - x0, y1 - y0)), cv2.COLOR_GRAY2BGR)
    assert reader.read(img, {"opp_left": roi}) == set()  # one read isn't enough
    assert reader.read(img, {"opp_left": roi}) == {"opp_left"}
    reader.reset()
    perc._read_towers(img, [])
    perc._read_towers(img, [])
    assert perc.towers == {"opp_left": 0}


def test_torch_detector_and_load_detector(tmp_path, monkeypatch):
    import droid_harness.domains.clash_royale.perception.detector as det_mod

    class _Boxes:
        xyxy = np.array([[10.0, 20.0, 30.0, 50.0]], dtype=np.float32)
        conf = np.array([0.85], dtype=np.float32)
        cls = np.array([1], dtype=np.float32)

        def __len__(self):
            return 1

    class _Res:
        boxes = _Boxes()

    class _FakeYolo:
        def __init__(self, names):
            self.names = names
            self.calls = []

        def predict(self, img, **kw):
            self.calls.append((img.shape, kw))
            return [_Res()]

    fake = _FakeYolo({0: "knight_me", 1: "knight_opp"})
    monkeypatch.setattr(det_mod, "_load_yolo", lambda _p: fake)

    pt = tmp_path / "m.pt"
    det = load_detector(pt, ["knight_me", "knight_opp"], imgsz=(640, 384), device="cpu")
    assert isinstance(det, TorchDetector)
    assert det.provider == "PyTorch(cpu)" and det.half is False
    out = det(np.zeros((100, 60, 3), np.uint8))
    assert out == [Detection("knight_opp", pytest.approx(0.85), (10.0, 20.0, 30.0, 50.0))]
    det.warmup(1)
    assert len(fake.calls) == 2

    with pytest.raises(ValueError, match="disagrees"):
        load_detector(pt, ["knight_opp", "knight_me"], device="cpu")

    # .onnx dispatches to OnnxDetector
    e2e = np.array([[[2, 4, 12, 24, 0.9, 0]]])
    onnx_path = _onnx(tmp_path / "d.onnx", e2e, names=["knight_me"])
    assert isinstance(load_detector(onnx_path, providers=["CPUExecutionProvider"]), OnnxDetector)
