"""M3 detector foundations: class list, synthetic data, arena-crop inference."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from droid_harness.domains.clash_royale.cards import CardDB
from droid_harness.domains.clash_royale.perception.classes import (
    base_classes,
    is_tower,
    load_class_names,
    missing_unit_classes,
    split_side,
)
from droid_harness.domains.clash_royale.perception.detector import Detection
from droid_harness.domains.clash_royale.perception.pipeline import CRPerception
from droid_harness.learn import synth
from droid_harness.learn.train_detector import TrainConfig, write_classes
from droid_harness.sim.cr_scene import CRScene, build_matcher_from_scene


def _cards():
    """The real (local, gitignored) card DB when present, else the starter-deck fixture."""
    root = Path(__file__).resolve().parents[1]
    real = root / "configs/domains/clash_royale/cards.json"
    return CardDB.load(real if real.exists() else root / "tests/fixtures/cards_starter.json")


def _names() -> list[str]:
    return load_class_names(Path(__file__).resolve().parents[1] / "models/classes.txt")


def test_classes_match_card_db():
    names = _names()
    assert len(names) == len(set(names))
    db = _cards()
    needed = {cls for card in db.cards.values() for cls in card.unit_classes}
    missing = missing_unit_classes(names, needed)
    assert not missing, (
        f"cards.json spawns {missing} but models/classes.txt can't produce them. Units need "
        f"<unit>_me and <unit>_opp; Minions should list unit_classes: ['minion'] (three minion units).")
    assert not any(is_tower(split_side(n)[0]) for n in names), "towers are read from fixed ROIs"


def _sprite(h=97, w=56, color=(0, 200, 0)):
    sp = np.zeros((h, w, 4), np.uint8)
    sp[:, :, :3] = color
    sp[:, :, 3] = 255
    return sp


def test_synth_uses_the_inference_geometry():
    import random

    from droid_harness.domains.clash_royale.perception.detector import letterbox
    bg = np.random.default_rng(0).integers(0, 255, (768, 464, 3), dtype=np.uint8)  # an arena crop
    cfg = synth.SynthConfig(imgsz=(640, 384), seed=1)
    empty, boxes = synth.compose(bg, [], cfg, random.Random(1))
    assert boxes == [] and np.array_equal(empty, letterbox(bg, (640, 384))[0])  # same transform as inference

    img, boxes = synth.compose(bg, [synth.Sprite(_sprite(), "knight_opp")] * 6, cfg, random.Random(2))
    _, r, (left, top) = letterbox(bg, (640, 384))
    for b in boxes:
        x1, y1, x2, y2 = b.box
        assert left <= x1 < x2 <= left + round(464 * r) and top <= y1 < y2 <= top + round(768 * r)
    # KataCR sprites come from 568-px arenas; ours is 464 px, then letterboxed by r.
    full = [b for b in boxes if b.box[3] - b.box[1] > 50]
    for b in full:
        assert b.box[3] - b.box[1] == pytest.approx(97 * 464 / 568 * r, rel=0.12)


def test_synth_z_order_occlusion_and_bars():
    import random
    bg = np.full((768, 464, 3), 60, np.uint8)
    cfg = synth.SynthConfig(imgsz=(640, 384), hp_bar_prob=1.0, scale_jitter=0.0)
    back = synth.Sprite(_sprite(color=(0, 200, 0)), "knight_opp")
    front = synth.Sprite(_sprite(color=(200, 200, 0)), "giant_me")
    # Same x, the "front" sprite's foot is lower on screen: it's drawn on top and hides "back".
    img, boxes = synth.compose(bg, [back, front], cfg, random.Random(0), feet=[(190, 300), (190, 302)])
    assert [b.cls for b in boxes] == ["giant_me"]  # fully hidden sprite isn't labeled
    assert ((img == (255, 0, 0)).all(-1)).any()  # blue bar for "me", drawn on top
    fx = synth.Sprite(_sprite(color=(0, 120, 255)), "fireball_effect")
    img2, boxes2 = synth.compose(bg, [fx], cfg, random.Random(0), feet=[(190, 300)])
    assert [b.cls for b in boxes2] == ["fireball_effect"]
    assert not ((img2 == (0, 0, 255)).all(-1)).any() and not ((img2 == (255, 0, 0)).all(-1)).any()
    # Mostly clipped at the arena edge: not labeled.
    _, boxes3 = synth.compose(bg, [back], cfg, random.Random(0), feet=[(190, 25)])
    assert boxes3 == []


def test_synth_dataset_layout_and_side_aware_sprites(tmp_path):
    import yaml
    seg = tmp_path / "segment"
    for d, files in {"knight": ["knight_0_0000001.png", "knight_1_0000002.png"],
                     "fireball_effect": ["fireball_1_0000003.png"], "hog_rider": ["hog-rider_1_0000004.png"]}.items():
        (seg / d).mkdir(parents=True)
        for f in files:
            cv2.imwrite(str(seg / d / f), _sprite(40, 24))
    names = _names()
    sprites, counts = synth.load_sprites(seg, names)
    assert sorted(s.cls for s in sprites) == ["fireball_effect", "knight_me", "knight_opp"]
    assert counts["skipped:hog_rider"] == 1  # not in our class list
    pool = list(sprites)
    bg = np.full((768, 464, 3), 60, np.uint8)
    data = synth.generate_dataset([bg], sprites, tmp_path / "ds", 6, names, synth.SynthConfig(seed=3),
                                  val_frac=0.5)
    assert sprites == pool  # caller's list untouched
    d = yaml.safe_load(data.read_text())
    assert d["names"][0] == "knight_me" and d["train"] == "images/train"
    imgs = list((tmp_path / "ds/images").rglob("*.jpg"))
    labels = list((tmp_path / "ds/labels").rglob("*.txt"))
    assert len(imgs) == len(labels) == 6
    for lb in labels:
        for row in lb.read_text().splitlines():
            cls, *xywh = row.split()
            assert 0 <= int(cls) < len(names) and all(0 <= float(v) <= 1 for v in xywh)
    with pytest.raises(ValueError, match="not in the class list"):
        synth.generate_dataset([bg], [synth.Sprite(_sprite(), "hog_rider_opp")], tmp_path / "x", 1, names)


def test_katacr_real_frames_convert_to_our_classes(tmp_path):
    from droid_harness.learn import katacr
    part2 = tmp_path / "part2"
    ep = part2 / "ep1" / "1"
    ep.mkdir(parents=True)
    (part2 / "ClashRoyale_detection.yaml").write_text(
        "names:\n  0: king-tower\n  1: knight\n  2: fireball\n  3: hog-rider\n")
    rows = ["0 0.5 0.13 0.2 0.16 1 0 0 0 0 0 0",   # tower: not a detector class
            "1 0.3 0.70 0.1 0.10 0 0 0 0 0 0 0",   # my knight
            "1 0.6 0.30 0.1 0.10 1 0 0 0 0 0 0",   # their knight
            "2 0.5 0.80 0.2 0.10 1 0 0 0 0 0 0",   # a fireball effect
            "3 0.4 0.40 0.1 0.10 1 0 0 0 0 0 0"]   # hog rider: not in our set
    (ep / "00000.txt").write_text("\n".join(rows) + "\n")
    cv2.imwrite(str(ep / "00000.jpg"), np.zeros((896, 568, 3), np.uint8))
    (ep / "00015.txt").write_text(rows[0] + "\n")  # nothing of ours: a negative, dropped at keep=0
    cv2.imwrite(str(ep / "00015.jpg"), np.zeros((896, 568, 3), np.uint8))
    names = _names()
    alias = katacr.load_alias(Path(__file__).resolve().parents[1] / "models/katacr_alias.json")
    data, counts = katacr.convert(part2, tmp_path / "out", names, alias, empty_keep=0.0)
    out = (tmp_path / "out/labels/val/ep1_1_00000.txt").read_text().split("\n")
    got = [names[int(r.split()[0])] for r in out if r]
    assert got == ["knight_me", "knight_opp", "fireball_effect"]
    assert counts["frames"] == 1 and counts["frames_dropped_empty"] == 1
    assert (tmp_path / "out/images/val/ep1_1_00000.jpg").exists() and data.exists()


def test_write_classes_roundtrip(tmp_path):
    p = write_classes(["knight", "tower_king"], tmp_path / "classes.txt")
    assert p.read_text().split() == ["knight", "tower_king"]
    cfg = TrainConfig(data_yaml=tmp_path / "data.yaml")
    assert cfg.model == "yolo26n.pt" and cfg.imgsz == 640


def test_arena_crop_and_flying_offset(cards, arena):
    scene = CRScene(cards, ["cannon", "musketeer", "ice_golem", "hog_rider",
                            "fireball", "the_log", "ice_spirit", "skeletons"])
    seen_tiles = []

    class CropDetector:
        def __call__(self, img):
            # Box in *crop* coordinates; pipeline must offset it to full-image pixels.
            h, w = img.shape[:2]
            assert (w, h) != scene.render().shape[:2][::-1]  # proves we got the crop, not full frame
            return [Detection("hog_rider", 0.9, (w / 2 - 5, h / 2 - 5, w / 2 + 5, h / 2 + 5))]

    class TileSpy:
        def __init__(self, real):
            self.real = real
        def px_to_tile(self, px, shape):
            seen_tiles.append(px)
            return self.real.px_to_tile(px, shape)

    perc = CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards, arena=arena,
                        matcher=build_matcher_from_scene(scene), detector=CropDetector())
    x0, y0, _, _ = scene.layout.arena.to_px(scene.render().shape)
    perc.tiles = TileSpy(scene.tiles)
    perc._detect(scene.render(), 1000)
    assert seen_tiles, "detector produced no tiles"
    # Crop offset applied: tile came from full-image coords, not crop-local ones.
    assert seen_tiles[0][0] > x0 and seen_tiles[0][1] > y0

    # Flying units: foot is lifted by flying_foot_frac * box height.
    assert cards["cannon"].flying is False
    from droid_harness.domains.clash_royale.cards import Card
    flying_cards = dict(cards.cards)
    base = cards.cards["hog_rider"]
    flying_cards["hog_rider"] = Card(**{**base.__dict__, "flying": True})
    cards_fly = CardDB(flying_cards)
    perc2 = CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards_fly, arena=arena,
                         matcher=build_matcher_from_scene(scene), detector=CropDetector(),
                         flying_foot_frac=0.5)
    tiles2 = []
    class Spy2(TileSpy):
        def px_to_tile(self, px, shape):
            tiles2.append(px)
            return self.real.px_to_tile(px, shape)
    perc2.tiles = Spy2(scene.tiles)
    perc2._detect(scene.render(), 1000)
    assert tiles2[0][1] < seen_tiles[0][1]  # lifted foot = smaller pixel y


def test_katacr_alias_covers_card_db():
    """Every card DB spawn class resolves through the KataCR alias map, whose
    keys exist in the upstream KataCR label list (MIT)."""
    import json
    root = Path(__file__).resolve().parents[1]
    alias = json.loads((root / "models/katacr_alias.json").read_text())
    bases = base_classes(_names())
    assert set(alias.values()) <= bases
    needed = {cls for card in _cards().cards.values() for cls in card.unit_classes}
    assert needed <= set(alias.values()), needed - set(alias.values())
    label_path = root / "models/katacr_label_list.py"
    if not label_path.exists():
        pytest.skip("run tools/fetch_katacr.py --label-list-only first (MIT upstream reference)")
    label_src = label_path.read_text()
    for kata_name in alias:
        assert f"'{kata_name}'" in label_src, kata_name
