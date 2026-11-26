"""M3 detector foundations: class list, synthetic data, arena-crop inference."""

from pathlib import Path

import numpy as np

from droid_harness.domains.clash_royale.cards import CardDB
from droid_harness.domains.clash_royale.perception.detector import Detection
from droid_harness.domains.clash_royale.perception.pipeline import CRPerception
from droid_harness.learn import synth
from droid_harness.learn.train_detector import TrainConfig, write_classes
from droid_harness.sim.cr_scene import CRScene, build_matcher_from_scene


def _cards():
    from pathlib import Path
    return CardDB.load(Path(__file__).resolve().parents[1] / "configs/domains/clash_royale/cards.json")


def _read_names(p: Path) -> list[str]:
    return [ln.strip() for ln in p.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def test_classes_match_card_db():
    root = Path("models/classes.txt")
    if not root.exists():
        root = Path(__file__).resolve().parents[1] / "models/classes.txt"
    names = _read_names(root)
    assert len(names) == len(set(names))  # no duplicates
    db = _cards()
    for card in db.cards.values():
        for cls in card.unit_classes:
            assert cls in names, cls
    for tower in ("tower_princess", "tower_king", "tower_rubble"):
        assert tower in names


def test_synth_compose_and_labels(tmp_path):
    import random
    bg = np.full((480, 640, 3), 60, np.uint8)
    sprite = np.zeros((40, 30, 4), np.uint8)
    sprite[:, :, :3] = (0, 200, 0)
    sprite[:, :, 3] = 255
    cfg = synth.SynthConfig(out_size=(320, 320), seed=1)
    img, boxes = synth.compose(bg, [(sprite, 0), (sprite, 1)], cfg, random.Random(1))
    assert img.shape == (320, 320, 3) and 1 <= len(boxes) <= 2
    for b in boxes:
        assert 0 <= b.cx <= 1 and 0 <= b.cy <= 1 and b.w > 0 and b.h > 0
    label = synth.write_yolo(tmp_path / "s.jpg", img, boxes)
    lines = label.read_text().splitlines()
    assert len(lines) == len(boxes) and all(len(x.split()) == 5 for x in lines)
    n = synth.generate_dataset([bg], [(sprite, 0)], tmp_path / "ds", 2, cfg)
    assert n == 2 and len(list((tmp_path / "ds").glob("*.jpg"))) == 2


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
