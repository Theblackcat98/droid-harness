"""Dataset combining (no train/val leakage), frame export split by match, per-class eval."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from droid_harness.domains.clash_royale.perception.detector import Detection
from droid_harness.learn.datasets import combine, export_frames
from droid_harness.learn.evaluate import evaluate, match, threat_gate

NAMES = ["knight_me", "knight_opp", "minion_opp"]


def _ds(root: Path, splits: dict[str, list[tuple[str, list[str]]]]) -> Path:
    """splits: split -> [(image stem, label rows)]"""
    for split, items in splits.items():
        for stem, rows in items:
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), np.zeros((100, 60, 3), np.uint8))
            (root / "labels" / split / f"{stem}.txt").write_text("".join(r + "\n" for r in rows))
    y = root / "data.yaml"
    y.write_text(yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val",
                                 "names": dict(enumerate(NAMES))}))
    return y


def test_combine_lists_sources_and_refuses_overlap(tmp_path):
    synth = _ds(tmp_path / "synth", {"train": [("s0", [])], "val": [("s1", [])]})
    real = _ds(tmp_path / "real", {"train": [("r0", [])], "val": [("r1", [])]})
    out = combine(tmp_path / "train.yaml", [(synth, "train"), (real, "train")], [(real, "val")])
    d = yaml.safe_load(out.read_text())
    assert len(d["train"]) == 2 and d["val"] == [str((tmp_path / "real/images/val").resolve())]
    with pytest.raises(ValueError, match="both train and val"):
        combine(tmp_path / "bad.yaml", [(real, "val")], [(real, "val")])
    other = tmp_path / "other"
    y = _ds(other, {"train": [("o", [])]})
    y.write_text(y.read_text().replace("minion_opp", "giant_opp"))
    with pytest.raises(ValueError, match="class list differs"):
        combine(tmp_path / "x.yaml", [(synth, "train"), (y, "train")], [(real, "val")])


def test_per_class_eval_and_threat_gate(tmp_path):
    # Every image: one enemy knight (found) and one enemy minion (missed).
    rows = ["1 0.5 0.5 0.2 0.2", "2 0.2 0.2 0.1 0.1"]
    data = _ds(tmp_path / "real", {"val": [(f"v{i}", rows) for i in range(25)]})

    def det(img):
        h, w = img.shape[:2]
        return [Detection("knight_opp", 0.9, (0.4 * w, 0.4 * h, 0.6 * w, 0.6 * h)),
                Detection("knight_me", 0.5, (0, 0, 5, 5))]  # a false positive

    rep = evaluate(det, data)
    pc = rep["per_class"]
    assert rep["images"] == 25
    assert pc["knight_opp"]["recall"] == 1.0 and pc["minion_opp"]["recall"] == 0.0
    assert pc["knight_me"]["precision"] == 0.0 and pc["knight_me"]["support"] == 0
    ok, failing = threat_gate(rep, 0.8, min_support=20)
    assert not ok and any("minion_opp: recall 0.00" in f for f in failing)
    ok, failing = threat_gate(rep, 0.8, min_support=30)
    assert not ok and all("labeled examples" in f for f in failing)  # unmeasured isn't passing


def test_match_is_greedy_by_confidence():
    gts = [("knight_opp", (0, 0, 10, 10))]
    dets = [Detection("knight_opp", 0.4, (0, 0, 10, 10)), Detection("knight_opp", 0.9, (1, 1, 10, 10))]
    st = match(dets, gts)
    assert st["knight_opp"] == {"tp": 1, "fp": 1, "fn": 0}


def test_export_frames_splits_by_match_and_prelabels(tmp_path, arena):
    from test_replay import _record

    from droid_harness.domains.clash_royale.cards import CardDB
    from droid_harness.sim.cr_scene import CRScene

    root = Path(__file__).resolve().parents[1]
    cards = CardDB.load(root / "tests/fixtures/cards_starter.json")
    deck = ["knight", "archers", "minions", "arrows", "fireball", "giant", "musketeer", "mini_pekka"]
    dirs = []
    for i in range(3):
        scene = CRScene(cards, deck, elixir=7.0, draw_tower_hp=False)
        _record(tmp_path / "matches" / f"m{i}", scene, 2.0, 15, {})
        dirs.append(tmp_path / "matches" / f"m{i}")

    def det(crop):
        h, w = crop.shape[:2]
        return [Detection("knight_opp", 0.9, (w * 0.4, h * 0.4, w * 0.5, h * 0.5)),
                Detection("hog_rider_opp", 0.9, (0, 0, 4, 4))]  # not in our list: dropped

    data, counts = export_frames(dirs, tmp_path / "own", scene.layout, NAMES, every_ms=500,
                                 val_frac=0.4, detector=det)
    assert counts["train"] > 0 and counts["val"] > 0
    groups = yaml.safe_load((tmp_path / "own/groups.json").read_text())
    for split in ("train", "val"):
        for p in (tmp_path / "own/images" / split).iterdir():
            assert groups[p.stem.split("__")[0]] == split  # a match never straddles the split
    lbl = next((tmp_path / "own/labels").rglob("*.txt")).read_text().split("\n")
    assert [r.split()[0] for r in lbl if r] == ["1"]
    img = next((tmp_path / "own/images").rglob("*.jpg"))
    x0, y0, x1, y1 = scene.layout.arena.to_px((scene.height, scene.width, 3))
    assert cv2.imread(str(img)).shape[:2] == (y1 - y0, x1 - x0)  # arena crops, like inference
