"""Tower-HP labels: offset search on real phone-a frames, offset locking, misread correction."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from droid_harness.app import load_digits
from droid_harness.core.transport.frame_slot import Frame
from droid_harness.domains.clash_royale.arena import Arena
from droid_harness.domains.clash_royale.cards import CardDB
from droid_harness.domains.clash_royale.perception.cards import CardMatcher
from droid_harness.domains.clash_royale.perception.digits import locate_number
from droid_harness.domains.clash_royale.perception.grid import Layout, Roi, TileMapper
from droid_harness.domains.clash_royale.perception.pipeline import CRPerception
from droid_harness.flywheel.eval.runner import dense_match_score

ROOT = Path(__file__).resolve().parents[1]
CR = ROOT / "configs/domains/clash_royale"
FRAME = CR / "screens/phone-a/IN_BATTLE__0.png"  # Training Camp, all four princess towers at 1400

pytestmark = pytest.mark.skipif(not FRAME.exists() or not (CR / "digits/phone-a/0.png").exists(),
                                reason="phone-a calibration not checked out")


def _frame() -> np.ndarray:
    # scrcpy streams 464x1024; the reference was saved at full resolution
    return cv2.resize(cv2.imread(str(FRAME)), (464, 1024), interpolation=cv2.INTER_LINEAR)


def test_search_finds_the_opponent_labels_the_fixed_region_misses():
    img, layout = _frame(), Layout.load(CR / "rois/phone-a.yaml")
    digits = load_digits(CR / "digits/phone-a")
    assert digits.read(layout.tower_hp["opp_left"].crop(img)) is None  # the calibrated region is ~16 px high
    v, (dx, dy), _ = locate_number(img, layout.tower_hp["opp_left"], digits)
    assert v == 1400 and 8 <= dy * 1024 <= 24
    v, (_, dy_me), _ = locate_number(img, layout.tower_hp["me_left"], digits)
    assert v == 1400 and abs(dy_me * 1024) <= 4


def test_pipeline_learns_offsets_and_reads_both_sides():
    layout = Layout.load(CR / "rois/phone-a.yaml")
    p = CRPerception(layout=layout, tiles=TileMapper.load(CR / "tiles/phone-a.yaml"),
                     cards=CardDB.load(ROOT / "tests/fixtures/cards_starter.json"), arena=Arena(),
                     matcher=CardMatcher.load(CR / "templates/phone-a"), digits=load_digits(CR / "digits/phone-a"))
    img = _frame()
    for i in range(12):  # 12 frames, 600 ms apart: tower reads run every 500 ms
        p.process(Frame(i + 1, img, 6_000_000 + i * 600_000, 0, 0))
    assert p.towers.get("opp_left") == 1400 and p.towers.get("opp_right") == 1400
    assert p.towers.get("me_left") == 1400 and p.towers.get("me_right") == 1400
    assert "opp_left" in p.hp_offsets or "opp_right" in p.hp_offsets
    assert p.tower_read_counts["opp_left"] >= 3
    assert p.warnings["tower_hp_increase"] == 0


class _Seq:
    """Digit reader stub: returns a scripted sequence, one value per read."""

    def __init__(self, values):
        self.values = list(values)

    def read(self, crop):
        return self.values.pop(0) if self.values else None


def test_one_low_misread_no_longer_locks_the_tower():
    from droid_harness.sim.cr_scene import sim_layout, sim_tiles

    lay = sim_layout()
    lay = replace(lay, towers={"me_left": Roi(0.1, 0.7, 0.3, 0.73)}, tower_hp={"me_left": Roi(0.1, 0.7, 0.3, 0.73)})
    p = CRPerception(layout=lay, tiles=sim_tiles(), cards=CardDB.load(ROOT / "tests/fixtures/cards_starter.json"),
                     arena=Arena(), matcher=CardMatcher(), digits=_Seq([1400, 444, 1400, 1400, 1400, 1380]))
    img = np.zeros((1024, 464, 3), np.uint8)
    p._last_t_ms = 10_000
    seen = []
    for _ in range(6):
        p._read_towers(img, [])
        seen.append(p.towers.get("me_left"))
    assert seen == [1400, 444, 444, 444, 1400, 1380]  # corrected after three agreeing reads
    assert any(e["kind"] == "tower_hp_corrected" for e in p.events)


def test_unknown_hp_difference_adds_nothing_to_the_dense_margin():
    base = {"result": "loss", "my_crowns": 0, "opp_crowns": 1, "duration_s": 180}
    assert dense_match_score({**base, "tower_hp_diff": None}) == dense_match_score(base)
    assert dense_match_score({**base, "tower_hp_diff": 2400}) > dense_match_score(base)


def test_tower_state_reader_rejects_mismatched_aspect(tmp_path):
    from droid_harness.domains.clash_royale.perception.towers import TowerStateReader

    lay = Layout.load(CR / "rois/phone-a.yaml")
    cv2.imwrite(str(tmp_path / "opp_left__alive__0.png"), np.zeros((26, 58), np.uint8))
    cv2.imwrite(str(tmp_path / "opp_left__rubble__0.png"), np.zeros((26, 67), np.uint8))
    with pytest.raises(ValueError, match="opp_left__rubble__0.png"):
        TowerStateReader.load(tmp_path, rois=lay.towers)


def test_princess_tower_marked_destroyed_when_bar_vanishes_and_intro_guarded():
    layout = Layout.load(CR / "rois/phone-a.yaml")
    p = CRPerception(layout=layout, tiles=TileMapper.load(CR / "tiles/phone-a.yaml"),
                     cards=CardDB.load(ROOT / "tests/fixtures/cards_starter.json"), arena=Arena(),
                     matcher=CardMatcher.load(CR / "templates/phone-a"), digits=load_digits(CR / "digits/phone-a"))
    img = _frame()
    # Build a clean frame where opp_left has no digits and no red/dark HP bar
    no_bar = img.copy()
    h, w = no_bar.shape[:2]
    r = layout.tower_hp["opp_left"]
    x0, y0 = max(0, int((r.x0 - 0.02) * w)), max(0, int((r.y0 - 0.02) * h))
    x1, y1 = min(w, int((r.x1 + 0.03) * w)), min(h, int((r.y1 + 0.05) * h))
    no_bar[y0:y1, x0:x1] = (120, 160, 180)  # sunlit sand, zero red/dark bar

    # Seed alive towers on real frame (battle HUD active)
    for i in range(6):
        p.process(Frame(i + 1, img, 6_000_000 + i * 600_000, 0, 0))
    assert p.towers["opp_left"] == 1400

    # Feed 4+ tower-read cycles with opp_left bar & digits gone -> marked 0 permanently
    for i in range(6, 12):
        p.process(Frame(i + 1, no_bar, 6_000_000 + i * 600_000, 0, 0))
    assert p.towers["opp_left"] == 0
    assert any(e["kind"] == "tower_destroyed" and e["tower"] == "opp_left" for e in p.events)
    # Returning the original frame cannot revive a destroyed tower
    p.process(Frame(13, img, 14_000_000, 0, 0))
    assert p.towers["opp_left"] == 0


def test_mid_match_clipped_leading_digit_drop_requires_confirmation():
    from droid_harness.sim.cr_scene import sim_layout, sim_tiles

    lay = sim_layout()
    lay = replace(lay, towers={"opp_right": Roi(0.7, 0.2, 0.9, 0.23)},
                  tower_hp={"opp_right": Roi(0.7, 0.2, 0.9, 0.23)})
    p = CRPerception(layout=lay, tiles=sim_tiles(), cards=CardDB.load(ROOT / "tests/fixtures/cards_starter.json"),
                     arena=Arena(), matcher=CardMatcher(), digits=_Seq([1750, 750, 1750, 750, 750]))
    img = np.zeros((1024, 464, 3), np.uint8)
    p._last_t_ms = 20_000
    seen = []
    for _ in range(5):
        p._read_towers(img, [])
        seen.append(p.towers.get("opp_right"))
    assert seen == [1750, 1750, 1750, 1750, 750]
