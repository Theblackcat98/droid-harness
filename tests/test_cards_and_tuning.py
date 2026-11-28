"""R5 targeted Speed 1, R6 unseen-card tooling, template harvesting."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from droid_harness.domains.clash_royale.arena import Arena
from droid_harness.domains.clash_royale.cards import CardDB
from droid_harness.domains.clash_royale.perception.classes import load_class_names
from droid_harness.domains.clash_royale.perception.detector import Detection
from droid_harness.domains.clash_royale.perception.harvest import TemplateHarvester, promote
from droid_harness.domains.clash_royale.policy.strategy import load_strategy
from droid_harness.flywheel.tuning.targeted import PATTERNS, scan_patterns, search_targeted
from droid_harness.learn.unseen_eval import check_classes, unseen_report

ROOT = Path(__file__).resolve().parents[1]
CARDS = CardDB.load(ROOT / "tests/fixtures/cards_starter.json")
EXP = ROOT / "data/matches-exp-v4-v6"


def test_harvester_saves_the_refill_card_and_promote_keeps_agreeing_crops(tmp_path):
    h = TemplateHarvester(tmp_path / "harvest", refill_ms=1000, min_gap_ms=0)
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (60, 48, 3), dtype=np.uint8)
    h.expect(2, "hog_rider", 10_000)
    assert h.observe(2, base, None, 10_500, greyed=False) is None  # still refilling
    for k in range(4):
        noisy = np.clip(base.astype(int) + rng.integers(-6, 6, base.shape), 0, 255).astype(np.uint8)
        assert h.observe(2, noisy, None, 11_200 + 100 * k, greyed=False) is not None
    assert h.observe(2, base, "hog_rider", 12_000, greyed=False) is None  # read fine: expectation cleared
    assert h.observe(2, base, None, 12_100, greyed=False) is None
    h.expect(1, "skeletons", 20_000)  # disagreeing crops for another card
    for k in range(3):
        h.observe(1, rng.integers(0, 255, (60, 48, 3), dtype=np.uint8), None, 21_500 + k, greyed=True)
    rep = {(r["card"], r["state"]): r for r in promote(tmp_path / "harvest", tmp_path / "tpl")}
    assert rep[("hog_rider", "affordable")]["written"].endswith("hog_rider__0.png")
    assert (tmp_path / "tpl/hog_rider__0.png").exists()
    assert rep[("skeletons", "greyed")]["written"] is None and "disagree" in rep[("skeletons", "greyed")]["skip"]


@pytest.mark.skipif(not EXP.exists(), reason="committed EXP-0001 data not present")
def test_targeted_search_writes_a_one_hypothesis_candidate(tmp_path):
    dirs = sorted(d for d in EXP.iterdir() if (d / "decisions.jsonl").exists())
    v6 = ROOT / "configs/strategies/ladder-vision-v6.yaml"
    counts = scan_patterns(v6, dirs, CARDS, Arena(), max_states=1500)
    assert set(counts) == set(PATTERNS) and counts["over_defend_bleed"] > 0
    out = tmp_path / "strategies/ladder-vision-v6-t1.yaml"
    res = search_targeted(v6, dirs, CARDS, Arena(), pattern="over_defend_bleed", trials=12, max_states=1500,
                          min_removed=0.1, max_collateral_pct=5.0, out_yaml=out)
    assert res["best"] is not None and res["candidate_yaml"] == str(out)
    assert 1 <= len(res["best"]["knobs"]) <= 2
    spec, params, _ = load_strategy(out)  # loads, with the champion's counters
    assert spec.strategy == "ladder-vision-v6-t1" and spec.parent == "ladder-vision-v6" and params.counters
    raw = yaml.safe_load(out.read_text())
    assert raw["deck"] == yaml.safe_load(v6.read_text())["deck"]


def test_unseen_report_counts_catch_all_wrong_name_and_missed(tmp_path):
    part2 = tmp_path / "part2" / "seq1"
    part2.mkdir(parents=True)
    (tmp_path / "part2/ClashRoyale_detection.yaml").write_text(
        yaml.safe_dump({"names": {0: "knight", 1: "hog-rider", 2: "golem", 3: "balloon"}}))
    cv2.imwrite(str(part2 / "f1.png"), np.zeros((896, 568, 3), np.uint8))
    rows = ["0 0.2 0.2 0.1 0.1 1", "1 0.5 0.5 0.1 0.1 1", "2 0.8 0.8 0.1 0.1 0", "3 0.5 0.2 0.1 0.1 1"]
    (part2 / "f1.txt").write_text("\n".join(rows) + "\n")
    alias = tmp_path / "alias.json"
    alias.write_text(json.dumps({"knight": "knight"}))
    generic = tmp_path / "generic.json"
    generic.write_text(json.dumps({"named": {}, "other_ground": ["hog-rider", "golem"], "other_air": ["balloon"]}))

    def box(cx, cy):
        return (cx * 568 - 28, cy * 896 - 45, cx * 568 + 28, cy * 896 + 45)

    def detector(img):  # hog rider -> catch-all (right side), golem -> named wrongly, balloon missed
        return [Detection("other_ground_opp", 0.9, box(0.5, 0.5)), Detection("giant_me", 0.8, box(0.8, 0.8)),
                Detection("knight_opp", 0.9, box(0.2, 0.2))]

    rep = unseen_report(detector, tmp_path / "part2", ["knight_me", "knight_opp", "giant_me", "other_ground_opp"],
                        alias_path=alias, generic_path=generic)
    assert rep["units"] == 3  # the knight is named in this model, so it's not "unseen"
    assert rep["per_card"]["hog-rider"]["catch_all"] == 1 and rep["per_card"]["hog-rider"]["side_ok"] == 1
    assert rep["per_card"]["golem"]["wrong_name"] == 1 and rep["per_card"]["golem"]["as:giant"] == 1
    assert rep["per_card"]["balloon"]["missed"] == 1


def test_next_class_list_keeps_current_indices_and_checks_against_cards():
    cur = load_class_names(ROOT / "models/classes.txt")
    nxt = load_class_names(ROOT / "models/classes-next.txt")
    assert cur[:31] == nxt[:31] and len(cur) >= len(nxt)
    assert {"hog_rider_me", "hog_rider_opp", "zap_effect", "log_effect", "cannon_me"} <= set(cur)
    rep = check_classes(cur, CARDS)
    assert "knight_me" not in rep["unresolved"] and "goblin_opp" in rep["unresolved"]  # starter fixture only
