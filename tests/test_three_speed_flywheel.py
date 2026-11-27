"""Tests for the 3-Speed Strategy Flywheel, Tower HP digits, Triage fix, M8 v2 Ranker,
Pro Engine Rules, Intelligent Weight Caching, and One-Command CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from matchgen import write_match

from droid_harness.app import load_digits, resolve_device_bundle, resolve_strategy_path, run_doctor
from droid_harness.cli import main as cli_main
from droid_harness.core.transport.frame_slot import Frame
from droid_harness.domains.clash_royale.perception.detector import load_detector, warmup_once
from droid_harness.domains.clash_royale.perception.grid import Layout
from droid_harness.domains.clash_royale.perception.pipeline import CRPerception, infer_king_max_hp
from droid_harness.domains.clash_royale.policy.candidate import Candidate
from droid_harness.domains.clash_royale.policy.strategy import load_strategy
from droid_harness.domains.clash_royale.policy.utility import CRPolicy
from droid_harness.domains.clash_royale.state import GameState, Unit
from droid_harness.flywheel.digest import build_digest, render_agent_markdown
from droid_harness.flywheel.engagements import segment
from droid_harness.flywheel.eval.runner import ExperimentConfig, ExperimentRunner
from droid_harness.flywheel.retention import retain_dir
from droid_harness.flywheel.timeline import load_match
from droid_harness.flywheel.triage import triage
from droid_harness.flywheel.tuning.offline_study import search_offline
from droid_harness.learn.label_from_video import label_from_video
from droid_harness.learn.train_ranker import FORMAT_V2, load_ranker, state_dim, train_ranker
from droid_harness.sim.cr_scene import CRScene, build_matcher_from_scene


def test_phone_a_digits_and_king_hp_seeding(cards, arena):
    root = Path(__file__).resolve().parents[1]
    digits_dir = root / "configs/domains/clash_royale/digits/phone-a"
    reader = load_digits(digits_dir)
    assert reader is not None and len(reader.templates) == 10
    for d, tmpl in reader.templates.items():
        # Every stored phone-a glyph must self-classify cleanly.
        got, sc = reader.classify(tmpl)
        assert got == d and sc >= 0.99
    assert infer_king_max_hp(1400) == 2400
    assert infer_king_max_hp(1750) == 2904

    # Verify Layout loads separate tower_hp ROIs + clock_countdown flag.
    lay = Layout.load(root / "configs/domains/clash_royale/rois/phone-a.yaml")
    assert lay.clock_countdown is True
    assert set(lay.tower_hp) == {"opp_left", "opp_right", "opp_king", "me_left", "me_right", "me_king"}
    assert lay.tower_hp["opp_left"] != lay.towers["opp_left"]


def test_countdown_clock_alignment_without_false_drift(cards, arena):
    deck = ["hog_rider", "musketeer", "ice_golem", "cannon", "fireball", "the_log", "ice_spirit", "skeletons"]
    scene = CRScene(cards, deck, draw_tower_hp=False)
    scene.layout.clock_countdown = True
    from droid_harness.domains.clash_royale.perception.digits import DigitReader

    reader = DigitReader.from_font(scale=0.5, thickness=1)
    perc = CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards, arena=arena,
                        matcher=build_matcher_from_scene(scene), digits=reader)
    img = scene.render()
    x0, _y0, _x1, y1 = scene.layout.clock.to_px(img.shape)
    import cv2

    # At match t=30s (pts=30_000_000), Clash Royale regulation clock shows 2:30 (150s left -> 30s elapsed).
    cv2.putText(img, "2:30", (x0 + 4, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    s = perc.process(Frame(1, img, 0, 0, 0))
    s = perc.process(Frame(2, img, 30_000_000, 0, 0))
    assert s.match_t_s == pytest.approx(30.0)
    assert "clock_drift" not in s.warnings


def test_triage_ignores_play_transition_slot_warnings_but_flags_sustained(tmp_path, arena, cards):
    d = write_match(tmp_path, "m_trans", result="loss", respond=True)
    # Inject 4 frames of transient hand_unreadable around our play at t=11_600 -> must NOT be PERCEPTION.
    dec_path = d / "decisions.jsonl"
    recs = [json.loads(line) for line in dec_path.read_text().splitlines()]
    for r in recs:
        if r["t_ms"] in (11_500, 12_000, 12_500):
            r["state"]["warnings"] = ["hand_unreadable", "affordability_mismatch"]
    dec_path.write_text("\n".join(json.dumps(r) for r in recs))
    tl = load_match(d)
    [e] = [triage(e, tl, arena) for e in segment(tl, arena, cards)]
    assert e.triage != "PERCEPTION"

    # Conversely, 100% broken templates across the entire engagement MUST flag PERCEPTION.
    for r in recs:
        r["state"]["warnings"] = ["hand_unreadable"]
    dec_path.write_text("\n".join(json.dumps(r) for r in recs))
    tl2 = load_match(d)
    [e2] = [triage(e, tl2, arena) for e in segment(tl2, arena, cards)]
    assert e2.triage == "PERCEPTION"


def test_pro_engine_rules_v6(arena):
    root = Path(__file__).resolve().parents[1]
    from droid_harness.domains.clash_royale.cards import CardDB

    starter_cards = CardDB.load(root / "tests/fixtures/cards_starter.json")
    _, p6, _ = load_strategy(root / "configs/strategies/ladder-vision-v6.yaml")
    pol = CRPolicy(starter_cards, arena, p6)

    # 1. GiantBackBuild is blocked during OPENING (t=2s), even at 9.2 elixir.
    s_open = GameState(t_ms=2000, phase="x1", my_elixir=9.2, opp_elixir=(7.0, 8.0),
                       hand=["giant", "knight", "archers", "arrows"],
                       towers={"me_left": 1400, "me_right": 1400, "opp_left": 1400, "opp_right": 1400})
    d_open = pol.decide(s_open)
    assert all(c.rule_id != "offense.giant_back_build" for c in d_open.candidates)

    # 2. Healthy-tower chip filter ignores a lone damaged skeleton (0.33 elixir value) at 1400/1400 HP.
    pol.reset_match()
    s_chip = GameState(
        t_ms=25_000, phase="x1", my_elixir=6.0, opp_elixir=(5.0, 5.0),
        hand=["knight", "musketeer", "fireball", "arrows"],
        towers={"me_left": 1400, "me_right": 1400, "opp_left": 1400, "opp_right": 1400},
        units=[Unit(1, "skeleton", "opp", (3.5, 13.0), vel=(0.0, -2.0), hp_pct=0.5)],
    )
    d_chip = pol.decide(s_chip)
    assert all(c.rule_id != "defend.lane_threat" for c in d_chip.candidates)

    # 3. OppositeLanePunish fires on right bridge when enemy drops Giant in back-left (y=26).
    pol.reset_match()
    s_pun = GameState(
        t_ms=35_000, phase="x1", my_elixir=8.5, opp_elixir=(3.0, 4.0),
        hand=["mini_pekka", "archers", "arrows", "fireball"],
        towers={"me_left": 1400, "me_right": 1400, "opp_left": 1400, "opp_right": 1400},
        units=[Unit(2, "giant", "opp", (3.5, 26.0), vel=(0.0, -1.0), hp_pct=1.0)],
    )
    d_pun = pol.decide(s_pun)
    punish_cands = [c for c in d_pun.candidates if c.rule_id == "offense.opposite_lane_punish"]
    assert punish_cands and punish_cands[0].card == "mini_pekka" and punish_cands[0].tile[0] > 9.0

    # 4. Exact level-calibrated lethal spell: Fireball (75 dmg) finishes a 70 HP enemy tower immediately.
    pol.reset_match()
    s_lethal = GameState(
        t_ms=90_000, phase="x1", my_elixir=5.0, opp_elixir=(5.0, 5.0),
        hand=["fireball", "knight", "archers", "giant"],
        towers={"me_left": 1400, "me_right": 1400, "opp_left": 70, "opp_right": 1400},
    )
    d_lethal = pol.decide(s_lethal)
    assert d_lethal.chosen is not None and d_lethal.chosen.rule_id == "t0.lethal_spell"


def test_ranker_v2_board_features_residual_and_cache(tmp_path, cards, arena):
    deck = ["knight", "archers", "minions", "arrows", "fireball", "giant", "musketeer", "mini_pekka"]
    m_root = tmp_path / "matches"
    write_match(m_root, "m1", result="win")
    # Re-write with starter-deck cards so label_from_video v2 + engagement credit runs end-to-end.
    d = m_root / "m2"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({
        "match_id": "m2", "status": "completed", "result": "win", "my_crowns": 2, "opp_crowns": 0,
        "events": [{"kind": "my_play", "card": "knight", "slot": 0, "t_ms": 5200}],
    }))
    (d / "decisions.jsonl").write_text(json.dumps({
        "seq": 1, "t_ms": 5000, "posture": "BALANCED", "chosen_rule": "defend.lane_threat",
        "chosen_card": "knight", "tile": [3.5, 11.0], "score": 0.6,
        "candidates": [
            {"rule_id": "defend.lane_threat", "card": "knight", "tile": [3.5, 11.0], "score": 0.6},
            {"rule_id": "cycle.cheap_cycle", "card": "archers", "tile": [13.5, 1.5], "score": 0.2},
        ],
        "state": {
            "t_ms": 5000, "phase": "x1", "my_elixir": 6.0, "opp_elixir": [3.0, 4.0],
            "hand": ["knight", "archers", "giant", "arrows"],
            "units": [[1, "mini_pekka", "opp", 3.5, 14.0, 0.0, -2.0, 1.0]],
            "towers": {"me_left": 1400, "opp_left": 1400}, "crowns": [0, 0],
            "opp_deck_seen": ["mini_pekka", "giant"], "opp_last_four": ["mini_pekka"],
            "posture": "BALANCED", "warnings": [],
        },
    }) + "\n")
    labels = tmp_path / "labels_v2.jsonl"
    st = label_from_video(d, labels, deck=deck, version=2, cards=cards, arena=arena)
    assert st["rows"] == 1
    rows = [json.loads(line) for line in labels.read_text().splitlines()]
    assert len(rows[0]["state"]) == state_dim(deck, version=2)

    onnx_out = tmp_path / "ranker_v2.onnx"
    meta = train_ranker(rows * 8, onnx_out, epochs=5, hidden=(16, 8), seed=0)
    assert meta["format"] == FORMAT_V2
    r1 = load_ranker(onnx_out, deck=deck)
    r2 = load_ranker(onnx_out, deck=deck)
    assert r1 is r2  # cached ONNX session!
    cands = [Candidate("defend.lane_threat", "knight", 0, (3.5, 11.0), 0.60)]
    res = r1.rescore_residual(rows[0]["state"], cands, weight=0.20)
    assert 0.45 <= res[0].score <= 0.75


def test_detector_cache_and_stage_a_dense_futility(tmp_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])
    outp = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, 6])
    const = helper.make_node("Constant", [], ["out"],
                             value=numpy_helper.from_array(np.zeros((1, 1, 6), np.float32)))
    m = helper.make_model(helper.make_graph([const], "g", [inp], [outp]),
                          opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    p = tmp_path / "tiny.onnx"
    onnx.save(m, p)
    d1 = load_detector(p, ["knight_me"], providers=["CPUExecutionProvider"])
    d2 = load_detector(p, ["knight_me"], providers=["CPUExecutionProvider"])
    assert d1 is d2
    assert warmup_once(d1, 1) is True
    assert warmup_once(d2, 1) is False

    # Stage-A dense-margin futility stops clearly losing candidate even on W-W / L-L ties
    runner = ExperimentRunner(ExperimentConfig(
        experiment_id="EXP-FUT", game_version="9.x", test="pairs", futility_min_pairs=6,
    ))
    for i in range(12):
        arm, pid = runner.next_assignment()
        # Both win, so discordant binary pairs == 0, BEFORE futility: candidate wins 1-0 (-900 HP),
        # champion stomps 3-0 (+2500 HP) -> dense margin strongly negative -> early reject!
        is_cand = arm == "candidate"
        runner.record({
            "match_id": f"m{i}", "arm": arm, "pair_id": pid, "game_version": "9.x",
            "status": "completed", "result": "win",
            "my_crowns": 1 if is_cand else 3, "opp_crowns": 0,
            "tower_hp_diff": -900 if is_cand else 2500, "duration_s": 179 if is_cand else 95,
        })
    assert runner.status == "rejected" and runner.futility_reason is not None


def test_digest_retention_offline_tuning_and_one_command_cli(tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    assert resolve_strategy_path("v6", root=root).name == "ladder-vision-v6.yaml"
    bundle = resolve_device_bundle("phone-a", root=root, with_detector=False)
    assert bundle["digits"] is not None and bundle["digits"].name == "phone-a"
    doc = run_doctor(device="phone-a", strategy="v6", root=root, live=False)
    assert doc["checks"]["digit_templates"] == 10
    assert doc["checks"]["deck_templates_missing"] == []

    # Build synthetic match root with both a win and a STRATEGY loss, plus dummy stream.h264
    m_root = tmp_path / "batch"
    w_dir = write_match(m_root, "win1", result="win", respond=True)
    l_dir = write_match(m_root, "loss1", result="loss", respond=False)
    (w_dir / "stream.h264").write_bytes(b"0" * 4096)
    (l_dir / "stream.h264").write_bytes(b"0" * 4096)

    from droid_harness.domains.clash_royale.arena import Arena
    from droid_harness.domains.clash_royale.cards import CardDB

    cards = CardDB.load(root / "tests/fixtures/cards_min.json")
    dig = build_digest(m_root, cards, Arena(), ledger_path=tmp_path / "exp.jsonl")
    md = render_agent_markdown(dig)
    assert dig["completed_matches"] == 2 and "## 4. Automated Failure Clusters" in md
    assert len(md.encode()) < 30_000

    # Offline counterfactual search over hog-cycle-v1
    out_cand = tmp_path / "hog-opt.yaml"
    res = search_offline(root / "configs/strategies/hog-cycle-v1.yaml", [w_dir, l_dir],
                         cards, Arena(), n_trials=6, out_yaml=out_cand, seed=1)
    assert out_cand.exists() and res["states_evaluated"] > 0

    # Retention prunes win stream.h264 and keeps loss stream.h264 by default
    ret = retain_dir(m_root)
    assert ret["matches_pruned"] == 1
    assert not (w_dir / "stream.h264").exists()
    assert (l_dir / "stream.h264").exists()

    # CLI digest smoke test
    out_md = tmp_path / "digest.md"
    cli_main(["digest", "--data", str(m_root), "--cards", str(root / "tests/fixtures/cards_min.json"),
              "--store", str(tmp_path / "exp.jsonl"), "--out", str(out_md)])
    assert out_md.exists()
    capsys.readouterr()
