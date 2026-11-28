"""Levels (R2/R3) and the counter prior (R7): opt-in, and off means unchanged."""

from __future__ import annotations

from pathlib import Path

import pytest

from droid_harness.domains.clash_royale.account import (
    AccountLevels,
    check_strategy,
    derived_spell_tower_damage,
)
from droid_harness.domains.clash_royale.arena import Arena
from droid_harness.domains.clash_royale.cards import CardDB
from droid_harness.domains.clash_royale.policy.candidate import Candidate
from droid_harness.domains.clash_royale.policy.counter_prior import effectiveness, prior_effectiveness
from droid_harness.domains.clash_royale.policy.levels import (
    Levels,
    king_level_from_princess_hp,
    level_gap,
    level_mult,
)
from droid_harness.domains.clash_royale.policy.strategy import load_strategy
from droid_harness.domains.clash_royale.policy.utility import CRPolicy
from droid_harness.domains.clash_royale.state import GameState
from droid_harness.flywheel.eval.runner import ExperimentConfig, ExperimentRunner

ROOT = Path(__file__).resolve().parents[1]
CARDS = CardDB.load(ROOT / "tests/fixtures/cards_starter.json")
STRATS = ROOT / "configs/strategies"


def test_king_level_from_tower_hp_and_level_scaling():
    assert king_level_from_princess_hp(1512) == 2 and king_level_from_princess_hp(1890) == 5
    assert king_level_from_princess_hp(1568) is None and king_level_from_princess_hp(None) is None
    assert level_mult("knight", 3) == pytest.approx(1.21)  # common: starts at 1
    assert level_mult("fireball", 5) == pytest.approx(1.21)  # rare: starts at 3
    assert level_mult("fireball", 2) == 1.0  # never below the starting level
    lv = Levels(king_level=2, cards={"archers": 3})
    assert lv.mine("archers") == pytest.approx(1.21) and lv.mine("knight") == pytest.approx(1.1)
    assert level_gap(GameState(t_ms=0, my_king_level=2, opp_king_level=5)) == 3
    assert level_gap(GameState(t_ms=0, my_king_level=2)) == 0


def test_king_levels_survive_the_decision_log():
    st = GameState(t_ms=10, my_king_level=2, opp_king_level=5)
    back = GameState.from_digest(st.digest())
    assert (back.my_king_level, back.opp_king_level) == (2, 5)


def test_level_gap_respect_is_off_unless_the_strategy_opts_in():
    _, v6, _ = load_strategy(STRATS / "ladder-vision-v6.yaml")
    _, v7, _ = load_strategy(STRATS / "ladder-vision-v7.yaml")
    assert not v6.level_aware and v7.level_aware and v7.level_gap_respect > 0
    d = Candidate("defend.lane_threat", "knight", 0, (3.5, 10.0), 0.5)
    o = Candidate("offense.win_condition_when_opp_low", "giant", 1, (3.5, 14.0), 0.5)
    p6, p7 = CRPolicy(CARDS, Arena(), v6), CRPolicy(CARDS, Arena(), v7)
    s = GameState(t_ms=1000, my_king_level=2, opp_king_level=5)
    p6.decide(s), p7.decide(s)  # sets the per-decision level gap
    assert p6.total(d, "BALANCED") == pytest.approx(0.5 * v6.weight(d.rule_id))
    k = 1 + v7.level_gap_respect * 3
    assert p7.total(d, "BALANCED") == pytest.approx(0.5 * v7.weight(d.rule_id) * k)
    assert p7.total(o, "BALANCED") == pytest.approx(0.5 * v7.weight(o.rule_id) / k)


def test_tier0_derives_spell_tower_damage_from_levels_only_when_opted_in():
    _, v6, _ = load_strategy(STRATS / "ladder-vision-v6.yaml")
    _, v7, _ = load_strategy(STRATS / "ladder-vision-v7.yaml")
    v7.tier0.spell_tower_damage.clear()  # a level-aware strategy that doesn't hard-code it
    lv = Levels(king_level=2, cards={"fireball": 5})
    pol = CRPolicy(CARDS, Arena(), v7, levels=lv)
    fb = CARDS["fireball"]
    assert pol.tier0.spell_tower_damage("fireball", v7) == int(fb.damage * 1.21 * fb.ct_damage_pct / 100)
    assert CRPolicy(CARDS, Arena(), v6, levels=lv).tier0.spell_tower_damage("fireball", v6) == 75  # explicit


def test_counter_prior_fills_missing_rows_but_never_overrides_the_csv():
    giant_tank = prior_effectiveness(CARDS["mini_pekka"], "giant", CARDS, flying=False)
    assert giant_tank >= 0.6
    assert prior_effectiveness(CARDS["mini_pekka"], "minion", CARDS, flying=True) == 0.0  # can't hit air
    assert prior_effectiveness(CARDS["musketeer"], "other_air", CARDS, flying=True) > 0
    _, p, _ = load_strategy(STRATS / "ladder-vision-v6.yaml")
    p.counters = {("knight", "giant"): 0.0}
    assert effectiveness(p, "knight", "giant", CARDS, flying=False) == 0.0  # explicit 0 wins
    assert effectiveness(p, "musketeer", "giant", CARDS, flying=False) == 0.0  # off: missing row = never
    p._p["counter_prior"] = True
    assert effectiveness(p, "knight", "giant", CARDS, flying=False) == 0.0
    assert effectiveness(p, "musketeer", "giant", CARDS, flying=False) > 0


def test_counter_csv_generic_fallback_when_prior_off():
    _, p, _ = load_strategy(STRATS / "ladder-vision-v4.yaml")
    assert not getattr(p, "counter_prior", False)
    # Named class not in v4's CSV falls back to other_ground / other_air instead of 0.0
    assert effectiveness(p, "knight", "wizard", CARDS, flying=False) == p.counters[("knight", "other_ground")]
    assert effectiveness(p, "musketeer", "lava_hound", CARDS, flying=True) == p.counters[("musketeer", "other_air")]
    p.counters = {("knight", "giant"): 0.0}
    assert effectiveness(p, "knight", "wizard", CARDS, flying=False) == 0.0


def test_account_levels_meta_doctor_and_experiment_void(tmp_path):
    # The live file tracks real progression (the upgrader rewrites it), so assert its shape and
    # the machinery around it, not today's numbers. Fixed levels below keep the doctor checks
    # reproducible no matter what the account has upgraded.
    acct = AccountLevels.load(ROOT / "configs/accounts/theblackcat99.yaml")
    assert acct.king_level is not None and acct.king_level >= 1
    assert set(acct.cards) == {"knight", "archers", "minions", "arrows", "fireball", "giant",
                               "musketeer", "mini_pekka"}
    assert all(1 <= v <= 15 for v in acct.known_cards.values())
    assert acct.spell_tower_damage.get("fireball")
    m = acct.to_meta()
    assert m["levels_id"] == acct.levels_id and len(acct.levels_id) == 10
    upgraded = AccountLevels(acct.account, acct.king_level, {**acct.cards, "musketeer": 4})
    assert upgraded.levels_id != acct.levels_id
    fixed = AccountLevels("fixture", 2, {"knight": 2, "archers": 3, "minions": 1, "arrows": 1,
                                         "fireball": 3, "giant": 3, "musketeer": 3, "mini_pekka": 4},
                          spell_tower_damage={"fireball": 75, "arrows": 30})
    _, v6, _ = load_strategy(STRATS / "ladder-vision-v6.yaml")
    # The pinned strategy matches the fixed account, so doctor has nothing to complain about.
    assert check_strategy(v6, CARDS, fixed)["problems"] == []
    # A value measured in game beats the card DB estimate (flat below level 4: it says ~97).
    rep = check_strategy(v6, CARDS, fixed)
    assert rep["derived_spell_tower_damage"]["fireball"] > rep["measured_spell_tower_damage"]["fireball"]
    # A partially filled file still reports what is missing.
    partial = AccountLevels("fixture", 2, {"archers": 3})
    rep = check_strategy(v6, CARDS, partial)
    assert "knight" in rep["card_levels_missing"] and "fireball" in rep["derived_spell_tower_damage"]
    assert derived_spell_tower_damage(CARDS, fixed, ["fireball", "knight"]).keys() == {"fireball"}

    r = ExperimentRunner(ExperimentConfig(experiment_id="E", game_version="v", account_levels_id=acct.levels_id))
    arm, pid = r.next_assignment()
    r.record({"arm": arm, "pair_id": pid, "game_version": "v", "status": "completed", "result": "win",
              "account_levels": acct.to_meta()})
    assert r.void_reason is None
    arm, pid = r.next_assignment()
    r.record({"arm": arm, "pair_id": pid, "game_version": "v", "status": "completed", "result": "win",
              "account_levels": upgraded.to_meta()})
    assert r.void_reason and "account levels changed" in r.void_reason and r.done
