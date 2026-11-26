import math

import pytest

from droid_harness.domains.clash_royale import path
from droid_harness.domains.clash_royale.elixir import (
    ElixirModel,
    OpponentElixir,
    generated_between,
    phase_at,
    time_to_reach,
)
from droid_harness.domains.clash_royale.opponent_model import CycleTracker
from droid_harness.domains.clash_royale.state import GameState, Unit


def test_official_phases():
    assert [phase_at(t) for t in (0, 119.9, 120, 179.9, 180, 239.9, 240)] == \
        ["x1", "x1", "x2", "x2", "ot_x2", "ot_x2", "ot_x3"]


def test_generation_rates():
    assert generated_between(0, 2.8) == pytest.approx(1.0)
    assert generated_between(120, 121.4) == pytest.approx(1.0)  # double
    assert generated_between(240, 240 + 2.8 / 3) == pytest.approx(1.0)  # triple
    assert generated_between(119, 121) == pytest.approx(1 / 2.8 + 2 / 2.8)
    assert time_to_reach(4.0, 5.0, 0.0) == pytest.approx(2.8, abs=0.01)


def test_elixir_model_caps_and_counts_leak():
    m = ElixirModel(start=9.0)
    m.advance(2.8 * 3)
    assert m.value == 10.0 and m.leaked == pytest.approx(2.0)


def test_elixir_model_snaps_to_hud_after_persistent_disagreement():
    m = ElixirModel(start=5.0, snap_after_s=1.0)
    m.observe(5.1, 0.0)
    assert m.discrepancy_s == 0
    m.observe(8.0, 0.1)  # big disagreement starts
    assert m.value < 7
    m.observe(8.0 + generated_between(0.1, 1.2), 1.2)
    assert m.discrepancy_s >= 1.0 and m.value == pytest.approx(8.0 + generated_between(0.1, 1.2))


def test_opponent_interval_collapses_at_cap_and_widens_on_unknown_cost():
    o = OpponentElixir(low=5, high=5)
    o.on_play((3, 5), t_s=0.0)  # unknown card costing 3..5
    assert o.interval() == (0.0, 2.0)
    o.advance(40.0)  # far longer than needed to refill
    assert o.interval() == (10.0, 10.0)  # a full bar is known
    o.on_play(4, t_s=40.0)
    assert o.interval() == (6.0, 6.0)


def test_opponent_play_proves_minimum():
    o = OpponentElixir(low=1, high=4)
    o.on_play(4, 0.0)  # they had at least 4
    assert o.low == 0 and o.high == 0


def test_cycle_tracker_hand_inference():
    c = CycleTracker()
    deck = ["golem", "night_witch", "baby_dragon", "lightning", "tornado", "zap", "mega_minion", "lumberjack"]
    for k in deck:
        c.on_play(k)
    assert c.deck_known
    assert c.certain_hand() == {"golem", "night_witch", "baby_dragon", "lightning"}
    assert c.next_card() == "tornado"
    assert c.out_of_cycle("zap") and not c.out_of_cycle("golem")


def test_path_enemy_walks_to_bridge_then_tower(arena):
    u = Unit(1, "hog_rider", "opp", (3.5, 20.0), vel=(0, -2.0))
    p1 = path.advance(u, 1.0, arena)
    assert p1 == pytest.approx((3.5, 18.0))
    p3 = path.advance(u, 3.0, arena)  # crossed the river (y 17 -> 15) and walking on
    assert p3[1] < 15
    far = path.advance(u, 60.0, arena)  # stops short of the tower
    assert math.dist(far, arena.towers["me_left"]) == pytest.approx(arena.tower_radius)


def test_threat_eta_and_sorting(arena, cards):
    s = GameState(t_ms=0, units=[Unit(1, "hog_rider", "opp", (3.5, 18), vel=(0, -2)),
                                 Unit(2, "golem", "opp", (14.5, 26), vel=(0, -0.75))],
                  towers={"me_left": 1000, "me_right": 1000, "me_king": 2000})
    th = s.threats(arena, cards)
    assert [t.kind for t in th] == ["hog_rider", "golem"]
    assert th[0].tower == "me_left" and th[0].eta_to_tower_s == pytest.approx(th[0].distance / 2.0)


def test_threat_targets_king_when_princess_down(arena, cards):
    s = GameState(t_ms=0, units=[Unit(1, "hog_rider", "opp", (3.5, 18), vel=(0, -2))],
                  towers={"me_left": 0, "me_right": 1000, "me_king": 2000})
    assert s.threats(arena, cards)[0].tower == "me_king"


def test_digest_roundtrip():
    s = GameState(t_ms=1234, hand=["hog_rider", "cannon", None, "fireball"],
                  units=[Unit(3, "knight", "opp", (4.0, 20.0), vel=(0.0, -1.0), hp_pct=0.5)],
                  towers={"opp_left": 100})
    s2 = GameState.from_digest(s.digest())
    assert s2.digest() == s.digest()
