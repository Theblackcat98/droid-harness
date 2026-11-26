import math

import pytest

from droid_harness.core.runtime.intent_bus import Intent
from droid_harness.domains.clash_royale.policy import posture
from droid_harness.domains.clash_royale.policy.candidate import Candidate
from droid_harness.domains.clash_royale.policy.constraints import Constraints
from droid_harness.domains.clash_royale.policy.placement import Placement
from droid_harness.domains.clash_royale.policy.strategy import strategy_id
from droid_harness.domains.clash_royale.policy.utility import CRPolicy
from droid_harness.domains.clash_royale.state import GameState, Unit

TOWERS = {"me_left": 3000, "me_right": 3000, "me_king": 4800, "opp_left": 3000, "opp_right": 3000,
          "opp_king": 4800}
HAND = ["hog_rider", "cannon", "fireball", "skeletons"]


def state(**kw):
    base = dict(t_ms=60_000, my_elixir=6.0, opp_elixir=(5.0, 5.0), hand=list(HAND), towers=dict(TOWERS))
    base.update(kw)
    return GameState(**base)


def test_strategy_loads_and_hash_is_stable(strategy):
    spec, p, sid = strategy
    assert spec.deck[0] == "hog_rider" and p.counters[("cannon", "hog_rider")] == 0.9
    assert p.act_threshold == 0.15 and p.min_action_gap_ms == 350
    yaml_bytes = open("tests/fixtures/strategy.yaml", "rb").read()
    csv_bytes = open("tests/fixtures/counters/hog-cycle-v1.csv", "rb").read()
    assert len(sid) == 12 and sid == strategy_id(yaml_bytes, csv_bytes)
    assert strategy_id(yaml_bytes + b"\n# edit", csv_bytes) != sid
    with pytest.raises(AttributeError):
        _ = p.not_a_param


def test_lead_target_accounts_for_deploy_and_latency(arena, cards, strategy):
    _, p, _ = strategy
    pl = Placement(arena, cards, p.placement, exec_latency_s=0.05)
    hog = Unit(1, "hog_rider", "opp", (3.5, 24.0), vel=(0.0, -2.0))
    s = state(units=[hog])
    lead = pl.lead_target(cards["knight"], hog, s, s.t_ms)  # 0.05 s latency + 1 s deploy
    assert lead == pytest.approx((3.5, 24.0 - 2.0 * 1.05))
    later = pl.lead_target(cards["knight"], hog, s, s.t_ms + 500)
    assert later[1] == pytest.approx(24.0 - 2.0 * 1.55)


def test_spell_lead_includes_travel_time(arena, cards, strategy):
    _, p, _ = strategy
    pl = Placement(arena, cards, p.placement, exec_latency_s=0.0)
    u = Unit(1, "knight", "opp", (3.5, 22.0), vel=(0.0, -1.0))  # walking straight down the left lane
    s = state(units=[u])
    aim = pl.lead_target(cards["fireball"], u, s, s.t_ms)
    travel = math.dist(arena.king_tower_spell_origin, aim) / cards["fireball"].projectile_speed_tiles_s
    assert aim[1] == pytest.approx(22.0 - travel, abs=0.05)  # fixed point converged


def test_building_waits_for_bridge(arena, cards, strategy):
    _, p, _ = strategy
    pl = Placement(arena, cards, p.placement, exec_latency_s=0.04)
    s = state(units=[Unit(1, "hog_rider", "opp", (3.5, 25.0), vel=(0, -2.0))])
    th = s.threats(arena, cards)[0]
    t = pl.timing(cards["cannon"], th, s)
    assert t == s.t_ms + int((8.0 / 2.0 - 1.0 - 0.04) * 1000)


def test_constraints(arena, cards):
    c = Constraints(arena, cards, min_action_gap_ms=350)
    s = state(my_elixir=3.0, last_action_ms=59_900)
    ok = Candidate("x", "cannon", 1, (9.0, 10.0), 1.0, execute_at_ms=60_400)
    assert c.why_not(s, ok) is None
    assert c.why_not(s, Candidate("x", "cannon", 0, (9, 10), 1)) == "not_in_hand"
    assert c.why_not(s, Candidate("x", "hog_rider", 0, (9, 10), 1)) == "not_enough_elixir"
    assert c.why_not(state(), Candidate("x", "cannon", 1, (9, 25), 1)) == "illegal_tile"
    assert c.why_not(state(), Candidate("x", "fireball", 2, (9, 25), 1)) is None  # spells anywhere
    assert c.why_not(s, Candidate("x", "cannon", 1, (9, 10), 1, execute_at_ms=60_000)) == "too_soon"
    pocket = state(pocket_open={"left": True, "right": False})
    assert c.why_not(pocket, Candidate("x", "cannon", 1, (3.5, 19), 1)) is None


def test_tier0_lethal_spell_prefers_override_damage(arena, cards, strategy):
    _, p, _ = strategy
    p.tier0.spell_tower_damage["fireball"] = 300
    pol = CRPolicy(cards, arena, p)
    towers = dict(TOWERS, opp_right=250)
    d = pol.decide(state(towers=towers))
    assert d.chosen.rule_id == "t0.lethal_spell" and d.chosen.tile == arena.towers["opp_right"]
    p.tier0.spell_tower_damage.clear()


def test_tier0_leak_guard_needs_duration(arena, cards, strategy):
    _, p, _ = strategy
    pol = CRPolicy(cards, arena, p)
    first = pol.decide(state(t_ms=40_000, my_elixir=10.0, opp_elixir=(9, 9)))
    assert first.chosen is None or first.chosen.rule_id != "t0.leak_guard"  # not yet 800 ms at the cap
    pol = CRPolicy(cards, arena, p)
    pol.tier0.check(state(t_ms=40_000, my_elixir=10.0, opp_elixir=(9, 9)), p)
    d = pol.decide(state(t_ms=40_900, my_elixir=10.0, opp_elixir=(9, 9)))
    assert d.chosen.rule_id == "t0.leak_guard" and d.chosen.card == "skeletons"


def test_posture_classification(arena, cards, strategy):
    _, p, _ = strategy
    pol = CRPolicy(cards, arena, p)
    dmg = pol.tier0.spell_tower_damage
    assert posture.classify(state(t_ms=5_000), p, cards, dmg) == "OPENING"
    assert posture.classify(state(opp_elixir=(1.0, 2.5)), p, cards, dmg) == "PRESS"
    assert posture.classify(state(t_ms=130_000, phase="x2", crowns=(1, 0)), p, cards, dmg) == "DEFEND_LEAD"
    ot = state(t_ms=200_000, phase="ot_x2", towers=dict(TOWERS, opp_left=250))
    assert posture.classify(ot, p, cards, dmg) == "SPELL_CYCLE"
    assert posture.classify(state(), p, cards, dmg, Intent(posture="PRESS")) == "PRESS"
    assert posture.classify(state(), p, cards, dmg, Intent(posture="OPENING")) == "BALANCED"  # not allowlisted


def test_policy_defends_hog_with_cannon_at_pull_tile(arena, cards, strategy):
    _, p, _ = strategy
    pol = CRPolicy(cards, arena, p)
    s = state(units=[Unit(1, "hog_rider", "opp", (3.5, 17.5), vel=(0, -2.0))])
    d = pol.decide(s)
    assert d.chosen.rule_id == "defend.lane_threat" and d.chosen.card == "cannon"
    assert d.chosen.tile == tuple(p.placement.building_pull["left"])
    rec = d.to_record()
    assert rec["chosen_rule"] == "defend.lane_threat" and len(rec["candidates"]) >= 2
    # one action in flight: no second decision until confirmed or timed out
    assert pol.decide(state(t_ms=s.t_ms + 100, units=s.units)).chosen is None
    pol.confirm_played("cannon", s.t_ms + 150)
    assert pol.pending is None


def test_policy_attacks_when_opponent_low(arena, cards, strategy):
    _, p, _ = strategy
    pol = CRPolicy(cards, arena, p)
    d = pol.decide(state(my_elixir=8.0, opp_elixir=(0.5, 1.5), towers=dict(TOWERS, opp_left=1200)))
    assert d.chosen.rule_id == "offense.win_condition_when_opp_low" and d.chosen.card == "hog_rider"
    assert d.chosen.tile[0] == pytest.approx(3.5)  # weaker (left) lane
    assert d.posture == "PRESS"


def test_spell_value_needs_enough_elixir_hit(arena, cards, strategy):
    _, p, _ = strategy
    pol = CRPolicy(cards, arena, p)
    trio = [Unit(i, "musketeer", "opp", (9.0 + 0.3 * i, 22.0)) for i in range(2)]
    d = pol.decide(state(units=trio, hand=["fireball", "cannon", "hog_rider", "skeletons"], my_elixir=5))
    assert d.chosen.rule_id == "spell.value" and d.chosen.card == "fireball"
    lone = [Unit(1, "musketeer", "opp", (9.0, 22.0), hp_pct=0.3)]
    pol2 = CRPolicy(cards, arena, p)
    d2 = pol2.decide(state(units=lone, hand=["fireball", "cannon", "hog_rider", "skeletons"], my_elixir=5))
    assert d2.chosen is None or d2.chosen.rule_id != "spell.value"
