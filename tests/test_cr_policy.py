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


def test_tier0_leak_guard_prefers_troops_and_aims_spells_at_enemy_tower(arena, cards, strategy):
    _, p, _ = strategy
    p.tier0.leak_card_preference = ["fireball", "skeletons"]
    pol = CRPolicy(cards, arena, p)
    pol.tier0.check(state(t_ms=40_000, my_elixir=10.0, opp_elixir=(9, 9)), p)
    d = pol.decide(state(t_ms=40_900, my_elixir=10.0, opp_elixir=(9, 9)))
    assert d.chosen.rule_id == "t0.leak_guard" and d.chosen.card == "skeletons"
    # When only spells are affordable, aim at weakest enemy tower, never behind our own King Tower.
    pol2 = CRPolicy(cards, arena, p)
    pol2.tier0.check(state(t_ms=40_000, my_elixir=10.0, hand=["fireball", None, None, None],
                           towers=dict(TOWERS, opp_left=900)), p)
    d2 = pol2.decide(state(t_ms=40_900, my_elixir=10.0, hand=["fireball", None, None, None],
                           towers=dict(TOWERS, opp_left=900)))
    assert d2.chosen.rule_id == "t0.leak_guard" and d2.chosen.card == "fireball"
    assert d2.chosen.tile == arena.towers["opp_left"]


def test_generic_other_threat_has_fallback_speed_for_eta_and_lead(arena, cards, strategy):
    _, p, _ = strategy
    pl = Placement(arena, cards, p.placement, exec_latency_s=0.05)
    u = Unit(1, "other_ground", "opp", (3.5, 22.0), vel=(0.0, 0.0))
    th = state(units=[u]).threats(arena, cards)[0]
    assert th.eta_to_tower_s < 20.0
    lead = pl.lead_target(cards["knight"], u, state(units=[u]), 60_000)
    assert lead[1] < 22.0


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


def test_ladder_v4_blocks_suicide_bridge_and_tracks_cost_cycle(arena, cards):
    from pathlib import Path

    from droid_harness.domains.clash_royale.cards import Card, CardDB
    from droid_harness.domains.clash_royale.policy.strategy import load_strategy

    ext = CardDB({
        **cards.cards,
        "giant": Card("giant", "Giant", 5, "troop", speed_tiles_s=0.75, range_tiles=1.2, hitpoints=1930,
                      buildings_only=True),
        "mini_pekka": Card("mini_pekka", "Mini P.E.K.K.A", 4, "troop", speed_tiles_s=1.5, range_tiles=0.8,
                           hitpoints=642),
        "archers": Card("archers", "Archers", 3, "troop", units=2, speed_tiles_s=1.0, range_tiles=5.0,
                        hitpoints=119, targets_air=True, unit_classes=("archer",)),
        "goblin_cage": Card("goblin_cage", "Goblin Cage", 4, "building", range_tiles=0.0, hitpoints=350),
    })
    _, p, _ = load_strategy(Path(__file__).resolve().parents[1] / "configs/strategies/ladder-vision-v4.yaml")
    pol = CRPolicy(ext, arena, p)

    # 1. t=0.8s with opp at 7.3 elixir: NEVER slam naked bridge Giant
    s_open = GameState(t_ms=800, my_elixir=7.0, opp_elixir=(7.3, 7.3),
                       hand=["giant", "minions", "knight", "arrows"], towers=dict(TOWERS))
    cands_open = [c for r in pol.rules for c in r.propose(s_open, p, pol.ctx, None)
                  if c.rule_id == "offense.win_condition_when_opp_low"]
    assert not cands_open

    # 2. Counter availability + affordability scaling on bridge Giant
    deck8 = ["knight", "archers", "minions", "arrows", "fireball", "giant", "musketeer", "goblin_cage"]
    # (a) Goblin Cage (cost 4) in hand AND affordable (opp 4.5)
    s_ready = GameState(t_ms=60_000, my_elixir=8.0, opp_elixir=(4.0, 4.5),
                        hand=["giant", "minions", "knight", "arrows"], towers=dict(TOWERS),
                        opp_deck_seen=deck8, opp_last_four=["knight", "arrows", "fireball", "musketeer"])
    sc_ready = [c for r in pol.rules for c in r.propose(s_ready, p, pol.ctx, None)
                if c.rule_id == "offense.win_condition_when_opp_low"][0]
    # (b) Goblin Cage in hand, UNAFFORDABLE (opp 1.5 < 3.5) -> boosted (punish!)
    s_broke = GameState(t_ms=60_000, my_elixir=8.0, opp_elixir=(1.0, 1.5),
                        hand=["giant", "minions", "knight", "arrows"], towers=dict(TOWERS),
                        opp_deck_seen=deck8, opp_last_four=["knight", "arrows", "fireball", "musketeer"])
    sc_broke = [c for r in pol.rules for c in r.propose(s_broke, p, pol.ctx, None)
                if c.rule_id == "offense.win_condition_when_opp_low"][0]
    # (c) Goblin Cage & Minions BOTH out of cycle -> boosted (1.40x)
    s_out = GameState(t_ms=60_000, my_elixir=8.0, opp_elixir=(1.0, 1.5),
                      hand=["giant", "minions", "knight", "arrows"], towers=dict(TOWERS),
                      opp_deck_seen=deck8, opp_last_four=["goblin_cage", "minions", "fireball", "musketeer"])
    sc_out = [c for r in pol.rules for c in r.propose(s_out, p, pol.ctx, None)
              if c.rule_id == "offense.win_condition_when_opp_low"][0]
    assert "may be in hand" in sc_ready.reason
    assert "unaffordable" in sc_broke.reason and sc_broke.score > sc_ready.score
    assert "out of cycle" in sc_out.reason and sc_out.score > sc_broke.score


def test_ladder_v4_spell_aware_minions_kill_spells_and_counter_saving(arena, cards):
    from pathlib import Path

    from droid_harness.domains.clash_royale.cards import Card, CardDB
    from droid_harness.domains.clash_royale.policy.strategy import load_strategy

    ext = CardDB({
        **cards.cards,
        "giant": Card("giant", "Giant", 5, "troop", speed_tiles_s=0.75, range_tiles=1.2, hitpoints=1930,
                      buildings_only=True),
        "mini_pekka": Card("mini_pekka", "Mini P.E.K.K.A", 4, "troop", speed_tiles_s=1.5, range_tiles=0.8,
                           hitpoints=642),
        "archers": Card("archers", "Archers", 3, "troop", units=2, speed_tiles_s=1.0, range_tiles=5.0,
                        hitpoints=119, targets_air=True, unit_classes=("archer",)),
    })
    _, p, _ = load_strategy(Path(__file__).resolve().parents[1] / "configs/strategies/ladder-vision-v4.yaml")
    pol = CRPolicy(ext, arena, p)

    # 1. Spell-aware Minions on counter-push
    survivors = [Unit(1, "knight", "me", (14.5, 13.0), vel=(0.0, 1.0), hp_pct=1.0)]
    s_arrows_up = GameState(t_ms=60_000, my_elixir=7.0, opp_elixir=(3.2, 3.5), units=survivors,
                            hand=["minions", "knight", "arrows", "fireball"], towers=dict(TOWERS),
                            opp_deck_seen=["arrows"], opp_last_four=[])
    s_arrows_down = GameState(t_ms=60_000, my_elixir=7.0, opp_elixir=(3.2, 3.5), units=survivors,
                              hand=["minions", "knight", "arrows", "fireball"], towers=dict(TOWERS),
                              opp_deck_seen=["arrows", "knight", "giant"], opp_last_four=["arrows", "knight", "giant"])
    c_up = [c for r in pol.rules for c in r.propose(s_arrows_up, p, pol.ctx, None)
            if c.rule_id == "offense.counter_push" and c.card == "minions"][0]
    c_down = [c for r in pol.rules for c in r.propose(s_arrows_down, p, pol.ctx, None)
              if c.rule_id == "offense.counter_push" and c.card == "minions"][0]
    assert "arrows in hand" in c_up.reason and "arrows out of cycle" in c_down.reason
    assert c_down.score > c_up.score

    # 2. Kill-aware spells: Fireball won't waste on full-HP solo Giant, Arrows DOES kill 3 Minions
    s_giant = GameState(t_ms=60_000, my_elixir=6.0, opp_elixir=(3.0, 3.0),
                        hand=["fireball", "arrows", "knight", "archers"], towers=dict(TOWERS),
                        units=[Unit(1, "giant", "opp", (3.5, 19.0), vel=(0.0, -0.75))])
    assert not [c for r in pol.rules for c in r.propose(s_giant, p, pol.ctx, None) if c.rule_id == "spell.value"]
    minions3 = [Unit(10 + i, "minion", "opp", (3.5 + 0.3 * i, 20.0), vel=(0.0, -1.5), flying=True) for i in range(3)]
    s_min = GameState(t_ms=60_000, my_elixir=6.0, opp_elixir=(3.0, 3.0),
                      hand=["fireball", "arrows", "knight", "archers"], towers=dict(TOWERS), units=minions3)
    assert [c for r in pol.rules for c in r.propose(s_min, p, pol.ctx, None)
            if c.rule_id == "spell.value" and c.card == "arrows"]

    # 3. Protect Mini P.E.K.K.A when opp has affordable Giant ready in hand
    s_cyc = GameState(t_ms=60_000, my_elixir=8.5, opp_elixir=(6.0, 7.0),
                      hand=["mini_pekka", "knight", "fireball", "arrows"], towers=dict(TOWERS),
                      opp_deck_seen=["giant", "knight"], opp_last_four=["knight"])
    # Make knight cost > mini_pekka in copy OR equal to check filter excludes mini_pekka even if knight costs 3 vs mini_pekka 4;
    # set mini_pekka=2 in a tiny override? Better: verify knight (cost 3) beat mini_pekka even if only mini_pekka & archers (cost 3):
    # Test with hand=["mini_pekka", "knight", ...] where both mini_pekka and archers=3:
    ext2 = CardDB({**ext.cards, "mini_pekka": Card("mini_pekka", "Mini P.E.K.K.A", 2, "troop", speed_tiles_s=1.5,
                                                     range_tiles=0.8, hitpoints=642)})
    pol2 = CRPolicy(ext2, arena, p)
    cyc = [c for r in pol2.rules for c in r.propose(s_cyc, p, pol2.ctx, None) if c.rule_id == "cycle.cheap_cycle"][0]
    assert cyc.card == "knight"


def test_ladder_v5_giant_back_build_block_tiles(arena, cards):
    from pathlib import Path

    from droid_harness.domains.clash_royale.cards import Card, CardDB
    from droid_harness.domains.clash_royale.policy.strategy import load_strategy

    ext = CardDB({
        **cards.cards,
        "giant": Card("giant", "Giant", 5, "troop", speed_tiles_s=0.75, range_tiles=1.2, hitpoints=1930,
                      buildings_only=True),
        "archers": Card("archers", "Archers", 3, "troop", units=2, speed_tiles_s=1.0, range_tiles=5.0,
                        hitpoints=119, targets_air=True, unit_classes=("archer",)),
    })
    _, p_v5, _ = load_strategy(Path(__file__).resolve().parents[1] / "configs/strategies/ladder-vision-v5.yaml")
    _, p_v4, _ = load_strategy(Path(__file__).resolve().parents[1] / "configs/strategies/ladder-vision-v4.yaml")
    pol5 = CRPolicy(ext, arena, p_v5)
    pol4 = CRPolicy(ext, arena, p_v4)
    tx, ty = arena.towers["me_left"]

    # Enemy at 11 tiles from tower: v5 back-builds Giant cleanly (11 > 9.0), while legacy v4 defers (11 <= 13).
    far_enemy = [Unit(1, "archers", "opp", (tx, ty + 11.0), vel=(0.0, -1.0))]
    s_far = GameState(t_ms=60_000, my_elixir=8.0, opp_elixir=(4.0, 4.0), units=far_enemy,
                      hand=["giant", "knight", "arrows", "fireball"], towers=dict(TOWERS))
    c5_far = [c for r in pol5.rules for c in r.propose(s_far, p_v5, pol5.ctx, None) if c.rule_id == "offense.giant_back_build"]
    c4_far = [c for r in pol4.rules for c in r.propose(s_far, p_v4, pol4.ctx, None) if c.rule_id == "offense.giant_back_build"]
    assert len(c5_far) == 1 and c5_far[0].card == "giant" and c4_far == []

    # Enemy within 7 tiles of tower: v5 still defers back-build Giant (7 <= 9.0).
    near_enemy = [Unit(2, "knight", "opp", (tx, ty + 7.0), vel=(0.0, -1.0))]
    s_near = GameState(t_ms=60_000, my_elixir=8.0, opp_elixir=(4.0, 4.0), units=near_enemy,
                       hand=["giant", "knight", "arrows", "fireball"], towers=dict(TOWERS))
    assert not [c for r in pol5.rules for c in r.propose(s_near, p_v5, pol5.ctx, None) if c.rule_id == "offense.giant_back_build"]
