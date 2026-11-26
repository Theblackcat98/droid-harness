"""Review fixes (stale hand, blind spell bookkeeping, target capability, dead-tower paths)
and the Giant-beatdown v3 experiments (E1, E3, E4, E5) plus catch-all classes."""

from pathlib import Path

import numpy as np
import pytest

from droid_harness.domains.clash_royale.arena import Arena
from droid_harness.domains.clash_royale.cards import CardDB
from droid_harness.domains.clash_royale.policy.strategy import load_strategy
from droid_harness.domains.clash_royale.policy.utility import CRPolicy
from droid_harness.domains.clash_royale.state import GameState, Unit

ROOT = Path(__file__).resolve().parents[1]
HAND = ["giant", "knight", "musketeer", "fireball"]


@pytest.fixture
def starter():
    return CardDB.load(ROOT / "tests/fixtures/cards_starter.json")


def _policy(starter, name, **kw):
    _, p, _ = load_strategy(ROOT / f"configs/strategies/{name}.yaml")
    return CRPolicy(starter, Arena(), p, **kw), p


def _s(t_ms=60_000, elixir=8.0, hand=HAND, **kw):
    kw.setdefault("opp_elixir", (2.0, 2.5))
    return GameState(t_ms=t_ms, my_elixir=elixir, hand=list(hand), frame_size=(464, 1024), **kw)


def _enemy(kind, tile, vel=(0.0, -1.0), flying=False, tid=1):
    return Unit(tid, kind, "opp", tile, vel, 1.0, 0, 0.9, flying=flying)


# --- stale hand ------------------------------------------------------------------------
def test_stale_slots_are_never_played_or_retried(starter):
    s = _s(hand_fresh=[True, False, True, True])
    assert [k for _, k in s.affordable_hand(starter)] == ["giant", "musketeer", "fireball"]
    pol, _ = _policy(starter, "training-camp-blind-v1", retry_after_ms=400)
    first = pol.decide(_s(40_000, 8.0, hand_fresh=[True] * 4)).chosen
    assert first.card == "giant" and first.slot == 0
    # The giant's slot failed to read (the next card is sliding in): no blind re-tap.
    d = pol.decide(_s(40_450, 8.1, hand_fresh=[False, True, True, True]))
    assert d.chosen is None or not d.chosen.reason.startswith("retry:")
    assert pol.constraints.why_not(_s(hand_fresh=[False] * 4), first) == "stale_slot"


def test_pipeline_marks_unreadable_slots_stale_but_still_detects_the_play(arena):
    from droid_harness.core.transport.frame_slot import Frame
    from droid_harness.domains.clash_royale.perception.pipeline import CRPerception
    from droid_harness.sim.cr_scene import CRScene, build_matcher_from_scene

    cards = CardDB.load(ROOT / "tests/fixtures/cards_starter.json")
    scene = CRScene(cards, ["knight", "archers", "minions", "arrows", "fireball", "giant", "musketeer",
                            "mini_pekka"], elixir=7.0, draw_tower_hp=False)
    plays = []
    perc = CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards, arena=arena,
                        matcher=build_matcher_from_scene(scene), on_my_play=lambda c, t: plays.append(c))
    img = scene.render()
    perc.start_match(Frame(1, img, 0, 0, 0))
    perc.process(Frame(1, img, 0, 0, 0))
    played = scene.hand[0]
    blank = img.copy()
    x0, y0, x1, y1 = scene.layout.hand[0].to_px(img.shape)
    blank[y0:y1, x0:x1] = 0  # slot 0 unreadable (card sliding in)
    s = perc.process(Frame(2, blank, 33_000, 0, 0))
    assert s.hand[0] == played and s.hand_fresh[0] is False and all(s.hand_fresh[1:])
    scene.play_card(0, (9.0, 10.0))
    s = perc.process(Frame(3, scene.render(), 66_000, 0, 0))
    assert plays == [played] and s.hand_fresh[0] and s.hand[0] != played


# --- blind spell bookkeeping ---------------------------------------------------------------
def test_blind_support_troop_does_not_use_up_the_push_spell(starter):
    pol, p = _policy(starter, "training-camp-blind-v1")
    rule = next(r for r in pol.rules if r.id == "offense.blind_push")
    pol.confirm_played("giant", 10_000)
    pol.confirm_played("knight", 10_000 + p.blind_spell_after_ms + 500)  # a troop inside the window
    assert rule.spelled_this_push is False
    pol.confirm_played("fireball", 10_000 + p.blind_spell_after_ms + 900)
    assert rule.spelled_this_push is True


# --- target capability -----------------------------------------------------------------
def test_counter_rows_never_override_what_a_card_can_hit(starter):
    pol, p = _policy(starter, "training-camp-v3")
    assert not starter["giant"].can_hit(False) and not starter["mini_pekka"].can_hit(True)
    assert starter["fireball"].can_hit(True) and starter["musketeer"].can_hit(True)
    # Pretend the table says Mini P.E.K.K.A counters minions and Giant counters knights.
    counters = dict(p.counters)
    counters[("mini_pekka", "minion")] = 0.9
    counters[("giant", "knight")] = 0.9
    p.counters = counters
    air = _s(elixir=9.0, hand=["mini_pekka", "giant", "knight", "arrows"],
             units=[_enemy("minion", (4.0, 10.0), flying=True)], towers={"me_left": 1400})
    cands = [c for r in pol.rules for c in r.propose(air, p, pol.ctx, None)]
    assert not [c for c in cands if c.rule_id.startswith("defend") and c.card in ("mini_pekka", "giant", "knight")]
    ground = _s(elixir=9.0, hand=["giant", "mini_pekka", "musketeer", "arrows"],
                units=[_enemy("knight", (4.0, 10.0))], towers={"me_left": 1400})
    cands = [c for r in pol.rules for c in r.propose(ground, p, pol.ctx, None)]
    defenders = {c.card for c in cands if c.rule_id.startswith("defend")}
    assert "giant" not in defenders and "mini_pekka" in defenders


def test_path_prediction_follows_destroyed_towers(starter):
    from droid_harness.domains.clash_royale import path
    a = Arena()
    u = _enemy("giant", (3.5, 11.0), vel=(0, -0.75))
    to_princess = path.advance(u, 30.0, a, speed=0.75)
    to_king = path.advance(u, 30.0, a, speed=0.75, alive={"me_left": False})
    assert to_princess[0] == pytest.approx(3.5, abs=1.0)  # stops at the left princess tower
    assert to_king[0] > 6.0  # walks on to the king once the princess tower is rubble
    pol, p = _policy(starter, "training-camp-v3")
    s = _s(towers={"me_left": 0, "me_right": 1400})
    lead = pol.placement.lead_target(starter["musketeer"], u, s, s.t_ms + 8000)
    assert lead[0] > 4.0


# --- v3 experiments ------------------------------------------------------------------------
def test_v3_never_sends_a_support_card_as_a_win_condition(starter):
    pol, p = _policy(starter, "training-camp-v3")
    assert p.win_conditions == ["giant"]
    s = _s(elixir=9.0, hand=["archers", "knight", "arrows", "fireball"], opp_elixir=(0.5, 1.0))
    cands = [c for r in pol.rules for c in r.propose(s, p, pol.ctx, None)]
    assert not [c for c in cands if c.rule_id == "offense.win_condition_when_opp_low"]


def test_e4_cycle_gate_scales_giant_pushes(starter):
    base_pol, base_p = _policy(starter, "training-camp-v3")
    pol, p = _policy(starter, "training-camp-v3-e4-cycle-gate")

    def push_score(policy, params, **kw):
        s = _s(elixir=8.0, opp_elixir=(1.0, 2.0), **kw)
        c = [c for r in policy.rules for c in r.propose(s, params, policy.ctx, None)
             if c.rule_id == "offense.win_condition_when_opp_low"]
        return c[0].score, c[0].reason

    base, _ = push_score(base_pol, base_p, opp_deck_seen=["mini_pekka", "knight"], opp_last_four=["knight"])
    held, why = push_score(pol, p, opp_deck_seen=["mini_pekka", "knight"], opp_last_four=["knight"])
    assert held == pytest.approx(base * 0.5) and "mini_pekka may be in hand" in why
    deck = ["mini_pekka", "knight", "archer", "goblin", "arrows", "giant", "musketeer", "bomber"]
    spent, why = push_score(pol, p, opp_deck_seen=deck, opp_last_four=["goblin", "arrows", "mini_pekka", "archer"])
    assert spent == pytest.approx(base * 1.3) and "out of cycle" in why
    unknown, _ = push_score(pol, p, opp_deck_seen=deck[:5], opp_last_four=["mini_pekka"])
    assert unknown == pytest.approx(base)  # an unseen card could still be a counter


def test_e3_tank_goes_ahead_of_survivors(starter):
    me = [Unit(i, "musketeer", "me", (14.5, 12.0 + i), (0.0, 1.0), 1.0, 0, 0.9) for i in range(2)]
    me += [Unit(9, "knight", "me", (14.5, 12.5), (0.0, 1.0), 1.0, 0, 0.9)]
    s = _s(elixir=7.0, units=me, hand=["giant", "knight", "arrows", "fireball"])
    for name, ahead in (("training-camp-v3", False), ("training-camp-v3-e3-tank-ahead", True)):
        pol, p = _policy(starter, name)
        giant = [c for r in pol.rules for c in r.propose(s, p, pol.ctx, None)
                 if c.rule_id == "offense.counter_push" and c.card == "giant"][0]
        lead_y = 13.0
        assert (giant.tile[1] > lead_y) == ahead, (name, giant.tile)


def test_e5_reserve_and_back_build(starter):
    pol, p = _policy(starter, "training-camp-v3-e5-reserve")
    quiet = _s(elixir=9.4, hand=["giant", "knight", "arrows", "fireball"], opp_elixir=(6.0, 10.0))
    d = pol.decide(quiet)
    assert d.chosen.rule_id == "offense.giant_back_build" and d.chosen.tile[1] < 3.0
    base_pol, base_p = _policy(starter, "training-camp-v3")
    assert not [c for r in base_pol.rules for c in r.propose(quiet, base_p, base_pol.ctx, None)
                if c.rule_id == "offense.giant_back_build"]
    # A bridge push at 7 elixir leaves 2 < 3 reserve while they may have elixir: halved.
    s = _s(elixir=7.0, opp_elixir=(1.0, 2.9))
    c_base = [c for r in base_pol.rules for c in r.propose(s, base_p, base_pol.ctx, None)
              if c.rule_id == "offense.win_condition_when_opp_low"][0]
    p._p["press_opp_elixir_max"] = 2.0  # they might have 2.9: not "low", so the reserve applies
    c_res = [c for r in pol.rules for c in r.propose(s, p, pol.ctx, None)
             if c.rule_id == "offense.win_condition_when_opp_low"][0]
    assert c_res.score == pytest.approx(c_base.score * 0.5) and "reserve" in c_res.reason
    p._p["press_opp_elixir_max"] = 3.0  # they're low: no reserve needed
    c_low = [c for r in pol.rules for c in r.propose(s, p, pol.ctx, None)
             if c.rule_id == "offense.win_condition_when_opp_low"][0]
    assert c_low.score == pytest.approx(c_base.score)


# --- catch-all classes ----------------------------------------------------------------------
def test_catch_all_enemy_is_a_threat_with_generic_counters(starter, arena):
    from droid_harness.domains.clash_royale.perception.classes import flying_of
    assert flying_of("other_air", None) and not flying_of("other_ground", None)
    pol, p = _policy(starter, "training-camp-v3")
    s = _s(elixir=9.0, hand=["musketeer", "mini_pekka", "giant", "arrows"],
           units=[_enemy("other_air", (14.0, 10.0), flying=True)], towers={"me_right": 1400})
    cands = [c for r in pol.rules for c in r.propose(s, p, pol.ctx, None) if c.rule_id == "defend.air_threat"]
    assert {c.card for c in cands} >= {"musketeer"} and "mini_pekka" not in {c.card for c in cands}


def test_synth_samples_classes_evenly(tmp_path):
    from droid_harness.domains.clash_royale.perception.classes import load_class_names
    from droid_harness.learn import synth
    names = load_class_names(ROOT / "models/classes.txt")
    sp = np.zeros((40, 24, 4), np.uint8)
    sp[..., 3] = 255
    sprites = [synth.Sprite(sp, "other_ground_opp")] * 500 + [synth.Sprite(sp, "archer_opp")]
    synth.generate_dataset([np.full((768, 464, 3), 60, np.uint8)], sprites, tmp_path, 30, names,
                           synth.SynthConfig(seed=1, min_units=4, max_units=4, min_visible=0.0))
    import yaml
    counts = yaml.safe_load((tmp_path / "label_counts.yaml").read_text())
    assert counts.get("archer_opp", 0) > 0.3 * counts.get("other_ground_opp", 1)
