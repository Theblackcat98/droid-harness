"""No-vision offense (offense.blind_push) and the execution retry."""

from pathlib import Path

import pytest

from droid_harness.domains.clash_royale.arena import Arena
from droid_harness.domains.clash_royale.cards import CardDB
from droid_harness.domains.clash_royale.policy.strategy import load_strategy
from droid_harness.domains.clash_royale.policy.utility import CRPolicy
from droid_harness.domains.clash_royale.state import GameState

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def starter():
    return CardDB.load(ROOT / "tests/fixtures/cards_starter.json")


def _policy(starter, name="training-camp-blind-v1", **kw):
    _, p, _ = load_strategy(ROOT / f"configs/strategies/{name}.yaml")
    return CRPolicy(starter, Arena(), p, **kw), p


def _s(t_ms, elixir, hand, **kw):
    kw.setdefault("opp_elixir", (4.0, 10.0))  # what perception reports when it can't see plays
    return GameState(t_ms=t_ms, my_elixir=elixir, hand=list(hand), phase=kw.pop("phase", "x1"),
                     frame_size=(464, 1024), **kw)


def test_vision_strategy_never_attacks_blind(starter):
    """The bug this rule exists for: no units seen -> opp elixir upper bound 10 -> no offense."""
    pol, _ = _policy(starter, "training-camp-v2")
    d = pol.decide(_s(40_000, 8.0, ["giant", "knight", "musketeer", "fireball"], opp_elixir=(4.0, 10.0)))
    assert d.chosen is None or d.chosen.rule_id != "offense.win_condition_when_opp_low"


def test_blind_push_tank_then_support_then_spell(starter):
    pol, p = _policy(starter)
    d = pol.decide(_s(40_000, 8.0, ["giant", "knight", "musketeer", "fireball"]))
    assert d.chosen.rule_id == "offense.blind_push" and d.chosen.card == "giant"
    assert d.chosen.tile == (14.5, 14.0)  # right bridge
    pol.confirm_played("giant", 40_200)

    # Support right after: knight (all we can afford) at the bridge.
    d = pol.decide(_s(41_000, 3.2, ["arrows", "knight", "musketeer", "fireball"]))
    assert d.chosen.card == "knight" and d.chosen.tile == (14.5, 14.0)
    pol.confirm_played("knight", 41_100)

    # Ranged support stands behind the bridge.
    d = pol.decide(_s(45_000, 4.5, ["arrows", "archers", "musketeer", "fireball"]))
    assert d.chosen.card == "musketeer" and d.chosen.tile == (14.5, 12.0)
    pol.confirm_played("musketeer", 45_100)

    # Tank should be at the tower: one spell on opp_right.
    d = pol.decide(_s(52_000, 4.0, ["arrows", "archers", "mini_pekka", "fireball"]))
    assert d.chosen.card in ("fireball", "arrows") and d.chosen.tile == Arena().towers["opp_right"]
    pol.confirm_played(d.chosen.card, 52_100)
    d = pol.decide(_s(53_000, 4.0, ["arrows", "archers", "mini_pekka", "knight"]))
    assert d.chosen is None  # one push-timed spell per push; 4 elixir is below pressure


def test_blind_pressure_and_holding_for_the_tank(starter):
    pol, _ = _policy(starter)
    # No Giant in hand: heaviest affordable troop at 7 elixir.
    d = pol.decide(_s(30_000, 7.0, ["knight", "mini_pekka", "arrows", "archers"]))
    assert d.chosen.card == "mini_pekka" and "pressure" in d.chosen.reason
    # Giant in hand but under the push threshold: hold the elixir.
    pol2, _ = _policy(starter)
    assert pol2.decide(_s(30_000, 7.5, ["knight", "giant", "arrows", "archers"])).chosen is None
    # Double elixir lowers the push threshold.
    d = pol2.decide(_s(130_000, 6.2, ["knight", "giant", "arrows", "archers"], phase="x2"))
    assert d.chosen.card == "giant"


def test_blind_lanes_alternate_and_dead_towers_retarget(starter):
    pol, p = _policy(starter)
    rule = next(r for r in pol.rules if r.id == "offense.blind_push")
    assert rule.lane(p) == "right"
    for t in (10_000, 30_000):
        pol.confirm_played("giant", t)
    assert rule.lane(p) == "left"
    pol.reset_match()
    assert rule.lane(p) == "right" and rule.tank_ms is None
    s = _s(10_000, 9.8, ["arrows", "fireball", "giant", "knight"], towers={"opp_right": 0})
    spells = [c for c in rule.propose(s, p, pol.ctx, None) if c.card in ("arrows", "fireball")]
    assert spells and all(c.tile == Arena().towers["opp_king"] for c in spells)


def test_blind_rule_is_off_for_vision_strategies(starter):
    pol, p = _policy(starter, "training-camp-v2")
    rule = next(r for r in pol.rules if r.id == "offense.blind_push")
    assert list(rule.propose(_s(40_000, 10.0, ["giant", "knight", "musketeer", "fireball"]), p,
                             pol.ctx, None)) == []


def test_missed_tap_is_retried_once(starter):
    pol, _ = _policy(starter, retry_after_ms=400, pending_timeout_ms=700)
    hand = ["giant", "knight", "musketeer", "fireball"]
    first = pol.decide(_s(40_000, 8.0, hand)).chosen
    assert first.card == "giant"
    assert pol.decide(_s(40_200, 8.1, hand)).chosen is None  # waiting for confirmation
    retry = pol.decide(_s(40_450, 8.2, hand)).chosen
    assert retry.card == "giant" and retry.slot == first.slot and retry.reason.startswith("retry:")
    assert pol.retries == 1
    assert pol.decide(_s(40_900, 8.3, hand)).chosen is None  # only once
    # The card left the hand (the hand read moved on): never re-tap whatever slid in.
    pol2, _ = _policy(starter, retry_after_ms=400)
    pol2.decide(_s(40_000, 8.0, hand))
    moved = ["archers", "knight", "musketeer", "fireball"]
    d = pol2.decide(_s(40_450, 3.2, moved))
    assert d.chosen is None or not d.chosen.reason.startswith("retry:")
