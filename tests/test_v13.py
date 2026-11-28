"""Ladder v13: the strategy for the deck the account plays (Skeletons, Cannon and Mega Minion in)."""

from __future__ import annotations

from pathlib import Path

import pytest

from droid_harness.domains.clash_royale.account import AccountLevels, check_strategy
from droid_harness.domains.clash_royale.arena import Arena
from droid_harness.domains.clash_royale.cards import CardDB
from droid_harness.domains.clash_royale.policy.counter_prior import effectiveness
from droid_harness.domains.clash_royale.policy.placement import Placement
from droid_harness.domains.clash_royale.policy.rules import RuleContext
from droid_harness.domains.clash_royale.policy.rules.cycle import _cheapest_cycle_card
from droid_harness.domains.clash_royale.policy.strategy import load_strategy
from droid_harness.domains.clash_royale.state import GameState

ROOT = Path(__file__).resolve().parents[1]
STRATS = ROOT / "configs/strategies"
CARDS = CardDB.load(ROOT / "configs/domains/clash_royale/cards.json")
DECK = ["knight", "skeletons", "cannon", "arrows", "fireball", "giant", "musketeer", "mega_minion"]


@pytest.fixture(scope="module")
def v13():
    return load_strategy(STRATS / "ladder-vision-v13.yaml")


def test_v13_is_the_deck_the_account_plays(v13):
    spec, p, _ = v13
    assert spec.deck == DECK and p.deck == DECK and spec.win_conditions == ["giant"]
    assert spec.parent == "ladder-vision-v12" and p.level_aware and p.counter_prior


def test_v13_changes_only_what_follows_from_the_cards(v13):
    """v12's measured parameters stay; what changed names a card that left the deck."""
    spec, p, _ = v13
    v12_spec, v12, _ = load_strategy(STRATS / "ladder-vision-v12.yaml")
    changed = {
        k for k in set(spec.params) | set(v12_spec.params) if spec.params.get(k) != v12_spec.params.get(k)
    }
    assert changed == {"counter_prior", "cycle_save_cards", "punish_cards"}
    assert set(p.tier0.leak_card_preference) <= set(DECK) <= set(CARDS.cards)
    assert not set(v12.tier0.leak_card_preference) <= set(DECK)  # v12 named cards the deck no longer has
    assert set(spec.params["punish_cards"]) <= set(DECK)
    assert spec.weights == v12_spec.weights and spec.posture_multipliers == v12_spec.posture_multipliers
    assert spec.placement == v12_spec.placement


def test_v13_counters_for_the_new_cards(v13):
    _, p, _ = v13

    def eff(card: str, threat: str, flying: bool = False) -> float:
        return effectiveness(p, card, threat, CARDS, flying=flying)

    assert eff("skeletons", "giant") > eff("skeletons", "valkyrie")  # DPS on a tank, never into splash
    assert eff("cannon", "hog_rider") >= 0.9 and eff("cannon", "minion", flying=True) == 0.0
    assert eff("mega_minion", "balloon", flying=True) >= 0.7 and eff("mega_minion", "pekka") >= 0.6
    assert (
        p.counters[("knight", "bomber")] == 0.85 and p.counters[("arrows", "minion")] == 0.9
    )  # kept from v9
    # A pair with no row now gets the card-DB prior instead of "never defend with this card".
    assert ("mega_minion", "royal_recruit") not in p.counters and eff("mega_minion", "royal_recruit") > 0


def _cycle(p, hand: list[str], opp_may_play: list[str]):
    ctx = RuleContext(CARDS, Arena(), Placement(Arena(), CARDS, p.placement))
    s = GameState(t_ms=60_000, my_elixir=9.0, opp_elixir=(9.0, 9.0), hand=hand, opp_deck_seen=opp_may_play)
    return _cheapest_cycle_card(s, p, ctx)


def test_the_tank_answers_are_not_cycled_away(v13):
    """Skeletons are the cheapest card, so the cycle rules would throw them away first."""
    _, p, _ = v13
    hand = ["skeletons", "knight", "giant", "fireball"]
    assert _cycle(p, hand, ["giant", "arrows"]) == (3, 1, "knight")  # their Giant may be in hand
    assert _cycle(p, hand, ["giant", "arrows", "knight", "musketeer", "fireball"]) == (3, 1, "knight")
    assert _cycle(p, hand, []) == (1, 0, "skeletons")  # nothing to save them for


def test_older_strategies_keep_their_own_saved_cards():
    _, v12, _ = load_strategy(STRATS / "ladder-vision-v12.yaml")
    assert v12.cycle_save_cards is None  # the starter deck's answers
    assert _cycle(v12, ["minions", "mini_pekka", "knight", "giant"], ["giant", "arrows"])[2] == "knight"


def test_doctor_has_nothing_to_say_about_v13(v13):
    _, p, _ = v13
    acct = AccountLevels.load(ROOT / "configs/accounts/theblackcat99.yaml")
    assert check_strategy(p, CARDS, acct)["problems"] == []
    assert p.tier0.spell_tower_damage == acct.spell_tower_damage
