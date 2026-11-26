from pathlib import Path

import pytest

from droid_harness.domains.clash_royale.arena import Arena
from droid_harness.domains.clash_royale.cards import CardDB
from droid_harness.domains.clash_royale.policy.strategy import load_strategy

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def cards():
    return CardDB.load(FIX / "cards_min.json")


@pytest.fixture
def arena():
    return Arena()


@pytest.fixture
def strategy():
    return load_strategy(FIX / "strategy.yaml")
