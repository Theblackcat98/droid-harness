"""Offline shadow-mode and match-clock tests (no phone, no replayed video)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from droid_harness.core.transport.frame_slot import Frame
from droid_harness.domains.clash_royale.perception.cards import CardMatcher
from droid_harness.domains.clash_royale.perception.digits import DigitReader
from droid_harness.domains.clash_royale.perception.pipeline import CRPerception
from droid_harness.domains.clash_royale.policy.candidate import Candidate, Decision
from droid_harness.domains.clash_royale.state import GameState
from droid_harness.flywheel.shadow import ShadowRecord, ShadowRunner, report
from droid_harness.sim.cr_scene import sim_layout, sim_tiles

# --- shadow mode ------------------------------------------------------------

def _states(n: int = 10) -> list[GameState]:
    return [GameState(t_ms=1_000 + 500 * i, my_elixir=float(i % 11)) for i in range(n)]


class _Choices:
    """Stub policy: one fixed rule id per tick."""

    def __init__(self, by_tick: dict[int, str]) -> None:
        self.by_tick = by_tick
        self.seen: list[int] = []

    def choose(self, state: GameState) -> str:
        self.seen.append(state.t_ms)
        return self.by_tick[state.t_ms]


def test_shadow_records_agreement_and_ranks_disagreements():
    states = _states(10)
    champion = _Choices({s.t_ms: "offense.push" for s in states})
    # ticks 0-4 agree, 5-7 disagree on B, 8-9 disagree on C
    candidate_map = {s.t_ms: ("offense.push" if i < 5 else "defend.air" if i < 8 else "spell.fireball")
                     for i, s in enumerate(states)}
    candidate = _Choices(candidate_map)
    values = [0.0] * 5 + [-2.0, -1.0, 0.0] + [2.0, 4.0]

    runner = ShadowRunner(champion, candidate)
    records = runner.run(states, values=values)

    assert len(records) == 10
    assert sum(r.agreement for r in records) == 5
    assert records[0] == ShadowRecord(1_000, "offense.push", "offense.push", True, 0.0)
    assert records[5] == ShadowRecord(3_500, "offense.push", "defend.air", False, -2.0)
    assert [r.tick for r in records] == [s.t_ms for s in states]
    assert runner.report(records) == [("defend.air", 3, -1.0), ("spell.fireball", 2, 3.0)]
    assert report(records) == [("defend.air", 3, -1.0), ("spell.fireball", 2, 3.0)]


def test_shadow_accepts_digests_embedded_values_and_same_snapshot():
    seen: dict[str, float] = {}

    class MutatingChampion:
        def choose(self, state: GameState) -> str:
            seen["champion_saw"] = state.my_elixir
            state.my_elixir = 0.0
            return "A"

    class ReaderCandidate:
        def choose(self, state: GameState) -> str:
            seen["candidate_saw"] = state.my_elixir
            return "B"

    state = GameState(t_ms=7_500, my_elixir=6.5, warnings=["w"])
    item = {"state": state.digest(), "engagement_value": -3.25}
    [rec] = ShadowRunner(MutatingChampion(), ReaderCandidate()).run([item])

    assert seen == {"champion_saw": 6.5, "candidate_saw": 6.5}
    assert rec.tick == 7_500 and rec.value == -3.25
    assert (rec.champion_choice, rec.candidate_choice, rec.agreement) == ("A", "B", False)
    assert report([rec]) == [("B", 1, -3.25)]


def test_shadow_loads_policies_by_qualified_name_and_normalizes_decisions(tmp_path, monkeypatch):
    module = tmp_path / "shadow_stub_policy.py"
    module.write_text(
        "def choose(state):\n"
        "    return 'defend.air' if state.my_elixir < 5 else 'offense.push'\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    states = [GameState(t_ms=100, my_elixir=2.0), GameState(t_ms=200, my_elixir=8.0)]
    records = ShadowRunner("shadow_stub_policy", "shadow_stub_policy").run(states)
    assert [r.candidate_choice for r in records] == ["defend.air", "offense.push"]
    assert all(r.agreement for r in records)

    class Champion:
        def decide(self, state: GameState) -> Decision:
            return Decision(state.t_ms, 0, "BALANCED",
                            Candidate("offense.win_condition", "giant", 0, (9.0, 10.0), 1.0))

    class CandidatePolicy:
        def choose(self, state: GameState) -> Candidate:
            return Candidate("defend.lane_threat", "knight", 0, (3.5, 12.0), 1.0)

    [rec] = ShadowRunner(Champion(), CandidatePolicy()).run([GameState(t_ms=300)])
    assert rec.champion_choice == "offense.win_condition"
    assert rec.candidate_choice == "defend.lane_threat"
    assert report([rec]) == [("defend.lane_threat", 1, 0.0)]


# --- match clock ------------------------------------------------------------

@pytest.fixture
def clock_perception(cards, arena):
    layout = sim_layout()
    reader = DigitReader.from_font(scale=0.8, thickness=1)

    def make(digits=reader):
        return CRPerception(layout=layout, tiles=sim_tiles(), cards=cards, arena=arena,
                            matcher=CardMatcher(), digits=digits)

    return layout, make


def _clock_frame(layout, text: str, pts_us: int, seq: int = 1) -> Frame:
    img = np.zeros((640, 360, 3), np.uint8)
    x0, _y0, _x1, y1 = layout.clock.to_px(img.shape)
    cv2.putText(img, text, (x0 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 1, cv2.LINE_AA)
    return Frame(seq, img, pts_us, 0, 0)


def test_clock_reads_once_per_second_and_parses_m_ss(clock_perception):
    layout, make = clock_perception
    perception = make()

    s = perception.process(_clock_frame(layout, "1:23", 0))
    assert s.match_t_s == pytest.approx(83.0)
    assert s.clock_drift_s == pytest.approx(-83.0)
    assert s.clock_activity and s.clock_activity > 0
    assert "clock_drift" not in s.warnings

    # 0.5 s later the clock changed on screen, but the cache is still 1:23.
    s = perception.process(_clock_frame(layout, "2:00", 500_000, 2))
    assert s.match_t_s == pytest.approx(83.0)
    assert s.clock_drift_s == pytest.approx(-82.5)

    # At 1.0 s the ROI is read again and the new value is used.
    s = perception.process(_clock_frame(layout, "2:00", 1_000_000, 3))
    assert s.match_t_s == pytest.approx(120.0)
    assert s.clock_drift_s == pytest.approx(-119.0)


def test_clock_drift_warns_after_one_second(clock_perception):
    layout, make = clock_perception
    perception = make()

    perception.process(_clock_frame(layout, "1:23", 0))
    s = perception.process(_clock_frame(layout, "1:23", 83_000_000, 2))
    assert s.match_t_s == pytest.approx(83.0)
    assert s.clock_drift_s == pytest.approx(0.0, abs=1e-3)
    assert "clock_drift" not in s.warnings

    s = perception.process(_clock_frame(layout, "1:23", 84_200_000, 3))
    assert s.clock_drift_s == pytest.approx(1.2)
    assert "clock_drift" in s.warnings


def test_clock_without_digit_templates_keeps_activity_signal(clock_perception):
    layout, make = clock_perception
    perception = make(digits=DigitReader())  # no templates loaded

    s = perception.process(_clock_frame(layout, "1:23", 0))
    assert s.match_t_s is None and s.clock_drift_s is None
    assert perception.clock_activity and perception.clock_activity > 0
    assert "clock_drift" not in s.warnings
