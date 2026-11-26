"""Dashboard tests over a fabricated match directory (and the real one when present)."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_dashboard():
    path = ROOT / "notebooks" / "dashboard.py"
    spec = importlib.util.spec_from_file_location("droid_dashboard_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dashboard = _load_dashboard()

DAY1 = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.UTC).timestamp()
DAY2 = dt.datetime(2026, 1, 2, 12, 0, tzinfo=dt.UTC).timestamp()


def _write_match(root: Path, name: str, summary: dict) -> Path:
    match = root / name
    match.mkdir(parents=True, exist_ok=True)
    (match / "summary.json").write_text(json.dumps(summary))
    return match


def _decision(seq: int, *, rule=None, posture="BALANCED", score=None, warnings=None,
              total: float = 10.0) -> dict:
    return {
        "seq": seq, "t_ms": seq * 200, "posture": posture, "chosen_rule": rule,
        "chosen_card": "knight" if rule else None, "score": score,
        "state": {"t_ms": seq * 200, "warnings": list(warnings or [])},
        "lat": {"decode": 1.0, "perceive": 5.0, "decide": 1.0, "total": total},
    }


@pytest.fixture
def matches(tmp_path):
    root = tmp_path / "matches"
    _write_match(root, "a", {
        "strategy_id": "s1", "opponent_kind": "ladder", "game_version": "v1",
        "status": "completed", "result": "win", "loop_p99_ms": 100.0, "started_unix": DAY1,
    })
    _write_match(root, "b", {
        "strategy_id": "s1", "opponent_kind": "ladder", "game_version": "v1",
        "status": "completed", "result": "loss", "loop_p99_ms": 200.0, "started_unix": DAY1,
    })
    # Missing status and some unrelated keys: .get must carry it through unchanged.
    _write_match(root, "c", {
        "strategy_id": "s2", "opponent_kind": "training_camp", "game_version": "v2",
        "result": "win", "started_unix": DAY2, "loop_p99_ms": 300.0,
    })

    records = [
        _decision(1, rule="offense.push", posture="PRESS", score=1.0),
        _decision(2, rule="offense.push", posture="PRESS", score=0.5),
        _decision(3, rule="defend.lane", posture="DEFEND_LEAD", score=-2.0,
                  warnings=["hand_unreadable"]),
        _decision(4, rule="defend.lane", posture="DEFEND_LEAD", score=-1.0, total=150.0),
        _decision(5),  # no chosen rule / score
        _decision(6, warnings=["elixir_discrepancy"]),
    ]
    (root / "a" / "decisions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n")
    return root


def test_win_rate_groups_and_wilson_ci(matches):
    rows = dashboard.win_rates(matches)
    by_key = {(r["strategy_id"], r["opponent_kind"], r["game_version"]): r for r in rows}

    s1 = by_key[("s1", "ladder", "v1")]
    lo, hi = dashboard.wilson(1, 2)
    assert (s1["wins"], s1["losses"], s1["draws"], s1["n"]) == (1, 1, 0, 2)
    assert s1["win_rate"] == 0.5
    assert s1["ci95"] == (round(lo, 3), round(hi, 3))

    s2 = by_key[("s2", "training_camp", "v2")]
    lo, hi = dashboard.wilson(1, 1)
    assert s2["win_rate"] == 1.0 and s2["ci95"] == (round(lo, 3), round(hi, 3))


def test_play_value_by_rule_and_posture(matches):
    rows = {(r["rule"], r["posture"]): r for r in dashboard.play_value(matches)}

    push = rows[("offense.push", "PRESS")]
    assert (push["n"], push["mean"], push["total"]) == (2, 0.75, 1.5)

    defend = rows[("defend.lane", "DEFEND_LEAD")]
    assert (defend["n"], defend["mean"], defend["total"]) == (2, -1.5, -3.0)


def test_triage_mix_and_daily_loop_p99(matches):
    assert dashboard.triage_mix(matches) == {"LATENCY": 0.167, "OK": 0.5, "PERCEPTION": 0.333}
    assert dashboard.loop_p99_by_day(matches) == {"2026-01-01": 200.0, "2026-01-02": 300.0}


def test_render_and_main_print_expected_numbers(matches, capsys):
    summary = dashboard.summarize(matches)
    text = dashboard.render(summary)
    assert "Win rate by strategy x opponent_kind x game_version" in text
    assert "s1 | ladder | v1: 1W-1L-0D win_rate=0.500" in text
    assert "offense.push | PRESS: n=2 mean=+0.750 total=+1.500" in text
    assert "PERCEPTION: 33.3%" in text
    assert "2026-01-01: 200.0 ms" in text

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(sys, "argv", ["dashboard.py", str(matches)])
        dashboard.main()
    finally:
        monkeypatch.undo()
    printed = capsys.readouterr().out
    assert "win_rate=0.500" in printed and "Triage mix" in printed


def test_dashboard_runs_on_real_matches_without_crashing():
    root = ROOT / "data" / "matches"
    if not any(root.glob("*/summary.json")):
        pytest.skip("no real match data in this checkout")
    summary = dashboard.summarize(root)
    assert isinstance(summary["win_rate"], list)
    assert isinstance(summary["play_value"], list)
    assert isinstance(summary["triage_mix"], dict)
    assert isinstance(summary["loop_p99_by_day"], dict)
