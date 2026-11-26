"""Offline ranker: video labeling → BC training → static ONNX → live rescore.

Everything here runs on CPU with a fabricated match directory and a hand-written labeled
set; no phone, no adb, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from droid_harness.domains.clash_royale.policy.candidate import Candidate
from droid_harness.learn.label_from_video import label_from_video
from droid_harness.learn.train_ranker import (
    OnnxRanker,
    build_examples,
    candidate_dim,
    load_ranker,
    load_rows,
    state_dim,
    train_ranker,
)

DECK = ["knight", "archers", "minions", "arrows", "fireball", "giant", "musketeer", "mini_pekka"]


def _state(hand: list[str | None], *, elixir: float = 7.0, phase: str = "x1") -> dict:
    return {
        "t_ms": 1000, "phase": phase, "my_elixir": elixir, "opp_elixir": [3.0, 4.0],
        "hand": hand, "next": "fireball", "towers": {"me_left": 1400, "opp_right": 900},
        "crowns": [1, 0], "posture": "PRESS", "warnings": [],
    }


def _candidates(card: str, *, tile=(14.5, 14.0)) -> list[dict]:
    other = DECK[(DECK.index(card) + 1) % len(DECK)]
    return [
        {"rule_id": "offense.blind_push", "card": card, "tile": list(tile), "score": 0.9},
        {"rule_id": "defend.lane_threat", "card": other, "tile": [3.5, 14.0], "score": 0.2},
    ]


def _decision(t_ms: int, card: str | None, hand: list[str | None], *, tile=(14.5, 14.0)) -> dict:
    return {
        "seq": t_ms, "t_ms": t_ms, "posture": "PRESS", "chosen_rule": "offense.blind_push",
        "chosen_card": card, "tile": list(tile), "score": 0.9,
        "candidates": _candidates(card, tile=tile) if card else [],
        "state": _state(hand),
    }


def _write_match(root: Path, name: str, summary: dict, decisions: list[dict]) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"status": "completed", **summary}))
    (d / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in decisions))
    return d


def test_label_from_video_weights_played_actions_and_skips_bad_ticks(tmp_path):
    root = tmp_path / "matches"
    _write_match(
        root, "win",
        {"match_id": "win", "result": "win", "my_crowns": 3, "opp_crowns": 1,
         "events": [{"kind": "my_play", "card": "giant", "slot": 0, "t_ms": 1300}]},
        [
            _decision(1000, "giant", ["giant", "knight", None, None]),   # played → row
            _decision(2000, None, ["giant", "knight", None, None]),      # no action
            _decision(3000, "knight", [None, None, None, None]),         # missing hand
            _decision(4000, "knight", ["knight", "giant", None, None]),  # event absent → tap missed
        ])
    _write_match(
        root, "loss",
        {"match_id": "loss", "result": "loss", "my_crowns": 0, "opp_crowns": 2,
         "events": [{"kind": "my_play", "card": "archers", "slot": 1, "t_ms": 1400},
                    {"kind": "my_play", "card": "fireball", "slot": 2, "t_ms": 2400}]},
        [_decision(1200, "archers", ["archers", "knight", None, None], tile=(3.5, 14.0)),
         _decision(2100, "fireball", ["fireball", "knight", None, None])])
    # An older recording with no events still demonstrates chosen_card plays.
    _write_match(
        root, "draw",
        {"match_id": "draw", "result": "draw", "my_crowns": 1, "opp_crowns": 1, "events": []},
        [_decision(1500, "musketeer", ["musketeer", "giant", None, None])])

    out = tmp_path / "labels.jsonl"
    stats = label_from_video(root, out, deck=DECK)

    assert stats["rows"] == 4 and stats["matches"] == 3
    assert stats["skipped"] == {"no_hand": 1, "no_action": 1, "unplayed": 1, "no_candidates": 0}
    rows = load_rows(out)
    assert {r["match_id"]: r["weight"] for r in rows} == {"win": 1.0, "loss": 0.3, "draw": 0.5}
    assert all(len(r["state"]) == state_dim(DECK) for r in rows)
    assert all(r["chosen_index"] == 0 and len(r["candidates"]) == 2 for r in rows)
    assert all(r["action"]["card"] == r["candidates"][r["chosen_index"]]["card"] for r in rows)
    assert next(r for r in rows if r["match_id"] == "loss")["crowns_delta"] == -2

    # The labeler's output is directly trainable.
    x, y, w, deck, used = build_examples(rows)
    assert deck == DECK and x.shape[1] == state_dim(DECK) + candidate_dim(DECK)
    assert used["rows"] == 4 and y.sum() == 4 and {round(float(v), 6) for v in np.unique(w)} == {0.3, 0.5, 1.0}


def _rows_20() -> list[dict]:
    rows = []
    for i in range(20):
        card = DECK[i % len(DECK)]
        rows.append({
            "deck": DECK,
            "state": _state([card, DECK[(i + 1) % len(DECK)], None, None], elixir=5.0 + i % 5),
            "candidates": _candidates(card),
            "chosen_index": 0,
            "weight": 1.0 if i < 14 else 0.3,
        })
    return rows


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("ranker") / "ranker.onnx"
    meta = train_ranker(_rows_20(), out, epochs=5, hidden=(16, 8), lr=0.05, batch_size=4, seed=0)
    return out, meta


def test_train_export_onnx_and_rescore(tiny_model):
    model, meta = tiny_model
    assert model.exists() and Path(str(model) + ".json").exists()
    assert meta["format"] == "droid-harness-ranker/1" and meta["deck"] == DECK
    assert meta["rows"] == 20 and meta["train_accuracy"] >= 0.9
    assert meta["state_dim"] + meta["candidate_dim"] == meta["input_dim"]

    ranker = load_ranker(model, deck=DECK)
    assert isinstance(ranker, OnnxRanker)
    state = _state(["giant", "knight", None, None], elixir=8.0)
    candidates = [
        Candidate("offense.blind_push", "giant", 0, (14.5, 14.0), 0.9),
        Candidate("defend.lane_threat", "knight", 1, (3.5, 14.0), 0.2),
        Candidate("cycle.cheap_cycle", "archers", 2, (9.0, 3.0), 0.1),
    ]
    rescored = ranker.rescore(state, candidates)
    assert [c.card for c in rescored] == [c.card for c in candidates]
    assert all(isinstance(c.score, float) and 0.0 <= c.score <= 1.0 for c in rescored)
    assert rescored[0].score > rescored[1].score and rescored[0].score > rescored[2].score
    assert rescored[0].rule_id == "offense.blind_push" and rescored[0].tile == (14.5, 14.0)
    # The policy calls rescore on an empty candidate list too.
    assert ranker.rescore(state, []) == []


def test_app_ranker_loader_rejects_bad_path_and_loads_good(tmp_path, tiny_model):
    model, meta = tiny_model

    with pytest.raises(ValueError, match="missing or unreadable"):
        load_ranker(tmp_path / "does-not-exist.onnx")

    with pytest.raises(ValueError, match="deck order"):
        load_ranker(model, deck=list(reversed(DECK)))

    corrupt = tmp_path / "corrupt.onnx"
    corrupt.write_bytes(b"not an onnx model")
    Path(str(corrupt) + ".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="ONNX unreadable"):
        load_ranker(corrupt, deck=DECK)

    loaded = load_ranker(model, deck=DECK)
    assert loaded.input_dim == meta["input_dim"] and loaded.max_candidates == meta["max_candidates"]
