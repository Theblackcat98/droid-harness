"""Replay regression: recorded H.264 + device timestamps -> perception -> events -> diff."""

import json
from fractions import Fraction
from pathlib import Path

import av
import pytest

from droid_harness.domains.clash_royale.cards import CardDB
from droid_harness.domains.clash_royale.perception.pipeline import CRPerception
from droid_harness.domains.clash_royale.replay import diff_events, diff_runs, replay_match
from droid_harness.sim.cr_scene import CRScene, build_matcher_from_scene

ROOT = Path(__file__).resolve().parents[1]
DECK = ["knight", "archers", "minions", "arrows", "fireball", "giant", "musketeer", "mini_pekka"]


def _record(match_dir: Path, scene: CRScene, seconds: float, fps: int, actions: dict[int, tuple]) -> int:
    """Encode the scene like the phone would; returns the start pts. Variable frame times."""
    match_dir.mkdir(parents=True)
    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height, enc.pix_fmt = scene.width, scene.height, "yuv420p"
    enc.time_base = Fraction(1, fps)
    enc.options = {"tune": "zerolatency", "preset": "ultrafast", "crf": "18"}
    raw, pts = bytearray(), []
    t0_us = 5_000_000  # device clock doesn't start at 0
    for i in range(int(seconds * fps)):
        if i in actions:
            scene.play_card(*actions[i])
        scene.step(1 / fps)
        vf = av.VideoFrame.from_ndarray(scene.render(), format="bgr24")
        vf.pts = i
        for pkt in enc.encode(vf):
            raw += bytes(pkt)
            pts.append(t0_us + int(pkt.pts * 1e6 / fps))
    for pkt in enc.encode(None):
        raw += bytes(pkt)
        pts.append(t0_us + int(pkt.pts * 1e6 / fps))
    (match_dir / "stream.h264").write_bytes(bytes(raw))
    (match_dir / "stream.pts").write_text("".join(f"{p}\n" for p in pts))
    (match_dir / "meta.json").write_text(json.dumps({"match_id": match_dir.name, "start_pts_us": t0_us}))
    (match_dir / "summary.json").write_text(json.dumps({"match_id": match_dir.name, "result": "win",
                                                        "duration_s": seconds}))
    return t0_us


def test_replay_reproduces_plays_with_device_timing(tmp_path, arena):
    cards = CardDB.load(ROOT / "tests/fixtures/cards_starter.json")
    scene = CRScene(cards, DECK, elixir=7.0, draw_tower_hp=False)
    played = scene.hand[1]
    _record(tmp_path / "m1", scene, 2.0, 30, {45: (1, (9.0, 10.0))})  # play at 1.5 s

    def perc():
        return CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards, arena=arena,
                            matcher=build_matcher_from_scene(scene))

    r = replay_match(tmp_path / "m1", perc())
    assert r["frames"] >= 55
    plays = [e for e in r["events"] if e["kind"] == "my_play"]
    assert [e["card"] for e in plays] == [played]
    assert plays[0]["t_ms"] == pytest.approx(1500, abs=100)  # device timestamps, not frame count guesses


def test_event_diff_reports_lost_and_gained():
    old = [{"kind": "opp_spawn_play", "card": "giant", "tile": (14.5, 20.0), "t_ms": 10_000},
           {"kind": "my_play", "card": "knight", "t_ms": 12_000}]
    new = [{"kind": "opp_spawn_play", "card": "giant", "tile": (14.0, 20.5), "t_ms": 10_300},
           {"kind": "opp_spawn_play", "card": "minion", "tile": (4.0, 22.0), "t_ms": 15_000}]
    d = diff_events(old, new)
    assert d["matched"] == 1 and [e["card"] for e in d["lost"]] == ["knight"]
    assert [e["card"] for e in d["gained"]] == ["minion"]
    assert d["by_kind"]["opp_spawn_play"] == {"old": 1, "new": 2, "lost": 0, "gained": 1}
    rep = diff_runs({"a": {"events": old}, "b": {"events": []}}, {"a": {"events": new}})
    assert rep["matches_compared"] == 1 and rep["only_old"] == ["b"] and rep["lost"] == 1
