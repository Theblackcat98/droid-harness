"""Studio (live view + replay inspector): snapshot schema, library, server routes, live tap."""

from __future__ import annotations

import gzip
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from droid_harness.core.actuate.executor import SentEvent
from droid_harness.core.actuate.gestures import TouchEvent
from droid_harness.core.runtime.orchestrator import BrainLoop
from droid_harness.core.transport.frame_slot import Frame
from droid_harness.domains.clash_royale.policy.candidate import Candidate, Decision
from droid_harness.domains.clash_royale.policy.strategy import load_strategy
from droid_harness.domains.clash_royale.state import GameState, Unit
from droid_harness.flywheel.eval.runner import ExperimentConfig, ExperimentRunner
from droid_harness.ui.library import MatchLibrary, find_match_dirs, read_ledger
from droid_harness.ui.server import Studio, StudioServer
from droid_harness.ui.snapshot import CostBook, StrategyIndex, candidate_rows, geometry, snapshot_from_record
from droid_harness.ui.telemetry import LivePublisher, TelemetryTap, experiment_state

ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "configs/strategies/ladder-vision-v6.yaml"


def _record(t_ms=15913, *, acted=True) -> dict:
    state = {"t_ms": t_ms, "phase": "x1", "my_elixir": 3.01, "opp_elixir": [2.72, 2.72],
             "hand": ["arrows", "archers", "mini_pekka", "fireball"], "hand_fresh": [True, True, False, True],
             "next": "giant",
             "units": [[7, "musketeer", "me", 15.32, 12.45, 0.27, 0.82, 0.32],
                       [33, "other_ground", "opp", 14.5, 18.21, 1.29, -5.72, 0.7]],
             "towers": {"me_right": 1400, "me_king": 2568, "me_left": 1512}, "crowns": [0, 0],
             "opp_last_four": ["giant", "minions"], "opp_deck_seen": ["giant", "minions", "knight"],
             "pocket_open": {"left": False, "right": False}, "posture": "PRESS", "last_action_ms": 8399,
             "warnings": ["hand_unreadable"]}
    cands = [
        {"rule_id": "defend.air_threat", "card": "arrows", "slot": 0, "tile": [13.5, 17.5], "score": 0.039,
         "reason": "arrows vs minion", "predicted_tile": [13.6, 17.3], "total": 0.039},
        {"rule_id": "defend.lane_threat", "card": "archers", "slot": 1, "tile": [14.5, 9.0], "score": 0.1988,
         "reason": "archers vs other_ground", "predicted_tile": [14.5, 12.1], "total": 0.1988},
        {"rule_id": "offense.counter_push", "card": "archers", "slot": 1, "tile": [14.5, 12.5], "score": -0.00286,
         "reason": "support", "total": -0.0034},
    ]
    return {"seq": 1, "t_ms": t_ms, "posture": "PRESS",
            "chosen_rule": "defend.lane_threat" if acted else None, "chosen_card": "archers" if acted else None,
            "tile": [14.5, 9.0] if acted else None, "predicted_tile": [14.5, 12.1] if acted else None,
            "score": 0.1988 if acted else None, "candidates": cands, "state": state,
            "lat": {"decode": 0.38, "perceive": 1.82, "decide": 0.16, "total": 2.4}}


V6_ID = load_strategy(V6)[2]  # strategy IDs hash the policy source, so compute it


def _write_match(root: Path, name: str, *, result="win", sid=V6_ID, n=5) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"match_id": name, "device": "phone-a", "strategy_id": sid,
                                             "started_unix": 1795768891.2}))
    (d / "summary.json").write_text(json.dumps({
        "match_id": name, "strategy_id": sid, "result": result, "status": "completed", "my_crowns": 1,
        "opp_crowns": 0, "duration_s": 12.0, "arm": "candidate", "experiment_id": "EXP-0001",
        "events": [{"kind": "my_play", "card": "archers", "slot": 1, "t_ms": 1000},
                   {"kind": "opp_spawn_play", "card": "giant", "tile": [3.5, 24.0], "t_ms": 2000}]}))
    lines = [json.dumps(_record(1000 * i, acted=i % 2 == 0)) for i in range(n)]
    (d / "decisions.jsonl").write_text("\n".join(lines) + "\n")
    return d


# --- snapshot schema ---------------------------------------------------------------------------

def test_snapshot_from_record_ranks_candidates_and_breaks_down_the_total():
    _, params, _ = load_strategy(V6)
    snap = snapshot_from_record(_record(), params=params, costs=CostBook(None))
    d = snap["decision"]
    assert [c["card"] for c in d["candidates"]] == ["archers", "arrows", "archers"]  # sorted by total
    chosen = [c for c in d["candidates"] if c["chosen"]]
    assert len(chosen) == 1 and chosen[0]["rule_id"] == "defend.lane_threat"
    assert d["threshold"] == params.act_threshold and d["acted"]
    # weight × posture multiplier comes from the strategy file when it is known
    assert chosen[0]["weight"] == params.weight("defend.lane_threat")
    assert chosen[0]["posture_mult"] == params.posture_multipliers.get("PRESS", {}).get("defend", 1.0)
    # hand: stale slot is not playable even though the card is known
    assert snap["hand"][2] == {"slot": 2, "card": "mini_pekka", "cost": 4.0, "fresh": False, "affordable": False}
    assert snap["hand"][0]["affordable"] and snap["next"] == "giant"
    # opponent cycle: giant was played 2 plays ago -> 3 more plays before it is back
    cyc = {c["card"]: c for c in snap["opp"]["cards"]}
    assert cyc["giant"]["until_back"] == 3 and not cyc["giant"]["in_hand"]
    assert cyc["knight"]["in_hand"] and cyc["knight"]["can_play_now"]  # 2.72 elixir >= 3 - 0.5 margin
    assert snap["opp"]["unknown_slots"] == 5
    assert {u["side"] for u in snap["units"]} == {"me", "opp"}
    json.dumps(snap)  # JSON-ready


def test_candidate_rows_without_a_known_strategy_derive_the_multiplier():
    rows = candidate_rows({"posture": "BALANCED", "candidates": [
        {"rule_id": "spell.value", "card": "fireball", "tile": [3, 20], "score": 0.5, "total": 0.4}]}, None)
    r = rows["candidates"][0]
    assert r["weight"] is None and r["wm"] == pytest.approx(0.8) and rows["threshold"] is None
    assert not rows["acted"] and not r["chosen"]


def test_geometry_carries_the_homography_and_rois():
    from droid_harness.sim.cr_scene import sim_layout, sim_tiles

    g = geometry(sim_layout(), sim_tiles(), None)
    H = np.array(g["tile_to_screen"])
    p = H @ np.array([9.0, 16.0, 1.0])
    assert 0 < p[0] / p[2] < 1 and 0 < p[1] / p[2] < 1
    assert len(g["rois"]["hand"]) == 4 and g["arena"]["width"] == 18.0


def test_strategy_index_resolves_current_ids():
    _, _, sid = load_strategy(V6)
    hit = StrategyIndex(ROOT / "configs/strategies").get(sid)
    assert hit is not None and hit[0] == "ladder-vision-v6"


# --- library + server ------------------------------------------------------------------------

def test_library_lists_and_loads_matches(tmp_path):
    _write_match(tmp_path / "batch", "aaa111")
    _write_match(tmp_path / "batch", "bbb222", result="loss")
    (tmp_path / "batch" / "not-a-match").mkdir()
    assert len(find_match_dirs([tmp_path])) == 2
    lib = MatchLibrary([tmp_path], strategies=StrategyIndex(ROOT / "configs/strategies"), root=ROOT)
    rows = lib.refresh()
    assert {r["id"] for r in rows} == {"batch/aaa111", "batch/bbb222"}
    assert all(r["strategy"] == "ladder-vision-v6" and not r["has_video"] for r in rows)
    m = lib.load("batch/aaa111")
    assert len(m["timeline"]) == 5 and m["strategy"]["known"] and m["geometry"]["tile_to_screen"]
    assert [s["decision"]["acted"] for s in m["timeline"]] == [True, False, True, False, True]
    assert m["timeline"][0]["tower_max"]["me_left"] == 1512
    assert lib.load("../../etc/passwd") is None and lib.video("batch/aaa111") is None
    # a single match directory works as a root too
    assert [r["id"] for r in MatchLibrary([tmp_path / "batch" / "aaa111"]).refresh()] == ["aaa111"]


def _get(url: str, *, gzip_ok: bool = False):
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"} if gzip_ok else {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            body = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            return r.status, dict(r.headers), body
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_replay_server_routes(tmp_path):
    _write_match(tmp_path, "ccc333", n=200)
    ledger = tmp_path / "experiments.jsonl"
    ledger.write_text("\n".join(json.dumps({"experiment_id": "EXP-0001", "status": s, "title": "t"})
                                for s in ("running", "inconclusive")) + "\n")
    lib = MatchLibrary([tmp_path], strategies=StrategyIndex(ROOT / "configs/strategies"), root=ROOT)
    server = StudioServer(Studio(library=lib, ledger=ledger), "127.0.0.1", 0).start()
    try:
        base = server.url
        code, _, body = _get(base)
        assert code == 200 and b"droid-harness" in body and b"<canvas" in body
        assert json.loads(_get(base + "api/info")[2]) ["replay"] is True
        rows = json.loads(_get(base + "api/matches")[2])
        assert rows[0]["id"] == "ccc333"
        code, headers, body = _get(base + "api/match?id=ccc333", gzip_ok=True)
        assert code == 200 and headers.get("Content-Encoding") == "gzip"
        assert len(json.loads(body)["timeline"]) == 200
        assert _get(base + "api/match?id=nope")[0] == 404
        assert _get(base + "api/match/frame.jpg?id=ccc333&t=0")[0] == 404  # compact files only, no video
        assert _get(base + "api/live/snapshot")[0] == 404  # replay-only server
        assert [e["status"] for e in json.loads(_get(base + "api/ledger")[2])] == ["inconclusive"]
    finally:
        server.stop()
    assert read_ledger(ledger)[0]["status"] == "inconclusive"


# --- live tap -----------------------------------------------------------------------------------

def _live_raw():
    st = GameState(t_ms=4200, frame_size=(64, 128), my_elixir=6.0, opp_elixir=(3.0, 4.0),
                   hand=["knight", "archers", "giant", "arrows"], hand_fresh=[True] * 4,
                   units=[Unit(1, "giant", "opp", (3.5, 20.0), (0.0, -1.0))],
                   towers={"me_left": 1512}, opp_deck_seen=["giant"], opp_last_four=["giant"])
    c = Candidate("defend.lane_threat", "knight", 0, (3.5, 12.0), 0.4, reason="knight vs giant")
    dec = Decision(st.t_ms, 0, "BALANCED", c, candidates=[c], totals={0: 0.4}, state_digest=st.digest())
    frame = Frame(7, np.full((128, 64, 3), 90, np.uint8), 1_000_000, 0, 0)

    class Obs:
        screen, state = "IN_BATTLE", st

    dets = [{"cls": "giant", "side": "opp", "conf": 0.91, "box": (10, 20, 30, 50), "tile": (3.5, 20.0), "hp": 1.0}]
    return frame, Obs(), dec, {"decode": 0.4, "total": 2.5}, dets


def test_live_publisher_builds_snapshot_and_jpeg():
    tap = TelemetryTap()
    tap.publish(_live_raw())
    tap.on_sent(SentEvent(TouchEvent(time.perf_counter_ns(), action=0, x=12, y=34, tag="place:down"),
                          time.perf_counter_ns()))
    tap.on_sent(SentEvent(TouchEvent(time.perf_counter_ns(), action=1, x=12, y=34, tag="place:up"),
                          time.perf_counter_ns()))
    _, params, _ = load_strategy(V6)
    ctx = {"params": params, "strategy": {"name": "ladder-vision-v6"}, "runner": {"completed": 2, "target": 10}}
    pub = LivePublisher(tap, lambda: ctx)
    snap = pub.publish_once(tap.latest()[1])
    assert snap["mode"] == "live" and snap["in_battle"] and snap["screen"] == "IN_BATTLE"
    assert snap["frame"] == {"seq": 7, "w": 64, "h": 128, "has_image": True}
    assert snap["detections"][0]["cls"] == "giant"
    assert snap["decision"]["candidates"][0]["chosen"] and snap["decision"]["threshold"] == params.act_threshold
    assert [t["tag"] for t in snap["taps"]] == ["place"]  # DOWN events only
    assert pub.jpeg[:2] == b"\xff\xd8" and pub.jpeg_seq == 7
    assert json.loads(pub.snapshot_json)["runner"]["completed"] == 2


def test_live_server_streams_events():
    tap = TelemetryTap()
    pub = LivePublisher(tap, lambda: {}, fps=30).start()
    server = StudioServer(Studio(live=pub, live_config={"rois": None}), "127.0.0.1", 0).start()
    try:
        tap.publish(_live_raw())
        deadline = time.time() + 3
        while pub.jpeg_seq != 7 and time.time() < deadline:
            time.sleep(0.02)
        base = server.url
        code, headers, body = _get(base + "api/live/frame.jpg")
        assert code == 200 and headers.get("X-Frame-Seq") == "7" and body[:2] == b"\xff\xd8"
        with urllib.request.urlopen(base + "api/live/events", timeout=5) as r:
            line = b""
            while not line.startswith(b"data: "):
                line = r.readline()
        assert json.loads(line[6:])["t_ms"] == 4200
    finally:
        server.stop()


def test_brain_loop_on_step_observer_cannot_break_the_loop():
    seen = []

    class P:
        def process(self, f):
            return "obs"

    class Pol:
        def decide(self, obs, intent):
            return None

    def observer(*args):
        seen.append(args)
        raise RuntimeError("ui bug")

    loop = BrainLoop(None, P(), Pol(), None, None, on_step=observer)
    frame = Frame(1, np.zeros((4, 4, 3), np.uint8), 0, time.perf_counter_ns(), time.perf_counter_ns())
    assert loop.step(frame) is None
    assert len(seen) == 1 and seen[0][0] is frame and seen[0][1] == "obs" and "total" in seen[0][3]


def test_observe_only_sink_never_touches_the_phone():
    from droid_harness.app import BotRunner, RunConfig

    assert RunConfig.__dataclass_fields__["observe_only"].default is False
    assert isinstance(BotRunner._observe_sink(0, 10, 10, 0), int)


def test_experiment_state_for_the_strip():
    r = ExperimentRunner(ExperimentConfig(experiment_id="EXP-9", game_version="v", max_games=40, seed=3))
    for _ in range(3):
        arm, pid = r.next_assignment()
        r.record({"arm": arm, "pair_id": pid, "game_version": "v", "status": "completed",
                  "result": "win" if arm == "candidate" else "loss"})
    st = experiment_state(r, active_arm="champion")
    assert st["id"] == "EXP-9" and st["active_arm"] == "champion" and st["status"] == "running"
    assert st["arms"]["candidate"]["wins"] + st["arms"]["champion"]["losses"] == 3
    assert st["lower"] < 0 < st["upper"]
    json.dumps(st)


def test_sim_with_ui_records_a_replayable_match(tmp_path):
    pytest.importorskip("av")
    from droid_harness.app import run_sim
    from droid_harness.ui.library import VideoCache

    r = run_sim(2.5, tmp_path / "sim", ui={"host": "127.0.0.1", "port": 0, "fps": 10})
    d = Path(r["match_dir"])
    lib = MatchLibrary([tmp_path / "sim"], root=ROOT)
    rows = lib.refresh()
    assert rows and rows[0]["has_video"]
    m = lib.load(rows[0]["id"])
    assert m["timeline"] and m["geometry"]["tile_to_screen"]  # sim geometry, no phone calibration
    vc = VideoCache(d, None, stride_ms=200)
    deadline = time.time() + 20
    while not vc.done and time.time() < deadline:
        time.sleep(0.05)
    assert vc.error is None and vc.status()["frames"] > 3
    t, jpeg = vc.frame_at(10_000)
    assert jpeg[:2] == b"\xff\xd8" and t <= 10_000
