"""Fabricate recorded match directories for flywheel tests."""

import json
from pathlib import Path

TOWERS = {"me_left": 2534, "me_right": 2534, "me_king": 4008, "opp_left": 2534, "opp_right": 2534,
          "opp_king": 4008}


def write_match(root: Path, match_id: str, *, result="win", respond=True, warnings_at=(), frame_gap_at=None,
                confirm_play=True, strategy_id="s1", arm=None, status="completed", game_version="9.x"):
    d = root / match_id
    d.mkdir(parents=True)
    decisions, events, frames = [], [], []
    cannon_hp, me_left = 1.0, TOWERS["me_left"]
    seq = 0
    for k in range(0, 41):  # 0..20 s every 500 ms
        t = k * 500
        units, towers = [], dict(TOWERS)
        if 10_000 <= t < 15_000 or (not respond and 10_000 <= t < 18_000):
            y = 24 - 2.0 * (t - 10_000) / 1000
            hp = 1.0 if t < 13_000 else (0.5 if respond else 1.0)
            units.append([1, "hog_rider", "opp", 3.5, max(y, 8.0), 0.0, -2.0, hp])
        if respond and t >= 11_600:
            cannon_hp = 0.6 if t >= 13_000 else 1.0
            units.append([2, "cannon", "me", 7.5, 10.5, 0.0, 0.0, cannon_hp])
        if t >= 14_000:
            me_left = TOWERS["me_left"] - (200 if respond else 800)
        towers["me_left"] = me_left
        warns = ["elixir_discrepancy"] if t in warnings_at else []
        seq += 1
        rec = {"seq": seq, "t_ms": t, "posture": "BALANCED", "chosen_rule": None, "chosen_card": None,
               "tile": None, "execute_at_ms": None,
               "state": {"t_ms": t, "units": units, "towers": towers, "warnings": warns, "posture": "BALANCED",
                         "my_elixir": 5.0}, "lat": {"total": 12.0}}
        if respond and t == 11_500:
            rec.update(chosen_rule="defend.lane_threat", chosen_card="cannon", tile=[7.5, 10.5],
                       execute_at_ms=11_500)
        decisions.append(rec)
    events.append({"kind": "opp_spawn_play", "card": "hog_rider", "tile": [3.5, 24.0], "t_ms": 10_000})
    if respond and confirm_play:
        events.append({"kind": "my_play", "card": "cannon", "slot": 1, "t_ms": 11_600})
    pts0 = 5_000_000
    for i in range(0, 20_000, 17):
        if frame_gap_at and frame_gap_at <= i < frame_gap_at + 400:
            continue
        frames.append({"pts_us": pts0 + i * 1000, "total": 10.0})
    summary = {"match_id": match_id, "device": "phone-a", "account": "acct-1", "strategy_id": strategy_id,
               "game_version": game_version, "balance_epoch": "2026-11", "opponent_kind": "practice",
               "opponent_id": "hog-cycle", "status": status, "result": result, "my_crowns": 1 if result == "win" else 0,
               "opp_crowns": 0 if result == "win" else 1, "duration_s": 20.0, "overtime": False,
               "elixir_leaked": 0.5, "start_pts_us": pts0, "events": events, "arm": arm, "started_unix": 0.0}
    (d / "summary.json").write_text(json.dumps(summary))
    (d / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in decisions))
    (d / "frames.jsonl").write_text("\n".join(json.dumps(f) for f in frames))
    return d
