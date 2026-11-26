"""Load a recorded match directory into a timeline the analysis code can use.

A match directory (written by the recorder) holds ``summary.json``, ``decisions.jsonl``
(sampled state digests + every chosen action) and ``frames.jsonl`` (per-frame latency).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Point:
    t_ms: int
    units: list[tuple]  # (track_id, kind, side, x, y, vx, vy, hp)
    towers: dict[str, int]
    warnings: list[str]
    posture: str = "BALANCED"
    my_elixir: float = 0.0
    loop_ms: float | None = None


@dataclass
class Play:
    t_ms: int
    side: str
    card: str
    tile: tuple[float, float] | None
    decision_seq: int | None = None
    rule: str | None = None


@dataclass
class ChosenAction:
    seq: int
    t_ms: int
    rule: str
    card: str
    tile: tuple[float, float]
    execute_at_ms: int | None
    posture: str


@dataclass
class MatchTimeline:
    match_id: str
    summary: dict
    points: list[Point] = field(default_factory=list)
    plays: list[Play] = field(default_factory=list)
    chosen: list[ChosenAction] = field(default_factory=list)
    frame_ms: list[tuple[float, float]] = field(default_factory=list)  # (match t_ms, total loop ms)


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_match(match_dir: str | Path) -> MatchTimeline:
    d = Path(match_dir)
    summary = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    tl = MatchTimeline(summary.get("match_id", d.name), summary)
    for rec in _jsonl(d / "decisions.jsonl"):
        st = rec.get("state") or {}
        if st:
            tl.points.append(Point(st["t_ms"], st.get("units", []), st.get("towers", {}), st.get("warnings", []),
                                   st.get("posture", "BALANCED"), st.get("my_elixir", 0.0),
                                   (rec.get("lat") or {}).get("total")))
        if rec.get("chosen_rule"):
            tl.chosen.append(ChosenAction(rec.get("seq", 0), rec["t_ms"], rec["chosen_rule"], rec["chosen_card"],
                                          tuple(rec["tile"]), rec.get("execute_at_ms"), rec.get("posture", "")))
    tl.points.sort(key=lambda p: p.t_ms)
    start_pts = summary.get("start_pts_us") or 0
    for f in _jsonl(d / "frames.jsonl"):
        tl.frame_ms.append(((f.get("pts_us", 0) - start_pts) / 1000, f.get("total", 0.0)))
    tl.plays = plays_from_events(summary.get("events", []), tl.chosen)
    return tl


def plays_from_events(events: list[dict], chosen: list[ChosenAction]) -> list[Play]:
    """Opponent plays come from spawns; ours from hand changes, matched to the decision that caused them."""
    plays = []
    used: set[int] = set()
    for e in events:
        if e["kind"] == "opp_spawn_play":
            plays.append(Play(e["t_ms"], "opp", e["card"], tuple(e["tile"]) if e.get("tile") else None))
        elif e["kind"] == "my_play":
            match = None
            for c in reversed(chosen):
                if c.seq in used or c.card != e["card"]:
                    continue
                if 0 <= e["t_ms"] - c.t_ms <= 2000:
                    match = c
                    break
            if match is not None:
                used.add(match.seq)
            plays.append(Play(e["t_ms"], "me", e["card"], match.tile if match else None,
                              match.seq if match else None, match.rule if match else None))
    return sorted(plays, key=lambda p: p.t_ms)
