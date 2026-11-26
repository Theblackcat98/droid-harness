"""DuckDB warehouse over recorded matches (plan §9.2)."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from ..domains.clash_royale.arena import Arena
from ..domains.clash_royale.cards import CardDB
from .engagements import Engagement, segment
from .eval.stats import wilson
from .timeline import MatchTimeline, load_match
from .triage import TriageParams, triage

SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
  strategy_id TEXT PRIMARY KEY, name TEXT, parent_id TEXT, git_sha TEXT,
  params_json JSON, deck TEXT[], created_at TIMESTAMP, created_by TEXT
);
CREATE TABLE IF NOT EXISTS matches (
  match_id TEXT PRIMARY KEY, started_unix DOUBLE,
  device TEXT, account TEXT, game_version TEXT, balance_epoch TEXT,
  strategy_id TEXT, opponent_kind TEXT, opponent_id TEXT, opp_deck_seen TEXT[],
  status TEXT, result TEXT, my_crowns INT, opp_crowns INT,
  duration_s DOUBLE, overtime BOOL, elixir_leaked DOUBLE, n_actions INT,
  loop_p50_ms DOUBLE, loop_p99_ms DOUBLE, perception_warnings INT,
  experiment_id TEXT, arm TEXT, video_path TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
  match_id TEXT, seq INT, t_match_ms INT, eng_id INT, posture TEXT,
  chosen_rule TEXT, chosen_card TEXT, tile_x DOUBLE, tile_y DOUBLE, execute_at_ms INT,
  play_value DOUBLE
);
CREATE TABLE IF NOT EXISTS engagements (
  match_id TEXT, eng_id INT, lane TEXT, t_start_ms INT, t_end_ms INT,
  spent_me DOUBLE, spent_opp DOUBLE,
  board_me_start DOUBLE, board_me_end DOUBLE, board_opp_start DOUBLE, board_opp_end DOUBLE,
  tower_dmg_dealt INT, tower_dmg_taken INT, value DOUBLE, triage TEXT, triage_evidence JSON
);
CREATE TABLE IF NOT EXISTS experiments (
  experiment_id TEXT PRIMARY KEY, title TEXT, hypothesis TEXT, proposer TEXT,
  champion_id TEXT, candidate_id TEXT, opponent_pool TEXT, test TEXT,
  game_version TEXT, balance_epoch TEXT, p0 DOUBLE, p1 DOUBLE, alpha DOUBLE, beta DOUBLE,
  status TEXT, games INT, wins INT, losses INT, draws INT, aborts_by_arm JSON, llr DOUBLE,
  decided_at TIMESTAMP, notes TEXT
);
"""


def connect(path: str | Path = "data/warehouse.duckdb") -> duckdb.DuckDBPyConnection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(SCHEMA)
    return con


def analyze_match(match_dir: str | Path, arena: Arena, cards: CardDB,
                  params: TriageParams | None = None) -> tuple[MatchTimeline, list[Engagement]]:
    tl = load_match(match_dir)
    engs = [triage(e, tl, arena, params) for e in segment(tl, arena, cards)]
    return tl, engs


def ingest(con: duckdb.DuckDBPyConnection, tl: MatchTimeline, engs: list[Engagement], video_path: str = "") -> None:
    s = tl.summary
    con.execute("DELETE FROM matches WHERE match_id = ?", [tl.match_id])
    con.execute("DELETE FROM decisions WHERE match_id = ?", [tl.match_id])
    con.execute("DELETE FROM engagements WHERE match_id = ?", [tl.match_id])
    con.execute(
        "INSERT INTO matches VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [tl.match_id, s.get("started_unix"), s.get("device"), s.get("account"), s.get("game_version"),
         s.get("balance_epoch"), s.get("strategy_id"), s.get("opponent_kind"), s.get("opponent_id"),
         s.get("opp_deck_seen", []), s.get("status"), s.get("result"), s.get("my_crowns"), s.get("opp_crowns"),
         s.get("duration_s"), s.get("overtime"), s.get("elixir_leaked"), len(tl.chosen), s.get("loop_p50_ms"),
         s.get("loop_p99_ms"), s.get("perception_warnings"), s.get("experiment_id"), s.get("arm"), video_path])
    eng_of: dict[int, Engagement] = {seq: e for e in engs for seq in e.credit}
    for c in tl.chosen:
        e = eng_of.get(c.seq)
        con.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    [tl.match_id, c.seq, c.t_ms, e.eng_id if e else None, c.posture, c.rule, c.card,
                     c.tile[0], c.tile[1], c.execute_at_ms, e.credit.get(c.seq) if e else None])
    for e in engs:
        con.execute("INSERT INTO engagements VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [tl.match_id, e.eng_id, e.lane, e.t_start, e.t_end, e.spent_me, e.spent_opp,
                     e.board_me_start, e.board_me_end, e.board_opp_start, e.board_opp_end, e.tower_dmg_dealt,
                     e.tower_dmg_taken, e.value, e.triage, json.dumps(e.evidence)])


def ingest_dir(con, root: str | Path, arena: Arena, cards: CardDB) -> int:
    n = 0
    for d in sorted(Path(root).iterdir()):
        if (d / "summary.json").exists():
            tl, engs = analyze_match(d, arena, cards)
            ingest(con, tl, engs, str(d / "stream.h264"))
            n += 1
    return n


def win_rates(con, game_version: str | None = None) -> list[dict]:
    q = """SELECT strategy_id, opponent_kind,
                  count(*) FILTER (WHERE result = 'win') AS wins,
                  count(*) FILTER (WHERE result = 'loss') AS losses,
                  count(*) FILTER (WHERE result = 'draw') AS draws
           FROM matches WHERE status = 'completed' {} GROUP BY ALL ORDER BY ALL"""
    rows = con.execute(q.format("AND game_version = ?" if game_version else ""),
                       [game_version] if game_version else []).fetchall()
    out = []
    for sid, kind, w, lo_, d in rows:
        lo, hi = wilson(w, w + lo_)
        out.append({"strategy_id": sid, "opponent_kind": kind, "wins": w, "losses": lo_, "draws": d,
                    "win_rate": round(w / (w + lo_), 3) if w + lo_ else None, "ci95": (round(lo, 3), round(hi, 3))})
    return out


def play_value_by_rule(con) -> list[tuple]:
    return con.execute("""SELECT chosen_rule, posture, count(*) n, round(avg(play_value), 3) mean_value
                          FROM decisions JOIN engagements USING (match_id, eng_id)
                          WHERE engagements.triage IN ('STRATEGY', 'OK')
                          GROUP BY ALL ORDER BY mean_value""").fetchall()
